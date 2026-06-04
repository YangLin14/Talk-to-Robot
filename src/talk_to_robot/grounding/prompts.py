"""Prompt templates for the three variants studied in proposal section 7:
zero-shot, few-shot (3 exemplars), chain-of-thought.

All prompts share the same workspace description (from workspace.py) so
the LLM grounder sees exactly the coordinate frame our regex grounder
and instruction-library ground truth use.
"""

import json

from talk_to_robot.workspace import workspace_description, WORKSPACE


_OUTPUT_FORMAT_ZS = (
    'Respond with a single JSON object and no other text:\n'
    '{"goal": [x, y, z]}\n'
    "where x, y, z are floats in meters."
)

_OUTPUT_FORMAT_COT = (
    'Respond with a single JSON object and no other text:\n'
    '{"reasoning": "<your step-by-step reasoning>", "goal": [x, y, z]}\n'
    "where x, y, z are floats in meters. Reason about the coordinate frame "
    "explicitly before choosing the goal."
)

_FEWSHOT_EXAMPLES = [
    {
        "context_text": None,
        "instruction": "push to (1.30, 0.85)",
        "output": {"goal": [1.30, 0.85, WORKSPACE["z"]]},
    },
    {
        "context_text": None,
        "instruction": "push to the upper-right corner",
        "output": {"goal": [WORKSPACE["x_max"], WORKSPACE["y_min"], WORKSPACE["z"]]},
    },
    {
        "context_text": "Cube position: [1.34, 0.75, 0.42]",
        "instruction": "move it 5 cm to the right",
        "output": {"goal": [1.34, 0.70, WORKSPACE["z"]]},
    },
]


def _format_context(context):
    if not context:
        return None
    parts = []
    if "cube_pos" in context and context["cube_pos"] is not None:
        cp = context["cube_pos"]
        parts.append(f"Cube position: [{cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f}]")
    if "reference_objects" in context and context["reference_objects"]:
        for name, pos in context["reference_objects"].items():
            parts.append(
                f"Reference object '{name}' at: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
            )
        parts.append(
            "Reference-object side rule: choose a nearby point about 0.06 meters "
            "from the named object. 'right side of' means -y from the object; "
            "'left side of' means +y; 'in front of' / 'forward of' means +x; "
            "'behind' / 'back of' means -x. Keep the goal inside the workspace."
        )
    return "\n".join(parts) if parts else None


def _instruction_block(instruction, context_text=None):
    lines = []
    if context_text:
        lines.append(context_text)
    lines.append(f'Instruction: "{instruction}"')
    return "\n".join(lines)


def build_zero_shot(instruction: str, tier: str, context=None) -> str:
    parts = [
        "You translate natural-language manipulation instructions into a 3D goal position for a robot.",
        workspace_description(),
        _instruction_block(instruction, _format_context(context)),
        _OUTPUT_FORMAT_ZS,
    ]
    return "\n\n".join(parts)


def build_few_shot(instruction: str, tier: str, context=None) -> str:
    parts = [
        "You translate natural-language manipulation instructions into a 3D goal position for a robot.",
        workspace_description(),
        "Examples:",
    ]
    for ex in _FEWSHOT_EXAMPLES:
        block = _instruction_block(ex["instruction"], ex["context_text"])
        parts.append(block + "\n" + json.dumps(ex["output"]))
    parts.append(_instruction_block(instruction, _format_context(context)))
    parts.append(_OUTPUT_FORMAT_ZS)
    return "\n\n".join(parts)


def build_cot(instruction: str, tier: str, context=None) -> str:
    parts = [
        "You translate natural-language manipulation instructions into a 3D goal position for a robot.",
        workspace_description(),
        "Think step by step. Identify what the instruction refers to (a literal "
        "coordinate, a named region, a relative offset, a reference object, or "
        "an intent), then map it to a concrete 3D point inside the workspace.",
        _instruction_block(instruction, _format_context(context)),
        _OUTPUT_FORMAT_COT,
    ]
    return "\n\n".join(parts)


PROMPT_BUILDERS = {
    "zero-shot": build_zero_shot,
    "few-shot": build_few_shot,
    "cot": build_cot,
}
