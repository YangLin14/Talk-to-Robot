"""Top-level LLM grounder.

`ground(instruction, tier, context=None, prompt_variant="zero-shot",
client=None)` mirrors the regex grounder's signature and returns a dict
with the same shared schema, plus LLM-specific fields:
    {
      instruction, tier, goal, grounder, prompt_variant,
      success, failure_mode, reasoning, raw_response,
      parse_error, usage, finish_reason, notes,
    }

Client is lazily constructed on first call so importing the module does
not require GEMINI_API_KEY.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from grounding.client import GeminiClient  # noqa: E402
from grounding.prompts import PROMPT_BUILDERS  # noqa: E402
from grounding.parser import parse_goal  # noqa: E402
from grounding.classifier import classify  # noqa: E402


_client_singleton: Optional[GeminiClient] = None


def _get_default_client() -> GeminiClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = GeminiClient()
    return _client_singleton


def ground(
    instruction: str,
    tier: str,
    context: Optional[dict] = None,
    prompt_variant: str = "zero-shot",
    client: Optional[GeminiClient] = None,
) -> dict:
    if prompt_variant not in PROMPT_BUILDERS:
        raise ValueError(
            f"unknown prompt_variant {prompt_variant!r}; "
            f"expected one of {list(PROMPT_BUILDERS)}"
        )
    prompt = PROMPT_BUILDERS[prompt_variant](instruction, tier, context)

    c = client or _get_default_client()
    base = {
        "instruction": instruction,
        "tier": tier,
        "grounder": c.model,
        "prompt_variant": prompt_variant,
    }

    try:
        raw = c.generate(prompt, json_mode=True)
    except Exception as e:  # noqa: BLE001 -- surface any SDK error verbatim
        return {
            **base,
            "goal": None,
            "success": False,
            "failure_mode": "api_error",
            "reasoning": None,
            "raw_response": None,
            "parse_error": None,
            "usage": None,
            "finish_reason": None,
            "notes": f"{type(e).__name__}: {e}",
        }

    parsed = parse_goal(raw["text"])
    failure_mode = classify(raw["text"], parsed)

    return {
        **base,
        "goal": parsed["goal"],
        "success": failure_mode is None,
        "failure_mode": failure_mode,
        "reasoning": parsed.get("reasoning"),
        "raw_response": raw["text"],
        "parse_error": parsed.get("parse_error"),
        "usage": raw.get("usage"),
        "finish_reason": raw.get("finish_reason"),
        "notes": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction")
    parser.add_argument("--tier", required=True,
                        choices=["T0", "T1", "T2", "T3", "T4"])
    parser.add_argument("--variant", default="zero-shot",
                        choices=list(PROMPT_BUILDERS))
    parser.add_argument("--cube-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the assembled prompt and exit (no API call)")
    args = parser.parse_args()

    context = {"cube_pos": args.cube_pos} if args.cube_pos else None

    if args.show_prompt:
        print(PROMPT_BUILDERS[args.variant](args.instruction, args.tier, context))
        return

    out = ground(args.instruction, args.tier, context=context,
                 prompt_variant=args.variant)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
