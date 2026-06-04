"""Regex grounder for T0-T2 instructions.

Output shape matches the LLM grounder (`grounding.ground`) so the eval
harness stays grounder-agnostic. T3-T4 deliberately return success=False
with failure_mode='unsupported_tier' -- per proposal section 8
(condition F), regex is not a fair baseline for reference-object or
intent reasoning.

Workspace + axis constants are imported from `talk_to_robot.workspace`
(single source of truth shared with the LLM grounder).
"""

import argparse
import json
import re
from typing import Optional

import numpy as np

from talk_to_robot.workspace import WORKSPACE, DIR_VECTORS as _DIR_TUPLES, REGIONS

DIR_VECTORS = {k: np.array(v) for k, v in _DIR_TUPLES.items()}

# T0: "(1.20, 0.50)" or "1.20, 0.50" or "(1.20, 0.50, 0.42)" or "[1.20, 0.50]".
COORD_RE = re.compile(
    r"[\(\[]?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*[\)\]]?"
)

# T0 named variant: "x=1.30, y=0.85" or "x=1.30 y=0.85 z=0.42".
NAMED_COORD_RE = re.compile(
    r"x\s*=\s*(-?\d+(?:\.\d+)?)\s*,?\s*y\s*=\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*,?\s*z\s*=\s*(-?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)

# T2: "<n> <unit> [to the] <direction>".
OFFSET_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(cm|centimeters?|centimetres?|m|meters?|metres?)"
    r"\s*(?:to\s+the\s+)?(right|left|forward|back|backward|backwards|north|south|east|west)",
    re.IGNORECASE,
)


def _result(goal, *, success, failure_mode=None, raw_match=None, notes=None):
    return {
        "goal": list(map(float, goal)) if goal is not None else None,
        "grounder": "regex",
        "success": success,
        "failure_mode": failure_mode,
        "raw_match": raw_match,
        "notes": notes,
    }


def ground_t0(instruction: str) -> dict:
    m = NAMED_COORD_RE.search(instruction) or COORD_RE.search(instruction)
    if not m:
        return _result(None, success=False, failure_mode="parse_error",
                       notes="no coordinate-like substring found")
    x, y, z = m.group(1), m.group(2), m.group(3)
    goal = [float(x), float(y), float(z) if z is not None else WORKSPACE["z"]]
    return _result(goal, success=True, raw_match=m.group(0))


def ground_t1(instruction: str) -> dict:
    s = instruction.lower()
    for (v_word, h_word), (x_key, y_key) in REGIONS.items():
        if v_word in s and h_word in s:
            goal = [WORKSPACE[x_key], WORKSPACE[y_key], WORKSPACE["z"]]
            return _result(goal, success=True, raw_match=f"{v_word} {h_word}")
    if "center" in s or "middle" in s or "centre" in s:
        goal = [(WORKSPACE["x_min"] + WORKSPACE["x_max"]) / 2,
                (WORKSPACE["y_min"] + WORKSPACE["y_max"]) / 2,
                WORKSPACE["z"]]
        return _result(goal, success=True, raw_match="center")
    return _result(None, success=False, failure_mode="region_unknown",
                   notes="no known region keyword found")


def ground_t2(instruction: str, cube_pos) -> dict:
    if cube_pos is None:
        return _result(None, success=False, failure_mode="missing_context",
                       notes="T2 requires cube_pos in context")
    m = OFFSET_RE.search(instruction)
    if not m:
        return _result(None, success=False, failure_mode="parse_error",
                       notes="no '<n> <unit> <direction>' phrase")
    magnitude = float(m.group(1))
    unit = m.group(2).lower()
    direction = m.group(3).lower()
    if unit.startswith("cm") or unit.startswith("centi"):
        magnitude /= 100.0
    if direction not in DIR_VECTORS:
        return _result(None, success=False, failure_mode="parse_error",
                       notes=f"unknown direction word: {direction}")
    cube = np.asarray(cube_pos, dtype=float).reshape(-1)
    if cube.shape[0] < 3:
        cube = np.concatenate([cube, [WORKSPACE["z"]]])
    goal = cube[:3] + magnitude * DIR_VECTORS[direction]
    return _result(goal.tolist(), success=True, raw_match=m.group(0))


def ground(instruction: str, tier: str, context: Optional[dict] = None) -> dict:
    """Regex grounder dispatch. Returns a grounder-output dict.

    `context` may carry: cube_pos=[x,y,z] (required for T2).
    """
    out = {"instruction": instruction, "tier": tier}
    if tier == "T0":
        out.update(ground_t0(instruction))
    elif tier == "T1":
        out.update(ground_t1(instruction))
    elif tier == "T2":
        cube_pos = (context or {}).get("cube_pos")
        out.update(ground_t2(instruction, cube_pos))
    elif tier in ("T3", "T4"):
        out.update(_result(None, success=False,
                           failure_mode="unsupported_tier",
                           notes=f"regex baseline does not address {tier}"))
    else:
        out.update(_result(None, success=False,
                           failure_mode="parse_error",
                           notes=f"unknown tier: {tier}"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction", help="natural-language instruction")
    parser.add_argument("--tier", required=True,
                        choices=["T0", "T1", "T2", "T3", "T4"])
    parser.add_argument("--cube-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="cube position; required for T2")
    args = parser.parse_args()

    context = {"cube_pos": args.cube_pos} if args.cube_pos else None
    result = ground(args.instruction, args.tier, context=context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
