"""Parse the LLM's raw text response into a {goal, reasoning, parse_error}
dict. Tolerates: pure JSON, JSON inside ```...``` code fences, JSON
followed by trailing prose. Falls back to a regex substring search.
"""

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from workspace import TABLE_Z  # noqa: E402


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_load(text: str):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def parse_goal(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        return {"goal": None, "reasoning": None, "parse_error": "empty response"}

    stripped = _FENCE_RE.sub("", text).strip()

    obj, err = _try_load(stripped)
    if obj is None:
        m = _BRACE_RE.search(stripped)
        if not m:
            return {"goal": None, "reasoning": None,
                    "parse_error": f"no JSON object found (json err: {err})"}
        obj, err2 = _try_load(m.group(0))
        if obj is None:
            return {"goal": None, "reasoning": None,
                    "parse_error": f"json decode failed: {err2}"}

    if not isinstance(obj, dict):
        return {"goal": None, "reasoning": None,
                "parse_error": f"top-level is not an object: {type(obj).__name__}"}

    goal = obj.get("goal")
    reasoning = obj.get("reasoning")

    if not isinstance(goal, list) or len(goal) not in (2, 3):
        return {"goal": None, "reasoning": reasoning,
                "parse_error": f"goal must be a 2- or 3-list, got {goal!r}"}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in goal):
        return {"goal": None, "reasoning": reasoning,
                "parse_error": f"goal entries must be numbers, got {goal!r}"}

    goal_3d = [float(goal[0]), float(goal[1]),
               float(goal[2]) if len(goal) == 3 else TABLE_Z]
    return {"goal": goal_3d, "reasoning": reasoning, "parse_error": None}
