"""Failure-mode classifier for LLM grounder outputs.

Operates on (raw_text, parsed) without ground truth -- the eval harness
layers GT-aware classification on top (region misidentification, etc.).

Buckets emitted here:
- refusal           : model declined / hedged instead of answering
- parse_error       : no extractable JSON / wrong shape
- hallucination     : goal numerically valid but very far outside workspace
- out_of_workspace  : goal just outside workspace bounds (small drift)
- None              : looks fine; downstream layers decide if it matches GT
"""

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from workspace import WORKSPACE  # noqa: E402


_REFUSAL_PATTERNS = [
    r"\bi (?:cannot|can'?t|won'?t|am unable|'m unable)\b",
    r"\b(?:sorry|apologi[sz]e)\b.*\b(?:cannot|can'?t|won'?t|unable)\b",
    r"\b(?:refuse|decline|not (?:able|possible))\b",
    r"\binsufficient (?:information|context)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# A goal more than this far outside the workspace counts as hallucination
# rather than drift.
_HALLUCINATION_SLACK_M = 0.5


def _looks_like_refusal(raw_text: str) -> bool:
    if not raw_text:
        return False
    return bool(_REFUSAL_RE.search(raw_text))


def _bounds_drift(goal):
    x, y = goal[0], goal[1]
    dx = max(0.0, WORKSPACE["x_min"] - x, x - WORKSPACE["x_max"])
    dy = max(0.0, WORKSPACE["y_min"] - y, y - WORKSPACE["y_max"])
    return max(dx, dy)


def classify(raw_text: str, parsed: dict):
    if parsed.get("parse_error"):
        if _looks_like_refusal(raw_text):
            return "refusal"
        return "parse_error"

    goal = parsed.get("goal")
    if goal is None:
        return "parse_error"

    drift = _bounds_drift(goal)
    if drift == 0.0:
        return None
    if drift > _HALLUCINATION_SLACK_M:
        return "hallucination"
    return "out_of_workspace"
