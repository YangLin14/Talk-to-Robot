"""Apply T4 human annotation points to data/instructions/instructions.json.

Input CSV columns:
  id,annotator,x,y,z

Each T4 instruction should have at least 3 annotator rows. The script computes:
  - annotator_points: the submitted points, sorted by annotator label
  - ground_truth_goal: mean of the annotator points
  - tolerance_m: mean pairwise annotator distance + --buffer-m
  - annotation_status: confirmed

Example:
  python scripts/apply_t4_annotations.py data/annotations/t4_annotations.csv
"""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


DEFAULT_INSTRUCTIONS = Path("data/instructions/instructions.json")


def _load_points(csv_path):
    grouped = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "annotator", "x", "y", "z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            entry_id = row["id"].strip()
            annotator = row["annotator"].strip()
            point = [
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ]
            grouped.setdefault(entry_id, {})[annotator] = point
    return grouped


def _mean_pairwise_distance(points):
    distances = [
        float(np.linalg.norm(np.array(a) - np.array(b)))
        for a, b in combinations(points, 2)
    ]
    return float(np.mean(distances)) if distances else 0.0


def apply_annotations(instructions, annotations, buffer_m, min_annotators):
    updated = 0
    for entry in instructions["instructions"]:
        entry_id = entry.get("id")
        if entry.get("tier") != "T4" or entry_id not in annotations:
            continue

        annotator_map = annotations[entry_id]
        if len(annotator_map) < min_annotators:
            raise SystemExit(
                f"{entry_id} has {len(annotator_map)} annotators; "
                f"need at least {min_annotators}."
            )

        points = [
            annotator_map[name]
            for name in sorted(annotator_map)
        ]
        mean_goal = np.mean(np.array(points, dtype=float), axis=0)
        tolerance = _mean_pairwise_distance(points) + buffer_m

        entry["annotator_points"] = points
        entry["ground_truth_goal"] = [float(v) for v in mean_goal]
        entry["tolerance_m"] = float(tolerance)
        entry["annotation_status"] = "confirmed"
        entry["design_note"] = (
            f"T4 ground truth frozen from {len(points)} human annotator "
            f"points; tolerance_m = mean pairwise distance + {buffer_m:.3f}m buffer."
        )
        updated += 1

    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_csv", type=Path)
    parser.add_argument(
        "--instructions",
        type=Path,
        default=DEFAULT_INSTRUCTIONS,
        help="Instruction JSON to update.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INSTRUCTIONS,
        help="Where to write the updated instruction JSON.",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=0.03,
        help="Buffer added to mean pairwise annotator distance.",
    )
    parser.add_argument(
        "--min-annotators",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    instructions = json.loads(args.instructions.read_text())
    annotations = _load_points(args.annotations_csv)
    updated = apply_annotations(
        instructions,
        annotations,
        buffer_m=args.buffer_m,
        min_annotators=args.min_annotators,
    )
    args.output.write_text(json.dumps(instructions, indent=2) + "\n")
    print(f"Updated {updated} T4 entries in {args.output}")


if __name__ == "__main__":
    main()
