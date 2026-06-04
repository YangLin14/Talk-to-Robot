# T4 Annotation Guide

T4 instructions are functional or intent-level commands such as "get it out of the way".
They do not have a single obvious coordinate answer, so they must be frozen before LLM
evaluation using human annotator points.

## Goal

For each T4 instruction, collect one acceptable target point from three human annotators.
The project then computes:

- `ground_truth_goal`: mean of the three points
- `tolerance_m`: mean pairwise distance between annotator points plus a small buffer
- `annotation_status`: `confirmed`

This avoids choosing the ground truth after seeing LLM outputs.

## Coordinate Frame

Use the project workspace from `src/talk_to_robot/workspace.py`:

- `x` in `[1.19, 1.49]`
- `y` in `[0.60, 0.90]`
- `z = 0.42`
- `+x` means forward / farther from the robot
- `-x` means back / nearer to the robot
- `+y` means left
- `-y` means right

Annotators should choose a point that satisfies the instruction while staying inside
the workspace unless the instruction explicitly implies an edge.

## CSV Format

Create `data/annotations/t4_annotations.csv` or another CSV with this format:

```csv
id,annotator,x,y,z
T4_001,A,1.49,0.60,0.42
T4_001,B,1.49,0.75,0.42
T4_001,C,1.49,0.90,0.42
T4_002,A,1.49,0.60,0.42
T4_002,B,1.49,0.75,0.42
T4_002,C,1.49,0.90,0.42
```

Each `T4_*` id needs at least three annotator rows.

## Applying Annotations

Run:

```bash
python scripts/apply_t4_annotations.py data/annotations/t4_annotations.csv
```

This updates `data/instructions/instructions.json` in place. To write to a separate file first:

```bash
python scripts/apply_t4_annotations.py data/annotations/t4_annotations.csv \
  --output /tmp/instructions_with_t4.json
```

After applying, check:

```bash
jq -r '.instructions[] | select(.tier=="T4") | [.id,.annotation_status,.ground_truth_goal,.tolerance_m] | @tsv' data/instructions/instructions.json
```

## Annotation Rules

- Do not use LLM output to choose or adjust annotator points.
- Freeze T4 annotations before running final LLM evaluation.
- If three annotators disagree strongly, keep the entry but mention low agreement in the report, or mark it for exclusion before evaluation.
- Keep all points on the table surface with `z = 0.42`.
