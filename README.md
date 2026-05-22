# Talk-to-Robot

Studying when one-shot LLM language grounding is enough for goal-conditioned robotic manipulation. SAC+HER policy on FetchPush-v4, Gemini 2.5 Flash frontend, regex baseline for grounding-necessity ablation.

## Teammates

Aditya Jadhav
Fong-Yu (Yang) Lin
Joseph
Nasser Al Nasser
Waleed Alghaithi

## Intro

Humans don't speak in coordinates. This project studies the simplest reliable way to bridge natural-language instructions and learned robotic controllers, in simulation. We compare a decoupled pipeline (LLM queried once per task) against a regex baseline across instruction tiers from literal to abstract, on the FetchPush-v4 MuJoCo benchmark. UCSD CSE 190 Spring 2026.

## Current Pipeline

1. Load an instruction from `instructions/instructions.json`.
2. Ground the instruction into a 3D goal with either:
   - regex baseline: `baselines/regex_grounder.py`
   - Gemini LLM grounder: `grounding/grounder.py`
3. Score the predicted goal against the tier's ground truth.
4. Optionally roll out the trained SAC+HER policy in FetchPush-v4.
5. Write JSON results and generate tables/plots.

Regex grounding itself does not require a GPU. Policy rollout also works on CPU, though GPU can be faster if available.

## Setup

See `docs/setup.md` for the full environment setup. For LLM runs, create a local `.env`:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
GEMINI_API_KEY=your_key_here
```

Do not commit `.env`.

## Running Evaluations

Grounding-only regex baseline, no SAC model rollout:

```bash
python eval/harness.py --grounder regex --tier T0 T1 T2 --skip-policy \
  --output results/regex_grounding_only.json
```

Regex baseline with policy rollout on CPU:

```bash
python eval/harness.py --grounder regex --tier T0 T1 T2 --device cpu \
  --n-episodes 1 --output results/regex_t0_t1_t2.json
```

LLM grounding with Gemini:

```bash
python eval/harness.py --grounder llm --tier T0 T1 T2 T3 \
  --variant zero-shot --device cpu --n-episodes 1 \
  --output results/llm_zero_shot_t0_t3.json
```

Prompt variants are `zero-shot`, `few-shot`, and `cot`.

T4 is skipped until human annotations are confirmed. See `docs/t4_annotation_guide.md`.

## Tables And Plots

Generate report-ready CSV, Markdown, and PNG summaries from harness JSON files:

```bash
python scripts/summarize_results.py results/*.json
```

Outputs are written to `results/summary/`:

- `summary.csv`
- `details.csv`
- `summary.md`
- `success_rates.png`
- `failure_modes.png` when failures exist

CSV/Markdown generation uses only the Python standard library. PNG plots require a working `matplotlib` install in the active environment. A notebook is optional for exploration, but the reproducible path should be the script.

## Recording A Rollout

Record one grounded instruction as video:

```bash
python scripts/record_rollout.py --instruction-id T1_001 --grounder regex \
  --device cpu --output videos/t1_001_regex.mp4
```

For GIF output:

```bash
python scripts/record_rollout.py --instruction-id T2_001 --grounder regex \
  --device cpu --output videos/t2_001_regex.gif
```

For LLM-grounded video, make sure `.env` contains `GEMINI_API_KEY` first:

```bash
python scripts/record_rollout.py --instruction-id T3_001 --grounder llm \
  --variant zero-shot --device cpu --output videos/t3_001_llm.mp4
```

You can also enter your own instruction without using the instruction library:

```bash
python scripts/record_rollout.py --instruction "push to the upper-left corner" \
  --tier T1 --grounder regex --device cpu --output videos/custom_t1_regex.mp4
```

For custom T2 instructions, the script uses the live cube position from the
current reset unless `--cube-pos X Y Z` is provided:

```bash
python scripts/record_rollout.py --instruction "push it 5 cm to the right" \
  --tier T2 --grounder regex --device cpu --output videos/custom_t2_regex.mp4
```

For custom T3 instructions, provide reference objects:

```bash
python scripts/record_rollout.py --instruction "push it next to the marker" \
  --tier T3 --grounder llm --variant zero-shot \
  --reference-object marker 1.40 0.68 0.42 \
  --device cpu --output videos/custom_t3_llm.mp4
```

For color-reference demos, the single physical FetchPush cube is colored based on
the first cube color named in the instruction. The other colored cube is sampled
as a random reference object at reset time and shown in the video mini-map. Omit
`--seed` for a fresh random placement every run; pass `--seed 0` or another
integer when you want reproducible placement:

```bash
python scripts/record_rollout.py \
  --instruction "push the red cube to the right side of the blue cube" \
  --tier T3 --grounder llm --variant zero-shot \
  --device cpu --output videos/red_to_right_of_blue.mp4

python scripts/record_rollout.py \
  --instruction "push the blue cube to the right side of the red cube" \
  --tier T3 --grounder llm --variant zero-shot \
  --device cpu --output videos/blue_to_right_of_red.mp4
```

Current limitation: only one cube is physically simulated and pushable at a time.
The other cube is rendered in the main MuJoCo scene as a non-colliding visual
reference marker and also shown in the mini-map. Making both cubes physically
pushable in the same MuJoCo scene requires a custom two-object environment and
retraining or replacing the policy.

## Proposal

[CSE190 Final Project Proposal Presentation Slides](https://docs.google.com/presentation/d/1CBNM3xPBS_gioZHrDHAHn269PESMauUR0sPaC5Z7mcA/edit?usp=sharing)
