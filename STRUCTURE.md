# Project Structure

```
Talk-to-Robot/
|
+-- workspace.py                  # single source of truth: coordinate bounds, axis conventions, region table
+-- requirements.txt
+-- .env.example                  # copy to .env and add GEMINI_API_KEY
|
+-- grounding/                    # LLM grounder pipeline
|   +-- client.py                 # Gemini API wrapper with rate limiting and retry
|   +-- grounder.py               # top-level ground() entry point
|   +-- prompts.py                # zero-shot / few-shot / CoT prompt builders
|   +-- parser.py                 # tolerant JSON extractor (handles code fences, missing z)
|   +-- classifier.py             # failure mode labeler (refusal, parse_error, hallucination, out_of_workspace)
|
+-- baselines/
|   +-- regex_grounder.py         # rule-based grounder for T0-T2
|
+-- eval/
|   +-- harness.py                # full eval loop: ground -> rollout -> score -> write JSON
|
+-- instructions/
|   +-- instructions.json         # 100 instructions with ground truth goals and per-tier tolerances
|
+-- scripts/
|   +-- eval_policy_fetchpush.py  # standalone policy rollout with optional fixed goal injection
|   +-- train_sac_her_fetchpush.py
|   +-- train_sac_her_retrain.py  # fine-tunes on grounder failure cases
|   +-- apply_t4_annotations.py   # freezes T4 ground truth from annotator CSV
|   +-- record_rollout.py         # records rollout videos and data
|   +-- summarize_results.py      # aggregates result JSONs into tables and plots
|
+-- results/
|   +-- original/                 # original SAC+HER model with LLM grounder
|   +-- retrained/                # retrained model with LLM grounder
|   +-- regex/                    # all regex grounder runs
|   +-- summary/                  # aggregated tables (CSV, MD) and plots (PNG)
|
+-- models/                       # SAC+HER checkpoints (.zip)
|   +-- sac_her_FetchPush-v4_seed0_best.zip
|   +-- retrain_best_model.zip
|
+-- logs/                         # TensorBoard training logs
|
+-- notebooks/                    # exploratory notebooks
|   +-- env_smoke_test.ipynb
|   +-- inspect_fetchpush.ipynb
|
+-- docs/
    +-- proposal/                 # project proposals (v1, v2)
    +-- milestones/               # milestone reports
    +-- setup.md                  # environment setup and installation
    +-- retraining.md             # retraining procedure
    +-- t4_annotation_guide.md    # instructions for T4 human annotation
    +-- t4_annotations.csv        # raw annotator points for T4 ground truth
```

## Key entry points

| Task | Command |
|------|---------|
| Run full eval (LLM, zero-shot) | `python eval/harness.py` |
| Run specific tier and variant | `python eval/harness.py --tier T3 --variant cot` |
| Run regex baseline | `python eval/harness.py --grounder regex --tier T0 T1 T2` |
| Grounding only (no policy) | `python eval/harness.py --skip-policy` |
| Summarize results | `python scripts/summarize_results.py` |
| Train from scratch | `python scripts/train_sac_her_fetchpush.py` |
| Retrain on failures | `python scripts/train_sac_her_retrain.py` |
