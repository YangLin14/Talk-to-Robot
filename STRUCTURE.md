# Project Structure

```text
Talk-to-Robot/
├── README.md
├── STRUCTURE.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   └── talk_to_robot/
│       ├── __init__.py
│       ├── workspace.py
│       ├── baselines/
│       │   ├── __init__.py
│       │   └── regex_grounder.py
│       ├── eval/
│       │   ├── __init__.py
│       │   └── harness.py
│       └── grounding/
│           ├── __init__.py
│           ├── client.py
│           ├── grounder.py
│           ├── prompts.py
│           ├── parser.py
│           └── classifier.py
├── scripts/
│   ├── apply_t4_annotations.py
│   ├── eval_policy_fetchpush.py
│   ├── record_rollout.py
│   ├── summarize_results.py
│   ├── train_sac_her_fetchpush.py
│   └── train_sac_her_retrain.py
├── data/
│   ├── instructions/
│   │   └── instructions.json
│   └── annotations/
│       ├── t4_annotations.csv
│       └── t4_annotation_guide.md
├── models/
├── results/
├── notebooks/
└── docs/
    ├── setup.md
    ├── retraining.md
    ├── milestones/
    └── proposal/
```

## Key entry points

| Task | Command |
|------|---------|
| Install package (src layout) | `pip install -e .` |
| Run full eval (LLM, zero-shot) | `python -m talk_to_robot.eval.harness` |
| Run specific tier and variant | `python -m talk_to_robot.eval.harness --tier T3 --variant cot` |
| Run regex baseline | `python -m talk_to_robot.eval.harness --grounder regex --tier T0 T1 T2` |
| Grounding only (no policy) | `python -m talk_to_robot.eval.harness --skip-policy` |
| Summarize results | `python scripts/summarize_results.py` |
| Train from scratch | `python scripts/train_sac_her_fetchpush.py` |
| Retrain on failures | `python scripts/train_sac_her_retrain.py` |
