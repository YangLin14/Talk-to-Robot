## Updates:
We landed the two pieces of W2 infrastructure that had not yet been written and were blocking the W3 grounding evaluation: a working policy-evaluation script and the regex-baseline grounder for T0–T2. The trained SAC+HER policy from W2 (`models/sac_her_FetchPush-v4_seed0_best.zip`, ~95–100% success around 600–700k timesteps) is now exercisable through a clean entry point that accepts either env-sampled goals or an externally-supplied goal vector. This is the same interface the LLM grounder will use, so swapping grounders in W3/W4 will not require touching the controller side.

The policy-evaluation script `scripts/eval_policy_fetchpush.py` loads the saved checkpoint, resets the environment with a configurable seed, and optionally overrides the env's randomly-sampled goal by writing the supplied vector to `env.unwrapped.goal` and patching the first observation's `desired_goal` field. Per-episode records (`success`, `final_distance`, `steps`, `achieved_goal`, `desired_goal`) are printed and can be written to JSON for downstream analysis. The script's two modes correspond directly to ablation row A (controller-only sanity, env-sampled goals) and rows D/F (injected goals from a grounder).

The regex baseline lives in `baselines/regex_grounder.py` and exposes a single `ground(instruction, tier, context=None)` entry point. The return shape — `{instruction, tier, goal, grounder, success, failure_mode, raw_match, notes}` — is the shared schema we will reuse for the Gemini grounder so the W3 eval harness can stay grounder-agnostic. T0 uses a numeric-coordinate regex (with or without parens, 2D or 3D), T1 keyword-matches against a small region table (corners plus center), and T2 parses `<n> <unit> <direction>` phrases, converts centimeters to meters, and applies the offset to a cube position supplied via `context`. T3 and T4 deliberately return `success=False` with `failure_mode="unsupported_tier"` so that the eval harness records them as a fair-baseline limit (proposal §8, condition F) rather than as parse errors that would bias the comparison against regex.

Both files were syntax-checked, and the two regex patterns were verified against the canonical example strings from the proposal (`(1.20, 0.50)`, `10 cm to the right`, etc.) plus a handful of negative cases. The numpy- and gymnasium-dependent code paths still need a smoke run inside the `spatial-rl-311` conda environment.

## Coordinate-frame assumptions (pending team confirmation)
Two sets of constants in `regex_grounder.py` encode workspace assumptions that the team should validate before authoring the full instruction library. They are isolated at the top of the file so a single edit propagates everywhere:

- `WORKSPACE` — `x ∈ [1.15, 1.45]`, `y ∈ [0.55, 0.95]`, `z = 0.42`. T1 region targets ("upper-left corner", etc.) are derived from these bounds.
- `DIR_VECTORS` — assumes robot frame: +x forward, +y to the robot's left, +z up. T2's "right" therefore maps to −y. T2 ground truth depends on this convention.

The proposal text uses 2D `(x, y)` coordinates throughout, but `FetchPush-v4`'s `desired_goal` is shape `(3,)`. The grounder defaults the z component to the workspace table height when only an `(x, y)` pair is parsed. The team should confirm this is the convention we want before finalizing the instruction-library schema.

## Open dependencies for W3
- **Instruction library (`instructions/tiers.json`).** Not yet authored. T0–T2 can be drafted quickly once the workspace/axis decisions land. T3 needs reference-object placement scenarios. T4 requires the three-annotator pre-registered hulls described in proposal §10; these must be frozen before any Gemini call to keep the evaluation valid.
- **`GEMINI_API_KEY`.** Required to smoke-test the grounding harness end-to-end (Step 3). The `google-genai==2.0.1` SDK is already pinned in `requirements.txt`.

## Report Summary
W2 deliverables were a trained SAC+HER policy, a complete 100-instruction tier library, and a grounding harness producing JSON logs. The policy is in good shape from the prior week's training. This milestone closes the W2 gap on the controller side by adding a reusable policy-evaluation script, and lands the regex baseline that will sit alongside the LLM grounder in the W3 evaluation. The instruction library and the LLM grounding harness remain the two W2 items still outstanding; the regex grounder's output schema is now the contract those follow-on components will implement against.
