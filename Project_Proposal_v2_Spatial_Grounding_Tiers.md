# Where Does LLM Spatial Grounding Break? An Instruction-Tier Study for Goal-Conditioned Robotic Manipulation

**Course:** CSE 190 — Intro to Deep RL (Spring 2026, UCSD)
**Team Size:** 5
**Timeline:** 5 weeks (April 28 – May 30, 2026)
**Version:** v2 — revised after Pitch 1 feedback

---

## Revision Notes (v1 → v2)

Following Pitch 1 feedback from the instructor, we have refocused this proposal. The original framing — *decoupled-pipeline-vs-coupled-architecture* — was identified as engineering territory that has been explored before, with the LLM-to-JSON extraction step being a solved problem rather than a research question. The genuine open problem lives one layer earlier: **how an LLM transitions free-form natural language into spatial coordinates, and where that grounding breaks as instructions become less literal.** The revised proposal foregrounds this as the central research question, organized around an explicit hierarchy of instruction abstraction. The pipeline architecture, RL controller, and simulation environment remain as supporting infrastructure.

---

## Abstract

A robot deployed in a real environment will not be commanded by coordinate vectors — it will be told things like "push it into the corner," "move it next to the marker," or "put it somewhere safer." The hard part of language-driven robotics is not formatting an LLM's output as JSON. The hard part is the **language-to-coordinate transition itself**: how reliably does a Large Language Model ground a free-form spatial instruction into a target position in the robot's frame, and how does that reliability degrade as instructions move from literal coordinates to abstract intent?

This project studies that transition empirically, in simulation. We define a five-tier hierarchy of instruction abstraction, from explicit coordinates (Tier 0) to functional/intent-level instructions (Tier 4). For each tier, we measure (a) the LLM's grounding accuracy against ground-truth target positions, (b) the failure-mode distribution (coordinate drift, region misidentification, refusal, hallucination), and (c) the propagation of grounding error into downstream task success when the LLM-emitted goal is executed by a goal-conditioned RL policy.

We implement the system on the FetchPush-v3 MuJoCo environment using SAC with Hindsight Experience Replay (HER) as the controller and Gemini 2.5 Flash as the grounding frontend. Our central contributions are: (1) the explicit instruction-tier hierarchy as an evaluation tool, (2) per-tier grounding-accuracy and failure-mode characterization for a frontier LLM, and (3) end-to-end measurement of how grounding error translates into manipulation outcomes. We position this work as an empirical first step toward language-driven robotics, conducted entirely in simulation, and we do not claim novelty for the pipeline architecture itself.

---

## 1. What Task Are We Doing?

We study **language-conditioned goal grounding for robotic pushing**. A user issues a natural-language instruction; an LLM translates it into a 3D goal coordinate; a pre-trained goal-conditioned RL policy pushes a cube to that goal in MuJoCo. The research question is not whether this composition works (it does, in the trivial case), but **how the LLM's grounding reliability scales — or breaks — across a structured hierarchy of instruction types**.

### 1.1 The Instruction Hierarchy (Central Object of Study)

| Tier | Name | Example | What the LLM Must Do |
|---|---|---|---|
| **T0** | Coordinate literal | "push to (1.20, 0.50)" | Extract numbers. Tests whether trivial cases are reliable. Baseline floor. |
| **T1** | Region reference | "push to the upper-left corner" | Map a named region to its coordinate. Requires a learned spatial vocabulary. |
| **T2** | Relative motion | "move it 10 cm to the right" | Read the cube's current position from context, apply a vector offset. Requires state-aware reasoning. |
| **T3** | Reference object | "push it next to the marker" | Resolve a reference object, then ground a relational predicate ("next to"). Requires multi-object reasoning. |
| **T4** | Functional / intent | "put it somewhere out of the way" | Infer intent, generate a plausible goal under under-specification. Requires open-ended commonsense. |

For each tier we generate **20 distinct instructions**, each evaluated against a ground-truth target position. Tier difficulty is operationalized by what the LLM must reason over, not by surface complexity of the sentence.

### 1.2 Stage 2: Control

Once the LLM emits a goal vector, a goal-conditioned RL policy — SAC + HER trained on FetchPush-v3 — executes for up to 50 timesteps to push the cube to that goal. The policy is trained once on ground-truth coordinate goals and reused as a fixed downstream consumer for all LLM-generated goals at evaluation time. **The policy is not retrained for each tier**; we are studying the LLM's grounding behavior, not adapting the controller to language.

### 1.3 Why This Is RL, Not Solvable by an LLM Alone

The downstream control problem requires continuous closed-loop control over a 7-DoF arm in contact with a rigid object whose dynamics depend on uncertain friction. LLMs cannot reliably emit motor commands at this granularity. The RL policy provides this lower layer, allowing us to isolate the **grounding** problem from the **control** problem and study it cleanly.

---

## 2. Research Questions

**Primary RQ.** As instructions move up the abstraction hierarchy (T0 → T4), how does LLM grounding accuracy degrade? Is the degradation gradual or does it exhibit a cliff?

**Secondary RQ-A (Failure Modes).** When the LLM fails, *how* does it fail? We classify failures into:
- *Coordinate drift* — correct region, off-by-distance
- *Region misidentification* — wrong region entirely
- *Refusal / hedge* — "I cannot determine"
- *Hallucination* — references nonexistent objects or coordinates outside the workspace
- *Format violation* — output cannot be parsed

The distribution of these modes, per tier, is itself a contribution.

**Secondary RQ-B (Error Propagation).** Given a grounding error of magnitude *δ*, what is the downstream task success rate? Does the RL policy tolerate small grounding errors gracefully, or is the manipulation task brittle to coordinate noise?

**Secondary RQ-C (Prompt Sensitivity).** For tiers that show high failure rates, can prompt restructuring (chain-of-thought, few-shot exemplars, explicit coordinate frame description) shift the failure curve? This tier-specific intervention question gives us a methods contribution beyond pure evaluation.

---

## 3. Environment

### 3.1 Base Environment

**FetchPush-v3** from Gymnasium-Robotics, a public MuJoCo-based environment simulating a 7-DoF Fetch manipulator pushing a cube on a table to a target position.

- **Observation:** robot end-effector state, cube position, desired goal (~25 dims continuous)
- **Action:** 4-D continuous Cartesian end-effector displacement (IK handled internally)
- **Reward:** sparse binary — 1 if cube within ε of goal, else 0
- **Episode horizon:** 50 timesteps

### 3.2 Are We Building a Simulator?

**No.** FetchPush-v3 is used unmodified for control training. The novel infrastructure we build is:

1. **Instruction Tier Library** — curated set of 100 instructions (20 per tier × 5 tiers) with ground-truth target positions
2. **LLM Grounding Wrapper** — handles prompting, retries, parsing, failure-mode logging
3. **Evaluation Harness** — for each instruction, runs the LLM, records grounding output, classifies failures, executes downstream RL policy, records task success
4. **Failure-Mode Classifier** — rule-based + manual review for ambiguous cases

### 3.3 LLM Frontend

**Gemini 2.5 Flash via Google AI Studio.** Free indefinite tier (15 RPM, 1M tokens/day, no expiration, no credit card). Each call ~300 tokens; the daily cap supports thousands of evaluations. Fallback: Groq Llama-3.3-70B free tier.

Each call uses structured output mode forcing JSON: `{"goal_x": ..., "goal_y": ..., "goal_z": ..., "reasoning": "..."}`. The `reasoning` field gives us an interpretable trace for failure-mode analysis. **Note:** structured output guarantees valid JSON; format violations come from the LLM refusing or returning out-of-range coordinates, not from parsing failures.

---

## 4. Why Is This Interesting? Who Will Care?

### 4.1 What We Are *Not* Claiming

We do not claim:
- Novelty for decoupled pipeline architectures (LM-Nav 2022, BabyAI 2019, and earlier instruction-grounding work explored these)
- Novelty for using LLMs as structured-output extractors (engineering, not research)
- Sim-to-real transfer (we operate purely in simulation)

### 4.2 What We Are Claiming

We claim:
- A **structured evaluation methodology** for LLM spatial grounding, organized around an explicit instruction-tier hierarchy
- An **empirical characterization** of where grounding breaks for a frontier LLM (Gemini 2.5 Flash) on a continuous-control manipulation task
- A **failure-mode taxonomy** with measured distributions per tier — not just "accuracy went down" but "failures changed character"
- A measurement of **error propagation** from grounding into downstream control, addressing whether grounding accuracy thresholds matter for end-to-end success

### 4.3 Who Will Care

- Anyone deploying robots that humans will instruct in natural language. The grounding-accuracy curve directly determines what instruction styles are safe to expose to users.
- Researchers studying LLM spatial reasoning. Continuous-control manipulation gives a different test surface than the gridworld and navigation benchmarks where most current grounding-accuracy work sits.
- Course staff and the PEARLS Lab. The work engages the LLM-agent and simulation-environment lecture content directly, with rigorous empirical claims rather than architectural rebranding.

### 4.4 Course Topic Mapping

- Simulated Environments — we use FetchPush-v3 and discuss its abstraction limits
- Deep RL (pre-LLM) — SAC and HER directly
- LLM Basics — Gemini grounding leverages frontier LLM capability
- LLM Agent Methods — structured tool use, prompt engineering, failure analysis

---

## 5. Initial Approach and Fallback Plans

### 5.1 Approach

**Step 1 — Train the controller.**
SAC + HER on FetchPush-v3 with ground-truth coordinate goals, target ≥ 70% success on standard test. This is well-established and not a research output, only infrastructure.

**Step 2 — Build the instruction-tier library.**
20 instructions per tier × 5 tiers = 100 instructions. Each instruction is paired with a ground-truth coordinate (or set of acceptable coordinates for the abstract tiers). For T4 (functional/intent), the "ground truth" is the *set* of coordinates a panel of three humans agrees would satisfy the instruction; the LLM is correct if its output falls within this set. **This is the most fragile design choice and is described in detail in Section 5.4.**

**Step 3 — Run grounding evaluation.**
For each instruction, query Gemini, log output, classify failure mode if any, compute coordinate error magnitude.

**Step 4 — Run end-to-end evaluation.**
Take each LLM-emitted goal, execute the RL policy 10 times, record success rate. This produces the error-propagation curve.

**Step 5 — Prompt-sensitivity intervention.**
For the worst-performing tier(s), test 3 prompt variants (zero-shot, few-shot with 3 exemplars, chain-of-thought with explicit coordinate-frame description) and report the per-variant failure curve.

### 5.2 Ablation Table

| Condition | Tier coverage | Purpose |
|---|---|---|
| **A. Controller-only sanity** | Ground-truth goals | Validate SAC+HER works (must hit ≥70%) |
| **B. Per-tier grounding** | T0–T4 | Primary RQ |
| **C. Failure-mode distribution** | T0–T4 | Secondary RQ-A |
| **D. Error propagation** | T0–T4 | Secondary RQ-B |
| **E. Prompt-sensitivity intervention** | Worst tier(s) | Secondary RQ-C |
| **F. Regex-baseline comparison** | T0–T2 only | Sanity bound — does LLM beat regex on tiers regex *can* address? |

Condition F is intentionally limited to T0–T2 because regex parsing cannot meaningfully address T3–T4; including it everywhere would make the LLM look artificially better. We are honest about where the comparison is fair.

### 5.3 What If the LLM Doesn't Work as Expected

**Failure mode: Gemini refuses or hedges on T3–T4.**
Not a project failure. Refusal rate is itself a measured failure mode (Secondary RQ-A). We report it.

**Failure mode: Gemini saturates near 100% even on T4.**
Unlikely given current LLM literature on spatial grounding, but if this happens we extend the hierarchy with harder tiers (e.g., counterfactual references, instruction-with-distractors). Document the extension.

**Failure mode: Gemini quota exhausted.**
Switch to Groq Llama-3.3-70B. Re-run T0 on both providers as a calibration check; report any cross-model differences.

### 5.4 What If the RL Doesn't Converge

**Hard fallback by end of Week 2.**
If SAC + HER does not reach 70% on FetchPush, switch to **FetchReach-v3** (reaching task). The instruction-tier methodology transfers identically; only the action target changes from "push cube to coordinate" to "move end-effector to coordinate." Reaching is provably solvable by SAC + HER in well under 1M timesteps.

**The grounding study is the contribution; the controller is infrastructure.** Switching tasks does not change the research narrative.

### 5.5 What If T4 Ground Truth Is Too Subjective

This is the most exposed methodological risk. Mitigation:
- For T4, ground truth is defined as the **convex hull** of three human annotators' chosen coordinates (or a tolerance region around their mean)
- We pre-register each T4 instruction's ground-truth region before running the LLM, to prevent post-hoc adjustment
- We report inter-annotator agreement as a check on whether T4 is well-defined enough to evaluate; if agreement is low for a given instruction, we flag and exclude it

### 5.6 Decision Gates

| Week | Gate | Action if Failed |
|---|---|---|
| End of W2 | SAC+HER converges on FetchPush | Switch to FetchReach |
| End of W3 | At least T0–T2 evaluation runs end-to-end | Drop T4, focus on T0–T3 |
| End of W4 | Ablation table populated | Drop Conditions E or F if behind |

---

## 6. Five-Week Timeline

| Week | Goals | Deliverables | Decision Gate |
|---|---|---|---|
| **W1 (4/28–5/4)** | Pivot framing per Pitch 1 feedback. Setup complete. SAC+HER smoke test. Instruction-tier library v1 (T0–T1, ~40 instructions). | Updated pitch deck, working `pip install`, smoke training, T0–T1 instruction set with ground truth | Did everyone install? Is tier library v1 complete? |
| **W2 (5/5–5/11)** | Full SAC+HER training. T2–T4 instruction library complete. Begin grounding evaluation harness. | Trained policy, full 100-instruction library, grounding harness produces JSON logs | **Did SAC+HER converge?** If not → FetchReach |
| **W3 (5/12–5/18)** | Run grounding evaluation across all tiers. Failure-mode classifier. Begin error-propagation runs. | Per-tier grounding accuracy table, failure-mode distribution, partial error-propagation data | **Does T3 or T4 evaluation produce meaningful results?** If not → drop to T0–T3 |
| **W4 (5/19–5/25)** | Final runs (3 seeds where applicable). Prompt-sensitivity intervention on worst tier. Regex baseline on T0–T2. Begin report. | Master results: tier × accuracy, tier × failure mode, intervention curve. Draft report sections 1–4. | Compute and time on track? |
| **W5 (5/26–5/30)** | Demo video (voice instruction → robot motion across tiers). Polish GitHub. Final report and slides. Submit by 5/30. | Final report, demo video, GitHub repo, slides | Final submission |

Submitting on 5/30 leaves week of 6/1 finals untouched.

---

## 7. Team Roles

| Member | Role | Primary Outputs |
|---|---|---|
| **Yang (Tech Lead)** | Architecture, RL pipeline, tier-library design, supercomputer runs, technical writing | Training pipeline, tier hierarchy, final-run management, Methods + Results sections |
| **Coder #2** | LLM wrapper, evaluation harness, prompt engineering for intervention experiments | Grounding wrapper, evaluation harness, prompt variants for Section 5.1 Step 5 |
| **Member #3** | Failure-mode classifier, plotting, statistical analysis | Failure-mode taxonomy, per-tier plots, error-propagation curves |
| **Member #4** | Instruction-tier library curation, regex baseline, related work | 100-instruction library with ground truth, regex parser for T0–T2, related work section |
| **Member #5** | Demo video, slide decks, README, report integration, T4 human-annotation panel coordination | Demo video showing tier progression, pitch deck, README, T4 annotator agreement check |

**Sync cadence:** full team Mondays, RL pair Wednesdays, async daily Slack.

---

## 8. Compute and API Resources

| Resource | Owner | Use |
|---|---|---|
| Google Colab Free | All members | Smoke tests, evaluation runs |
| UCSD Supercomputer | Yang (will request) | Final RL training runs in W2 and W4 |
| Google AI Studio (Gemini 2.5 Flash) | All members | LLM grounding |
| Groq free tier (Llama-3.3-70B) | Backup | LLM fallback if Gemini limited |

**Estimated usage:** ~30–50 GPU-hours total on supercomputer, ~10,000 LLM API calls total (well under 1M tokens/day cap).

---

## 9. Success Criteria

**Minimum Viable Project (must achieve):**
- SAC+HER on FetchPush ≥ 60% success on coordinate goals
- T0–T2 grounding evaluation complete with per-tier accuracy and failure-mode distribution
- One end-to-end demo connecting natural-language instruction → grounding → robot motion

**Target Outcome (likely to achieve):**
- Full T0–T4 grounding evaluation across all 100 instructions
- Failure-mode taxonomy with measured distributions
- Error-propagation curve linking grounding error magnitude to task success
- Regex baseline on T0–T2

**Stretch Goals (if time permits):**
- Prompt-sensitivity intervention on worst-performing tier
- Cross-LLM comparison (Gemini vs. Llama via Groq)
- Extended hierarchy (T5+ with adversarial or distractor instructions)

---

## 10. Honest Limitations

- We do not claim novelty for the pipeline paradigm; we claim a structured evaluation methodology for spatial grounding.
- All work is in simulation. We do not address sim-to-real transfer.
- We use a single frontier LLM (Gemini 2.5 Flash) as the primary subject; cross-model generalization is a stretch goal.
- T4 (functional/intent) ground truth depends on human annotation, with the methodological risks described in Section 5.5.
- 100 instructions is a small sample; we report effect sizes with appropriate uncertainty intervals rather than claiming statistical significance broadly.
- Three random seeds for RL training is a minimum; we report variance honestly.

---

## 11. Key References

- Andrychowicz et al. 2017, *Hindsight Experience Replay*
- Haarnoja et al. 2018, *Soft Actor-Critic*
- Plappert et al. 2018, *Multi-Goal RL: Challenging Robotics Environments*
- Misra et al. 2017, *Mapping Instructions to Actions* — early instruction grounding
- Shah et al. 2022, *LM-Nav* — pipeline-style grounding
- Chevalier-Boisvert et al. 2019, *BabyAI* — grounding evaluation methodology
- Ahn et al. 2022, *SayCan* — closed-loop grounding (contrast)
- Gemini 2.5 documentation, Google AI Studio
- Recent literature on LLM spatial reasoning evaluation (SpatialRGPT, SpatialBot, 2024–2025) — to be added in W1 lit review

---

*Last updated: April 28, 2026, post-Pitch 1*
