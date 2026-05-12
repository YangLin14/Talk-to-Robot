# Language-Grounded Goal-Conditioned Reinforcement Learning for Robotic Pushing

**Course:** CSE 190 — Intro to Deep RL (Spring 2026, UCSD)
**Team Size:** 5
**Timeline:** 5 weeks (April 28 – May 30, 2026)

---

## Abstract

When humans ask each other for help, they speak in natural language: "put it in the corner," "move it forward," "send it over there." But the robots we train today expect structured goal vectors — explicit coordinates in a known frame. As robots move closer to operating alongside humans in homes, warehouses, and labs, the gap between how humans give instructions and how robots interpret them becomes a deployment bottleneck. **This project studies, in simulation, how to bridge that gap with the simplest architecture that works.**

We focus on a **decoupled pipeline architecture**, where a Large Language Model (LLM) is queried **once per task** to translate a free-form natural language instruction (e.g., *"push the cube to the upper-left corner"*) into a structured goal vector, and a separately-trained goal-conditioned reinforcement learning (RL) policy executes the task autonomously thereafter. This contrasts with the current trend in robotics-LLM research toward tightly-coupled systems (SayCan, Inner Monologue, ReAct) that query the LLM throughout execution, paying for that flexibility with latency, API cost, and runtime brittleness.

We implement and evaluate the pipeline on the FetchPush-v3 MuJoCo environment using SAC with Hindsight Experience Replay (HER) for the low-level controller and Gemini 2.5 Flash for the language frontend. Our evaluation answers four empirical questions: (1) does HER solve the sparse-reward problem on FetchPush relative to vanilla SAC; (2) how does LLM grounding accuracy degrade as instructions become more abstract (literal → paraphrased → abstract); (3) does end-to-end task success hold when the components are composed; and (4) does the LLM contribute value over a simple regex parser, or is classical parsing sufficient for narrow goal spaces?

Our **primary research question** asks under what task conditions one-shot LLM grounding suffices, and where it begins to fail. Our **secondary research question** asks whether LLMs are necessary at all for narrow goal spaces, or whether classical parsing suffices. **We position this as an empirical first step toward language-driven robotics, conducted entirely in simulation, rather than a sim-to-real claim.** This scope is rigorously defensible within five weeks and produces directly comparable architectural data points that future tightly-coupled or sim-to-real work can build on.

---

## 1. What Task Are We Doing?

A robot must push a block to a target location specified by a free-form natural language instruction. The pipeline has two stages:

1. **Stage 1 (Language Grounding):** A user provides an instruction in natural language. An LLM translates the instruction into a structured goal — specifically, a 3D target position vector for the cube.
2. **Stage 2 (Control):** A goal-conditioned RL policy, trained with SAC + HER on FetchPush-v3, accepts the target position and executes a sequence of end-effector displacements until the cube is pushed to the goal or the episode terminates.

Once Stage 1 emits the goal, the LLM is no longer in the loop. The RL policy executes autonomously based purely on physical state and the static goal vector.

Concrete example:

| Input (natural language) | Stage 1 output (goal vector) | Stage 2 (RL execution) |
|---|---|---|
| "push the cube to the upper-left corner" | `[0.20, 0.50, 0.42]` | SAC+HER policy pushes |
| "move the block forward by 10 centimeters" | `[1.30, 0.75, 0.42]` | SAC+HER policy pushes |
| "send it to the back-right area" | `[1.40, 0.95, 0.42]` | SAC+HER policy pushes |

This mirrors the early language-conditioned RL literature (Misra et al. 2017, BabyAI 2019, LM-Nav 2022) but applies it to continuous-control manipulation rather than navigation or gridworld tasks, and uses a frontier LLM for the grounding step rather than a custom-trained instruction parser.

### Why This Is RL, Not Solvable by an LLM Alone

The Stage 2 control problem requires continuous closed-loop control over a 7-DoF arm manipulating a rigid object whose dynamics depend on uncertain friction. LLMs cannot reliably emit motor commands at this granularity (this is empirically demonstrated by SayCan, Inner Monologue, and the LLM+A paper). Stage 2 is squarely an RL problem. Our contribution is showing that the LLM can be removed from the inner loop entirely once the instruction is grounded, and that the system still operates end-to-end.

---

## 2. Environment

### 2.1 Base Environment

**FetchPush-v3** from Gymnasium-Robotics, a publicly available MuJoCo-based environment. It simulates a 7-DoF Fetch manipulator pushing a cube on a table to a target position.

- **Observation:** robot end-effector state, cube position, desired goal (continuous, ~25 dimensions)
- **Action:** 4-dimensional continuous Cartesian end-effector displacement (inverse kinematics handled by MuJoCo internally)
- **Reward:** sparse binary — 1 if cube is within ε of goal at any point, else 0
- **Episode horizon:** 50 timesteps

### 2.2 Are We Building a Simulator?

**No.** We use FetchPush-v3 unmodified for the core training. We build only a thin wrapper that:

1. accepts a natural-language string at episode reset,
2. queries Gemini 2.5 Flash to convert the string into a goal vector,
3. injects that goal vector into the underlying environment.

Building a simulator from scratch would consume 1–2 weeks we cannot spare. Reusing FetchPush-v3 lets us focus on the research question (language-grounding pipeline behavior) rather than infrastructure.

### 2.3 LLM Frontend

**Gemini 2.5 Flash via Google AI Studio.** Free indefinite tier with 15 RPM and 1M tokens/day, no credit card required, no expiration. This is the only major provider with a true indefinite free tier as of April 2026. Each call uses ~300 tokens, so daily cap of 1M tokens supports thousands of evaluations. Backup: Groq's free Llama-3.3-70B if Gemini is rate-limited.

The LLM is given a system prompt describing the table coordinate frame, valid goal regions, and instruction templates. We use structured-output prompting to force the LLM to emit a JSON object containing the goal coordinates.

---

## 3. Time Estimates and Difficulty

### 3.1 What Makes This Hard

- **Sparse reward:** FetchPush has a binary success signal. Vanilla SAC is known to fail on this task without HER. This is precisely why HER is the right tool, and why we expect a clean ablation.
- **LLM grounding accuracy:** The LLM must reliably translate variable natural-language instructions to a small structured goal space. Failures cascade — if Stage 1 outputs the wrong goal, Stage 2's success is irrelevant.
- **Pipeline integration:** End-to-end testing requires both components functional simultaneously. Debugging failures requires distinguishing translation errors from control errors.
- **Hyperparameter sensitivity:** SAC + HER requires tuning of `n_sampled_goal`, `tau`, `learning_rate`, `batch_size`. Wrong choices waste training compute.

### 3.2 What Reduces Risk

- FetchPush-v3 is a **standard, public benchmark** — we are not building infrastructure.
- Stable-Baselines3 has **HER built in** as `HerReplayBuffer`. We do not implement HER from scratch.
- Gemini 2.5 Flash is **free and accessible** with no signup friction.
- The LLM grounding task (free-form instruction to 3D coordinate) is **well within frontier LLM capability** — no fine-tuning required.

### 3.3 Estimated Effort by Component

| Component | Estimated Effort | Risk |
|---|---|---|
| Environment setup, dependency pinning | 2–3 days | Low |
| SAC + HER training pipeline | 4–6 days | Medium |
| LLM prompt design + parser | 2–3 days | Low |
| Pipeline integration | 3–4 days | Medium |
| Evaluation harness (ID + ambiguity tests) | 3–4 days | Low |
| Final runs (3 seeds, 3 conditions) | 5–7 days compute | High (compute-bound) |
| Report, demo video, presentation | 5–6 days | Low |

**Total realistic effort: 5 weeks with a 5-person team, assuming 1–2 strong RL coders and 3 members on supporting tasks.** Expected success probability for the full ablation: 60–65%. Probability of at least the MVP (single-seed SAC + HER + working LLM pipeline): 80%.

---

## 4. Why Is This Interesting? Who Will Care?

### 4.1 Research Interest

The robotics-LLM literature has bifurcated into two camps:

- **Tightly-coupled** systems (SayCan, Inner Monologue, ReAct, Code-as-Policies) where LLMs are queried throughout execution
- **Pipeline / one-shot** systems (LM-Nav, BabyAI, Misra et al. 2017) where LLMs ground instructions once and then step out

The current research consensus has drifted toward tight coupling, citing better long-horizon reasoning. However, this comes with three real costs: latency, API expense, and brittleness when the LLM hallucinates mid-execution. The pipeline approach trades long-horizon flexibility for **deployment simplicity, lower cost, and runtime independence**.

Our contribution is an **empirical characterization** of when this trade-off is favorable. Specifically: for short-horizon, well-defined goal spaces like cube-pushing to a 3D target, does one-shot LLM grounding suffice? At what level of instruction ambiguity does it break? We do not claim to invent the pipeline approach — we claim to rigorously evaluate it on a continuous-control manipulation benchmark, which is underrepresented in the existing pipeline literature (most pipeline work is on navigation or gridworld).

A related but distinct question we also address: **even within the pipeline paradigm, is an LLM necessary at all?** For a narrow 3D goal space, a regex or template-matching parser is a plausible substitute. By comparing LLM grounding against a regex baseline across instruction tiers (literal → paraphrased → abstract), we directly measure where the LLM contributes value rather than assuming it does. This avoids the common failure mode of LLM-augmented systems papers that never test the obvious null hypothesis.

### 4.2 Who Will Care

- **Anyone deploying robots that humans will instruct.** Home robots, warehouse pickers, lab assistants — humans will speak in natural language, not coordinates. The architectural choice between one-shot grounding and per-step coupling directly affects what's deployable on what hardware budget.
- **Robotics deployment engineers.** Anyone with a real product and a real inference budget cares whether a frontier LLM needs to sit in the inner loop or can be amortized across an entire episode.
- **Researchers studying language grounding.** Empirical evidence on the failure modes of pipeline architectures helps calibrate when tight coupling is justified vs. when it's overkill.
- **Course staff and the PEARLS Lab.** The work directly engages with several lecture topics (search and planning, deep RL, LLM agents, tool use, simulation environments) and connects to the lab's broader interest in how language interfaces with structured environments.

### 4.3 Direct Course Topic Mapping

- **Lecture: What Are Agents?** — We design an agent with structured input/output interfaces.
- **Lecture: Simulated Environments** — We use a public MuJoCo simulator and discuss its abstraction choices.
- **Lecture: Deep RL Pre-LLMs** — SAC and HER are direct lecture content.
- **Lecture: LLM Basics** — Gemini-based grounding leverages frontier LLM capability.
- **Lecture: Other LLM Agent Methods** — One-shot translation is an alternative agent design pattern compared to closed-loop tool use.

---

## 5. Initial Ideas on How to Solve It (and What If X Doesn't Work)

### 5.1 Primary Approach

1. **Train SAC + HER on FetchPush-v3 with ground-truth goals.** Get a working policy with success rate ≥ 70% on standard test conditions.
2. **Build LLM grounding wrapper.** Design ~30 instruction templates spanning explicit ("push to coordinates X, Y") to abstract ("push to the back of the table"). Use Gemini 2.5 Flash with structured output to emit `{"goal_x": ..., "goal_y": ..., "goal_z": ...}`.
3. **Compose pipeline.** Test end-to-end with held-out instructions.
4. **Stress test.** Evaluate with ambiguous instructions, instructions with distractors, and OOD phrasings to characterize failure modes.

### 5.2 Ablation Plan

| Condition | Description | Question Answered |
|---|---|---|
| **A. SAC, no HER, ground-truth goals** | RL baseline | Does SAC fail under sparse reward? (Expected: yes, sanity check) |
| **B. SAC + HER, ground-truth goals** | Main RL method | Does HER solve the sparse-reward problem? |
| **C. SAC + HER + LLM goals (literal instructions)** | Pipeline tier 1 | Does the LLM frontend preserve performance on simple instructions? |
| **D. SAC + HER + LLM goals (abstract instructions)** | Pipeline tier 2 | Where does grounding break down? |
| **E. SAC + HER + regex parser (all instruction tiers)** | Grounding necessity test | Is the LLM actually contributing, or could a simple parser substitute? |

**Condition E rationale:** A natural objection to the LLM frontend is that the goal space (3D coordinates on a table) is small enough that a regex- or template-matching parser could plausibly substitute. We test this directly. Expected outcome: regex matches LLM on literal templates ("push to coordinates X, Y"), degrades on paraphrased instructions, and fails on abstract instructions ("send it to the back"). The LLM's value is concretely measured by the gap between Conditions C/D and Condition E across tiers — not assumed.

### 5.3 What If the LLM Doesn't Work?

**Failure mode 1: Gemini emits malformed JSON or wrong coordinates.**
Mitigation: structured output mode + JSON schema validation + retry-on-fail with one re-prompt. If it still fails, fall back to a regex parser on the natural-language instruction. Document failure rate as a finding.

**Failure mode 2: Gemini is rate-limited or down.**
Mitigation: Groq-hosted Llama-3.3-70B as a drop-in replacement. Both expose OpenAI-compatible APIs.

**Failure mode 3: LLM accuracy is bad on abstract instructions.**
This is itself a finding, not a failure. Report it. The weakness of pipeline architectures on ambiguous language is exactly what motivates tightly-coupled systems — confirming this empirically is a contribution.

### 5.4 What If the RL Doesn't Work?

**Failure mode 1: SAC + HER does not converge on FetchPush.**
Mitigation: switch to TD3 + HER (also supported by SB3). Both have HER integration via `HerReplayBuffer`.

**Failure mode 2: Convergence is too slow within compute budget.**
Mitigation: drop to 2 seeds instead of 3 for the baseline condition. Document in limitations.

**Failure mode 3: FetchPush is unexpectedly hard.**
Hard fallback by end of Week 2: switch to **FetchReach-v3** (reaching task, not pushing). The LLM grounding pipeline transfers identically. Reaching is provably solvable by SAC + HER in well under 1M timesteps.

### 5.5 What If Pipeline Integration Reveals an Unforeseen Problem?

The plan has a **decision gate at end of Week 3**. If the pipeline does not produce a single end-to-end success by Friday of Week 3, we drop the LLM frontend and submit a focused **SAC + HER vs. SAC ablation on FetchPush** as a backup project. This is a complete, defensible course project on its own.

---

## 6. Five-Week Timeline

| Week | Goals | Deliverables | Decision Gate |
|---|---|---|---|
| **Week 1** (4/28–5/4) | Setup, dependency lock, SAC+HER smoke test, Gemini API key working, pitch deck | Pitch deck (5/3), working `pip install`, random-policy run, 50K-step smoke training | Did everyone pip install? |
| **Week 2** (5/5–5/11) | Full SAC+HER training to convergence; vanilla SAC baseline; LLM prompt design begins | Working trained policy, baseline failure documented, draft prompt templates | **Did SAC+HER converge?** If not → switch to FetchReach |
| **Week 3** (5/12–5/18) | LLM grounding wrapper, end-to-end pipeline integration, evaluation harness | Pipeline runs end-to-end on at least 5 instructions, evaluation harness produces JSON metrics | **Did the pipeline produce any end-to-end success?** If not → drop LLM frontend |
| **Week 4** (5/19–5/25) | 9 final training runs (3 conditions × 3 seeds) on supercomputer, full evaluation across all instruction tiers, report draft begins | Master results table, learning curves, draft report sections 1–4 | Compute budget on track? |
| **Week 5** (5/26–5/30) | Demo video (with voice instructions), GitHub polish, final report, presentation rehearsal | Final report, demo video, GitHub repo, slides, submitted by 5/30 | Final submission |

**Buffer:** Submitting on 5/30 leaves the week of 6/1 finals untouched.

---

## 7. Team Roles

5 members, mixed RL experience. Roles assume only 1–2 members will write substantial RL code; the others contribute through experiments, evaluation, writing, and presentation.

| Member | Role | Primary Outputs |
|---|---|---|
| **Yang** (Tech Lead) | Architecture, RL pipeline, supercomputer runs, LLM prompt design, technical writing | Training pipeline, prompt templates, final-run management, Methods + Results sections |
| **Coder #2** | Environment wrappers, training scripts, evaluation harness | LLM wrapper, eval harness, hyperparameter sweeps |
| **Member #3** | Seed experiments, plotting, statistical analysis | Learning curve plots, seed analysis, results tables |
| **Member #4** | Evaluation runs, instruction template design, regex baseline parser, OOD testing | Instruction template library, regex baseline, OOD eval reports, related work section |
| **Member #5** | Demo video, slide decks, README, report integration | Demo video with voice instructions, pitch deck, README, report formatting |

**Sync cadence:** full team Mondays, RL pair (Yang + Coder #2) Wednesdays, async daily Slack updates.

---

## 8. Compute Resources

| Resource | Owner | Use |
|---|---|---|
| Google Colab Free | All members | Smoke tests, eval runs, short training |
| UCSD Supercomputer | Yang (will request access) | 9 final training runs in Week 4 |
| Google AI Studio (Gemini 2.5 Flash) | All members | Free LLM API for grounding |

**Compute budget estimate:** ~30–50 GPU-hours total on supercomputer, ~50,000 LLM API calls (well under the 1M tokens/day Gemini cap).

---

## 9. Success Criteria

**Minimum Viable Project (must achieve):**
- One trained SAC + HER policy on FetchPush with ≥ 60% success on standard test
- Working LLM grounding wrapper that translates ≥ 80% of literal instructions correctly
- One end-to-end pipeline run demonstrating language → goal → robot motion

**Target Outcome (likely to achieve):**
- All five conditions trained / evaluated across 3 seeds
- LLM grounding accuracy characterized across instruction tiers (literal, paraphrased, abstract)
- Regex baseline comparison quantifying where LLM grounding adds value
- Robustness analysis: end-to-end success vs. instruction ambiguity

**Stretch Goals (if time permits):**
- Domain randomization on cube mass / friction
- Larger instruction template library (100+ templates)
- Comparison against tightly-coupled baseline (LLM-in-loop)

---

## 10. Honest Limitations We Will Disclose

- We do not claim novelty for the pipeline paradigm itself; we claim empirical characterization on a continuous-control manipulation benchmark.
- We do not address long-horizon multi-step tasks where pipeline approaches are known to fail.
- **We do not claim sim-to-real transfer; all training and evaluation is in simulation.** The motivation references real-world deployment as the long-term direction this kind of work points toward, but our concrete claims are about simulation behavior only.
- Three random seeds is a minimum, not a definitive sample size. We will report uncertainty intervals honestly.
- Our LLM is a single frontier model (Gemini 2.5 Flash). Generalization to weaker or different LLMs is not studied.

---

## 11. Key References

- Andrychowicz et al. 2017, *Hindsight Experience Replay*
- Haarnoja et al. 2018, *Soft Actor-Critic*
- Plappert et al. 2018, *Multi-Goal RL: Challenging Robotics Environments*
- Ahn et al. 2022, *SayCan: Grounding Language in Robotic Affordances*
- Carta et al. 2023, *GLAM: Grounding LLMs in Interactive Environments*
- Shah et al. 2022, *LM-Nav* — pipeline-style instruction following
- Misra et al. 2017, *Mapping Instructions to Actions* — early pipeline work
- Plappert et al. 2018, *Gymnasium-Robotics*

---

*Last updated: April 27, 2026*
