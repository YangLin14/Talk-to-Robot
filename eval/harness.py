"""Eval harness: pipes instruction library -> grounder -> fetchpush policy.

Overall workflow:
    1. Parses Arguments (--tier, --variant, --output, etc.), Load instructions, filter entries
    2. For each desired instruction:
        A. Build context if necessary for the entry, call ground(), get back goal + failure_mode
        B. Check if the ground was successful, feed goal -> eval_policy_fetchpush
        C. Check for policy success and end-to-end success; Record output for the given instruction tier
            # Per-entry tolerance: entry.get("tolerance_m") or DEFAULT_TOLERANCE_M (0.05m).
            # T0/T1/T2 entries omit tolerance_m and fall back to 0.05m.
            # T3 entries set tolerance_m=0.08m (proximity ambiguity).
            # T4 entries set tolerance_m from annotator spread (mean pairwise + 0.03m).
            Also record if it failed
    3. Write results to json file
Usage:
  # Run all tiers, zero-shot, default model
  python eval/harness.py

  # Run only T1 with specified variant ("zero-shot", "few-shot", "cot")
  python eval/harness.py --tier T1 --variant ARGUMENT

  # Run T0-T2, write results to file
  python eval/harness.py --tier T0 T1 T2 --output results/run_001.json

  # Use regex grounder instead of LLM
  python eval/harness.py --grounder regex

  # Grounding-only regex baseline, no SAC model rollout
  python eval/harness.py --grounder regex --tier T0 T1 T2 --skip-policy

Some Design Decisions/Notes:
  - T2: Before grounding, the harness resets the env to get the live cube position,
    recomputes ground truth by preserving the original offset, and passes the live
    cube_pos as context so grounding is evaluated against the actual sim state.
  - T4: entries with annotation_status != 'confirmed' are not run
  - Scoring has three layers: grounder success, policy success, and end-to-end success (strict, grounding failure = e2e failure) tracked separately.
  - Default tolerance is 0.05m (T0/T1/T2). T3 entries override to 0.08m; T4 to annotator spread.
"""

import argparse
import json
import sys
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

#path setup
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from grounding.grounder import ground as llm_ground          # noqa: E402
from baselines.regex_grounder import ground as regex_ground  # noqa: E402


# Constants:
DEFAULT_INSTRUCTIONS_PATH = _PROJECT_ROOT / "instructions" / "instructions.json"
DEFAULT_MODEL_PATH        = _PROJECT_ROOT / "models" / "sac_her_FetchPush-v4_seed0_best.zip"
DEFAULT_ENV_ID            = "FetchPush-v4"
DEFAULT_N_EPISODES        = 3      #per instruction
DEFAULT_MAX_STEPS         = 50     # FetchPush horizon
DEFAULT_SEED              = 0
DEFAULT_TOLERANCE_M       = 0.05   # T0, T1 scoring tolerance


# Scoring helper functions

def _goal_distance(goal_a, goal_b):
    """Euclidean distance between two [x, y, z] goals."""
    return float(np.linalg.norm(np.array(goal_a) - np.array(goal_b)))


def score_grounder(entry, ground_result):
    """
    Check whether the grounder output (LLM) matches ground truth (from instructions.json parsing).

    Returns:
        grounder_success (bool)
        grounder_distance (float | None)  -- distance from ground truth, None if no ground truth
        skip_reason (str | None)          -- if entry should be skipped (e.g. "T4 not confirmed")
    """
    tier = entry["tier"]

    # T4: skip if not confirmed
    if tier == "T4":
        if entry.get("annotation_status") != "confirmed":
            return False, None, "pending_annotation"
        if entry.get("ground_truth_goal") is None:
            return False, None, "no_ground_truth"

    # Grounder failed to produce a goal
    if not ground_result.get("success") or ground_result.get("goal") is None:
        return False, None, None

    predicted = ground_result["goal"]
    ground_truth = entry["ground_truth_goal"]

    if ground_truth is None:
        return True, None, None  # grounder succeeded, no ground_truth to score against, only applies to T4 instructions

    dist = _goal_distance(predicted, ground_truth)

    # T3 and T4 set tolerance_m per entry; T0/T1/T2 fall back to DEFAULT_TOLERANCE_M (0.05m).
    tolerance = entry.get("tolerance_m") or DEFAULT_TOLERANCE_M
    if dist <= tolerance:
        grounder_success = True
    else:
        grounder_success = False

    return grounder_success, dist, None


# Evaulation loop (steps 1 and 2)

def run_eval(
    instructions_path,
    model_path,
    env_id,
    tiers,
    variant,
    grounder_name,
    n_episodes,
    max_steps,
    seed,
    device,
    skip_policy,
):
    # Load instructions
    with open(instructions_path) as f:
        all_entries = json.load(f)["instructions"]

    # Filter by tier
    if tiers:
        all_entries = [e for e in all_entries if e["tier"] in tiers]

    if not all_entries:
        print(f"No instructions found for tiers: {tiers}")
        return []

    print(f"Loaded {len(all_entries)} instructions "
          f"(tiers={tiers or 'all'}, grounder={grounder_name}, variant={variant})")

    env = None
    model = None
    needs_env = (not skip_policy) or any(e.get("tier") == "T2" for e in all_entries)
    make_env = rollout = None

    if needs_env:
        try:
            import gymnasium as gym
            import gymnasium_robotics
            from scripts.eval_policy_fetchpush import make_env, rollout

            gym.register_envs(gymnasium_robotics)
            env = make_env(env_id, seed)
        except ModuleNotFoundError as e:
            if not skip_policy:
                raise
            print(
                "Gymnasium is unavailable; running --skip-policy without live "
                f"T2 cube context ({e})."
            )

    if skip_policy:
        print("Policy rollout skipped; running grounding/scoring only.")
    else:
        from stable_baselines3 import SAC

        model = SAC.load(model_path, env=env, device=device)
        print(f"Model loaded: {model_path.name} (device={device})")

    results = []

    for i, entry in enumerate(all_entries):
        entry_id   = entry["id"]
        tier       = entry["tier"]
        instruction = entry["instruction"]
        context    = entry.get("context")
        entry_seed = seed + i * max(n_episodes, 1)

        # T2: replace fixed cube_pos with live position from env reset
        if env is not None and tier == "T2" and context and "cube_pos" in context:
            obs_pre, _ = env.reset(seed=entry_seed)
            live_cube_pos = list(map(float, obs_pre["achieved_goal"]))
            old_cube = np.array(context["cube_pos"])
            old_gt = np.array(entry["ground_truth_goal"])
            offset = old_gt - old_cube
            context = dict(context)
            context["cube_pos"] = live_cube_pos
            entry = dict(entry)
            entry["ground_truth_goal"] = list(map(float,
                np.array(live_cube_pos) + offset))

        print(f"\n[{i+1}/{len(all_entries)}] {entry_id}: \"{instruction}\"")

        #A. Ground instruction entry
        if grounder_name == "regex":
            ground_result = regex_ground(instruction, tier, context)
        else:
             #LLM Grounding (rate limiting + retry handled inside GeminiClient)
            ground_result = llm_ground(
                instruction, tier,
                context=context,
                prompt_variant=variant,
            )

        #B + C, Score
        grounder_success, grounder_dist, skip_reason = score_grounder(
            entry, ground_result
        )

        if skip_reason:
            print(f"  SKIPPED: {skip_reason}")
            results.append({
                "id": entry_id,
                "tier": tier,
                "instruction": instruction,
                "skipped": True,
                "skip_reason": skip_reason,
            })
            continue

        goal = ground_result.get("goal")
        print(f"  grounder={'OK' if grounder_success else 'FAIL'} "
              f"| goal={goal} "
              f"| failure_mode={ground_result.get('failure_mode')}"
              f"| dist={f'{grounder_dist:.4f}m' if grounder_dist is not None else 'N/A'}")

        policy_records = []
        if skip_policy:
            policy_success_rate = None
            mean_distance = None
            e2e_success_rate = None
            print("  policy: skipped (--skip-policy)")
        elif goal is not None:
            policy_records = rollout(
                model, env,
                goal=goal,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed=entry_seed,
            )
            policy_success_rate = sum(
                r["success"] for r in policy_records
            ) / len(policy_records)
            mean_distance = float(np.mean(
                [r["final_distance"] for r in policy_records]
            ))
            print(f"  policy success={policy_success_rate:.0%} "
                  f"| mean_dist={mean_distance:.4f}m")
            # End-to-end scoring
            if goal is not None and entry.get("ground_truth_goal") is not None:
                tolerance = entry.get("tolerance_m") or DEFAULT_TOLERANCE_M
                e2e_successes = [
                    _goal_distance(r["achieved_goal"], entry["ground_truth_goal"]) <= tolerance
                    for r in policy_records
                ]
                e2e_success_rate = sum(e2e_successes) / len(e2e_successes)
            else:
                e2e_success_rate = None

        else:
            policy_success_rate = None
            mean_distance = None
            e2e_success_rate = 0.0
            print("  policy: skipped (no valid goal)")

        #C. Record
        results.append({
            "id": entry_id,
            "tier": tier,
            "instruction": instruction,
            "skipped": False,
            "skip_reason": None,
            # grounder
            "grounder": ground_result.get("grounder"),
            "prompt_variant": ground_result.get("prompt_variant"),
            "grounder_success": grounder_success,
            "grounder_distance_m": grounder_dist,
            "grounder_failure_mode": ground_result.get("failure_mode"),
            "predicted_goal": goal,
            "ground_truth_goal": entry.get("ground_truth_goal"),
            "reasoning": ground_result.get("reasoning"),
            "usage": ground_result.get("usage"),
            # policy
            "policy_success_rate": policy_success_rate,
            "policy_mean_distance_m": mean_distance,
            "policy_episodes": policy_records,
            #end-to-end
            "e2e_success_rate": e2e_success_rate
        })

    if env is not None:
        env.close()
    return results


#Aggregation
def aggregate(results):
    """Compute per-tier summary stats."""
    summary = {}
    tiers = sorted(set(r["tier"] for r in results if not r.get("skipped")))

    for tier in tiers:
        tier_results = [r for r in results if r["tier"] == tier and not r.get("skipped")]
        if not tier_results:
            continue

        n = len(tier_results)
        grounder_successes = [r for r in tier_results if r["grounder_success"]]
        policy_rates = [
            r["policy_success_rate"]
            for r in tier_results
            if r["policy_success_rate"] is not None
        ]
        e2e_rates = [
            r["e2e_success_rate"]
            for r in tier_results
            if r["e2e_success_rate"] is not None
        ]
        failure_modes = [
            r["grounder_failure_mode"]
            for r in tier_results
            if r["grounder_failure_mode"]
        ]

        summary[tier] = {
            "n": n,
            "grounder_success_rate": len(grounder_successes) / n,
            "policy_success_rate": float(np.mean(policy_rates)) if policy_rates else None,
            "e2e_success_rate": float(np.mean(e2e_rates)) if e2e_rates else None,
            "failure_mode_counts": {
                mode: failure_modes.count(mode)
                for mode in set(failure_modes)
            },
            "mean_grounder_distance_m": float(np.mean([
                r["grounder_distance_m"]
                for r in tier_results
                if r["grounder_distance_m"] is not None
            ])) if any(r["grounder_distance_m"] is not None for r in tier_results) else None,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Eval harness: ground instructions and run RL policy."
    )
    parser.add_argument(
        "--instructions", type=Path,
        default=DEFAULT_INSTRUCTIONS_PATH,
        help="Path to instructions.json instruction library"
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to SAC+HER checkpoint"
    )
    parser.add_argument(
        "--env-id", type=str, default=DEFAULT_ENV_ID,
    )
    parser.add_argument(
        "--tier", nargs="+",
        choices=["T0", "T1", "T2", "T3", "T4"],
        default=None,
        help="Tiers to evaluate. Omit to run all."
    )
    parser.add_argument(
        "--variant", default="zero-shot",
        choices=["zero-shot", "few-shot", "cot"],
        help="Prompt variant for LLM grounder"
    )
    parser.add_argument(
        "--grounder", default="llm",
        choices=["llm", "regex"],
        help="Which grounder to use"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=DEFAULT_N_EPISODES,
        help="Policy rollouts per instruction"
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for SAC policy rollout. Regex grounding itself does not need a GPU."
    )
    parser.add_argument(
        "--skip-policy", action="store_true",
        help="Only run the grounder and ground-truth scoring; do not load or roll out the SAC policy."
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write full results JSON here"
    )
    args = parser.parse_args()

    results = run_eval(
        instructions_path=args.instructions,
        model_path=args.model_path,
        env_id=args.env_id,
        tiers=args.tier,
        variant=args.variant,
        grounder_name=args.grounder,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
        skip_policy=args.skip_policy,
    )

    summary = aggregate(results)

    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for tier, stats in summary.items():
        print(f"\n{tier} (n={stats['n']})")
        print(f"  Grounder success : {stats['grounder_success_rate']:.0%}")
        if stats['policy_success_rate'] is not None:
            print(f"  Policy success   : {stats['policy_success_rate']:.0%}")
        if stats['e2e_success_rate'] is not None:
            print(f"  E2E success      : {stats['e2e_success_rate']:.0%}")
        if stats['mean_grounder_distance_m'] is not None:
            print(f"  Mean GT distance : {stats['mean_grounder_distance_m']:.4f}m")
        if stats['failure_mode_counts']:
            print(f"  Failure modes    : {stats['failure_mode_counts']}")

    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "grounder": args.grounder,
            "prompt_variant": args.variant,
            "tiers": args.tier or ["T0","T1","T2","T3","T4"],
            "n_episodes": args.n_episodes,
            "device": args.device,
            "skip_policy": args.skip_policy,
            "summary": summary,
            "results": results,
        }
        args.output.write_text(json.dumps(output, indent=2))
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
