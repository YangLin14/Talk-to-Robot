"""Evaluate the trained SAC+HER policy on FetchPush-v4.

Two modes:
  - default: roll out N episodes with the env's randomly-sampled goals
    (ablation row A in the proposal -- controller-only sanity).
  - --goal x y z: roll out N episodes with a fixed injected goal vector
    (the form the LLM/regex grounders will produce; ablation rows D/F).

Per-episode records are printed and optionally written to --output as JSON.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC


def make_env(env_id, seed):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def rollout(model, env, goal, n_episodes, max_steps, seed):
    """Roll out the policy. If goal is not None, inject it after reset."""
    records = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if goal is not None:
            env.unwrapped.goal = np.array(goal, dtype=np.float32)
            obs["desired_goal"] = env.unwrapped.goal.copy()

        steps = 0
        ep_success = False
        terminated = truncated = False
        while not (terminated or truncated) and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            ep_success = ep_success or bool(info.get("is_success", False))

        final_distance = float(
            np.linalg.norm(np.asarray(obs["achieved_goal"]) - np.asarray(obs["desired_goal"]))
        )
        records.append({
            "episode": ep,
            "success": ep_success,
            "final_distance": final_distance,
            "steps": steps,
            "achieved_goal": list(map(float, obs["achieved_goal"])),
            "desired_goal": list(map(float, obs["desired_goal"])),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="models/sac_her_FetchPush-v4_seed0_best.zip")
    parser.add_argument("--env-id", type=str, default="FetchPush-v4")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50,
                        help="cap per episode; FetchPush default horizon is 50")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="3D goal vector. Omit to use env-sampled goals.")
    parser.add_argument("--output", type=str, default=None,
                        help="optional JSON path for per-episode records")
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)
    env = make_env(args.env_id, args.seed)
    model = SAC.load(args.model_path, env=None, device="auto")

    records = rollout(
        model, env,
        goal=args.goal,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    success_rate = sum(r["success"] for r in records) / len(records)
    mean_distance = float(np.mean([r["final_distance"] for r in records]))

    summary = {
        "env_id": args.env_id,
        "model_path": args.model_path,
        "goal": args.goal,
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "success_rate": success_rate,
        "mean_final_distance": mean_distance,
        "episodes": records,
    }

    print(f"env={args.env_id}  episodes={args.n_episodes}  goal={args.goal}")
    print(f"success_rate         = {success_rate:.2%}")
    print(f"mean_final_distance  = {mean_distance:.4f} m")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_path}")

    env.close()


if __name__ == "__main__":
    main()
