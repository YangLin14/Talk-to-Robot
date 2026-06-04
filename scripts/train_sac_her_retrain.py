"""
train_sac_her_retrain.py

SAC+HER retraining script with three targeted improvements:
  Phase 1: Goal Distribution Augmentation (T2 egdes and T4 corners)
  Phase 2: Shaped reward for T3 offset correction
  Phase 3: Targeted Replay on failure cases (seed policy w/ documented failure cases from original runs)

The --failure-json argument accepts one of the eval result files (t0123_llm_*.json). 
The script will extract the failed episodes and use them to seed the replay buffer before training begins.
"""

import argparse
import json
import os
import random
from pathlib import Path
import sys

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer as _BaseHER
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from talk_to_robot.workspace import WORKSPACE, TABLE_Z, in_workspace


WS_X = (WORKSPACE["x_min"], WORKSPACE["x_max"])
WS_Y = (WORKSPACE["y_min"], WORKSPACE["y_max"])
WS_Z = TABLE_Z

# Corner goals used by the grounder for T4
WORKSPACE_CORNERS = np.array([
    [WORKSPACE["x_min"], WORKSPACE["y_min"], WS_Z],
    [WORKSPACE["x_max"], WORKSPACE["y_min"], WS_Z],
    [WORKSPACE["x_min"], WORKSPACE["y_max"], WS_Z],
    [WORKSPACE["x_max"], WORKSPACE["y_max"], WS_Z],
    [(WORKSPACE["x_min"] + WORKSPACE["x_max"]) / 2,
     (WORKSPACE["y_min"] + WORKSPACE["y_max"]) / 2, WS_Z],
], dtype=np.float32)

# Fraction of HER virtual goals to replace with biased samples (20/10 split)
BOUNDARY_GOAL_FRACTION = 0.20
CORNER_GOAL_FRACTION   = 0.10

# Boundary margin
BOUNDARY_MARGIN = 0.05

#Phase 1
class BiasedGoalHerReplayBuffer(_BaseHER):
    """
    Extends HerReplayBuffer to inject edge/corner goals during virtual
    experience generation, alongside the standard future-strategy goals.

    Sampling split per HER virtual transition:
        70%  standard                 (unchanged baseline behaviour)
        20%  near-boundary goals      (T2: policy learns workspace edges)
        10%  workspace corner goals   (T4: policy reliable at grounder corners)
    """
    #Override default sample goals
    def _sample_goals(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
    ) -> np.ndarray:
        n = len(batch_indices)
        n_corner   = max(1, int(n * CORNER_GOAL_FRACTION))
        n_boundary = max(1, int(n * BOUNDARY_GOAL_FRACTION))
        n_standard = n - n_corner - n_boundary

        standard_goals = self._sample_future_goals(
            batch_indices[:n_standard], env_indices[:n_standard]
        )

        boundary_goals = self._sample_boundary_goals(n_boundary)

        corner_idx    = np.random.randint(0, len(WORKSPACE_CORNERS), size=n_corner)
        corner_goals  = WORKSPACE_CORNERS[corner_idx]

        return np.concatenate([standard_goals, boundary_goals, corner_goals], axis=0)


    def _sample_future_goals(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
    ) -> np.ndarray:
        """Exact replica of the parent's FUTURE strategy logic."""
        batch_ep_start  = self.ep_start[batch_indices, env_indices]
        batch_ep_length = self.ep_length[batch_indices, env_indices]
        current_indices_in_episode = (batch_indices - batch_ep_start) % self.buffer_size
        transition_indices_in_episode = np.random.randint(
            current_indices_in_episode, batch_ep_length
        )
        transition_indices = (
            transition_indices_in_episode + batch_ep_start
        ) % self.buffer_size
        return self.next_observations["achieved_goal"][transition_indices, env_indices]

    def _sample_boundary_goals(self, n: int) -> np.ndarray:
        """
        Sample goals uniformly within BOUNDARY_MARGIN of the workspace edge.
        Each goal is either near the x-boundary or y-boundary with equal prob.
        """
        goals = np.zeros((n, 3), dtype=np.float32)
        goals[:, 2] = WS_Z

        for i in range(n):
            if np.random.random() < 0.5:
                # Near x boundary
                side = np.random.choice([-1, 1])
                if side == 1:
                    goals[i, 0] = np.random.uniform(WS_X[1] - BOUNDARY_MARGIN, WS_X[1])
                else:
                    goals[i, 0] = np.random.uniform(WS_X[0], WS_X[0] + BOUNDARY_MARGIN)
                goals[i, 1] = np.random.uniform(WS_Y[0], WS_Y[1])
            else:
                # Near y boundary
                goals[i, 0] = np.random.uniform(WS_X[0], WS_X[1])
                side = np.random.choice([-1, 1])
                if side == 1:
                    goals[i, 1] = np.random.uniform(WS_Y[1] - BOUNDARY_MARGIN, WS_Y[1])
                else:
                    goals[i, 1] = np.random.uniform(WS_Y[0], WS_Y[0] + BOUNDARY_MARGIN)

        return goals

#Phase 2
class ShapedRewardWrapper(gym.Wrapper):
    """
    Adds a small distance-proportional penalty on top of FetchPush's sparse
    reward to help the policy care about proximity to goal.

    r_total = r_sparse + alpha * shaping_bonus
    shaping_bonus = clip(1 - dist / dist_threshold, 0, 1)

    alpha is 0.1 so it doesn't dominate the sparse signal.
    """

    def __init__(self, env: gym.Env, alpha: float = 0.1, dist_threshold: float = 0.10):
        super().__init__(env)
        self.alpha          = alpha
        self.dist_threshold = dist_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        achieved = obs["achieved_goal"]
        desired  = obs["desired_goal"]
        dist     = float(np.linalg.norm(achieved - desired))

        shaping_bonus = np.clip(1.0 - dist / self.dist_threshold, 0.0, 1.0)
        shaped_reward = float(reward) + self.alpha * shaping_bonus

        return obs, shaped_reward, terminated, truncated, info



#Phase 3
def load_failure_transitions(json_paths: list[str], tiers: list[str] = None) -> list[dict]:
    """
    Parse eval result JSON files and extract failed episode transitions.

    Args:
        json_paths: Paths to eval result JSON files (t0123_llm_*.json etc.)
        tiers:      If set, only extract failures from these tiers (e.g. ['T2','T3'])

    Returns:
        List of failure records with predicted_goal, ground_truth_goal,
        and policy_episodes data.
    """
    failures = []
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", [])
        for r in results:
            if tiers and r.get("tier") not in tiers:
                continue
            # T2/T3: grounder failed OR policy failed at least one episode
            grounder_failed = not r.get("grounder_success", True)
            policy_sr       = r.get("policy_success_rate", 1.0)
            if grounder_failed or policy_sr < 1.0:
                failures.append(r)
    print(f"[Phase 3] Loaded {len(failures)} failure records from {len(json_paths)} file(s)")
    return failures


def seed_replay_buffer(
    model: SAC,
    env: DummyVecEnv,
    failures: list[dict],
    n_repeats: int = 5,
    seed: int = 0,
) -> None:
    """
    Run short rollouts from failure-adjacent starting states and store the
    transitions in the replay buffer before training begins.

    Because we can't set arbitrary block positions in FetchPush without
    modifying the env, we approximate this by:
      1. Resetting the env normally
      2. Overriding the desired_goal with the failure's ground_truth_goal
      3. Rolling out a random policy for a few steps to populate the buffer
         with transitions around hard goal regions

    This biases early replay toward the goal locations that were hard in eval.
    """
    rng = np.random.default_rng(seed)
    n_seeded = 0

    for record in failures:
        goal = np.array(record["ground_truth_goal"], dtype=np.float32)
        if not in_workspace(goal):
            continue

        for _ in range(n_repeats):
            obs = env.reset()

            # Override the desired goal in the observation dict
            if isinstance(obs, dict) and "desired_goal" in obs:
                obs["desired_goal"][:] = goal

            done = False
            steps = 0
            while not done and steps < 50:
                action = np.array([env.action_space.sample()])
                next_obs, reward, done_arr, info = env.step(action)

                # Override desired goal in next obs too
                if isinstance(next_obs, dict) and "desired_goal" in next_obs:
                    next_obs["desired_goal"][:] = goal

                model.replay_buffer.add(
                    obs, next_obs, action,
                    reward, done_arr,
                    [info] if not isinstance(info, list) else info,
                )
                obs   = next_obs
                done  = done_arr[0]
                steps += 1
            n_seeded += 1

    print(f"[Phase 3] Seeded replay buffer with ~{n_seeded} failure rollouts "
          f"({n_seeded * 50} transitions approx)")






def make_env(env_id: str, seed: int, rank: int = 0, shaped_reward: bool = True,
             alpha: float = 0.1, dist_threshold: float = 0.10):
    def _init():
        env = gym.make(env_id)
        if shaped_reward:
            env = ShapedRewardWrapper(env, alpha=alpha, dist_threshold=dist_threshold)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="SAC+HER retrain with T2/T3 fixes")
    parser.add_argument("--total-timesteps",  type=int,   default=1_000_000)
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--log-dir",          type=str,   default="logs/sac_her_retrain")
    parser.add_argument("--model-dir",        type=str,   default="models")
    parser.add_argument("--eval-freq",        type=int,   default=25_000)
    parser.add_argument("--n-eval-episodes",  type=int,   default=20)

    parser.add_argument("--reward-alpha",         type=float, default=0.1,
                        help="Weight of shaping bonus (0 = disable shaped reward)")
    parser.add_argument("--reward-dist-threshold",type=float, default=0.10,
                        help="Distance at which shaping bonus goes to zero (metres)")

    parser.add_argument("--failure-json", type=str, nargs="+", default=[],
                        help="Paths to eval result JSON files for failure seeding")
    parser.add_argument("--failure-tiers", type=str, nargs="+", default=["T2", "T3"],
                        help="Which tiers to pull failures from (default: T2 T3)")
    parser.add_argument("--failure-repeats", type=int, default=5,
                        help="How many rollouts to seed per failure record")
    parser.add_argument("--disable-goal-biasing", action="store_true",
                    help="Use standard HerReplayBuffer instead of biased version")

    args = parser.parse_args()

    set_seed(args.seed)
    gym.register_envs(gymnasium_robotics)
    env_id = "FetchPush-v4"

    log_dir   = Path(args.log_dir) / f"seed_{args.seed}"
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    shaped = args.reward_alpha > 0
    train_env = DummyVecEnv([make_env(
        env_id, args.seed, 0,
        shaped_reward=shaped,
        alpha=args.reward_alpha,
        dist_threshold=args.reward_dist_threshold,
    )])
    # Eval env uses clean sparse reward for apples-to-apples comparison
    eval_env = DummyVecEnv([make_env(
        env_id, args.seed + 10_000, 0,
        shaped_reward=False,
    )])

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="sac_her_retrain",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    print("=" * 80)
    print(f"Training env   : {env_id}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Seed           : {args.seed}")
    print(f"Shaped reward  : {'yes (alpha={}, threshold={}m)'.format(args.reward_alpha, args.reward_dist_threshold) if shaped else 'no'}")
    print(f"Goal biasing   : boundary={BOUNDARY_GOAL_FRACTION:.0%}  corners={CORNER_GOAL_FRACTION:.0%}  standard={1-BOUNDARY_GOAL_FRACTION-CORNER_GOAL_FRACTION:.0%}")
    print(f"Failure seeding: {len(args.failure_json)} file(s), tiers={args.failure_tiers}")
    print(f"Logs           : {log_dir}")
    print("=" * 80)

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer if args.disable_goal_biasing else BiasedGoalHerReplayBuffer,  # Phase 1
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(log_dir / "tb"),
        seed=args.seed,
    )

    if args.failure_json:
        failures = load_failure_transitions(args.failure_json, args.failure_tiers)
        if failures:
            seed_replay_buffer(
                model, train_env, failures,
                n_repeats=args.failure_repeats,
                seed=args.seed,
            )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
    )

    final_path = model_dir / f"sac_her_{env_id}_retrain_seed{args.seed}.zip"
    model.save(final_path)

    print("=" * 80)
    print(f"Saved final model: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()