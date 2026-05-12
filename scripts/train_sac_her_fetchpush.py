import argparse
import os
import random
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(env_id, seed, rank=0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="logs/sac_her_fetchpush")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    gym.register_envs(gymnasium_robotics)
    env_id = "FetchPush-v4"

    log_dir = Path(args.log_dir) / f"seed_{args.seed}"
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(env_id, args.seed, 0)])
    eval_env = DummyVecEnv([make_env(env_id, args.seed + 10_000, 0)])

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="sac_her_fetchpush",
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

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer,
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

    print("=" * 80)
    print(f"Training env: {env_id}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Seed: {args.seed}")
    print(f"Logs: {log_dir}")
    print("=" * 80)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
    )

    final_path = model_dir / f"sac_her_{env_id}_seed{args.seed}.zip"
    model.save(final_path)

    print("=" * 80)
    print(f"Saved final model to: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()