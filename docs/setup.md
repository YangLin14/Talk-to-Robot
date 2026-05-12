# Setup Guide: Spatial Grounding RL Environment

This document records the environment setup and smoke tests used for the spatial-grounding robotic manipulation project.

The goal of this setup is to validate the simulation and RL infrastructure before running full-scale SAC + HER training or connecting the LLM grounding wrapper.

## 1. Environment Choice

We use a dedicated Conda environment:

```bash
spatial-rl-311
```

Important: do not install project packages into `(base)`.

Or you can use python .venv if you would like.

## 2. Project Directory

Move into the project directory first:

```bash
cd ~/Talk-to-Robot
```

## 3. Create the Conda Environment

```bash
conda create -n spatial-rl-311 python=3.11 -y
conda activate spatial-rl-311
```

Verify that the correct environment is active:

```bash
which python
python --version
which pip
conda info --envs
```

Expected result:

```text
Python 3.11.x
```

The active environment should be marked as:

```text
spatial-rl-311 *
```

## 4. Install Dependencies

Install the core simulation, RL, LLM, and utility packages:

```bash
python -m pip install --upgrade "pip<26" "setuptools<82" wheel

pip install mujoco
pip install gymnasium-robotics
pip install "stable-baselines3[extra]"
pip install google-genai pydantic python-dotenv
pip install numpy pandas matplotlib tqdm tensorboard ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name spatial-rl-311 --display-name "Python (spatial-rl-311)"
```

Save the exact package versions:

```bash
pip freeze > requirements.txt
```

This produces the file:

```text
requirements.txt
```

## 5. Important Version Note: FetchPush-v4

The original proposal referenced `FetchPush-v3`, but the installed Gymnasium-Robotics version does not register `FetchPush-v3` as the supported environment. Instead, the available FetchPush environments are:

```text
FetchPush-v1
FetchPush-v4
FetchPushDense-v1
FetchPushDense-v4
```

Therefore, the implementation uses:

```python
ENV_ID = "FetchPush-v4"
```

Do not write that we "installed FetchPush-v4." FetchPush-v4 is not a separate package. We installed `gymnasium-robotics`, and `FetchPush-v4` is one of the registered environment IDs provided by that package.

## 6. Smoke Test 1: FetchPush-v4 Environment Test

This test verifies that MuJoCo, Gymnasium-Robotics, and FetchPush-v4 are installed correctly.

Run:

```bash
python - <<'PY'
import gymnasium as gym
import gymnasium_robotics
import mujoco

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPush-v4"

print("MuJoCo:", mujoco.__version__)
print("Available FetchPush envs:")
print([spec.id for spec in gym.registry.values() if "FetchPush" in spec.id])

env = gym.make(ENV_ID)
obs, info = env.reset(seed=0)

print("Using:", ENV_ID)
print("Observation keys:", obs.keys())
print("observation shape:", obs["observation"].shape)
print("achieved_goal shape:", obs["achieved_goal"].shape)
print("desired_goal shape:", obs["desired_goal"].shape)
print("Action space:", env.action_space)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

print("passed")
env.close()
PY
```

Example output:

```text
AdroitHandRelocateDense-v1, AdroitHandHammerDense-v1, AdroitHandDoorDense-v1 environment's reward functions were updated in v1.2.1 without an environment version update. Therefore, use gymnasium-robotics==1.2.0 for v1 reproducibility or use v2 in gymnasium-robotics>=1.4.3. See https://github.com/Farama-Foundation/Gymnasium-Robotics/pull/220 for more details
MuJoCo: 3.8.1
Available FetchPush envs:
['FetchPush-v1', 'FetchPush-v4', 'FetchPushDense-v1', 'FetchPushDense-v4']
Using: FetchPush-v4
Observation keys: dict_keys(['observation', 'achieved_goal', 'desired_goal'])
observation shape: (25,)
achieved_goal shape: (3,)
desired_goal shape: (3,)
Action space: Box(-1.0, 1.0, (4,), float32)
passed
```

Interpretation:

- `FetchPush-v4` is registered and usable.
- The environment can be created.
- The environment can be reset with a fixed seed.
- The environment can be stepped with random actions.
- The observation is goal-conditioned and contains:
  - `observation`
  - `achieved_goal`
  - `desired_goal`
- The action space is a 4-dimensional continuous `Box`.

The Adroit warning can be ignored for this project because we are not using Adroit hand environments.

## 7. Smoke Test 2: SAC + HER Training Test

This test verifies that Stable-Baselines3, SAC, HER replay buffer, CUDA, logging, and checkpoint saving work with FetchPush-v4.

Run:

```bash
mkdir -p checkpoints runs

python - <<'PY'
import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC, HerReplayBuffer

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPush-v4"

env = gym.make(ENV_ID)

model = SAC(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    learning_starts=1000,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    tensorboard_log="runs/fetchpush_sac_her_smoke",
)

model.learn(total_timesteps=2000)
model.save("checkpoints/sac_her_fetchpush_v4_smoke")

print("SAC + HER smoke training finished.")
env.close()
PY
```

Example output:

```text
Using cuda device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
Logging to runs/fetchpush_sac_her_smoke/SAC_1
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50       |
|    ep_rew_mean     | -37.5    |
|    success_rate    | 0.25     |
| time/              |          |
|    episodes        | 4        |
|    fps             | 488      |
|    time_elapsed    | 0        |
|    total_timesteps | 200      |
---------------------------------
...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50       |
|    ep_rew_mean     | -45.8    |
|    success_rate    | 0.0833   |
| time/              |          |
|    episodes        | 24       |
|    fps             | 218      |
|    time_elapsed    | 5        |
|    total_timesteps | 1200     |
| train/             |          |
|    actor_loss      | -6.31    |
|    critic_loss     | 0.223    |
|    ent_coef        | 0.942    |
|    ent_coef_loss   | -0.401   |
|    learning_rate   | 0.0003   |
|    n_updates       | 199      |
---------------------------------
...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50       |
|    ep_rew_mean     | -47.2    |
|    success_rate    | 0.05     |
| time/              |          |
|    episodes        | 40       |
|    fps             | 112      |
|    time_elapsed    | 17       |
|    total_timesteps | 2000     |
| train/             |          |
|    actor_loss      | -11.8    |
|    critic_loss     | 0.186    |
|    ent_coef        | 0.741    |
|    ent_coef_loss   | -2.01    |
|    learning_rate   | 0.0003   |
|    n_updates       | 999      |
---------------------------------
SAC + HER smoke training finished.
```

Interpretation:

- Stable-Baselines3 successfully detected CUDA.
- The FetchPush-v4 environment was wrapped by SB3 using `Monitor` and `DummyVecEnv`.
- TensorBoard logs were created under:

```text
runs/fetchpush_sac_her_smoke/
```

- SAC began gradient updates after the replay buffer warm-up period.
- Training metrics appeared, including:
  - `actor_loss`
  - `critic_loss`
  - `ent_coef`
  - `ent_coef_loss`
  - `n_updates`
- A checkpoint was saved under:

```text
checkpoints/sac_her_fetchpush_v4_smoke.zip
```

The low success rate during the 2,000-step run is expected. This run is only a smoke test, not a performance result. FetchPush requires much longer training for meaningful controller performance.

## 8. What This Setup Confirms

After both smoke tests pass, the project has validated:

```text
Conda environment: spatial-rl-311
Python: 3.11
MuJoCo: working
Gymnasium-Robotics: working
FetchPush-v4: working
Stable-Baselines3: working
SAC: working
HER replay buffer: working
MultiInputPolicy: working
CUDA: detected
TensorBoard logging: working
Checkpoint saving: working
```

This means the simulation and RL training stack is ready for longer SAC + HER training runs.