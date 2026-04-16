# Project Instinct Agent Guide

## Purpose

This document is a model-facing guide for understanding and navigating the Project Instinct codebase. It summarizes the responsibilities, boundaries, and integration points of the three major subprojects:

- `InstinctLab`
- `instinct_rl`
- `instinct_onboard`

The repository should be interpreted as a full pipeline for humanoid whole-body control:

`task/environment definition -> policy training -> ONNX export -> onboard inference`

## System Partition

### InstinctLab

Role: environment-side project built on top of Isaac Lab.

Primary responsibilities:

- define tasks, scenes, robots, terrain, sensors
- define observation, action, reward, events, terminations, curriculum
- manage motion-reference-driven tasks
- provide custom environment extensions for `instinct_rl`

Important concepts:

- `InstinctRlEnv`
- `MultiRewardManager`
- `MonitorManager`
- `MotionReferenceManager`
- noisy cameras
- virtual obstacle terrain abstractions

Reading rule:

- treat it as the task/environment layer
- focus on what it adds beyond vanilla Isaac Lab
- do not spend time re-deriving standard Isaac Lab behavior unless the extension point depends on it

### instinct_rl

Role: training-side RL framework derived from Rsl-RL with major extensions.

Primary responsibilities:

- construct algorithms and policy/value networks
- run rollout/update training loops
- support structured observation inputs
- support PPO and advanced variants
- store rollout/checkpoint/training state

Important concepts:

- `obs_format`
- `obs_segment`
- `obs_pack`
- algorithm/network factory pattern
- swappable actor-critic classes

Reading rule:

- treat it as the algorithm/runtime layer
- prioritize the runner, observation interfaces, algorithm interfaces, and network factories
- understand how environment-side grouped observations are flattened and consumed

### instinct_onboard

Role: onboard deployment and ROS2 inference runtime.

Primary responsibilities:

- read training outputs from `logdir`
- load `env.yaml` and `agent.yaml`
- reconstruct observations exactly as training expected
- run ONNX models
- send commands to robot-side control interfaces

Important concepts:

- onboard agents
- ROS nodes
- action scaling handled on the ROS side, not by the network output itself
- deployment reuses training configuration rather than redefining interfaces manually

Reading rule:

- treat it as the deployment/runtime layer
- always relate onboard observation logic back to training configs
- check how ONNX submodels correspond to training-time architecture decomposition

## End-to-End Data Flow

1. A task is registered in `InstinctLab`.
2. `scripts/instinct_rl/train.py` resolves the task config and RL config.
3. A Gym/IsaacLab environment is created with `instinctlab.envs:InstinctRlEnv`.
4. `InstinctRlVecEnvWrapper` adapts grouped observations and multi-reward outputs for `instinct_rl`.
5. `instinct_rl.runners.OnPolicyRunner` builds the configured algorithm and network.
6. Training runs and stores outputs under a timestamped `logdir`.
7. Policy export produces ONNX files.
8. `instinct_onboard` reads `logdir`, rebuilds observation logic, loads ONNX, and performs inference on the robot.

## What Is Project-Specific

The most project-specific parts are not generic Isaac Lab or generic PPO mechanics. The highest-value areas are:

- motion-reference-driven humanoid tasks
- shadowing/tracking/imitation task design
- multi-reward support
- structured observation handling across environment and RL layers
- sim-to-real deployment through shared configs and ONNX export

## Key Design Contracts

### Contract 1: Task Registration

Tasks should register with:

- `entry_point="instinctlab.envs:InstinctRlEnv"`
- `env_cfg_entry_point`
- `instinct_rl_cfg_entry_point`

This is a stable cross-layer contract between task definition and training runtime.

### Contract 2: Structured Observations

Environment-side observations are grouped and named. The RL side expects:

- explicit observation group structure
- stable flattening behavior
- consistent slicing based on `obs_format`

Any new task or module must preserve this alignment.

### Contract 3: Motion Reference Integration

Motion reference is not an isolated dataset utility. It must align with:

- commands
- observations
- rewards
- resets
- deployment assumptions where applicable

### Contract 4: Deployment Reproducibility

Onboard execution assumes the training `logdir` is the source of truth. If training config semantics change, onboard logic may also need to change.

## Recommended Reading Path

### First pass: one task, one full path

Use:

- `Instinct-Perceptive-Shadowing-G1-v0`

Read in order:

1. task registration
2. `scripts/instinct_rl/train.py`
3. environment config
4. `InstinctRlEnv`
5. `InstinctRlVecEnvWrapper`
6. policy config
7. `OnPolicyRunner`
8. algorithm implementation
9. onboard agent counterpart

### Second pass: project-specific abstractions

Focus on:

- `motion_reference`
- `envs/mdp/commands`
- `envs/mdp/observations`
- `envs/mdp/rewards`
- multi-reward and monitor managers

### Third pass: training variants

Compare:

- PPO
- WASABI / AMP
- TPPO
- VAE distillation
- actor critic variants
- encoder-based models
- MoE models

## Questions a Model Should Be Able to Answer

After reading the codebase, the model should be able to explain:

1. what each of the three subprojects is responsible for
2. how a task is registered and launched
3. how structured observations flow from IsaacLab managers into `instinct_rl`
4. how multi-reward support changes the training interface
5. how motion reference drives commands, observations, and rewards
6. how training outputs are reused by onboard inference

## Common Misreadings To Avoid

- Do not treat `InstinctLab` as only a thin task collection.
- Do not assume `instinct_rl` only consumes flat vectors without structure.
- Do not treat `instinct_onboard` as a separate ad hoc deployment codebase; it is coupled to training outputs.
- Do not start from `thirdparty/IsaacLab` unless an extension point truly requires it.
- Do not read tasks horizontally before understanding one representative task end to end.
