# InstinctLab 概览

## 设计原则

InstinctLab 是 Project Instinct 的环境侧部分，构建在 **Isaac Lab** 之上，设计目标包括：

- **隔离性**：遵循 Isaac Lab 的 “own project” 工作流，使开发工作独立于 Isaac Lab 核心仓库，同时仍然依赖其能力。
- **灵活性**：支持以 Omniverse Extension 的形式运行。
- **统一生态**：与 `instinct_rl` 和 `instinct_onboard` 协同集成。
- **模块化环境**：扩展 Isaac Lab 的 `ManagerBasedRLEnv`，支持多 critic 强化学习、完整监控等高级能力。

## 核心组件

### 1. InstinctRlEnv（`instinctlab.envs.InstinctRlEnv`）

**说明**：
这是项目中的核心环境类，继承自 `ManagerBasedRLEnv`。

**关键扩展**：

- **MultiRewardManager**
  替换默认 reward manager，用于支持多 reward group，这对多 critic RL 非常有用。
- **MonitorManager**
  专门用于记录仿真状态和指标，并写入 TensorBoard。

### 2. MultiRewardManager（`instinctlab.managers.MultiRewardManager`）

**使用方式**：
当 `cfg.rewards` 是 `MultiRewardCfg` 的实例时启用。

**功能**：
对定义好的 reward group 分别计算 reward。

**配置示例**：

```python
@configclass
class RewardGroupsCfg(MultiRewardCfg):
    rewards = RewardsCfg()  # 标准 reward group
    # rewards_group_1 = RewardsCfg()  # 第二个 critic 对应的额外 group
```

### 3. Motion Reference（`instinctlab.motion_reference`）

**说明**：
用于管理运动数据源，例如 AMASS、shadowing command，以及 tracking / imitation reward。

**组成部分**：

- **MotionReferenceManager**
  负责加载和流式管理运动数据，并管理多个 `MotionBuffer`。
- **MotionBuffer**
  作为不同运动来源的统一接口，例如数据集、生成模型等。
- **Shadowing Commands**
  生成机器人需要跟随的命令。
- **Rewards**
  - `*_tracking_*`：基于时间点对齐的 tracking reward
  - `*_imitation_*`：基于当前帧对齐的 imitation reward

### 4. Virtual Obstacles（`instinctlab.terrains.virtual_obstacle`）

**说明**：
从 terrain mesh 中生成抽象几何表示，例如边缘圆柱体等。

**用途**：
可注册到传感器中，例如 `VolumePointsSensor`，用于在没有显式物理碰撞几何的情况下进行碰撞检测和穿透计算。

### 5. Noisy Grouped Sensor Camera（`instinctlab.sensors.NoisyGroupedRayCasterCamera`）

**说明**：
继承自 `GroupedRayCaster`，加入可配置噪声管线和历史缓冲区。

**目的**：
用于 sim-to-real，通过模拟深度噪声、双目噪声、延迟等传感器伪影，提升迁移能力。

### 6. Mesh Spawning（`instinctlab.sim.spawners.MeshFileCfg`）

**说明**：
用于从 mesh 文件（OBJ、STL、FBX）生成刚体对象，并自动转换成 USD。

**配置项**：

- **`make_instanceable`**
  来自 `MeshConverterCfg`。为 `True` 时使用 instancing 以节省内存，默认值为 `True`。
- **`apply_collision_props_at_spawn`**
  默认值为 `False`。
  - 为 `False` 时，碰撞属性在转换阶段 baked 进去，spawn 时不会再修改，适合 instanced geometry，且不会产生 warning。
  - 为 `True` 时，碰撞属性会在 spawn 阶段应用，便于逐实例覆盖，但要求 `make_instanceable=False`。如果两者同时为 `True`，会在 `__post_init__` 中自动修正。

**行为矩阵**：

| make_instanceable | apply_collision_props_at_spawn | 结果 |
|---|---|---|
| True | False | 默认行为，无 warning，启用 instancing，碰撞属性来自 converter |
| False | True | 支持在 spawn 时覆盖碰撞属性 |
| False | False | 不使用 instancing，碰撞属性仅来自 converter |
| True | True | 自动修正为 `make_instanceable=False` |

**示例：默认配置，无 warning**

```python
MeshFileCfg(asset_path=..., mass_props=..., collision_props=...)
```

**示例：spawn 时覆盖碰撞属性**

```python
MeshFileCfg(
    asset_path=...,
    collision_props=...,
    apply_collision_props_at_spawn=True,
    make_instanceable=False,
)
```

## Motion Buffer 的设计与用法

### 概念

`MotionBuffer`（`instinctlab.motion_reference.motion_buffer.MotionBuffer`）是运动数据源的统一接口。
它的角色有点类似 PyTorch Dataset，但专门为仿真环境设计。

其职责包括：

- **环境分配**
  每个 buffer 负责管理一部分环境实例。
- **填充数据**
  它会向 `MotionReferenceData` buffer 中填充特定时间戳下的运动帧，例如 joint positions、base pose 等。
- **状态管理**
  它负责初始化和 reset 自己所管理环境的运动状态。

### 关键类

- **`MotionBuffer`**
  基类。如果你要支持新的 motion source，需要继承它。
- **`MotionReferenceData`**
  数据类，用于保存运动帧张量，例如 `joint_pos`、`base_pos_w`、`link_pos_w` 等。
- **`MotionReferenceState`**
  数据类，用于保存 reset 时的初始状态。

### 扩展 Motion Reference：创建自定义 Buffer

如果你想支持新的数据格式或生成式 motion source，需要继承 `MotionBuffer` 并实现以下内容：

1. **`reset(self, env_ids, ...)`**
   - 重置指定环境的内部状态
   - 如有需要，更新 `symmetric_augmentation_mask_buffer`
2. **`fill_init_reference_state(self, env_ids, ...)`**
   - 用起始姿态填充 `MotionReferenceState`
   - 注意：`get_init_reference_state(env_ids)` 返回的已经是 `state[env_ids]`，shape 为 `[len(env_ids), ...]`，不要再次用 `env_ids` 重复索引
3. **`fill_motion_data(self, env_ids, sample_timestamp, ...)`**
   - 在指定时间戳采样 motion data，并填入 `MotionReferenceData`
   - **非常关键**：必须正确计算并填充 joint position、base pose、link pose 等所有张量
4. **属性**
   - `num_trajectories`：总运动轨迹数
   - `complete_motion_lengths`：每条轨迹的总时长，单位秒
5. **`get_current_motion_identifiers(self, env_ids)`**
   - 返回每个环境当前所使用运动的唯一字符串标识，例如文件名

### 关键注意事项

- **FPS 不匹配**
  motion source 的帧率可能和仿真帧率不同。
  `MotionBuffer` 必须正确处理插值或跳帧逻辑。
- **关节顺序必须一致**
  运动源文件中的 joint 顺序必须与仿真环境中的 URDF 关节顺序一致，否则 motion tracking 会完全错误。
  - **修复方式**：使用 `motion_buffer.isaac_joint_names` 获取 Isaac Sim 中的正确关节顺序，并确保在 motion loading / retargeting 阶段按此顺序重排数据。

## 如何使用

1. **安装**
   - 确保 Isaac Lab 和 `instinct_rl` 已安装
   - 通过 pip 安装 `instinctlab`
2. **训练**
   - 使用 `scripts/instinct_rl/train.py`，并传入目标 task

```bash
python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless
```

## 如何扩展

### 添加一个新任务

1. **创建独立仓库**
   - 使用 Isaac Lab Template Generator 创建新的外部项目
   - 这样能确保项目隔离，并遵循标准实践
2. **复制训练脚本**
   - 新生成的仓库默认 **不会** 包含 `instinct_rl` 的训练脚本
   - 需要把本仓库中的 `scripts/instinct_rl` 目录复制到新仓库的 `scripts/` 目录
   - 确认新仓库里包含 `scripts/instinct_rl/train.py` 等文件
3. **实现任务**
   - 在 `source/<your_project>/<your_project>/tasks/<task_name>` 下创建任务目录
   - 确保每一级目录都有 `__init__.py`
4. **注册任务**
   - 在 `__init__.py` 中使用 `gym.register`
   - **Entry Point 必须使用** `instinctlab.envs:InstinctRlEnv`
   - 需要提供 `env_cfg_entry_point` 和 `instinct_rl_cfg_entry_point`

```python
gym.register(
    id="My-Task-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env_cfg:MyEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.my_ppo_cfg:MyPPOCfg",
    },
)
```

### 自定义环境

1. **环境配置**
   - 创建继承自 `InstinctLabRLEnvCfg` 或 `ManagerBasedRLEnvCfg` 的配置类
2. **Reward**
   - 使用 `RewTermCfg` 定义 `RewardsCfg`
   - 如果是多 critic 结构，使用 `MultiRewardCfg`
3. **Sensors / Events**
   - 在对应配置类中加入自定义 sensor 或 event term

### 添加新的 Manager 或组件

- 继承 `isaaclab` 或 `instinctlab` 中的基类，例如 `ManagerTermBase`、`SensorBase`
- 然后在环境配置中注册它们
