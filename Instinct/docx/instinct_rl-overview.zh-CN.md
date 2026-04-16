# Instinct-RL 概览

## 设计原则

Instinct-RL 是一个基于 **Rsl-RL** 的模块化强化学习库，设计目标包括：

- **灵活性**：算法与网络模块可以互换
- **模块化**：算法、存储、网络架构之间边界清晰
- **高级特性**：支持状态估计、运动先验 AMP/WASABI、蒸馏 DAgger/TPPO，以及基于 VAE 的方法

## 核心组件

### 1. Observation Format（`obs_format`）

- **结构**：一个 `OrderedDict`，用于定义输入观测
- **层次结构**：
  - `obs_format`：顶层字典，例如 `{'policy': ..., 'critic': ...}`
  - `obs_segment`：定义观测项及其形状
  - `obs_pack`：实际传给算法的 observation tensor 字典

### 2. Algorithms（`instinct_rl.algorithms`）

- **PPO**：标准 Proximal Policy Optimization
- **State Estimator**：用于学习隐状态表征
- **AMP（WASABI）**：用于模仿学习的对抗式运动先验
- **TPPO**：教师-学生蒸馏与 DAgger
- **VAE Distillation**：基于变分自编码器学生网络的蒸馏方法

### 3. Modules（`instinct_rl.modules`）

- **ActorCritic**：策略网络与价值网络的基础类
- **MoE Actor Critic**：Mixture of Experts 架构
- **VAE Actor Critic**：基于 VAE 的策略架构
- **Encoder Modules**：用于处理图像等高维输入的编码器结构

### 4. Factory Pattern

- **用法**：使用 `modules.build_actor_critic` 根据配置构建网络
- **动态加载**：支持通过类名或完整导入路径加载自定义架构

## 如何使用

1. **安装**
   - 通过 pip 安装：`pip install -e instinct_rl`
2. **集成方式**
   - 主要被 `instinctlab` 的训练脚本调用
3. **配置方式**
   - 算法和策略通过字典配置传入 runner

## 如何扩展

### 添加一个新算法

1. **位置**
   - 在 `instinct_rl/algorithms/` 中创建新文件
2. **接口要求**
   - 新算法必须兼容 `instinct_rl.runners.on_policy_runner.OnPolicyRunner`
   - 需要实现如下方法：
     - `__init__(self, actor_critic, ...)`
     - `act(self, obs, critic_obs)`
     - `process_env_step(self, rewards, dones, infos, ...)`
     - `update(self, current_learning_iteration)`
     - `init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)`
     - `test_mode(self)` / `train_mode(self)`
     - `save(self, path)` / `load(self, path)`
3. **网络兼容性**
   - 算法接收的网络模块不一定必须继承 `ActorCritic`
   - 但如果算法内部复用了 PPO，网络就必须满足 PPO 所需的 `ActorCritic` 接口
4. **注册**
   - 建议将其导入 `instinct_rl/algorithms/__init__.py`

### 添加一个新网络模块

1. **位置**
   - 在 `instinct_rl/modules/` 中创建新文件
2. **继承方式**
   - 普通自定义网络可以继承 `nn.Module`
   - 只有在网络要配合标准 PPO 或依赖 `ActorCritic` 接口的算法时，才需要继承 `ActorCritic`
3. **观测处理**
   - 模块必须正确使用 `obs_format` 来切分和处理输入

```python
class MyNetwork(nn.Module):  # 或者使用 ActorCritic
    def __init__(self, obs_format, num_actions, ...):
        super().__init__()
        # 构建自定义网络层
```

4. **使用方式**
   - 在 policy 配置中引用该类，例如：
     - `class_name: MyNetwork`
     - `module.path:MyNetwork`

### 自定义 PPO

- **继承方式**
  - 可以继承 `PPO` 并重写 `compute_losses` 或 `compute_auxiliary_reward`
- **辅助奖励**
  - 可以通过实现 `compute_auxiliary_reward`，基于 observation 或内部状态添加自定义 reward term
