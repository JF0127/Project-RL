# Instinct RL（基于 Rsl-RL 代码库，但做了大量修改）

## 警告

本代码库采用 [CC BY-NC 4.0 license](LICENSE) 许可证，并继承 IsaacLab 中的相关许可证。你不得将本仓库内容用于商业用途，例如用于展示你的商业产品、为你的商业产品做宣传演示，或将代码重新封装用于你自己的商业目的。

## 贡献

贡献指南见 [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md)。一旦你提交贡献或发起 Pull Request，即表示你同意将所提交内容的版权归属转让给项目维护者。

贡献者名单见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

---

## 安装

- 将本仓库单独克隆到 Project Instinct 安装目录之外：

  ```bash
  # 方式 1：HTTPS
  git clone https://github.com/project-instinct/instinct_rl.git
  # 方式 2：SSH
  git clone git@github.com:project-instinct/instinct_rl.git
  ```

- 使用任意 Python 解释器安装：

  ```bash
  python -m pip install -e instinct_rl
  ```

---

## 重要的新概念

### 通用观测格式

- **obs_format**：一个 `OrderedDict`，用于定义输入观测的格式。
  键：`obs_segment` 的名称。对于算法侧来说，它对应 IsaacLab 中 `ManagerBasedRlEnv` 里的 `obs_group`。
  值：一个 `obs_segment` 对象，见下文。

- **obs_segment**：一个 `OrderedDict`，用于定义观测中的某个片段。
  键：观测项名称，对应 IsaacLab 中 `ManagerBasedRlEnv` 里的 `obs_term_name`。
  值：该观测片段各部分的形状。

- **obs_pack**：按照 `obs_format` 定义组织起来的观测字典。
  键：`obs_segment` 名称，对应 IsaacLab 中 `ManagerBasedRlEnv` 的 `observation_manager`。
  值：每个观测片段展平后的 tensor / vector，算法侧要求使用这种形式。

- **obs_component**：对应 IsaacLab 中 `ManagerBasedRlEnv` 的 `obs_term`。

---

## 如何使用可替换算法与网络模块设计

在 `on_policy_runner.py` 中，有如下代码：

```python
actor_critic = modules.build_actor_critic(
    self.policy_cfg.pop("class_name"),
    self.policy_cfg,
    obs_format,
    num_actions=env.num_actions,
    num_rewards=env.num_rewards,
).to(self.device)
```

这段代码用于构建 actor-critic 网络。`class_name` 是 actor-critic 类名，`policy_cfg` 是 actor-critic 的配置，`obs_format` 是观测格式，`num_actions` 是动作维度数量，`num_rewards` 是奖励维度数量。

`modules.build_actor_critic` 是一个工厂函数，它根据 `class_name` 和 `policy_cfg` 来构建 actor-critic 网络。

`class_name` 可以是本仓库中实现的某个 actor-critic 类名，也可以是你自己实现的 actor-critic 类的完整导入路径，格式为 `module_name:class_name`。

---

## 算法

### PPO

标准的 Proximal Policy Optimization（PPO）强化学习算法。实现了 PPO 的 clipping surrogate loss、value function loss 和 entropy regularization。支持每个环境步进行多轮学习 epoch 和 mini-batch 更新。

### State Estimator

用于从部分观测中学习状态表征与状态估计的算法。它结合重建目标与预测目标，学习有意义的隐状态表示，并可进一步用于控制任务。

### AMP（WASABI）

WASABI 的实现，以 AMP 的算法框架为起点。它使用带判别器的对抗训练，从专家演示中学习运动先验。支持多种判别器结构，包括 BCE loss、Wasserstein loss 和 MSELoss，并包含 gradient penalty 以及多种判别器架构。

### Distillation 和 DAgger（TPPO）

教师-学生蒸馏框架，在 PPO 基础上增加教师网络指导。实现了 Dataset Aggregation（DAgger）和知识蒸馏技术。允许在保持 PPO 稳定性的前提下，利用专家演示进行学习。支持多种教师动作选择概率和蒸馏损失系数。

### VAE Distillation

教师-学生蒸馏框架，在 TPPO 基础上进一步引入 VAE 学生网络。它使用 VAE 对学生动作进行编码和解码来生成输出动作。该方法会生成独立于动作分布之外的额外 latent distribution，因此为将 VAE 蒸馏与 PPO 结合提供了可能性。

---

## 网络模块

### Actor Critic

标准 actor-critic 架构，策略网络（actor）和价值网络（critic）分离。支持可配置的隐藏层维度、激活函数以及观测处理方式。能够正确处理连续动作空间和离散动作空间，并构造相应的概率分布。

### MoE Actor Critic

Mixture of Experts（MoE）版 actor-critic 架构。它使用多个 expert 网络，并通过 gating 机制让不同 expert 专门处理状态空间或动作空间中的不同区域。这样可以在保持参数效率的同时，通过专业化分工提升复杂任务的性能。

### VAE Actor Critic

Variational Autoencoder（VAE）版 actor-critic 架构。它先使用 VAE 学习观测的潜在表示，再由 actor 根据潜在表示预测动作。可用于降维、特征学习，或者作为更大强化学习流水线的一部分。

### 与 Encoder 相关的模块

用于处理不同输入模态的一组编码器架构：

- **Encoder Actor Critic**：带编码器骨干网络的 actor-critic，用于处理图像或复杂状态表示等高维输入
- **All Mixer**：用于组合不同观测类型的多模态输入处理模块
- **VQ-VAE**：用于离散潜变量表示的 Vector Quantized Variational Autoencoder
- **State Estimator**：用于从部分观测中估计隐藏状态的网络
