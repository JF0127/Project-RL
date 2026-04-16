# Project Instinct 项目导读

## 项目是什么

Project Instinct 是一个围绕人形机器人全身控制构建的强化学习生态。它不是单一仓库，而是由三个相互衔接的子系统组成：

- `InstinctLab`：环境与任务定义层
- `instinct_rl`：训练算法与网络模块层
- `instinct_onboard`：真机推理与 ROS2 部署层

整个项目的目标是把 humanoid whole-body control 的训练、导出和部署组织成一条可复用的工程链路，而不是只做单次实验代码。

## 三个子系统分别做什么

### 1. InstinctLab

`InstinctLab` 是项目的环境侧，建立在 Isaac Lab 之上。

它主要负责：

- 定义任务、场景、机器人、地形、传感器
- 定义 observation、action、reward、termination、curriculum
- 管理 `motion reference`、`shadowing command`、tracking/imitation reward
- 扩展 Isaac Lab 的 `ManagerBasedRLEnv`
- 把环境包装成能够交给 `instinct_rl` 训练的形式

它最重要的扩展包括：

- `InstinctRlEnv`
- `MultiRewardManager`
- `MonitorManager`
- `MotionReferenceManager`

### 2. instinct_rl

`instinct_rl` 是项目的强化学习训练库，来源于 Rsl-RL，但做了大量扩展。

它主要负责：

- runner
- PPO 及其扩展算法
- 可替换的 actor-critic / encoder / MoE / VAE 网络
- rollout storage
- obs 格式化与 normalizer
- checkpoint 保存和恢复

它的核心设计特点是：

- 算法和网络模块可以互换
- 支持结构化 observation，而不是只支持单一平铺向量
- 支持 AMP/WASABI、TPPO、VAE distillation 等高级训练形式

### 3. instinct_onboard

`instinct_onboard` 是板载部署部分，用于把训练好的策略运行在真实机器人上。

它主要负责：

- 读取训练日志目录中的 `env.yaml` 和 `agent.yaml`
- 读取导出的 ONNX 模型
- 在 ROS2 节点中收集机器人状态与传感器数据
- 以和训练时一致的方式构造 observation
- 运行 ONNX 推理得到动作
- 将动作发送给机器人控制接口

它不是训练代码，而是训练结果的执行层。

## 项目主链路

这个项目的典型工作流如下：

1. 在 `InstinctLab` 中定义任务
2. 用 `scripts/instinct_rl/train.py` 启动训练
3. 由 `instinct_rl` 构建策略网络和算法并完成训练
4. 在 `play.py` 等流程中导出 ONNX
5. 将实验 `logdir` 复制到机器人端
6. 用 `instinct_onboard` 读取配置和模型进行真机推理

可以把它简化理解为：

`定义训练世界 -> 训练策略 -> 真机运行策略`

## 项目里的关键概念

### 1. Structured Observation

`instinct_rl` 使用结构化观测格式：

- `obs_format`：定义有哪些 observation group
- `obs_segment`：定义每个 group 中有哪些观测项及其形状
- `obs_pack`：算法真正接收到的观测张量字典

这使得项目能够自然支持：

- policy / critic 不同输入
- 图像和本体状态混合输入
- motion reference 等多模态输入

### 2. Motion Reference

`motion reference` 是 Instinct 中最核心的任务抽象之一。

它负责：

- 读取运动数据
- 在仿真过程中按时间提供参考轨迹
- 生成 shadowing command
- 为 tracking / imitation reward 提供参考目标

它常见于：

- shadowing
- perceptive shadowing
- parkour

### 3. Multi Reward / Multi Critic

`InstinctRlEnv` 支持多 reward group，这使得环境可以输出多路 reward，供多 critic 或 advantage mixing 使用。

### 4. Sim-to-Real

这个项目从设计上就考虑了 sim-to-real：

- 环境里有 noisy camera、virtual obstacle 等机制
- 训练结果支持导出 ONNX
- 板载推理会复用训练配置来重建 observation

## 推荐阅读顺序

如果你要快速掌握这个项目，建议按以下顺序阅读：

1. `InstinctLab/README.md`
2. `instinct_rl/README.md`
3. `instinct_onboard/README.md`
4. `InstinctLab/.cursor/rules/instinctlab-overview.mdc`
5. `instinct_rl/.cursor/rules/instinct_rl-overview.mdc`
6. 一个具体 task 的完整链路

推荐从这个任务开始：

- `Instinct-Perceptive-Shadowing-G1-v0`

因为它覆盖了：

- task 注册
- motion reference
- perceptive observation
- InstinctLab 到 instinct_rl 的桥接
- 真机 shadowing agent 的部署对应关系

## 一条任务链路应该怎么读

按下面顺序追一条完整任务最有效：

1. task 注册文件
2. `train.py`
3. env config
4. `InstinctRlEnv`
5. `InstinctRlVecEnvWrapper`
6. policy config
7. `OnPolicyRunner`
8. 对应算法
9. `instinct_onboard` 中对应 agent

## 适合你的学习目标

如果你的目标是“完全熟悉代码”，不要从目录遍历开始，而应该先回答这几个问题：

1. `InstinctLab` 相对 Isaac Lab 多了什么
2. `instinct_rl` 相对 Rsl-RL 多了什么
3. 一个 task 如何从注册走到训练
4. `motion reference` 如何进入 obs、command、reward
5. 训练结果如何导出并在 `instinct_onboard` 上运行

把这几条主线吃透之后，再去横向比较不同 task，会比从文件夹扫读高效得多。
