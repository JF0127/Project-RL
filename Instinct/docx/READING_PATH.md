# Project Instinct 阅读路径

## 目标

这份文档用于帮助你在较短时间内系统掌握 Project Instinct。默认前提是：

- 你已经熟悉 Isaac Lab
- 你要重点理解 Instinct 相对 Isaac Lab 多出来的部分
- 你的目标不是只会运行，而是能追完整条任务链路

## 总原则

不要按目录顺序读，也不要先横扫所有 task。

推荐策略是：

1. 先建立项目全图
2. 再打通一个代表性任务
3. 再按专题精读核心模块
4. 最后横向比较不同任务和算法变体

## 最推荐的起点

先读这个任务：

- `Instinct-Perceptive-Shadowing-G1-v0`

原因：

- 它覆盖 `InstinctLab` 的核心任务抽象
- 它包含 `motion reference`
- 它包含 `perceptive` 输入
- 它能连到 `instinct_rl`
- 它有明确的 `instinct_onboard` 对应 agent

## 阶段一：建立全图

先读以下文件：

1. [AGENT.md](/home/jhl/projects/Project-RL/Instinct/docx/AGENT.md)
2. [InstinctLab-README.zh-CN.md](/home/jhl/projects/Project-RL/Instinct/docx/InstinctLab-README.zh-CN.md)
3. [instinct_rl-README.zh-CN.md](/home/jhl/projects/Project-RL/Instinct/docx/instinct_rl-README.zh-CN.md)
4. [instinct_onboard-README.zh-CN.md](/home/jhl/projects/Project-RL/Instinct/docx/instinct_onboard-README.zh-CN.md)
5. [InstinctLab-overview.zh-CN.md](/home/jhl/projects/Project-RL/Instinct/docx/InstinctLab-overview.zh-CN.md)
6. [instinct_rl-overview.zh-CN.md](/home/jhl/projects/Project-RL/Instinct/docx/instinct_rl-overview.zh-CN.md)

这一阶段你要能回答：

- `InstinctLab`、`instinct_rl`、`instinct_onboard` 分别负责什么
- 一个训练实验的产物最后为什么能在真机上复用
- 项目最重要的特有概念是什么

## 阶段二：打通一个任务

推荐按下面顺序阅读：

1. task 注册文件
2. 训练入口
3. env config
4. `InstinctRlEnv`
5. `InstinctRlVecEnvWrapper`
6. policy config
7. `OnPolicyRunner`
8. 对应算法
9. onboard agent

如果你以 `Perceptive Shadowing G1` 为例，建议重点追问：

- task id 是什么
- 任务的 env cfg 和 instinct_rl cfg 分别在哪里
- 参考动作是如何进入环境的
- depth image 是如何进入 policy 的
- obs 是怎样从 Isaac Lab manager 输出，变成 instinct_rl 输入的
- 训练产物怎样在 onboard 侧重建

## 阶段三：按专题精读

### 专题 1：Motion Reference

这是项目最关键的任务抽象之一。

建议读：

- `InstinctLab/DOCS.md`
- `instinctlab.motion_reference`
- 与 reference 强相关的 commands / observations / rewards

你要重点搞清楚：

- `MotionBuffer`
- `MotionReferenceData`
- `MotionReferenceState`
- shadowing、tracking、imitation 的区别

### 专题 2：环境扩展

建议读：

- `InstinctRlEnv`
- `MultiRewardManager`
- `MonitorManager`

你要重点搞清楚：

- 它比标准 `ManagerBasedRLEnv` 多了什么
- 多 reward 如何进入训练侧
- 监控项如何进入日志系统

### 专题 3：RL 框架

建议读：

- `OnPolicyRunner`
- `PPO`
- `rollout_storage`
- `modules`

你要重点搞清楚：

- `obs_format` 的作用
- actor-critic 如何根据配置构建
- 为什么它强调 algorithm/network 可替换

### 专题 4：部署

建议读：

- `instinct_onboard/agents/base.py`
- `shadowing_agent.py`
- `scripts/`

你要重点搞清楚：

- 为什么 onboard 直接读取训练 logdir
- observation 是如何按训练配置重建的
- action scale 为什么不在网络输出侧做

## 阶段四：横向比较

在你吃透一个代表性任务后，再横向比较不同任务族：

- `locomotion`
- `shadowing/whole_body`
- `shadowing/perceptive`
- `shadowing/perceptive_hoi`
- `shadowing/beyondmimic`
- `parkour`

你要比较的是：

- 输入模态差异
- reference 依赖差异
- reward 设计差异
- policy 结构差异
- 是否使用 AMP / distillation / VAE / MoE

## 阅读时的固定问题

每读一个文件，都建议问这 6 个问题：

1. 这个文件在整个系统中的位置是什么
2. 它消费哪些上游数据
3. 它产出哪些下游数据
4. 它是 Isaac Lab 原生逻辑，还是 Instinct 增加的逻辑
5. 它是否和 `motion reference` 相关
6. 它是否影响 sim-to-real 部署

## 不推荐的阅读方式

- 不要先按目录递归扫完所有文件
- 不要先读 `thirdparty/IsaacLab`
- 不要一开始就看所有算法
- 不要先横向比较所有 task
- 不要把注意力放在琐碎工具函数上

## 建议输出

每完成一个阶段，至少写下：

- 一张流程图
- 一页结构化笔记
- 5 个你已经能回答的问题
- 3 个还不确定的问题

如果你最终能清楚回答下面这些问题，就说明主干已经通了：

1. `InstinctLab` 相对 Isaac Lab 增加了什么
2. `instinct_rl` 相对 Rsl-RL 增加了什么
3. 一个 task 如何从注册走到训练
4. `motion reference` 如何进入 obs、command、reward
5. policy 怎样支持视觉和参考动作混合输入
6. 训练产物如何被 `instinct_onboard` 复用
