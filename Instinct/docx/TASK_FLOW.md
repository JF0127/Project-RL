# 任务全流程拆解

## 目的

这份文档用于帮助你追踪一个任务从注册到训练，再到导出和 onboard 部署的完整链路。

推荐以：

- `Instinct-Perceptive-Shadowing-G1-v0`

作为主样本任务。

## 全局流程

一个任务在 Project Instinct 中的典型生命周期如下：

1. 在 `InstinctLab` 中注册 task
2. 训练入口加载 task 对应的 env config 和 instinct_rl config
3. 创建 Isaac Lab 环境，并使用 `InstinctRlEnv` 扩展
4. 用 wrapper 将环境输出桥接给 `instinct_rl`
5. 由 runner 构建网络和算法并执行训练
6. 保存日志、配置和 checkpoint
7. 导出 ONNX
8. 在 `instinct_onboard` 中读取 `logdir` 和 ONNX 模型运行真机推理

## 第 1 步：Task 注册

task 注册文件通常位于：

- `source/instinctlab/instinctlab/tasks/.../__init__.py`

它的职责是：

- 定义 task id
- 指定 `entry_point`
- 指定 `env_cfg_entry_point`
- 指定 `instinct_rl_cfg_entry_point`

在这个项目里，注册时最关键的约定是：

- `entry_point` 应使用 `instinctlab.envs:InstinctRlEnv`

这意味着后续环境不是标准 Isaac Lab env，而是带有 Instinct 扩展功能的环境。

## 第 2 步：训练入口

主入口通常是：

- `InstinctLab/scripts/instinct_rl/train.py`

它会做这些事情：

- 解析 CLI 参数
- 通过 Hydra 根据 task 读取 env cfg 和 instinct_rl cfg
- 启动 Isaac Sim
- 创建 gym 环境
- 将环境包装成 `InstinctRlVecEnvWrapper`
- 构建 `OnPolicyRunner`
- 启动训练

这一层的关键作用是把 task 配置和 RL 配置真正接到一起。

## 第 3 步：环境配置

env config 决定任务“训练世界”的内容。

这里通常会定义：

- scene
- robot
- terrain
- sensors
- actions
- observations
- rewards
- terminations
- events
- curriculum
- motion_reference

对 `Perceptive Shadowing G1` 这种任务来说，最重要的不是基础 scene，而是：

- 参考动作从哪里来
- depth image 如何配置
- 机器人要跟随什么 reference
- reward 是围绕 tracking、imitation 还是 shadowing 设计的

## 第 4 步：InstinctRlEnv

`InstinctRlEnv` 是对 Isaac Lab `ManagerBasedRLEnv` 的扩展。

它比基础环境多出的关键能力是：

- `MultiRewardManager`
- `MonitorManager`

它的意义是：

- 让环境支持多 reward group
- 让环境额外记录更丰富的监控项

因此它不是简单替代品，而是整个项目的环境扩展核心。

## 第 5 步：Wrapper 桥接

环境和 RL 框架之间通过 `InstinctRlVecEnvWrapper` 连接。

它负责：

- 读取 Isaac Lab observation manager 输出
- 按 group flatten observation
- 把 reward dict 堆叠成训练可用 tensor
- 提供 `obs_format`

这一层非常关键，因为：

- `InstinctLab` 是任务定义层
- `instinct_rl` 是训练层
- wrapper 是两者的接口协议层

## 第 6 步：Policy / Algorithm 配置

`instinct_rl` 的配置一般与 env config 分开。

它通常定义：

- 使用哪个 policy 类
- 使用哪个 algorithm 类
- actor / critic hidden dims
- encoder 配置
- PPO 参数
- normalizer 配置

如果是 perceptive task，你通常会看到：

- depth image encoder
- proprioception 与 image encoder 混合输入
- policy / critic 的不同结构

## 第 7 步：Runner

`OnPolicyRunner` 的职责是：

- 从 env 读取 `obs_format`
- 根据配置构建 actor-critic
- 根据配置构建算法类
- 执行 rollout
- 调用 update
- 写日志
- 保存 checkpoint

这一层是训练主循环的控制中心。

## 第 8 步：Algorithm

算法层负责：

- 计算动作
- 接收环境反馈
- 存储 rollout
- 计算 returns
- 执行更新

标准样本是 `PPO`，但这个项目还支持：

- WASABI / AMP
- TPPO
- VAE distillation

因此在看任务时，要区分：

- 任务结构是谁定义的
- 优化过程是谁定义的

## 第 9 步：训练产物

训练运行后，通常会产生一个结构化 `logdir`，其中包括：

- `params/env.yaml`
- `params/agent.yaml`
- checkpoint
- 可能的导出模型

这个 `logdir` 不只是日志目录，也是后续部署的重要输入。

## 第 10 步：ONNX 导出

在某些运行流程中，策略会导出成 ONNX。

项目设计上允许将导出的模型和整个 `logdir` 一起复制到机器人电脑上。

这样做的意义是：

- 模型参数和训练配置一起迁移
- onboard 不需要手工重写 observation 规格

## 第 11 步：Onboard 部署

`instinct_onboard` 读取训练时保存的配置和导出模型。

它会：

- 从 `env.yaml` 中解析 action scale、offset、PD gains、observation 配置
- 从 `agent.yaml` 中解析网络和 encoder 的结构约定
- 加载 ONNX
- 重建 observation
- 执行动作推理

因此，训练和部署之间的桥梁不是“口头约定”，而是具体落在 `logdir` 中的配置文件和导出模型。

## 你在追任务时应该重点回答的问题

1. task 在哪里注册
2. env cfg 和 instinct_rl cfg 分别在哪里
3. 参考动作是如何进入环境的
4. observations 是如何组织的
5. wrapper 如何把环境输出变成训练输入
6. policy 如何处理不同模态输入
7. reward 是单组还是多组
8. checkpoint 和 ONNX 在哪里产生
9. onboard 如何读取训练产物

## 一个任务读通后的标志

如果你已经能独立说清楚下面这些内容，说明这条任务链路基本读通了：

- task id 对应哪个 env config 和哪个 policy config
- 机器人看到了什么，输出了什么
- reward 依赖哪些 reference 或 sensor
- 训练循环的核心对象有哪些
- 训练结果如何被部署侧复用
