# MuJoCo Sim2Sim

`depoly/` 现在只服务一个目标：
把 `Mvr_22dof_six` 在 Isaac Lab 训练出的策略接到 MuJoCo 做 sim2sim。

## 当前结构

```text
depoly/
├── main.py
├── config/
│   ├── base_config.py
│   └── mvr_22dof.py
├── runners/
│   └── sim2sim_runner.py
└── utils/
    ├── engine.py
    ├── logger.py
    ├── math_utils.py
    ├── observation.py
    └── viz.py
```

## 运行方式

仓库根目录下执行：

```bash
python depoly/main.py
```

可选覆盖路径：

```bash
python depoly/main.py \
  --xml-path /abs/path/to/Mvr_22dof.xml \
  --policy-path /abs/path/to/policy.pt
```

## 日志可视化

默认读取 `depoly/logs` 下最新的日志，并按关节名画图：

```bash
python depoly/utils/viz.py --joint left_knee_joint
```

保存图片而不是弹窗显示：

```bash
python depoly/utils/viz.py --joint left_knee_joint --save-fig
```

指定某个日志文件：

```bash
python depoly/utils/viz.py \
  --csv-file depoly/logs/sim2sim_log_xxx.csv \
  --joint right_knee_joint
```

一次看多个关节的动作：

```bash
python depoly/utils/viz.py \
  --mode multi \
  --joints left_hip_pitch_joint left_knee_joint right_hip_pitch_joint right_knee_joint
```

左右腿同名关节对比：

```bash
python depoly/utils/viz.py \
  --mode compare \
  --left-joint left_knee_joint \
  --right-joint right_knee_joint
```

只看步态相位：

```bash
python depoly/utils/viz.py --mode phase
```

列出当前支持的策略控制关节：

```bash
python depoly/utils/viz.py --list-joints
```

## 当前对齐约定

- 机器人固定为 `Mvr_22dof`
- 引擎固定为 MuJoCo
- runner 固定为 sim2sim
- 动作维度固定为 `6`
- 观测维度固定为 `55`
- 控制关节固定为 `Mvr_22dof_six` 的 6 个腿部关节
- 全身其余关节保持在 default pose

## 配置项

当前所有运行参数都集中在 `config/mvr_22dof.py`：

- 路径: `xml_path`, `policy_path`, `log_path`
- 时序: `dt`, `decimation`
- 初始状态: `base_init_pos`, `default_dof_pos`
- 命令: `command`, `gait_period`
- 观测缩放: `lin_vel_scale`, `ang_vel_scale`, `dof_pos_scale`, `dof_vel_scale`
- 控制与对齐: `joint_names`, `policy_joint_names`, `observation_joint_names`
- 控制参数: `kps`, `kds`, `action_scales`, `clip_actions`, `torque_limits`

## 调试重点

- `config/mvr_22dof.py`
  这里定义 joint 顺序、policy 关节、观测关节、PD 参数和默认姿态
- `runners/sim2sim_runner.py`
  这里定义观测构造、策略推理、action 拼回 full-body target 的逻辑
- `utils/engine.py`
  这里定义 MuJoCo joint/actuator 映射和 PD 力矩执行

## 常见问题

- 启动即报维度错误
  检查 `policy.pt` 是否真的是 `Mvr_22dof_six` 导出的策略
- 启动即报配置错误
  检查 `num_actions` / `num_observations` 是否和 `policy_joint_names` / `observation_joint_names` 一致
- 动作表现不对
  优先检查 `policy_joint_names`、`observation_joint_names` 和 `joint_names` 的顺序是否一致
