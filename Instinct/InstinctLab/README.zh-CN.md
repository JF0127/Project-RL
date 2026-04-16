# Project Instinct

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## 概览

这个仓库是 [Project-Instinct](https://project-instinct.github.io/) 的环境侧代码仓库。

我们的目标是将面向人形机器人（足式机器人）全身控制的强化学习工程化、工业化。

**主要特性：**

- `隔离性` 在 Isaac Lab 核心仓库之外开展工作，确保你的开发过程保持自包含。
- `灵活性` 这个模板支持你的代码以 Omniverse 扩展的形式运行。
- `统一生态` 这个仓库属于 Project-Instinct 生态的一部分，该生态还包括 [instinct_rl](https://github.com/project-instinct/instinct_rl) 和 [instinct_onboard](https://github.com/project-instinct/instinct_onboard) 仓库。
  - 这个生态的核心设计理念是：将每个实验都视为一个独立的结构化文件夹，并以时间戳作为唯一标识开头。
  - 在 `play.py` 脚本中加入 `--exportonnx` 参数会将策略导出为 ONNX 模型。之后，你可以直接把整个日志目录复制到机器人电脑上，并使用 `instinct_onboard` 工作流在真机上运行该策略。

**关键词：** extension, template, isaaclab

## 警告

本代码库采用 [CC BY-NC 4.0 license](LICENSE) 许可证，并继承 IsaacLab 中的相关许可证。你不得将本仓库内容用于商业用途，例如用于展示你的商业产品、为你的商业产品做宣传演示，或将代码重新封装用于你自己的商业目的。

## 贡献

贡献指南见 [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md)。一旦你提交贡献或发起 Pull Request，即表示你同意将所提交内容的版权归属转让给项目维护者。

贡献者名单见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

## 安装

- 按照 [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab，并且**切换到 5.1.0 版本**。我们建议使用 conda 方式安装，这样更方便从终端直接调用 Python 脚本。当前发布版本所使用的 IsaacLab 提交为 `37ddf626871758333d6ed89cf64ad702aef127d0`，日期为 2026 年 1 月 30 日。

- 按照 [installation guide](https://github.com/project-instinct/instinct_rl/blob/main/README.md) 安装 Instinct-RL。
  简要步骤如下：
  ```bash
  git clone https://github.com/project-instinct/instinct_rl.git
  python -m pip install -e instinct_rl
  ```

- 将本仓库单独克隆到 Isaac Lab 安装目录之外，也就是不要放到 `IsaacLab` 目录内部：

  ```bash
  # 方式 1：HTTPS
  git clone https://github.com/project-instinct/instinctlab.git

  # 方式 2：SSH
  git clone git@github.com:project-instinct/instinctlab.git
  ```

- 使用已经安装好 Isaac Lab 的 Python 解释器安装本库：

  ```bash
  python -m pip install -e source/instinctlab
  ```

- 如果你要配合 `instinct-rl` 运行，在安装好 [instinct-rl](https://github.com/project-instinct/instinct_rl) 后，可以执行：

  ```bash
  python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless
  ```

## 关键组件文档

- [Instinct-RL Documentation](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [InstinctLab Documentation](https://github.com/project-instinct/instinctlab/blob/main/DOCS.md)

### 配置 IDE（可选）

如需配置 IDE，请按以下步骤操作：

- 在 VSCode 中运行任务：按 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，然后在下拉列表中运行 `setup_python_env`。执行过程中，系统会提示你输入 Isaac Sim 安装目录的绝对路径。

如果一切执行正常，它会在 `.vscode` 目录中生成一个 `.python.env` 文件。该文件包含 Isaac Sim 和 Omniverse 提供的所有扩展的 Python 路径，有助于 IDE 正确索引这些模块，从而在编写代码时提供更完整的智能提示。

## 代码格式化

我们提供了一个 `pre-commit` 模板用于自动格式化代码。
安装 `pre-commit`：

```bash
pip install pre-commit
```

然后你可以这样运行：

```bash
pre-commit run --all-files
```

如果你希望每次提交时自动执行 `pre-commit`，可以运行：

```bash
pre-commit install
```

## 训练你自己的项目

***为了保留你自己的代码开发历史和进展，强烈建议你参考 https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html 创建你自己的独立仓库。***

然后把 `scripts/instinct_rl` 复制到你自己的仓库中。

### 或者，如果你坚持要直接 fork 并修改这个仓库

- 请在 `source/instinctlab/instinctlab/tasks` 目录下创建一个新的文件夹，文件夹名应当是你的项目名。并且务必在每一级子目录中都添加 `__init__.py`。很多人容易漏掉这一步，结果导致本应注册成功的任务无法被发现。

- 我们继承了 IsaacLab 的 manager-based RL environment，并在此基础上增加了新功能。你在 `gym.register` 调用里应当使用 `instinctlab.envs:InstinctRlEnv` 作为 `entry_point`。例如，如果你想添加一个新任务，可以参考下面的写法：

```python
import gymnasium as gym
from . import agents
task_entry = "instinctlab.tasks.shadowing.perceptive.config.g1"
gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)
```

## 故障排查

### Pylance 无法完整索引扩展

在某些 VSCode 版本中，部分扩展可能无法被完整索引。此时可以在 `.vscode/settings.json` 中的 `"python.analysis.extraPaths"` 键下手动加入你的扩展路径。

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/instinctlab"
    ]
}
```
