# Instinct Onboard

这是 Project Instinct 的板载端代码，旨在支持在不同机器人硬件平台上进行网络推理。

***注意*** 当前项目仅在 Ubuntu 22.04、ROS2 Humble，以及 Unitree G1 的 Jetson Orin NX（29 自由度版本）上进行过测试。

## 前置依赖

- Ubuntu
- ROS2
- Python

### 安装（Unitree G1 Jetson Orin NX）

- JetPack
    ```bash
    sudo apt-get update
    sudo apt install nvidia-jetpack
    ```

- 安装 `crc` 模块

    按照 [crc_module](https://github.com/ZiwenZhuang/g1_crc) 的说明进行安装，并将产物文件 `crc_module.so` 复制到你运行 Python 脚本的位置。

- 安装 `unitree_hg` 和 `unitree_go` 的消息定义


### 安装（通用）

- 确保已经安装 ROS2 的 mcap 存储支持
    ```bash
    sudo apt install ros-{ROS_VERSION}-rosbag2-storage-mcap
    ```

- Python 虚拟环境
    ```bash
    sudo apt-get install python3-venv
    python3 -m venv instinct_venv
    source instinct_venv/bin/activate
    ```

- 安装板载侧 Python 包，并自动检测 GPU
  但默认不包含 OpenCV 相关库：
    ```bash
    pip install -e .
    ```
    这会自动检测当前环境中是否有可用的 GPU / CUDA，并安装对应版本的 ONNX Runtime。

    - 安装选项：
        ```bash
        # 默认安装（包含全部依赖，包括 OpenCV 相关库）
        pip install -e .[all]

        # 不安装 OpenCV 相关依赖
        pip install -e .[noopencv]
        ```

- 确保 `cv2` 在当前 Python 环境中可用。你可以在 Python shell 中运行 `import cv2` 进行测试。

    - 可以执行 `pip install opencv-python`，或者参考 [Geek for Geeks](https://www.geeksforgeeks.org/python/getting-started-with-opencv-cuda-module/) 的说明，自行编译支持 CUDA 的 OpenCV。

- 说明：
    - ONNX Runtime 版本会自动检测并选择，有 GPU 就装 GPU 版，否则装 CPU 版
    - 你也可以通过环境变量覆盖自动检测：`FORCE_CPU=1 pip install -e .` 或 `FORCE_GPU=1 pip install -e .`
    - 如果你打算自行从源码构建 GPU 版 OpenCV，可以使用 `[noopencv]` 选项安装 `instinct_onboard`


## 代码结构简介

### ROS 节点

- 在 `instinct_onboard/ros_nodes/` 中，可以找到用于与机器人通信的 ROS 节点。

- 为避免菱形继承，每个按功能划分的 ROS 节点都应在单独文件中以 Mixin class 的形式实现。

- 在主入口脚本中，你应当按需继承所需内容，并同时组合状态机逻辑。主入口脚本位于 `scripts/`。

### Agents

- 在 `instinct_onboard/agents/` 中，可以找到用于运行网络以及收集观测的 agents。

- 不要对网络输出动作做缩放。动作缩放是在 ROS 节点一侧完成的。
