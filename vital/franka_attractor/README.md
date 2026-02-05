# Franka Attractor 项目

Franka Panda 机器人通过虚拟吸引子进行位置控制和抓取-放置任务演示。

## 项目结构

```
franka_attractor/
├── main.py                          # 主程序入口
├── config.py                        # 全局配置参数
│
├── core/                            # 核心模拟功能模块
│   ├── __init__.py
│   ├── simulation.py                # 模拟环境初始化、光照、地面等
│   ├── assets.py                    # 资产加载和管理
│   └── scene.py                     # 场景构建（机器人、立方体等）
│
├── systems/                         # 独立的功能系统模块
│   ├── __init__.py
│   ├── camera.py                    # 摄像头管理系统
│   ├── attractor.py                 # 吸引子控制系统
│   ├── gripper.py                   # 夹爪控制
│   ├── planner.py                   # 抓取-放置动作规划
│   └── sensor.py                    # 力传感器数据处理
│
├── utils/                           # 工具函数模块
│   ├── __init__.py
│   ├── math_utils.py                # 数学运算（lerp、slerp等）
│   ├── io_utils.py                  # 数据保存和I/O操作
│   └── visualization.py             # 坐标系绘制、可视化
│
├── data/                            # 数据输出目录（运行时生成）
│   └── .gitkeep
│
└── README.md                        # 项目说明文档
```

## 功能模块说明

### core/ - 核心模拟功能

- **simulation.py**: 创建物理模拟环境、配置参数、初始化查看器
- **assets.py**: 加载 URDF 文件中的机器人、工作台和立方体等资产，配置力传感器
- **scene.py**: 构建完整的模拟场景，包括机器人演员、环境对象、吸引子等

### systems/ - 独立功能系统

- **camera.py**: 创建、配置和管理多个摄像头，包括固定摄像头和眼在手上摄像头
- **sensor.py**: 处理力传感器数据，包括零点校准、数据读取和保存
- **gripper.py**: 控制夹爪的开合动作
- **attractor.py**: 管理吸引子位置更新和可视化
- **planner.py**: 生成动作规划序列，管理采集系统

### utils/ - 工具函数

- **math_utils.py**: 向量/四元数插值、指尖位置计算等
- **io_utils.py**: 目录管理、文件名生成等
- **visualization.py**: 坐标系绘制、指尖标记等可视化函数

## 配置参数

所有可配置参数集中在 `config.py` 中的 `SimulationConfig` 类中，包括：

- **物理参数**: 时间步长、子步数、求解器参数等
- **资产路径**: URDF 文件路径
- **场景参数**: 立方体位置、工作台位置等
- **摄像头参数**: 分辨率、位置、旋转角度等
- **控制参数**: 吸引子刚度/阻尼、夹爪宽度等
- **采集参数**: 摄像头采集频率、持续时间等

## 运行方法

### 基本运行

```bash
python main.py
```

### 带参数运行

```bash
# 使用 PhysX 物理引擎
python main.py --physics_engine=physx

# 使用 GPU 加速
python main.py --use_gpu

# 指定 GPU 设备
python main.py --graphics_device_id=0
```

## 运行输出

运行时会在以下位置生成输出数据：

```
camera_outputs/
├── camera_0/        # 摄像头1的图像序列
├── camera_1/        # 摄像头2的图像序列
├── camera_2/        # 摄像头3的图像序列
├── camera_3/        # 摄像头4的图像序列
├── camera_4/        # 眼在手上摄像头的图像序列
└── gel/             # 力传感器数据（numpy 和文本格式）
    ├── 0000.npy     # 第0帧的力/力矩数据
    ├── 0000.txt     # 第0帧的力/力矩数据（文本格式）
    └── ...
```

## 动作规划序列

程序执行以下 10 个步骤的抓取-放置任务：

1. **move_pregrasp** (2.0s): 从初始位置移动到上方立方体上方
2. **descend_grasp** (2.0s): 向下靠近立方体
3. **close_gripper** (1.5s): 关闭夹爪抓住立方体
4. **stabilize_after_grasp** (0.5s): 稳定抓取
5. **lift** (1.5s): 向上提起立方体
6. **move_over_drop** (3.0s): 移动到下方立方体上方
7. **stabilize_before_place** (0.5s): 放置前稳定
8. **place_release** (2.0s): 降低立方体到放置位置
9. **open_gripper** (1.5s): 打开夹爪释放立方体
10. **retreat** (1.5s): 撤退回到安全位置

总耗时约 17 秒。

## 可视化坐标系

启用 `Config.VISUALIZE_AXES = True` 来显示：

- **基座坐标系**: 机器人基座的世界坐标系
- **手坐标系**: 末端执行器（手）的坐标系
- **指尖坐标系**: 左右指尖的实际接触点位置
- **摄像头坐标系**: 各摄像头的视图方向

## 力传感器

在右指尖安装了力/力矩传感器，可以测量：

- 力的三个分量 (Fx, Fy, Fz) - 单位：N
- 力矩的三个分量 (Tx, Ty, Tz) - 单位：Nm

传感器在任务开始前 0.5 秒进行自动零点校准。

## 依赖

- Python 3.x
- Isaac Gym
- PyTorch
- NumPy

## 许可证

Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

## 注意事项

1. 所有参数都可以在 `config.py` 中修改
2. 摄像头参数使用列表格式，易于添加新摄像头
3. 摄像头采集通过计时器自动管理，不会阻塞模拟循环
4. 力传感器数据保存为 numpy 文件和文本文件便于查看
