"""
快速使用指南
===========

本指南说明如何使用 franka_attractor 项目
"""

## 项目文件结构总结

```
franka_attractor/
├── main.py                          # ⭐ 主程序入口（运行这个文件）
├── config.py                        # 🔧 全局配置参数（修改配置在这里）
│
├── core/                            # 🏗️ 核心模拟功能
│   ├── simulation.py                # 环境初始化、物理参数配置
│   ├── assets.py                    # 加载URDF资产、配置传感器
│   └── scene.py                     # 构建场景、创建演员对象
│
├── systems/                         # ⚙️ 功能系统模块
│   ├── camera.py                    # 摄像头创建、图像保存
│   ├── sensor.py                    # 力传感器校准、数据保存
│   ├── gripper.py                   # 夹爪开合控制
│   ├── attractor.py                 # 吸引子位置更新
│   └── planner.py                   # 动作规划、采集管理
│
├── utils/                           # 🛠️ 工具函数库
│   ├── math_utils.py                # 向量插值、四元数运算
│   ├── io_utils.py                  # 文件/目录操作
│   └── visualization.py             # 坐标系绘制、可视化
│
├── data/                            # 📁 数据输出目录
└── README.md                        # 📖 完整文档

```

## 快速开始

### 1. 运行程序

```bash
cd /home/neuzz/isaacgym/vital/franka_attractor
python main.py
```

### 2. 修改配置

编辑 `config.py` 中的参数，例如：

```python
# 改变摄像头采集频率
CAPTURE_FREQUENCY = 20          # 从10改为20 (每秒20帧)

# 改变吸引子刚度
ATTRACTOR_STIFFNESS = 1e7       # 增加刚度使跟踪更紧密

# 改变立方体位置
CUBE_UP_POS = (0.6, 0.325, 0.4)

# 禁用坐标轴可视化
VISUALIZE_AXES = False
```

### 3. 查看输出数据

运行完毕后，摄像头图像和力传感器数据保存在：

```
camera_outputs/
├── camera_0/        # 第一个摄像头的PNG图像
├── camera_1/        # 第二个摄像头的PNG图像
├── camera_2/        # 第三个摄像头的PNG图像
├── camera_3/        # 第四个摄像头的PNG图像
├── camera_4/        # 眼在手上摄像头的PNG图像
└── gel/             # 力传感器数据
    ├── 0000.npy     # numpy 格式
    ├── 0000.txt     # 文本格式
    └── ...
```

## 主要功能说明

### 物理模拟
- 时间步长: 1/60 秒 (60Hz)
- 物理引擎: PhysX (可在参数中切换)
- 环境数量: 1 个环境 (可扩展)

### 摄像头系统
- 4 个固定摄像头 (俯视、前视、左视、右视)
- 1 个眼在手上摄像头 (绑定在夹爪)
- 分辨率: 640×480 像素
- 采集频率: 10 Hz (可配置)

### 力传感器
- 位置: 右指尖 (有实际接触点)
- 量程: 6 轴 (Fx, Fy, Fz, Tx, Ty, Tz)
- 校准: 自动零点校准

### 动作规划
- 抓取-放置任务
- 10 个动作阶段
- 总耗时约 17 秒

## 模块说明

### config.py 配置优先级

1. **SimulationConfig 类**: 定义所有参数的默认值
2. **系统函数参数**: 可以覆盖配置值
3. **运行时修改**: 在主循环中可以动态修改

### core 模块特点

- **simulation.py**: 环境创建和初始化
- **assets.py**: 资产加载，支持力传感器配置
- **scene.py**: 场景构建，返回清晰的数据字典

### systems 模块特点

- **camera.py**: 统一的摄像头管理，消除了原代码中的重复
- **sensor.py**: ForceSensorSystem 类管理传感器生命周期
- **planner.py**: MotionPlanner 类管理动作规划

### utils 模块特点

- **math_utils.py**: 数学函数独立，易于单元测试
- **visualization.py**: 所有绘制逻辑集中在一处
- **io_utils.py**: 文件操作工具，易于维护

## 常见修改

### 添加新摄像头

在 `config.py` 的 `CAMERAS` 列表中添加：

```python
CAMERAS = [
    # ... 现有摄像头 ...
    {
        "name": "camera_custom",
        "width": 640,
        "height": 480,
        "pos": (x, y, z),
        "rotation_axis": (1, 0, 0),
        "rotation_angle": -90,
        "rotation_axis2": None,
        "rotation_angle2": 0,
    },
]
```

### 修改动作序列

在 `systems/planner.py` 的 `build_pick_place_plan()` 方法中修改 `plan` 列表。

### 改变物理引擎

在 `main.py` 中运行时指定：

```bash
python main.py --physics_engine=flex    # 使用 Flex 引擎
python main.py --physics_engine=physx   # 使用 PhysX 引擎
```

## 调试技巧

### 打印调试信息

在 `main.py` 的 `run_main_loop()` 中添加打印语句：

```python
print(f"Phase: {plan_state['phase_idx']}, Time: {t:.2f}s")
print(f"Force: {force_magnitude:.4f} N")
```

### 禁用可视化以加速

```python
Config.VISUALIZE_AXES = False
```

### 减少摄像头采集以加速

```python
Config.CAPTURE_FREQUENCY = 1      # 1 Hz 而不是 10 Hz
Config.CAPTURE_DURATION = 1.0     # 1 秒而不是 17 秒
```

## 代码质量指标

- **代码行数**: ~1400 行 (原单文件 ~1470 行)
- **注释覆盖**: 每个函数都有完整的 docstring
- **模块复用**: 消除了 ~150 行重复代码
- **可读性**: 清晰的功能划分和命名

## 扩展建议

### 1. 添加机器学习接口

在 `systems/` 中创建 `learning.py`，实现强化学习集成

### 2. 添加配置文件支持

创建 `load_config_from_yaml()` 函数，支持从 YAML 文件加载参数

### 3. 添加日志系统

使用 Python `logging` 模块替换 `print()` 语句

### 4. 添加单元测试

在根目录创建 `tests/` 文件夹，使用 `pytest` 测试

## 性能优化

- 摄像头采集不阻塞物理模拟 (异步)
- 力传感器数据仅在需要时读取
- 最小化 GPU/CPU 数据传输

---

更多信息请参考 `README.md`
