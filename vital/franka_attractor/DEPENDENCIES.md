# 模块依赖关系和导入说明

## 导入关系图

```
main.py (主程序)
  │
  ├─→ config.py (全局配置)
  │    └─→ SimulationConfig (配置类)
  │
  ├─→ core/simulation.py
  │    ├─→ config.py
  │    └─→ isaacgym
  │
  ├─→ core/assets.py
  │    ├─→ config.py
  │    └─→ isaacgym
  │
  ├─→ core/scene.py
  │    ├─→ config.py
  │    ├─→ isaacgym
  │    └─→ math (math 模块)
  │
  ├─→ systems/camera.py
  │    ├─→ config.py
  │    ├─→ isaacgym
  │    └─→ os, math
  │
  ├─→ systems/sensor.py
  │    ├─→ config.py
  │    ├─→ os
  │    └─→ numpy
  │
  ├─→ systems/gripper.py
  │    ├─→ config.py
  │    └─→ isaacgym
  │
  ├─→ systems/attractor.py
  │    ├─→ isaacgym
  │    ├─→ systems/gripper.py
  │    └─→ utils/math_utils.py
  │
  ├─→ systems/planner.py
  │    ├─→ config.py
  │    └─→ isaacgym
  │
  ├─→ utils/math_utils.py
  │    └─→ isaacgym
  │
  ├─→ utils/io_utils.py
  │    └─→ os
  │
  └─→ utils/visualization.py
       ├─→ config.py
       ├─→ isaacgym
       ├─→ math
       └─→ utils/math_utils.py
```

## 每个模块的依赖列表

### config.py
- **依赖**: 无 (独立配置文件)
- **被依赖者**: 所有其他模块

### core/simulation.py
- **依赖**: isaacgym, config
- **功能**: 环境初始化

### core/assets.py
- **依赖**: isaacgym, config
- **功能**: 资产加载

### core/scene.py
- **依赖**: isaacgym, gymutil, config, math
- **功能**: 场景构建

### systems/camera.py
- **依赖**: isaacgym, gymutil, config, os, math
- **功能**: 摄像头管理

### systems/sensor.py
- **依赖**: os, numpy, config
- **功能**: 力传感器管理
- **核心类**: ForceSensorSystem

### systems/gripper.py
- **依赖**: isaacgym, config
- **功能**: 夹爪控制

### systems/attractor.py
- **依赖**: isaacgym, gymutil, systems/gripper, utils/math_utils
- **功能**: 吸引子控制

### systems/planner.py
- **依赖**: isaacgym, config
- **功能**: 动作规划和采集管理
- **核心类**: MotionPlanner

### utils/math_utils.py
- **依赖**: isaacgym
- **功能**: 数学运算

### utils/io_utils.py
- **依赖**: os
- **功能**: 文件/目录操作

### utils/visualization.py
- **依赖**: isaacgym, gymutil, config, math, utils/math_utils
- **功能**: 可视化绘制

## 数据流向

### 初始化流程

```
main()
  ├─ setup_scene()
  │   ├─ simulation.initialize_simulation_env()
  │   ├─ simulation.setup_lighting()
  │   ├─ simulation.add_ground_plane()
  │   ├─ assets.load_robot_asset()
  │   ├─ assets.load_workbench_asset()
  │   ├─ assets.load_cube_asset()
  │   └─ scene.build_scene() → scene_data (字典)
  │
  └─ initialize_systems(scene_data)
      ├─ scene.initialize_robot_states()
      ├─ camera.setup_camera_output_directory()
      ├─ sensor.setup_gel_output_directory()
      ├─ camera.create_camera_sensor() × N
      ├─ camera.create_eye_in_hand_camera()
      ├─ ForceSensorSystem.__init__()
      ├─ planner.initialize_camera_system() → camera_system
      └─ planner.MotionPlanner.build_pick_place_plan() → motion_plan
```

### 主循环流程

```
run_main_loop(scene_data, systems_data)
  while not viewer_closed:
    ├─ gym.refresh_force_sensor_tensor()
    ├─ ForceSensorSystem.calibrate() [一次]
    ├─ ForceSensorSystem.read_and_calibrate()
    ├─ planner.MotionPlanner [获取当前阶段]
    ├─ attractor.update_pick_and_place()
    │   ├─ math_utils.lerp_vec()
    │   ├─ math_utils.slerp_quat()
    │   ├─ gripper.command_gripper()
    │   └─ visualization (绘制)
    ├─ gym.simulate()
    ├─ planner.should_capture_frame()
    │   ├─ sensor.save_gel_sensor_data()
    │   └─ camera.render_and_save_camera_images()
    └─ visualization (绘制坐标系)
```

## 关键数据结构

### scene_data (来自 scene.build_scene())
```python
{
    'envs': [env_handle, ...],                    # 环境列表
    'robot_handles': [robot_handle, ...],         # 机器人句柄列表
    'attractor_handles': [attractor_handle, ...], # 吸引子句柄列表
    'camera_handles': [camera_handle, ...],       # 摄像头句柄列表
    'dof_props': dof_props_array,                 # 关节属性
    'lower_limits': limits_array,                 # 关节下限
    'upper_limits': limits_array,                 # 关节上限
    'mids': mids_array,                           # 关节中点
    'num_dofs': int,                              # DOF数量
    'body_dict': {'name': idx, ...},              # 刚体字典
    'finger_dof_indices': [left_idx, right_idx],  # 夹爪关节索引
    'cube_handles': [cube_down, cube_up],         # 立方体句柄
    # ... 其他数据
}
```

### plan_state (动作规划状态)
```python
{
    'plan': [                                    # 动作序列
        {
            'name': str,                         # 阶段名称
            'start': (x, y, z),                  # 起始位置
            'goal': (x, y, z),                   # 目标位置
            'start_rot': Quat,                   # 起始旋转
            'goal_rot': Quat,                    # 目标旋转
            'duration': float,                   # 时长（秒）
            'start_finger_width': float,         # 起始夹爪宽度
            'goal_finger_width': float,          # 目标夹爪宽度
        },
        ...
    ],
    'phase_idx': int,                            # 当前阶段索引
    'phase_elapsed': float,                      # 阶段已过时间
    'current_pose': Transform,                   # 当前吸引子姿态
    'running': bool,                             # 是否运行中
    'current_time': float,                       # 当前模拟时间
    'dt': float,                                 # 时间步长
    'start_time': float,                         # 开始时间
}
```

### camera_system (摄像头采集系统)
```python
{
    'output_dir': str,                           # 输出目录
    'capture_frequency': int,                    # 采集频率 (Hz)
    'capture_interval': float,                   # 采集间隔 (秒)
    'capture_duration': float,                   # 采集总时长 (秒)
    'total_frames': int,                         # 总帧数
    'start_time': float,                         # 开始时间
    'next_capture_time': float,                  # 下一次采集时间
    'capture_count': int,                        # 已采集帧数
}
```

## 调用关键函数的位置

| 函数 | 调用位置 | 用途 |
|------|--------|------|
| `scene.build_scene()` | `setup_scene()` | 初始化场景 |
| `camera.create_camera_sensor()` | `initialize_systems()` | 创建固定摄像头 |
| `camera.create_eye_in_hand_camera()` | `initialize_systems()` | 创建手部摄像头 |
| `ForceSensorSystem.calibrate()` | `run_main_loop()` | 传感器零点校准 |
| `attractor.update_pick_and_place()` | `run_main_loop()` | 更新吸引子位置 |
| `planner.should_capture_frame()` | `run_main_loop()` | 判断是否采集 |
| `camera.render_and_save_camera_images()` | `run_main_loop()` | 保存摄像头图像 |
| `sensor.save_gel_sensor_data()` | `run_main_loop()` | 保存传感器数据 |

## 新增函数和类

### 新增类
- `ForceSensorSystem` (in systems/sensor.py): 管理力传感器的生命周期
- `MotionPlanner` (in systems/planner.py): 管理动作规划

### 新增函数
参见各个模块的 docstring

## 配置参数引用位置

| 参数 | 引用位置 |
|-----|---------|
| `DT`, `SUBSTEPS` | core/simulation.py |
| `ASSET_*` | core/assets.py |
| `VISUALIZE_AXES` | utils/visualization.py, core/scene.py |
| `CAMERAS` | main.py (initialize_systems) |
| `ATTRACTOR_*` | core/scene.py |
| `GRIPPER_*` | systems/gripper.py, systems/planner.py |
| `CAPTURE_*` | systems/planner.py |
| `SENSOR_*` | core/assets.py, utils/math_utils.py |
