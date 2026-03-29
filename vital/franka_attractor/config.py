"""
全局配置参数
=============
集中管理所有模拟、摄像头、控制和路径相关的配置参数
"""

class SimulationConfig:
    """模拟环境全局配置"""
    
    # ========== 物理模拟参数 ==========
    DT = 1.0 / 60.0  # 时间步长：60Hz（0.0167秒）
    SUBSTEPS = 2     # 每帧的物理子步数
    USE_GPU_PIPELINE = False  # 是否启用GPU物理管线（如果可用）
    
    # Flex引擎参数
    FLEX_SOLVER_TYPE = 5             # Flex 求解器类型
    FLEX_NUM_OUTER_ITERATIONS = 4    # Flex 外循环迭代次数
    FLEX_NUM_INNER_ITERATIONS = 15   # Flex 内循环迭代次数
    FLEX_RELAXATION = 0.75           # Flex 松弛系数
    FLEX_WARM_START = 0.8            # Flex 热启动系数
    
    # PhysX引擎参数
    PHYSX_SOLVER_TYPE = 1            # PhysX 求解器类型
    PHYSX_NUM_POSITION_ITERATIONS = 4  # PhysX 位置迭代次数
    PHYSX_NUM_VELOCITY_ITERATIONS = 1  # PhysX 速度迭代次数
    
    # ========== 资产路径 ==========
    ASSET_ROOT = "../urdf"  # 资产根目录
    FRANKA_URDF = "franka_description/robots/franka_panda.urdf"  # Franka 机械臂 URDF 路径
    WORKBENCH_URDF = "workbench.urdf"  # 工作台 URDF 路径
    CUBE_DOWN_URDF = "cube_down.urdf"  # 下层物块 URDF 路径
    CUBE_UP_URDF = "cube_up.urdf"  # 上层物块 URDF 路径
    
    # ========== 场景参数 ==========
    NUM_ENVS = 1  # 并行环境数量
    SPACING = 1.0  # 环境间距
    HAND_NAME = "panda_hand"  # 末端执行器刚体名称

    # ========== 限时关闭参数 ==========
    TIMEOUT_ENABLED = False       # 是否启用限时关闭
    TIMEOUT_SECONDS = 20.0        # 运行超时（秒）
    
    # 工作台和立方体初始位置(物块范围x:[0.25,0.60],y:0.325,z:[-0.4,0.4])
    WORKBENCH_POS = (0.8, 0.2, 0.0)  # 工作台初始位置
    CUBE_DOWN_POS = (0.25, 0.325, -0.4)  # 下层物块默认位置
    CUBE_UP_POS = (0.6, 0.325, 0.4)  # 上层物块默认位置
    CUBE_RANGE_X = (0.25, 0.60)  # 随机采样 x 范围
    CUBE_RANGE_Y = 0.325  # 随机采样 y 固定高度
    CUBE_RANGE_Z = (-0.4, 0.4)  # 随机采样 z 范围
    RANDOM_SEED = 0               # 随机种子（修改后可复现不同结果）
    ENABLE_RANDOM_CUBE_POS = True  # 是否用随机种子生成初始物块位置
    CUBE_SIZE = 0.05               # 物块边长
    MIN_CUBE_DISTANCE = 0.12       # 两物块最小距离（中心点距离）
    MAX_SAMPLE_TRIES = 1000        # 随机采样最大尝试次数
    CAPTURE_OUTPUT_DIR = f"/media/neuzz/HLX/zz/DataSet/camera_outputs_{RANDOM_SEED}"      # 摄像头数据输出目录(临时)

    # ========== 吸引子控制参数 ==========
    ATTRACTOR_STIFFNESS = 5e6      # 增加10倍刚度以提高精度
    ATTRACTOR_DAMPING = 5e4        # 增加10倍阻尼以减少振荡
    
    # ========== 夹爪参数 ==========
    GRIPPER_FINGER_OPEN = 0.08     # 两指张开的总宽度
    GRIPPER_FINGER_CLOSED = 0.045  # 两指闭合的总宽度
    GRIPPER_MIN_GAP = 0.001        # 最小间隙，避免硬碰撞
    
    # ========== 光照参数 ==========
    LIGHT_AMBIENT_COLOR = (0.5, 0.5, 0.5)  # 环境光颜色
    LIGHT_DIRECTION_COLOR = (0.8, 0.8, 0.8)  # 方向光颜色
    LIGHT_DIRECTION = (0, -1, 0)  # 方向光方向向量
    
    # ========== 可视化参数 ==========
    VISUALIZE_AXES = False  # 是否绘制坐标轴可视化
    BASE_AXES_SIZE = 2.0  # 世界坐标轴长度
    HAND_AXES_SIZE = 0.15  # 手部坐标轴长度
    FINGERTIP_AXES_SIZE = 0.08  # 指尖坐标轴长度
    CAMERA_AXES_SIZE = 0.1  # 相机坐标轴长度
    ATTRACTOR_SPHERE_SIZE = 0.03  # 吸引子球体可视化半径
    
    # ========== 摄像头配置 ==========
    CAMERAS = [
        {
            "name": "camera_top",
            "width": 640,
            "height": 480,
            "pos": (0.4, 1.0, 0.0),
            "rotation_axis": (1, 0, 0),
            "rotation_angle": -90,
            "rotation_axis2": None,
            "rotation_angle2": 0,
        },
        {
            "name": "camera_front",
            "width": 640,
            "height": 480,
            "pos": (0.8, 0.8, 0.0),
            "rotation_axis": (0, 1, 0),
            "rotation_angle": 90,
            "rotation_axis2": (0, 0, 1),
            "rotation_angle2": 45,
        },
        {
            "name": "camera_side_left",
            "width": 640,
            "height": 480,
            "pos": (0.4, 0.8, 0.6),
            "rotation_axis": (1, 0, 0),
            "rotation_angle": -45,
            "rotation_axis2": None,
            "rotation_angle2": 0,
        },
        {
            "name": "camera_side_right",
            "width": 640,
            "height": 480,
            "pos": (0.4, 0.8, -0.6),
            "rotation_axis": (0, 1, 0),
            "rotation_angle": 180,
            "rotation_axis2": (1, 0, 0),
            "rotation_angle2": 45,
        },
    ]
    
    # 眼在手上摄像头配置
    HAND_CAMERA_OFFSET = (0.05, 0.0, 0.0)  # 手眼相机相对手坐标偏移
    HAND_CAMERA_AXIS_PRIMARY = (1, 0, 0)  # 手眼相机主旋转轴
    HAND_CAMERA_ANGLE_PRIMARY = 180  # 手眼相机主旋转角度
    HAND_CAMERA_AXIS_SECONDARY = (0, 0, 1)  # 手眼相机次旋转轴
    HAND_CAMERA_ANGLE_SECONDARY = 90  # 手眼相机次旋转角度
    HAND_CAMERA_WIDTH = 640  # 手眼相机分辨率宽
    HAND_CAMERA_HEIGHT = 480  # 手眼相机分辨率高
    
    # ========== 摄像头采集参数 ==========
    CAPTURE_FREQUENCY = 10         # 每秒10帧
    CAPTURE_DURATION = 18.0        # 总共采集18秒
    CAPTURE_START_TIME = 1.5       # 采集开始时间

    # ========== 在线策略控制参数 ==========
    # 控制模式："planner" 使用规则规划器，"policy" 使用模型在线推理
    CONTROL_MODE = "planner"  # 可切换为 "policy" 进行在线推理控制

    # 在线推理模型文件路径
    POLICY_CKPT = "./model/policy_best.ckpt"  # 策略模型权重文件
    POLICY_ARGS = "./model/args.json"  # 策略模型参数文件
    POLICY_STATS = "./model/dataset_stats.pkl"  # 训练数据统计量文件

    # 是否启用影子模式（仅推理不控制，便于联调）
    POLICY_SHADOW_MODE = False

    # 推理频率（Hz）
    POLICY_RATE_HZ = 10

    # 策略异常时是否回退到 planner
    POLICY_FALLBACK_TO_PLANNER = True

    # 安全约束：单步最大位置增量（米）
    MAX_DELTA_XYZ = 0.01

    # 安全约束：工作空间边界（米）
    #物块范围x:[0.25,0.60],y:0.325,z:[-0.4,0.4]
    POLICY_WORKSPACE_X = (0.20, 0.80)
    POLICY_WORKSPACE_Y = (0.20, 1.20)
    POLICY_WORKSPACE_Z = (-0.60, 0.60)

    # 安全约束：夹爪开度范围（与动作第4维对应）
    GRIPPER_ACTION_MIN = 0.0
    GRIPPER_ACTION_MAX = 0.08

    # 安全约束：输出平滑系数（EMA），0表示不平滑
    POLICY_ACTION_EMA_ALPHA = 0.0

    # ========== 动作规划参数 ==========
    MOTION_PLAN_HOVER_OFFSET = 0.2      # 悬停在立方体上方20cm
    MOTION_PLAN_GRASP_OFFSET = 0.1      # 抓取时夹爪在立方体上方10cm
    MOTION_PLAN_RELEASE_OFFSET = 0.15   # 放置时留15cm高度
    
    # ========== 打印和日志参数 ==========
    PRINT_INTERVAL = 0.2           # 每0.2秒打印一次
    LOG_CAPTURE_INTERVAL = 10      # 每采集10帧打印一次
    # 触觉绘图开关：False 时不创建 matplotlib 窗口（更省资源）
    ENABLE_TACTILE_PLOT = False
    # True: 左右指尖在同一个窗口对比；False: 左右各一个窗口
    TACTILE_COMBINED_PLOT = False
    # 触觉数据导出：保存为 converter 兼容的 gel/<frame>.npy (6,)
    ENABLE_GEL_NPY_EXPORT = True
    GEL_OUTPUT_SUBDIR = "gel"
    # gel 向量来源："right" | "left" | "average"
    GEL_VECTOR_SOURCE = "right"
    # 是否按真实时间同步渲染；设为 False 可提高离线采样吞吐（会“快进”运行）
    SYNC_TO_REALTIME = False
    
    # ========== 资产加载选项 ==========
    ASSET_FIX_BASE_LINK = True  # 是否固定基座链接
    ASSET_FLIP_VISUAL = True  # 是否翻转可视网格
    ASSET_ARMATURE = 0.01  # 资产等效转动惯量系数
    ASSET_DENSITY = 100.0  # 资产默认密度
    
    # ========== 接触属性 ==========
    CONTACT_FRICTION = 2.0  # 接触摩擦系数
    CONTACT_RESTITUTION = 0.0  # 接触恢复系数（弹性）
    CONTACT_OFFSET = 0.001  # 接触偏移
    REST_OFFSET = 0.0  # 静止偏移
    
    # ========== 关节控制参数 ==========
    DOF_STIFFNESS = 10.0           # 降低以让attractor主导控制
    DOF_DAMPING = 10.0             # 降低以减少阻尼干扰
    GRIPPER_STIFFNESS = 1e10  # 夹爪刚度
    GRIPPER_DAMPING = 1.0  # 夹爪阻尼
