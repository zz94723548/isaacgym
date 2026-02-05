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
    USE_GPU_PIPELINE = False
    
    # Flex引擎参数
    FLEX_SOLVER_TYPE = 5
    FLEX_NUM_OUTER_ITERATIONS = 4
    FLEX_NUM_INNER_ITERATIONS = 15
    FLEX_RELAXATION = 0.75
    FLEX_WARM_START = 0.8
    
    # PhysX引擎参数
    PHYSX_SOLVER_TYPE = 1
    PHYSX_NUM_POSITION_ITERATIONS = 4
    PHYSX_NUM_VELOCITY_ITERATIONS = 1
    
    # ========== 资产路径 ==========
    ASSET_ROOT = "../urdf"
    FRANKA_URDF = "franka_description/robots/franka_panda.urdf"
    WORKBENCH_URDF = "workbench.urdf"
    CUBE_DOWN_URDF = "cube_down.urdf"
    CUBE_UP_URDF = "cube_up.urdf"
    
    # ========== 场景参数 ==========
    NUM_ENVS = 1
    SPACING = 1.0
    HAND_NAME = "panda_hand"

    # ========== 限时关闭参数 ==========
    TIMEOUT_ENABLED = True       # 是否启用限时关闭
    TIMEOUT_SECONDS = 20.0        # 运行超时（秒）
    
    # 工作台和立方体初始位置(物块范围x:[0.25,0.60],y:0.325,z:[-0.4,0.4])
    WORKBENCH_POS = (0.8, 0.2, 0.0)
    CUBE_DOWN_POS = (0.25, 0.325, -0.4)
    CUBE_UP_POS = (0.6, 0.325, 0.4)
    CUBE_RANGE_X = (0.25, 0.60)
    CUBE_RANGE_Y = 0.325
    CUBE_RANGE_Z = (-0.4, 0.4)
    RANDOM_SEED = 100               # 随机种子（修改后可复现不同结果）
    ENABLE_RANDOM_CUBE_POS = True  # 是否用随机种子生成初始物块位置
    CUBE_SIZE = 0.05               # 物块边长
    MIN_CUBE_DISTANCE = 0.12       # 两物块最小距离（中心点距离）
    MAX_SAMPLE_TRIES = 1000        # 随机采样最大尝试次数
    CAPTURE_OUTPUT_DIR = f"/media/neuzz/HLX/zz/camera_outputs_{RANDOM_SEED}"      # 摄像头数据输出目录(临时)

    # ========== 吸引子控制参数 ==========
    ATTRACTOR_STIFFNESS = 5e6      # 增加10倍刚度以提高精度
    ATTRACTOR_DAMPING = 5e4        # 增加10倍阻尼以减少振荡
    
    # ========== 夹爪参数 ==========
    GRIPPER_FINGER_OPEN = 0.08     # 两指张开的总宽度
    GRIPPER_FINGER_CLOSED = 0.045  # 两指闭合的总宽度
    GRIPPER_MIN_GAP = 0.001        # 最小间隙，避免硬碰撞
    
    # ========== 光照参数 ==========
    LIGHT_AMBIENT_COLOR = (0.5, 0.5, 0.5)
    LIGHT_DIRECTION_COLOR = (0.8, 0.8, 0.8)
    LIGHT_DIRECTION = (0, -1, 0)
    
    # ========== 可视化参数 ==========
    VISUALIZE_AXES = False
    BASE_AXES_SIZE = 2.0
    HAND_AXES_SIZE = 0.15
    FINGERTIP_AXES_SIZE = 0.08
    CAMERA_AXES_SIZE = 0.1
    ATTRACTOR_SPHERE_SIZE = 0.03
    
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
    HAND_CAMERA_OFFSET = (0.05, 0.0, 0.0)
    HAND_CAMERA_AXIS_PRIMARY = (1, 0, 0)
    HAND_CAMERA_ANGLE_PRIMARY = 180
    HAND_CAMERA_AXIS_SECONDARY = (0, 0, 1)
    HAND_CAMERA_ANGLE_SECONDARY = 90
    HAND_CAMERA_WIDTH = 640
    HAND_CAMERA_HEIGHT = 480
    
    # ========== 摄像头采集参数 ==========
    CAPTURE_FREQUENCY = 10         # 每秒10帧
    CAPTURE_DURATION = 18.0        # 总共采集17秒
    CAPTURE_START_TIME = 1.5       # 采集开始时间
    # CAPTURE_OUTPUT_DIR = "/media/neuzz/HLX/camera_outputs"  # 摄像头数据输出目录

    
    # ========== 传感器校准参数 ==========
    SENSOR_CALIBRATION_TIME = 1.0  # 在任务开始前1秒进行校准
    SENSOR_FINGERTIP_OFFSET_Z = 0.045  # 沿局部Z轴的偏移距离(米)
    
    # ========== 动作规划参数 ==========
    MOTION_PLAN_HOVER_OFFSET = 0.2      # 悬停在立方体上方20cm
    MOTION_PLAN_GRASP_OFFSET = 0.1      # 抓取时夹爪在立方体上方10cm
    MOTION_PLAN_RELEASE_OFFSET = 0.15   # 放置时留15cm高度
    
    # ========== 打印和日志参数 ==========
    PRINT_INTERVAL = 0.2           # 每0.2秒打印一次
    LOG_CAPTURE_INTERVAL = 10      # 每采集10帧打印一次
    
    # ========== 资产加载选项 ==========
    ASSET_FIX_BASE_LINK = True
    ASSET_FLIP_VISUAL = True
    ASSET_ARMATURE = 0.01
    ASSET_DENSITY = 100.0
    
    # ========== 接触属性 ==========
    CONTACT_FRICTION = 2.0
    CONTACT_RESTITUTION = 0.0
    CONTACT_OFFSET = 0.03
    REST_OFFSET = 0.0
    
    # ========== 关节控制参数 ==========
    DOF_STIFFNESS = 10.0           # 降低以让attractor主导控制
    DOF_DAMPING = 10.0             # 降低以减少阻尼干扰
    GRIPPER_STIFFNESS = 1e10
    GRIPPER_DAMPING = 1.0
