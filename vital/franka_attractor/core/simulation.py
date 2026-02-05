"""
模拟环境初始化模块
==================
负责创建物理模拟环境、查看器、光照和地面
"""

from isaacgym import gymapi, gymutil
from config import SimulationConfig as Config


def configure_sim_params(args):
    """配置物理模拟参数（Flex和PhysX）
    
    Args:
        args: 命令行参数对象，包含physics_engine、use_gpu、num_threads等
        
    Returns:
        SimParams: 配置好的模拟参数对象
    """
    sim_params = gymapi.SimParams()
    sim_params.dt = Config.DT
    sim_params.substeps = Config.SUBSTEPS
    
    if args.physics_engine == gymapi.SIM_FLEX:
        # 使用 NVIDIA Flex 物理引擎的配置参数
        sim_params.flex.solver_type = Config.FLEX_SOLVER_TYPE
        sim_params.flex.num_outer_iterations = Config.FLEX_NUM_OUTER_ITERATIONS
        sim_params.flex.num_inner_iterations = Config.FLEX_NUM_INNER_ITERATIONS
        sim_params.flex.relaxation = Config.FLEX_RELAXATION
        sim_params.flex.warm_start = Config.FLEX_WARM_START
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # 使用 NVIDIA PhysX 物理引擎的配置参数
        sim_params.physx.solver_type = Config.PHYSX_SOLVER_TYPE
        sim_params.physx.num_position_iterations = Config.PHYSX_NUM_POSITION_ITERATIONS
        sim_params.physx.num_velocity_iterations = Config.PHYSX_NUM_VELOCITY_ITERATIONS
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    
    sim_params.use_gpu_pipeline = Config.USE_GPU_PIPELINE
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")
    
    return sim_params


def initialize_simulation_env(gym, args):
    """初始化模拟对象和查看器
    
    Args:
        gym: Isaac Gym 对象
        args: 命令行参数
        
    Returns:
        tuple: (sim, viewer) - 模拟环境和可视化查看器
    """
    sim_params = configure_sim_params(args)
    
    # 创建物理模拟环境
    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params
    )
    
    if sim is None:
        print("*** Failed to create sim")
        quit()
    
    # 创建查看器（可视化窗口）
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    
    return sim, viewer


def setup_lighting(gym, sim):
    """设置光照参数
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
    """
    gym.set_light_parameters(
        sim, 0,
        gymapi.Vec3(*Config.LIGHT_AMBIENT_COLOR),      # 环境光颜色/强度
        gymapi.Vec3(*Config.LIGHT_DIRECTION_COLOR),    # 方向光颜色/强度
        gymapi.Vec3(*Config.LIGHT_DIRECTION)           # 光源方向（从上往下）
    )


def add_ground_plane(gym, sim):
    """添加地面平面
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
    """
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)
