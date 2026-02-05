"""
场景构建模块
===========
负责在环境中创建机器人、吸引子、物体和配置场景
"""

import math
from isaacgym import gymapi, gymutil
from config import SimulationConfig as Config


def create_attractor_properties(stiffness=None, damping=None):
    """创建吸引子属性
    
    Args:
        stiffness: 刚度系数，如果为None使用配置值
        damping: 阻尼系数，如果为None使用配置值
        
    Returns:
        AttractorProperties: 吸引子属性对象
    """
    if stiffness is None:
        stiffness = Config.ATTRACTOR_STIFFNESS
    if damping is None:
        damping = Config.ATTRACTOR_DAMPING
        
    attractor_props = gymapi.AttractorProperties()
    attractor_props.stiffness = stiffness
    attractor_props.damping = damping
    attractor_props.axes = gymapi.AXIS_ALL
    return attractor_props


def create_robot_pose(pos_x=0.0, pos_y=0.0, pos_z=0.0,
                      quat_x=-0.707107, quat_y=0.0, quat_z=0.0, quat_w=0.707107):
    """创建机器人初始姿态
    
    Args:
        pos_x, pos_y, pos_z: 位置坐标
        quat_x, quat_y, quat_z, quat_w: 旋转四元数
        
    Returns:
        Transform: 机器人初始姿态
    """
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(pos_x, pos_y, pos_z)
    pose.r = gymapi.Quat(quat_x, quat_y, quat_z, quat_w)
    return pose


def create_visualization_geometries():
    """创建可视化几何体（吸引子）
    
    Returns:
        tuple: (axes_geom, sphere_geom) - 坐标轴和球体几何体
    """
    axes_geom = gymutil.AxesGeometry(0.1)
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(
        Config.ATTRACTOR_SPHERE_SIZE, 12, 12, sphere_pose, color=(1, 0, 0)
    )
    return axes_geom, sphere_geom


def create_robot_actor(gym, env, robot_asset, pose, env_id, hand_name="panda_hand"):
    """在环境中创建机器人演员
    
    Args:
        gym: Isaac Gym 对象
        env: 环境对象
        robot_asset: 加载的机器人资产
        pose: 机器人初始姿态
        env_id: 环境ID
        hand_name: 末端执行器（手）的名称
        
    Returns:
        tuple: (robot_handle, body_dict, dof_dict, props, hand_handle, left_finger_handle, right_finger_handle)
    """
    robot_handle = gym.create_actor(env, robot_asset, pose, "franka", env_id, 0)
    body_dict = gym.get_actor_rigid_body_dict(env, robot_handle)
    dof_dict = gym.get_actor_dof_dict(env, robot_handle)
    props = gym.get_actor_rigid_body_states(env, robot_handle, gymapi.STATE_POS)
    hand_handle = gym.find_actor_rigid_body_handle(env, robot_handle, hand_name)
    
    left_finger_handle = gym.find_actor_rigid_body_handle(env, robot_handle, "panda_leftfinger")
    right_finger_handle = gym.find_actor_rigid_body_handle(env, robot_handle, "panda_rightfinger")
    
    return robot_handle, body_dict, dof_dict, props, hand_handle, left_finger_handle, right_finger_handle


def setup_robot_dof_properties(dof_props):
    """设置机器人关节属性的具体参数
    
    Args:
        dof_props: 关节属性数组
        
    Returns:
        ndarray: 更新后的关节属性
    """
    # 设置所有关节的基础刚度和阻尼
    dof_props['stiffness'].fill(Config.DOF_STIFFNESS)
    dof_props['damping'].fill(Config.DOF_DAMPING)
    
    # 前两个关节使用位置驱动模式
    dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS
    
    # 夹爪：两个指头都驱动
    dof_props["driveMode"][7] = gymapi.DOF_MODE_POS
    dof_props["driveMode"][8] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][7] = Config.GRIPPER_STIFFNESS
    dof_props['damping'][7] = Config.GRIPPER_DAMPING
    dof_props['stiffness'][8] = Config.GRIPPER_STIFFNESS
    dof_props['damping'][8] = Config.GRIPPER_DAMPING
    
    return dof_props


def initialize_robot_states(gym, envs, robot_handles, joint_mids, num_dofs):
    """初始化机器人关节状态
    
    Args:
        gym: Isaac Gym 对象
        envs: 环境列表
        robot_handles: 机器人句柄列表
        joint_mids: 关节中点位置
        num_dofs: 关节自由度数量
    """
    for i in range(len(envs)):
        dof_states = gym.get_actor_dof_states(envs[i], robot_handles[i], gymapi.STATE_NONE)
        for j in range(num_dofs):
            dof_states['pos'][j] = joint_mids[j]
        gym.set_actor_dof_states(envs[i], robot_handles[i], dof_states, gymapi.STATE_POS)


def configure_actor_shape_properties(gym, env, actor_handle,
                                     friction=None, restitution=None,
                                     contact_offset=None, rest_offset=None):
    """配置演员的刚体形状属性
    
    Args:
        gym: Isaac Gym 对象
        env: 环境对象
        actor_handle: 演员句柄
        friction: 摩擦系数，如果为None使用配置值
        restitution: 恢复系数，如果为None使用配置值
        contact_offset: 接触偏移，如果为None使用配置值
        rest_offset: 静止偏移，如果为None使用配置值
    """
    if friction is None:
        friction = Config.CONTACT_FRICTION
    if restitution is None:
        restitution = Config.CONTACT_RESTITUTION
    if contact_offset is None:
        contact_offset = Config.CONTACT_OFFSET
    if rest_offset is None:
        rest_offset = Config.REST_OFFSET
        
    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for p in shape_props:
        p.friction = friction
        p.restitution = restitution
        p.contact_offset = contact_offset
        p.rest_offset = rest_offset
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)


def create_workbench_actor(gym, env, workbench_asset, position, env_id=0):
    """在环境中创建工作台演员
    
    Args:
        gym: Isaac Gym 对象
        env: 环境对象
        workbench_asset: 工作台资产
        position: 工作台位置 (x, y, z)
        env_id: 环境ID
        
    Returns:
        Actor: 工作台演员句柄
    """
    workbench_pose = gymapi.Transform()
    workbench_pose.p = gymapi.Vec3(position[0], position[1], position[2])
    workbench_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    workbench_handle = gym.create_actor(env, workbench_asset, workbench_pose, "workbench", env_id, 0)
    return workbench_handle


def create_cube_actor(gym, env, cube_asset, position, env_id=0):
    """创建立方体演员
    
    Args:
        gym: Isaac Gym 对象
        env: 环境对象
        cube_asset: 立方体资产
        position: 立方体位置 (x, y, z)
        env_id: 环境ID
        
    Returns:
        Actor: 立方体演员句柄
    """
    cube_pose = gymapi.Transform()
    cube_pose.p = gymapi.Vec3(position[0], position[1], position[2])
    cube_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    cube_handle = gym.create_actor(env, cube_asset, cube_pose, "cube", env_id, 0)
    return cube_handle


def build_scene(gym, sim, viewer, robot_asset, workbench_asset, cube_down_asset, cube_up_asset,
                num_envs=1, spacing=1.0, hand_name="panda_hand",
                attractor_stiffness=None, attractor_damping=None):
    """构建完整的模拟场景
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
        viewer: 查看器对象
        robot_asset: 机器人资产
        workbench_asset: 工作台资产
        cube_down_asset: 下方立方体资产
        cube_up_asset: 上方立方体资产
        num_envs: 环境数量
        spacing: 环境间距
        hand_name: 末端执行器名称
        attractor_stiffness: 吸引子刚度
        attractor_damping: 吸引子阻尼
        
    Returns:
        dict: 场景数据字典，包含所有创建的对象
    """
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))
    
    envs = []
    robot_handles = []
    attractor_handles = []
    camera_handles = []
    
    attractor_props = create_attractor_properties(attractor_stiffness, attractor_damping)
    robot_pose = create_robot_pose()
    axes_geom, sphere_geom = create_visualization_geometries()
    
    print(f"Creating {num_envs} environments")
    
    # 创建第一个环境
    temp_env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    
    # 为第一个环境创建机器人
    robot_handle, body_dict, dof_dict, props, hand_handle, left_finger_handle, right_finger_handle = create_robot_actor(
        gym, temp_env, robot_asset, robot_pose, 0, hand_name
    )
    
    # 获取机器人DOF属性并配置
    dof_props = gym.get_actor_dof_properties(temp_env, robot_handle)
    dof_props = setup_robot_dof_properties(dof_props)
    
    # 获取关节信息
    lower_limits = dof_props['lower'].copy()
    upper_limits = dof_props['upper'].copy()
    mids = 0.5 * (upper_limits + lower_limits)
    num_dofs = len(dof_props)
    
    # 记录夹爪 DOF 索引
    finger_dof_indices = [
        dof_dict.get("panda_finger_joint1"),
        dof_dict.get("panda_finger_joint2"),
    ]
    
    if None in finger_dof_indices:
        raise ValueError("Finger DOF indices not found in Franka asset")
    
    # 设置吸引子初始位置
    attractor_props.target = props['pose'][:][body_dict[hand_name]]
    attractor_props.target.p.y = attractor_props.target.p.y - 0.1
    attractor_props.rigid_handle = hand_handle
    
    # 绘制吸引子可视化
    gymutil.draw_lines(axes_geom, gym, viewer, temp_env, attractor_props.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, temp_env, attractor_props.target)
    
    # 创建工作台
    workbench_handle = create_workbench_actor(gym, temp_env, workbench_asset,
                                              Config.WORKBENCH_POS, env_id=0)
    configure_actor_shape_properties(gym, temp_env, workbench_handle)
    
    # 创建立方体
    cube_down_handle = create_cube_actor(gym, temp_env, cube_down_asset,
                                        Config.CUBE_DOWN_POS, env_id=0)
    cube_up_handle = create_cube_actor(gym, temp_env, cube_up_asset,
                                      Config.CUBE_UP_POS, env_id=0)
    
    configure_actor_shape_properties(gym, temp_env, cube_down_handle)
    configure_actor_shape_properties(gym, temp_env, cube_up_handle)
    
    # 绘制机器人基座坐标系
    if Config.VISUALIZE_AXES:
        base_axes_geom = gymutil.AxesGeometry(Config.BASE_AXES_SIZE)
        base_transform = gymapi.Transform()
        base_transform.p = gymapi.Vec3(0.0, 0.0, 0.0)
        base_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        gymutil.draw_lines(base_axes_geom, gym, viewer, temp_env, base_transform)
    
    # 创建吸引子
    attractor_handle = gym.create_rigid_body_attractor(temp_env, attractor_props)
    
    # 创建摄像头几何体
    camera_axes_geom = gymutil.AxesGeometry(Config.CAMERA_AXES_SIZE)
    
    envs.append(temp_env)
    robot_handles.append(robot_handle)
    attractor_handles.append(attractor_handle)
    cube_handles = [cube_down_handle, cube_up_handle]
    
    # 为所有环境应用DOF属性
    for i in range(num_envs):
        gym.set_actor_dof_properties(envs[i], robot_handles[i], dof_props)
    
    return {
        'envs': envs,
        'robot_handles': robot_handles,
        'attractor_handles': attractor_handles,
        'camera_handles': camera_handles,
        'dof_props': dof_props,
        'lower_limits': lower_limits,
        'upper_limits': upper_limits,
        'mids': mids,
        'num_dofs': num_dofs,
        'axes_geom': axes_geom,
        'sphere_geom': sphere_geom,
        'camera_axes_geom': camera_axes_geom,
        'cube_handles': cube_handles,
        'body_dict': body_dict,
        'hand_name': hand_name,
        'finger_dof_indices': finger_dof_indices,
        'cube_down_pos': Config.CUBE_DOWN_POS,
        'cube_up_pos': Config.CUBE_UP_POS,
        'initial_hand_pose': attractor_props.target,
        'left_finger_handle': left_finger_handle,
        'right_finger_handle': right_finger_handle,
    }
