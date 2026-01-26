"""
版权所有 (c) 2020, NVIDIA CORPORATION. 保留所有权利。

NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
及其任何修改的所有知识产权和专有权。未经 NVIDIA CORPORATION
的明确许可协议，严禁以任何形式使用、复制、披露或
分发本软件及相关文档。


Franka 吸引子示例
-----------------
这个脚本演示了如何使用 Isaac Gym 进行 Franka Panda 机器人的位置控制。
机器人通过一个虚拟吸引子点进行运动控制，吸引子点的位置根据时间函数实时更新。
"""

import math
import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# ===============================
# 坐标系可视化开关
# ===============================
VISUALIZE_AXES = False  # 设置为 True 显示所有坐标系，False 关闭所有坐标系绘制


# ===============================
# 第一部分：模拟环境初始化
# ===============================

#　配置物理模拟参数（Flex和PhysX）
def configure_sim_params(args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0  # 时间步长：60Hz（0.0167秒）
    sim_params.substeps = 2     # 每帧的物理子步数    
    if args.physics_engine == gymapi.SIM_FLEX:
        # 使用 NVIDIA Flex 物理引擎的配置参数
        sim_params.flex.solver_type = 5            # 求解器类型
        sim_params.flex.num_outer_iterations = 4   # 外层迭代次数
        sim_params.flex.num_inner_iterations = 15  # 内层迭代次数
        sim_params.flex.relaxation = 0.75          # 松弛因子
        sim_params.flex.warm_start = 0.8           # 热启动参数
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # 使用 NVIDIA PhysX 物理引擎的配置参数
        sim_params.physx.solver_type = 1                    # 求解器类型
        sim_params.physx.num_position_iterations = 4        # 位置迭代次数
        sim_params.physx.num_velocity_iterations = 1        # 速度迭代次数
        sim_params.physx.num_threads = args.num_threads     # 使用的线程数
        sim_params.physx.use_gpu = args.use_gpu             # 是否使用 GPU 加速
    
    sim_params.use_gpu_pipeline = False  # 禁用 GPU 渲染管道
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")
    
    return sim_params

# 初始化模拟对象和查看器
def initialize_simulation_env(gym, args):
    # 配置模拟参数
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

#　设置光照参数
def setup_lighting(gym, sim):
    gym.set_light_parameters(
        sim, 0,
        gymapi.Vec3(0.5, 0.5, 0.5),   # 环境光颜色/强度
        gymapi.Vec3(0.8, 0.8, 0.8),   # 方向光颜色/强度
        gymapi.Vec3(0, -1, 0)         # 光源方向（从上往下）
    )

# 添加地面平面
def add_ground_plane(gym, sim):
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)


# ===============================
# 第二部分：资产加载
# ===============================

#　创建资产加载选项
def create_asset_options(fix_base=True, flip_visual=True, armature=0.01):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fix_base     # 固定基座链接
    asset_options.flip_visual_attachments = flip_visual     # 翻转视觉附着物
    asset_options.armature = armature       # 装甲参数（惯性）
    return asset_options

# 加载机器人资产
def load_robot_asset(gym, sim, asset_root, asset_file, asset_options):
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    
    # asset_root: 资产文件根目录
    # asset_file: 机器人模型文件相对路径
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    if robot_asset is None:
        print("*** Failed to load asset: %s" % asset_file)
        quit()
    
    return robot_asset

# 加载工作台资产
def load_workbench_asset(gym, sim, asset_root, asset_file):
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    
    workbench_asset_options = gymapi.AssetOptions()
    workbench_asset_options.fix_base_link = True
    workbench_asset_options.density = 100.0         # 设置密度
    workbench_asset = gym.load_asset(sim, asset_root, asset_file, workbench_asset_options)
    
    if workbench_asset is None:
        print("*** Failed to load workbench asset: %s" % asset_file)
        quit()
    
    return workbench_asset

# 加载立方体资产
def load_cube_asset(gym, sim, asset_root, asset_file):
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    
    cube_asset_options = gymapi.AssetOptions()
    cube_asset_options.fix_base_link = False
    cube_asset_options.density = 100.0         # 设置密度
    cube_asset = gym.load_asset(sim, asset_root, asset_file, cube_asset_options)
    
    if cube_asset is None:
        print("*** Failed to load cube asset: %s" % asset_file)
        quit()
    
    return cube_asset


# ===============================
# 第三部分：场景构建
# ===============================

# 创建吸引子属性
def create_attractor_properties(stiffness=5e5, damping=5e3):
    attractor_props = gymapi.AttractorProperties()
    attractor_props.stiffness = stiffness       # 刚度系数
    attractor_props.damping = damping           # 阻尼系数
    attractor_props.axes = gymapi.AXIS_ALL      # 作用于所有轴（X、Y、Z和平移/旋转）
    return attractor_props

# 创建机器人初始姿态
def create_robot_pose(pos_x=0.0, pos_y=0.0, pos_z=0.0, 
                      quat_x=-0.707107, quat_y=0.0, quat_z=0.0, quat_w=0.707107):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(pos_x, pos_y, pos_z)   # 位置坐标
    pose.r = gymapi.Quat(quat_x, quat_y, quat_z, quat_w)  # 旋转（四元数）
    return pose

# 创建可视化几何体（吸引子）
def create_visualization_geometries():
    axes_geom = gymutil.AxesGeometry(0.1)        # 坐标轴几何体，长度0.1m
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)    # 球体旋转四元数
    sphere_pose = gymapi.Transform(r=sphere_rot)    # 球体变换
    sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))   # 红色球体,半径0.03m
    
    return axes_geom, sphere_geom

# 在环境中创建工作台演员
def create_workbench_actor(gym, env, workbench_asset, position, scale=1.0, env_id=0):
    workbench_pose = gymapi.Transform()
    workbench_pose.p = gymapi.Vec3(position[0], position[1], position[2])
    workbench_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    workbench_handle = gym.create_actor(env, workbench_asset, workbench_pose, "workbench", env_id, 0)
    return workbench_handle

# 创建立方体演员
def create_cube_actor(gym, env, cube_asset, position, scale=1.0, env_id=0):
    cube_pose = gymapi.Transform()
    cube_pose.p = gymapi.Vec3(position[0], position[1], position[2])
    cube_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    cube_handle = gym.create_actor(env, cube_asset, cube_pose, "cube", env_id, 0)
    return cube_handle

# 在环境中创建机器人演员
def create_robot_actor(gym, env, robot_asset, pose, env_id, hand_name="panda_hand"):
    robot_handle = gym.create_actor(env, robot_asset, pose, "franka", env_id, 0)        # 创建机器人演员
    body_dict = gym.get_actor_rigid_body_dict(env, robot_handle)            # 获取刚体字典
    dof_dict = gym.get_actor_dof_dict(env, robot_handle)                    # 获取DOF字典
    props = gym.get_actor_rigid_body_states(env, robot_handle, gymapi.STATE_POS)    # 获取刚体状态
    hand_handle = gym.find_actor_rigid_body_handle(env, robot_handle, hand_name)    # 获取末端执行器句柄
    
    return robot_handle, body_dict, dof_dict, props, hand_handle

# 配置机器人关节属性
def configure_robot_dof_properties(gym, env, robot_handle, franka_dof_props):
    gym.set_actor_dof_properties(env, robot_handle, franka_dof_props)

# 设置机器人关节属性的具体参数
def setup_robot_dof_properties(dof_props):
    # 设置所有关节的基础刚度和阻尼
    dof_props['stiffness'].fill(1000.0)
    dof_props['damping'].fill(1000.0)
    
    # 前两个关节使用位置驱动模式
    dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS
    
    # 夹爪：两个指头都驱动，在代码中手动同步mimic（Isaac Gym不自动处理URDF mimic）
    dof_props["driveMode"][7] = gymapi.DOF_MODE_POS
    dof_props["driveMode"][8] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][7] = 1e10
    dof_props['damping'][7] = 1.0
    dof_props['stiffness'][8] = 1e10
    dof_props['damping'][8] = 1.0
    
    return dof_props

# 初始化机器人关节状态
def initialize_robot_states(gym, envs, robot_handles, joint_mids, num_dofs):

    for i in range(len(envs)):
        # 获取机器人的当前DOF状态
        dof_states = gym.get_actor_dof_states(envs[i], robot_handles[i], gymapi.STATE_NONE)
        # 将所有关节设置为中点位置
        for j in range(num_dofs):
            dof_states['pos'][j] = joint_mids[j]
        # 应用DOF状态
        gym.set_actor_dof_states(envs[i], robot_handles[i], dof_states, gymapi.STATE_POS)

# 配置演员的刚体形状属性
def configure_actor_shape_properties(gym, env, actor_handle, 
                                     friction=2.0, restitution=0.0,
                                     contact_offset=0.02, rest_offset=0.0):
    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for p in shape_props:
        p.friction = friction       # 设置摩擦系数
        p.restitution = restitution     # 设置恢复系数（弹性系数）
        p.contact_offset = contact_offset       # 设置接触偏移
        p.rest_offset = rest_offset       # 设置静止偏移
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)

# 在环境中创建摄像头传感器
def create_camera_sensor(
    gym,
    env,
    width=640,
    height=480,
    pos=(0.0, 1.5, 0.0),
    rotation_axis=(1, 0, 0),
    rotation_angle=-90,
    rotation_axis2=None,
    rotation_angle2=0,
):
    # 创建摄像头属性
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    
    # 创建摄像头传感器
    camera_handle = gym.create_camera_sensor(env, camera_props)
    
    # 设置摄像头变换（位置和旋转）
    primary_rot = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(*rotation_axis),
        np.deg2rad(rotation_angle),
    )

    # 支持二次旋转（例如先绕X再绕Y）
    if rotation_axis2 is not None:
        secondary_rot = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(*rotation_axis2),
            np.deg2rad(rotation_angle2),
        )
        camera_rot = secondary_rot * primary_rot  # 先primary后secondary
    else:
        camera_rot = primary_rot
    camera_transform = gymapi.Transform(
        p=gymapi.Vec3(*pos),
        r=camera_rot
    )
    
    gym.set_camera_transform(camera_handle, env, camera_transform)
    
    return camera_handle, camera_transform

#　计算摄像头变换
def compute_camera_transform(pos, axis_primary, angle_primary, axis_secondary=None, angle_secondary=0):
    primary_rot = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(axis_primary[0], axis_primary[1], axis_primary[2]),
        np.deg2rad(angle_primary),
    )
    if axis_secondary is not None:
        secondary_rot = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(axis_secondary[0], axis_secondary[1], axis_secondary[2]),
            np.deg2rad(angle_secondary),
        )
        rot = secondary_rot * primary_rot
    else:
        rot = primary_rot
    return gymapi.Transform(p=gymapi.Vec3(pos[0], pos[1], pos[2]), r=rot)


# 创建并绑定在夹爪上的摄像头
def create_eye_in_hand_camera(
    gym,
    env,
    hand_handle,
    width,
    height,
    offset=(0.0, 0.0, 0.1),
    rotation_axis_primary=(1, 0, 0),
    rotation_angle_primary=-90,
    rotation_axis_secondary=None,
    rotation_angle_secondary=0,
):
    # 构建摄像头属性
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height

    camera_handle = gym.create_camera_sensor(env, camera_props)

    # 计算安装在手坐标系下的相对变换
    mount_tf = compute_camera_transform(
        offset,
        rotation_axis_primary,
        rotation_angle_primary,
        rotation_axis_secondary,
        rotation_angle_secondary,
    )

    # 绑定摄像头到夹爪（跟随手的姿态）
    gym.attach_camera_to_body(
        camera_handle,
        env,
        hand_handle,
        mount_tf,
        gymapi.FOLLOW_TRANSFORM,
    )

    return camera_handle, mount_tf

# 绘制单个摄像头的坐标系
def draw_camera_axes_single(
    gym,
    viewer,
    env,
    camera_axes_geom,
    pos,
    rotation_axis_primary,
    rotation_angle_primary,
    rotation_axis_secondary=None,
    rotation_angle_secondary=0,
):
    cam_tf = compute_camera_transform(
        pos,
        rotation_axis_primary,
        rotation_angle_primary,
        rotation_axis_secondary,
        rotation_angle_secondary,
    )
    gymutil.draw_lines(camera_axes_geom, gym, viewer, env, cam_tf)

#　摄像头参数
camera_params = {
    "axis_length": 0.1,
    "width": 640,
    "height": 480,
    "pos1": (0.4, 1.0, 0.0),
    "rotation_axis1": (1, 0, 0),
    "rotation_angle1": -90,
    "pos2": (0.8, 0.8, 0.0),
    "rotation_axis2a": (0, 1, 0),
    "rotation_angle2a": 90,
    "rotation_axis2b": (0, 0, 1),
    "rotation_angle2b": 45,
    "pos3": (0.4, 0.8, 0.6),
    "rotation_axis3": (1, 0, 0),
    "rotation_angle3": -45,
    "pos4": (0.4, 0.8, -0.6),
    "rotation_axis4a": (0, 1, 0),
    "rotation_angle4a": 180,
    "rotation_axis4b": (1, 0, 0),
    "rotation_angle4b": 45,
    # 夹爪摄像头（眼在手上）的相对参数
    "hand_cam_offset": (0.05, 0.0, 0.0),
    "hand_cam_axis_primary": (1, 0, 0),
    "hand_cam_angle_primary": 180,
    "hand_cam_axis_secondary": (0, 0, 1),
    "hand_cam_angle_secondary": 90,

}

# 构建完整的模拟场景
def build_scene(gym, sim, viewer, robot_asset, workbench_asset, cube_down_asset, cube_up_asset, num_envs=1, spacing=1.0, 
                hand_name="panda_hand", attractor_stiffness=5e5, attractor_damping=5e3):
    # 环境布局参数
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))
    
    # 初始化存储结构
    envs = []       # 环境列表
    robot_handles = []      # 机器人句柄列表
    attractor_handles = []      # 吸引子句柄列表
    camera_handles = []     # 摄像头句柄列表
    
    # 创建吸引子属性
    attractor_props = create_attractor_properties(attractor_stiffness, attractor_damping)
    
    # 创建机器人初始姿态
    robot_pose = create_robot_pose()
    
    # 创建可视化几何体（坐标系和吸引子）
    axes_geom, sphere_geom = create_visualization_geometries()
    
    print("Creating %d environments" % num_envs)
    
    # 创建第一个环境
    temp_env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    
    # 为第一个环境创建机器人
    robot_handle, body_dict, dof_dict, props, hand_handle = create_robot_actor(
        gym, temp_env, robot_asset, robot_pose, 0, hand_name
    )
    
    # 获取机器人DOF属性并配置
    dof_props = gym.get_actor_dof_properties(temp_env, robot_handle)
    dof_props = setup_robot_dof_properties(dof_props)
    
    # 获取关节信息
    lower_limits = dof_props['lower'].copy()
    upper_limits = dof_props['upper'].copy()
    mids = 0.5 * (upper_limits + lower_limits)  # 计算中点位置
    num_dofs = len(dof_props)

    # 记录夹爪 DOF 索引，便于后续开合控制
    finger_dof_indices = [
        dof_dict.get("panda_finger_joint1"),
        dof_dict.get("panda_finger_joint2"),
    ]

    if None in finger_dof_indices:
        raise ValueError("Finger DOF indices not found in Franka asset")

    # 设置吸引子初始位置
    attractor_props.target = props['pose'][:][body_dict[hand_name]]
    attractor_props.target.p.y = attractor_props.target.p.y - 0.1  # 吸引子在手下方0.1m
    attractor_props.rigid_handle = hand_handle 
    
    # 绘制吸引子可视化
    gymutil.draw_lines(axes_geom, gym, viewer, temp_env, attractor_props.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, temp_env, attractor_props.target)

    # 创建工作台
    workbench_handle = create_workbench_actor(gym, temp_env, workbench_asset, position=(0.8, 0.2, 0.0), env_id=0)
    
    # 配置工作台接触属性
    configure_actor_shape_properties(gym, temp_env, workbench_handle, 
                                    friction=2.0, restitution=0.0, 
                                    contact_offset=0.03, rest_offset=0.0)

    # 创建立方体
    # ｘ轴范围（0.225-0.625）z轴范围（-0.4-0.4）
    cube_down_pos = (0.225, 0.325, -0.4)
    cube_up_pos = (0.6, 0.325, 0.4)
    cube_down_handle = create_cube_actor(gym, temp_env, cube_down_asset, position=cube_down_pos, env_id=0)
    cube_up_handle = create_cube_actor(gym, temp_env, cube_up_asset, position=cube_up_pos, env_id=0)
    
    # 配置立方体接触属性
    configure_actor_shape_properties(gym, temp_env, cube_down_handle, 
                                    friction=2.0, restitution=0.0,
                                    contact_offset=0.03, rest_offset=0.0)
    configure_actor_shape_properties(gym, temp_env, cube_up_handle, 
                                    friction=2.0, restitution=0.0,
                                    contact_offset=0.03, rest_offset=0.0)

    # 绘制机器人基座坐标系
    if VISUALIZE_AXES:
        base_axes_geom = gymutil.AxesGeometry(2.0)  # 较大的坐标系，长度2.0m
        base_transform = gymapi.Transform()
        base_transform.p = gymapi.Vec3(0.0, 0.0, 0.0)  # 基座位置
        base_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 无旋转
        gymutil.draw_lines(base_axes_geom, gym, viewer, temp_env, base_transform)
    
    # 创建吸引子
    attractor_handle = gym.create_rigid_body_attractor(temp_env, attractor_props)
    
    # 创建摄像头几何体
    camera_axes_geom = gymutil.AxesGeometry(camera_params["axis_length"])  # 较小的坐标系，长度0.1m

    # 创建摄像头1 - 俯视图
    camera_handle_1, camera_transform_1 = create_camera_sensor(
        gym, temp_env,
        width=camera_params["width"], height=camera_params["height"],
        pos=camera_params["pos1"],
        rotation_axis=camera_params["rotation_axis1"],
        rotation_angle=camera_params["rotation_angle1"],
    )
    if VISUALIZE_AXES:
        draw_camera_axes_single(
            gym, viewer, temp_env, camera_axes_geom,
            camera_params["pos1"],
            camera_params["rotation_axis1"],
            camera_params["rotation_angle1"],
        )
    
    # 创建摄像头2 - 正前方视图（朝向机器人）
    camera_handle_2, camera_transform_2 = create_camera_sensor(
        gym, temp_env,
        width=camera_params["width"], height=camera_params["height"],
        pos=camera_params["pos2"],
        rotation_axis=camera_params["rotation_axis2a"],
        rotation_angle=camera_params["rotation_angle2a"],
        rotation_axis2=camera_params["rotation_axis2b"],
        rotation_angle2=camera_params["rotation_angle2b"],
    )
    if VISUALIZE_AXES:
        draw_camera_axes_single(
            gym, viewer, temp_env, camera_axes_geom,
            camera_params["pos2"],
            camera_params["rotation_axis2a"],
            camera_params["rotation_angle2a"],
            camera_params.get("rotation_axis2b"),
            camera_params.get("rotation_angle2b", 0),
        )

    # 创建摄像头3 - 侧视图１
    camera_handle_3, camera_transform_3 = create_camera_sensor(
        gym, temp_env,
        width=camera_params["width"], height=camera_params["height"],
        pos=camera_params["pos3"],
        rotation_axis=camera_params["rotation_axis3"],
        rotation_angle=camera_params["rotation_angle3"],
    )
    if VISUALIZE_AXES:
        draw_camera_axes_single(
            gym, viewer, temp_env, camera_axes_geom,
            camera_params["pos3"],
            camera_params["rotation_axis3"],
            camera_params["rotation_angle3"],
        )
    
    # 创建摄像头4 - 侧视图２
    camera_handle_4, camera_transform_4 = create_camera_sensor(
        gym, temp_env,
        width=camera_params["width"], height=camera_params["height"],
        pos=camera_params["pos4"],
        rotation_axis=camera_params["rotation_axis4a"],
        rotation_angle=camera_params["rotation_angle4a"],
        rotation_axis2=camera_params["rotation_axis4b"],
        rotation_angle2=camera_params["rotation_angle4b"],
    )
    if VISUALIZE_AXES:
        draw_camera_axes_single(
            gym, viewer, temp_env, camera_axes_geom,
            camera_params["pos4"],
            camera_params["rotation_axis4a"],
            camera_params["rotation_angle4a"],
            camera_params.get("rotation_axis4b"),
            camera_params.get("rotation_angle4b", 0),
        )
    
    # 创建摄像头5 - 眼在手上的摄像头（绑定在 panda_hand 上）
    hand_camera_handle, hand_camera_mount_tf = create_eye_in_hand_camera(
        gym,
        temp_env,
        hand_handle,
        width=camera_params["width"],
        height=camera_params["height"],
        offset=camera_params["hand_cam_offset"],
        rotation_axis_primary=camera_params["hand_cam_axis_primary"],
        rotation_angle_primary=camera_params["hand_cam_angle_primary"],
        rotation_axis_secondary=camera_params["hand_cam_axis_secondary"],
        rotation_angle_secondary=camera_params["hand_cam_angle_secondary"],
    )
    

    envs.append(temp_env)
    robot_handles.append(robot_handle)
    attractor_handles.append(attractor_handle)
    camera_handles.append(camera_handle_1)
    camera_handles.append(camera_handle_2)
    camera_handles.append(camera_handle_3)
    camera_handles.append(camera_handle_4)
    camera_handles.append(hand_camera_handle)
    cube_handles = [cube_down_handle, cube_up_handle]
    
    '''
    # 为其他环境创建场景
    for i in range(1, num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)
        
        # 创建机器人
        robot_handle, body_dict, props, hand_handle = create_robot_actor(
            gym, env, robot_asset, robot_pose, i, hand_name
        )
        robot_handles.append(robot_handle)
        
        # 配置关节属性
        configure_robot_dof_properties(gym, env, robot_handle, dof_props)
        
        # 设置吸引子
        attractor_props.target = props['pose'][:][body_dict[hand_name]]
        attractor_props.target.p.y -= 0.1
        attractor_props.target.p.z = 0.1
        attractor_props.rigid_handle = hand_handle
        
        # 绘制吸引子可视化
        gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_props.target)
        gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_props.target)
        
        # 创建吸引子
        attractor_handle = gym.create_rigid_body_attractor(env, attractor_props)
        attractor_handles.append(attractor_handle)
        
        # 创建摄像头
        camera_handle, camera_transform = create_camera_sensor(gym, env)
        gymutil.draw_lines(camera_axes_geom, gym, viewer, env, camera_transform)
        camera_handles.append(camera_handle)
    '''
    # 为所有环境应用DOF属性
    for i in range(num_envs):
        configure_robot_dof_properties(gym, envs[i], robot_handles[i], dof_props)
    
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
        'cube_down_pos': cube_down_pos,
        'cube_up_pos': cube_up_pos,
        'initial_hand_pose': attractor_props.target,
        'hand_camera_mount_tf': hand_camera_mount_tf,
    }

# ===============================
# 第四部分：摄像头系统
# ===============================

#　创建摄像头输出目录
def setup_camera_output_directory(output_dir="./camera_outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Camera output directory created at {output_dir}")

# 渲染并保存摄像头图像
def render_and_save_camera_images(gym, sim, envs, camera_handles, 
                                   output_dir="./camera_outputs", 
                                   capture_count=0):
    # 渲染所有摄像头传感器
    gym.render_all_camera_sensors(sim)
    
    # 保存所有摄像头的图像
    camera_idx = 0
    for env_idx in range(len(envs)):
        env = envs[env_idx]
        
        # 遍历该环境的所有摄像头
        for camera_handle in camera_handles:
            # 为每个摄像头创建子文件夹
            camera_dir = f"{output_dir}/camera_{camera_idx}"
            if not os.path.exists(camera_dir):
                os.makedirs(camera_dir)
            
            # 生成输出文件名 - 保存在对应的摄像头文件夹中
            rgb_filename = f"{camera_dir}/{capture_count:04d}.png"
            
            # 保存RGB图像
            gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
            camera_idx += 1
    
    return capture_count + 1

#　初始化摄像头系统参数
def initialize_camera_system(output_dir="./camera_outputs", 
                            capture_frequency=10, 
                            capture_duration=10.0,
                            start_time=1.5):
    # 创建输出目录
    setup_camera_output_directory(output_dir)
    
    # 计算采集间隔
    capture_interval = 1.0 / capture_frequency
    
    # 计算总帧数
    total_frames = int(capture_duration * capture_frequency)
    
    return {
        'output_dir': output_dir,
        'capture_frequency': capture_frequency,
        'capture_interval': capture_interval,
        'capture_duration': capture_duration,
        'total_frames': total_frames,
        'start_time': start_time,
        'next_capture_time': start_time,
        'capture_count': 0
    }

# 判断是否应该采集当前帧
def should_capture_frame(current_time, camera_system_data):
    current_count = camera_system_data['capture_count']
    total_frames = camera_system_data['total_frames']
    next_capture_time = camera_system_data['next_capture_time']
    
    # 检查是否达到总帧数或时间限制
    if current_count >= total_frames:
        return False
    
    # 检查是否到达采集时间
    if current_time >= next_capture_time:
        return True
    
    return False

# 更新摄像头采集时间和计数
def update_camera_capture_time(camera_system_data):
    camera_system_data['next_capture_time'] += camera_system_data['capture_interval']
    camera_system_data['capture_count'] += 1
    return camera_system_data

# 记录摄像头采集进度日志
def log_capture_progress(capture_count, current_time, log_interval=10):
    if (capture_count + 1) % log_interval == 0:
        print(f"Captured {capture_count + 1} camera frames at {current_time:.2f}s")


# ===============================
# 第五部分：吸引子更新（抓取/放置规划）
# ===============================

def lerp_vec(a, b, alpha):
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )

def slerp_quat(q1, q2, alpha):
    """四元数球面线性插值"""
    # 计算点积
    dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w
    
    # 如果点积为负，反转一个四元数以选择最短路径
    if dot < 0.0:
        q2 = gymapi.Quat(-q2.x, -q2.y, -q2.z, -q2.w)
        dot = -dot
    
    # 线性插值（简化版，适用于小角度）
    result = gymapi.Quat(
        q1.x + (q2.x - q1.x) * alpha,
        q1.y + (q2.y - q1.y) * alpha,
        q1.z + (q2.z - q1.z) * alpha,
        q1.w + (q2.w - q1.w) * alpha
    )
    
    # 归一化
    norm = (result.x**2 + result.y**2 + result.z**2 + result.w**2)**0.5
    if norm > 0:
        result.x /= norm
        result.y /= norm
        result.z /= norm
        result.w /= norm
    
    return result

# 构建抓取/放置动作规划
def build_pick_place_plan(initial_pose, cube_up_pos, cube_down_pos,
                          hover_offset, grasp_offset, release_offset):

    # 起始位置和姿态
    start_pos = (initial_pose.p.x, initial_pose.p.y, initial_pose.p.z)
    start_rot = initial_pose.r
    
    # 夹爪向下的旋转（绕X轴180度，使夹爪指向-Y方向）
    grasp_rot = gymapi.Quat(0.707106, 0.0, 0.0, 0.707106)  # 向下姿态
    
    # 计算吸引子目标位置
    hover_up = (cube_up_pos[0], cube_up_pos[1] + hover_offset, cube_up_pos[2])
    grasp_pos = (cube_up_pos[0], cube_up_pos[1] + grasp_offset, cube_up_pos[2])
    lift_pos = hover_up
    hover_down = (cube_down_pos[0], cube_down_pos[1] + hover_offset, cube_down_pos[2])
    place_pos = (cube_down_pos[0], cube_down_pos[1] + release_offset, cube_down_pos[2])
    retreat_pos = (cube_down_pos[0], cube_down_pos[1] + hover_offset + 0.05, cube_down_pos[2])

    # finger_width 代表两指张开的总宽度
    finger_open = 0.08
    finger_closed = 0.045

    plan = [
        {"name": "move_pregrasp", "start": start_pos, "goal": hover_up, "start_rot": start_rot, "goal_rot": grasp_rot, "duration": 2.0, "start_finger_width": finger_open, "goal_finger_width": finger_open},
        {"name": "descend_grasp", "start": hover_up, "goal": grasp_pos, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 2.0, "start_finger_width": finger_open, "goal_finger_width": finger_open},
        {"name": "close_gripper", "start": grasp_pos, "goal": grasp_pos, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 1.5, "start_finger_width": finger_open, "goal_finger_width": finger_closed},
        {"name": "stabilize_after_grasp", "start": grasp_pos, "goal": grasp_pos, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 0.5, "start_finger_width": finger_closed, "goal_finger_width": finger_closed},
        {"name": "lift", "start": grasp_pos, "goal": lift_pos, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 1.5, "start_finger_width": finger_closed, "goal_finger_width": finger_closed},
        {"name": "move_over_drop", "start": lift_pos, "goal": hover_down, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 3.0, "start_finger_width": finger_closed, "goal_finger_width": finger_closed},
        {"name": "stabilize_before_place", "start": hover_down, "goal": hover_down, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 0.5, "start_finger_width": finger_closed, "goal_finger_width": finger_closed},
        {"name": "place_release", "start": hover_down, "goal": place_pos, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 2.0, "start_finger_width": finger_closed, "goal_finger_width": finger_closed},
        {"name": "open_gripper", "start": place_pos, "goal": place_pos, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 1.5, "start_finger_width": finger_closed, "goal_finger_width": finger_open},
        {"name": "retreat", "start": place_pos, "goal": retreat_pos, "start_rot": grasp_rot, "goal_rot": grasp_rot, "duration": 1.5, "start_finger_width": finger_open, "goal_finger_width": finger_open},
    ]

    return plan

# 控制夹爪开合
def command_gripper(gym, envs, robot_handles, finger_dof_indices, target_width, base_dof_pos):
    """Set gripper finger targets via per-actor DOF position targets array."""
    # 添加最小间隙，避免目标为0时的硬碰撞导致非对称回弹
    min_gap = 0.001  # 1mm 最小间隙
    single_finger_pos = max(target_width * 0.5, min_gap * 0.5)
    
    for env, handle in zip(envs, robot_handles):
        # 获取当前DOF目标，只修改夹爪，不干扰手臂关节
        dof_states = gym.get_actor_dof_states(env, handle, gymapi.STATE_NONE)
        targets = dof_states['pos'].copy()
        # 手动同步两个指头（Isaac Gym不自动处理URDF mimic约束）
        targets[finger_dof_indices[0]] = single_finger_pos
        targets[finger_dof_indices[1]] = single_finger_pos
        gym.set_actor_dof_position_targets(env, handle, targets)
        

# 更新抓取/放置吸引子位置和夹爪状态
def update_pick_and_place(gym, viewer, envs, attractor_handles, axes_geom, sphere_geom,
                          plan_state, finger_dof_indices, robot_handles, base_dof_pos):
    if not plan_state['running']:
        return plan_state

    t = plan_state['current_time']      # 当前时间
    dt = plan_state['dt']               # 时间步长
    phase_idx = plan_state['phase_idx'] # 当前阶段索引
    
    if phase_idx >= len(plan_state['plan']):
        return plan_state  # 已完成

    gym.clear_lines(viewer)

    phase = plan_state['plan'][phase_idx]
    plan_state['phase_elapsed'] += dt
    alpha = min(plan_state['phase_elapsed'] / max(phase['duration'], 1e-4), 1.0)

    target_pos = lerp_vec(phase['start'], phase['goal'], alpha)
    target_rot = slerp_quat(phase['start_rot'], phase['goal_rot'], alpha)
    
    # 对夹爪宽度也进行线性插值，实现平滑闭合/打开
    target_finger_width = phase['start_finger_width'] + (phase['goal_finger_width'] - phase['start_finger_width']) * alpha

    # 更新吸引子目标（位置和旋转）
    pose = plan_state['current_pose']
    pose.p.x, pose.p.y, pose.p.z = target_pos
    pose.r = target_rot
    gym.set_attractor_target(envs[0], attractor_handles[0], pose)

    # 可视化当前吸引子
    gymutil.draw_lines(axes_geom, gym, viewer, envs[0], pose)
    gymutil.draw_lines(sphere_geom, gym, viewer, envs[0], pose)

    # 控制夹爪（使用插值后的宽度）
    command_gripper(gym, envs, robot_handles, finger_dof_indices, target_finger_width, base_dof_pos)

    # 阶段切换
    if plan_state['phase_elapsed'] >= phase['duration']:
        plan_state['phase_idx'] += 1
        plan_state['phase_elapsed'] = 0.0
        # 下一段的起点用当前目标，避免跳变
        if plan_state['phase_idx'] < len(plan_state['plan']):
            plan_state['plan'][plan_state['phase_idx']]['start'] = target_pos
            plan_state['plan'][plan_state['phase_idx']]['start_rot'] = target_rot

    return plan_state


# ===============================
# 第六部分：主循环
# ===============================
# 主模拟循环
def run_simulation(gym, sim, viewer, envs, franka_handles, attractor_handles, camera_handles,
                   franka_mids, franka_num_dofs, axes_geom, sphere_geom,camera_axes_geom,
                   camera_system, cube_handles, plan_state, finger_dof_indices,
                   gravity_toggle_supported=True, sim_start_time=1.5, body_dict=None, hand_name="panda_hand"):
    last_t = gym.get_sim_time(sim)
    last_print_time = 0.0  # 用于控制打印频率
    print_interval = 0.2   # 每0.2秒打印一次

    # 打印初始姿态
    robot_props = gym.get_actor_rigid_body_states(envs[0], franka_handles[0], gymapi.STATE_POS)
    hand_pose = robot_props['pose'][:][body_dict[hand_name]]
    print(f"Initial panda_hand position: x={hand_pose['p']['x']:.4f}, y={hand_pose['p']['y']:.4f}, z={hand_pose['p']['z']:.4f}")
    print(f"Initial panda_hand orientation: x={hand_pose['r']['x']:.4f}, y={hand_pose['r']['y']:.4f}, z={hand_pose['r']['z']:.4f}, w={hand_pose['r']['w']:.4f}")

    # 主模拟循环
    while not gym.query_viewer_has_closed(viewer):
        # 获取当前模拟时间
        t = gym.get_sim_time(sim)
        dt = t - last_t
        last_t = t
        
        # 实时打印位置信息
        if body_dict and t - last_print_time >= print_interval:
            last_print_time = t
            # 获取机器人刚体状态
            robot_props = gym.get_actor_rigid_body_states(envs[0], franka_handles[0], gymapi.STATE_POS)
            
            # 获取立方体状态
            cube_up_props = gym.get_actor_rigid_body_states(envs[0], cube_handles[0], gymapi.STATE_POS)
            cube_down_props = gym.get_actor_rigid_body_states(envs[0], cube_handles[1], gymapi.STATE_POS)
            
            # 获取各个刚体的位置
            hand_idx = body_dict[hand_name]
            panda_hand_pose_data = robot_props['pose'][:][hand_idx]
            panda_hand_pos = panda_hand_pose_data['p']
            
            cube_up_pose_data = cube_up_props['pose'][:][0]
            cube_up_pos_current = cube_up_pose_data['p']
            
            cube_down_pose_data = cube_down_props['pose'][:][0]
            cube_down_pos_current = cube_down_pose_data['p']
            
            # 获取吸引子位置（从 plan_state 中取）
            attractor_pos = plan_state['current_pose'].p
            
            print(f"\n[t={t:.2f}s] === 实时位置信息 ===")
            print(f"cube_up:      x={cube_up_pos_current['x']:.4f}, y={cube_up_pos_current['y']:.4f}, z={cube_up_pos_current['z']:.4f}")
            print(f"cube_down:    x={cube_down_pos_current['x']:.4f}, y={cube_down_pos_current['y']:.4f}, z={cube_down_pos_current['z']:.4f}")
            print(f"attractor:    x={attractor_pos.x:.4f}, y={attractor_pos.y:.4f}, z={attractor_pos.z:.4f}")
            print(f"panda_hand:   x={panda_hand_pos['x']:.4f}, y={panda_hand_pos['y']:.4f}, z={panda_hand_pos['z']:.4f}")
            print(f"=========================")
        
        # 启动抓取/放置规划
        if (not plan_state['running']) and t >= plan_state['start_time']:
            plan_state['running'] = True
            plan_state['current_time'] = t
            plan_state['dt'] = dt
            plan_state = update_pick_and_place(
                gym, viewer, envs, attractor_handles, axes_geom, sphere_geom,
                plan_state, finger_dof_indices, franka_handles, franka_mids,
            )
        elif plan_state['running']:
            plan_state['current_time'] = t
            plan_state['dt'] = dt
            plan_state = update_pick_and_place(
                gym, viewer, envs, attractor_handles, axes_geom, sphere_geom,
                plan_state, finger_dof_indices, franka_handles, franka_mids,
            )

        # 执行物理模拟步骤
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        #　可视化摄像头坐标系
        if VISUALIZE_AXES:
            draw_camera_axes_single(
                gym, viewer, envs[0], camera_axes_geom,
                camera_params["pos1"],
                camera_params["rotation_axis1"],
                camera_params["rotation_angle1"],
            )
            draw_camera_axes_single(
                gym, viewer, envs[0], camera_axes_geom,
                camera_params["pos2"],
                camera_params["rotation_axis2a"],
                camera_params["rotation_angle2a"],
                camera_params.get("rotation_axis2b"),
                camera_params.get("rotation_angle2b", 0),
            )
            draw_camera_axes_single(
                gym, viewer, envs[0], camera_axes_geom,
                camera_params["pos3"],
                camera_params["rotation_axis3"],
                camera_params["rotation_angle3"],
            )
            draw_camera_axes_single(
                gym, viewer, envs[0], camera_axes_geom,
                camera_params["pos4"],
                camera_params["rotation_axis4a"],
                camera_params["rotation_angle4a"],
                camera_params.get("rotation_axis4b"),
                camera_params.get("rotation_angle4b", 0),
            )

        # 可视化机器人基座坐标系
        if VISUALIZE_AXES:
            base_axes_geom = gymutil.AxesGeometry(2.0)  # 较大的坐标系，长度2.0m
            base_transform = gymapi.Transform()
            base_transform.p = gymapi.Vec3(0.0, 0.0, 0.0)  # 基座位置
            base_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 无旋转
            gymutil.draw_lines(base_axes_geom, gym, viewer, envs[0], base_transform)

        # 执行渲染步骤（更新可视化）
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

        # 摄像头拍摄逻辑
        if should_capture_frame(t, camera_system):
            # 渲染并保存摄像头图像
            render_and_save_camera_images(
                gym, sim, envs, camera_handles,
                output_dir=camera_system['output_dir'],
                capture_count=camera_system['capture_count']
            )
            # 记录进度
            log_capture_progress(camera_system['capture_count'], t)
            # 更新摄像头系统状态
            camera_system = update_camera_capture_time(camera_system)


# ===============================
# 主程序入口
# ===============================

# 初始化 Isaac Gym 环境
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(description="Franka Attractor Example")

# 初始化模拟环境和查看器
sim, viewer = initialize_simulation_env(gym, args)

# 设置光照
setup_lighting(gym, sim)

# 添加地面
add_ground_plane(gym, sim)

# 定义资产路径
asset_root = "./urdf"
franka_asset_file = "franka_description/robots/franka_panda.urdf"
workbench_asset_file = "workbench.urdf"
cube_down_asset_file = "cube_down.urdf"
cube_up_asset_file = "cube_up.urdf"

# 创建资产加载选项
asset_options = create_asset_options(
    fix_base=True,
    flip_visual=True,
    armature=0.01
)

# 加载Franka机器人资产
franka_asset = load_robot_asset(gym, sim, asset_root, franka_asset_file, asset_options)

# 加载工作台资产
workbench_asset = load_workbench_asset(gym, sim, asset_root, workbench_asset_file)

# 加载立方体资产
cube_down_asset = load_cube_asset(gym, sim, asset_root, cube_down_asset_file)
cube_up_asset = load_cube_asset(gym, sim, asset_root, cube_up_asset_file)

# 场景参数
num_envs = 1
spacing = 1.0
hand_name = "panda_hand"

# 构建场景
scene_data = build_scene(
    gym, sim, viewer, franka_asset, workbench_asset,cube_down_asset,cube_up_asset,
    num_envs=num_envs,
    spacing=spacing,
    hand_name=hand_name
)

# 提取场景数据供后续使用
envs = scene_data['envs']
franka_handles = scene_data['robot_handles']
attractor_handles = scene_data['attractor_handles']
camera_handles = scene_data['camera_handles']
franka_dof_props = scene_data['dof_props']
franka_lower_limits = scene_data['lower_limits']
franka_upper_limits = scene_data['upper_limits']
franka_mids = scene_data['mids']
franka_num_dofs = scene_data['num_dofs']
axes_geom = scene_data['axes_geom']
sphere_geom = scene_data['sphere_geom']
camera_axes_geom = scene_data['camera_axes_geom']
cube_handles = scene_data['cube_handles']
finger_dof_indices = scene_data['finger_dof_indices']
cube_down_pos = scene_data['cube_down_pos']
cube_up_pos = scene_data['cube_up_pos']
initial_hand_pose = scene_data['initial_hand_pose']

# 初始化机器人状态
initialize_robot_states(gym, envs, franka_handles, franka_mids, franka_num_dofs)

# 初始化后重新获取手的位置并更新吸引子
robot_props_updated = gym.get_actor_rigid_body_states(envs[0], franka_handles[0], gymapi.STATE_POS)
hand_pose_updated = robot_props_updated['pose'][:][scene_data['body_dict'][hand_name]]

print(
    "Eye-in-hand mount base (panda_hand) world pose -> "
    f"pos: ({hand_pose_updated['p']['x']:.4f}, {hand_pose_updated['p']['y']:.4f}, {hand_pose_updated['p']['z']:.4f}), "
    f"quat: ({hand_pose_updated['r']['x']:.4f}, {hand_pose_updated['r']['y']:.4f}, {hand_pose_updated['r']['z']:.4f}, {hand_pose_updated['r']['w']:.4f})"
)

# 绘制夹爪(panda_hand)的坐标系用于可视化
hand_axes_geom = gymutil.AxesGeometry(0.15)  # 手坐标系，长度0.15m
hand_transform = gymapi.Transform()
hand_transform.p = gymapi.Vec3(hand_pose_updated['p']['x'], hand_pose_updated['p']['y'], hand_pose_updated['p']['z'])
hand_transform.r = gymapi.Quat(hand_pose_updated['r']['x'], hand_pose_updated['r']['y'], hand_pose_updated['r']['z'], hand_pose_updated['r']['w'])
if VISUALIZE_AXES:
    gymutil.draw_lines(hand_axes_geom, gym, viewer, envs[0], hand_transform)
print(f"Hand (panda_hand) coordinate frame drawn at position")

print(
    "Eye-in-hand relative mount (hand frame) -> offset: "
    f"{camera_params['hand_cam_offset']}, primary axis/angle: "
    f"{camera_params['hand_cam_axis_primary']}/{camera_params['hand_cam_angle_primary']} deg, "
    f"secondary axis/angle: {camera_params['hand_cam_axis_secondary']}/{camera_params['hand_cam_angle_secondary']} deg"
)

# 更新吸引子目标位置为初始化后的手的位置（下方0.1m）
updated_attractor_target = gymapi.Transform(
    p=gymapi.Vec3(hand_pose_updated['p']['x'], hand_pose_updated['p']['y'] - 0.1, hand_pose_updated['p']['z']),
    r=gymapi.Quat(hand_pose_updated['r']['x'], hand_pose_updated['r']['y'], hand_pose_updated['r']['z'], hand_pose_updated['r']['w'])
)
gym.set_attractor_target(envs[0], attractor_handles[0], updated_attractor_target)

# 设置观察角度
cam_pos = gymapi.Vec3(2.0, 2.0, 2.0)
cam_target = gymapi.Vec3(0.5, 0.3, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

sim_start_time = 1.5

# 构建抓取-放置规划（使用更新后的吸引子位置）
plan_initial_pose = gymapi.Transform(
    p=gymapi.Vec3(updated_attractor_target.p.x, updated_attractor_target.p.y, updated_attractor_target.p.z),
    r=gymapi.Quat(updated_attractor_target.r.x, updated_attractor_target.r.y, updated_attractor_target.r.z, updated_attractor_target.r.w),
)
motion_plan = build_pick_place_plan(
    plan_initial_pose, 
    cube_up_pos, 
    cube_down_pos,
    hover_offset=0.2,      # 悬停在立方体上方20cm
    grasp_offset=0.1,      # 抓取时夹爪在立方体上方10cm（考虑立方体高度约10cm）
    release_offset=0.15    # 放置时留15cm高度
)

plan_state = {
    'plan': motion_plan,        # 抓取-放置动作序列
    'phase_idx': 0,             # 当前阶段索引
    'phase_elapsed': 0.0,           # 当前阶段已过时间
    'current_pose': plan_initial_pose,      # 当前吸引子姿态
    'running': False,           # 规划是否正在运行
    'current_time': 0.0,            # 当前模拟时间
    'dt': 0.0,          # 时间步长
    'start_time': sim_start_time,           # 规划开始时间
}

# 初始化摄像头系统
camera_system = initialize_camera_system(
    output_dir="./camera_outputs",
    capture_frequency=10,           # 每秒10帧
    capture_duration=17.0,          # 总共采集17秒
    start_time=sim_start_time
)



# 运行主模拟循环
print("\nStarting simulation...")
run_simulation(
    gym, sim, viewer, envs, franka_handles, attractor_handles, camera_handles,
    franka_mids, franka_num_dofs, axes_geom, sphere_geom, camera_axes_geom,
    camera_system, cube_handles, plan_state, finger_dof_indices,
    sim_start_time=sim_start_time,
    body_dict=scene_data['body_dict'],
    hand_name=hand_name
)

# 模拟完成
print("Done")

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

