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
    props = gym.get_actor_rigid_body_states(env, robot_handle, gymapi.STATE_POS)    # 获取刚体状态
    hand_handle = gym.find_actor_rigid_body_handle(env, robot_handle, hand_name)    # 获取末端执行器句柄
    
    return robot_handle, body_dict, props, hand_handle

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
    
    # 夹爪关节（7及以后）使用位置驱动模式，并设置高刚度
    dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][7:] = 1e10
    dof_props['damping'][7:] = 1.0
    
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
    "pos1": (0.0, 1.5, 0.0),
    "rotation_axis1": (1, 0, 0),
    "rotation_angle1": -90,
    "pos2": (0.5, 1.5, 0.0),
    "rotation_axis2a": (0, 1, 0),
    "rotation_angle2a": 90,
    "rotation_axis2b": (0, 0, 1),
    "rotation_angle2b": 45,
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
    robot_handle, body_dict, props, hand_handle = create_robot_actor(
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
    
    # 设置吸引子目标位置
    attractor_props.target = props['pose'][:][body_dict[hand_name]]
    attractor_props.target.p.y -= 0.1
    attractor_props.target.p.z = 0.1
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
    cube_pos_down = (0.8, 0.325, -0.3)
    cube_pos_up = (0.8, 0.325, 0.3)
    cube_handle_down = create_cube_actor(gym, temp_env, cube_down_asset, position=cube_pos_down, env_id=0)
    cube_handle_up = create_cube_actor(gym, temp_env, cube_up_asset, position=cube_pos_up, env_id=0)
    
    # 配置立方体接触属性
    configure_actor_shape_properties(gym, temp_env, cube_handle_down, 
                                    friction=2.0, restitution=0.0,
                                    contact_offset=0.03, rest_offset=0.0)
    configure_actor_shape_properties(gym, temp_env, cube_handle_up, 
                                    friction=2.0, restitution=0.0,
                                    contact_offset=0.03, rest_offset=0.0)

    # 绘制机器人基座坐标系
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
    draw_camera_axes_single(
        gym, viewer, temp_env, camera_axes_geom,
        camera_params["pos2"],
        camera_params["rotation_axis2a"],
        camera_params["rotation_angle2a"],
        camera_params.get("rotation_axis2b"),
        camera_params.get("rotation_angle2b", 0),
    )
    
    envs.append(temp_env)
    robot_handles.append(robot_handle)
    attractor_handles.append(attractor_handle)
    camera_handles.append(camera_handle_1)
    camera_handles.append(camera_handle_2)
    cube_handles = [cube_handle_down, cube_handle_up]
    
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
        'hand_name': hand_name
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
# 第五部分：吸引子更新
# ===============================

# 更新所有环境中吸引子的目标位置
def update_franka(t):
    gym.clear_lines(viewer)  # 清除之前绘制的所有线条
    for i in range(num_envs):
        # 更新吸引子的目标位置
        attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i])
        pose = attractor_properties.target
        
        # 计算随时间变化的目标轨迹（使用正弦和余弦波形）
        pose.p.x = 0.2 * math.sin(1.5 * t - math.pi * float(i) / num_envs)  # X 方向：正弦波
        pose.p.y = 0.7 + 0.1 * math.cos(2.5 * t - math.pi * float(i) / num_envs)  # Y 方向：余弦波加偏移
        pose.p.z = 0.2 * math.cos(1.5 * t - math.pi * float(i) / num_envs)  # Z 方向：余弦波

        # 设置吸引子的新目标位置
        gym.set_attractor_target(envs[i], attractor_handles[i], pose)

        # 在更新的位置重新绘制吸引子的可视化元素
        gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)

        # 重新绘制机器人基座坐标系
        base_axes_geom = gymutil.AxesGeometry(2.0)  # 较大的坐标系，长度2.0m
        base_transform = gymapi.Transform()
        base_transform.p = gymapi.Vec3(0.0, 0.0, 0.0)  # 基座位置
        base_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 无旋转
        gymutil.draw_lines(base_axes_geom, gym, viewer, envs[i], base_transform)

        # 重新绘制两个摄像头的坐标轴
        draw_camera_axes_single(
            gym, viewer, envs[i], camera_axes_geom,
            camera_params["pos1"],
            camera_params["rotation_axis1"],
            camera_params["rotation_angle1"],
        )
        draw_camera_axes_single(
            gym, viewer, envs[i], camera_axes_geom,
            camera_params["pos2"],
            camera_params["rotation_axis2a"],
            camera_params["rotation_angle2a"],
            camera_params.get("rotation_axis2b"),
            camera_params.get("rotation_angle2b", 0),
        )


# ===============================
# 第六部分：主循环
# ===============================
# 主模拟循环
def run_simulation(gym, sim, viewer, envs, franka_handles, attractor_handles, camera_handles,
                   franka_mids, franka_num_dofs, axes_geom, sphere_geom,
                   camera_system, cube_handles, gravity_toggle_supported=True,
                   sim_start_time=1.5, ):
    update_time = sim_start_time
    # 主模拟循环
    while not gym.query_viewer_has_closed(viewer):
        # 获取当前模拟时间
        t = gym.get_sim_time(sim)
        
        # 更新吸引子目标
        if t >= update_time:
            update_franka(t)
            update_time += 0.01

        # 执行物理模拟步骤
        gym.simulate(sim)
        gym.fetch_results(sim, True)

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

# 初始化机器人状态
initialize_robot_states(gym, envs, franka_handles, franka_mids, franka_num_dofs)

# 设置观察角度
cam_pos = gymapi.Vec3(-4.0, 4.0, -1.0)
cam_target = gymapi.Vec3(0.0, 2.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

sim_start_time = 1.5

# 初始化摄像头系统
camera_system = initialize_camera_system(
    output_dir="./camera_outputs",
    capture_frequency=10,
    capture_duration=10.0,
    start_time=sim_start_time
)

# 运行主模拟循环
print("\nStarting simulation...")
run_simulation(
    gym, sim, viewer, envs, franka_handles, attractor_handles, camera_handles,
    franka_mids, franka_num_dofs, axes_geom, sphere_geom,
    camera_system, cube_handles,
    sim_start_time=sim_start_time
)

# 模拟完成
print("Done")

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

