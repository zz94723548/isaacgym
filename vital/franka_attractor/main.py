"""
Franka 吸引子示例 - 主程序
========================

这个脚本演示了如何使用 Isaac Gym 进行 Franka Panda 机器人的位置控制。
机器人通过一个虚拟吸引子点进行运动控制，吸引子点的位置根据动作规划实时更新。

运行方法:
    python main.py
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch

# 导入配置和各个模块
from config import SimulationConfig as Config
from core import simulation, assets, scene
from systems import camera, sensor, gripper, attractor, planner
from utils import math_utils, visualization, io_utils


def apply_random_cube_positions():
    """根据随机种子生成可复现的物块位置"""
    if not Config.ENABLE_RANDOM_CUBE_POS:
        return
    x_min, x_max = Config.CUBE_RANGE_X
    z_min, z_max = Config.CUBE_RANGE_Z
    y = Config.CUBE_RANGE_Y
    
    rng = np.random.RandomState(Config.RANDOM_SEED)
    cube_down_x = rng.uniform(x_min, x_max)
    cube_down_z = rng.uniform(z_min, z_max)
    cube_up_x = rng.uniform(x_min, x_max)
    cube_up_z = rng.uniform(z_min, z_max)
    min_dist = Config.MIN_CUBE_DISTANCE
    for _ in range(Config.MAX_SAMPLE_TRIES):
        dx = cube_up_x - cube_down_x
        dz = cube_up_z - cube_down_z
        if (dx * dx + dz * dz) ** 0.5 >= min_dist:
            break
        cube_up_x = rng.uniform(x_min, x_max)
        cube_up_z = rng.uniform(z_min, z_max)
    
    Config.CUBE_DOWN_POS = (cube_down_x, y, cube_down_z)
    Config.CUBE_UP_POS = (cube_up_x, y, cube_up_z)


def setup_scene(gym, sim, viewer):
    """初始化和配置场景
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境
        viewer: 查看器
        
    Returns:
        dict: 场景数据
    """
    # 设置光照和地面
    simulation.setup_lighting(gym, sim)
    simulation.add_ground_plane(gym, sim)
    
    # 应用随机种子生成初始物块位置
    apply_random_cube_positions()
    
    # 加载资产
    franka_asset = assets.load_robot_asset(
        gym, sim, Config.ASSET_ROOT, Config.FRANKA_URDF
    )
    workbench_asset = assets.load_workbench_asset(
        gym, sim, Config.ASSET_ROOT, Config.WORKBENCH_URDF
    )
    cube_down_asset = assets.load_cube_asset(
        gym, sim, Config.ASSET_ROOT, Config.CUBE_DOWN_URDF
    )
    cube_up_asset = assets.load_cube_asset(
        gym, sim, Config.ASSET_ROOT, Config.CUBE_UP_URDF
    )
    
    # 构建场景
    scene_data = scene.build_scene(
        gym, sim, viewer,
        franka_asset, workbench_asset, cube_down_asset, cube_up_asset,
        num_envs=Config.NUM_ENVS,
        spacing=Config.SPACING,
        hand_name=Config.HAND_NAME,
        attractor_stiffness=Config.ATTRACTOR_STIFFNESS,
        attractor_damping=Config.ATTRACTOR_DAMPING
    )
    
    return scene_data


def initialize_systems(gym, sim, scene_data, viewer, output_dir=None):
    """初始化各个系统（传感器、摄像头等）
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境
        scene_data: 场景数据
        viewer: 查看器
        
    Returns:
        dict: 各个系统的初始化数据
    """
    envs = scene_data['envs']
    robot_handles = scene_data['robot_handles']
    body_dict = scene_data['body_dict']
    
    # 初始化机器人状态
    scene.initialize_robot_states(
        gym, envs, robot_handles,
        scene_data['mids'], scene_data['num_dofs']
    )
    
    # 创建摄像头
    if output_dir is None:
        output_dir = Config.CAPTURE_OUTPUT_DIR
    camera.setup_camera_output_directory(output_dir)
    sensor.setup_gel_output_directory(f"{output_dir}/gel")
    
    camera_handles = []
    for cam_config in Config.CAMERAS:
        cam_handle, _ = camera.create_camera_sensor(
            gym, envs[0],
            width=cam_config["width"],
            height=cam_config["height"],
            pos=cam_config["pos"],
            rotation_axis=cam_config["rotation_axis"],
            rotation_angle=cam_config["rotation_angle"],
            rotation_axis2=cam_config["rotation_axis2"],
            rotation_angle2=cam_config["rotation_angle2"],
        )
        camera_handles.append(cam_handle)
    
    # 创建眼在手上摄像头
    hand_handle = gym.find_actor_rigid_body_handle(
        envs[0], robot_handles[0], Config.HAND_NAME
    )
    hand_cam_handle, _ = camera.create_eye_in_hand_camera(
        gym, envs[0], hand_handle,
        width=Config.HAND_CAMERA_WIDTH,
        height=Config.HAND_CAMERA_HEIGHT,
        offset=Config.HAND_CAMERA_OFFSET,
        rotation_axis_primary=Config.HAND_CAMERA_AXIS_PRIMARY,
        rotation_angle_primary=Config.HAND_CAMERA_ANGLE_PRIMARY,
        rotation_axis_secondary=Config.HAND_CAMERA_AXIS_SECONDARY,
        rotation_angle_secondary=Config.HAND_CAMERA_ANGLE_SECONDARY,
    )
    camera_handles.append(hand_cam_handle)
    
    # 初始化传感器系统
    force_sensor_system = sensor.ForceSensorSystem()
    
    # 初始化摄像头系统
    camera_system = planner.initialize_camera_system(
        output_dir=output_dir,
        capture_frequency=Config.CAPTURE_FREQUENCY,
        capture_duration=Config.CAPTURE_DURATION,
        start_time=Config.CAPTURE_START_TIME
    )
    
    # 构建动作规划
    robot_props = gym.get_actor_rigid_body_states(envs[0], robot_handles[0], gymapi.STATE_POS)
    hand_pose = robot_props['pose'][:][body_dict[Config.HAND_NAME]]
    
    initial_pose = gymapi.Transform(
        p=gymapi.Vec3(hand_pose['p']['x'], hand_pose['p']['y'] - 0.1, hand_pose['p']['z']),
        r=gymapi.Quat(hand_pose['r']['x'], hand_pose['r']['y'], hand_pose['r']['z'], hand_pose['r']['w']),
    )
    
    motion_plan = planner.MotionPlanner.build_pick_place_plan(
        initial_pose,
        Config.CUBE_UP_POS,
        Config.CUBE_DOWN_POS,
        hover_offset=Config.MOTION_PLAN_HOVER_OFFSET,
        grasp_offset=Config.MOTION_PLAN_GRASP_OFFSET,
        release_offset=Config.MOTION_PLAN_RELEASE_OFFSET
    )
    
    plan_state = {
        'plan': motion_plan,
        'phase_idx': 0,
        'phase_elapsed': 0.0,
        'current_pose': initial_pose,
        'running': False,
        'current_time': 0.0,
        'dt': 0.0,
        'start_time': Config.CAPTURE_START_TIME,
    }
    
    return {
        'camera_handles': camera_handles,
        'force_sensor_system': force_sensor_system,
        'camera_system': camera_system,
        'plan_state': plan_state,
    }


def run_main_loop(gym, sim, viewer, scene_data, systems_data, stop_when_capture_done=True):
    """运行主模拟循环
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境
        viewer: 查看器
        scene_data: 场景数据
        systems_data: 系统初始化数据
    """
    # 提取数据
    envs = scene_data['envs']
    robot_handles = scene_data['robot_handles']
    attractor_handles = scene_data['attractor_handles']
    body_dict = scene_data['body_dict']
    
    camera_handles = systems_data['camera_handles']
    force_sensor_system = systems_data['force_sensor_system']
    camera_system = systems_data['camera_system']
    plan_state = systems_data['plan_state']
    
    # 设置观察角度
    cam_pos = gymapi.Vec3(2.0, 2.0, 2.0)
    cam_target = gymapi.Vec3(0.5, 0.3, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    # 获取力传感器张量
    gym.prepare_sim(sim)
    sensor_tensor = gym.acquire_force_sensor_tensor(sim)
    force_sensor_data = gymtorch.wrap_tensor(sensor_tensor)
    
    start_t = gym.get_sim_time(sim)
    last_t = start_t
    last_print_time = 0.0
    
    print("\nStarting simulation...")
    print(f"Initial panda_hand position: x={body_dict}, y=..., z=...")
    
    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        t = gym.get_sim_time(sim)
        dt = t - last_t
        last_t = t
        
        # 限时关闭
        if Config.TIMEOUT_ENABLED and (t - start_t) >= Config.TIMEOUT_SECONDS:
            print(f"\nTimeout reached: {Config.TIMEOUT_SECONDS:.1f}s. Exiting simulation loop.")
            break
        
        # 刷新力传感器数据
        gym.refresh_force_sensor_tensor(sim)
        
        # 传感器校准（仅执行一次）
        if not force_sensor_system.calibration_done and \
           t >= (Config.CAPTURE_START_TIME - Config.SENSOR_CALIBRATION_TIME):
            sensor_reading_initial = force_sensor_data.view(-1, 6)[0]
            force_sensor_system.calibrate(sensor_reading_initial)
            print(f"\n[t={t:.2f}s] 传感器零点校准完成")
        
        # 实时打印位置信息
        if body_dict and t - last_print_time >= Config.PRINT_INTERVAL:
            last_print_time = t
            robot_props = gym.get_actor_rigid_body_states(envs[0], robot_handles[0], gymapi.STATE_POS)
            cube_up_props = gym.get_actor_rigid_body_states(envs[0], scene_data['cube_handles'][0], gymapi.STATE_POS)
            cube_down_props = gym.get_actor_rigid_body_states(envs[0], scene_data['cube_handles'][1], gymapi.STATE_POS)
            
            hand_idx = body_dict[Config.HAND_NAME]
            panda_hand_pose_data = robot_props['pose'][:][hand_idx]
            
            # 获取传感器数据
            sensor_idx = 0
            sensor_reading = force_sensor_data.view(-1, 6)[sensor_idx]
            force_x, force_y, force_z, torque_x, torque_y, torque_z = \
                force_sensor_system.read_and_calibrate(sensor_reading)
            force_magnitude = force_sensor_system.get_force_magnitude(force_x, force_y, force_z)
            
            cube_up_pos = cube_up_props['pose'][:][0]['p']
            cube_down_pos = cube_down_props['pose'][:][0]['p']
            
            current_phase_name = "idle"
            if plan_state['running'] and plan_state['phase_idx'] < len(plan_state['plan']):
                current_phase_name = plan_state['plan'][plan_state['phase_idx']]['name']
            
            print(f"\n[t={t:.2f}s] === 实时位置信息 === [阶段: {current_phase_name}]")
            print(f"panda_hand: x={panda_hand_pose_data['p']['x']:.4f}, "
                  f"y={panda_hand_pose_data['p']['y']:.4f}, "
                  f"z={panda_hand_pose_data['p']['z']:.4f}")
            print(f"cube_up:   x={cube_up_pos['x']:.4f}, y={cube_up_pos['y']:.4f}, z={cube_up_pos['z']:.4f}")
            print(f"cube_down: x={cube_down_pos['x']:.4f}, y={cube_down_pos['y']:.4f}, z={cube_down_pos['z']:.4f}")
            print(f"force sensor: |F|={force_magnitude:.4f} N")
        
        # 启动动作规划
        if (not plan_state['running']) and t >= plan_state['start_time']:
            plan_state['running'] = True
            plan_state['current_time'] = t
            plan_state['dt'] = dt
            plan_state = attractor.update_pick_and_place(
                gym, viewer, envs, attractor_handles,
                scene_data['axes_geom'], scene_data['sphere_geom'],
                plan_state, scene_data['finger_dof_indices'],
                robot_handles, scene_data['mids'],
                body_dict=body_dict, hand_name=Config.HAND_NAME
            )
        elif plan_state['running']:
            plan_state['current_time'] = t
            plan_state['dt'] = dt
            plan_state = attractor.update_pick_and_place(
                gym, viewer, envs, attractor_handles,
                scene_data['axes_geom'], scene_data['sphere_geom'],
                plan_state, scene_data['finger_dof_indices'],
                robot_handles, scene_data['mids'],
                body_dict=body_dict, hand_name=Config.HAND_NAME
            )
        
        # 执行物理模拟
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 可视化
        if Config.VISUALIZE_AXES:
            visualization.draw_base_coordinate_frame(gym, viewer, envs[0])
            
            robot_props = gym.get_actor_rigid_body_states(envs[0], robot_handles[0], gymapi.STATE_POS)
            left_finger_idx = body_dict['panda_leftfinger']
            right_finger_idx = body_dict['panda_rightfinger']
            left_finger = robot_props['pose'][:][left_finger_idx]
            right_finger = robot_props['pose'][:][right_finger_idx]
            
            visualization.draw_fingertip_markers(gym, viewer, envs[0], left_finger, right_finger)
            
            # 绘制摄像头坐标系
            for cam_config in Config.CAMERAS:
                camera.draw_camera_axes_single(
                    gym, viewer, envs[0], scene_data['camera_axes_geom'],
                    cam_config["pos"],
                    cam_config["rotation_axis"],
                    cam_config["rotation_angle"],
                    cam_config.get("rotation_axis2"),
                    cam_config.get("rotation_angle2", 0),
                )
        
        # 更新图形
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)
        
        # 摄像头拍摄逻辑
        if planner.should_capture_frame(t, camera_system):
            sensor_reading = force_sensor_data.view(-1, 6)[0]
            force_x, force_y, force_z, torque_x, torque_y, torque_z = \
                force_sensor_system.read_and_calibrate(sensor_reading)
            
            # 保存传感器数据
            sensor.save_gel_sensor_data(
                force_x, force_y, force_z, torque_x, torque_y, torque_z,
                output_dir=f"{camera_system['output_dir']}/gel",
                capture_count=camera_system['capture_count']
            )
            
            # 保存摄像头图像
            camera.render_and_save_camera_images(
                gym, sim, envs, camera_handles,
                output_dir=camera_system['output_dir'],
                capture_count=camera_system['capture_count']
            )
            
            planner.log_capture_progress(camera_system['capture_count'], t)
            camera_system = planner.update_camera_capture_time(camera_system)
            if stop_when_capture_done and camera_system['capture_count'] >= camera_system['total_frames']:
                break
    
    print("\nSimulation completed.")
    return camera_system


def main():
    """主程序入口"""
    # 初始化 Isaac Gym
    gym = gymapi.acquire_gym()
    
    # 解析命令行参数
    args = gymutil.parse_arguments(description="Franka Attractor Example")
    
    # 初始化模拟环境
    sim, viewer = simulation.initialize_simulation_env(gym, args)
    
    try:
        # 设置场景
        scene_data = setup_scene(gym, sim, viewer)
        
        # 初始化各个系统
        systems_data = initialize_systems(gym, sim, scene_data, viewer)
        
        # 运行主循环
        run_main_loop(gym, sim, viewer, scene_data, systems_data)
        
    finally:
        # 清理资源
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        print("Cleanup completed.")


if __name__ == "__main__":
    main()
