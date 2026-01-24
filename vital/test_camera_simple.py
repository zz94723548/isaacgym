"""
Franka 吸引子示例 - 摄像头集成测试版本
这是一个简化版本，用于测试摄像头功能
"""

import math
import numpy as np
import os
from isaacgym import gymapi
from isaacgym import gymutil
from PIL import Image

# 初始化
gym = gymapi.acquire_gym()      # 获取 Gym 接口，用于创建和管理模拟环境
args = gymutil.parse_arguments(description="Franka with Camera - Test")     #　解析命令行参数

# 配置模拟
sim_params = gymapi.SimParams()     # 设置物理参数
sim_params.dt = 1.0 / 60.0      # 时间步长（1/60秒==60帧每秒）
sim_params.substeps = 2     # 每个时间步的子步数（dt/2进行物理计算）

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1        # 设置 PhysX 求解器类型，1 代表 PGS（投影高斯-赛德尔），0 代表 TGS（张量高斯-赛德尔）
    sim_params.physx.num_position_iterations = 4        # 位置迭代次数
    sim_params.physx.num_velocity_iterations = 1        # 速度迭代次数
    sim_params.physx.use_gpu = args.use_gpu     # 是否使用 GPU 加速

# 创建模拟环境
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)  # compute_device_id 和 graphics_device_id 分别指定用于计算和渲染的设备 ID，默认为 0

if sim is None:
    print("*** Failed to create sim")
    quit()

# 创建查看器
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 添加地面
plane_params = gymapi.PlaneParams()     # 获取地面参数
gym.add_ground(sim, plane_params)       #　添加地面到模拟环境

# 加载机器人
asset_root = "../assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# 创建输出目录
output_dir = "camera_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ 创建输出目录: {output_dir}")

# 创建环境和机器人（只要1个）
num_envs = 1
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs = []
franka_handles = []
camera_handles = []

# 配置摄像头
camera_props = gymapi.CameraProperties()
camera_props.width = 640
camera_props.height = 480
camera_props.horizontal_fov = 75.0

num_per_row = 1

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 创建机器人
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0.0, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 2)
    franka_handles.append(franka_handle)

    # 创建摄像头
    camera_handle = gym.create_camera_sensor(env, camera_props)
    
    # 摄像头放在前上方，俯视基座
    camera_pose = gymapi.Transform()
    camera_pose.p = gymapi.Vec3(0.8, 1.0, 0.6)
    camera_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -math.pi / 3)
    gym.set_camera_transform(camera_handle, env, camera_pose)
    
    camera_handles.append(camera_handle)
    print(f"✓ 摄像头已创建在环境 {i}")

# 获取 DOF 属性
franka_dof_props = gym.get_actor_dof_properties(envs[0], franka_handles[0])
franka_lower_limits = franka_dof_props['lower']
franka_upper_limits = franka_dof_props['upper']
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props)

# 设置 DOF 属性
franka_dof_props['stiffness'].fill(1000.0)
franka_dof_props['damping'].fill(1000.0)
franka_dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS
franka_dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
franka_dof_props['stiffness'][7:] = 1e10
franka_dof_props['damping'][7:] = 1.0

for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)

# 初始化机器人姿态
for i in range(num_envs):
    franka_dof_states = gym.get_actor_dof_states(envs[i], franka_handles[i], gymapi.STATE_NONE)
    for j in range(franka_num_dofs):
        franka_dof_states['pos'][j] = franka_mids[j]
    gym.set_actor_dof_states(envs[i], franka_handles[i], franka_dof_states, gymapi.STATE_POS)

# 设置查看器摄像头
cam_pos = gymapi.Vec3(-4.0, 4.0, -1.0)
cam_target = gymapi.Vec3(0.0, 2.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("\n✓ 模拟环境初始化完成")
print("✓ 摄像头数量:", len(camera_handles))
print(f"✓ 输出目录: {os.path.abspath(output_dir)}")
print("\n运行模拟 (按 'q' 关闭窗口)...")

frame_count = 0
test_duration = 500  # 运行 8.3 秒（500 帧）给予充分的模拟时间
images_saved = False  # 标记是否已保存图像

while not gym.query_viewer_has_closed(viewer):
    # 执行物理模拟
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    
    # 获取摄像头图像
    frame_count += 1
    
    # 在第 250 帧时保存图像（给予充分的模拟时间）
    if frame_count == 250 and not images_saved:
        images_saved = True
        print(f"\n✓ 开始保存摄像头图像...\n")
        try:
            i = 0
            camera_handle = camera_handles[0]
            # 直接用官方接口写出文件，避免手动解码大小不匹配
            rgb_filename = os.path.join(output_dir, "camera_rgb.png")
            gym.write_camera_image_to_file(sim, envs[0], camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
            print(f"  ✓ 已保存: {rgb_filename}")

            # 为调试再读取一次，稳健解析形状
            rgb_image = gym.get_camera_image(sim, envs[0], camera_handle, gymapi.IMAGE_COLOR)
            rgb_array = np.array(rgb_image)

            expected_rgba = camera_props.height * camera_props.width * 4
            if rgb_array.ndim == 1 and rgb_array.size == expected_rgba:
                rgb_array = rgb_array.reshape(camera_props.height, camera_props.width, 4)[:, :, :3]
            elif rgb_array.ndim == 2 and rgb_array.shape[0] == camera_props.height:
                # 可能被平铺成 (H, W*4)
                if rgb_array.shape[1] == camera_props.width * 4:
                    rgb_array = rgb_array.reshape(camera_props.height, camera_props.width, 4)[:, :, :3]
            elif rgb_array.ndim == 3 and rgb_array.shape[2] >= 3:
                rgb_array = rgb_array[:, :, :3]
            else:
                print(f"  警告: 无法识别的RGB形状 {rgb_array.shape}，将填充黑图")
                rgb_array = np.zeros((camera_props.height, camera_props.width, 3), dtype=np.uint8)

            print(f"\n  摄像头图像信息:")
            print(f"    RGB 形状: {rgb_array.shape}, 数据类型: {rgb_array.dtype}")
            print(f"    RGB 值范围: [{rgb_array.min()}, {rgb_array.max()}]")
            print(f"    RGB 平均值: {rgb_array.mean():.2f}\n")

        except Exception as e:
            print(f"✗ 摄像头读取或保存失败: {e}")
    
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    
    # 如果运行时间足够，退出
    if frame_count >= test_duration:
        break

print("\n✓ 测试完成")
print("✓ 总帧数:", frame_count)

# 清理
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

print("✓ 程序正常退出")
