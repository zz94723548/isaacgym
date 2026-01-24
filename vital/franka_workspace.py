"""
Franka Panda 机器人工作空间可视化
=====================================
通过采样关节空间并计算末端执行器位置，绘制机器人的三维工作空间。
"""

import math
import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil


# ===============================
# 第一部分：模拟环境初始化
# ===============================

def configure_sim_params(args):
    """配置物理模拟参数"""
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    
    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    
    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")
    
    return sim_params

def initialize_simulation_env(gym, args):
    """初始化模拟对象和查看器"""
    sim_params = configure_sim_params(args)
    
    sim = gym.create_sim(
        args.compute_device_id, 
        args.graphics_device_id, 
        args.physics_engine, 
        sim_params
    )
    
    if sim is None:
        print("*** Failed to create sim")
        quit()
    
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    
    return sim, viewer

def setup_lighting(gym, sim):
    """设置光照参数"""
    gym.set_light_parameters(
        sim, 0,
        gymapi.Vec3(0.5, 0.5, 0.5),
        gymapi.Vec3(0.8, 0.8, 0.8),
        gymapi.Vec3(0, -1, 0)
    )

def add_ground_plane(gym, sim):
    """添加地面平面"""
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)


# ===============================
# 第二部分：资产加载
# ===============================

def create_asset_options(fix_base=True, flip_visual=True, armature=0.01):
    """创建资产加载选项"""
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fix_base
    asset_options.flip_visual_attachments = flip_visual
    asset_options.armature = armature
    return asset_options

def load_robot_asset(gym, sim, asset_root, asset_file, asset_options):
    """加载机器人资产"""
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    if robot_asset is None:
        print("*** Failed to load asset: %s" % asset_file)
        quit()
    
    return robot_asset


# ===============================
# 第三部分：工作空间采样与计算
# ===============================

class FrankaWorkspaceAnalyzer:
    """Franka工作空间分析器"""
    
    def __init__(self, gym, sim, env, robot_handle, hand_name="panda_hand"):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.robot_handle = robot_handle
        self.hand_name = hand_name
        
        # 获取机器人信息
        self.body_dict = gym.get_actor_rigid_body_dict(env, robot_handle)
        
        # 关节范围（从URDF文件）
        self.joint_limits = {
            0: (-2.8973, 2.8973),    # Joint 1
            1: (-1.7628, 1.7628),    # Joint 2
            2: (-2.8973, 2.8973),    # Joint 3
            3: (-3.0718, -0.0698),   # Joint 4
            4: (-2.8973, 2.8973),    # Joint 5
            5: (-0.0175, 3.7525),    # Joint 6
            6: (-2.8973, 2.8973),    # Joint 7
        }
        
        self.end_effector_positions = []
        self.end_effector_orientations = []
    
    def sample_workspace(self, samples_per_joint=5):
        """
        采样工作空间
        
        args:
            samples_per_joint: 每个关节采样的点数
        """
        print(f"Sampling workspace with {samples_per_joint} samples per joint...")
        
        # 为每个关节生成采样角度
        joint_samples = []
        for i in range(7):
            lower, upper = self.joint_limits[i]
            samples = np.linspace(lower, upper, samples_per_joint)
            joint_samples.append(samples)
        
        # 生成所有关节配置的组合
        total_configs = samples_per_joint ** 7
        print(f"Total configurations to evaluate: {total_configs}")
        
        # 迭代遍历所有关节组合
        config_count = 0
        for j1 in joint_samples[0]:
            for j2 in joint_samples[1]:
                for j3 in joint_samples[2]:
                    for j4 in joint_samples[3]:
                        for j5 in joint_samples[4]:
                            for j6 in joint_samples[5]:
                                for j7 in joint_samples[6]:
                                    # 设置关节角度
                                    dof_states = self.gym.get_actor_dof_states(
                                        self.env, self.robot_handle, gymapi.STATE_NONE
                                    )
                                    dof_states['pos'][0] = j1
                                    dof_states['pos'][1] = j2
                                    dof_states['pos'][2] = j3
                                    dof_states['pos'][3] = j4
                                    dof_states['pos'][4] = j5
                                    dof_states['pos'][5] = j6
                                    dof_states['pos'][6] = j7
                                    # 夹爪保持在中间位置
                                    dof_states['pos'][7] = 0.02
                                    dof_states['pos'][8] = 0.02
                                    
                                    self.gym.set_actor_dof_states(
                                        self.env, self.robot_handle, dof_states, gymapi.STATE_POS
                                    )
                                    
                                    # 执行一步物理模拟以更新运动学
                                    self.gym.simulate(self.sim)
                                    self.gym.fetch_results(self.sim, True)
                                    
                                    # 获取末端执行器位置
                                    hand_states = self.gym.get_actor_rigid_body_states(
                                        self.env, self.robot_handle, gymapi.STATE_POS
                                    )
                                    hand_idx = self.body_dict[self.hand_name]
                                    hand_pose = hand_states['pose'][hand_idx]
                                    
                                    # 从numpy结构化数组中获取位置和旋转
                                    pos_array = hand_pose['p']
                                    rot_array = hand_pose['r']
                                    
                                    self.end_effector_positions.append([pos_array[0], pos_array[1], pos_array[2]])
                                    self.end_effector_orientations.append([rot_array[0], rot_array[1], rot_array[2], rot_array[3]])
                                    
                                    config_count += 1
                                    if config_count % 100 == 0:
                                        print(f"  Processed {config_count}/{total_configs} configurations")
        
        self.end_effector_positions = np.array(self.end_effector_positions)
        self.end_effector_orientations = np.array(self.end_effector_orientations)
        print(f"Workspace sampling complete! Total valid configurations: {len(self.end_effector_positions)}")
    
    def get_workspace_stats(self):
        """获取工作空间统计信息"""
        if len(self.end_effector_positions) == 0:
            print("No positions sampled yet")
            return
        
        pos = self.end_effector_positions
        
        print("\n=== Franka Panda 工作空间统计 ===")
        print(f"采样点数: {len(pos)}")
        print(f"\nX轴范围: {pos[:, 0].min():.4f} ~ {pos[:, 0].max():.4f} m")
        print(f"Y轴范围: {pos[:, 1].min():.4f} ~ {pos[:, 1].max():.4f} m")
        print(f"Z轴范围: {pos[:, 2].min():.4f} ~ {pos[:, 2].max():.4f} m")
        
        # 计算距基座的距离
        distances = np.linalg.norm(pos, axis=1)
        print(f"\n距基座距离范围: {distances.min():.4f} ~ {distances.max():.4f} m")
        print(f"平均可达距离: {distances.mean():.4f} m")
        
        # 计算工作空间体积（概估）
        volume = np.prod(pos.max(axis=0) - pos.min(axis=0))
        print(f"\n工作空间边界盒体积: {volume:.4f} m³")
    
    def visualize_workspace_points(self, gym, viewer, env):
        """在查看器中绘制工作空间点"""
        print("Rendering workspace points in viewer...")
        
        # 创建点的几何体
        for i, pos in enumerate(self.end_effector_positions):
            # 创建小球表示采样点
            sphere_pose = gymapi.Transform()
            sphere_pose.p = gymapi.Vec3(pos[0], pos[1], pos[2])
            
            # 根据Z轴高度调整颜色（热力图效果）
            z_min = self.end_effector_positions[:, 2].min()
            z_max = self.end_effector_positions[:, 2].max()
            z_norm = (pos[2] - z_min) / (z_max - z_min) if z_max > z_min else 0.5
            
            # 从蓝色（低）到红色（高）
            if z_norm < 0.5:
                color = (0, z_norm * 2, 1)  # 蓝到青
            else:
                color = (2 * (z_norm - 0.5), 1 - 2 * (z_norm - 0.5), 0)  # 绿到红
            
            # 创建小球几何体
            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, sphere_pose, color=color)
            gymutil.draw_lines(sphere_geom, gym, viewer, env, sphere_pose)
            
            if (i + 1) % 100 == 0:
                print(f"  Rendered {i + 1}/{len(self.end_effector_positions)} points")


# ===============================
# 第四部分：场景构建与可视化
# ===============================

def build_workspace_visualization_scene(gym, sim, viewer, robot_asset):
    """构建工作空间可视化场景"""
    
    # 创建环境
    env_lower = gymapi.Vec3(-1.0, 0.0, -1.0)
    env_upper = gymapi.Vec3(1.0, 1.0, 1.0)
    
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    # 创建机器人
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    
    robot_handle = gym.create_actor(env, robot_asset, pose, "franka", 0, 2)
    
    # 获取DOF属性并配置
    dof_props = gym.get_actor_dof_properties(env, robot_handle)
    
    # 设置所有关节为位置驱动模式（不施加力）
    dof_props['stiffness'].fill(0.0)
    dof_props['damping'].fill(0.0)
    
    gym.set_actor_dof_properties(env, robot_handle, dof_props)
    
    # 绘制基座坐标系
    base_axes_geom = gymutil.AxesGeometry(0.5)
    base_transform = gymapi.Transform()
    base_transform.p = gymapi.Vec3(0.0, 0.0, 0.0)
    base_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    gymutil.draw_lines(base_axes_geom, gym, viewer, env, base_transform)
    
    return env, robot_handle


# ===============================
# 主程序
# ===============================

def main():
    # 初始化 Isaac Gym
    gym = gymapi.acquire_gym()
    
    # 解析命令行参数
    args = gymutil.parse_arguments(description="Franka Workspace Visualization")
    
    # 初始化模拟环境
    sim, viewer = initialize_simulation_env(gym, args)
    
    # 设置光照
    setup_lighting(gym, sim)
    
    # 添加地面
    add_ground_plane(gym, sim)
    
    # 加载机器人资产
    asset_root = "../assets"
    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    asset_options = create_asset_options(fix_base=True, flip_visual=True, armature=0.01)
    franka_asset = load_robot_asset(gym, sim, asset_root, franka_asset_file, asset_options)
    
    # 构建场景
    print("\nBuilding workspace visualization scene...")
    env, robot_handle = build_workspace_visualization_scene(gym, sim, viewer, franka_asset)
    
    # 设置观察角度
    cam_pos = gymapi.Vec3(1.5, 1.5, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.5, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    # 创建工作空间分析器
    print("Initializing Franka workspace analyzer...")
    analyzer = FrankaWorkspaceAnalyzer(gym, sim, env, robot_handle, hand_name="panda_hand")
    
    # 采样工作空间（使用较少的样本以加快速度，可以调整）
    analyzer.sample_workspace(samples_per_joint=4)
    
    # 获取统计信息
    analyzer.get_workspace_stats()
    
    # 在查看器中绘制工作空间点
    analyzer.visualize_workspace_points(gym, viewer, env)
    
    # 主查看循环
    print("\nShowing workspace visualization. Close window to exit.")
    while not gym.query_viewer_has_closed(viewer):
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)
    
    # 清理资源
    print("Cleaning up...")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done!")


if __name__ == "__main__":
    main()
