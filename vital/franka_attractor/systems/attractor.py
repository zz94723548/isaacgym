"""
吸引子控制系统
=============
负责吸引子位置的更新和可视化
"""

from isaacgym import gymapi, gymutil
from systems.gripper import command_gripper


def update_pick_and_place(gym, viewer, envs, attractor_handles, axes_geom, sphere_geom,
                         plan_state, finger_dof_indices, robot_handles, base_dof_pos,
                         body_dict=None, hand_name="panda_hand"):
    """更新抓取/放置吸引子位置和夹爪状态
    
    根据当前运动规划的阶段，更新吸引子的目标位置和夹爪的开合状态
    
    Args:
        gym: Isaac Gym 对象
        viewer: 查看器对象
        envs: 环境列表
        attractor_handles: 吸引子句柄列表
        axes_geom: 坐标轴几何体
        sphere_geom: 球体几何体
        plan_state: 规划状态字典
        finger_dof_indices: 夹爪关节索引列表
        robot_handles: 机器人句柄列表
        base_dof_pos: 基础DOF位置
        body_dict: 刚体字典
        hand_name: 末端执行器名称
        
    Returns:
        dict: 更新后的规划状态
    """
    if not plan_state['running']:
        return plan_state
    
    t = plan_state['current_time']
    dt = plan_state['dt']
    phase_idx = plan_state['phase_idx']
    
    if phase_idx >= len(plan_state['plan']):
        return plan_state  # 已完成
    
    gym.clear_lines(viewer)
    
    phase = plan_state['plan'][phase_idx]
    plan_state['phase_elapsed'] += dt
    
    # 线性插值计算当前目标位置和姿态
    from utils.math_utils import lerp_vec, slerp_quat
    
    alpha = min(plan_state['phase_elapsed'] / max(phase['duration'], 1e-4), 1.0)
    target_pos = lerp_vec(phase['start'], phase['goal'], alpha)
    target_rot = slerp_quat(phase['start_rot'], phase['goal_rot'], alpha)
    
    # 对夹爪宽度也进行线性插值
    target_finger_width = phase['start_finger_width'] + \
                         (phase['goal_finger_width'] - phase['start_finger_width']) * alpha
    
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


def create_attractor_visualization(gym, viewer, env, pose, axes_geom, sphere_geom):
    """创建吸引子可视化
    
    Args:
        gym: Isaac Gym 对象
        viewer: 查看器对象
        env: 环境对象
        pose: 吸引子姿态
        axes_geom: 坐标轴几何体
        sphere_geom: 球体几何体
    """
    gymutil.draw_lines(axes_geom, gym, viewer, env, pose)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, pose)
