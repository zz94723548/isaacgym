"""
夹爪控制系统
===========
负责夹爪的开合控制
"""

from isaacgym import gymapi
from config import SimulationConfig as Config


def command_gripper(gym, envs, robot_handles, finger_dof_indices, target_width, base_dof_pos):
    """控制夹爪开合
    
    设置夹爪指头的目标位置，实现开合动作
    
    Args:
        gym: Isaac Gym 对象
        envs: 环境列表
        robot_handles: 机器人句柄列表
        finger_dof_indices: 夹爪指头关节索引列表 [left_finger_dof, right_finger_dof]
        target_width: 目标宽度（两指张开的总宽度）
        base_dof_pos: 基础DOF位置（未使用但保留接口兼容性）
    """
    # 添加最小间隙，避免目标为0时的硬碰撞导致非对称回弹
    min_gap = Config.GRIPPER_MIN_GAP
    single_finger_pos = max(target_width * 0.5, min_gap * 0.5)
    
    for env, handle in zip(envs, robot_handles):
        # 获取当前DOF目标，只修改夹爪，不干扰手臂关节
        dof_states = gym.get_actor_dof_states(env, handle, gymapi.STATE_NONE)
        targets = dof_states['pos'].copy()
        
        # 手动同步两个指头（Isaac Gym不自动处理URDF mimic约束）
        targets[finger_dof_indices[0]] = single_finger_pos
        targets[finger_dof_indices[1]] = single_finger_pos
        
        gym.set_actor_dof_position_targets(env, handle, targets)
