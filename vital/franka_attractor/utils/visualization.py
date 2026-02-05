"""
可视化工具函数
=============
包含坐标系绘制、可视化标记等函数
"""

import math
from isaacgym import gymapi, gymutil
from config import SimulationConfig as Config


def draw_coordinate_frame(gym, viewer, env, pos, quat, size=0.1, color=(1, 1, 1)):
    """绘制通用坐标系框架
    
    Args:
        gym: Isaac Gym 对象
        viewer: 查看器对象
        env: 环境对象
        pos: 位置 (x, y, z)
        quat: 旋转四元数
        size: 坐标轴长度
        color: 颜色 (R, G, B)
    """
    axes_geom = gymutil.AxesGeometry(size)
    transform = gymapi.Transform()
    transform.p = gymapi.Vec3(pos[0], pos[1], pos[2])
    transform.r = quat
    gymutil.draw_lines(axes_geom, gym, viewer, env, transform)


def draw_base_coordinate_frame(gym, viewer, env):
    """绘制机器人基座坐标系
    
    Args:
        gym: Isaac Gym 对象
        viewer: 查看器对象
        env: 环境对象
    """
    if not Config.VISUALIZE_AXES:
        return
        
    base_axes_geom = gymutil.AxesGeometry(Config.BASE_AXES_SIZE)
    base_transform = gymapi.Transform()
    base_transform.p = gymapi.Vec3(0.0, 0.0, 0.0)
    base_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    gymutil.draw_lines(base_axes_geom, gym, viewer, env, base_transform)


def draw_hand_coordinate_frame(gym, viewer, env, hand_pose):
    """绘制手部坐标系
    
    Args:
        gym: Isaac Gym 对象
        viewer: 查看器对象
        env: 环境对象
        hand_pose: 手部姿态 (Transform或字典)
    """
    if not Config.VISUALIZE_AXES:
        return
        
    hand_axes_geom = gymutil.AxesGeometry(Config.HAND_AXES_SIZE)
    
    if isinstance(hand_pose, dict):
        # 字典格式
        hand_transform = gymapi.Transform()
        hand_transform.p = gymapi.Vec3(
            hand_pose['p']['x'],
            hand_pose['p']['y'],
            hand_pose['p']['z']
        )
        hand_transform.r = gymapi.Quat(
            hand_pose['r']['x'],
            hand_pose['r']['y'],
            hand_pose['r']['z'],
            hand_pose['r']['w']
        )
    else:
        # Transform格式
        hand_transform = hand_pose
    
    gymutil.draw_lines(hand_axes_geom, gym, viewer, env, hand_transform)


def draw_fingertip_markers(gym, viewer, env, left_finger_pose, right_finger_pose):
    """绘制指尖标记球
    
    左指尖绿色，右指尖红色（有传感器）
    
    Args:
        gym: Isaac Gym 对象
        viewer: 查看器对象
        env: 环境对象
        left_finger_pose: 左指尖姿态
        right_finger_pose: 右指尖姿态
    """
    if not Config.VISUALIZE_AXES:
        return
    
    from utils.math_utils import compute_fingertip_position
    
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    
    # 计算真实指尖位置
    left_true_tip = compute_fingertip_position(left_finger_pose, Config.SENSOR_FINGERTIP_OFFSET_Z)
    right_true_tip = compute_fingertip_position(right_finger_pose, Config.SENSOR_FINGERTIP_OFFSET_Z)
    
    # 绘制坐标系
    true_tip_axes_geom = gymutil.AxesGeometry(Config.FINGERTIP_AXES_SIZE)
    
    left_true_tip_tf = gymapi.Transform(
        p=gymapi.Vec3(left_true_tip[0], left_true_tip[1], left_true_tip[2]),
        r=gymapi.Quat(
            left_finger_pose['r']['x'],
            left_finger_pose['r']['y'],
            left_finger_pose['r']['z'],
            left_finger_pose['r']['w']
        )
    )
    right_true_tip_tf = gymapi.Transform(
        p=gymapi.Vec3(right_true_tip[0], right_true_tip[1], right_true_tip[2]),
        r=gymapi.Quat(
            right_finger_pose['r']['x'],
            right_finger_pose['r']['y'],
            right_finger_pose['r']['z'],
            right_finger_pose['r']['w']
        )
    )
    
    gymutil.draw_lines(true_tip_axes_geom, gym, viewer, env, left_true_tip_tf)
    gymutil.draw_lines(true_tip_axes_geom, gym, viewer, env, right_true_tip_tf)
    
    # 绘制标记球
    left_marker_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, sphere_pose, color=(0, 1, 0))
    right_marker_geom = gymutil.WireframeSphereGeometry(0.015, 12, 12, sphere_pose, color=(1, 0, 0))
    
    left_tip_marker_tf = gymapi.Transform(p=gymapi.Vec3(left_true_tip[0], left_true_tip[1], left_true_tip[2]))
    right_tip_marker_tf = gymapi.Transform(p=gymapi.Vec3(right_true_tip[0], right_true_tip[1], right_true_tip[2]))
    
    gymutil.draw_lines(left_marker_geom, gym, viewer, env, left_tip_marker_tf)
    gymutil.draw_lines(right_marker_geom, gym, viewer, env, right_tip_marker_tf)


def draw_attractor_visualization(gym, viewer, env, pose, axes_geom, sphere_geom):
    """绘制吸引子的坐标系和球体标记
    
    Args:
        gym: Isaac Gym 对象
        viewer: 查看器对象
        env: 环境对象
        pose: 吸引子姿态 (Transform)
        axes_geom: 坐标轴几何体
        sphere_geom: 球体几何体
    """
    gymutil.draw_lines(axes_geom, gym, viewer, env, pose)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, pose)
