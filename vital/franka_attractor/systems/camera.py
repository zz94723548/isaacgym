"""
摄像头管理系统
=============
负责摄像头的创建、配置、坐标系绘制和图像保存
"""

import os
import math
import numpy as np
from isaacgym import gymapi, gymutil
from config import SimulationConfig as Config


def create_camera_sensor(gym, env, width=640, height=480,
                        pos=(0.0, 1.5, 0.0),
                        rotation_axis=(1, 0, 0),
                        rotation_angle=-90,
                        rotation_axis2=None,
                        rotation_angle2=0):
    """在环境中创建摄像头传感器
    
    Args:
        gym: Isaac Gym 对象
        env: 环境对象
        width: 摄像头宽度像素
        height: 摄像头高度像素
        pos: 摄像头位置 (x, y, z)
        rotation_axis: 主旋转轴 (x, y, z)
        rotation_angle: 主旋转角度（度）
        rotation_axis2: 次旋转轴，如果为None则无次旋转
        rotation_angle2: 次旋转角度（度）
        
    Returns:
        tuple: (camera_handle, camera_transform) - 摄像头句柄和变换
    """
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    
    camera_handle = gym.create_camera_sensor(env, camera_props)
    
    # 计算旋转
    primary_rot = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(*rotation_axis),
        math.radians(rotation_angle),
    )
    
    if rotation_axis2 is not None:
        secondary_rot = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(*rotation_axis2),
            math.radians(rotation_angle2),
        )
        camera_rot = secondary_rot * primary_rot
    else:
        camera_rot = primary_rot
    
    camera_transform = gymapi.Transform(
        p=gymapi.Vec3(*pos),
        r=camera_rot
    )
    
    gym.set_camera_transform(camera_handle, env, camera_transform)
    
    return camera_handle, camera_transform


def compute_camera_transform(pos, axis_primary, angle_primary, axis_secondary=None, angle_secondary=0):
    """计算摄像头变换
    
    Args:
        pos: 位置 (x, y, z)
        axis_primary: 主旋转轴
        angle_primary: 主旋转角度（度）
        axis_secondary: 次旋转轴，可选
        angle_secondary: 次旋转角度（度）
        
    Returns:
        Transform: 摄像头变换对象
    """
    primary_rot = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(axis_primary[0], axis_primary[1], axis_primary[2]),
        math.radians(angle_primary),
    )
    if axis_secondary is not None:
        secondary_rot = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(axis_secondary[0], axis_secondary[1], axis_secondary[2]),
            math.radians(angle_secondary),
        )
        rot = secondary_rot * primary_rot
    else:
        rot = primary_rot
    return gymapi.Transform(p=gymapi.Vec3(pos[0], pos[1], pos[2]), r=rot)


def create_eye_in_hand_camera(gym, env, hand_handle, width, height,
                             offset=(0.0, 0.0, 0.1),
                             rotation_axis_primary=(1, 0, 0),
                             rotation_angle_primary=-90,
                             rotation_axis_secondary=None,
                             rotation_angle_secondary=0):
    """创建并绑定在夹爪上的摄像头（眼在手上）
    
    Args:
        gym: Isaac Gym 对象
        env: 环境对象
        hand_handle: 手的刚体句柄
        width: 摄像头宽度像素
        height: 摄像头高度像素
        offset: 相对于手的偏移 (x, y, z)
        rotation_axis_primary: 主旋转轴
        rotation_angle_primary: 主旋转角度（度）
        rotation_axis_secondary: 次旋转轴
        rotation_angle_secondary: 次旋转角度（度）
        
    Returns:
        tuple: (camera_handle, mount_tf) - 摄像头句柄和安装变换
    """
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
    
    # 绑定摄像头到夹爪
    gym.attach_camera_to_body(
        camera_handle,
        env,
        hand_handle,
        mount_tf,
        gymapi.FOLLOW_TRANSFORM,
    )
    
    return camera_handle, mount_tf


def draw_camera_axes_single(gym, viewer, env, camera_axes_geom,
                           pos, rotation_axis_primary, rotation_angle_primary,
                           rotation_axis_secondary=None, rotation_angle_secondary=0):
    """绘制单个摄像头的坐标系
    
    Args:
        gym: Isaac Gym 对象
        viewer: 查看器对象
        env: 环境对象
        camera_axes_geom: 坐标轴几何体
        pos: 摄像头位置
        rotation_axis_primary: 主旋转轴
        rotation_angle_primary: 主旋转角度（度）
        rotation_axis_secondary: 次旋转轴
        rotation_angle_secondary: 次旋转角度（度）
    """
    cam_tf = compute_camera_transform(
        pos,
        rotation_axis_primary,
        rotation_angle_primary,
        rotation_axis_secondary,
        rotation_angle_secondary,
    )
    gymutil.draw_lines(camera_axes_geom, gym, viewer, env, cam_tf)


def setup_camera_output_directory(output_dir=None):
    """创建摄像头输出目录
    
    Args:
        output_dir: 输出目录路径，如果为None使用配置值
    """
    if output_dir is None:
        output_dir = Config.CAPTURE_OUTPUT_DIR
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Camera output directory created at {output_dir}")


def get_camera_rgb_image(gym, sim, env, camera_handle, render_first=False,
                         expected_width=None, expected_height=None):
    """从相机句柄读取单帧 RGB 图像（内存，不落盘）

    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
        env: 环境对象
        camera_handle: 摄像头句柄
        render_first: 是否在读取前先渲染所有相机
        expected_width: 期望图像宽度（用于一维数组重塑）
        expected_height: 期望图像高度（用于一维数组重塑）

    Returns:
        ndarray: uint8, 形状 (H, W, 3)
    """
    if render_first:
        gym.render_all_camera_sensors(sim)

    raw = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    img = np.asarray(raw)

    # 常见格式：H x W x 4 (RGBA)
    if img.ndim == 3:
        if img.shape[-1] >= 3:
            rgb = img[..., :3]
        else:
            raise ValueError(f"Unexpected camera image shape: {img.shape}")
    else:
        # 兜底：使用调用方提供的分辨率进行重塑（旧版 Isaac Gym 无 get_camera_properties）
        if expected_width is None or expected_height is None:
            # 默认采用项目主相机分辨率
            w = int(Config.CAMERAS[0]["width"])
            h = int(Config.CAMERAS[0]["height"])
        else:
            w = int(expected_width)
            h = int(expected_height)
        if img.size == h * w * 4:
            rgb = img.reshape(h, w, 4)[..., :3]
        elif img.size == h * w * 3:
            rgb = img.reshape(h, w, 3)
        else:
            raise ValueError(f"Cannot reshape camera image: size={img.size}, target=({h},{w},3/4)")

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def get_policy_camera_frames(gym, sim, env, camera_handles,
                             camera_indices=(0, 1),
                             camera_names=("realsence1", "realsence2"),
                             render_first=True):
    """读取在线推理使用的多路相机图像（内存，不落盘）

    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
        env: 环境对象
        camera_handles: 已创建的相机句柄列表
        camera_indices: 需要读取的相机索引（默认前两路）
        camera_names: 输出字典中的键名，需与模型输入名对齐
        render_first: 是否在读取前统一渲染

    Returns:
        dict: {camera_name: ndarray(H, W, 3), ...}
    """
    if len(camera_indices) != len(camera_names):
        raise ValueError("camera_indices and camera_names must have the same length")

    if render_first:
        gym.render_all_camera_sensors(sim)

    frames = {}
    for idx, name in zip(camera_indices, camera_names):
        if idx < 0 or idx >= len(camera_handles):
            raise IndexError(f"camera index out of range: {idx}, total={len(camera_handles)}")

        # 根据索引推断分辨率（前4路固定相机 + 最后一路手眼相机）
        if idx < len(Config.CAMERAS):
            exp_w = int(Config.CAMERAS[idx]["width"])
            exp_h = int(Config.CAMERAS[idx]["height"])
        else:
            exp_w = int(getattr(Config, "HAND_CAMERA_WIDTH", 640))
            exp_h = int(getattr(Config, "HAND_CAMERA_HEIGHT", 480))

        frames[name] = get_camera_rgb_image(
            gym, sim, env, camera_handles[idx], render_first=False,
            expected_width=exp_w, expected_height=exp_h,
        )

    return frames


def render_and_save_camera_images(gym, sim, envs, camera_handles,
                                  output_dir=None, capture_count=0):
    """渲染并保存摄像头图像
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
        envs: 环境列表
        camera_handles: 摄像头句柄列表
        output_dir: 输出目录，如果为None使用配置值
        capture_count: 采集计数
        
    Returns:
        int: 下一个采集计数
    """
    if output_dir is None:
        output_dir = Config.CAPTURE_OUTPUT_DIR
        
    gym.render_all_camera_sensors(sim)
    
    camera_idx = 0
    for env_idx in range(len(envs)):
        env = envs[env_idx]
        
        for camera_handle in camera_handles:
            camera_dir = f"{output_dir}/camera_{camera_idx}"
            if not os.path.exists(camera_dir):
                os.makedirs(camera_dir)
            
            rgb_filename = f"{camera_dir}/{capture_count:04d}.png"
            gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
            camera_idx += 1
    
    return capture_count + 1
