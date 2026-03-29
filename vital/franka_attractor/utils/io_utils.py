"""
I/O 工具函数
===========
包含数据保存、文件操作等输入输出相关的函数
"""

import os
import numpy as np


def ensure_directory_exists(directory_path):
    """确保目录存在，如果不存在则创建
    
    Args:
        directory_path: 目录路径
        
    Returns:
        str: 目录路径
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def get_frame_filename(directory, frame_number, extension=".png"):
    """生成带编号的文件名
    
    Args:
        directory: 目录路径
        frame_number: 帧号
        extension: 文件扩展名
        
    Returns:
        str: 完整文件路径
    """
    return f"{directory}/{frame_number:04d}{extension}"


def create_nested_directory(base_path, *subdirs):
    """创建嵌套目录结构
    
    Args:
        base_path: 基础路径
        *subdirs: 子目录名称
        
    Returns:
        str: 最终目录路径
    """
    path = base_path
    for subdir in subdirs:
        path = os.path.join(path, subdir)
    
    ensure_directory_exists(path)
    return path


def save_action_data(output_dir, capture_count, action_vector):
    """
    保存单帧动作向量到 output_dir/actions/<frame>.npy
    action_vector: iterable of 4 floats: [ax, ay, az, gripper_gap]
    """
    actions_dir = os.path.join(output_dir, "actions")
    os.makedirs(actions_dir, exist_ok=True)
    fname = os.path.join(actions_dir, f"{capture_count:04d}.npy")
    np.save(fname, np.asarray(action_vector, dtype=np.float32))


def save_gel_data(output_dir, capture_count, gel_vector, subdir="gel"):
    """
    保存单帧触觉向量到 output_dir/<subdir>/<frame>.npy
    gel_vector: iterable of 6 floats: [Fx, Fy, Fz, Tx, Ty, Tz]
    """
    gel_dir = os.path.join(output_dir, subdir)
    os.makedirs(gel_dir, exist_ok=True)
    fname = os.path.join(gel_dir, f"{capture_count:04d}.npy")
    np.save(fname, np.asarray(gel_vector, dtype=np.float32))
