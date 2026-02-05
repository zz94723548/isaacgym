"""
I/O 工具函数
===========
包含数据保存、文件操作等输入输出相关的函数
"""

import os


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
