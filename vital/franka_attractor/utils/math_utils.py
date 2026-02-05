"""
数学工具函数
===========
包含向量插值、四元数插值等数学运算函数
"""

from isaacgym import gymapi


def lerp_vec(a, b, alpha):
    """向量线性插值
    
    Args:
        a: 起始向量 (x, y, z)
        b: 终止向量 (x, y, z)
        alpha: 插值参数，范围 [0, 1]
        
    Returns:
        tuple: 插值结果 (x, y, z)
    """
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


def slerp_quat(q1, q2, alpha):
    """四元数球面线性插值
    
    Args:
        q1: 起始四元数 (Quat)
        q2: 终止四元数 (Quat)
        alpha: 插值参数，范围 [0, 1]
        
    Returns:
        Quat: 插值结果四元数
    """
    # 计算点积
    dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w
    
    # 如果点积为负，反转一个四元数以选择最短路径
    if dot < 0.0:
        q2 = gymapi.Quat(-q2.x, -q2.y, -q2.z, -q2.w)
        dot = -dot
    
    # 线性插值（简化版，适用于小角度）
    result = gymapi.Quat(
        q1.x + (q2.x - q1.x) * alpha,
        q1.y + (q2.y - q1.y) * alpha,
        q1.z + (q2.z - q1.z) * alpha,
        q1.w + (q2.w - q1.w) * alpha
    )
    
    # 归一化
    norm = (result.x**2 + result.y**2 + result.z**2 + result.w**2)**0.5
    if norm > 0:
        result.x /= norm
        result.y /= norm
        result.z /= norm
        result.w /= norm
    
    return result


def compute_fingertip_position(finger_pose, offset_z=0.045):
    """计算指尖真实位置
    
    从指尖刚体原点沿其局部Z轴正方向偏移，得到真正的指尖位置
    
    Args:
        finger_pose: 刚体姿态字典，包含'p'(位置)和'r'(四元数旋转)
        offset_z: 沿局部Z轴的偏移距离(米)
        
    Returns:
        tuple: 真正指尖的世界坐标 (x, y, z)
    """
    # 获取刚体的位置和四元数
    pos = finger_pose['p']
    quat = finger_pose['r']
    
    # 提取四元数分量
    x, y, z, w = quat['x'], quat['y'], quat['z'], quat['w']
    
    # 计算旋转矩阵的第三列（局部Z轴方向）
    z_axis_x = 2 * (x * z + w * y)
    z_axis_y = 2 * (y * z - w * x)
    z_axis_z = 1 - 2 * (x * x + y * y)
    
    # 沿局部Z轴偏移
    true_tip_x = pos['x'] + offset_z * z_axis_x
    true_tip_y = pos['y'] + offset_z * z_axis_y
    true_tip_z = pos['z'] + offset_z * z_axis_z
    
    return (true_tip_x, true_tip_y, true_tip_z)
