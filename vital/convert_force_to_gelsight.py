"""
将力传感器数据转换为模拟的GelSight深度应变图像
将6维力数据映射为32x32x3的深度应变图像
"""

import numpy as np
import glob
import os
from PIL import Image


def create_contact_gaussian(center_x, center_y, force_magnitude, image_size=32):
    """
    创建基于高斯分布的接触压力分布
    
    Args:
        center_x, center_y: 接触点中心位置（归一化到0-1）
        force_magnitude: 力的大小
        image_size: 输出图像尺寸
    
    Returns:
        pressure_map: (image_size, image_size) 压力分布图
    """
    # 创建网格
    x = np.linspace(0, 1, image_size)
    y = np.linspace(0, 1, image_size)
    X, Y = np.meshgrid(x, y)
    
    # 高斯分布参数（增大sigma使分布更宽）
    sigma = 0.15  # 接触区域大小
    
    # 计算高斯分布
    gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
    
    # 乘以力的大小并放大
    pressure_map = gaussian * force_magnitude * 2.0  # 增加倍数
    
    return pressure_map


def parse_force_from_txt(txt_file):
    """
    从txt文件解析力数据
    
    Format:
    Frame: XXXX
    Force  (N):  x=..., y=..., z=...
    Torque (Nm): x=..., y=..., z=...
    """
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        force_x = force_y = force_z = 0
        torque_x = torque_y = torque_z = 0
        
        for line in lines:
            if 'Force' in line and '(N)' in line:
                # 解析: Force  (N):  x=0.060224, y=-0.155365, z=-0.042712
                parts = line.split('x=')[1].split(',')
                force_x = float(parts[0])
                force_y = float(parts[1].split('y=')[1])
                force_z = float(parts[2].split('z=')[1])
            elif 'Torque' in line and '(Nm)' in line:
                # 解析: Torque (Nm): x=-0.006913, y=-0.002710, z=0.000120
                parts = line.split('x=')[1].split(',')
                torque_x = float(parts[0])
                torque_y = float(parts[1].split('y=')[1])
                torque_z = float(parts[2].split('z=')[1])
        
        return force_x, force_y, force_z, torque_x, torque_y, torque_z
    except:
        return 0, 0, 0, 0, 0, 0


def force_to_gelsight_image(force_x, force_y, force_z, torque_x, torque_y, torque_z, image_size=32):
    """
    将6维力传感器数据转换为GelSight深度应变图像
    
    Args:
        force_x/y/z: 力的三个分量 (N)
        torque_x/y/z: 力矩的三个分量 (Nm)
        image_size: 输出图像尺寸
    
    Returns:
        depth_strain_image: (image_size, image_size, 3) float32数组
    """
    # 确保输入是标量值
    force_x = float(np.asarray(force_x))
    force_y = float(np.asarray(force_y))
    force_z = float(np.asarray(force_z))
    torque_x = float(np.asarray(torque_x))
    torque_y = float(np.asarray(torque_y))
    torque_z = float(np.asarray(torque_z))
    
    # 计算力的大小
    force_magnitude = np.sqrt(force_x**2 + force_y**2 + force_z**2)
    
    # 即使没有接触力也生成图像（降低阈值）
    if force_magnitude < 0.0001:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)
    
    # 根据力矩估算接触点位置
    torque_magnitude = np.sqrt(torque_x**2 + torque_y**2)
    
    # 接触点中心位置
    center_x = 0.5 + np.clip(torque_y * 10, -0.3, 0.3)
    center_y = 0.5 - np.clip(torque_x * 10, -0.3, 0.3)
    
    # 生成压力分布
    depth_map = create_contact_gaussian(center_x, center_y, force_magnitude, image_size)
    
    depth_strain_image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    
    # 使用激进的非线性映射：平方根和指数增强
    # 注意：实际力数据很小（max ~0.17N），需要大幅放大
    # R通道：Z方向深度（垂直压力）
    abs_force_z = abs(force_z) + 0.05
    r_channel = np.sqrt(depth_map) * np.sqrt(abs_force_z) * 50.0
    depth_strain_image[:, :, 0] = np.clip(r_channel, 0, 1)
    
    # G通道：X方向应变（切向力）
    abs_force_x = abs(force_x) + 0.03
    x_strain = create_contact_gaussian(center_x, center_y, abs_force_x, image_size)
    g_channel = np.sqrt(x_strain) * np.sqrt(abs_force_x) * 30.0
    depth_strain_image[:, :, 1] = np.clip(g_channel, 0, 1)
    
    # B通道：Y方向应变（夹持力）
    abs_force_y = abs(force_y) + 0.03
    y_strain = create_contact_gaussian(center_x, center_y, abs_force_y, image_size)
    b_channel = np.sqrt(y_strain) * np.sqrt(abs_force_y) * 30.0
    depth_strain_image[:, :, 2] = np.clip(b_channel, 0, 1)
    
    return depth_strain_image


def convert_all_force_data(input_dir="./camera_outputs/gel", output_dir="./camera_outputs/gel_images", image_size=32):
    """
    批量转换所有力传感器数据为GelSight图像
    
    Args:
        input_dir: 输入的txt/npy文件目录 (camera_outputs/gel)
        output_dir: 输出的GelSight图像目录 (camera_outputs/gel_images)
        image_size: 图像尺寸
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有txt文件（优先txt，再试npy）
    txt_files = sorted(glob.glob(f"{input_dir}/*.txt"))
    npy_files = sorted(glob.glob(f"{input_dir}/*.npy"))
    
    # 使用txt文件（更可靠）
    if txt_files:
        data_files = txt_files
        is_txt = True
        print(f"找到 {len(txt_files)} 个力传感器数据文件 (txt格式)")
    elif npy_files:
        data_files = npy_files
        is_txt = False
        print(f"找到 {len(npy_files)} 个力传感器数据文件 (npy格式)")
    else:
        print(f"未找到力数据文件在 {input_dir}")
        return
    
    print(f"开始转换为GelSight深度应变图像...")
    print(f"输入位置: {input_dir}")
    print(f"输出位置: {output_dir}")
    print()
    
    # 显示样本数据
    if is_txt:
        sample_force_x, sample_force_y, sample_force_z, sample_torque_x, sample_torque_y, sample_torque_z = parse_force_from_txt(data_files[0])
    else:
        sample_data = np.load(data_files[0]).flatten()
        sample_force_x, sample_force_y, sample_force_z = float(sample_data[0]), float(sample_data[1]), float(sample_data[2])
        sample_torque_x, sample_torque_y, sample_torque_z = float(sample_data[3]), float(sample_data[4]), float(sample_data[5])
    
    print(f"样本数据 (Frame 0):")
    print(f"  Force: [{sample_force_x:.6f}, {sample_force_y:.6f}, {sample_force_z:.6f}] N")
    print(f"  Torque: [{sample_torque_x:.6f}, {sample_torque_y:.6f}, {sample_torque_z:.6f}] Nm")
    print()
    
    for idx, data_file in enumerate(data_files):
        # 读取力数据
        if is_txt:
            force_x, force_y, force_z, torque_x, torque_y, torque_z = parse_force_from_txt(data_file)
        else:
            force_data = np.load(data_file).flatten()
            force_x = float(force_data[0])
            force_y = float(force_data[1])
            force_z = float(force_data[2])
            torque_x = float(force_data[3])
            torque_y = float(force_data[4])
            torque_z = float(force_data[5])
        
        # 转换为GelSight图像
        gelsight_image = force_to_gelsight_image(
            force_x, force_y, force_z, 
            torque_x, torque_y, torque_z,
            image_size=image_size
        )
        
        # 保存为PNG格式
        basename = os.path.basename(data_file).replace('.txt', '.png').replace('.npy', '.png')
        output_file = os.path.join(output_dir, basename)
        
        # 归一化到0-255范围
        gelsight_image_normalized = np.clip(gelsight_image * 255, 0, 255).astype(np.uint8)
        
        # 保存为PNG
        img = Image.fromarray(gelsight_image_normalized, mode='RGB')
        img.save(output_file)
        
        # 显示进度
        if (idx + 1) % 20 == 0 or idx == len(data_files) - 1:
            print(f"  已完成: {idx + 1}/{len(data_files)}")
    
    print(f"\n✅ 转换完成！共处理 {len(data_files)} 个文件")
    print(f"输出位置: {output_dir}")
    print(f"图像格式: PNG (32x32)")


def main():
    print("=" * 60)
    print("力传感器数据 -> GelSight深度应变图像转换")
    print("=" * 60)
    print()
    
    # 转换数据：从gel读取，输出到gel_images
    convert_all_force_data(
        input_dir="./camera_outputs/gel",
        output_dir="./camera_outputs/gel_images",
        image_size=32
    )
    
    print("\n✅ 完成！GelSight图像已生成。")


if __name__ == "__main__":
    main()
