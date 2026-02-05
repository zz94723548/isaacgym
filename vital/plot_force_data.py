"""
力传感器数据可视化脚本
读取所有gel传感器数据并绘制连续的时间序列图
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_all_force_data(gel_dir="./camera_outputs/gel"):
    """
    加载所有力传感器数据（从txt文件）
    
    Returns:
        forces: [N, 3] 数组，N个时间步的力数据 (fx, fy, fz)
        torques: [N, 3] 数组，N个时间步的力矩数据 (tx, ty, tz)
    """
    # 获取所有.txt文件并排序
    txt_files = sorted(glob.glob(f"{gel_dir}/*.txt"))
    
    if not txt_files:
        print(f"未找到数据文件在 {gel_dir}")
        return None, None
    
    print(f"找到 {len(txt_files)} 个数据文件")
    
    forces = []
    torques = []
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                # 格式: Force  (N):  x=0.081531, y=-0.073433, z=-0.253063
                for line in lines:
                    if line.startswith('Force  (N):'):
                        # 解析力数据
                        parts = line.split('x=')[1].split(', y=')
                        fx = float(parts[0])
                        parts = parts[1].split(', z=')
                        fy = float(parts[0])
                        fz = float(parts[1])
                        forces.append([fx, fy, fz])
                    elif line.startswith('Torque (Nm):'):
                        # 解析力矩数据
                        parts = line.split('x=')[1].split(', y=')
                        tx = float(parts[0])
                        parts = parts[1].split(', z=')
                        ty = float(parts[0])
                        tz = float(parts[1])
                        torques.append([tx, ty, tz])
        except Exception as e:
            print(f"  警告：无法读取 {txt_file}: {e}")
            continue
    
    return np.array(forces), np.array(torques)


def plot_force_timeline(forces, torques, output_file="force_timeline.png", fps=10):
    """
    绘制力和力矩的时间序列图
    
    Args:
        forces: [N, 3] 力数据
        torques: [N, 3] 力矩数据
        output_file: 输出文件名
        fps: 采样频率（用于计算时间轴）
    """
    num_frames = len(forces)
    time = np.arange(num_frames) / fps  # 时间轴（秒）
    
    # 计算力和力矩的大小
    force_magnitude = np.linalg.norm(forces, axis=1)
    torque_magnitude = np.linalg.norm(torques, axis=1)
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 子图1: 力的三个分量
    ax1 = axes[0]
    ax1.plot(time, forces[:, 0], 'r-', label='Force X', linewidth=1.5)
    ax1.plot(time, forces[:, 1], 'g-', label='Force Y', linewidth=1.5)
    ax1.plot(time, forces[:, 2], 'b-', label='Force Z', linewidth=1.5)
    ax1.plot(time, force_magnitude, 'k--', label='Total Force', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
    ax1.set_title('Right Fingertip Force Components', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 子图2: 力矩的三个分量
    ax2 = axes[1]
    ax2.plot(time, torques[:, 0], 'r-', label='Torque X', linewidth=1.5)
    ax2.plot(time, torques[:, 1], 'g-', label='Torque Y', linewidth=1.5)
    ax2.plot(time, torques[:, 2], 'b-', label='Torque Z', linewidth=1.5)
    ax2.plot(time, torque_magnitude, 'k--', label='Total Torque', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Torque (Nm)', fontsize=12, fontweight='bold')
    ax2.set_title('Right Fingertip Torque Components', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 子图3: 力和力矩的总大小对比
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(time, force_magnitude, 'b-', label='Force Magnitude', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Force Magnitude (N)', fontsize=12, fontweight='bold', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    
    line2 = ax3_twin.plot(time, torque_magnitude, 'r-', label='Torque Magnitude', linewidth=2)
    ax3_twin.set_ylabel('Torque Magnitude (Nm)', fontsize=12, fontweight='bold', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    ax3.set_title('Force and Torque Magnitudes Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    # 添加统计信息
    stats_text = f"Frames: {num_frames} | Duration: {time[-1]:.2f}s | Sampling: {fps}Hz\n"
    stats_text += f"Force - Max: {force_magnitude.max():.3f}N, Mean: {force_magnitude.mean():.3f}N, Min: {force_magnitude.min():.3f}N\n"
    stats_text += f"Torque - Max: {torque_magnitude.max():.4f}Nm, Mean: {torque_magnitude.mean():.4f}Nm, Min: {torque_magnitude.min():.4f}Nm"
    
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    # 保存图像
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图像已保存到: {output_file}")
    
    # 显示图像（可选）
    # plt.show()
    plt.close()


def main():
    gel_dir = "/media/neuzz/HLX/zz/camera_outputs_25/gel"
    
    # 加载数据
    print("正在加载力传感器数据...")
    forces, torques = load_all_force_data(gel_dir)
    
    if forces is None:
        return
    
    # 绘制图形
    print("正在生成时间序列图...")
    plot_force_timeline(forces, torques, output_file="force_timeline.png", fps=10)
    
    print("完成！")


if __name__ == "__main__":
    main()
