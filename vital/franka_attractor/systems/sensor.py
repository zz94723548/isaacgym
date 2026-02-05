"""
力传感器管理系统
===============
负责力传感器数据的读取、校准和保存
"""

import os
import numpy as np
from config import SimulationConfig as Config


class ForceSensorSystem:
    """管理力传感器的校准和数据处理"""
    
    def __init__(self):
        self.zero_offset = None
        self.calibration_done = False
    
    def calibrate(self, sensor_reading):
        """进行零点校准
        
        Args:
            sensor_reading: 传感器读数（张量）
            
        Returns:
            ndarray: 零点偏移
        """
        self.zero_offset = sensor_reading.clone()
        self.calibration_done = True
        return self.zero_offset
    
    def read_and_calibrate(self, sensor_reading):
        """读取并校准传感器数据
        
        Args:
            sensor_reading: 传感器读数（张量，形状为[6]，包含Fx, Fy, Fz, Tx, Ty, Tz）
            
        Returns:
            tuple: (force_x, force_y, force_z, torque_x, torque_y, torque_z) - 校准后的值
        """
        if self.calibration_done:
            sensor_reading_calibrated = sensor_reading - self.zero_offset
        else:
            sensor_reading_calibrated = sensor_reading
        
        force_x = sensor_reading_calibrated[0].item()
        force_y = sensor_reading_calibrated[1].item()
        force_z = sensor_reading_calibrated[2].item()
        torque_x = sensor_reading_calibrated[3].item()
        torque_y = sensor_reading_calibrated[4].item()
        torque_z = sensor_reading_calibrated[5].item()
        
        return force_x, force_y, force_z, torque_x, torque_y, torque_z
    
    def get_force_magnitude(self, force_x, force_y, force_z):
        """计算力的大小
        
        Args:
            force_x, force_y, force_z: 力的三个分量
            
        Returns:
            float: 力的大小
        """
        return np.sqrt(force_x**2 + force_y**2 + force_z**2)
    
    def get_torque_magnitude(self, torque_x, torque_y, torque_z):
        """计算力矩的大小
        
        Args:
            torque_x, torque_y, torque_z: 力矩的三个分量
            
        Returns:
            float: 力矩的大小
        """
        return np.sqrt(torque_x**2 + torque_y**2 + torque_z**2)


def setup_gel_output_directory(output_dir=None):
    """创建触觉传感器输出目录
    
    Args:
        output_dir: 输出目录路径，如果为None使用默认值
    """
    if output_dir is None:
        output_dir = f"{Config.CAPTURE_OUTPUT_DIR}/gel"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Gel sensor output directory created at {output_dir}")


def save_gel_sensor_data(force_x, force_y, force_z, torque_x, torque_y, torque_z,
                        output_dir=None, capture_count=0):
    """保存触觉传感器数据为numpy和文本文件
    
    Args:
        force_x, force_y, force_z: 力的三个分量 (N)
        torque_x, torque_y, torque_z: 力矩的三个分量 (Nm)
        output_dir: 输出目录，如果为None使用默认值
        capture_count: 采集计数
    """
    if output_dir is None:
        output_dir = f"{Config.CAPTURE_OUTPUT_DIR}/gel"
        
    # 组织数据
    sensor_data = np.array([force_x, force_y, force_z, torque_x, torque_y, torque_z])
    
    # 保存为numpy二进制文件
    npy_filename = f"{output_dir}/{capture_count:04d}.npy"
    np.save(npy_filename, sensor_data)
    
    # 同时保存为文本文件（方便查看）
    txt_filename = f"{output_dir}/{capture_count:04d}.txt"
    force_magnitude = np.sqrt(force_x**2 + force_y**2 + force_z**2)
    torque_magnitude = np.sqrt(torque_x**2 + torque_y**2 + torque_z**2)
    
    with open(txt_filename, 'w') as f:
        f.write(f"Frame: {capture_count:04d}\n")
        f.write(f"Force  (N):  x={force_x:.6f}, y={force_y:.6f}, z={force_z:.6f}\n")
        f.write(f"Torque (Nm): x={torque_x:.6f}, y={torque_y:.6f}, z={torque_z:.6f}\n")
        f.write(f"Force Magnitude:  {force_magnitude:.6f} N\n")
        f.write(f"Torque Magnitude: {torque_magnitude:.6f} Nm\n")
