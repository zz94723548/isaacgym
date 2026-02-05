import h5py
import numpy as np
import os

os.makedirs('data/data_dir/data', exist_ok=True)
print("✅ 目录创建成功！")

print(f"目标目录: {os.path.abspath('data/data_dir/data')}")

# 生成 5 个测试 episode
for ep_id in range(5):
    filename = f'data/data_dir/data/episode_{ep_id}.hdf5'
    print(f"正在创建: {filename}")
    
    with h5py.File(filename, 'w') as f:
        # 属性
        f.attrs['sim'] = True
        f.attrs['image_height'] = 480
        f.attrs['image_width'] = 640
        f.attrs['gelsight_height'] = 32
        f.attrs['gelsight_width'] = 32
        f.attrs['episode_id'] = ep_id
        
        # 数据 (100 timesteps)
        num_timesteps = 100
        f.attrs['num_timesteps'] = num_timesteps
        
        # 关节位置 (4维：末端位置变化XYZ + 夹爪开合度)当前位置
        f.create_dataset('observations/qpos', 
                        data=np.random.randn(num_timesteps, 4).astype(np.float32))
        
        # 相机RGB图像
        f.create_dataset('observations/images/realsence1', 
                        data=np.random.randint(0, 255, 
                                              (num_timesteps, 480, 640, 3), 
                                              dtype=np.uint8))
        
        # GelSight触觉传感器的深度应变图像
        f.create_dataset('observations/gelsight/depth_strain_image',
                        data=np.random.randn(num_timesteps, 32, 32, 3).astype(np.float32))
        
        # 动作数据 (4维：末端位置变化XYZ + 夹爪开合度)目标位置
        f.create_dataset('action', 
                        data=np.random.randn(num_timesteps, 4).astype(np.float32))
        
        # 可选：添加时间戳
        f.create_dataset('timestamp', 
                        data=np.arange(num_timesteps).astype(np.float32))
        
        print(f"    创建了 {num_timesteps} 个时间步的数据")
        
    print(f"  ✅ 完成 episode_{ep_id}")

print("✅ 测试数据生成完成！")
print(f"共生成 {len(os.listdir('data/data_dir/data'))} 个文件")

# 验证文件大小
print("\n文件大小统计:")
for ep_id in range(5):
    filename = f'data/data_dir/data/episode_{ep_id}.hdf5'
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"  {filename}: {size_mb:.2f} MB")