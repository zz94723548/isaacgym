import h5py
import numpy as np
import os
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ─────────────────────────────────────────────────────────────────────────────
# VITaL 格式应变图转换（参照论文 arXiv:2403.11898 Fig.5）
#   通道定义（LAB 色彩空间）:
#     channel 0 → 法向应变（深度）= Fz   → L（亮度）
#     channel 1 → x 切向应变      = Fx   → B（蓝-黄色谱）
#     channel 2 → y 切向应变      = Fy   → A（红-绿色谱）
# ─────────────────────────────────────────────────────────────────────────────

def gel_vector_to_strain_map(gel_vec, h=32, w=32, scale=1e4):
    """
    将单帧六轴力矩向量 [Fx, Fy, Fz, Tx, Ty, Tz] 映射为
    VITaL 格式空间应变图 (H, W, 3), float32。

    输出通道:
        [0] normal_strain = Fz + Tx*dy - Ty*dx  （法向/深度）
        [1] x_strain      = Fx + Tz*dy           （x 切向）
        [2] y_strain      = Fy - Tz*dx           （y 切向）
    dx, dy ∈ [-1,1] 为归一化像素坐标，力矩产生线性梯度分布。
    """
    v = np.asarray(gel_vec, dtype=np.float64) * scale
    if v.shape[0] != 6:
        raise ValueError(f"gel 向量维度应为 6，实际为 {v.shape[0]}")
    Fx, Fy, Fz, Tx, Ty, Tz = v

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    dy = (ys - cy) / (h / 2.0)
    dx = (xs - cx) / (w / 2.0)

    normal_strain = Fz + Tx * dy - Ty * dx
    x_strain      = Fx + Tz * dy
    y_strain      = Fy - Tz * dx

    return np.stack([normal_strain, x_strain, y_strain], axis=-1).astype(np.float32)


def gel_sequence_to_depth_strain_images(gel_data, h=32, w=32):
    """将 (T, 6) gel 序列映射为 (T, H, W, 3) VITaL 格式应变图序列。"""
    gel_data = np.asarray(gel_data, dtype=np.float32)
    if gel_data.ndim != 2 or gel_data.shape[1] != 6:
        raise ValueError(f"gel_data 期望形状为 (T, 6)，实际为 {gel_data.shape}")

    imgs = [gel_vector_to_strain_map(g, h=h, w=w) for g in gel_data]
    return np.stack(imgs, axis=0).astype(np.float32)

def process_single_episode(episode_id, input_base_dir, output_dir):
    """
    处理单条轨迹数据，将其转换为HDF5格式
    
    Parameters:
    -----------
    episode_id : int
        轨迹ID (0-99)
    input_base_dir : str
        输入数据的根目录 (如: /path/to/camera_outputs_0)
    output_dir : str
        输出HDF5文件的目录
    """
    
    print(f"\n处理 Episode {episode_id}...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 输入文件夹路径
    action_dir = os.path.join(input_base_dir, 'actions')
    camera_0_dir = os.path.join(input_base_dir, 'camera_0')
    camera_1_dir = os.path.join(input_base_dir, 'camera_1')
    camera_2_dir = os.path.join(input_base_dir, 'camera_2')
    camera_3_dir = os.path.join(input_base_dir, 'camera_3')
    camera_4_dir = os.path.join(input_base_dir, 'camera_4')
    gel_dir = os.path.join(input_base_dir, 'gel')
    
    # 验证所有输入文件夹存在
    input_dirs = [action_dir, camera_0_dir, camera_1_dir, camera_2_dir, camera_3_dir, camera_4_dir, gel_dir]
    for dir_path in input_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ 文件夹不存在: {dir_path}")
            return False
    
    # 读取action数据 (180 timesteps, 每条是末端位置+夹爪开度)
    print(f"  读取action数据...")
    action_files = sorted([f for f in os.listdir(action_dir) if f.endswith('.npy')])
    if len(action_files) != 180:
        print(f"  ⚠️  警告: action文件数量不是180，而是{len(action_files)}")
    
    action_data = []
    for action_file in action_files:
        action_path = os.path.join(action_dir, action_file)
        action = np.load(action_path)  # 1D数组 (4,): [x, y, z, gripper]
        action_data.append(action)
    action_data = np.array(action_data).astype(np.float32)  # (180, 4)
    
    # 读取5个摄像头数据
    print(f"  读取摄像头数据...")
    camera_dirs = [camera_0_dir, camera_1_dir, camera_2_dir, camera_3_dir, camera_4_dir]    
    from PIL import Image
    camera_names = ['realsence1', 'realsence2', 'realsence3', 'realsence4', 'realsence5']
    camera_data = {}
    
    for cam_name, cam_dir in zip(camera_names, camera_dirs):
        image_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.png')])
        if len(image_files) != 180:
            print(f"  ⚠️  警告: {cam_name}文件数量不是180，而是{len(image_files)}")
        
        images = []
        for img_file in image_files:
            img_path = os.path.join(cam_dir, img_file)
            img = np.array(Image.open(img_path).convert('RGB'))  # (480, 640, 3) RGB格式
            images.append(img)
        camera_data[cam_name] = np.array(images).astype(np.uint8)  # (180, H, W, 3)
    
    # 读取gel数据
    print(f"  读取gel数据...")
    gel_files = sorted([f for f in os.listdir(gel_dir) if f.endswith('.npy')])
    if len(gel_files) != 180:
        print(f"  ⚠️  警告: gel文件数量不是180，而是{len(gel_files)}")
    
    gel_data = []
    for gel_file in gel_files:
        gel_path = os.path.join(gel_dir, gel_file)
        gel = np.load(gel_path)  # 1D数组 (6,): gel传感器数据
        gel_data.append(gel)
    gel_data = np.array(gel_data).astype(np.float32)  # (180, 6)

    # 将原始 gel 向量转换为 GelSight 深度应变图 (T, 32, 32, 3)
    gelsight_depth_strain = gel_sequence_to_depth_strain_images(gel_data, h=32, w=32)
    
    # 创建HDF5文件
    output_file = os.path.join(output_dir, f'episode_{episode_id}.hdf5')
    print(f"  保存到HDF5文件: {output_file}")
    
    num_timesteps = action_data.shape[0]

    with h5py.File(output_file, 'w') as f:
        # 保存属性信息
        f.attrs['episode_id'] = episode_id
        f.attrs['num_timesteps'] = num_timesteps
        f.attrs['action_dim'] = action_data.shape[1]
        f.attrs['image_height'] = 480
        f.attrs['image_width'] = 640
        f.attrs['gelsight_height'] = 32
        f.attrs['gelsight_width'] = 32
        f.attrs['sim'] = False

        # 顶层 action 数据集（末端目标位置 + 夹爪开合度）
        f.create_dataset('action', data=action_data)

        # 时间戳
        f.create_dataset('timestamp',
                         data=np.arange(num_timesteps).astype(np.float32))

        # 保存observations - qpos 当前末端状态
        f.create_dataset('observations/qpos', data=action_data)

        # 保存observations - 摄像头 RGB 图像 (180, 480, 640, 3)
        for cam_name, images in camera_data.items():
            f.create_dataset(f'observations/images/{cam_name}', data=images)

        # 保存observations - gel 原始六轴力矩 (180, 6)
        f.create_dataset('observations/gel', data=gel_data)

        # 保存observations - GelSight 深度应变图 VITaL 格式 (180, 32, 32, 3) float32
        f.create_dataset(
            'observations/gelsight/depth_strain_image',
            data=gelsight_depth_strain
        )
    
    # 验证HDF5文件
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  ✅ Episode {episode_id} 完成 | 文件大小: {file_size_mb:.2f} MB")
    
    return True


def batch_process_episodes(input_base_path, output_base_dir, num_episodes=100):
    """
    批量处理所有轨迹数据
    
    Parameters:
    -----------
    input_base_path : str
        输入数据的基础路径 (如: /path/to 包含 camera_outputs_0, camera_outputs_1, ...)
    output_base_dir : str
        输出目录
    num_episodes : int
        总轨迹数
    """
    
    print(f"开始批量处理 {num_episodes} 条轨迹...")
    successful = 0
    failed = 0
    
    if tqdm is not None:
        ep_iter = tqdm(range(num_episodes), desc="Converting episodes", unit="ep")
    else:
        ep_iter = range(num_episodes)

    for ep_id in ep_iter:
        input_episode_dir = os.path.join(input_base_path, f'camera_outputs_{ep_id}')
        
        if not os.path.exists(input_episode_dir):
            print(f"❌ Episode {ep_id}: 目录不存在 {input_episode_dir}")
            failed += 1
            continue
        
        try:
            if process_single_episode(ep_id, input_episode_dir, output_base_dir):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Episode {ep_id}: 处理失败 - {str(e)}")
            failed += 1

        if tqdm is None:
            print(f"进度: {ep_id + 1}/{num_episodes} | 成功: {successful} | 失败: {failed}")
    
    print(f"\n✅ 批量处理完成!")
    print(f"   成功: {successful}, 失败: {failed}")


if __name__ == '__main__':
    # 修改这些路径为实际路径
    INPUT_BASE_PATH = '/media/neuzz/HLX/zz/DataSet'  # 实际数据路径
    OUTPUT_DIR = '/media/neuzz/HLX/zz/DataSet_HDF5'  # 输出HDF5文件的目录
    
    # 先处理第一条轨迹测试
    print("=" * 60)
    print("先处理第一条轨迹进行测试")
    print("=" * 60)
    process_single_episode(
        episode_id=0,
        input_base_dir=os.path.join(INPUT_BASE_PATH, 'camera_outputs_0'),
        output_dir=OUTPUT_DIR
    )
    
    # 确认无误后，取消注释下面的代码进行批量处理
    print("\n" + "=" * 60)
    print("开始批量处理所有轨迹")
    print("=" * 60)
    batch_process_episodes(INPUT_BASE_PATH, OUTPUT_DIR, num_episodes=100)
