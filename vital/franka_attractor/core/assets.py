"""
资产加载模块
===========
负责加载机器人、工作台和物体等资产
"""

from isaacgym import gymapi
from config import SimulationConfig as Config


def create_asset_options(fix_base=True, flip_visual=True, armature=0.01):
    """创建资产加载选项
    
    Args:
        fix_base: 是否固定基座链接
        flip_visual: 是否翻转视觉附着物
        armature: 装甲参数（惯性）
        
    Returns:
        AssetOptions: 资产加载选项对象
    """
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fix_base
    asset_options.flip_visual_attachments = flip_visual
    asset_options.armature = armature
    return asset_options


def load_robot_asset(gym, sim, asset_root, asset_file):
    """加载Franka机器人资产
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
        asset_root: 资产文件根目录
        asset_file: 机器人模型文件相对路径
        
    Returns:
        Asset: 加载的机器人资产
    """
    print(f"Loading asset '{asset_file}' from '{asset_root}'")
    
    asset_options = create_asset_options(
        fix_base=Config.ASSET_FIX_BASE_LINK,
        flip_visual=Config.ASSET_FLIP_VISUAL,
        armature=Config.ASSET_ARMATURE
    )
    
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    if robot_asset is None:
        print(f"*** Failed to load asset: {asset_file}")
        quit()
    
    return robot_asset


def load_workbench_asset(gym, sim, asset_root, asset_file):
    """加载工作台资产
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
        asset_root: 资产文件根目录
        asset_file: 工作台模型文件相对路径
        
    Returns:
        Asset: 加载的工作台资产
    """
    print(f"Loading asset '{asset_file}' from '{asset_root}'")
    
    workbench_asset_options = gymapi.AssetOptions()
    workbench_asset_options.fix_base_link = True
    workbench_asset_options.density = Config.ASSET_DENSITY
    
    workbench_asset = gym.load_asset(sim, asset_root, asset_file, workbench_asset_options)
    
    if workbench_asset is None:
        print(f"*** Failed to load workbench asset: {asset_file}")
        quit()
    
    return workbench_asset


def load_cube_asset(gym, sim, asset_root, asset_file):
    """加载立方体资产
    
    Args:
        gym: Isaac Gym 对象
        sim: 模拟环境对象
        asset_root: 资产文件根目录
        asset_file: 立方体模型文件相对路径
        
    Returns:
        Asset: 加载的立方体资产
    """
    print(f"Loading asset '{asset_file}' from '{asset_root}'")
    
    cube_asset_options = gymapi.AssetOptions()
    cube_asset_options.fix_base_link = False
    cube_asset_options.density = Config.ASSET_DENSITY
    
    cube_asset = gym.load_asset(sim, asset_root, asset_file, cube_asset_options)
    
    if cube_asset is None:
        print(f"*** Failed to load cube asset: {asset_file}")
        quit()
    
    return cube_asset
