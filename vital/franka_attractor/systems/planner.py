"""
动作规划系统
===========
负责生成抓取-放置动作序列和规划管理
"""

from isaacgym import gymapi
from config import SimulationConfig as Config


class MotionPlanner:
    """管理抓取-放置动作规划"""
    
    @staticmethod
    def build_pick_place_plan(initial_pose, cube_up_pos, cube_down_pos,
                             hover_offset=None, grasp_offset=None, release_offset=None):
        """构建抓取/放置动作规划
        
        Args:
            initial_pose: 初始吸引子姿态 (Transform)
            cube_up_pos: 上方立方体位置 (x, y, z)
            cube_down_pos: 下方立方体位置 (x, y, z)
            hover_offset: 悬停偏移量（米），如果为None使用配置值
            grasp_offset: 抓取偏移量（米），如果为None使用配置值
            release_offset: 放置偏移量（米），如果为None使用配置值
            
        Returns:
            list: 运动规划阶段列表，每个阶段包含运动目标和时间
        """
        if hover_offset is None:
            hover_offset = Config.MOTION_PLAN_HOVER_OFFSET
        if grasp_offset is None:
            grasp_offset = Config.MOTION_PLAN_GRASP_OFFSET
        if release_offset is None:
            release_offset = Config.MOTION_PLAN_RELEASE_OFFSET
        
        # 起始位置和姿态
        start_pos = (initial_pose.p.x, initial_pose.p.y, initial_pose.p.z)
        start_rot = initial_pose.r
        
        # 夹爪向下的旋转（绕X轴180度，使夹爪指向-Y方向）
        grasp_rot = gymapi.Quat(0.707106, 0.0, 0.0, 0.707106)
        
        # 各个关键位置
        hover_up = (cube_up_pos[0], cube_up_pos[1] + hover_offset, cube_up_pos[2])
        grasp_pos = (cube_up_pos[0], cube_up_pos[1] + grasp_offset, cube_up_pos[2])
        lift_pos = hover_up
        hover_down = (cube_down_pos[0], cube_down_pos[1] + hover_offset, cube_down_pos[2])
        place_pos = (cube_down_pos[0], cube_down_pos[1] + release_offset, cube_down_pos[2])
        retreat_pos = (cube_down_pos[0], cube_down_pos[1] + hover_offset + 0.05, cube_down_pos[2])
        
        print("Planning to pick and place the cube.")
        
        # 夹爪宽度
        finger_open = Config.GRIPPER_FINGER_OPEN
        finger_closed = Config.GRIPPER_FINGER_CLOSED
        
        # 构建动作序列
        plan = [
            {
                "name": "move_pregrasp",
                "start": start_pos,
                "goal": hover_up,
                "start_rot": start_rot,
                "goal_rot": grasp_rot,
                "duration": 2.0,
                "start_finger_width": finger_open,
                "goal_finger_width": finger_open
            },
            {
                "name": "descend_grasp",
                "start": hover_up,
                "goal": grasp_pos,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 2.0,
                "start_finger_width": finger_open,
                "goal_finger_width": finger_open
            },
            {
                "name": "close_gripper",
                "start": grasp_pos,
                "goal": grasp_pos,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 1.5,
                "start_finger_width": finger_open,
                "goal_finger_width": finger_closed
            },
            {
                "name": "stabilize_after_grasp",
                "start": grasp_pos,
                "goal": grasp_pos,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 0.5,
                "start_finger_width": finger_closed,
                "goal_finger_width": finger_closed
            },
            {
                "name": "lift",
                "start": grasp_pos,
                "goal": lift_pos,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 1.5,
                "start_finger_width": finger_closed,
                "goal_finger_width": finger_closed
            },
            {
                "name": "move_over_drop",
                "start": lift_pos,
                "goal": hover_down,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 3.0,
                "start_finger_width": finger_closed,
                "goal_finger_width": finger_closed
            },
            {
                "name": "stabilize_before_place",
                "start": hover_down,
                "goal": hover_down,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 0.5,
                "start_finger_width": finger_closed,
                "goal_finger_width": finger_closed
            },
            {
                "name": "place_release",
                "start": hover_down,
                "goal": place_pos,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 2.0,
                "start_finger_width": finger_closed,
                "goal_finger_width": finger_closed
            },
            {
                "name": "open_gripper",
                "start": place_pos,
                "goal": place_pos,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 1.5,
                "start_finger_width": finger_closed,
                "goal_finger_width": finger_open
            },
            {
                "name": "retreat",
                "start": place_pos,
                "goal": retreat_pos,
                "start_rot": grasp_rot,
                "goal_rot": grasp_rot,
                "duration": 1.5,
                "start_finger_width": finger_open,
                "goal_finger_width": finger_open
            },
        ]
        
        return plan


def initialize_camera_system(output_dir=None, capture_frequency=None,
                            capture_duration=None, start_time=None):
    """初始化摄像头系统参数
    
    Args:
        output_dir: 输出目录，如果为None使用配置值
        capture_frequency: 采集频率（Hz），如果为None使用配置值
        capture_duration: 采集总时长（秒），如果为None使用配置值
        start_time: 采集开始时间（秒），如果为None使用配置值
        
    Returns:
        dict: 摄像头系统配置字典
    """
    if output_dir is None:
        output_dir = Config.CAPTURE_OUTPUT_DIR
    if capture_frequency is None:
        capture_frequency = Config.CAPTURE_FREQUENCY
    if capture_duration is None:
        capture_duration = Config.CAPTURE_DURATION
    if start_time is None:
        start_time = Config.CAPTURE_START_TIME
        
    # 计算采集间隔
    capture_interval = 1.0 / capture_frequency
    
    # 计算总帧数
    total_frames = int(capture_duration * capture_frequency)
    
    return {
        'output_dir': output_dir,
        'capture_frequency': capture_frequency,
        'capture_interval': capture_interval,
        'capture_duration': capture_duration,
        'total_frames': total_frames,
        'start_time': start_time,
        'next_capture_time': start_time,
        'capture_count': 0
    }


def should_capture_frame(current_time, camera_system_data):
    """判断是否应该采集当前帧
    
    Args:
        current_time: 当前模拟时间
        camera_system_data: 摄像头系统数据
        
    Returns:
        bool: 是否应该采集当前帧
    """
    current_count = camera_system_data['capture_count']
    total_frames = camera_system_data['total_frames']
    next_capture_time = camera_system_data['next_capture_time']
    
    if current_count >= total_frames:
        return False
    
    if current_time >= next_capture_time:
        return True
    
    return False


def update_camera_capture_time(camera_system_data):
    """更新摄像头采集时间和计数
    
    Args:
        camera_system_data: 摄像头系统数据
        
    Returns:
        dict: 更新后的摄像头系统数据
    """
    camera_system_data['next_capture_time'] += camera_system_data['capture_interval']
    camera_system_data['capture_count'] += 1
    return camera_system_data


def log_capture_progress(capture_count, current_time, log_interval=None):
    """记录摄像头采集进度日志
    
    Args:
        capture_count: 采集计数
        current_time: 当前时间
        log_interval: 日志间隔，如果为None使用配置值
    """
    if log_interval is None:
        log_interval = Config.LOG_CAPTURE_INTERVAL
        
    if (capture_count + 1) % log_interval == 0:
        print(f"Captured {capture_count + 1} camera frames at {current_time:.2f}s")
