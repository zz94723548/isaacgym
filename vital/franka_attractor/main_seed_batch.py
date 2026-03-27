"""
Franka 批量轨迹采集入口
=====================

按随机种子范围连续运行，生成多条轨迹。
为避免 Isaac Gym / GPU PhysX 在同一进程中反复创建和销毁模拟器导致段错误，
每个 seed 都在独立子进程中运行。

示例：
    python main_seed_batch.py --seed_start 0 --seed_end 99 --output_root /media/neuzz/HLX/zz/DataSet
"""

import subprocess
import sys
from pathlib import Path

from isaacgym import gymapi, gymutil

from config import SimulationConfig as Config
from core import simulation
from main import setup_scene, initialize_systems, run_main_loop


def run_single_trajectory(args, seed, output_dir):
    """运行单条轨迹采集。"""
    gym = gymapi.acquire_gym()
    Config.RANDOM_SEED = int(seed)
    Config.CAPTURE_OUTPUT_DIR = output_dir

    sim, viewer = simulation.initialize_simulation_env(gym, args)

    try:
        scene_data = setup_scene(gym, sim, viewer)
        systems_data = initialize_systems(gym, sim, scene_data, viewer, output_dir=output_dir)
        run_main_loop(gym, sim, viewer, scene_data, systems_data)
    finally:
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        print("Cleanup completed.")


def build_child_command(script_path, args, seed, output_dir):
    """构造单个 seed 的子进程命令。"""
    command = [
        sys.executable,
        str(script_path),
        "--single_seed",
        str(seed),
        "--output_dir",
        output_dir,
    ]

    physics_engine = getattr(args, "physics_engine", None)
    if physics_engine == gymapi.SIM_PHYSX:
        command.append("--physx")
    elif physics_engine == gymapi.SIM_FLEX:
        command.append("--flex")

    for arg_name in ("sim_device", "pipeline", "graphics_device_id", "num_threads", "subscenes", "slices"):
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            command.extend([f"--{arg_name}", str(arg_value)])

    return command


def run_seed_in_subprocess(script_path, args, seed, output_dir):
    """在独立子进程中运行单条轨迹，避免 PhysX 资源在同一进程中累积。"""
    command = build_child_command(script_path, args, seed, output_dir)
    result = subprocess.run(command, cwd=str(script_path.parent))
    if result.returncode != 0:
        raise RuntimeError(f"seed={seed} 运行失败，子进程退出码: {result.returncode}")


def main():
    script_path = Path(__file__).resolve()
    args = gymutil.parse_arguments(
        description="Franka Batch Trajectory Generator",
        custom_parameters=[
            {"name": "--seed_start", "type": int, "default": 0, "help": "起始随机种子（含）"},
            {"name": "--seed_end", "type": int, "default": 99, "help": "结束随机种子（含）"},
            {"name": "--single_seed", "type": int, "default": None, "help": "内部使用：运行单个 seed"},
            {"name": "--output_dir", "type": str, "default": None, "help": "内部使用：单个 seed 输出目录"},
            {
                "name": "--output_root",
                "type": str,
                "default": "/media/neuzz/HLX/zz/DataSet",
                "help": "输出根目录（每个 seed 保存到 camera_outputs_<seed>）",
            },
        ],
    )

    if args.single_seed is not None:
        if args.output_dir is None:
            raise ValueError("单 seed 模式必须提供 output_dir")
        run_single_trajectory(args, seed=args.single_seed, output_dir=args.output_dir)
        return

    if args.seed_end < args.seed_start:
        raise ValueError("seed_end 必须大于等于 seed_start")

    total = args.seed_end - args.seed_start + 1
    print(f"\n批量轨迹采集: seeds={args.seed_start}..{args.seed_end} (共 {total} 条)")

    for i, seed in enumerate(range(args.seed_start, args.seed_end + 1), start=1):
        output_dir = f"{args.output_root}/camera_outputs_{seed}"
        print(f"\n[{i}/{total}] 开始 seed={seed}, 输出目录: {output_dir}")
        run_seed_in_subprocess(script_path, args, seed=seed, output_dir=output_dir)

    print("\n批量轨迹采集完成。")


if __name__ == "__main__":
    main()
