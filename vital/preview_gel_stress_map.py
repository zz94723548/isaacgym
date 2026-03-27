import os
import argparse
import numpy as np
from PIL import Image
from skimage.color import lab2rgb

# ─────────────────────────────────────────────────────────────────────────────
# VITaL 格式说明（参照论文 arXiv:2403.11898 Fig.5）
#   应变图三通道定义（LAB 色彩空间）:
#     channel 0 → 法向应变（深度）   = Fz   → L 通道（亮度）
#     channel 1 → x 切向应变        = Fx   → B 通道（蓝-黄色谱）
#     channel 2 → y 切向应变        = Fy   → A 通道（红-绿色谱）
#   力矩 Tx, Ty, Tz 用于在传感器面上添加空间梯度，使单点接触力形成
#   符合物理直觉的 2D 分布。
# ─────────────────────────────────────────────────────────────────────────────


def _norm_to_range(x, lo, hi, eps=1e-10):
    """将数组线性缩放到 [lo, hi]，若范围极小则置中值。"""
    xmin, xmax = float(x.min()), float(x.max())
    if xmax - xmin < eps:
        return np.full_like(x, (lo + hi) / 2.0, dtype=np.float32)
    return (lo + (x - xmin) / (xmax - xmin) * (hi - lo)).astype(np.float32)


def gel_vector_to_strain_map(gel_vec, h=32, w=32, scale=1e4):
    """
    将单帧六轴力矩向量 [Fx, Fy, Fz, Tx, Ty, Tz] (单位 N / Nm)
    映射为 VITaL 格式的空间应变图 (H, W, 3), float32。

    参数:
        scale  : 放大系数，将量级极小的原始力矩值放大便于计算

    输出通道:
        [0] normal_strain   = Fz + Tx*dy - Ty*dx   （法向/深度）
        [1] x_strain        = Fx + Tz*dy            （x 切向）
        [2] y_strain        = Fy - Tz*dx            （y 切向）
    其中 dx, dy ∈ [-1, 1] 为归一化像素坐标，力矩在传感器面上
    产生线性梯度分布。
    """
    v = np.asarray(gel_vec, dtype=np.float64) * scale
    if v.shape[0] != 6:
        raise ValueError(f"gel 向量维度应为 6，实际为 {v.shape[0]}")
    Fx, Fy, Fz, Tx, Ty, Tz = v

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    dy = (ys - cy) / (h / 2.0)   # -1 ~ +1
    dx = (xs - cx) / (w / 2.0)   # -1 ~ +1

    normal_strain = Fz + Tx * dy - Ty * dx   # 法向
    x_strain      = Fx + Tz * dy             # x 切向（力矩使之沿 y 方向有梯度）
    y_strain      = Fy - Tz * dx             # y 切向（力矩使之沿 x 方向有梯度）

    return np.stack([normal_strain, x_strain, y_strain], axis=-1).astype(np.float32)


def strain_map_to_lab_rgb(strain_map):
    """
    将应变图 (H, W, 3) 渲染为 LAB 色彩空间的 RGB uint8 图像 (H, W, 3)。

    通道映射（对齐 VITaL 论文 Fig.5）:
        strain[:, :, 0] (法向)  → L  (亮度,  0 ~ 100)
        strain[:, :, 1] (x切向) → B  (蓝-黄, -128 ~ 127)
        strain[:, :, 2] (y切向) → A  (红-绿, -128 ~ 127)
    """
    L = _norm_to_range(strain_map[:, :, 0],    0,  100)
    B = _norm_to_range(strain_map[:, :, 1], -128,  127)  # x切向 → 蓝-黄
    A = _norm_to_range(strain_map[:, :, 2], -128,  127)  # y切向 → 红-绿

    lab = np.stack([L, A, B], axis=-1).astype(np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb_float = lab2rgb(lab)                           # skimage: LAB→RGB [0,1]
    return np.clip(rgb_float * 255, 0, 255).astype(np.uint8)


def save_strain_map_png(strain_map, out_png):
    """以 LAB 色彩空间保存应变图为 PNG（用于可视化）。"""
    rgb = strain_map_to_lab_rgb(strain_map)
    Image.fromarray(rgb).save(out_png)


def main():
    parser = argparse.ArgumentParser(
        description="从 gel 原始力矩数据生成 VITaL 格式应变图预览"
    )
    parser.add_argument("--gel_dir", type=str, required=True,
                        help="gel 文件夹路径，如 /.../camera_outputs_0/gel")
    parser.add_argument("--out_dir", type=str, default="./gel_strain_preview",
                        help="输出目录")
    parser.add_argument("--num", type=int, default=10,
                        help="导出前多少帧")
    parser.add_argument("--scale", type=float, default=1e4,
                        help="力矩放大系数（原始量级极小，默认 1e4）")
    parser.add_argument("--size", type=int, default=32,
                        help="输出应变图分辨率（正方形边长，默认 32）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gel_files = sorted([f for f in os.listdir(args.gel_dir) if f.endswith('.npy')])
    if len(gel_files) == 0:
        raise RuntimeError(f"未在 {args.gel_dir} 找到 .npy gel 文件")

    n = min(args.num, len(gel_files))
    all_maps = []

    print(f"找到 {len(gel_files)} 个 gel 帧，导出前 {n} 帧 → {args.out_dir}")
    print(f"输出格式: ({args.size}, {args.size}, 3) float32  /  LAB-RGB PNG")
    print(f"通道: [0]=法向(Fz)  [1]=x切向(Fx)  [2]=y切向(Fy)  +力矩梯度")
    print()

    for i in range(n):
        gel_path = os.path.join(args.gel_dir, gel_files[i])
        gel_vec = np.load(gel_path)                           # (6,)
        strain_map = gel_vector_to_strain_map(
            gel_vec, h=args.size, w=args.size, scale=args.scale
        )                                                     # (H, W, 3)

        out_png = os.path.join(args.out_dir, f"{i:04d}_strain.png")
        save_strain_map_png(strain_map, out_png)
        all_maps.append(strain_map)

        Fx, Fy, Fz, Tx, Ty, Tz = gel_vec
        print(f"[{i+1:3d}/{n}] {gel_files[i]}  "
              f"Fz={Fz:.2e} Fx={Fx:.2e} Fy={Fy:.2e} "
              f"Tx={Tx:.2e} Ty={Ty:.2e} Tz={Tz:.2e}  → {out_png}")

    all_maps = np.stack(all_maps, axis=0).astype(np.float32)   # (N, H, W, 3)
    npy_path = os.path.join(args.out_dir, "strain_maps.npy")
    np.save(npy_path, all_maps)
    print(f"\n✅ 完成。应变图序列已保存: {npy_path}  shape={all_maps.shape}")


if __name__ == "__main__":
    main()
