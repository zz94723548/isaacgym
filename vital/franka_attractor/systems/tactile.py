import os
import numpy as np
import matplotlib.pyplot as plt
from isaacgym import gymapi, gymutil


def init_wrench_plot_window(title="Tactile 6D wrench", max_points=600):
    """创建 6D 力/力矩实时曲线窗口。"""
    plt.ion()
    fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(10, 6), num=title)

    ax_f.set_title("Force (world frame)")
    ax_f.set_ylabel("N")
    ax_f.grid(True, alpha=0.3)

    ax_t.set_title("Torque (world frame)")
    ax_t.set_ylabel("Nm")
    ax_t.set_xlabel("time (s)")
    ax_t.grid(True, alpha=0.3)

    (line_fx,) = ax_f.plot([], [], label="Fx")
    (line_fy,) = ax_f.plot([], [], label="Fy")
    (line_fz,) = ax_f.plot([], [], label="Fz")
    ax_f.legend(loc="upper right")

    (line_tx,) = ax_t.plot([], [], label="Tx")
    (line_ty,) = ax_t.plot([], [], label="Ty")
    (line_tz,) = ax_t.plot([], [], label="Tz")
    ax_t.legend(loc="upper right")

    fig.tight_layout()

    return {
        "fig": fig,
        "ax_f": ax_f,
        "ax_t": ax_t,
        "line_fx": line_fx,
        "line_fy": line_fy,
        "line_fz": line_fz,
        "line_tx": line_tx,
        "line_ty": line_ty,
        "line_tz": line_tz,
        "times": [],
        "fx": [],
        "fy": [],
        "fz": [],
        "tx": [],
        "ty": [],
        "tz": [],
        "max_points": int(max_points),
        "last_draw_time": -1.0,
    }


def update_wrench_plot_window(plot_state, t, force_world, torque_world, draw_period=0.05):
    """更新 6D 力/力矩实时曲线。"""
    if plot_state is None:
        return

    plot_state["times"].append(float(t))
    plot_state["fx"].append(float(force_world[0]))
    plot_state["fy"].append(float(force_world[1]))
    plot_state["fz"].append(float(force_world[2]))
    plot_state["tx"].append(float(torque_world[0]))
    plot_state["ty"].append(float(torque_world[1]))
    plot_state["tz"].append(float(torque_world[2]))

    if len(plot_state["times"]) > plot_state["max_points"]:
        for k in ("times", "fx", "fy", "fz", "tx", "ty", "tz"):
            plot_state[k] = plot_state[k][-plot_state["max_points"]:]

    if plot_state["last_draw_time"] < 0.0 or (t - plot_state["last_draw_time"]) >= draw_period:
        plot_state["line_fx"].set_data(plot_state["times"], plot_state["fx"])
        plot_state["line_fy"].set_data(plot_state["times"], plot_state["fy"])
        plot_state["line_fz"].set_data(plot_state["times"], plot_state["fz"])
        plot_state["line_tx"].set_data(plot_state["times"], plot_state["tx"])
        plot_state["line_ty"].set_data(plot_state["times"], plot_state["ty"])
        plot_state["line_tz"].set_data(plot_state["times"], plot_state["tz"])

        plot_state["ax_f"].relim()
        plot_state["ax_f"].autoscale_view()
        plot_state["ax_t"].relim()
        plot_state["ax_t"].autoscale_view()

        plot_state["fig"].canvas.draw_idle()
        plot_state["fig"].canvas.flush_events()
        plt.pause(0.001)
        plot_state["last_draw_time"] = float(t)


def init_dual_wrench_plot_window(title="Finger-Cube Tactile Compare 6D wrench", max_points=600):
    """创建左右指尖对比的 6D 力/力矩曲线窗口（同一窗口）。"""
    plt.ion()
    fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(12, 7), num=title)

    ax_f.set_title("Force compare (world frame)")
    ax_f.set_ylabel("N")
    ax_f.grid(True, alpha=0.3)

    ax_t.set_title("Torque compare (world frame)")
    ax_t.set_ylabel("Nm")
    ax_t.set_xlabel("time (s)")
    ax_t.grid(True, alpha=0.3)

    (line_rfx,) = ax_f.plot([], [], label="R_Fx")
    (line_rfy,) = ax_f.plot([], [], label="R_Fy")
    (line_rfz,) = ax_f.plot([], [], label="R_Fz")
    (line_lfx,) = ax_f.plot([], [], "--", label="L_Fx")
    (line_lfy,) = ax_f.plot([], [], "--", label="L_Fy")
    (line_lfz,) = ax_f.plot([], [], "--", label="L_Fz")
    ax_f.legend(loc="upper right", ncol=2)

    (line_rtx,) = ax_t.plot([], [], label="R_Tx")
    (line_rty,) = ax_t.plot([], [], label="R_Ty")
    (line_rtz,) = ax_t.plot([], [], label="R_Tz")
    (line_ltx,) = ax_t.plot([], [], "--", label="L_Tx")
    (line_lty,) = ax_t.plot([], [], "--", label="L_Ty")
    (line_ltz,) = ax_t.plot([], [], "--", label="L_Tz")
    ax_t.legend(loc="upper right", ncol=2)

    fig.tight_layout()

    return {
        "fig": fig,
        "ax_f": ax_f,
        "ax_t": ax_t,
        "line_rfx": line_rfx,
        "line_rfy": line_rfy,
        "line_rfz": line_rfz,
        "line_lfx": line_lfx,
        "line_lfy": line_lfy,
        "line_lfz": line_lfz,
        "line_rtx": line_rtx,
        "line_rty": line_rty,
        "line_rtz": line_rtz,
        "line_ltx": line_ltx,
        "line_lty": line_lty,
        "line_ltz": line_ltz,
        "times": [],
        "rfx": [], "rfy": [], "rfz": [], "rtx": [], "rty": [], "rtz": [],
        "lfx": [], "lfy": [], "lfz": [], "ltx": [], "lty": [], "ltz": [],
        "max_points": int(max_points),
        "last_draw_time": -1.0,
    }


def update_dual_wrench_plot_window(
    plot_state,
    t,
    right_force,
    right_torque,
    left_force,
    left_torque,
    draw_period=0.05,
):
    """更新左右指尖对比曲线。"""
    if plot_state is None:
        return

    plot_state["times"].append(float(t))

    plot_state["rfx"].append(float(right_force[0]))
    plot_state["rfy"].append(float(right_force[1]))
    plot_state["rfz"].append(float(right_force[2]))
    plot_state["rtx"].append(float(right_torque[0]))
    plot_state["rty"].append(float(right_torque[1]))
    plot_state["rtz"].append(float(right_torque[2]))

    plot_state["lfx"].append(float(left_force[0]))
    plot_state["lfy"].append(float(left_force[1]))
    plot_state["lfz"].append(float(left_force[2]))
    plot_state["ltx"].append(float(left_torque[0]))
    plot_state["lty"].append(float(left_torque[1]))
    plot_state["ltz"].append(float(left_torque[2]))

    if len(plot_state["times"]) > plot_state["max_points"]:
        for k in (
            "times", "rfx", "rfy", "rfz", "rtx", "rty", "rtz",
            "lfx", "lfy", "lfz", "ltx", "lty", "ltz",
        ):
            plot_state[k] = plot_state[k][-plot_state["max_points"]:]

    if plot_state["last_draw_time"] < 0.0 or (t - plot_state["last_draw_time"]) >= draw_period:
        times = plot_state["times"]
        plot_state["line_rfx"].set_data(times, plot_state["rfx"])
        plot_state["line_rfy"].set_data(times, plot_state["rfy"])
        plot_state["line_rfz"].set_data(times, plot_state["rfz"])
        plot_state["line_lfx"].set_data(times, plot_state["lfx"])
        plot_state["line_lfy"].set_data(times, plot_state["lfy"])
        plot_state["line_lfz"].set_data(times, plot_state["lfz"])

        plot_state["line_rtx"].set_data(times, plot_state["rtx"])
        plot_state["line_rty"].set_data(times, plot_state["rty"])
        plot_state["line_rtz"].set_data(times, plot_state["rtz"])
        plot_state["line_ltx"].set_data(times, plot_state["ltx"])
        plot_state["line_lty"].set_data(times, plot_state["lty"])
        plot_state["line_ltz"].set_data(times, plot_state["ltz"])

        plot_state["ax_f"].relim()
        plot_state["ax_f"].autoscale_view()
        plot_state["ax_t"].relim()
        plot_state["ax_t"].autoscale_view()

        plot_state["fig"].canvas.draw_idle()
        plot_state["fig"].canvas.flush_events()
        plt.pause(0.001)
        plot_state["last_draw_time"] = float(t)


def init_wrench_logger(output_dir, filename="tactile_wrench.csv"):
    """初始化 6D 力数据日志文件（CSV）。"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    f = open(csv_path, "w", encoding="utf-8")
    f.write("time,fx,fy,fz,tx,ty,tz\n")
    f.flush()
    print(f"[TactileLogger] logging 6D wrench to: {csv_path}")
    return f, csv_path


def append_wrench_log(log_file, t, force_world, torque_world):
    """追加写入一条 6D 力/力矩数据。"""
    log_file.write(
        f"{t:.6f},"
        f"{float(force_world[0]):.9f},{float(force_world[1]):.9f},{float(force_world[2]):.9f},"
        f"{float(torque_world[0]):.9f},{float(torque_world[1]):.9f},{float(torque_world[2]):.9f}\n"
    )


def vec3_to_np(v):
    """将 gymapi.Vec3 或结构化字段转换为 numpy 向量。"""
    if hasattr(v, "x"):
        return np.array([v.x, v.y, v.z], dtype=np.float32)
    try:
        return np.array([v['x'], v['y'], v['z']], dtype=np.float32)
    except Exception:
        return np.array([v[0], v[1], v[2]], dtype=np.float32)


def quat_rotate_vec3(q, v_xyz):
    """用四元数旋转向量（numpy实现）。"""
    qv = np.array([q.x, q.y, q.z], dtype=np.float32)
    v = np.array(v_xyz, dtype=np.float32)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (q.w * uv + uuv)


def contact_field(c, name, default=None):
    """兼容对象属性 / numpy 结构化数组字段访问（含别名）。"""
    aliases = {
        "local_pos0": ("local_pos0", "localPos0", "offset0", "position0", "pos0"),
        "local_pos1": ("local_pos1", "localPos1", "offset1", "position1", "pos1"),
        "lambda": ("lambda",),
        "normal": ("normal",),
        "body0": ("body0",),
        "body1": ("body1",),
    }

    candidates = aliases.get(name, (name,))

    for key in candidates:
        if hasattr(c, key):
            return getattr(c, key)
        try:
            return c[key]
        except Exception:
            pass

    return default


def compute_body_pair_contact_wrench(
    gym,
    env,
    body_a_sim_idx,
    body_b_sim_idx,
    body_a_pos,
    body_a_rot,
    ref_world_pos,
):
    """计算 body_b 作用在 body_a 上的法向接触力与相对参考点力矩。"""
    contacts = gym.get_env_rigid_contacts(env)
    return compute_body_pair_contact_wrench_from_contacts(
        contacts,
        body_a_sim_idx,
        body_b_sim_idx,
        body_a_pos,
        body_a_rot,
        ref_world_pos,
    )


def compute_body_pair_contact_wrench_from_contacts(
    contacts,
    body_a_sim_idx,
    body_b_sim_idx,
    body_a_pos,
    body_a_rot,
    ref_world_pos,
):
    """基于已获取 contacts 计算接触力/力矩，避免重复获取接触列表。"""

    force_world = np.zeros(3, dtype=np.float32)
    torque_world = np.zeros(3, dtype=np.float32)

    pair_count = 0
    for c in contacts:
        b0 = int(contact_field(c, 'body0'))
        b1 = int(contact_field(c, 'body1'))

        if not ((b0 == body_a_sim_idx and b1 == body_b_sim_idx) or
            (b0 == body_b_sim_idx and b1 == body_a_sim_idx)):
            continue

        lam = float(contact_field(c, 'lambda'))
        if lam <= 0.0:
            continue

        n = vec3_to_np(contact_field(c, 'normal'))

        if b0 == body_a_sim_idx:
            f = -n * lam
            local_pos_raw = contact_field(c, 'local_pos0', default=None)
        else:
            f = n * lam
            local_pos_raw = contact_field(c, 'local_pos1', default=None)

        if local_pos_raw is None:
            contact_world_pos = ref_world_pos
        else:
            local_pos_a = vec3_to_np(local_pos_raw)
            contact_world_pos = body_a_pos + quat_rotate_vec3(body_a_rot, local_pos_a)

        force_world += f
        torque_world += np.cross(contact_world_pos - ref_world_pos, f)
        pair_count += 1

    return force_world, torque_world, pair_count


def draw_sensor_wrench(
    gym,
    viewer,
    env,
    sensor_world_pos,
    force_world,
    torque_world,
    sensor_axes_geom=None,
    sensor_marker_geom=None,
    marker_color=(0.0, 1.0, 1.0),
):
    """绘制传感器位置、力向量、力矩向量。"""
    sensor_tf = gymapi.Transform(p=gymapi.Vec3(*sensor_world_pos), r=gymapi.Quat(0, 0, 0, 1))

    if sensor_axes_geom is not None:
        gymutil.draw_lines(sensor_axes_geom, gym, viewer, env, sensor_tf)

    sensor_marker = sensor_marker_geom
    if sensor_marker is None:
        sensor_marker = gymutil.WireframeSphereGeometry(0.008, 10, 10, color=marker_color)
    gymutil.draw_lines(sensor_marker, gym, viewer, env, sensor_tf)

    f_mag = float(np.linalg.norm(force_world))
    t_mag = float(np.linalg.norm(torque_world))

    f_len = np.clip(0.02 * f_mag, 0.005, 0.25) if f_mag > 1e-6 else 0.0
    t_len = np.clip(0.06 * t_mag, 0.005, 0.20) if t_mag > 1e-6 else 0.0

    p0 = gymapi.Vec3(*sensor_world_pos)

    if f_len > 0.0:
        f_dir = force_world / (f_mag + 1e-9)
        p1 = gymapi.Vec3(*(sensor_world_pos + f_dir * f_len))
        gymutil.draw_line(p0, p1, gymapi.Vec3(0.0, 1.0, 0.0), gym, viewer, env)

    if t_len > 0.0:
        t_dir = torque_world / (t_mag + 1e-9)
        p2 = gymapi.Vec3(*(sensor_world_pos + t_dir * t_len))
        gymutil.draw_line(p0, p2, gymapi.Vec3(1.0, 1.0, 0.0), gym, viewer, env)