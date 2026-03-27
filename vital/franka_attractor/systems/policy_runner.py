"""
在线策略推理模块
================
负责加载策略模型、构造在线观测、执行动作预测，并提供基础安全限幅。

说明：
- 当前工程尚未包含 ACT 网络结构定义，本模块先提供“可运行骨架”。
- 若 ckpt 可直接作为可调用模型（TorchScript 或带 __call__ 的对象），会直接启用。
- 若无法解析模型结构，将退化为“保持当前姿态”的安全输出，不阻塞主循环。
"""

import json
import os
import pickle
import sys
import types
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torchvision import transforms
try:
    import cv2
except Exception:
    cv2 = None

from config import SimulationConfig as Config


def _to_numpy(x: Any, dtype=np.float32) -> np.ndarray:
    """将输入转换为 numpy 数组。"""
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


class PolicyRunner:
    """在线策略推理器。"""

    def __init__(self, config_cls=Config, device: Optional[str] = None):
        self.cfg = config_cls
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = None
        self.model_loaded = False
        self.model_error = None
        self.model_kind = "unknown"

        self.args: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.camera_names: List[str] = ["realsence1", "realsence2", "gelsight"]
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        # 参考 robot_operation.py 预处理尺寸（H, W）
        self.act_image_size = (400, 480)

        self.prev_action = None

    def load_model(self) -> bool:
        """加载策略参数、归一化统计和模型权重。"""
        self.model_loaded = False
        self.model_error = None

        # 1) 加载 args
        args_path = self.cfg.POLICY_ARGS
        if os.path.exists(args_path):
            with open(args_path, "r", encoding="utf-8") as f:
                self.args = json.load(f)
            self.camera_names = self.args.get("camera_names", self.camera_names)
        else:
            self.args = {}

        # 2) 加载 stats
        stats_path = self.cfg.POLICY_STATS
        self.stats = self._load_stats(stats_path)

        # 3) 加载模型权重
        ckpt_path = self.cfg.POLICY_CKPT
        if not os.path.exists(ckpt_path):
            self.model_error = f"checkpoint not found: {ckpt_path}"
            return False

        # 3.1 优先尝试 TorchScript
        try:
            self.model = torch.jit.load(ckpt_path, map_location=self.device)
            self.model.eval()
            self.model_loaded = True
            self.model_kind = "torchscript"
            return True
        except Exception:
            pass

        # 3.2 尝试 torch.load（如直接保存了可调用对象）
        try:
            obj = torch.load(ckpt_path, map_location=self.device)
            if hasattr(obj, "eval"):
                obj.eval()
            if callable(obj):
                self.model = obj
                self.model_loaded = True
                self.model_kind = "callable"
                return True

            # 常见情况：state_dict 或 lightning checkpoint（需要网络结构才能真正恢复）
            if isinstance(obj, dict):
                # 尝试使用 act_policy 中的真实 ACT 网络定义加载
                if self._try_load_act_policy(ckpt_path, args_path):
                    return True

                # 若 ACT 加载已设置更具体错误，优先保留
                if not self.model_error:
                    keys = list(obj.keys())
                    self.model_error = (
                        "checkpoint loaded but network definition is missing in current repo; "
                        f"top-level keys={keys[:8]}"
                    )
                return False

            self.model_error = "unsupported checkpoint object type"
            return False
        except Exception as e:
            self.model_error = f"failed to load checkpoint: {e}"
            return False

    def _try_load_act_policy(self, ckpt_path: str, args_path: str) -> bool:
        """使用本仓库的 act_policy 源码加载真实 ACT 模型。"""
        try:
            # 某些训练源码硬依赖 IPython，这里提供兼容桩避免运行环境缺包时报错
            if "IPython" not in sys.modules:
                ipy_stub = types.ModuleType("IPython")
                ipy_stub.embed = lambda *args, **kwargs: None
                sys.modules["IPython"] = ipy_stub

            # 兼容旧版 torch(1.8) 与部分新库接口不一致的问题
            try:
                nm_sig = inspect.signature(torch.nn.Module.named_modules)
                if "remove_duplicate" not in nm_sig.parameters:
                    _orig_named_modules = torch.nn.Module.named_modules

                    def _named_modules_compat(self, memo=None, prefix='', remove_duplicate=True):
                        return _orig_named_modules(self, memo=memo, prefix=prefix)

                    torch.nn.Module.named_modules = _named_modules_compat

                np_sig = inspect.signature(torch.nn.Module.named_parameters)
                if "remove_duplicate" not in np_sig.parameters:
                    _orig_named_parameters = torch.nn.Module.named_parameters

                    def _named_parameters_compat(self, prefix='', recurse=True, remove_duplicate=True):
                        return _orig_named_parameters(self, prefix=prefix, recurse=recurse)

                    torch.nn.Module.named_parameters = _named_parameters_compat

                if not hasattr(torch.nn.Module, "get_submodule"):
                    def _get_submodule_compat(self, target: str):
                        if target == "":
                            return self
                        mod = self
                        for item in target.split("."):
                            mod = getattr(mod, item)
                        return mod

                    torch.nn.Module.get_submodule = _get_submodule_compat
            except Exception:
                pass

            # 训练源码中广泛使用 cv2，若环境缺失则提供最小兼容实现（主要保证导入成功）
            if "cv2" not in sys.modules and cv2 is None:
                cv2_stub = types.ModuleType("cv2")
                cv2_stub.COLOR_LAB2BGR = 0

                def _resize(img, dsize, fx=0, fy=0):
                    # dsize: (w, h)
                    w, h = dsize
                    t = torch.from_numpy(img).float()
                    if t.ndim == 2:
                        t = t.unsqueeze(0).unsqueeze(0)
                    else:
                        t = t.permute(2, 0, 1).unsqueeze(0)
                    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
                    out = t.squeeze(0)
                    if out.ndim == 3:
                        out = out.permute(1, 2, 0)
                    return out.cpu().numpy().astype(img.dtype, copy=False)

                cv2_stub.resize = _resize
                cv2_stub.bitwise_and = lambda a, b, mask=None: a if mask is None else (a * (mask[..., None] > 0))
                cv2_stub.cvtColor = lambda img, code: img
                sys.modules["cv2"] = cv2_stub

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            act_src = os.path.join(project_root, "model", "act_policy")
            if not os.path.exists(os.path.join(act_src, "load_ACT.py")):
                self.model_error = f"act_policy loader not found: {act_src}"
                return False

            if act_src not in sys.path:
                sys.path.insert(0, act_src)

            detr_src = os.path.join(act_src, "detr")
            if os.path.exists(detr_src) and detr_src not in sys.path:
                sys.path.insert(0, detr_src)

            from load_ACT import load_ACT  # type: ignore

            override_args = {}
            try:
                with open(args_path, "r", encoding="utf-8") as f:
                    act_args = json.load(f)

                g_path = act_args.get("gelsight_backbone_path", "none")
                v_path = act_args.get("vision_backbone_path", "none")

                # 兼容相对路径
                def _resolve(p):
                    if not isinstance(p, str):
                        return None
                    return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.path.dirname(args_path), p))

                g_resolved = _resolve(g_path)
                v_resolved = _resolve(v_path)

                # 若路径不存在，自动关闭预训练 backbone 加载（与 robot_operation.py 逻辑一致）
                if isinstance(g_path, str) and g_path != "none" and (g_resolved is None or not os.path.exists(g_resolved)):
                    override_args["gelsight_backbone_path"] = "none"
                if isinstance(v_path, str) and v_path != "none" and (v_resolved is None or not os.path.exists(v_resolved)):
                    override_args["vision_backbone_path"] = "none"
            except Exception:
                # args 解析失败时不阻断，走默认加载
                override_args = {}

            self.model = load_ACT(
                ckpt_path,
                args_path,
                override_args=override_args if len(override_args) > 0 else None,
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_kind = "act"
            self.model_loaded = True
            return True
        except Exception as e:
            self.model_error = f"failed to load ACT policy: {e}"
            return False

    @staticmethod
    def _load_stats(stats_path: str) -> Dict[str, Any]:
        """加载归一化统计，支持 pkl/json。"""
        if not os.path.exists(stats_path):
            return {}

        if stats_path.endswith(".json"):
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            with open(stats_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            return {}

    def build_obs(
        self,
        qpos: np.ndarray,
        camera_frames: Optional[Dict[str, np.ndarray]] = None,
        gelsight_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        构建模型输入观测。

        参数
        ----
        qpos : ndarray (4,)
            当前状态 [x, y, z, gripper]
        camera_frames : dict
            e.g. {'realsence1': HxWx3, 'realsence2': HxWx3}
        gelsight_image : ndarray
            触觉图像 (H, W, 3)
        """
        qpos = _to_numpy(qpos).reshape(-1)
        qpos_n = self._normalize_qpos(qpos)

        obs = {
            "qpos": torch.from_numpy(qpos_n).float().to(self.device).unsqueeze(0),
            "images": {},
            "gelsight": None,
            "image_list": None,
        }

        if camera_frames:
            for name, img in camera_frames.items():
                img_np = _to_numpy(img, dtype=np.float32)
                obs["images"][name] = self._preprocess_rgb_image(img_np)

        if gelsight_image is not None:
            gel_np = _to_numpy(gelsight_image, dtype=np.float32)
            obs["gelsight"] = self._preprocess_gelsight(gel_np)

        # ACT 需要按 camera_names 顺序提供 list[Tensor]
        obs["image_list"] = self._build_act_image_list(obs["images"], obs["gelsight"])

        return obs

    def predict_action(self, obs: Dict[str, Any], qpos_fallback: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测动作，返回 shape=(4,) 的 numpy: [ax, ay, az, gripper]
        """
        # 模型不可用时：安全回退（保持当前位置）
        if not self.model_loaded or self.model is None:
            if qpos_fallback is None:
                return np.zeros(4, dtype=np.float32)
            q = _to_numpy(qpos_fallback).reshape(-1)
            return np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)

        with torch.no_grad():
            if self.model_kind == "act":
                pred = self.model(obs.get("qpos"), obs.get("image_list"))
                action = self._act_to_action(pred, qpos_fallback)
            else:
                try:
                    pred = self.model(obs)
                except TypeError:
                    # 某些模型签名可能是 (qpos, images, gelsight)
                    pred = self.model(obs.get("qpos"), obs.get("images"), obs.get("gelsight"))

                action = self._extract_action(pred)
                action = self._denormalize_action(action)

        action = self.safe_clip_action(action)
        return action.astype(np.float32)

    def _act_to_action(self, pred: Any, qpos_fallback: Optional[np.ndarray]) -> np.ndarray:
        """将 ACT 输出（通常为归一化 delta 序列）还原为绝对动作 [x,y,z,gripper]。"""
        arr = _to_numpy(pred)
        # 期望形状 [B, T, 4]，取当前时刻第一条查询
        if arr.ndim == 3:
            delta_n = arr[0, 0, :]
        elif arr.ndim == 2:
            delta_n = arr[0, :]
        else:
            delta_n = arr.reshape(-1)[:4]

        if qpos_fallback is None:
            qpos = np.zeros(4, dtype=np.float32)
        else:
            qpos = _to_numpy(qpos_fallback).reshape(-1)[:4]

        # 若有 delta 统计，按 robot_operation.py 的逻辑还原
        delta_mean = self._get_stat_vector("delta_mean", 4, default=0.0)
        delta_std = self._get_stat_vector("delta_std", 4, default=1.0)
        has_delta_stats = bool(np.any(np.abs(delta_std - 1.0) > 1e-8) or np.any(np.abs(delta_mean) > 1e-8))

        if has_delta_stats:
            delta = delta_n * delta_std + delta_mean
            action = np.array(
                [
                    qpos[0] + delta[0],
                    qpos[1] + delta[1],
                    qpos[2] + delta[2],
                    delta[3],
                ],
                dtype=np.float32,
            )
            return action

        # 回退：当成绝对动作标准化输出
        return self._denormalize_action(delta_n)

    def _preprocess_rgb_image(self, img: np.ndarray) -> torch.Tensor:
        """RGB 预处理：resize + imagenet normalize + CHW。"""
        img_np = _to_numpy(img, dtype=np.float32)
        h, w = self.act_image_size
        if img_np.shape[0] != h or img_np.shape[1] != w:
            if cv2 is not None:
                img_np = cv2.resize(img_np, (w, h))
            else:
                # 无 cv2 时用 torch 插值
                t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().unsqueeze(0)
                t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
                img_np = np.transpose(t.squeeze(0).cpu().numpy(), (1, 2, 0))

        if img_np.max() > 1.5:
            img_np = img_np / 255.0

        img_t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float()
        img_t = self.image_normalize(img_t)
        return img_t.to(self.device).unsqueeze(0)

    def _preprocess_gelsight(self, gel: np.ndarray) -> torch.Tensor:
        """GelSight 预处理：按统计量标准化 + CHW。"""
        g = _to_numpy(gel, dtype=np.float32)
        mean = self._get_stat_vector("gelsight_mean", 3, default=0.0).reshape(1, 1, 3)
        std = self._get_stat_vector("gelsight_std", 3, default=1.0).reshape(1, 1, 3)
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        g = (g - mean) / std
        g_t = torch.from_numpy(np.transpose(g, (2, 0, 1))).float()
        return g_t.to(self.device).unsqueeze(0)

    def _build_act_image_list(self, images: Dict[str, torch.Tensor], gelsight: Optional[torch.Tensor]) -> List[torch.Tensor]:
        """按 args.json 中 camera_names 顺序构造 ACT 输入列表。"""
        image_list: List[torch.Tensor] = []
        h, w = self.act_image_size
        blank = torch.zeros((1, 3, h, w), dtype=torch.float32, device=self.device)

        for cam_name in self.camera_names:
            if cam_name == "gelsight":
                if gelsight is not None:
                    image_list.append(gelsight)
                else:
                    image_list.append(blank)
            elif cam_name == "blank":
                image_list.append(blank)
            else:
                # 常见别名兼容
                candidates = [cam_name]
                if cam_name == "1":
                    candidates.append("realsence1")
                if cam_name == "2":
                    candidates.append("realsence2")

                found = None
                for c in candidates:
                    if c in images:
                        found = images[c]
                        break

                image_list.append(found if found is not None else blank)

        return image_list

    def safe_clip_action(self, action: np.ndarray) -> np.ndarray:
        """动作安全约束：工作空间、单步位移、夹爪限幅、EMA。"""
        a = _to_numpy(action).reshape(-1)
        if a.shape[0] < 4:
            out = np.zeros(4, dtype=np.float32)
            out[: a.shape[0]] = a
            a = out

        # 夹爪限幅
        a[3] = np.clip(a[3], self.cfg.GRIPPER_ACTION_MIN, self.cfg.GRIPPER_ACTION_MAX)

        # 工作空间限幅
        a[0] = np.clip(a[0], self.cfg.POLICY_WORKSPACE_X[0], self.cfg.POLICY_WORKSPACE_X[1])
        a[1] = np.clip(a[1], self.cfg.POLICY_WORKSPACE_Y[0], self.cfg.POLICY_WORKSPACE_Y[1])
        a[2] = np.clip(a[2], self.cfg.POLICY_WORKSPACE_Z[0], self.cfg.POLICY_WORKSPACE_Z[1])

        # 单步最大位移约束（相对上一次动作）
        if self.prev_action is not None:
            delta = a[:3] - self.prev_action[:3]
            step_norm = float(np.linalg.norm(delta))
            max_step = float(self.cfg.MAX_DELTA_XYZ)
            if step_norm > max_step and step_norm > 1e-9:
                delta = delta / step_norm * max_step
                a[:3] = self.prev_action[:3] + delta

        # EMA 平滑
        alpha = float(self.cfg.POLICY_ACTION_EMA_ALPHA)
        if self.prev_action is not None and alpha > 0.0:
            alpha = np.clip(alpha, 0.0, 1.0)
            a = alpha * self.prev_action + (1.0 - alpha) * a

        self.prev_action = a.copy()
        return a

    def _normalize_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """使用统计量归一化 qpos。"""
        mean = self._get_stat_vector("qpos_mean", qpos.shape[0], default=0.0)
        std = self._get_stat_vector("qpos_std", qpos.shape[0], default=1.0)
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        return (qpos - mean) / std

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """将模型输出从标准化空间还原到物理量空间。"""
        action = _to_numpy(action).reshape(-1)
        mean = self._get_stat_vector("action_mean", action.shape[0], default=0.0)
        std = self._get_stat_vector("action_std", action.shape[0], default=1.0)
        return action * std + mean

    def _get_stat_vector(self, key: str, dim: int, default: float) -> np.ndarray:
        """读取统计量向量，不存在则返回常数向量。"""
        stats = self.stats if isinstance(self.stats, dict) else {}
        # 兼容 args.json 中内嵌 norm_stats
        if key not in stats and isinstance(self.args, dict):
            norm_stats = self.args.get("norm_stats", {})
            if key in norm_stats:
                stats = norm_stats

        if key in stats:
            v = _to_numpy(stats[key]).reshape(-1)
            if v.shape[0] == dim:
                return v
            out = np.full((dim,), default, dtype=np.float32)
            n = min(dim, v.shape[0])
            out[:n] = v[:n]
            return out

        return np.full((dim,), default, dtype=np.float32)

    @staticmethod
    def _extract_action(pred: Any) -> np.ndarray:
        """
        解析模型输出，提取 4 维动作。

        支持：
        - Tensor: [B,4] / [4]
        - dict: {'action': ...} / {'pred_actions': ...}
        - list/tuple: 取最后一个元素
        """
        if isinstance(pred, dict):
            for k in ("action", "actions", "pred_action", "pred_actions", "a_hat"):
                if k in pred:
                    return _to_numpy(pred[k]).reshape(-1)[-4:]
            # 兜底：取第一个 value
            if len(pred) > 0:
                first_val = next(iter(pred.values()))
                return _to_numpy(first_val).reshape(-1)[-4:]

        if isinstance(pred, (list, tuple)) and len(pred) > 0:
            return _to_numpy(pred[-1]).reshape(-1)[-4:]

        arr = _to_numpy(pred).reshape(-1)
        if arr.shape[0] >= 4:
            return arr[-4:]

        out = np.zeros(4, dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out


def load_policy_runner(config_cls=Config, device: Optional[str] = None) -> PolicyRunner:
    """创建并加载策略推理器。"""
    runner = PolicyRunner(config_cls=config_cls, device=device)
    ok = runner.load_model()
    if ok:
        print(f"[PolicyRunner] model loaded on {runner.device}")
    else:
        print(f"[PolicyRunner] model not fully loaded, fallback enabled. reason: {runner.model_error}")
    return runner
