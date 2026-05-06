#!/usr/bin/env python3
"""Bridge-based teleop + LeRobot-style demo recorder.

This is a bridge-backed copy for recording demos without rclpy subscriptions.
- Robot control/observation backend: stretch_ai bridge worker (HomeRobotZmqClient)
- Manual controls: base, arm, head, wrist, gripper
- Recording output format: same as robot_teleop_ui_ros_v5.py (LeRobotStyleRecorder)
"""

from __future__ import annotations

import base64
import colorsys
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np
from ultralytics import SAM
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QThread
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import QSize
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtWidgets import QGroupBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QListWidget
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QMenu
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QTableWidget
from PyQt6.QtWidgets import QTableWidgetItem
from PyQt6.QtWidgets import QSlider
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QAction
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QImage
from PyQt6.QtGui import QPixmap

try:
    import roslibpy
except ModuleNotFoundError:
    roslibpy = None


###############################################################################
# Runtime configuration
###############################################################################
STRETCH_AI_REPO = "/home/ibk5106/Desktop/Projects/stretch_ai"
STRETCH_AI_ENV_NAME = "stretch_ai"
STRETCH_AI_WORKER_LAUNCHER = "mamba"
STRETCH_AI_WORKER_PYTHON = "python"

STRETCH_AI_ROBOT_IP = "192.168.1.10"
STRETCH_AI_USE_REMOTE_COMPUTER = True
STRETCH_AI_RECV_PORT = 4401
STRETCH_AI_SEND_PORT = 4402
STRETCH_AI_RECV_STATE_PORT = 4403
STRETCH_AI_RECV_SERVO_PORT = 4404

STRETCH_AI_CONNECT_TIMEOUT_S = 30.0
STRETCH_AI_RPC_TIMEOUT_S = 8.0
STRETCH_AI_OBS_POLL_HZ = 10.0
STRETCH_AI_WORKER_JPEG_QUALITY = 85
STRETCH_AI_ROTATE_HEAD_90_CW = False
STRETCH_AI_WORKER_DEBUG = True
# Direct ROS topic path can have a different camera orientation than bridge worker output.
ROS_TOPICS_ROTATE_HEAD_90_CW = True

ROSBRIDGE_HOST = "127.0.0.1"
ROSBRIDGE_PORT = 9090
ROSBRIDGE_TIMEOUT_S = 8.0

HEAD_RGB_TOPIC = "/camera/color/image_raw"
HEAD_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
HEAD_CAMERA_INFO_TOPIC = "/camera/aligned_depth_to_color/camera_info"

WRIST_RGB_TOPIC = "/gripper_camera/color/image_rect_raw"
WRIST_DEPTH_TOPIC = "/gripper_camera/aligned_depth_to_color/image_raw"
WRIST_CAMERA_INFO_TOPIC = "/gripper_camera/aligned_depth_to_color/camera_info"

JOINT_STATE_TOPIC = "/stretch/joint_states"
ODOM_TOPIC = "/odom"
IMU_MOBILE_BASE_TOPIC = "/imu_mobile_base"
IMU_WRIST_TOPIC = "/imu_wrist"
IMU_CAMERA_ACCEL_TOPIC = "/camera/accel/sample"
IMU_CAMERA_GYRO_TOPIC = "/camera/gyro/sample"
MAGNETOMETER_TOPIC = "/magnetometer_mobile_base"
BATTERY_TOPIC = "/battery"

# Recorder defaults (same style as v5)
DEMO_RECORD_FPS = 10
DEMO_RECORD_QUEUE_MAX = 8
DEFAULT_DATASET_ROOT = str((Path.cwd() / "stretch_recordings_v2").resolve())

# Command smoothing (same values as v5)
COMMAND_SMOOTH_STEP_SIZES = [
    0.005,
    0.005,
    0.020,
    0.020,
    0.020,
    0.020,
    0.020,
    0.005,
    0.020,
    0.030,
]

DEFAULT_COMMAND_SMOOTH_DELAY_S = 0.110
UI_REFRESH_MS = 100
DEFAULT_BASE_ROTATE_STEP_DEG = 0.2
DEFAULT_BASE_ROTATE_STEP_DELAY_S = 0.10
# In bridge arm_to mode, base motion is available as manipulation base_x only.
MANIP_BASE_X_LIMITS = (-0.35, 0.35)

# v5 behavior constants (kept for full feature parity with robot_teleop_ui_ros_v5.py).
COMPENSATE_HEAD_ON_ROTATE = True
REACH_HEIGHT_CLEARANCE = 0.20
GRASP_PITCH_DEG = -30.0
GRASP_LATERAL_TRIM_M = 0.03
GRASP_REACH_TRIM_M = 0.025
GRASP_CLOSE_EXTRA_M = 0.015
GRASP_STALK_LENGTH_M = 0.2716
GRASP_RESIDUAL_ROT_GAIN = 0.60
GRASP_RESIDUAL_ROT_MAX_DEG = 8.0
GRASP_REACH_CORR_GAIN = 0.8
GRASP_REACH_CORR_MAX_STEP_M = 0.04
GRASP_REACH_CORR_THRESH_M = 0.008
GRASP_REACH_CORR_ITERS = 2
GRASP_PRELOWER_VERIFY_TIP_MARGIN_M = 0.03
GRASP_TIP_Z_MARGIN_M = 0.08
GRASP_TARGET_Z_OFFSET_M = 0.03
SCRIPT_STAGE_WAIT_CAP_S = 6.0
SAM_CC_MIN_AREA_PX = 600
SAM_CC_MIN_WIDTH_PX = 18
SAM_CC_MIN_HEIGHT_PX = 18

# v6: use stretch_ai IK/open-loop planning for reach/grasp instead of custom geometry.
USE_STRETCH_AI_IK_GRASP_PIPELINE = True
IK_PREGRASP_DISTANCE_M = 0.10
IK_LIFT_DISTANCE_M = 0.20
IK_SAFE_LIFT_M = 0.95
# Calibration offset applied to IK grasp base_x before execution.
# Positive pushes farther forward; negative pulls back.
IK_GRASP_BASE_X_OFFSET_M = 0.10
# Gripper partial-close tuning for grasp:
# close target = open_target - (IK_GRIPPER_CLOSE_DELTA_M / 0.22)
# where 0.22m per joint-unit comes from existing width->joint mapping.
IK_GRIPPER_CLOSE_DELTA_M = 0.035  # tune between 0.02 .. 0.05 (2-5 cm)
IK_GRIPPER_CLOSE_MIN_JOINT = -0.02  # prevent hard full-close motor load


class _DummyRclpy:
    class time:
        class Time:
            def __init__(self, *args, **kwargs):
                pass

    class duration:
        class Duration:
            def __init__(self, *, seconds: float = 0.0):
                self.seconds = float(seconds)


rclpy = _DummyRclpy()


class PointStamped:
    """Minimal geometry_msgs.msg.PointStamped-compatible container."""

    def __init__(self):
        self.header = types.SimpleNamespace(frame_id="", stamp=None)
        self.point = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)


###############################################################################
# Recorder (copied to preserve format)
###############################################################################
def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


class LeRobotStyleRecorder:
    """Episode recorder with LeRobot-like directory and naming conventions."""

    def __init__(self, robot_type="stretch3", target_fps=DEMO_RECORD_FPS, queue_maxsize=DEMO_RECORD_QUEUE_MAX):
        self.robot_type = robot_type
        self.target_fps = float(target_fps)
        self.queue_maxsize = int(queue_maxsize)
        self.active = False
        self.root = None
        self.prompt = ""
        self.episode_index = -1
        self.frame_index = 0
        self.start_wall_time = 0.0
        self.writers = {}
        self.paths = {}
        self.episode_name = ""
        self.chunk_name = ""
        self.data_format = "jsonl"
        self._sample_queue = queue.Queue(maxsize=self.queue_maxsize)
        self._writer_thread = None
        self._stop_requested = False
        self._last_sample_ts = None
        self._dropped_frames = 0
        self._row_file = None

    def _ensure(self, p: Path):
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _next_episode_index(self, root: Path):
        episodes_meta = root / "meta" / "episodes.jsonl"
        if not episodes_meta.exists():
            return 0
        idx = -1
        with open(episodes_meta, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    idx = max(idx, int(rec.get("episode_index", -1)))
                except Exception:
                    continue
        return idx + 1

    def _depth_preview_rgb(self, depth_m):
        depth = np.array(depth_m, dtype=np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth, 0.0, 3.0)
        depth_u8 = (depth / 3.0 * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    def start(self, dataset_root: str, prompt: str):
        root = Path(dataset_root).expanduser().resolve()
        self.root = root
        self.prompt = (prompt or "").strip()
        self.episode_index = self._next_episode_index(root)
        self.chunk_name = f"chunk-{self.episode_index // 1000:03d}"
        self.episode_name = f"episode_{self.episode_index:06d}"
        self.frame_index = 0
        self.start_wall_time = time.time()
        self.writers = {}
        self._last_sample_ts = None
        self._dropped_frames = 0
        self._stop_requested = False

        meta_dir = self._ensure(root / "meta")
        data_dir = self._ensure(root / "data" / self.chunk_name)
        images_dir = self._ensure(root / "images" / self.chunk_name)
        prompts_dir = self._ensure(root / "prompts" / self.chunk_name)
        depth_dir = self._ensure(root / "depth" / self.chunk_name)

        self.paths = {
            "meta_dir": meta_dir,
            "data_file_jsonl": data_dir / f"{self.episode_name}.jsonl",
            "prompt_file": prompts_dir / f"{self.episode_name}.txt",
            "head_rgb_frames": self._ensure(images_dir / "observation.images.head_rgb" / self.episode_name),
            "wrist_rgb_frames": self._ensure(images_dir / "observation.images.wrist_rgb" / self.episode_name),
            "head_depth_preview_frames": self._ensure(images_dir / "observation.images.head_depth" / self.episode_name),
            "wrist_depth_preview_frames": self._ensure(images_dir / "observation.images.wrist_depth" / self.episode_name),
            "head_depth_frames": self._ensure(depth_dir / "observation.depth.head" / self.episode_name),
            "wrist_depth_frames": self._ensure(depth_dir / "observation.depth.wrist" / self.episode_name),
        }

        with open(self.paths["prompt_file"], "w", encoding="utf-8") as f:
            f.write(self.prompt + "\n")

        self._row_file = open(self.paths["data_file_jsonl"], "w", encoding="utf-8", buffering=1)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self.active = True

    def _rel(self, p: Path):
        return str(p.relative_to(self.root))

    def record_step(self, sample: dict):
        if not self.active:
            return

        ts_abs = float(sample.get("timestamp", time.time()))
        if self._last_sample_ts is not None:
            min_dt = 1.0 / max(1e-6, self.target_fps)
            if (ts_abs - self._last_sample_ts) < min_dt:
                return
        self._last_sample_ts = ts_abs

        try:
            self._sample_queue.put_nowait(sample)
        except queue.Full:
            self._dropped_frames += 1

    def _process_step(self, sample: dict):
        ts_abs = float(sample.get("timestamp", time.time()))
        ts_rel = ts_abs - self.start_wall_time

        head_rgb = sample.get("head_rgb")
        wrist_rgb = sample.get("wrist_rgb")
        head_depth = sample.get("head_depth")
        wrist_depth = sample.get("wrist_depth")

        head_rgb_png_rel = None
        if head_rgb is not None:
            head_rgb_png = self.paths["head_rgb_frames"] / f"frame_{self.frame_index:06d}.png"
            cv2.imwrite(str(head_rgb_png), cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR))
            head_rgb_png_rel = self._rel(head_rgb_png)

        wrist_rgb_png_rel = None
        if wrist_rgb is not None:
            wrist_rgb_png = self.paths["wrist_rgb_frames"] / f"frame_{self.frame_index:06d}.png"
            cv2.imwrite(str(wrist_rgb_png), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
            wrist_rgb_png_rel = self._rel(wrist_rgb_png)

        head_depth_png_rel = None
        head_depth_preview_png_rel = None
        if head_depth is not None:
            head_depth_rgb = self._depth_preview_rgb(head_depth)
            head_depth_preview_png = self.paths["head_depth_preview_frames"] / f"frame_{self.frame_index:06d}.png"
            cv2.imwrite(str(head_depth_preview_png), cv2.cvtColor(head_depth_rgb, cv2.COLOR_RGB2BGR))
            head_depth_preview_png_rel = self._rel(head_depth_preview_png)
            depth_mm = np.clip(np.array(head_depth, dtype=np.float32) * 1000.0, 0, 65535).astype(np.uint16)
            depth_png = self.paths["head_depth_frames"] / f"frame_{self.frame_index:06d}.png"
            cv2.imwrite(str(depth_png), depth_mm)
            head_depth_png_rel = self._rel(depth_png)

        wrist_depth_png_rel = None
        wrist_depth_preview_png_rel = None
        if wrist_depth is not None:
            wrist_depth_rgb = self._depth_preview_rgb(wrist_depth)
            wrist_depth_preview_png = self.paths["wrist_depth_preview_frames"] / f"frame_{self.frame_index:06d}.png"
            cv2.imwrite(str(wrist_depth_preview_png), cv2.cvtColor(wrist_depth_rgb, cv2.COLOR_RGB2BGR))
            wrist_depth_preview_png_rel = self._rel(wrist_depth_preview_png)
            depth_mm = np.clip(np.array(wrist_depth, dtype=np.float32) * 1000.0, 0, 65535).astype(np.uint16)
            depth_png = self.paths["wrist_depth_frames"] / f"frame_{self.frame_index:06d}.png"
            cv2.imwrite(str(depth_png), depth_mm)
            wrist_depth_png_rel = self._rel(depth_png)

        row = {
            "episode_index": int(self.episode_index),
            "frame_index": int(self.frame_index),
            "timestamp": ts_abs,
            "timestamp_sec": ts_rel,
            "task": self.prompt,
            "observation.state": _to_jsonable(sample.get("state", [])),
            "action": _to_jsonable(sample.get("action", [])),
            "action_command": _to_jsonable(sample.get("action_command", [])),
            "observation.images.head_rgb": head_rgb_png_rel,
            "observation.images.wrist_rgb": wrist_rgb_png_rel,
            "observation.images.head_depth": head_depth_preview_png_rel,
            "observation.images.wrist_depth": wrist_depth_preview_png_rel,
            "observation.depth.head_frame": head_depth_png_rel,
            "observation.depth.wrist_frame": wrist_depth_png_rel,
        }
        sensors = _to_jsonable(sample.get("sensors", {}))
        for k, v in sensors.items():
            row[k] = v

        if self._row_file is not None:
            self._row_file.write(json.dumps(_to_jsonable(row), ensure_ascii=True) + "\n")
        self.frame_index += 1

    def _writer_loop(self):
        while True:
            if self._stop_requested and self._sample_queue.empty():
                break
            try:
                sample = self._sample_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if sample is None:
                self._sample_queue.task_done()
                break
            try:
                self._process_step(sample)
            except Exception as e:
                print(f"Recorder worker error: {e}")
            finally:
                self._sample_queue.task_done()

        if self._row_file is not None:
            try:
                self._row_file.flush()
                self._row_file.close()
            except Exception:
                pass
            self._row_file = None

    def _append_jsonl(self, path: Path, obj: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(obj), ensure_ascii=True) + "\n")

    def stop(self):
        if not self.active:
            return None

        self.active = False
        self._stop_requested = True
        try:
            self._sample_queue.put_nowait(None)
        except Exception:
            pass
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10.0)
            self._writer_thread = None

        data_rel = self._rel(self.paths["data_file_jsonl"])
        data_format = "jsonl"

        meta_dir = self.paths["meta_dir"]
        episodes_path = meta_dir / "episodes.jsonl"
        tasks_path = meta_dir / "tasks.jsonl"
        info_path = meta_dir / "info.json"

        ep_meta = {
            "episode_index": int(self.episode_index),
            "episode_name": self.episode_name,
            "length": int(self.frame_index),
            "task": self.prompt,
            "data_path": data_rel,
            "images": {
                "head_rgb": self._rel(self.paths["head_rgb_frames"]),
                "wrist_rgb": self._rel(self.paths["wrist_rgb_frames"]),
                "head_depth": self._rel(self.paths["head_depth_preview_frames"]),
                "wrist_depth": self._rel(self.paths["wrist_depth_preview_frames"]),
            },
            "prompt_path": self._rel(self.paths["prompt_file"]),
            "created_at": time.time(),
            "tabular_format": data_format,
            "dropped_frames": int(self._dropped_frames),
        }
        self._append_jsonl(episodes_path, ep_meta)
        self._append_jsonl(tasks_path, {
            "episode_index": int(self.episode_index),
            "task": self.prompt,
        })

        info = {}
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
        total_episodes = int(info.get("total_episodes", 0)) + 1
        total_frames = int(info.get("total_frames", 0)) + int(self.frame_index)
        info.update({
            "dataset_type": "lerobot_style",
            "codebase_version": "v2.1_style",
            "robot_type": self.robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "fps": self.target_fps,
            "rgb_storage": "png_frames",
            "depth_preview_storage": "png_frames",
            "tabular_format": data_format,
            "last_episode_dropped_frames": int(self._dropped_frames),
            "features": [
                "observation.images.head_rgb",
                "observation.images.wrist_rgb",
                "observation.images.head_depth",
                "observation.images.wrist_depth",
                "observation.depth.head_frame",
                "observation.depth.wrist_frame",
                "observation.state",
                "action",
                "sensors.*",
                "task",
            ],
        })
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

        summary = {
            "episode_index": int(self.episode_index),
            "num_frames": int(self.frame_index),
            "data_format": data_format,
            "dataset_root": str(self.root),
            "task": self.prompt,
            "dropped_frames": int(self._dropped_frames),
        }

        self._sample_queue = queue.Queue(maxsize=self.queue_maxsize)
        return summary


###############################################################################
# stretch_ai worker RPC
###############################################################################
class _StretchAIWorkerRPC:
    def __init__(self, cmd: list[str], *, env: dict[str, str], cwd: str | None = None):
        self._cmd = cmd
        self._env = env
        self._cwd = cwd
        self._proc: subprocess.Popen[Any] | None = None
        self._resp_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._req_lock = threading.Lock()
        self._next_id = 1

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self._cwd,
            env=self._env,
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            raise RuntimeError("Failed to create pipes for stretch_ai worker process")
        self._stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _stdout_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                # Worker responses are JSON objects. Ignore array/scalar JSON logs
                # emitted by dependencies so they do not break request matching.
                if isinstance(parsed, dict):
                    self._resp_queue.put(parsed)
                else:
                    print(f"[stretch_ai_worker] Ignoring non-object JSON stdout: {line}", file=sys.stderr)
            except Exception:
                # Treat arbitrary stdout text as log output; do not poison the RPC queue.
                print(f"[stretch_ai_worker] Ignoring non-JSON stdout: {line}", file=sys.stderr)

    def _stderr_loop(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            msg = line.rstrip()
            if msg:
                print(f"[stretch_ai_worker] {msg}", file=sys.stderr)

    def request(self, method: str, params: dict[str, Any] | None = None, *, timeout_s: float) -> dict[str, Any]:
        if self._proc is None:
            raise RuntimeError("stretch_ai worker process is not started")
        if self._proc.poll() is not None:
            raise RuntimeError(f"stretch_ai worker exited with code {self._proc.returncode}")
        if self._proc.stdin is None:
            raise RuntimeError("stretch_ai worker stdin is unavailable")

        with self._req_lock:
            req_id = self._next_id
            self._next_id += 1
            payload = {"id": req_id, "method": method, "params": params or {}}
            self._proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()

            deadline = time.time() + float(timeout_s)
            while True:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"stretch_ai worker exited with code {self._proc.returncode} during {method!r}"
                    )
                remaining = max(0.0, deadline - time.time())
                if remaining <= 0.0:
                    raise TimeoutError(f"Timeout waiting for worker response to {method!r}")
                try:
                    resp = self._resp_queue.get(timeout=remaining)
                except queue.Empty as exc:
                    if self._proc.poll() is not None:
                        raise RuntimeError(
                            f"stretch_ai worker exited with code {self._proc.returncode} during {method!r}"
                        ) from exc
                    raise TimeoutError(f"Timeout waiting for worker response to {method!r}") from exc
                if not isinstance(resp, dict):
                    continue
                if resp.get("id") != req_id:
                    continue
                if not resp.get("ok", False):
                    tb = resp.get("traceback")
                    msg = str(resp.get("error", "Unknown worker error"))
                    if tb:
                        msg = f"{msg}\n{tb}"
                    raise RuntimeError(msg)
                return resp.get("result", {})

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    self.request("close", {}, timeout_s=2.0)
                except Exception:
                    pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self._proc = None


###############################################################################
# Bridge backend
###############################################################################
class _DummyNow:
    def to_msg(self):
        sec = int(time.time())
        nanosec = int((time.time() - sec) * 1e9)
        return {"sec": sec, "nanosec": nanosec}


class _DummyClock:
    def now(self):
        return _DummyNow()


class _CameraInfoCompat:
    def __init__(self, raw: dict[str, Any]):
        self.width = int(raw.get("width", 0))
        self.height = int(raw.get("height", 0))
        self.k = [float(v) for v in (raw.get("k") or [0.0] * 9)]
        self.d = [float(v) for v in (raw.get("d") or [])]
        self.r = [float(v) for v in (raw.get("r") or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])]
        self.p = [float(v) for v in (raw.get("p") or [0.0] * 12)]
        self.distortion_model = str(raw.get("distortion_model", "plumb_bob"))


def _quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    if q.shape[0] < 4:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q[:4]
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_dict_to_msg(tf: dict[str, Any]):
    t = tf.get("translation") or [0.0, 0.0, 0.0]
    q = tf.get("rotation") or [0.0, 0.0, 0.0, 1.0]
    return types.SimpleNamespace(
        header=types.SimpleNamespace(
            frame_id=str(tf.get("target_frame", "")),
            stamp={"sec": int(time.time()), "nanosec": 0},
        ),
        child_frame_id=str(tf.get("source_frame", "")),
        transform=types.SimpleNamespace(
            translation=types.SimpleNamespace(x=float(t[0]), y=float(t[1]), z=float(t[2])),
            rotation=types.SimpleNamespace(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
        ),
    )


def _apply_transform_point(point_stamped: PointStamped, tf_msg) -> PointStamped:
    q = np.array(
        [
            float(tf_msg.transform.rotation.x),
            float(tf_msg.transform.rotation.y),
            float(tf_msg.transform.rotation.z),
            float(tf_msg.transform.rotation.w),
        ],
        dtype=np.float64,
    )
    r = _quat_xyzw_to_rotmat(q)
    t = np.array(
        [
            float(tf_msg.transform.translation.x),
            float(tf_msg.transform.translation.y),
            float(tf_msg.transform.translation.z),
        ],
        dtype=np.float64,
    )
    p = np.array([float(point_stamped.point.x), float(point_stamped.point.y), float(point_stamped.point.z)])
    out = PointStamped()
    out.header.frame_id = str(tf_msg.header.frame_id)
    out.header.stamp = point_stamped.header.stamp
    p_out = r @ p + t
    out.point.x = float(p_out[0])
    out.point.y = float(p_out[1])
    out.point.z = float(p_out[2])
    return out


class _BridgeTFBuffer:
    def __init__(self, bridge: "StretchAIDemoBridge"):
        self._bridge = bridge

    def lookup_transform(self, target_frame: str, source_frame: str, *args, **kwargs):
        return self._bridge.lookup_transform(target_frame, source_frame)


class RosTopicImageClient:
    """Low-latency ROS topic observation client via rosbridge topics."""

    def __init__(
        self,
        *,
        host: str = ROSBRIDGE_HOST,
        port: int = ROSBRIDGE_PORT,
        head_rgb_topic: str = HEAD_RGB_TOPIC,
        head_depth_topic: str = HEAD_DEPTH_TOPIC,
        head_info_topic: str = HEAD_CAMERA_INFO_TOPIC,
        wrist_rgb_topic: str = WRIST_RGB_TOPIC,
        wrist_depth_topic: str = WRIST_DEPTH_TOPIC,
        wrist_info_topic: str = WRIST_CAMERA_INFO_TOPIC,
        joint_topic: str = JOINT_STATE_TOPIC,
        odom_topic: str = ODOM_TOPIC,
        imu_mobile_base_topic: str = IMU_MOBILE_BASE_TOPIC,
        imu_wrist_topic: str = IMU_WRIST_TOPIC,
        imu_camera_accel_topic: str = IMU_CAMERA_ACCEL_TOPIC,
        imu_camera_gyro_topic: str = IMU_CAMERA_GYRO_TOPIC,
        magnetometer_topic: str = MAGNETOMETER_TOPIC,
        battery_topic: str = BATTERY_TOPIC,
    ):
        self._host = str(host)
        self._port = int(port)
        self._head_rgb_topic_name = str(head_rgb_topic)
        self._head_depth_topic_name = str(head_depth_topic)
        self._head_info_topic_name = str(head_info_topic)
        self._wrist_rgb_topic_name = str(wrist_rgb_topic)
        self._wrist_depth_topic_name = str(wrist_depth_topic)
        self._wrist_info_topic_name = str(wrist_info_topic)
        self._joint_topic_name = str(joint_topic)
        self._odom_topic_name = str(odom_topic)
        self._imu_mobile_base_topic_name = str(imu_mobile_base_topic)
        self._imu_wrist_topic_name = str(imu_wrist_topic)
        self._imu_camera_accel_topic_name = str(imu_camera_accel_topic)
        self._imu_camera_gyro_topic_name = str(imu_camera_gyro_topic)
        self._magnetometer_topic_name = str(magnetometer_topic)
        self._battery_topic_name = str(battery_topic)

        self._lock = threading.Lock()
        self._ros = None
        self._head_rgb_sub = None
        self._head_depth_sub = None
        self._head_info_sub = None
        self._wrist_rgb_sub = None
        self._wrist_depth_sub = None
        self._wrist_info_sub = None
        self._joint_sub = None
        self._odom_sub = None
        self._imu_mobile_sub = None
        self._imu_wrist_sub = None
        self._imu_cam_accel_sub = None
        self._imu_cam_gyro_sub = None
        self._mag_sub = None
        self._battery_sub = None
        self._connected = False

        self._head_rgb: np.ndarray | None = None
        self._wrist_rgb: np.ndarray | None = None
        self._head_depth: np.ndarray | None = None
        self._wrist_depth: np.ndarray | None = None
        self._head_info: dict[str, Any] | None = None
        self._wrist_info: dict[str, Any] | None = None
        self._warned_head_depth_shape = False
        self._warned_wrist_depth_shape = False

        self._actual_qpos: list[float] | None = None
        self._base_pose_xytheta: list[float] | None = None
        self._joint_state_name: list[str] = []
        self._joint_state_position: list[float] = []
        self._joint_state_velocity: list[float] = []
        self._joint_state_effort: list[float] = []
        self._imu_mobile: dict[str, Any] | None = None
        self._imu_wrist: dict[str, Any] | None = None
        self._imu_cam_accel: dict[str, Any] | None = None
        self._imu_cam_gyro: dict[str, Any] | None = None
        self._mag_mobile: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None
        self._odom: dict[str, Any] | None = None
        self._base_lin = 0.0
        self._base_ang = 0.0
        self._last_cb_error_t = 0.0

    def _log_cb_error(self, cb_name: str, exc: Exception) -> None:
        now = time.time()
        if now - self._last_cb_error_t < 1.0:
            return
        self._last_cb_error_t = now
        print(f"[ros_topic_image_client] callback {cb_name} error: {exc}", file=sys.stderr, flush=True)

    @staticmethod
    def _decode_raw_image_rgb(msg: dict[str, Any]) -> np.ndarray | None:
        data_b64 = msg.get("data")
        if not isinstance(data_b64, str):
            return None
        try:
            width = int(msg.get("width"))
            height = int(msg.get("height"))
            step = int(msg.get("step", 0))
            encoding = str(msg.get("encoding", "")).lower()
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        try:
            buf = base64.b64decode(data_b64)
        except Exception:
            return None

        def _reshape(bytes_per_pixel: int) -> np.ndarray | None:
            row_stride = step if step > 0 else width * bytes_per_pixel
            expected = row_stride * height
            if len(buf) < expected:
                return None
            raw = np.frombuffer(buf, dtype=np.uint8, count=expected).reshape(height, row_stride)
            return raw[:, : width * bytes_per_pixel].reshape(height, width, bytes_per_pixel)

        if encoding == "rgb8":
            arr = _reshape(3)
            return None if arr is None else arr.copy()
        if encoding == "bgr8":
            arr = _reshape(3)
            return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if encoding == "rgba8":
            arr = _reshape(4)
            return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        if encoding == "bgra8":
            arr = _reshape(4)
            return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        if encoding in {"mono8", "8uc1"}:
            arr = _reshape(1)
            if arr is None:
                return None
            return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2RGB)
        return None

    @staticmethod
    def _decode_raw_depth_m(msg: dict[str, Any]) -> np.ndarray | None:
        data_b64 = msg.get("data")
        if not isinstance(data_b64, str):
            return None
        try:
            width = int(msg.get("width"))
            height = int(msg.get("height"))
            step = int(msg.get("step", 0))
            encoding = str(msg.get("encoding", "")).lower()
            is_big = int(msg.get("is_bigendian", 0))
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        try:
            buf = base64.b64decode(data_b64)
        except Exception:
            return None

        if encoding in {"16uc1", "mono16"}:
            bpp = 2
            row_stride = step if step > 0 else width * bpp
            expected = row_stride * height
            if len(buf) < expected:
                return None
            raw = np.frombuffer(buf, dtype=np.uint8, count=expected).reshape(height, row_stride)
            pix = raw[:, : width * bpp]
            dtype = np.dtype(">u2") if is_big else np.dtype("<u2")
            depth_u16 = pix.view(dtype).reshape(height, width)
            return depth_u16.astype(np.float32) / 1000.0

        if encoding in {"32fc1"}:
            bpp = 4
            row_stride = step if step > 0 else width * bpp
            expected = row_stride * height
            if len(buf) < expected:
                return None
            raw = np.frombuffer(buf, dtype=np.uint8, count=expected).reshape(height, row_stride)
            pix = raw[:, : width * bpp]
            dtype = np.dtype(">f4") if is_big else np.dtype("<f4")
            depth_f32 = pix.view(dtype).reshape(height, width).astype(np.float32)
            depth_f32[~np.isfinite(depth_f32)] = 0.0
            return depth_f32

        return None

    @staticmethod
    def _camera_info_dict(msg: dict[str, Any]) -> dict[str, Any] | None:
        try:
            width = int(msg.get("width", 0))
            height = int(msg.get("height", 0))
            k = [float(v) for v in (msg.get("k") or [0.0] * 9)]
            d = [float(v) for v in (msg.get("d") or [])]
            r = [float(v) for v in (msg.get("r") or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])]
            p = [float(v) for v in (msg.get("p") or [0.0] * 12)]
            distortion_model = str(msg.get("distortion_model", "plumb_bob"))
        except (TypeError, ValueError):
            return None
        return {
            "width": width,
            "height": height,
            "k": k,
            "d": d,
            "r": r,
            "p": p,
            "distortion_model": distortion_model,
        }

    @staticmethod
    def _imu_to_dict(msg: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(msg, dict):
            return None
        ori = msg.get("orientation", {})
        av = msg.get("angular_velocity", {})
        la = msg.get("linear_acceleration", {})
        try:
            return {
                "orientation": [
                    float(ori.get("x", 0.0)),
                    float(ori.get("y", 0.0)),
                    float(ori.get("z", 0.0)),
                    float(ori.get("w", 1.0)),
                ],
                "angular_velocity": [
                    float(av.get("x", 0.0)),
                    float(av.get("y", 0.0)),
                    float(av.get("z", 0.0)),
                ],
                "linear_acceleration": [
                    float(la.get("x", 0.0)),
                    float(la.get("y", 0.0)),
                    float(la.get("z", 0.0)),
                ],
            }
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _mag_to_dict(msg: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(msg, dict):
            return None
        field = msg.get("magnetic_field", {})
        try:
            return {
                "magnetic_field": [
                    float(field.get("x", 0.0)),
                    float(field.get("y", 0.0)),
                    float(field.get("z", 0.0)),
                ]
            }
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _battery_to_dict(msg: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(msg, dict):
            return None
        try:
            return {
                "voltage": float(msg.get("voltage", 0.0)),
                "current": float(msg.get("current", 0.0)),
                "charge": float(msg.get("charge", 0.0)),
                "capacity": float(msg.get("capacity", 0.0)),
                "percentage": float(msg.get("percentage", 0.0)),
                "power_supply_status": int(msg.get("power_supply_status", 0)),
                "power_supply_health": int(msg.get("power_supply_health", 0)),
                "power_supply_technology": int(msg.get("power_supply_technology", 0)),
                "present": bool(msg.get("present", False)),
            }
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_qpos(msg: dict[str, Any], *, base_lin: float, base_ang: float) -> list[float] | None:
        names = msg.get("name", [])
        positions = msg.get("position", [])
        if not isinstance(names, list) or not isinstance(positions, list):
            return None
        try:
            idx = {str(name): i for i, name in enumerate(names)}
            arm_lift = positions[idx["joint_lift"]]
            arm_extension = 4.0 * positions[idx["joint_arm_l0"]]
            wrist_yaw = positions[idx["joint_wrist_yaw"]]
            wrist_pitch = positions[idx["joint_wrist_pitch"]]
            wrist_roll = positions[idx["joint_wrist_roll"]]
            head_pan = positions[idx["joint_head_pan"]]
            head_tilt = positions[idx["joint_head_tilt"]]
            gripper = positions[idx["joint_gripper_finger_left"]]
        except (KeyError, IndexError, TypeError, ValueError):
            return None

        return [
            float(arm_extension),
            float(arm_lift),
            float(wrist_yaw),
            float(wrist_pitch),
            float(wrist_roll),
            float(head_pan),
            float(head_tilt),
            float(gripper),
            float(base_lin),
            float(base_ang),
        ]

    def _head_rgb_cb(self, msg: dict[str, Any]) -> None:
        try:
            img = self._decode_raw_image_rgb(msg)
            if img is None:
                return
            if ROS_TOPICS_ROTATE_HEAD_90_CW:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            with self._lock:
                self._head_rgb = img
        except Exception as exc:
            self._log_cb_error("head_rgb", exc)

    def _wrist_rgb_cb(self, msg: dict[str, Any]) -> None:
        try:
            img = self._decode_raw_image_rgb(msg)
            if img is None:
                return
            with self._lock:
                self._wrist_rgb = img
        except Exception as exc:
            self._log_cb_error("wrist_rgb", exc)

    def _head_depth_cb(self, msg: dict[str, Any]) -> None:
        try:
            depth = self._decode_raw_depth_m(msg)
            if depth is None:
                return
            if ROS_TOPICS_ROTATE_HEAD_90_CW:
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            with self._lock:
                self._head_depth = depth
        except Exception as exc:
            self._log_cb_error("head_depth", exc)

    def _wrist_depth_cb(self, msg: dict[str, Any]) -> None:
        try:
            depth = self._decode_raw_depth_m(msg)
            if depth is None:
                return
            with self._lock:
                self._wrist_depth = depth
        except Exception as exc:
            self._log_cb_error("wrist_depth", exc)

    def _head_info_cb(self, msg: dict[str, Any]) -> None:
        try:
            info = self._camera_info_dict(msg)
            if info is None:
                return
            with self._lock:
                self._head_info = info
        except Exception as exc:
            self._log_cb_error("head_camera_info", exc)

    def _wrist_info_cb(self, msg: dict[str, Any]) -> None:
        try:
            info = self._camera_info_dict(msg)
            if info is None:
                return
            with self._lock:
                self._wrist_info = info
        except Exception as exc:
            self._log_cb_error("wrist_camera_info", exc)

    def _joint_cb(self, msg: dict[str, Any]) -> None:
        try:
            with self._lock:
                base_lin = float(self._base_lin)
                base_ang = float(self._base_ang)

            measured = self._extract_qpos(msg, base_lin=base_lin, base_ang=base_ang)

            names = msg.get("name", [])
            positions = msg.get("position", [])
            velocities = msg.get("velocity", [])
            efforts = msg.get("effort", [])
            with self._lock:
                self._joint_state_name = [str(v) for v in names] if isinstance(names, list) else []
                self._joint_state_position = [float(v) for v in positions] if isinstance(positions, list) else []
                self._joint_state_velocity = [float(v) for v in velocities] if isinstance(velocities, list) else []
                self._joint_state_effort = [float(v) for v in efforts] if isinstance(efforts, list) else []
                if measured is not None:
                    self._actual_qpos = measured
        except Exception as exc:
            self._log_cb_error("joint_state", exc)

    def _odom_cb(self, msg: dict[str, Any]) -> None:
        try:
            pose = msg.get("pose", {}).get("pose", {})
            pos = pose.get("position", {})
            ori = pose.get("orientation", {})
            twist = msg.get("twist", {}).get("twist", {})
            lin = twist.get("linear", {})
            ang = twist.get("angular", {})
            try:
                x = float(pos.get("x", 0.0))
                y = float(pos.get("y", 0.0))
                z = float(pos.get("z", 0.0))
                qx = float(ori.get("x", 0.0))
                qy = float(ori.get("y", 0.0))
                qz = float(ori.get("z", 0.0))
                qw = float(ori.get("w", 1.0))
                lin_x = float(lin.get("x", 0.0))
                lin_y = float(lin.get("y", 0.0))
                lin_z = float(lin.get("z", 0.0))
                ang_x = float(ang.get("x", 0.0))
                ang_y = float(ang.get("y", 0.0))
                ang_z = float(ang.get("z", 0.0))
            except (TypeError, ValueError):
                return

            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            theta = float(np.arctan2(siny_cosp, cosy_cosp))

            odom_dict = {
                "position": [x, y, z],
                "orientation": [qx, qy, qz, qw],
                "linear_velocity": [lin_x, lin_y, lin_z],
                "angular_velocity": [ang_x, ang_y, ang_z],
            }

            with self._lock:
                self._base_pose_xytheta = [x, y, theta]
                self._base_lin = lin_x
                self._base_ang = ang_z
                self._odom = odom_dict
                if self._actual_qpos is not None and len(self._actual_qpos) >= 10:
                    self._actual_qpos[8] = lin_x
                    self._actual_qpos[9] = ang_z
        except Exception as exc:
            self._log_cb_error("odom", exc)

    def _imu_mobile_cb(self, msg: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._imu_mobile = self._imu_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("imu_mobile", exc)

    def _imu_wrist_cb(self, msg: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._imu_wrist = self._imu_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("imu_wrist", exc)

    def _imu_cam_accel_cb(self, msg: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._imu_cam_accel = self._imu_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("imu_cam_accel", exc)

    def _imu_cam_gyro_cb(self, msg: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._imu_cam_gyro = self._imu_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("imu_cam_gyro", exc)

    def _mag_cb(self, msg: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._mag_mobile = self._mag_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("magnetometer", exc)

    def _battery_cb(self, msg: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._battery = self._battery_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("battery", exc)

    def connect(self, timeout_s: float = ROSBRIDGE_TIMEOUT_S) -> None:
        if self._connected:
            return
        if roslibpy is None:
            raise RuntimeError("roslibpy is not installed in this environment")

        ros = roslibpy.Ros(host=self._host, port=self._port)
        ros.run()
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline and not ros.is_connected:
            time.sleep(0.05)
        if not ros.is_connected:
            try:
                ros.terminate()
            except Exception:
                pass
            raise RuntimeError(f"Failed to connect to rosbridge ws://{self._host}:{self._port}")

        head_rgb_sub = roslibpy.Topic(
            ros, self._head_rgb_topic_name, "sensor_msgs/msg/Image", throttle_rate=66, queue_length=1
        )
        wrist_rgb_sub = roslibpy.Topic(
            ros, self._wrist_rgb_topic_name, "sensor_msgs/msg/Image", throttle_rate=66, queue_length=1
        )
        head_depth_sub = roslibpy.Topic(
            ros, self._head_depth_topic_name, "sensor_msgs/msg/Image", throttle_rate=66, queue_length=1
        )
        wrist_depth_sub = roslibpy.Topic(
            ros, self._wrist_depth_topic_name, "sensor_msgs/msg/Image", throttle_rate=66, queue_length=1
        )
        head_info_sub = roslibpy.Topic(
            ros, self._head_info_topic_name, "sensor_msgs/msg/CameraInfo", throttle_rate=500, queue_length=1
        )
        wrist_info_sub = roslibpy.Topic(
            ros, self._wrist_info_topic_name, "sensor_msgs/msg/CameraInfo", throttle_rate=500, queue_length=1
        )
        joint_sub = roslibpy.Topic(
            ros, self._joint_topic_name, "sensor_msgs/msg/JointState", throttle_rate=50, queue_length=1
        )
        odom_sub = roslibpy.Topic(
            ros, self._odom_topic_name, "nav_msgs/msg/Odometry", throttle_rate=50, queue_length=1
        )
        imu_mobile_sub = roslibpy.Topic(
            ros, self._imu_mobile_base_topic_name, "sensor_msgs/msg/Imu", throttle_rate=100, queue_length=1
        )
        imu_wrist_sub = roslibpy.Topic(
            ros, self._imu_wrist_topic_name, "sensor_msgs/msg/Imu", throttle_rate=100, queue_length=1
        )
        imu_cam_accel_sub = roslibpy.Topic(
            ros, self._imu_camera_accel_topic_name, "sensor_msgs/msg/Imu", throttle_rate=100, queue_length=1
        )
        imu_cam_gyro_sub = roslibpy.Topic(
            ros, self._imu_camera_gyro_topic_name, "sensor_msgs/msg/Imu", throttle_rate=100, queue_length=1
        )
        mag_sub = roslibpy.Topic(
            ros, self._magnetometer_topic_name, "sensor_msgs/msg/MagneticField", throttle_rate=100, queue_length=1
        )
        battery_sub = roslibpy.Topic(
            ros, self._battery_topic_name, "sensor_msgs/msg/BatteryState", throttle_rate=300, queue_length=1
        )

        head_rgb_sub.subscribe(self._head_rgb_cb)
        wrist_rgb_sub.subscribe(self._wrist_rgb_cb)
        head_depth_sub.subscribe(self._head_depth_cb)
        wrist_depth_sub.subscribe(self._wrist_depth_cb)
        head_info_sub.subscribe(self._head_info_cb)
        wrist_info_sub.subscribe(self._wrist_info_cb)
        joint_sub.subscribe(self._joint_cb)
        odom_sub.subscribe(self._odom_cb)
        imu_mobile_sub.subscribe(self._imu_mobile_cb)
        imu_wrist_sub.subscribe(self._imu_wrist_cb)
        imu_cam_accel_sub.subscribe(self._imu_cam_accel_cb)
        imu_cam_gyro_sub.subscribe(self._imu_cam_gyro_cb)
        mag_sub.subscribe(self._mag_cb)
        battery_sub.subscribe(self._battery_cb)

        self._ros = ros
        self._head_rgb_sub = head_rgb_sub
        self._wrist_rgb_sub = wrist_rgb_sub
        self._head_depth_sub = head_depth_sub
        self._wrist_depth_sub = wrist_depth_sub
        self._head_info_sub = head_info_sub
        self._wrist_info_sub = wrist_info_sub
        self._joint_sub = joint_sub
        self._odom_sub = odom_sub
        self._imu_mobile_sub = imu_mobile_sub
        self._imu_wrist_sub = imu_wrist_sub
        self._imu_cam_accel_sub = imu_cam_accel_sub
        self._imu_cam_gyro_sub = imu_cam_gyro_sub
        self._mag_sub = mag_sub
        self._battery_sub = battery_sub
        self._connected = True

    def close(self) -> None:
        subs = [
            self._head_rgb_sub,
            self._wrist_rgb_sub,
            self._head_depth_sub,
            self._wrist_depth_sub,
            self._head_info_sub,
            self._wrist_info_sub,
            self._joint_sub,
            self._odom_sub,
            self._imu_mobile_sub,
            self._imu_wrist_sub,
            self._imu_cam_accel_sub,
            self._imu_cam_gyro_sub,
            self._mag_sub,
            self._battery_sub,
        ]
        for sub in subs:
            if sub is None:
                continue
            try:
                sub.unsubscribe()
            except Exception:
                pass

        if self._ros is not None:
            try:
                self._ros.terminate()
            except Exception:
                pass

        self._head_rgb_sub = None
        self._wrist_rgb_sub = None
        self._head_depth_sub = None
        self._wrist_depth_sub = None
        self._head_info_sub = None
        self._wrist_info_sub = None
        self._joint_sub = None
        self._odom_sub = None
        self._imu_mobile_sub = None
        self._imu_wrist_sub = None
        self._imu_cam_accel_sub = None
        self._imu_cam_gyro_sub = None
        self._mag_sub = None
        self._battery_sub = None
        self._ros = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return bool(self._connected)

    def get_snapshot(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any] | None, dict[str, Any] | None]:
        with self._lock:
            hr = None if self._head_rgb is None else self._head_rgb.copy()
            wr = None if self._wrist_rgb is None else self._wrist_rgb.copy()
            hd = None if self._head_depth is None else self._head_depth.copy()
            wd = None if self._wrist_depth is None else self._wrist_depth.copy()
            hi = None if self._head_info is None else dict(self._head_info)
            wi = None if self._wrist_info is None else dict(self._wrist_info)

        # Keep depth grids aligned to RGB for segmentation/click-to-3D workflows.
        if hr is not None and hd is not None and hd.shape[:2] != hr.shape[:2]:
            if not self._warned_head_depth_shape:
                print(
                    f"[ros_topic_image_client] head depth/rgb shape mismatch: "
                    f"depth={hd.shape[:2]} rgb={hr.shape[:2]} (resizing depth to rgb)",
                    file=sys.stderr,
                    flush=True,
                )
                self._warned_head_depth_shape = True
            hd = cv2.resize(hd.astype(np.float32), (int(hr.shape[1]), int(hr.shape[0])), interpolation=cv2.INTER_NEAREST)
        if wr is not None and wd is not None and wd.shape[:2] != wr.shape[:2]:
            if not self._warned_wrist_depth_shape:
                print(
                    f"[ros_topic_image_client] wrist depth/rgb shape mismatch: "
                    f"depth={wd.shape[:2]} rgb={wr.shape[:2]} (resizing depth to rgb)",
                    file=sys.stderr,
                    flush=True,
                )
                self._warned_wrist_depth_shape = True
            wd = cv2.resize(wd.astype(np.float32), (int(wr.shape[1]), int(wr.shape[0])), interpolation=cv2.INTER_NEAREST)
        return hr, wr, hd, wd, hi, wi

    def get_observation_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "actual_qpos": list(self._actual_qpos) if self._actual_qpos is not None else None,
                "base_pose_xytheta": list(self._base_pose_xytheta) if self._base_pose_xytheta is not None else None,
                "joint_state_name": list(self._joint_state_name),
                "joint_state_position": list(self._joint_state_position),
                "joint_state_velocity": list(self._joint_state_velocity),
                "joint_state_effort": list(self._joint_state_effort),
                "imu_mobile_base": None if self._imu_mobile is None else dict(self._imu_mobile),
                "imu_wrist": None if self._imu_wrist is None else dict(self._imu_wrist),
                "imu_camera_accel": None if self._imu_cam_accel is None else dict(self._imu_cam_accel),
                "imu_camera_gyro": None if self._imu_cam_gyro is None else dict(self._imu_cam_gyro),
                "magnetometer_mobile_base": None if self._mag_mobile is None else dict(self._mag_mobile),
                "battery": None if self._battery is None else dict(self._battery),
                "odom": None if self._odom is None else dict(self._odom),
                "camera_info_head": None if self._head_info is None else dict(self._head_info),
                "camera_info_wrist": None if self._wrist_info is None else dict(self._wrist_info),
            }


class StretchAIDemoBridge:
    JOINT_LIMITS = [
        (0.00, 0.51),
        (0.08, 1.05),
        (-2.6, 2.6),
        (-1.0, 1.57),
        (-1.57, 1.57),
        (-1.57, 1.57),
        (-1.0, 1.0),
        (-0.1, 0.5501),
        (-2.0, 2.0),
        (-5.0, 5.0),
    ]

    CONTROL_MAP = {
        "arm_extension": 0,
        "arm_lift": 1,
        "wrist_yaw": 2,
        "wrist_pitch": 3,
        "wrist_roll": 4,
        "head_pan": 5,
        "head_tilt": 6,
        "gripper": 7,
        "base_linear": 8,
        "base_angular": 9,
    }

    JOINT_STATE_NAMES = [
        "joint_base_x",
        "joint_base_y",
        "joint_base_theta",
        "joint_lift",
        "joint_arm_l0",
        "joint_gripper_finger_left",
        "joint_wrist_roll",
        "joint_wrist_pitch",
        "joint_wrist_yaw",
        "joint_head_pan",
        "joint_head_tilt",
    ]

    def __init__(self):
        self._lock = threading.Lock()
        self._rpc: _StretchAIWorkerRPC | None = None
        self._poll_thread: threading.Thread | None = None
        self._command_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self.head_rgb: np.ndarray | None = None
        self.wrist_rgb: np.ndarray | None = None
        self.head_depth: np.ndarray | None = None
        self.wrist_depth: np.ndarray | None = None

        self.actual_qpos: list[float] | None = None
        self.qpos: list[float] | None = None
        self.published_qpos: list[float] | None = None

        self._base_x = 0.0
        self._base_y = 0.0
        self._base_theta = 0.0
        self._manip_base_x = 0.0
        self.command_base_pose_xytheta: list[float] | None = None
        self.command_base_pose_last_wall_time: float | None = None

        self._joint_positions: list[float] | None = None
        self._joint_velocities: list[float] | None = None
        self._joint_efforts: list[float] | None = None
        self._camera_info_head: dict[str, Any] | None = None
        self._camera_info_wrist: dict[str, Any] | None = None
        self._mode: str = "unknown"
        self._at_goal: bool | None = None
        self._is_homed: bool | None = None
        self._is_runstopped: bool | None = None

        self._last_exec_result: dict[str, Any] | None = None
        self._last_cmd_error_t = 0.0
        self.command_smooth_delay_s = float(DEFAULT_COMMAND_SMOOTH_DELAY_S)
        self.base_rotate_step_rad = float(np.deg2rad(DEFAULT_BASE_ROTATE_STEP_DEG))
        self.base_rotate_step_delay_s = float(DEFAULT_BASE_ROTATE_STEP_DELAY_S)
        self._base_linear_cmd = 0.0
        self._base_angular_cmd = 0.0
        self._base_cmd_last_wall_time: float | None = None
        self._last_base_step_error_t = 0.0
        self._last_base_theta_warn_t = 0.0
        self._needs_mode_retry = False
        self._clock = _DummyClock()
        self.tf_buffer = _BridgeTFBuffer(self)
        self.odom = None
        self._image_source: str = "bridge"
        self._ros_image_client: RosTopicImageClient | None = None
        self._next_ros_topics_connect_t: float = 0.0

    @staticmethod
    def _decode_jpg_rgb(data_b64: str | None) -> np.ndarray | None:
        if not data_b64:
            return None
        try:
            buf = base64.b64decode(data_b64)
            arr = np.frombuffer(buf, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    @staticmethod
    def _decode_depth_png(data_b64: str | None) -> np.ndarray | None:
        if not data_b64:
            return None
        try:
            buf = base64.b64decode(data_b64)
            arr = np.frombuffer(buf, dtype=np.uint8)
            depth_u16 = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if depth_u16 is None:
                return None
            if depth_u16.dtype != np.uint16:
                depth_u16 = depth_u16.astype(np.uint16)
            return depth_u16.astype(np.float32) / 1000.0
        except Exception:
            return None

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return float(np.arctan2(np.sin(theta), np.cos(theta)))

    def _worker_script_path(self) -> Path:
        return Path(__file__).with_name("stretch_ai_bridge_worker.py")

    def _worker_cmd(self) -> tuple[list[str], dict[str, str]]:
        worker_path = self._worker_script_path()
        if not worker_path.exists():
            raise RuntimeError(f"Missing worker script: {worker_path}")

        cmd = [STRETCH_AI_WORKER_LAUNCHER, "run", "-n", STRETCH_AI_ENV_NAME, STRETCH_AI_WORKER_PYTHON, "-u"]
        cmd.append(str(worker_path))
        cmd += ["--jpeg-quality", str(STRETCH_AI_WORKER_JPEG_QUALITY)]
        if STRETCH_AI_ROTATE_HEAD_90_CW:
            cmd.append("--rotate-head-90-cw")

        env = os.environ.copy()
        src_path = str(Path(STRETCH_AI_REPO) / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
        env["PYTHONNOUSERSITE"] = "1"
        if STRETCH_AI_WORKER_DEBUG:
            env["STRETCH_AI_WORKER_DEBUG"] = "1"
        return cmd, env

    def connect(self, timeout_s: float = STRETCH_AI_CONNECT_TIMEOUT_S) -> None:
        cmd, env = self._worker_cmd()
        self._rpc = _StretchAIWorkerRPC(cmd, env=env, cwd=str(Path(__file__).resolve().parent.parent))
        self._rpc.start()
        self._rpc.request("ping", {}, timeout_s=timeout_s)
        self._rpc.request(
            "connect",
            {
                "robot_ip": STRETCH_AI_ROBOT_IP,
                "use_remote_computer": STRETCH_AI_USE_REMOTE_COMPUTER,
                "recv_port": STRETCH_AI_RECV_PORT,
                "send_port": STRETCH_AI_SEND_PORT,
                "recv_state_port": STRETCH_AI_RECV_STATE_PORT,
                "recv_servo_port": STRETCH_AI_RECV_SERVO_PORT,
            },
            timeout_s=timeout_s,
        )

        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self._command_thread.start()

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            with self._lock:
                if self.actual_qpos is not None and self.head_rgb is not None:
                    return
            time.sleep(0.05)
        raise RuntimeError("stretch_ai worker connected, but no observations arrived before timeout")

    def _poll_loop(self) -> None:
        period = 1.0 / max(1e-3, float(STRETCH_AI_OBS_POLL_HZ))
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                if self._rpc is None:
                    break
                obs = self._rpc.request("observe", {}, timeout_s=STRETCH_AI_RPC_TIMEOUT_S)

                head = self._decode_jpg_rgb(obs.get("head_jpg_b64"))
                wrist = self._decode_jpg_rgb(obs.get("wrist_jpg_b64"))
                head_depth = self._decode_depth_png(obs.get("head_depth_png_b64"))
                wrist_depth = self._decode_depth_png(obs.get("wrist_depth_png_b64"))

                q = obs.get("actual_qpos")
                q_out = None
                if isinstance(q, list):
                    q_out = [float(v) for v in q[:10]]
                    if len(q_out) < 10:
                        q_out += [0.0] * (10 - len(q_out))

                base_pose = obs.get("base_pose")
                if isinstance(base_pose, list) and len(base_pose) >= 3:
                    bx, by, bt = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
                else:
                    bx, by, bt = self._base_x, self._base_y, self._base_theta
                manip_base_x_obs = obs.get("manip_base_x")
                if isinstance(manip_base_x_obs, (int, float)):
                    manip_base_x = float(manip_base_x_obs)
                else:
                    manip_base_x = self._manip_base_x

                with self._lock:
                    if head is not None:
                        self.head_rgb = head
                    if wrist is not None:
                        self.wrist_rgb = wrist
                    if head_depth is not None:
                        self.head_depth = head_depth
                    if wrist_depth is not None:
                        self.wrist_depth = wrist_depth
                    if q_out is not None:
                        self.actual_qpos = q_out
                        if self.qpos is None:
                            self.qpos = list(q_out)
                            self.qpos[8] = 0.0
                            self.qpos[9] = 0.0
                        if self.published_qpos is None:
                            self.published_qpos = list(self.qpos)
                    self._base_x = bx
                    self._base_y = by
                    self._base_theta = bt
                    self._manip_base_x = float(manip_base_x)
                    if self.command_base_pose_xytheta is None:
                        self.command_base_pose_xytheta = [bx, by, bt]
                    if self.command_base_pose_last_wall_time is None:
                        self.command_base_pose_last_wall_time = time.time()

                    self._joint_positions = obs.get("joint_positions") if isinstance(obs.get("joint_positions"), list) else None
                    self._joint_velocities = obs.get("joint_velocities") if isinstance(obs.get("joint_velocities"), list) else None
                    self._joint_efforts = obs.get("joint_efforts") if isinstance(obs.get("joint_efforts"), list) else None
                    self._camera_info_head = obs.get("camera_info_head") if isinstance(obs.get("camera_info_head"), dict) else None
                    self._camera_info_wrist = obs.get("camera_info_wrist") if isinstance(obs.get("camera_info_wrist"), dict) else None
                    mode = obs.get("mode")
                    if isinstance(mode, str) and mode:
                        self._mode = mode
                    self._at_goal = bool(obs.get("at_goal")) if obs.get("at_goal") is not None else None
                    self._is_homed = bool(obs.get("is_homed")) if obs.get("is_homed") is not None else None
                    self._is_runstopped = bool(obs.get("is_runstopped")) if obs.get("is_runstopped") is not None else None
                    lin_v = float(q_out[8]) if q_out is not None and len(q_out) > 8 else 0.0
                    ang_v = float(q_out[9]) if q_out is not None and len(q_out) > 9 else 0.0
                    self.odom = types.SimpleNamespace(
                        pose=types.SimpleNamespace(
                            pose=types.SimpleNamespace(
                                position=types.SimpleNamespace(x=float(bx), y=float(by), z=0.0),
                                orientation=types.SimpleNamespace(
                                    x=0.0,
                                    y=0.0,
                                    z=float(np.sin(bt / 2.0)),
                                    w=float(np.cos(bt / 2.0)),
                                ),
                            )
                        ),
                        twist=types.SimpleNamespace(
                            twist=types.SimpleNamespace(
                                linear=types.SimpleNamespace(x=lin_v, y=0.0, z=0.0),
                                angular=types.SimpleNamespace(x=0.0, y=0.0, z=ang_v),
                            )
                        ),
                    )
            except Exception:
                time.sleep(0.1)

            dt = time.time() - t0
            sleep_s = max(0.0, period - dt)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _command_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    dx = float(self._base_linear_cmd)
                    dtheta = float(self._base_angular_cmd)
                    # Consume latched base step once so a press issues one relative
                    # x/theta goal instead of repeated goals while held.
                    if (abs(dx) > 1e-6) or (abs(dtheta) > 1e-6):
                        self._base_linear_cmd = 0.0
                        self._base_angular_cmd = 0.0
                    self._base_cmd_last_wall_time = time.time()
                base_active = (abs(dx) > 1e-6) or (abs(dtheta) > 1e-6)

                # Base control path: apply fixed relative x/theta chunks each tick.
                if base_active:
                    ok = self.move_base_relative(
                        dx=dx,
                        dy=0.0,
                        dtheta=dtheta,
                        blocking=False,
                        timeout_s=max(1.5, 2.0 + 4.0 * (abs(dx) + abs(dtheta))),
                    )
                    # Track commanded base target in x/y/theta space for recording/debug.
                    self._advance_command_base_pose_by_relative_step(dx, 0.0, dtheta)
                    if not ok:
                        t_now = time.time()
                        if t_now - self._last_base_step_error_t > 2.0:
                            self._last_base_step_error_t = t_now
                            print("[stretch_ai_bridge] base relative step command failed", file=sys.stderr)

                pending = self.has_pending_command()
                if pending:
                    self.publish_commands(force=False)
            except Exception:
                pass
            time.sleep(max(0.01, float(self.command_smooth_delay_s)))

    def _integrate_command_base_pose_until(self, now_wall_time: float):
        if self.command_base_pose_xytheta is None:
            self.command_base_pose_xytheta = [self._base_x, self._base_y, self._base_theta]
        if self.command_base_pose_last_wall_time is None:
            self.command_base_pose_last_wall_time = float(now_wall_time)
            return
        dt = float(now_wall_time) - float(self.command_base_pose_last_wall_time)
        if dt <= 0.0:
            self.command_base_pose_last_wall_time = float(now_wall_time)
            return
        dt = min(dt, 1.0)
        if self.published_qpos is None or len(self.published_qpos) < 10:
            self.command_base_pose_last_wall_time = float(now_wall_time)
            return
        v = float(self.published_qpos[8])
        w = float(self.published_qpos[9])
        x, y, theta = [float(vv) for vv in self.command_base_pose_xytheta]
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta = self._wrap_angle(theta + w * dt)
        self.command_base_pose_xytheta = [x, y, theta]
        self.command_base_pose_last_wall_time = float(now_wall_time)

    def _advance_command_base_pose_by_relative_step(self, dx: float, dy: float, dtheta: float) -> None:
        """Advance command base x/y/theta by a relative step in robot frame."""
        with self._lock:
            if self.command_base_pose_xytheta is None:
                self.command_base_pose_xytheta = [self._base_x, self._base_y, self._base_theta]
            x, y, theta = [float(v) for v in self.command_base_pose_xytheta]
            x += float(dx) * np.cos(theta) - float(dy) * np.sin(theta)
            y += float(dx) * np.sin(theta) + float(dy) * np.cos(theta)
            theta = self._wrap_angle(theta + float(dtheta))
            self.command_base_pose_xytheta = [x, y, theta]
            self.command_base_pose_last_wall_time = time.time()

    def publish_commands(self, *, force: bool = False) -> None:
        rpc = self._rpc
        if rpc is None:
            return

        prev_published: list[float] | None = None
        with self._lock:
            if self.qpos is None:
                return
            if self.published_qpos is None or len(self.published_qpos) != len(self.qpos):
                self.published_qpos = list(self.qpos)
            prev_published = list(self.published_qpos)

            for i, target in enumerate(self.qpos):
                current = float(self.published_qpos[i])
                target = float(target)
                # Base is controlled via relative x/y/theta commands. Keep qpos base channels at zero.
                if i >= 8:
                    self.published_qpos[i] = 0.0
                    continue
                step_limit = float(COMMAND_SMOOTH_STEP_SIZES[i]) if i < len(COMMAND_SMOOTH_STEP_SIZES) else 0.0
                if step_limit <= 0.0:
                    self.published_qpos[i] = target
                    continue
                delta = target - current
                if abs(delta) <= step_limit:
                    self.published_qpos[i] = target
                else:
                    self.published_qpos[i] = current + (step_limit if delta > 0.0 else -step_limit)

            cmd = [float(v) for v in self.published_qpos]

        try:
            # print(cmd)
            result = rpc.request(
                "execute_qpos_cmd",
                {
                    "qpos_cmd": cmd,
                    "force": bool(force),
                    # Base is commanded through move_base_relative()/rotate_base_relative(),
                    # not through qpos base velocity channels.
                    "base_mode": "none",
                },
                timeout_s=max(STRETCH_AI_RPC_TIMEOUT_S, 20.0),
            )
            # print(result)
            retry_needed = False
            if isinstance(result, dict):
                if result.get("waiting_for_mode") is not None:
                    retry_needed = True
            with self._lock:
                self._last_exec_result = dict(result) if isinstance(result, dict) else {"result": result}
                mode = self._last_exec_result.get("mode") if self._last_exec_result else None
                if isinstance(mode, str) and mode:
                    self._mode = mode
                if retry_needed and prev_published is not None:
                    # Keep target->published delta so control tick retries after mode transition.
                    self.published_qpos = list(prev_published)
                    self._needs_mode_retry = True
                else:
                    self._needs_mode_retry = False
        except Exception as exc:
            now = time.time()
            with self._lock:
                if prev_published is not None:
                    self.published_qpos = list(prev_published)
                self._needs_mode_retry = True
            if now - self._last_cmd_error_t > 2.0:
                self._last_cmd_error_t = now
                print(f"[stretch_ai_bridge] command error: {exc}", file=sys.stderr)

    def set_control(self, control_name: str, value: float) -> None:
        idx = self.CONTROL_MAP.get(control_name)
        if idx is None:
            return
        actual_for_init = self.get_actual_qpos()
        with self._lock:
            if self.qpos is None:
                if not actual_for_init:
                    return
                self.qpos = list(actual_for_init[:10]) + [0.0] * max(0, 10 - len(actual_for_init))
                self.qpos[8] = 0.0
                self.qpos[9] = 0.0
            lo, hi = self.JOINT_LIMITS[idx]
            clipped = float(np.clip(value, lo, hi))
            if idx == 8:
                self._base_linear_cmd = clipped
                # Keep qpos base channels at zero; base is controlled via xyt relative actions.
                self.qpos[8] = 0.0
                self.qpos[9] = 0.0
                return
            if idx == 9:
                self._base_angular_cmd = clipped
                self.qpos[8] = 0.0
                self.qpos[9] = 0.0
                return
            self.qpos[idx] = clipped

    def adjust_control(self, control_name: str, delta: float) -> None:
        idx = self.CONTROL_MAP.get(control_name)
        if idx is None:
            return
        actual_for_init = self.get_actual_qpos()
        with self._lock:
            if self.qpos is None:
                if not actual_for_init:
                    return
                self.qpos = list(actual_for_init[:10]) + [0.0] * max(0, 10 - len(actual_for_init))
                self.qpos[8] = 0.0
                self.qpos[9] = 0.0
            lo, hi = self.JOINT_LIMITS[idx]
            if idx == 8:
                self._base_linear_cmd = float(np.clip(self._base_linear_cmd + float(delta), lo, hi))
                self.qpos[8] = 0.0
                self.qpos[9] = 0.0
                return
            if idx == 9:
                self._base_angular_cmd = float(np.clip(self._base_angular_cmd + float(delta), lo, hi))
                self.qpos[8] = 0.0
                self.qpos[9] = 0.0
                return
            self.qpos[idx] = float(np.clip(self.qpos[idx] + float(delta), lo, hi))

    def stop_base(self) -> None:
        self.set_control("base_linear", 0.0)
        self.set_control("base_angular", 0.0)
        with self._lock:
            self._base_cmd_last_wall_time = time.time()

    def move_base_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dtheta: float = 0.0,
        *,
        blocking: bool = False,
        timeout_s: float = 3.0,
    ) -> bool:
        """Send a relative base x/y/theta command through bridge navigation action."""
        rpc = self._rpc
        if rpc is None:
            return False
        try:
            res = rpc.request(
                "move_base_relative",
                {
                    "dx": float(dx),
                    "dy": float(dy),
                    "dtheta": float(dtheta),
                    "blocking": bool(blocking),
                    "timeout_s": float(timeout_s),
                },
                timeout_s=max(2.0, float(timeout_s) + 2.0),
            )
            # print(res)
            with self._lock:
                self._last_exec_result = dict(res) if isinstance(res, dict) else {"result": res}
                pose = res.get("base_pose") if isinstance(res, dict) else None
                if isinstance(pose, list) and len(pose) >= 3:
                    self.command_base_pose_xytheta = [float(pose[0]), float(pose[1]), float(pose[2])]
                    self.command_base_pose_last_wall_time = time.time()
            return bool(isinstance(res, dict) and res.get("ok", False))
        except Exception as exc:
            now = time.time()
            if now - self._last_cmd_error_t > 2.0:
                self._last_cmd_error_t = now
                print(f"[stretch_ai_bridge] move_base_relative error: {exc}", file=sys.stderr)
            return False

    def rotate_base_relative(self, theta_rad: float, timeout_s: float = 10.0) -> bool:
        """Rotate base by relative yaw angle using bridge xyt navigation action."""
        rpc = self._rpc
        if rpc is None:
            return False
        try:
            # Ensure streaming velocity channels are zero before one-shot rotate.
            with self._lock:
                if self.qpos is not None and len(self.qpos) >= 10:
                    self.qpos[8] = 0.0
                    self.qpos[9] = 0.0
                if self.published_qpos is not None and len(self.published_qpos) >= 10:
                    self.published_qpos[8] = 0.0
                    self.published_qpos[9] = 0.0
            res = rpc.request(
                "rotate_base_relative",
                {"theta_rad": float(theta_rad), "timeout_s": float(timeout_s)},
                timeout_s=max(5.0, float(timeout_s) + 3.0),
            )
            # print(res)
            with self._lock:
                self._last_exec_result = dict(res) if isinstance(res, dict) else {"result": res}
                pose = res.get("base_pose") if isinstance(res, dict) else None
                if isinstance(pose, list) and len(pose) >= 3:
                    self.command_base_pose_xytheta = [float(pose[0]), float(pose[1]), float(pose[2])]
                    self.command_base_pose_last_wall_time = time.time()
            return bool(isinstance(res, dict) and res.get("ok", False))
        except Exception as exc:
            now = time.time()
            if now - self._last_cmd_error_t > 2.0:
                self._last_cmd_error_t = now
                print(f"[stretch_ai_bridge] rotate_base_relative error: {exc}", file=sys.stderr)
            return False

    def execute_arm_to(
        self,
        joint6: list[float],
        *,
        gripper: float | None = None,
        head: list[float] | None = None,
        blocking: bool = True,
        timeout_s: float = 8.0,
        reliable: bool = False,
    ) -> bool:
        """Direct arm_to wrapper using stretch_ai client semantics.

        joint6 ordering:
          [base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll]
        """
        rpc = self._rpc
        if rpc is None:
            return False
        try:
            res = rpc.request(
                "execute_arm_to",
                {
                    "joint": [float(v) for v in np.asarray(joint6, dtype=np.float32).reshape(-1)[:6].tolist()],
                    "gripper": None if gripper is None else float(gripper),
                    "head": None if head is None else [float(head[0]), float(head[1])],
                    "blocking": bool(blocking),
                    "timeout_s": float(timeout_s),
                    "reliable": bool(reliable),
                },
                timeout_s=max(float(timeout_s) + 2.0, STRETCH_AI_RPC_TIMEOUT_S),
            )
            ok = bool(isinstance(res, dict) and res.get("ok", False))
            with self._lock:
                self._last_exec_result = dict(res) if isinstance(res, dict) else {"result": res}
                if ok:
                    # Keep local command targets aligned with the last successful arm_to pose
                    # so the next manual increment starts from current robot posture.
                    if self.qpos is None or len(self.qpos) < 10:
                        actual_now = self.get_actual_qpos()
                        self.qpos = list(actual_now[:10]) + [0.0] * max(0, 10 - len(actual_now)) if actual_now else ([0.0] * 10)
                    # joint6 ordering is [base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll].
                    # qpos ordering is  [arm,    lift, wrist_yaw, wrist_pitch, wrist_roll, ...].
                    # Keep manipulation base_x in its own field; do not write it into qpos[0].
                    self._manip_base_x = float(np.clip(float(joint6[0]), MANIP_BASE_X_LIMITS[0], MANIP_BASE_X_LIMITS[1]))
                    self.qpos[0] = float(np.clip(float(joint6[2]), self.JOINT_LIMITS[0][0], self.JOINT_LIMITS[0][1]))  # arm_extension
                    self.qpos[1] = float(np.clip(float(joint6[1]), self.JOINT_LIMITS[1][0], self.JOINT_LIMITS[1][1]))  # lift
                    self.qpos[2] = float(np.clip(float(joint6[3]), self.JOINT_LIMITS[2][0], self.JOINT_LIMITS[2][1]))  # wrist_yaw
                    self.qpos[3] = float(np.clip(float(joint6[4]), self.JOINT_LIMITS[3][0], self.JOINT_LIMITS[3][1]))  # wrist_pitch
                    self.qpos[4] = float(np.clip(float(joint6[5]), self.JOINT_LIMITS[4][0], self.JOINT_LIMITS[4][1]))  # wrist_roll
                    if gripper is not None:
                        self.qpos[7] = float(np.clip(float(gripper), self.JOINT_LIMITS[7][0], self.JOINT_LIMITS[7][1]))
                    if isinstance(head, list) and len(head) >= 2:
                        self.qpos[5] = float(np.clip(float(head[0]), self.JOINT_LIMITS[5][0], self.JOINT_LIMITS[5][1]))
                        self.qpos[6] = float(np.clip(float(head[1]), self.JOINT_LIMITS[6][0], self.JOINT_LIMITS[6][1]))
                    self.qpos[8] = 0.0
                    self.qpos[9] = 0.0
                    self.published_qpos = list(self.qpos)
                    self._base_linear_cmd = 0.0
                    self._base_angular_cmd = 0.0
                    self._needs_mode_retry = False
            if isinstance(res, dict) and res.get("status") == "accepted_stalled":
                now = time.time()
                if now - self._last_cmd_error_t > 2.0:
                    self._last_cmd_error_t = now
                    print(
                        "[stretch_ai_bridge] execute_arm_to reported stalled-but-accepted; continuing.",
                        file=sys.stderr,
                    )
            return ok
        except Exception as exc:
            now = time.time()
            if now - self._last_cmd_error_t > 2.0:
                self._last_cmd_error_t = now
                print(f"[stretch_ai_bridge] execute_arm_to error: {exc}", file=sys.stderr)
            return False

    def sync_command_targets_to_actual(self) -> bool:
        """Reset local command targets to measured robot state for safe manual continuation."""
        actual = self.get_actual_qpos()
        if len(actual) < 10:
            return False
        with self._lock:
            self.qpos = list(actual[:10])
            self.qpos[8] = 0.0
            self.qpos[9] = 0.0
            self.published_qpos = list(self.qpos)
            self._base_linear_cmd = 0.0
            self._base_angular_cmd = 0.0
            self._needs_mode_retry = False
        return True

    def plan_open_loop_grasp(
        self,
        object_xyz_global: tuple[float, float, float] | list[float] | np.ndarray,
        *,
        pregrasp_distance: float = IK_PREGRASP_DISTANCE_M,
        lift_distance: float = IK_LIFT_DISTANCE_M,
        wrist_yaw_target: float | None = None,
        wrist_pitch_target: float | None = None,
        wrist_roll_target: float | None = None,
        timeout_s: float = 35.0,
    ) -> dict[str, Any] | None:
        """Request stretch_ai open-loop grasp IK targets for a world-frame point."""
        rpc = self._rpc
        if rpc is None:
            return None
        try:
            res = rpc.request(
                "plan_open_loop_grasp",
                {
                    "object_xyz_global": [
                        float(object_xyz_global[0]),
                        float(object_xyz_global[1]),
                        float(object_xyz_global[2]),
                    ],
                    "pregrasp_distance": float(pregrasp_distance),
                    "lift_distance": float(lift_distance),
                    "wrist_yaw_target": None if wrist_yaw_target is None else float(wrist_yaw_target),
                    "wrist_pitch_target": None if wrist_pitch_target is None else float(wrist_pitch_target),
                    "wrist_roll_target": None if wrist_roll_target is None else float(wrist_roll_target),
                },
                timeout_s=max(float(timeout_s), STRETCH_AI_RPC_TIMEOUT_S),
            )
            return dict(res) if isinstance(res, dict) else {"ok": False, "error": "invalid plan response"}
        except Exception as exc:
            now = time.time()
            if now - self._last_cmd_error_t > 2.0:
                self._last_cmd_error_t = now
                print(f"[stretch_ai_bridge] plan_open_loop_grasp error: {exc}", file=sys.stderr)
            return {"ok": False, "error": str(exc)}

    def rotate_base_relative_chunked(
        self,
        theta_rad: float,
        *,
        step_rad: float | None = None,
        step_delay_s: float | None = None,
    ) -> bool:
        """Execute a relative rotation as equal small-angle sub-commands with delay."""
        total = float(theta_rad)
        if abs(total) < 1e-6:
            return True

        with self._lock:
            sr = float(step_rad) if step_rad is not None else float(self.base_rotate_step_rad)
            sd = float(step_delay_s) if step_delay_s is not None else float(self.base_rotate_step_delay_s)
        sr = max(1e-4, abs(sr))
        sd = float(np.clip(sd, 0.0, 0.5))

        n = int(np.ceil(abs(total) / sr))
        n = max(1, n)
        step = total / float(n)
        for i in range(n):
            # Allocate a bounded timeout per chunk to avoid over-blocking on one failed step.
            timeout_s = max(2.0, abs(step) * 8.0 + 1.5)
            ok = self.rotate_base_relative(step, timeout_s=timeout_s)
            if not ok:
                return False
            if i < (n - 1) and sd > 0.0:
                time.sleep(sd)
        return True

    def set_command_smoothing_delay(self, delay_s: float) -> None:
        self.command_smooth_delay_s = float(np.clip(float(delay_s), 0.01, 0.5))

    def set_base_rotate_step_deg(self, step_deg: float) -> None:
        self.base_rotate_step_rad = float(np.deg2rad(np.clip(float(step_deg), 0.01, 45.0)))

    def set_base_rotate_step_delay(self, delay_s: float) -> None:
        self.base_rotate_step_delay_s = float(np.clip(float(delay_s), 0.0, 0.5))

    def _ensure_ros_image_client_connected(self, timeout_s: float = ROSBRIDGE_TIMEOUT_S) -> None:
        client = self._ros_image_client
        if client is None:
            client = RosTopicImageClient(
                host=ROSBRIDGE_HOST,
                port=ROSBRIDGE_PORT,
                head_rgb_topic=HEAD_RGB_TOPIC,
                head_depth_topic=HEAD_DEPTH_TOPIC,
                head_info_topic=HEAD_CAMERA_INFO_TOPIC,
                wrist_rgb_topic=WRIST_RGB_TOPIC,
                wrist_depth_topic=WRIST_DEPTH_TOPIC,
                wrist_info_topic=WRIST_CAMERA_INFO_TOPIC,
                joint_topic=JOINT_STATE_TOPIC,
                odom_topic=ODOM_TOPIC,
                imu_mobile_base_topic=IMU_MOBILE_BASE_TOPIC,
                imu_wrist_topic=IMU_WRIST_TOPIC,
                imu_camera_accel_topic=IMU_CAMERA_ACCEL_TOPIC,
                imu_camera_gyro_topic=IMU_CAMERA_GYRO_TOPIC,
                magnetometer_topic=MAGNETOMETER_TOPIC,
                battery_topic=BATTERY_TOPIC,
            )
            self._ros_image_client = client
        if not client.connected:
            client.connect(timeout_s=float(timeout_s))

    def _try_connect_ros_topics(self, timeout_s: float = 1.0) -> tuple[bool, str | None]:
        now = time.time()
        client = self._ros_image_client
        if client is not None and client.connected:
            return True, None
        if now < float(self._next_ros_topics_connect_t):
            return False, "ROS topic client reconnect backoff active"
        try:
            self._ensure_ros_image_client_connected(timeout_s=float(timeout_s))
            self._next_ros_topics_connect_t = 0.0
            return True, None
        except Exception as exc:
            self._next_ros_topics_connect_t = now + 2.0
            return False, str(exc)

    def set_image_source(self, source: str) -> dict[str, Any]:
        src = str(source).strip().lower()
        if src in {"ros", "ros_topic", "ros_topics", "topic", "topics"}:
            src = "ros_topics"
        elif src in {"bridge", "worker"}:
            src = "bridge"
        else:
            return {"ok": False, "error": f"Unknown image source {source!r}"}

        with self._lock:
            self._image_source = src

        if src == "ros_topics":
            ok, err = self._try_connect_ros_topics(timeout_s=min(3.0, ROSBRIDGE_TIMEOUT_S))
            if not ok:
                return {"ok": False, "source": src, "error": err}
        return {"ok": True, "source": src}

    def get_image_source(self) -> str:
        with self._lock:
            return str(self._image_source)

    def head_image_rotated_90_cw(self) -> bool:
        source = self.get_image_source()
        if source == "ros_topics":
            return bool(ROS_TOPICS_ROTATE_HEAD_90_CW)
        return bool(STRETCH_AI_ROTATE_HEAD_90_CW)

    def publish_hold_stop(self) -> None:
        self.stop_base()
        for _ in range(3):
            self.publish_commands(force=True)
            rpc = self._rpc
            if rpc is not None:
                try:
                    rpc.request("stop_base", {}, timeout_s=2.0)
                except Exception:
                    pass
            time.sleep(0.02)

    def get_images(self):
        source = self.get_image_source()
        if source == "ros_topics":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_image_client
            if client is not None and client.connected:
                hr, wr, hd, wd, _, _ = client.get_snapshot()
                return hr, wr, hd, wd
            return None, None, None, None

        with self._lock:
            hr = None if self.head_rgb is None else self.head_rgb.copy()
            wr = None if self.wrist_rgb is None else self.wrist_rgb.copy()
            hd = None if self.head_depth is None else self.head_depth.copy()
            wd = None if self.wrist_depth is None else self.wrist_depth.copy()
        return hr, wr, hd, wd

    def get_actual_qpos(self):
        source = self.get_image_source()
        if source == "ros_topics":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_image_client
            if client is not None and client.connected:
                obs = client.get_observation_snapshot()
                actual = obs.get("actual_qpos")
                if isinstance(actual, list) and len(actual) > 0:
                    out = [float(v) for v in actual[:10]]
                    if len(out) < 10:
                        out += [0.0] * (10 - len(out))
                    return out
            return []
        with self._lock:
            if self.actual_qpos is not None:
                return list(self.actual_qpos)
        return []

    def get_published_qpos(self):
        with self._lock:
            if self.published_qpos is not None:
                return list(self.published_qpos)
            if self.qpos is not None:
                return list(self.qpos)
        return []

    def get_target_qpos(self):
        with self._lock:
            if self.qpos is not None:
                return list(self.qpos)
        return []

    def is_base_command_active(self, threshold: float = 1e-3) -> bool:
        with self._lock:
            return (
                abs(float(self._base_linear_cmd)) > float(threshold)
                or abs(float(self._base_angular_cmd)) > float(threshold)
            )

    def has_pending_command(self, eps: float = 1e-4) -> bool:
        with self._lock:
            if self._needs_mode_retry:
                return True
            if self.qpos is None:
                return False
            if self.published_qpos is None:
                return True
            n = min(len(self.qpos), len(self.published_qpos))
            if n == 0:
                return False
            for i in range(n):
                if abs(float(self.qpos[i]) - float(self.published_qpos[i])) > float(eps):
                    return True
            return len(self.qpos) != len(self.published_qpos)

    def get_measured_base_pose_xytheta(self):
        source = self.get_image_source()
        if source == "ros_topics":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_image_client
            if client is not None and client.connected:
                obs = client.get_observation_snapshot()
                pose = obs.get("base_pose_xytheta")
                if isinstance(pose, list) and len(pose) >= 3:
                    return [float(pose[0]), float(pose[1]), float(pose[2])]
            return [0.0, 0.0, 0.0]
        with self._lock:
            return [float(self._base_x), float(self._base_y), float(self._base_theta)]

    def get_command_base_pose_xytheta(self):
        with self._lock:
            if self.command_base_pose_xytheta is not None:
                return list(self.command_base_pose_xytheta)
            return [float(self._base_x), float(self._base_y), float(self._base_theta)]

    @property
    def camera_info(self):
        source = self.get_image_source()
        if source == "ros_topics":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_image_client
            if client is not None and client.connected:
                _, _, _, _, head_info, _ = client.get_snapshot()
                if head_info is not None:
                    return _CameraInfoCompat(head_info)
            return None

        with self._lock:
            raw = dict(self._camera_info_head) if self._camera_info_head is not None else None
        return _CameraInfoCompat(raw) if raw is not None else None

    def get_clock(self):
        return self._clock

    def lookup_transform(self, target_frame: str, source_frame: str):
        rpc = self._rpc
        if rpc is None:
            raise RuntimeError("Bridge worker is not connected")
        tf = rpc.request(
            "lookup_transform",
            {"target_frame": str(target_frame), "source_frame": str(source_frame)},
            timeout_s=max(2.0, STRETCH_AI_RPC_TIMEOUT_S),
        )
        return _transform_dict_to_msg(tf)

    def pixel_to_3d_point(self, pixel_x: int, pixel_y: int, depth: float):
        cam = self.camera_info
        if cam is None:
            return None
        fx = float(cam.k[0])
        fy = float(cam.k[4])
        cx = float(cam.k[2])
        cy = float(cam.k[5])
        if fx == 0.0 or fy == 0.0:
            return None

        # Keep mapping consistent with active image source orientation.
        if self.head_image_rotated_90_cw():
            original_x = float(pixel_y)
            original_y = float(cam.height - 1 - pixel_x)
        else:
            original_x = float(pixel_x)
            original_y = float(pixel_y)

        pt = PointStamped()
        pt.header.frame_id = "camera_color_optical_frame"
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = float((original_x - cx) * float(depth) / fx)
        pt.point.y = float((original_y - cy) * float(depth) / fy)
        pt.point.z = float(depth)
        return pt

    def transform_point_to_base(self, point_stamped: PointStamped):
        try:
            tf_msg = self.lookup_transform("base_link", point_stamped.header.frame_id)
        except Exception as exc:
            print(f"[stretch_ai_bridge] transform lookup failed: {exc}", file=sys.stderr)
            return None
        return _apply_transform_point(point_stamped, tf_msg)

    def get_sensor_snapshot(self):
        source = self.get_image_source()
        with self._lock:
            qpos = list(self.qpos) if self.qpos is not None else []
            published = list(self.published_qpos) if self.published_qpos is not None else []
            base_pose_bridge = [float(self._base_x), float(self._base_y), float(self._base_theta)]
            command_pose = (
                list(self.command_base_pose_xytheta)
                if self.command_base_pose_xytheta is not None
                else list(base_pose_bridge)
            )
            actual_bridge = list(self.actual_qpos) if self.actual_qpos is not None else []
            jp_bridge = list(self._joint_positions) if self._joint_positions is not None else []
            jv_bridge = list(self._joint_velocities) if self._joint_velocities is not None else []
            je_bridge = list(self._joint_efforts) if self._joint_efforts is not None else []
            head_info_bridge = dict(self._camera_info_head) if self._camera_info_head is not None else None
            wrist_info_bridge = dict(self._camera_info_wrist) if self._camera_info_wrist is not None else None
            mode = self._mode
            at_goal = self._at_goal
            is_homed = self._is_homed
            is_runstopped = self._is_runstopped

        if source == "ros_topics":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_image_client
            ros_obs = client.get_observation_snapshot() if (client is not None and client.connected) else {}

            actual = ros_obs.get("actual_qpos") if isinstance(ros_obs.get("actual_qpos"), list) else []
            actual = [float(v) for v in actual[:10]]
            if len(actual) < 10:
                actual += [0.0] * (10 - len(actual))

            base_pose = ros_obs.get("base_pose_xytheta") if isinstance(ros_obs.get("base_pose_xytheta"), list) else [0.0, 0.0, 0.0]
            if len(base_pose) < 3:
                base_pose = [0.0, 0.0, 0.0]
            base_pose = [float(base_pose[0]), float(base_pose[1]), float(base_pose[2])]

            joint_names = [str(v) for v in ros_obs.get("joint_state_name", [])] if isinstance(ros_obs.get("joint_state_name"), list) else []
            jp = [float(v) for v in ros_obs.get("joint_state_position", [])] if isinstance(ros_obs.get("joint_state_position"), list) else []
            jv = [float(v) for v in ros_obs.get("joint_state_velocity", [])] if isinstance(ros_obs.get("joint_state_velocity"), list) else []
            je = [float(v) for v in ros_obs.get("joint_state_effort", [])] if isinstance(ros_obs.get("joint_state_effort"), list) else []
            imu_mobile = ros_obs.get("imu_mobile_base")
            imu_wrist = ros_obs.get("imu_wrist")
            imu_cam_accel = ros_obs.get("imu_camera_accel")
            imu_cam_gyro = ros_obs.get("imu_camera_gyro")
            mag_mobile = ros_obs.get("magnetometer_mobile_base")
            battery = ros_obs.get("battery")
            odom = ros_obs.get("odom") if isinstance(ros_obs.get("odom"), dict) else None
            if odom is None:
                odom = {
                    "position": [base_pose[0], base_pose[1], 0.0],
                    "orientation": [0.0, 0.0, np.sin(base_pose[2] / 2.0), np.cos(base_pose[2] / 2.0)],
                    "linear_velocity": [float(actual[8]) if len(actual) > 8 else 0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, float(actual[9]) if len(actual) > 9 else 0.0],
                }
            head_info = ros_obs.get("camera_info_head") if isinstance(ros_obs.get("camera_info_head"), dict) else None
            wrist_info = ros_obs.get("camera_info_wrist") if isinstance(ros_obs.get("camera_info_wrist"), dict) else None
        else:
            actual = actual_bridge
            base_pose = base_pose_bridge
            joint_names = list(self.JOINT_STATE_NAMES)
            jp = jp_bridge
            jv = jv_bridge
            je = je_bridge
            imu_mobile = None
            imu_wrist = None
            imu_cam_accel = None
            imu_cam_gyro = None
            mag_mobile = None
            battery = None
            odom = {
                "position": [base_pose[0], base_pose[1], 0.0],
                "orientation": [0.0, 0.0, np.sin(base_pose[2] / 2.0), np.cos(base_pose[2] / 2.0)],
                "linear_velocity": [float(actual[8]) if len(actual) > 8 else 0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, float(actual[9]) if len(actual) > 9 else 0.0],
            }
            head_info = head_info_bridge
            wrist_info = wrist_info_bridge

        return {
            "observation.qpos_full": qpos,
            "observation.qpos_actual": actual,
            "observation.base_pose_xytheta": base_pose,
            "observation.command_base_pose_xytheta": command_pose,
            "observation.joint_state.name": joint_names,
            "observation.joint_state.position": jp,
            "observation.joint_state.velocity": jv,
            "observation.joint_state.effort": je,
            "observation.imu.mobile_base": imu_mobile,
            "observation.imu.wrist": imu_wrist,
            "observation.imu.camera_accel": imu_cam_accel,
            "observation.imu.camera_gyro": imu_cam_gyro,
            "observation.magnetometer.mobile_base": mag_mobile,
            "observation.battery": battery,
            "observation.odom": odom,
            "observation.camera_info.head": head_info,
            "observation.camera_info.wrist": wrist_info,
            "observation.bridge.control_mode": mode,
            "observation.bridge.at_goal": at_goal,
            "observation.bridge.is_homed": is_homed,
            "observation.bridge.is_runstopped": is_runstopped,
            "observation.qpos_published": published,
            "observation.images.source": source,
        }

    def is_ready(self):
        actual = self.get_actual_qpos()
        with self._lock:
            return self.qpos is not None and len(actual) > 0

    def close(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        if self._command_thread is not None and self._command_thread.is_alive():
            self._command_thread.join(timeout=2.0)
        if self._ros_image_client is not None:
            self._ros_image_client.close()
        if self._rpc is not None:
            self._rpc.close()
            self._rpc = None


###############################################################################
# UI
###############################################################################
class RobotTeleopBridgeUI(QMainWindow):
    TABLE_JOINT_NAMES = [
        "arm_extension",
        "arm_lift",
        "wrist_yaw",
        "wrist_pitch",
        "wrist_roll",
        "head_pan",
        "head_tilt",
        "gripper",
        "base_x",
        "base_y",
        "base_theta",
    ]

    def __init__(self, ros_node: StretchAIDemoBridge):
        super().__init__()
        self.ros_node = ros_node
        self.image_source = "bridge"
        if hasattr(self.ros_node, "get_image_source"):
            try:
                self.image_source = str(self.ros_node.get_image_source())
            except Exception:
                self.image_source = "bridge"

        self.head_rgb = None
        self.wrist_rgb = None
        self.depth_image = None
        self.wrist_depth = None

        # Base manual controls use per-tick relative chunks (dx in meters, dtheta in degrees).
        self.base_linear_step_m = 0.03
        self.base_theta_step_deg = 4.0
        self.arm_speed = 0.01
        self.head_speed = 0.05
        self.wrist_speed = 0.05
        self.gripper_step = 0.03
        self.command_smoothing_delay_s = float(self.ros_node.command_smooth_delay_s)

        self.dataset_root = DEFAULT_DATASET_ROOT
        self.record_prompt = ""
        self.is_recording_demo = False
        self.demo_recorder = LeRobotStyleRecorder(robot_type="stretch3", target_fps=DEMO_RECORD_FPS)

        self._active_increments: dict[str, float] = {}

        self._init_ui()

        self._control_timer = QTimer(self)
        self._control_timer.timeout.connect(self._control_tick)
        self._control_timer.start(max(10, int(round(self.command_smoothing_delay_s * 1000.0))))

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._refresh)
        self._ui_timer.start(UI_REFRESH_MS)

    def _init_ui(self):
        self.setWindowTitle("Robot Teleop + Recording (stretch_ai bridge)")
        self.resize(1700, 960)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        root.addLayout(left, stretch=4)

        cams = QGroupBox("RGB Cameras")
        cams_layout = QGridLayout(cams)
        self.head_label = QLabel("Waiting for head RGB...")
        self.wrist_label = QLabel("Waiting for wrist RGB...")
        for lbl in (self.head_label, self.wrist_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #111; color: #ddd;")
            lbl.setMinimumSize(760, 430)
        cams_layout.addWidget(self.head_label, 0, 0)
        cams_layout.addWidget(self.wrist_label, 0, 1)
        left.addWidget(cams, stretch=3)

        mon = QGroupBox("Joint State (Command vs Measured)")
        mon_layout = QVBoxLayout(mon)
        self.state_table = QTableWidget(len(self.TABLE_JOINT_NAMES), 3)
        self.state_table.setHorizontalHeaderLabels(["Joint", "Command", "Measured"])
        self.state_table.verticalHeader().setVisible(False)
        self.state_table.setAlternatingRowColors(True)
        self.state_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.state_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.state_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.state_table.setMaximumHeight(320)
        self.state_table.setColumnWidth(0, 150)
        self.state_table.setColumnWidth(1, 130)
        self.state_table.setColumnWidth(2, 130)
        self.state_table.horizontalHeader().setStretchLastSection(True)
        for row, name in enumerate(self.TABLE_JOINT_NAMES):
            self.state_table.setItem(row, 0, QTableWidgetItem(name))
            self.state_table.setItem(row, 1, QTableWidgetItem("--"))
            self.state_table.setItem(row, 2, QTableWidgetItem("--"))
        mon_layout.addWidget(self.state_table)
        left.addWidget(mon, stretch=1)

        ctrl = QVBoxLayout()
        root.addLayout(ctrl, stretch=2)

        ctrl.addWidget(self._build_smoothing_controls())
        ctrl.addWidget(self._build_base_controls())
        ctrl.addWidget(self._build_arm_controls())
        ctrl.addWidget(self._build_head_controls())
        ctrl.addWidget(self._build_wrist_controls())
        ctrl.addWidget(self._build_gripper_controls())

        stop_btn = QPushButton("STOP / HOLD")
        stop_btn.setMinimumHeight(42)
        stop_btn.setStyleSheet("QPushButton { background: #d32f2f; color: white; font-weight: bold; }")
        stop_btn.clicked.connect(self._on_stop)
        ctrl.addWidget(stop_btn)

        rec = QGroupBox("LeRobot Demo Recording")
        rec_layout = QGridLayout(rec)
        rec_layout.addWidget(QLabel("Prompt"), 0, 0)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g. pick up the red block and place it in the tray")
        self.prompt_input.textChanged.connect(self.on_prompt_changed)
        rec_layout.addWidget(self.prompt_input, 0, 1, 1, 2)

        rec_layout.addWidget(QLabel("Folder"), 1, 0)
        self.record_folder_input = QLineEdit(self.dataset_root)
        self.record_folder_input.textChanged.connect(self.on_record_folder_changed)
        rec_layout.addWidget(self.record_folder_input, 1, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_record_folder)
        rec_layout.addWidget(browse_btn, 1, 2)

        self.record_toggle_button = QPushButton("Record")
        self.record_toggle_button.clicked.connect(self.toggle_demo_recording)
        rec_layout.addWidget(self.record_toggle_button, 2, 0, 1, 3)

        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        rec_layout.addWidget(self.status_label, 3, 0, 1, 3)

        self.fps_label = QLabel("FPS: --")
        rec_layout.addWidget(self.fps_label, 4, 0, 1, 3)
        root.addWidget(rec, stretch=1)

    def _make_speed_slider(
        self,
        *,
        min_val: float,
        max_val: float,
        default_val: float,
        on_change,
        label_text: str = "Step Size",
    ):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(72)
        row.addWidget(label)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        pos = int((default_val - min_val) / (max_val - min_val) * 100)
        slider.setValue(int(np.clip(pos, 0, 100)))
        value_label = QLabel(f"{default_val:.3f}")
        value_label.setFixedWidth(52)

        def _handle(v: int):
            value = min_val + (max_val - min_val) * (v / 100.0)
            value_label.setText(f"{value:.3f}")
            on_change(value)

        slider.valueChanged.connect(_handle)
        row.addWidget(slider)
        row.addWidget(value_label)
        return row

    def _build_smoothing_controls(self):
        group = QGroupBox("Command Smoothing")
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Delay between small command steps (seconds)"))
        layout.addLayout(
            self._make_speed_slider(
                min_val=0.20,
                max_val=0.01,
                default_val=self.command_smoothing_delay_s,
                on_change=self._on_smoothing_delay_changed,
                label_text="Speed"
            )
        )
        return group

    def _on_smoothing_delay_changed(self, delay_s: float):
        self.command_smoothing_delay_s = float(np.clip(delay_s, 0.01, 0.5))
        self.ros_node.set_command_smoothing_delay(self.command_smoothing_delay_s)
        self._control_timer.setInterval(max(10, int(round(self.command_smoothing_delay_s * 1000.0))))

    def _bind_hold_button(self, *, button: QPushButton, control_name: str, sign: float, speed_attr: str):
        button.pressed.connect(lambda: self._start_increment(control_name, sign * float(getattr(self, speed_attr))))
        button.released.connect(lambda: self._stop_increment(control_name))

    def _build_base_controls(self):
        group = QGroupBox("Base")
        layout = QGridLayout(group)

        btn_fwd = QPushButton("Forward")
        btn_left = QPushButton("Turn Left")
        btn_back = QPushButton("Backward")
        btn_right = QPushButton("Turn Right")

        btn_fwd.pressed.connect(lambda: self._set_base(+self.base_linear_step_m, 0.0))
        btn_fwd.released.connect(self._stop_base)
        btn_back.pressed.connect(lambda: self._set_base(-self.base_linear_step_m, 0.0))
        btn_back.released.connect(self._stop_base)
        btn_left.pressed.connect(lambda: self._set_base(0.0, +self.base_theta_step_deg))
        btn_left.released.connect(self._stop_base)
        btn_right.pressed.connect(lambda: self._set_base(0.0, -self.base_theta_step_deg))
        btn_right.released.connect(self._stop_base)

        layout.addWidget(btn_fwd, 0, 1)
        layout.addWidget(btn_left, 1, 0)
        layout.addWidget(btn_back, 1, 1)
        layout.addWidget(btn_right, 1, 2)

        layout.addLayout(
            self._make_speed_slider(
                min_val=0.005,
                max_val=0.20,
                default_val=self.base_linear_step_m,
                on_change=lambda v: setattr(self, "base_linear_step_m", v),
                label_text="Linear Step",
            ),
            2, 0, 1, 3,
        )
        layout.addLayout(
            self._make_speed_slider(
                min_val=0.5,
                max_val=20.0,
                default_val=self.base_theta_step_deg,
                on_change=lambda v: setattr(self, "base_theta_step_deg", v),
                label_text="Rotation Step",
            ),
            3, 0, 1, 3,
        )

        layout.addWidget(QLabel("Move (cm)"), 4, 0)
        self.base_distance_cm_input = QLineEdit("0")
        self.base_distance_cm_input.setPlaceholderText("+100 / -100")
        move_btn = QPushButton("Move Distance")
        move_btn.clicked.connect(self._move_base_distance_from_input)
        layout.addWidget(self.base_distance_cm_input, 4, 1)
        layout.addWidget(move_btn, 4, 2)
        return group

    def _move_base_distance_from_input(self) -> None:
        text = self.base_distance_cm_input.text().strip()
        if not text:
            return
        try:
            dist_cm = float(text)
        except ValueError:
            self.status_label.setText("Base move requires numeric cm value")
            return

        dx_m = dist_cm / 100.0
        if abs(dx_m) < 1e-6:
            return

        try:
            self.ros_node.stop_base()
            ok = self.ros_node.move_base_relative(
                dx=dx_m,
                dy=0.0,
                dtheta=0.0,
                blocking=False,
                timeout_s=max(2.0, 4.0 + abs(dx_m) * 6.0),
            )
            self.status_label.setText(
                f"Base move command {dx_m:+.3f} m ({'sent' if ok else 'failed'})"
            )
        except Exception as exc:
            self.status_label.setText(f"Base move failed: {exc}")

    def _build_arm_controls(self):
        group = QGroupBox("Arm (Lift / Stretch)")
        layout = QGridLayout(group)
        b1 = QPushButton("Lift +")
        b2 = QPushButton("Lift -")
        b3 = QPushButton("Stretch +")
        b4 = QPushButton("Stretch -")

        self._bind_hold_button(button=b1, control_name="arm_lift", sign=+1.0, speed_attr="arm_speed")
        self._bind_hold_button(button=b2, control_name="arm_lift", sign=-1.0, speed_attr="arm_speed")
        self._bind_hold_button(button=b3, control_name="arm_extension", sign=+1.0, speed_attr="arm_speed")
        self._bind_hold_button(button=b4, control_name="arm_extension", sign=-1.0, speed_attr="arm_speed")

        layout.addWidget(b1, 0, 0)
        layout.addWidget(b2, 0, 1)
        layout.addWidget(b3, 1, 0)
        layout.addWidget(b4, 1, 1)
        layout.addLayout(
            self._make_speed_slider(min_val=0.002, max_val=0.05, default_val=self.arm_speed,
                                    on_change=lambda v: setattr(self, "arm_speed", v)),
            2, 0, 1, 2,
        )
        return group

    def _build_head_controls(self):
        group = QGroupBox("Head")
        layout = QGridLayout(group)
        b1 = QPushButton("Pan +")
        b2 = QPushButton("Pan -")
        b3 = QPushButton("Tilt +")
        b4 = QPushButton("Tilt -")

        self._bind_hold_button(button=b1, control_name="head_pan", sign=+1.0, speed_attr="head_speed")
        self._bind_hold_button(button=b2, control_name="head_pan", sign=-1.0, speed_attr="head_speed")
        self._bind_hold_button(button=b3, control_name="head_tilt", sign=+1.0, speed_attr="head_speed")
        self._bind_hold_button(button=b4, control_name="head_tilt", sign=-1.0, speed_attr="head_speed")

        layout.addWidget(b1, 0, 0)
        layout.addWidget(b2, 0, 1)
        layout.addWidget(b3, 1, 0)
        layout.addWidget(b4, 1, 1)
        layout.addLayout(
            self._make_speed_slider(min_val=0.01, max_val=0.30, default_val=self.head_speed,
                                    on_change=lambda v: setattr(self, "head_speed", v)),
            2, 0, 1, 2,
        )
        return group

    def _build_wrist_controls(self):
        group = QGroupBox("Wrist")
        layout = QGridLayout(group)
        b1 = QPushButton("Yaw +")
        b2 = QPushButton("Yaw -")
        b3 = QPushButton("Pitch +")
        b4 = QPushButton("Pitch -")
        b5 = QPushButton("Roll +")
        b6 = QPushButton("Roll -")

        self._bind_hold_button(button=b1, control_name="wrist_yaw", sign=+1.0, speed_attr="wrist_speed")
        self._bind_hold_button(button=b2, control_name="wrist_yaw", sign=-1.0, speed_attr="wrist_speed")
        self._bind_hold_button(button=b3, control_name="wrist_pitch", sign=+1.0, speed_attr="wrist_speed")
        self._bind_hold_button(button=b4, control_name="wrist_pitch", sign=-1.0, speed_attr="wrist_speed")
        self._bind_hold_button(button=b5, control_name="wrist_roll", sign=+1.0, speed_attr="wrist_speed")
        self._bind_hold_button(button=b6, control_name="wrist_roll", sign=-1.0, speed_attr="wrist_speed")

        layout.addWidget(b1, 0, 0)
        layout.addWidget(b2, 0, 1)
        layout.addWidget(b3, 1, 0)
        layout.addWidget(b4, 1, 1)
        layout.addWidget(b5, 2, 0)
        layout.addWidget(b6, 2, 1)
        layout.addLayout(
            self._make_speed_slider(min_val=0.01, max_val=0.30, default_val=self.wrist_speed,
                                    on_change=lambda v: setattr(self, "wrist_speed", v)),
            3, 0, 1, 2,
        )
        return group

    def _build_gripper_controls(self):
        group = QGroupBox("Gripper")
        layout = QGridLayout(group)
        open_btn = QPushButton("Open +")
        close_btn = QPushButton("Close -")
        open_btn.clicked.connect(lambda: self._gripper_step(+1.0))
        close_btn.clicked.connect(lambda: self._gripper_step(-1.0))
        layout.addWidget(open_btn, 0, 0)
        layout.addWidget(close_btn, 0, 1)
        layout.addLayout(
            self._make_speed_slider(min_val=0.005, max_val=0.10, default_val=self.gripper_step,
                                    on_change=lambda v: setattr(self, "gripper_step", v)),
            1, 0, 1, 2,
        )
        return group

    def _set_base(self, linear_step_m: float, theta_step_deg: float):
        self.ros_node.set_control("base_linear", float(linear_step_m))
        self.ros_node.set_control("base_angular", float(np.deg2rad(theta_step_deg)))

    def _stop_base(self):
        self.ros_node.stop_base()

    def _start_increment(self, control_name: str, delta: float):
        # Safety: if a base command is still latched, stop it before starting
        # any non-base hold control. This prevents staying in navigation mode.
        if control_name not in ("base_linear", "base_angular") and self.ros_node.is_base_command_active():
            self.ros_node.stop_base()
        self._active_increments[control_name] = float(delta)

    def _stop_increment(self, control_name: str):
        self._active_increments.pop(control_name, None)

    def _gripper_step(self, sign: float):
        if self.ros_node.is_base_command_active():
            self.ros_node.stop_base()
        self.ros_node.adjust_control("gripper", sign * self.gripper_step)

    def _on_stop(self):
        self._active_increments.clear()
        self.ros_node.publish_hold_stop()
        self.status_label.setText("STOP/HOLD published")

    def _control_tick(self):
        for control_name, delta in list(self._active_increments.items()):
            self.ros_node.adjust_control(control_name, delta)
        pending = self.ros_node.has_pending_command()
        hold_active = bool(self._active_increments)
        # Publish only for arm/head/wrist/gripper holds or pending joint smoothing deltas.
        # Base streaming is handled independently by move_base_relative() in command-loop.
        if hold_active or pending:
            self.ros_node.publish_commands(force=False)

    def on_prompt_changed(self, text):
        self.record_prompt = text

    def on_record_folder_changed(self, text):
        self.dataset_root = text.strip()

    def browse_record_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select LeRobot Dataset Folder", self.dataset_root)
        if folder:
            self.dataset_root = folder
            self.record_folder_input.setText(folder)

    def _build_record_sample(self):
        sensors = self.ros_node.get_sensor_snapshot()
        actual_qpos = self.ros_node.get_actual_qpos()
        command_qpos = self.ros_node.get_published_qpos()
        measured_pose = self.ros_node.get_measured_base_pose_xytheta() or [0.0, 0.0, 0.0]
        command_pose = self.ros_node.get_command_base_pose_xytheta() or list(measured_pose)

        state_raw_v5 = list(actual_qpos) if actual_qpos else []
        action_raw_v5 = list(actual_qpos) if actual_qpos else []
        action_command_raw_v5 = list(command_qpos) if command_qpos else []

        state = state_raw_v5[:8] + [float(v) for v in measured_pose]
        action = action_raw_v5[:8] + [float(v) for v in measured_pose]
        action_command = action_command_raw_v5[:8] + [float(v) for v in command_pose]

        sensors["observation.state_raw_v5"] = state_raw_v5
        sensors["observation.qpos_actual_raw_v5"] = action_raw_v5
        sensors["observation.qpos_full_raw_v5"] = action_command_raw_v5
        sensors["action_raw_v5"] = action_raw_v5
        sensors["action_command_raw_v5"] = action_command_raw_v5

        return {
            "timestamp": time.time(),
            "head_rgb": self.head_rgb if self.head_rgb is not None else None,
            "wrist_rgb": self.wrist_rgb if self.wrist_rgb is not None else None,
            "head_depth": self.depth_image if self.depth_image is not None else None,
            "wrist_depth": self.wrist_depth if self.wrist_depth is not None else None,
            "state": state,
            "action": action,
            "action_command": action_command,
            "sensors": sensors,
        }

    def _set_manual_gripper_override(self, value: float | None) -> None:
        if value is None or not isinstance(value, (int, float)):
            self._manual_gripper_override = None
            return
        try:
            v = float(value)
            if not math.isfinite(v):
                self._manual_gripper_override = None
                return
        except (TypeError, ValueError):
            self._manual_gripper_override = None
            return
        lo, hi = self.ros_node.JOINT_LIMITS[7]
        self._manual_gripper_override = float(np.clip(v, lo, hi))

    def _get_manual_gripper_target(self, fallback: float | None = None) -> float | None:
        if self._manual_gripper_override is not None:
            return float(self._manual_gripper_override)

        target = self.ros_node.get_target_qpos()
        if isinstance(target, list) and len(target) >= 8:
            try:
                v = float(target[7])
                if math.isfinite(v):
                    lo, hi = self.ros_node.JOINT_LIMITS[7]
                    return float(np.clip(v, lo, hi))
            except (TypeError, ValueError):
                pass

        actual = self.ros_node.get_actual_qpos()
        if isinstance(actual, list) and len(actual) >= 8:
            try:
                v = float(actual[7])
                if math.isfinite(v):
                    lo, hi = self.ros_node.JOINT_LIMITS[7]
                    return float(np.clip(v, lo, hi))
            except (TypeError, ValueError):
                pass

        return None if fallback is None else float(fallback)

    def start_demo_recording(self):
        if self.is_recording_demo:
            return
        if not self.dataset_root:
            self.status_label.setText("Set a recording folder first")
            return
        prompt = (self.record_prompt or "").strip()
        if not prompt:
            prompt = "unspecified_task"
            self.prompt_input.setText(prompt)
        try:
            self.demo_recorder.start(self.dataset_root, prompt)
            self.is_recording_demo = True
            self.record_toggle_button.setText("Stop Recording")
            self.status_label.setText(f"Recording demo: {prompt} (fps={self.demo_recorder.target_fps:.1f})")
        except Exception as exc:
            self.status_label.setText(f"Record start failed: {exc}")

    def stop_demo_recording(self):
        if not self.is_recording_demo:
            return
        try:
            summary = self.demo_recorder.stop()
            self.is_recording_demo = False
            self.record_toggle_button.setText("Record")
            if summary is None:
                self.status_label.setText("Recording stopped")
            else:
                self.status_label.setText(
                    f"Saved demo ep {summary['episode_index']} ({summary['num_frames']} frames, "
                    f"dropped={summary.get('dropped_frames', 0)})"
                )
                print(f"Demo saved: {summary}")
        except Exception as exc:
            self.status_label.setText(f"Record stop failed: {exc}")

    def toggle_demo_recording(self):
        if self.is_recording_demo:
            self.stop_demo_recording()
        else:
            self.start_demo_recording()

    @staticmethod
    def _to_pixmap(rgb: np.ndarray, target_w: int = 900, target_h: int = 520) -> QPixmap:
        img = np.ascontiguousarray(rgb)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _refresh(self):
        head, wrist, head_depth, wrist_depth = self.ros_node.get_images()
        self.head_rgb = head
        self.wrist_rgb = wrist
        self.depth_image = head_depth
        self.wrist_depth = wrist_depth

        if head is not None:
            self.head_label.setPixmap(
                self._to_pixmap(
                    head,
                    target_w=max(320, self.head_label.width() - 10),
                    target_h=max(240, self.head_label.height() - 10),
                )
            )
        if wrist is not None:
            self.wrist_label.setPixmap(
                self._to_pixmap(
                    wrist,
                    target_w=max(320, self.wrist_label.width() - 10),
                    target_h=max(240, self.wrist_label.height() - 10),
                )
            )

        actual = self.ros_node.get_actual_qpos()
        cmd = self.ros_node.get_published_qpos()
        mpose = self.ros_node.get_measured_base_pose_xytheta()
        cpose = self.ros_node.get_command_base_pose_xytheta()

        cmd10 = list(cmd[:10]) + [0.0] * max(0, 10 - len(cmd))
        actual10 = list(actual[:10]) + [0.0] * max(0, 10 - len(actual))
        cmd_rows = cmd10[:8] + [float(cpose[0]), float(cpose[1]), float(cpose[2])]
        measured_rows = actual10[:8] + [float(mpose[0]), float(mpose[1]), float(mpose[2])]
        for row, (cmd_v, meas_v) in enumerate(zip(cmd_rows, measured_rows)):
            self.state_table.item(row, 1).setText(f"{cmd_v:+.5f}")
            self.state_table.item(row, 2).setText(f"{meas_v:+.5f}")

        fps_status = "Ready" if self.ros_node.is_ready() else "Initializing"
        hs = head.shape if head is not None else "N/A"
        ws = wrist.shape if wrist is not None else "N/A"
        sensors = self.ros_node.get_sensor_snapshot()
        mode = sensors.get("observation.bridge.control_mode", "unknown")
        self.fps_label.setText(
            f"Robot: {fps_status} | Head: {hs} | Wrist: {ws} | "
            f"Mode: {mode} | Delay: {self.command_smoothing_delay_s:.3f}s"
        )

        if self.is_recording_demo:
            try:
                self.demo_recorder.record_step(self._build_record_sample())
            except Exception as exc:
                print(f"Recording step error: {exc}")
                self.stop_demo_recording()
                self.status_label.setText(f"Recording stopped due error: {exc}")

    def closeEvent(self, event):
        if self.is_recording_demo:
            self.stop_demo_recording()
        self._control_timer.stop()
        self._ui_timer.stop()
        try:
            self.ros_node.publish_hold_stop()
        except Exception:
            pass
        self.ros_node.close()
        event.accept()

class RobotController:
    """Robot controller that interfaces with ROS node"""

    def __init__(self, ros_node):
        self.ros_node = ros_node
        self.running = True
        self.step_count = 0
        self.last_fps_time = time.time()

        # Callbacks (will be set by UI)
        self.on_images_updated = None
        self.on_error = None
        self.on_fps_updated = None

    def step(self):
        """Update step - called by QTimer in main thread"""
        if not self.running:
            return

        try:
            # Get images from ROS node
            head_rgb, wrist_rgb, depth_image, wrist_depth = self.ros_node.get_images()

            # Create dummy images if not available
            if head_rgb is None:
                # Create placeholder
                if self.step_count % 30 == 0:
                    print("Waiting for camera feed...")
                return

            # Ensure images are in correct format
            if len(head_rgb.shape) == 2:
                head_rgb = cv2.cvtColor(head_rgb, cv2.COLOR_GRAY2RGB)
            elif head_rgb.shape[2] == 4:
                head_rgb = head_rgb[:, :, :3]

            if head_rgb.dtype != np.uint8:
                head_rgb = (np.clip(head_rgb, 0, 255)).astype(np.uint8)

            # Handle wrist camera
            if wrist_rgb is None:
                wrist_rgb = np.zeros_like(head_rgb)
            else:
                if len(wrist_rgb.shape) == 2:
                    wrist_rgb = cv2.cvtColor(wrist_rgb, cv2.COLOR_GRAY2RGB)
                elif wrist_rgb.shape[2] == 4:
                    wrist_rgb = wrist_rgb[:, :, :3]
                if wrist_rgb.dtype != np.uint8:
                    wrist_rgb = (np.clip(wrist_rgb, 0, 255)).astype(np.uint8)

            # Handle depth
            if depth_image is None:
                depth_image = np.zeros((head_rgb.shape[0], head_rgb.shape[1]), dtype=np.float32)
            if wrist_depth is None:
                wrist_depth = np.zeros((wrist_rgb.shape[0], wrist_rgb.shape[1]), dtype=np.float32)

            # Call callback with images
            if self.on_images_updated:
                self.on_images_updated(
                    head_rgb.copy(),
                    wrist_rgb.copy(),
                    depth_image.copy(),
                    wrist_depth.copy(),
                )

            # FPS counter (every 30 frames)
            self.step_count += 1
            if self.step_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - self.last_fps_time)
                if self.on_fps_updated:
                    self.on_fps_updated(fps)
                self.last_fps_time = current_time

        except Exception as e:
            if self.on_error:
                self.on_error(f"Robot control error: {str(e)}")
            print(f"Error in robot control: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def set_control(self, control_name, value):
        """Set control value - maps to ROS node"""
        self.ros_node.set_control(control_name, value)

    def adjust_control(self, control_name, delta):
        """Adjust control value - maps to ROS node"""
        self.ros_node.adjust_control(control_name, delta)

    def stop(self):
        """Stop the controller"""
        self.running = False


class SegmentationThread(QThread):
    """Separate thread for SAM segmentation"""

    # Signals
    segmentation_complete = pyqtSignal(list, np.ndarray)  # segments, mask_overlay
    segmentation_error = pyqtSignal(str)
    model_loading = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.sam_model = None
        self.rgb_image = None
        self.depth_image = None
        self.color_palette = self._generate_colors(50)

    def _generate_colors(self, n):
        """Generate n distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def set_images(self, rgb_image, depth_image):
        """Set images to segment"""
        self.rgb_image = rgb_image
        self.depth_image = depth_image

    def run(self):
        """Run segmentation"""
        try:
            if self.rgb_image is None:
                self.segmentation_error.emit("No image available")
                return

            # Load SAM model if needed
            if self.sam_model is None:
                self.model_loading.emit()
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"Loading SAM model on {device.upper()}...", flush=True)
                self.sam_model = SAM("/home/ibk5106/Desktop/Projects/teleop_in_behavior/sam_b.pt")
                self.sam_model.to(device)
                print(f"SAM model loaded on {device.upper()}!", flush=True)

            # Run segmentation
            print("Running SAM inference on GPU...", flush=True)
            import time
            start_time = time.time()
            results = self.sam_model(self.rgb_image, verbose=False)
            inference_time = time.time() - start_time
            print(f"SAM inference completed in {inference_time:.1f} seconds", flush=True)

            # Extract segments
            segments = []
            masks_combined = np.zeros_like(self.rgb_image)
            seg_id = 0
            filtered_small = 0

            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()

                for mask in masks:
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    if mask_binary.sum() == 0:
                        continue

                    # Split disconnected parts of one SAM mask into separate instances.
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        mask_binary, connectivity=8
                    )
                    for comp_id in range(1, num_labels):  # 0 = background
                        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
                        y = int(stats[comp_id, cv2.CC_STAT_TOP])
                        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
                        area_px = int(stats[comp_id, cv2.CC_STAT_AREA])

                        # Filter tiny fragmented components.
                        if (area_px < SAM_CC_MIN_AREA_PX or
                                w < SAM_CC_MIN_WIDTH_PX or
                                h < SAM_CC_MIN_HEIGHT_PX):
                            filtered_small += 1
                            continue

                        comp_mask = (labels == comp_id).astype(np.uint8)
                        center_x = int(round(float(centroids[comp_id, 0])))
                        center_y = int(round(float(centroids[comp_id, 1])))

                        # Get depth at component (median of valid depths, fallback to center pixel)
                        depth_value = 0.0
                        if self.depth_image is not None:
                            comp_depth = self.depth_image[comp_mask > 0]
                            valid_depth = comp_depth[(comp_depth > 0.1) & (comp_depth < 5.0)]
                            if len(valid_depth) > 0:
                                depth_value = float(np.median(valid_depth))
                            elif (0 <= center_y < self.depth_image.shape[0] and
                                  0 <= center_x < self.depth_image.shape[1]):
                                depth_value = float(self.depth_image[center_y, center_x])

                        segment_info = {
                            'id': seg_id,
                            'mask': comp_mask,
                            'bbox': (x, y, w, h),
                            'center': (center_x, center_y),
                            'depth': depth_value,
                            'area': area_px,
                            'color': self.color_palette[seg_id % len(self.color_palette)]
                        }
                        segments.append(segment_info)
                        seg_id += 1

                        # Add to combined mask
                        color = segment_info['color']
                        masks_combined[comp_mask > 0] = color

            if filtered_small > 0:
                print(f"SAM post-process: filtered {filtered_small} tiny components "
                      f"(area<{SAM_CC_MIN_AREA_PX} or w<{SAM_CC_MIN_WIDTH_PX} or h<{SAM_CC_MIN_HEIGHT_PX})",
                      flush=True)

            self.segmentation_complete.emit(segments, masks_combined)

        except Exception as e:
            self.segmentation_error.emit(f"Segmentation error: {str(e)}")
            print(f"Segmentation error: {e}")
            import traceback
            traceback.print_exc()


class RobotTeleopUI(QMainWindow):
    """Main UI class for robot teleoperation with SAM segmentation"""
    ui_status_signal = pyqtSignal(str, str)
    ui_return_enabled_signal = pyqtSignal(bool)
    ui_action_state_signal = pyqtSignal(str)

    def __init__(self, ros_node):
        super().__init__()

        self.ros_node = ros_node
        self.image_source = "bridge"
        if hasattr(self.ros_node, "get_image_source"):
            try:
                self.image_source = str(self.ros_node.get_image_source())
            except Exception:
                self.image_source = "bridge"

        # State
        self.head_rgb = None
        self.wrist_rgb = None
        self.depth_image = None
        self.wrist_depth = None
        self.mask_overlay = None
        self._grasp_debug_info = None  # rect_info with axis_pixels for debug overlay
        self.segments = []
        self.selected_segment = None
        self.use_head_for_segmentation = True  # Use head camera for segmentation by default
        self._pre_action_state = None  # saved before reach/grasp so we can return
        self._action_lock = threading.Lock()
        self._action_state = 'idle'  # idle|running|paused|awaiting_confirm|awaiting_post_grasp
        self._action_mode = None
        self._action_abort_requested = False
        self._manual_gripper_override: float | None = None
        # Queued two-goal workflow (grasp -> reach) planned from the same frame.
        self.queued_goals = {"grasp": None, "reach": None}
        self.queued_goal_cursor = 0
        self.queued_sequence_started = False
        self._deferred_next_goal_start = False
        self._skip_to_next_goal_requested = False
        self.dataset_root = str((Path.cwd() / "stretch_recordings_v2").resolve())
        self.record_prompt = ""
        self.is_recording_demo = False
        self.demo_recorder = LeRobotStyleRecorder(robot_type="stretch3", target_fps=DEMO_RECORD_FPS)

        # Control parameters (increased for better responsiveness)
        self.linear_speed = 0.103      # m/s for base translation
        self.angular_speed = 0.2       # rad/s for base rotation
        self.arm_speed = 0.005         # m or rad increment per update
        self.head_speed = 0.02         # rad increment per update
        self.wrist_speed = 0.02        # rad increment per update
        self.gripper_step = 0.02       # joint increment per click (open/close)
        self.command_smoothing_delay = DEFAULT_COMMAND_SMOOTH_DELAY_S
        self.base_angle_step_deg = float(DEFAULT_BASE_ROTATE_STEP_DEG)

        # Grasp planner (will be adapted for ROS)
        # For now, we'll implement basic grasping behavior
        self.grasp_planner_available = True  # Always show grasp button
        print("Note: Grasp planner will use basic approach behavior")

        # Setup robot controller
        self.robot_controller = RobotController(ros_node)
        self.robot_controller.on_images_updated = self.on_images_updated
        self.robot_controller.on_error = self.on_error
        self.robot_controller.on_fps_updated = self.on_fps_updated
        self.ros_node.set_command_smoothing_delay(self.command_smoothing_delay)
        if hasattr(self.ros_node, 'set_base_rotate_step_deg'):
            self.ros_node.set_base_rotate_step_deg(self.base_angle_step_deg)
        if hasattr(self.ros_node, 'set_base_rotate_step_delay'):
            self.ros_node.set_base_rotate_step_delay(self.command_smoothing_delay)

        # Setup segmentation thread
        self.seg_thread = SegmentationThread()
        self.seg_thread.segmentation_complete.connect(self.on_segmentation_complete)
        self.seg_thread.segmentation_error.connect(self.on_error)
        self.seg_thread.model_loading.connect(self.on_model_loading)

        # Setup UI
        self.init_ui()
        # Thread-safe UI update signals (worker threads -> main Qt thread)
        self.ui_status_signal.connect(self._apply_status_update)
        self.ui_return_enabled_signal.connect(self.return_button.setEnabled)
        self.ui_action_state_signal.connect(self._apply_action_state_ui)
        self._update_goal_queue_label()
        self._update_next_goal_button_state()

        # Setup timer for update loop (runs in main thread)
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.robot_controller.step)
        self.control_timer.start(100)  # ~10 FPS (gives GIL time to camera thread)

        print("UI initialized successfully", flush=True)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Robot Teleoperation with SAM (ROS2)")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)  # Set minimum window size

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        main_layout.setSpacing(5)

        # Content layout (horizontal)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(5)

        # Left side: Camera feeds (takes more space)
        camera_widget = self.create_camera_widget()
        content_layout.addWidget(camera_widget, stretch=3)

        # Right side: Controls and objects (fixed width range)
        control_widget = self.create_control_widget()
        content_layout.addWidget(control_widget, stretch=2)

        main_layout.addLayout(content_layout, stretch=1)

        # Status bar at bottom (fixed height)
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("QLabel { padding: 5px; background-color: #2c3e50; color: white; font-size: 11px; }")
        self.fps_label.setMaximumHeight(30)
        main_layout.addWidget(self.fps_label)

    def create_camera_widget(self):
        """Create camera feed widget"""
        widget = QGroupBox("Camera Feeds (Bridge or ROS Topics)")
        layout = QVBoxLayout()

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Image Source"))
        self.image_source_combo = QComboBox()
        self.image_source_combo.addItem("Bridge (worker)", "bridge")
        self.image_source_combo.addItem("ROS Topics (direct)", "ros_topics")
        idx = 0 if self.image_source != "ros_topics" else 1
        self.image_source_combo.setCurrentIndex(idx)
        self.image_source_combo.currentIndexChanged.connect(self.on_image_source_changed)
        source_row.addWidget(self.image_source_combo, stretch=1)
        self.image_source_status = QLabel(f"Using: {self.image_source}")
        self.image_source_status.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        source_row.addWidget(self.image_source_status)
        layout.addLayout(source_row)

        # Head RGB Camera
        head_label = QLabel("Head RGB Camera (/camera/color/image_raw)")
        head_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(head_label)

        # Wrapper container expands to fill space; label inside shrinks to image
        self.head_container = QWidget()
        self.head_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        head_container_layout = QHBoxLayout(self.head_container)
        head_container_layout.setContentsMargins(0, 0, 0, 0)
        head_container_layout.addStretch()
        self.head_display = QLabel("Waiting for head camera feed...")
        self.head_display.setScaledContents(True)
        self.head_display.setStyleSheet("QLabel { background-color: black; color: white; }")
        self.head_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.head_display.mousePressEvent = self.on_image_click
        head_container_layout.addWidget(self.head_display)
        head_container_layout.addStretch()
        layout.addWidget(self.head_container, stretch=6)

        # Wrist/Gripper RGB Camera
        self.wrist_label = QLabel("Gripper RGB Camera (no feed)")
        self.wrist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.wrist_label)

        self.wrist_container = QWidget()
        self.wrist_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.wrist_container.setMaximumHeight(40)  # Small placeholder when no image
        wrist_container_layout = QHBoxLayout(self.wrist_container)
        wrist_container_layout.setContentsMargins(0, 0, 0, 0)
        wrist_container_layout.addStretch()
        self.wrist_display = QLabel("No wrist camera feed")
        self.wrist_display.setScaledContents(True)
        self.wrist_display.setStyleSheet("QLabel { background-color: #1a1a1a; color: gray; }")
        self.wrist_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wrist_container_layout.addWidget(self.wrist_display)
        wrist_container_layout.addStretch()
        self._wrist_feed_active = False  # Track whether we've received an image
        layout.addWidget(self.wrist_container, stretch=0)

        # Segmentation button
        self.segment_button = QPushButton("Run SAM Segmentation (Head Camera)")
        self.segment_button.clicked.connect(self.run_segmentation)
        self.segment_button.setMinimumHeight(40)
        self.segment_button.setMaximumHeight(60)  # Prevent button from getting too tall
        layout.addWidget(self.segment_button)

        widget.setLayout(layout)
        return widget

    def on_image_source_changed(self, _index: int):
        source = "bridge"
        if hasattr(self, "image_source_combo"):
            data = self.image_source_combo.currentData()
            source = str(data) if data is not None else "bridge"
        try:
            result = None
            if hasattr(self.ros_node, "set_image_source"):
                result = self.ros_node.set_image_source(source)
            self.image_source = source
            if hasattr(self, "image_source_status"):
                self.image_source_status.setText(f"Using: {self.image_source}")
                if isinstance(result, dict) and not result.get("ok", False):
                    self.image_source_status.setStyleSheet("QLabel { color: #ef6c00; font-size: 10px; }")
                else:
                    self.image_source_status.setStyleSheet("QLabel { color: #1e88e5; font-size: 10px; }")
            if hasattr(self, "status_label"):
                if isinstance(result, dict) and not result.get("ok", False):
                    self.status_label.setText(
                        f"Image source set to {self.image_source}, waiting for data: "
                        f"{result.get('error', 'connect pending')}"
                    )
                    self.status_label.setStyleSheet("QLabel { color: #ef6c00; font-size: 10px; }")
                else:
                    self.status_label.setText(f"Image source switched to {self.image_source}")
                    self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")
        except Exception as exc:
            if hasattr(self, "image_source_status"):
                self.image_source_status.setText(f"Using: {self.image_source}")
                self.image_source_status.setStyleSheet("QLabel { color: #d32f2f; font-size: 10px; }")
            if hasattr(self, "status_label"):
                self.status_label.setText(f"Image source switch failed: {exc}")
                self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")
            print(f"[image_source] switch error: {exc}", file=sys.stderr)

    def create_control_widget(self):
        """Create two-column right side: controls (middle) + detected objects/actions (right)."""
        widget = QWidget()
        widget.setMinimumWidth(620)
        widget.setMaximumWidth(980)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Middle column: robot controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.create_robot_controls())
        left_layout.addStretch(1)
        left_scroll.setWidget(left_container)

        # Right-most column: detected objects + action flow + recording widgets
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.create_object_list())
        right_layout.addStretch(1)
        right_scroll.setWidget(right_container)

        layout.addWidget(left_scroll, stretch=1)
        layout.addWidget(right_scroll, stretch=1)
        widget.setLayout(layout)
        return widget

    def _create_speed_slider(
        self,
        min_val,
        max_val,
        default_val,
        callback,
        reverse: bool = False,
        value_fmt: str = "{:.3f}",
        label_text: str = "Step Size",
    ):
        """Create a compact speed slider row: [Slow] --slider-- [Fast] value_label."""
        row = QHBoxLayout()
        row.setSpacing(4)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        lo = float(min_val)
        hi = float(max_val)
        if hi < lo:
            lo, hi = hi, lo

        value = float(default_val) if np.isfinite(default_val) else lo
        value = float(np.clip(value, lo, hi))
        pos_norm = (value - lo) / (hi - lo) if hi > lo else 0.0
        if reverse:
            pos_norm = 1.0 - pos_norm
        slider.setValue(int(round(max(0.0, min(1.0, pos_norm)) * 100.0)))
        slider.setFixedHeight(20)

        val_label = QLabel(value_fmt.format(float(default_val)))
        val_label.setFixedWidth(45)
        val_label.setStyleSheet("QLabel { font-size: 10px; }")

        def on_change(v):
            p = float(int(v)) / 100.0
            if reverse:
                p = 1.0 - p
            speed = lo + (hi - lo) * p
            val_label.setText(value_fmt.format(float(speed)))
            callback(speed)

        slider.valueChanged.connect(on_change)

        row.addWidget(QLabel(label_text))
        row.addWidget(slider)
        row.addWidget(val_label)
        return row

    def _create_delay_slider(self, min_val, max_val, default_val, callback):
        """Create compact slider row for smoothing loop delay."""
        return self._create_speed_slider(
            min_val,
            max_val,
            default_val,
            callback,
            reverse=True,
            value_fmt="{:.3f}s",
            label_text="Speed",
        )

    def create_robot_controls(self):
        """Create robot control buttons"""
        widget = QGroupBox("Robot Controls (Bridge Commands)")
        layout = QVBoxLayout()

        # Base controls
        base_group = QGroupBox("Base Movement")
        base_layout = QGridLayout()
        base_layout.setSpacing(3)

        btn_forward = QPushButton("↑")
        btn_forward.setToolTip("Move Forward")
        btn_forward.setMinimumHeight(40)
        btn_forward.pressed.connect(lambda: self.start_control('base_linear', self.linear_speed))
        btn_forward.released.connect(lambda: self.stop_control('base_linear'))
        base_layout.addWidget(btn_forward, 0, 1)

        btn_left = QPushButton("←")
        btn_left.setToolTip("Rotate Left")
        btn_left.setMinimumHeight(40)
        btn_left.pressed.connect(lambda: self.start_control('base_angular', self.angular_speed))
        btn_left.released.connect(lambda: self.stop_control('base_angular'))
        base_layout.addWidget(btn_left, 1, 0)

        btn_backward = QPushButton("↓")
        btn_backward.setToolTip("Move Backward")
        btn_backward.setMinimumHeight(40)
        btn_backward.pressed.connect(lambda: self.start_control('base_linear', -self.linear_speed))
        btn_backward.released.connect(lambda: self.stop_control('base_linear'))
        base_layout.addWidget(btn_backward, 1, 1)

        btn_right = QPushButton("→")
        btn_right.setToolTip("Rotate Right")
        btn_right.setMinimumHeight(40)
        btn_right.pressed.connect(lambda: self.start_control('base_angular', -self.angular_speed))
        btn_right.released.connect(lambda: self.stop_control('base_angular'))
        base_layout.addWidget(btn_right, 1, 2)

        base_layout.addWidget(QLabel("Linear Step"), 2, 0)
        base_layout.addWidget(QLabel("Rotation Step"), 3, 0)

        def set_base_linear_step(v):
            self.linear_speed = v

        def set_base_angle_step_deg(v):
            self.base_angle_step_deg = float(v)
            if hasattr(self.ros_node, 'set_base_rotate_step_deg'):
                self.ros_node.set_base_rotate_step_deg(self.base_angle_step_deg)

        base_layout.addLayout(
            self._create_speed_slider(0.005, 0.20, self.linear_speed, set_base_linear_step), 2, 1, 1, 2)
        base_layout.addLayout(
            self._create_speed_slider(0.01, 20.0, self.base_angle_step_deg, set_base_angle_step_deg), 3, 1, 1, 2)

        base_layout.addWidget(QLabel("Move (cm)"), 4, 0)
        self.base_distance_cm_input = QLineEdit("0")
        self.base_distance_cm_input.setPlaceholderText("+100 / -100")
        move_dist_btn = QPushButton("Move Distance")

        def _move_base_distance():
            txt = self.base_distance_cm_input.text().strip()
            if not txt:
                return
            try:
                dx_m = float(txt) / 100.0
            except ValueError:
                if hasattr(self, "status_label"):
                    self.status_label.setText("Base move requires numeric cm value")
                return
            if abs(dx_m) < 1e-6:
                return
            try:
                if hasattr(self.ros_node, "stop_base"):
                    self.ros_node.stop_base()
                ok = self.ros_node.move_base_relative(
                    dx=dx_m,
                    dy=0.0,
                    dtheta=0.0,
                    blocking=False,
                    timeout_s=max(2.0, 4.0 + abs(dx_m) * 6.0),
                )
                if hasattr(self, "status_label"):
                    self.status_label.setText(
                        f"Base move command {dx_m:+.3f} m ({'sent' if ok else 'failed'})"
                    )
            except Exception as exc:
                if hasattr(self, "status_label"):
                    self.status_label.setText(f"Base move failed: {exc}")

        move_dist_btn.clicked.connect(_move_base_distance)
        base_layout.addWidget(self.base_distance_cm_input, 4, 1)
        base_layout.addWidget(move_dist_btn, 4, 2)

        base_group.setLayout(base_layout)
        layout.addWidget(base_group)

        # Command smoothing controls (applies to all published qpos channels).
        smooth_group = QGroupBox("Command Smoothing")
        smooth_layout = QVBoxLayout()
        smooth_layout.setSpacing(3)

        def set_command_smoothing_delay(v):
            self.command_smoothing_delay = v
            self.ros_node.set_command_smoothing_delay(v)
            if hasattr(self.ros_node, 'set_base_rotate_step_delay'):
                self.ros_node.set_base_rotate_step_delay(v)

        smooth_layout.addLayout(
            self._create_delay_slider(0.20, 0.01, self.command_smoothing_delay, set_command_smoothing_delay)
        )
        smooth_hint = QLabel("Large target jumps are split into small per-channel steps before publish.")
        smooth_hint.setWordWrap(True)
        smooth_hint.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        smooth_layout.addWidget(smooth_hint)
        smooth_group.setLayout(smooth_layout)
        layout.addWidget(smooth_group)

        # Arm controls
        arm_group = QGroupBox("Arm")
        arm_layout = QGridLayout()
        arm_layout.setSpacing(3)

        btn_lift_up = QPushButton("Lift ↑")
        btn_lift_up.setMinimumHeight(35)
        btn_lift_up.pressed.connect(lambda: self.start_control_incremental('arm_lift', self.arm_speed))
        btn_lift_up.released.connect(lambda: self.stop_control('arm_lift'))
        arm_layout.addWidget(btn_lift_up, 0, 0)

        btn_lift_down = QPushButton("Lift ↓")
        btn_lift_down.setMinimumHeight(35)
        btn_lift_down.pressed.connect(lambda: self.start_control_incremental('arm_lift', -self.arm_speed))
        btn_lift_down.released.connect(lambda: self.stop_control('arm_lift'))
        arm_layout.addWidget(btn_lift_down, 0, 1)

        btn_extend = QPushButton("Extend →")
        btn_extend.setMinimumHeight(35)
        btn_extend.pressed.connect(lambda: self.start_control_incremental('arm_extension', self.arm_speed))
        btn_extend.released.connect(lambda: self.stop_control('arm_extension'))
        arm_layout.addWidget(btn_extend, 1, 0)

        btn_retract = QPushButton("Retract ←")
        btn_retract.setMinimumHeight(35)
        btn_retract.pressed.connect(lambda: self.start_control_incremental('arm_extension', -self.arm_speed))
        btn_retract.released.connect(lambda: self.stop_control('arm_extension'))
        arm_layout.addWidget(btn_retract, 1, 1)

        arm_layout.addLayout(
            self._create_speed_slider(0.005, 0.10, self.arm_speed, lambda v: setattr(self, 'arm_speed', v)), 2, 0, 1, 2)

        arm_group.setLayout(arm_layout)
        layout.addWidget(arm_group)

        # Head controls
        head_group = QGroupBox("Head")
        head_layout = QGridLayout()
        head_layout.setSpacing(3)

        btn_pan_left = QPushButton("Pan ←")
        btn_pan_left.setMinimumHeight(35)
        btn_pan_left.pressed.connect(lambda: self.start_control_incremental('head_pan', self.head_speed))
        btn_pan_left.released.connect(lambda: self.stop_control('head_pan'))
        head_layout.addWidget(btn_pan_left, 0, 0)

        btn_pan_right = QPushButton("Pan →")
        btn_pan_right.setMinimumHeight(35)
        btn_pan_right.pressed.connect(lambda: self.start_control_incremental('head_pan', -self.head_speed))
        btn_pan_right.released.connect(lambda: self.stop_control('head_pan'))
        head_layout.addWidget(btn_pan_right, 0, 1)

        btn_tilt_up = QPushButton("Tilt ↑")
        btn_tilt_up.setMinimumHeight(35)
        btn_tilt_up.pressed.connect(lambda: self.start_control_incremental('head_tilt', self.head_speed))
        btn_tilt_up.released.connect(lambda: self.stop_control('head_tilt'))
        head_layout.addWidget(btn_tilt_up, 1, 0)

        btn_tilt_down = QPushButton("Tilt ↓")
        btn_tilt_down.setMinimumHeight(35)
        btn_tilt_down.pressed.connect(lambda: self.start_control_incremental('head_tilt', -self.head_speed))
        btn_tilt_down.released.connect(lambda: self.stop_control('head_tilt'))
        head_layout.addWidget(btn_tilt_down, 1, 1)

        head_layout.addLayout(
            self._create_speed_slider(0.02, 0.30, self.head_speed, lambda v: setattr(self, 'head_speed', v)), 2, 0, 1, 2)

        head_group.setLayout(head_layout)
        layout.addWidget(head_group)

        # Wrist controls
        wrist_group = QGroupBox("Wrist")
        wrist_layout = QGridLayout()
        wrist_layout.setSpacing(3)

        btn_roll_left = QPushButton("Roll ↶")
        btn_roll_left.setMinimumHeight(30)
        btn_roll_left.pressed.connect(lambda: self.start_control_incremental('wrist_roll', -self.wrist_speed))
        btn_roll_left.released.connect(lambda: self.stop_control('wrist_roll'))
        wrist_layout.addWidget(btn_roll_left, 0, 0)

        btn_roll_right = QPushButton("Roll ↷")
        btn_roll_right.setMinimumHeight(30)
        btn_roll_right.pressed.connect(lambda: self.start_control_incremental('wrist_roll', self.wrist_speed))
        btn_roll_right.released.connect(lambda: self.stop_control('wrist_roll'))
        wrist_layout.addWidget(btn_roll_right, 0, 1)

        btn_pitch_down = QPushButton("Pitch ↓")
        btn_pitch_down.setMinimumHeight(30)
        btn_pitch_down.pressed.connect(lambda: self.start_control_incremental('wrist_pitch', -self.wrist_speed))
        btn_pitch_down.released.connect(lambda: self.stop_control('wrist_pitch'))
        wrist_layout.addWidget(btn_pitch_down, 1, 0)

        btn_pitch_up = QPushButton("Pitch ↑")
        btn_pitch_up.setMinimumHeight(30)
        btn_pitch_up.pressed.connect(lambda: self.start_control_incremental('wrist_pitch', self.wrist_speed))
        btn_pitch_up.released.connect(lambda: self.stop_control('wrist_pitch'))
        wrist_layout.addWidget(btn_pitch_up, 1, 1)

        btn_yaw_left = QPushButton("Yaw ←")
        btn_yaw_left.setMinimumHeight(30)
        btn_yaw_left.pressed.connect(lambda: self.start_control_incremental('wrist_yaw', self.wrist_speed))
        btn_yaw_left.released.connect(lambda: self.stop_control('wrist_yaw'))
        wrist_layout.addWidget(btn_yaw_left, 2, 0)

        btn_yaw_right = QPushButton("Yaw →")
        btn_yaw_right.setMinimumHeight(30)
        btn_yaw_right.pressed.connect(lambda: self.start_control_incremental('wrist_yaw', -self.wrist_speed))
        btn_yaw_right.released.connect(lambda: self.stop_control('wrist_yaw'))
        wrist_layout.addWidget(btn_yaw_right, 2, 1)

        wrist_layout.addLayout(
            self._create_speed_slider(0.02, 0.30, self.wrist_speed, lambda v: setattr(self, 'wrist_speed', v)), 3, 0, 1, 2)

        wrist_group.setLayout(wrist_layout)
        layout.addWidget(wrist_group)

        # Gripper controls
        gripper_group = QGroupBox("Gripper")
        gripper_layout = QGridLayout()
        gripper_layout.setSpacing(3)

        btn_gripper_open = QPushButton("Open +")
        btn_gripper_open.setMinimumHeight(35)
        btn_gripper_open.setToolTip("Open gripper by one step")
        btn_gripper_open.clicked.connect(lambda: self.adjust_gripper_step(+1))
        gripper_layout.addWidget(btn_gripper_open, 0, 0)

        btn_gripper_close = QPushButton("Close -")
        btn_gripper_close.setMinimumHeight(35)
        btn_gripper_close.setToolTip("Close gripper by one step")
        btn_gripper_close.clicked.connect(lambda: self.adjust_gripper_step(-1))
        gripper_layout.addWidget(btn_gripper_close, 0, 1)

        btn_gripper_open_full = QPushButton("Open Full")
        btn_gripper_open_full.setMinimumHeight(35)
        btn_gripper_open_full.setToolTip("Open gripper to maximum limit")
        btn_gripper_open_full.clicked.connect(lambda: self.set_gripper(self.ros_node.JOINT_LIMITS[7][1]))
        gripper_layout.addWidget(btn_gripper_open_full, 1, 0)

        btn_gripper_close_full = QPushButton("Close Full")
        btn_gripper_close_full.setMinimumHeight(35)
        btn_gripper_close_full.setToolTip("Close gripper to minimum limit")
        btn_gripper_close_full.clicked.connect(lambda: self.set_gripper(self.ros_node.JOINT_LIMITS[7][0]))
        gripper_layout.addWidget(btn_gripper_close_full, 1, 1)

        gripper_layout.addLayout(
            self._create_speed_slider(
                0.005, 0.10, self.gripper_step,
                lambda v: setattr(self, 'gripper_step', v)
            ),
            2, 0, 1, 2
        )

        gripper_group.setLayout(gripper_layout)
        layout.addWidget(gripper_group)

        widget.setLayout(layout)
        return widget

    def create_object_list(self):
        """Create object list widget"""
        widget = QGroupBox("Detected Objects")
        layout = QVBoxLayout()

        # List widget (scrollable)
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.on_object_selected)
        self.object_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.object_list, stretch=1)

        # Action buttons
        btn_layout = QHBoxLayout()

        self.center_button = QPushButton("Center")
        self.center_button.setToolTip("Center camera on selected object")
        self.center_button.clicked.connect(self.center_camera_on_object)
        self.center_button.setEnabled(False)
        self.center_button.setMinimumHeight(35)
        btn_layout.addWidget(self.center_button)

        self.reach_button = QPushButton("Reach")
        self.reach_button.setToolTip("Move arm above selected object (10cm clearance)")
        self.reach_button.clicked.connect(self.reach_object)
        self.reach_button.setEnabled(False)
        self.reach_button.setMinimumHeight(35)
        btn_layout.addWidget(self.reach_button)

        self.grasp_button = QPushButton("Grasp")
        self.grasp_button.setToolTip("Move to and grasp selected object")
        self.grasp_button.clicked.connect(self.grasp_object)
        self.grasp_button.setEnabled(False)
        self.grasp_button.setMinimumHeight(35)
        btn_layout.addWidget(self.grasp_button)

        layout.addLayout(btn_layout)

        # Return button (full width, below action buttons)
        self.return_button = QPushButton("Return")
        self.return_button.setToolTip("Return arm and base to position before last reach/grasp")
        self.return_button.clicked.connect(self.return_to_start)
        self.return_button.setEnabled(False)
        self.return_button.setMinimumHeight(35)
        layout.addWidget(self.return_button)

        # Play/Pause/Continue button for long actions
        self.play_pause_button = QPushButton("Pause")
        self.play_pause_button.setToolTip("Pause running action / Continue paused action")
        self.play_pause_button.clicked.connect(self.on_play_pause_clicked)
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setMinimumHeight(35)
        layout.addWidget(self.play_pause_button)

        self.next_goal_button = QPushButton("Go To Next Goal")
        self.next_goal_button.setToolTip("Execute next queued goal (or skip current action and move to next queued goal)")
        self.next_goal_button.clicked.connect(self.go_to_next_goal)
        self.next_goal_button.setEnabled(False)
        self.next_goal_button.setMinimumHeight(35)
        layout.addWidget(self.next_goal_button)

        self.goal_queue_label = QLabel("Queued goals: (none)")
        self.goal_queue_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        self.goal_queue_label.setWordWrap(True)
        layout.addWidget(self.goal_queue_label)

        # Demonstration recording controls
        record_group = QGroupBox("LeRobot Demo Recording")
        record_layout = QGridLayout()
        record_layout.setSpacing(4)

        record_layout.addWidget(QLabel("Prompt"), 0, 0)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g. pick up the red block and place it in the tray")
        self.prompt_input.textChanged.connect(self.on_prompt_changed)
        record_layout.addWidget(self.prompt_input, 0, 1, 1, 2)

        record_layout.addWidget(QLabel("Folder"), 1, 0)
        self.record_folder_input = QLineEdit(self.dataset_root)
        self.record_folder_input.textChanged.connect(self.on_record_folder_changed)
        record_layout.addWidget(self.record_folder_input, 1, 1)

        self.browse_record_folder_button = QPushButton("Browse")
        self.browse_record_folder_button.clicked.connect(self.browse_record_folder)
        self.browse_record_folder_button.setMinimumHeight(30)
        record_layout.addWidget(self.browse_record_folder_button, 1, 2)

        self.record_toggle_button = QPushButton("Record")
        self.record_toggle_button.setMinimumHeight(35)
        self.record_toggle_button.setToolTip("Start/stop recording demonstration in LeRobot-style layout")
        self.record_toggle_button.clicked.connect(self.toggle_demo_recording)
        record_layout.addWidget(self.record_toggle_button, 2, 0, 1, 3)

        record_group.setLayout(record_layout)
        layout.addWidget(record_group)

        # Status label
        self.status_label = QLabel("No segmentation yet")
        self.status_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumHeight(60)
        layout.addWidget(self.status_label)

        widget.setLayout(layout)
        return widget

    def _set_action_state(self, state):
        """Thread-safe action-state update (logic state + UI state)."""
        with self._action_lock:
            self._action_state = state
        self.ui_action_state_signal.emit(state)

    def _apply_action_state_ui(self, state):
        """Apply action-state visuals on the Qt thread."""
        if state == 'idle':
            self.play_pause_button.setEnabled(False)
            self.play_pause_button.setText("Pause")
        elif state == 'running':
            self.play_pause_button.setEnabled(True)
            self.play_pause_button.setText("Pause")
        elif state in ('paused', 'awaiting_confirm'):
            self.play_pause_button.setEnabled(True)
            self.play_pause_button.setText("Continue")
            self.return_button.setEnabled(True)
        elif state == 'awaiting_post_grasp':
            self.play_pause_button.setEnabled(True)
            self.play_pause_button.setText("Continue")
            self.return_button.setEnabled(True)
        self._update_next_goal_button_state()

    def _goal_sequence_order(self):
        goals = []
        if self.queued_goals.get("grasp") is not None:
            goals.append(self.queued_goals["grasp"])
        if self.queued_goals.get("reach") is not None:
            goals.append(self.queued_goals["reach"])
        return goals

    def _goal_sequence_has_next(self):
        return self.queued_goal_cursor < len(self._goal_sequence_order())

    def _update_goal_queue_label(self):
        goals = self._goal_sequence_order()
        if not hasattr(self, "goal_queue_label"):
            return
        if not goals:
            self.goal_queue_label.setText("Queued goals: (none)")
            self.goal_queue_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
            return
        labels = []
        for idx, g in enumerate(goals):
            prefix = "-> " if idx == self.queued_goal_cursor and self._goal_sequence_has_next() else "   "
            labels.append(f"{prefix}{idx+1}. {g['kind']} ({g['px']},{g['py']})")
        self.goal_queue_label.setText("Queued goals:\n" + "\n".join(labels))
        self.goal_queue_label.setStyleSheet("QLabel { color: #555; font-size: 10px; }")

    def _update_next_goal_button_state(self):
        if not hasattr(self, "next_goal_button"):
            return
        with self._action_lock:
            st = self._action_state
        enabled = False
        if self._goal_sequence_has_next():
            # Allow starting next goal from idle, or skipping to next during an active/paused action.
            enabled = st in ('idle', 'running', 'paused', 'awaiting_confirm')
        self.next_goal_button.setEnabled(enabled)

    def _reset_goal_sequence_progress(self):
        self.queued_goal_cursor = 0
        self.queued_sequence_started = False
        self._deferred_next_goal_start = False
        self._skip_to_next_goal_requested = False
        self._update_goal_queue_label()
        self._update_next_goal_button_state()

    def _apply_status_update(self, text, style):
        self.status_label.setText(text)
        if style:
            self.status_label.setStyleSheet(style)

    def _set_status(self, text, style=None):
        self.ui_status_signal.emit(text, style or "")

    def _set_return_enabled(self, enabled):
        self.ui_return_enabled_signal.emit(bool(enabled))

    def _begin_action(self, mode):
        """Start a new reach/grasp action if none is running."""
        with self._action_lock:
            if self._action_state != 'idle':
                return False
            self._action_mode = mode
            self._action_abort_requested = False
            self._action_state = 'running'
        self._set_action_state('running')
        self.return_button.setEnabled(False)
        return True

    def _is_abort_requested(self):
        with self._action_lock:
            return self._action_abort_requested

    def on_play_pause_clicked(self):
        """Pause/continue active action. In post-grasp hold, Continue triggers lift."""
        with self._action_lock:
            state = self._action_state

        if state == 'running':
            self._set_action_state('paused')
            self.status_label.setText("Paused. Press Continue or Return.")
            self.status_label.setStyleSheet("QLabel { color: orange; font-size: 10px; }")
            return

        if state in ('paused', 'awaiting_confirm'):
            self._set_action_state('running')
            self.status_label.setText("Resuming action...")
            self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")
            return

        if state == 'awaiting_post_grasp':
            self._set_action_state('running')
            self.status_label.setText("Resuming action...")
            self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")

    def start_control(self, control_name, value):
        """Start continuous control (velocity-based)"""
        self.robot_controller.set_control(control_name, value)

    def start_control_incremental(self, control_name, delta):
        """Start incremental control (position-based)"""
        # For position controls, we use a timer to apply incremental changes
        if not hasattr(self, 'control_timers'):
            self.control_timers = {}

        if control_name in self.control_timers and self.control_timers[control_name].isActive():
            return

        def apply_increment():
            self.robot_controller.adjust_control(control_name, delta)

        timer = QTimer()
        timer.timeout.connect(apply_increment)
        timer.start(50)  # 20Hz updates
        self.control_timers[control_name] = timer

    def stop_control(self, control_name):
        """Stop continuous control"""
        # Stop velocity-based controls
        if control_name in ['base_linear', 'base_angular']:
            self.robot_controller.set_control(control_name, 0.0)

        # Stop incremental timers
        if hasattr(self, 'control_timers') and control_name in self.control_timers:
            self.control_timers[control_name].stop()

    def set_gripper(self, value):
        """Set gripper state"""
        lo, hi = self.ros_node.JOINT_LIMITS[7]
        try:
            clipped = float(np.clip(float(value), lo, hi))
        except (TypeError, ValueError):
            return
        self._set_manual_gripper_override(clipped)
        self.robot_controller.set_control('gripper', clipped)

    def _set_manual_gripper_override(self, value: float | None) -> None:
        if value is None or not isinstance(value, (int, float)):
            self._manual_gripper_override = None
            return
        try:
            v = float(value)
            if not math.isfinite(v):
                self._manual_gripper_override = None
                return
        except (TypeError, ValueError):
            self._manual_gripper_override = None
            return
        lo, hi = self.ros_node.JOINT_LIMITS[7]
        self._manual_gripper_override = float(np.clip(v, lo, hi))

    def _get_manual_gripper_target(self, fallback: float | None = None) -> float | None:
        if self._manual_gripper_override is not None:
            return float(self._manual_gripper_override)

        target = self.ros_node.get_target_qpos()
        if isinstance(target, list) and len(target) >= 8:
            try:
                v = float(target[7])
                if math.isfinite(v):
                    lo, hi = self.ros_node.JOINT_LIMITS[7]
                    return float(np.clip(v, lo, hi))
            except (TypeError, ValueError):
                pass

        actual = self.ros_node.get_actual_qpos()
        if isinstance(actual, list) and len(actual) >= 8:
            try:
                v = float(actual[7])
                if math.isfinite(v):
                    lo, hi = self.ros_node.JOINT_LIMITS[7]
                    return float(np.clip(v, lo, hi))
            except (TypeError, ValueError):
                pass

        return None if fallback is None else float(fallback)

    def on_prompt_changed(self, text):
        self.record_prompt = text

    def on_record_folder_changed(self, text):
        self.dataset_root = text.strip()

    def browse_record_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select LeRobot Dataset Folder",
            self.dataset_root or str(Path.cwd()),
        )
        if folder:
            self.dataset_root = folder
            self.record_folder_input.setText(folder)

    def _build_record_sample(self):
        sensors = self.ros_node.get_sensor_snapshot()
        actual_qpos = self.ros_node.get_actual_qpos()
        command_qpos = self.ros_node.get_published_qpos()
        measured_pose = self.ros_node.get_measured_base_pose_xytheta() or [0.0, 0.0, 0.0]
        command_pose = self.ros_node.get_command_base_pose_xytheta() or list(measured_pose)

        # Raw state is the exact 10D measured qpos vector used internally:
        # [first 8 joint positions, base linear vel, base angular vel]
        state_raw_v5 = list(actual_qpos) if actual_qpos else []
        action_raw_v5 = list(actual_qpos) if actual_qpos else []
        action_command_raw_v5 = list(command_qpos) if command_qpos else []

        state = state_raw_v5[:8] + [float(v) for v in measured_pose]
        action = action_raw_v5[:8] + [float(v) for v in measured_pose]
        action_command = action_command_raw_v5[:8] + [float(v) for v in command_pose]

        sensors["observation.state_raw_v5"] = state_raw_v5
        sensors["observation.qpos_actual_raw_v5"] = action_raw_v5
        sensors["observation.qpos_full_raw_v5"] = action_command_raw_v5
        sensors["action_raw_v5"] = action_raw_v5
        sensors["action_command_raw_v5"] = action_command_raw_v5

        return {
            "timestamp": time.time(),
            "head_rgb": self.head_rgb if self.head_rgb is not None else None,
            "wrist_rgb": self.wrist_rgb if self.wrist_rgb is not None else None,
            "head_depth": self.depth_image if self.depth_image is not None else None,
            "wrist_depth": self.wrist_depth if self.wrist_depth is not None else None,
            "state": state,
            "action": action,
            "action_command": action_command,
            "sensors": sensors,
        }

    def start_demo_recording(self):
        if self.is_recording_demo:
            return
        if not self.dataset_root:
            self.status_label.setText("Set a recording folder first")
            self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")
            return
        prompt = (self.record_prompt or "").strip()
        if not prompt:
            prompt = "unspecified_task"
            self.prompt_input.setText(prompt)
        try:
            self.demo_recorder.start(self.dataset_root, prompt)
            self.is_recording_demo = True
            self.record_toggle_button.setText("Stop Recording")
            self.status_label.setText(f"Recording demo: {prompt} (fps={self.demo_recorder.target_fps:.1f})")
            self.status_label.setStyleSheet("QLabel { color: #1e88e5; font-size: 10px; }")
        except Exception as e:
            self.status_label.setText(f"Record start failed: {e}")
            self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")

    def stop_demo_recording(self):
        if not self.is_recording_demo:
            return
        try:
            summary = self.demo_recorder.stop()
            self.is_recording_demo = False
            self.record_toggle_button.setText("Record")
            if summary is None:
                self.status_label.setText("Recording stopped")
                self.status_label.setStyleSheet("QLabel { color: green; font-size: 10px; }")
            else:
                self.status_label.setText(
                    f"Saved demo ep {summary['episode_index']} "
                    f"({summary['num_frames']} frames, dropped={summary.get('dropped_frames', 0)})"
                )
                self.status_label.setStyleSheet("QLabel { color: green; font-size: 10px; }")
                print(f"Demo saved: {summary}")
        except Exception as e:
            self.status_label.setText(f"Record stop failed: {e}")
            self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")

    def toggle_demo_recording(self):
        if self.is_recording_demo:
            self.stop_demo_recording()
        else:
            self.start_demo_recording()

    def adjust_gripper_step(self, direction):
        """Incrementally open/close gripper by configured step size."""
        grip_min, grip_max = self.ros_node.JOINT_LIMITS[7]
        getter = getattr(self, "_get_manual_gripper_target", None)
        if getter is None and self.ros_node is not None:
            getter = getattr(self.ros_node, "_get_manual_gripper_target", None)
        if callable(getter):
            current = getter(fallback=float(self.ros_node.JOINT_LIMITS[7][1]))
        else:
            target = self.ros_node.get_target_qpos()
            if isinstance(target, list) and len(target) >= 8:
                try:
                    current = float(target[7])
                except (TypeError, ValueError):
                    current = float(self.ros_node.JOINT_LIMITS[7][1])
            else:
                current = float(self.ros_node.JOINT_LIMITS[7][1])
        if current is None:
            try:
                current = float(self.ros_node.JOINT_LIMITS[7][1])
            except (TypeError, ValueError):
                current = 0.0

        delta = self.gripper_step if direction > 0 else -self.gripper_step
        target = max(grip_min, min(grip_max, current + delta))
        self.set_gripper(target)
        self.status_label.setText(
            f"Gripper -> {target:.3f} (step {self.gripper_step:.3f})"
        )
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")

    def on_images_updated(self, head_rgb, wrist_rgb, depth_image, wrist_depth):
        """Handle updated images from robot thread"""
        import time
        # print(time.ctime(), " >> I am called >> ", head_rgb is not None)
        self.head_rgb = head_rgb.copy()
        self.wrist_rgb = wrist_rgb.copy()
        self.depth_image = depth_image.copy()
        self.wrist_depth = wrist_depth.copy()
        if self.is_recording_demo:
            try:
                self.demo_recorder.record_step(self._build_record_sample())
            except Exception as e:
                print(f"Recording step error: {e}")
                self.stop_demo_recording()
                self.status_label.setText(f"Recording stopped due error: {e}")
                self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")
        self.update_camera_displays()

    def on_fps_updated(self, fps):
        """Handle FPS updates"""
        head_shape = self.head_rgb.shape if self.head_rgb is not None else 'N/A'
        wrist_shape = self.wrist_rgb.shape if self.wrist_rgb is not None else 'N/A'
        robot_status = "Ready" if self.ros_node.is_ready() else "Initializing..."
        image_source = "bridge"
        if hasattr(self.ros_node, "get_image_source"):
            try:
                image_source = str(self.ros_node.get_image_source())
            except Exception:
                image_source = "bridge"
        status_color = "#27ae60" if self.ros_node.is_ready() else "#f39c12"
        self.fps_label.setText(
            f"FPS: {fps:.1f} | Robot: {robot_status} | ImageSrc: {image_source} | "
            f"Head: {head_shape} | Wrist: {wrist_shape}"
        )
        self.fps_label.setStyleSheet(f"QLabel {{ padding: 5px; background-color: {status_color}; color: white; }}")

    def update_camera_displays(self):
        """Update camera display widgets"""
        # Update Head RGB
        # print(self.head_rgb.shape)
        if self.head_rgb is not None:
            head_display = self.head_rgb.copy()
            # Apply mask overlay if available and using head camera for segmentation
            if self.mask_overlay is not None and self.use_head_for_segmentation:
                head_display = cv2.addWeighted(head_display, 0.7, self.mask_overlay, 0.3, 0)

            # Draw 3D grasp axes and bounding box for debugging
            if self._grasp_debug_info is not None and 'axis_pixels' in self._grasp_debug_info:
                ap = self._grasp_debug_info['axis_pixels']
                # Bounding rectangle corners — CYAN outline
                if 'corners' in ap and len(ap['corners']) >= 3:
                    corners = ap['corners']
                    for i in range(len(corners)):
                        cv2.line(head_display,
                                 corners[i], corners[(i + 1) % len(corners)],
                                 (255, 255, 0), 1, cv2.LINE_AA)
                # Long axis — GREEN line
                if 'long1' in ap and 'long2' in ap:
                    cv2.line(head_display, ap['long1'], ap['long2'],
                             (0, 255, 0), 2, cv2.LINE_AA)
                # Narrow axis (grasp direction) — RED line
                if 'narrow1' in ap and 'narrow2' in ap:
                    cv2.line(head_display, ap['narrow1'], ap['narrow2'],
                             (0, 0, 255), 2, cv2.LINE_AA)
                # Center dot — YELLOW
                if 'center' in ap:
                    cv2.circle(head_display, ap['center'], 5, (0, 255, 255), -1)
                # Labels
                if 'long1' in ap:
                    cv2.putText(head_display, "long", ap['long1'],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if 'narrow1' in ap:
                    cv2.putText(head_display, "grasp", ap['narrow1'],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            head_pixmap = self.numpy_to_pixmap(head_display)
            # Scale to fit container keeping aspect ratio, then size label to match
            available = self.head_container.size()
            scaled_pixmap = head_pixmap.scaled(
                available,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.head_display.setFixedSize(scaled_pixmap.size())
            self.head_display.setPixmap(scaled_pixmap)

        # Update Wrist RGB
        if self.wrist_rgb is not None:
            # First image received — expand container to full size
            if not self._wrist_feed_active:
                self._wrist_feed_active = True
                self.wrist_container.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
                self.wrist_container.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.wrist_label.setText("Gripper RGB Camera")
                # Give it layout stretch comparable to head camera
                layout = self.wrist_container.parentWidget().layout()
                if layout is not None:
                    idx = layout.indexOf(self.wrist_container)
                    if idx >= 0:
                        layout.setStretch(idx, 4)

            wrist_display = self.wrist_rgb.copy()
            wrist_pixmap = self.numpy_to_pixmap(wrist_display)
            available = self.wrist_container.size()
            scaled_pixmap = wrist_pixmap.scaled(
                available,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.wrist_display.setFixedSize(scaled_pixmap.size())
            self.wrist_display.setPixmap(scaled_pixmap)

    def numpy_to_pixmap(self, image):
        """Convert numpy array to QPixmap"""
        try:
            # Ensure image is in correct format
            if len(image.shape) == 2:
                # Grayscale - convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                # RGBA - convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Ensure uint8
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 255)).astype(np.uint8)

            height, width, channel = image.shape
            bytes_per_line = 3 * width

            # Make sure data is contiguous
            image = np.ascontiguousarray(image)

            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(q_image)
        except Exception as e:
            print(f"Error converting image to pixmap: {e}")
            # Return empty pixmap
            return QPixmap()

    def run_segmentation(self):
        """Run SAM segmentation"""
        # Use head camera for segmentation
        rgb_for_seg = self.head_rgb if self.use_head_for_segmentation else self.wrist_rgb

        if rgb_for_seg is None:
            self.status_label.setText("No camera image available")
            self.status_label.setStyleSheet("QLabel { color: red; }")
            return

        if self.seg_thread.isRunning():
            print("Segmentation already running")
            return

        self.segment_button.setEnabled(False)
        self.segment_button.setText("Segmenting...")
        self.status_label.setText("Running SAM segmentation on GPU...")
        self.status_label.setStyleSheet("QLabel { color: orange; font-size: 10px; }")

        # Set images and start thread
        self.seg_thread.set_images(rgb_for_seg.copy(), self.depth_image.copy())
        self.seg_thread.start()

    def on_model_loading(self):
        """Handle model loading signal"""
        self.status_label.setText("Loading SAM model on GPU...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")

    def on_segmentation_complete(self, segments, mask_overlay):
        """Handle segmentation completion"""
        self.segments = segments
        self.mask_overlay = mask_overlay
        self._grasp_debug_info = None  # Clear old axis overlay

        # Update object list
        self.object_list.clear()
        for i, seg in enumerate(segments):
            color_hex = '#%02x%02x%02x' % seg['color']
            label = f"Object {i+1} - Area: {seg['area']:.0f}px - Depth: {seg['depth']:.2f}m"
            item = QListWidgetItem(label)
            item.setBackground(QColor(color_hex))
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.object_list.addItem(item)

        self.status_label.setText(f"Found {len(segments)} objects")
        self.status_label.setStyleSheet("QLabel { color: green; }")
        self.segment_button.setEnabled(True)
        self.segment_button.setText("Run SAM Segmentation")

        # Update display
        self.update_camera_displays()

    def on_error(self, error_msg):
        """Handle error messages"""
        print(f"Error: {error_msg}")
        self.status_label.setText(error_msg)
        self.status_label.setStyleSheet("QLabel { color: red; }")
        if not self.segment_button.isEnabled():
            self.segment_button.setEnabled(True)
            self.segment_button.setText("Run SAM Segmentation")

    def on_object_selected(self, item):
        """Handle object selection from list"""
        idx = item.data(Qt.ItemDataRole.UserRole)
        self.selected_segment = self.segments[idx]
        self.center_button.setEnabled(True)
        self.reach_button.setEnabled(True)
        self.grasp_button.setEnabled(True)
        print(f"Selected object {idx}: center={self.selected_segment['center']}, depth={self.selected_segment['depth']:.3f}m")

    def on_image_click(self, event):
        """Handle click on head RGB image — left-click selects segment, right-click shows context menu"""
        if self.head_rgb is None:
            return

        # Get click position in image coordinates
        px, py = self._pixel_from_event(event)
        if px is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Left-click: select segment under cursor
            if not self.segments:
                return
            for i, seg in enumerate(self.segments):
                if py < seg['mask'].shape[0] and px < seg['mask'].shape[1]:
                    if seg['mask'][py, px] > 0:
                        self.object_list.setCurrentRow(i)
                        self.selected_segment = seg
                        self.center_button.setEnabled(True)
                        self.reach_button.setEnabled(True)
                        self.grasp_button.setEnabled(True)
                        print(f"Clicked on object {i}")
                        break

        elif event.button() == Qt.MouseButton.RightButton:
            # Right-click: show context menu with Center / Reach / Grasp
            clicked_on_segment = False
            if self.segments:
                for i, seg in enumerate(self.segments):
                    if py < seg['mask'].shape[0] and px < seg['mask'].shape[1]:
                        if seg['mask'][py, px] > 0:
                            clicked_on_segment = True
                            clicked_segment = seg
                            break

            menu = QMenu(self)

            center_action = QAction("Center", self)
            center_action.triggered.connect(lambda: self._center_on_pixel(px, py))
            menu.addAction(center_action)

            reach_action = QAction("Reach", self)
            reach_action.triggered.connect(lambda: self._reach_to_pixel(px, py))
            menu.addAction(reach_action)

            grasp_action = QAction("Grasp", self)
            if clicked_on_segment:
                grasp_action.triggered.connect(lambda: self._grasp_at_pixel(px, py))
            else:
                grasp_action.setEnabled(False)
            menu.addAction(grasp_action)

            menu.addSeparator()

            add_reach_goal_action = QAction("Add Reach Goal", self)
            add_reach_goal_action.triggered.connect(lambda: self.add_reach_goal_at_pixel(px, py))
            menu.addAction(add_reach_goal_action)

            add_grasp_goal_action = QAction("Add Grasp Goal", self)
            if clicked_on_segment:
                add_grasp_goal_action.triggered.connect(
                    lambda: self.add_grasp_goal_at_pixel(px, py, segment=clicked_segment)
                )
            else:
                add_grasp_goal_action.setEnabled(False)
            menu.addAction(add_grasp_goal_action)

            menu.exec(self.head_display.mapToGlobal(event.pos()))

    def center_camera_on_object(self):
        """Center camera on selected object"""
        if self.selected_segment is None or self.head_rgb is None:
            return

        cx, cy = self.selected_segment['center']
        img_height, img_width = self.head_rgb.shape[:2]

        # Calculate offset
        offset_x = (cx - img_width // 2) / img_width
        offset_y = (cy - img_height // 2) / img_height

        # Pan/tilt adjustment (small increments)
        pan_adj = -offset_x * 0.2
        tilt_adj = offset_y * 0.2

        print(f"Centering camera: pan={pan_adj:.3f}, tilt={tilt_adj:.3f}")

        # Apply adjustment incrementally
        self.robot_controller.adjust_control('head_pan', pan_adj)
        self.robot_controller.adjust_control('head_tilt', tilt_adj)

        self.status_label.setText("Centering camera...")
        self.status_label.setStyleSheet("QLabel { color: blue; }")

    def grasp_object(self):
        """Grasp selected object using orientation-aware approach (button handler)"""
        if self.selected_segment is None or self.head_rgb is None:
            self.status_label.setText("No object selected")
            self.status_label.setStyleSheet("QLabel { color: red; }")
            return

        cx, cy = self.selected_segment['center']
        self._grasp_at_pixel(cx, cy, segment=self.selected_segment)

    def _get_3d_point_at_pixel(self, px, py):
        """Convert a pixel (in rotated image coords) to a 3D point in base_link.
        Returns (point_base, depth) or (None, None) on failure."""
        if self.head_rgb is None or self.depth_image is None:
            return None, None

        h, w = self.depth_image.shape
        if not (0 <= py < h and 0 <= px < w):
            return None, None

        # Average depth over a 5x5 window for robustness
        y_min, y_max = max(0, py - 2), min(h, py + 3)
        x_min, x_max = max(0, px - 2), min(w, px + 3)
        region = self.depth_image[y_min:y_max, x_min:x_max]
        valid = region[(region > 0.1) & (region < 5.0)]
        if len(valid) == 0:
            return None, None
        depth = float(valid.mean())

        point_camera = self.ros_node.pixel_to_3d_point(px, py, depth)
        if point_camera is None:
            return None, None

        point_base = self.ros_node.transform_point_to_base(point_camera)
        return point_base, depth

    @staticmethod
    def _compute_gripper_reach_drop(wrist_yaw=0.0, wrist_pitch=0.0, wrist_roll=0.0):
        """Compute effective horizontal reach and vertical drop of the grasp
        center relative to the arm extension axis (link_arm_l0 end), given
        the current wrist joint angles.

        The kinematic chain (from URDF, Stretch SE3 with SG3 gripper):
          link_arm_l0
            -> joint_wrist_yaw  (revolute, Z-axis)
               offset from arm end: x=0.083, y=-0.031
            -> joint_wrist_pitch (revolute)
               offset: y=0.019, z=-0.031
            -> joint_wrist_roll  (revolute)
               offset: x=-0.019, y=-0.024, z=0.020
            -> gripper body  (fixed)
               offset: z=0.021
            -> grasp center  (fixed)
               offset: z=0.230

        For our grasp planning we care about two numbers in the
        arm-aligned frame AFTER base rotation has aligned the arm
        with the object:
          - reach: how far the grasp center extends along the arm
                   extension direction (-Y in base_link before rotation,
                   i.e. outward from the robot)
          - drop:  how far the grasp center hangs below the lift height

        At the default home pose (yaw=0, pitch=0, roll=0) the gripper
        points straight down (-Z in base_link), so:
          reach ≈ 0.083m (wrist yaw origin offset)
          drop  ≈ 0.031 + 0.031 + 0.024 + 0.021 + 0.230 ≈ 0.337m
        But empirical observation shows the Stretch gripper in home pose
        has a smaller effective drop (~0.20-0.25m) because the joints
        have non-trivial rpy that fold the chain differently.

        We use a simplified planar model:
          - The gripper has a fixed 'stalk' of length L from the wrist
            pitch axis to the grasp center.
          - wrist_pitch rotates this stalk: pitch=0 means pointing down,
            positive pitch tilts the stalk forward (toward -Y / outward).
          - wrist_yaw rotates the horizontal projection: yaw=0 means
            the stalk swings in the arm plane.

        Returns (reach_m, drop_m, lateral_m).
        """
        import math

        # Distances from URDF (meters)
        WRIST_YAW_OFFSET_FORWARD = 0.083   # arm end to wrist_yaw along arm (-Y)
        WRIST_YAW_OFFSET_RIGHT = 0.031     # lateral offset

        # Combined stalk length from wrist_pitch axis to grasp center
        # (pitch->roll->gripper_body->grasp_center, summing along the
        # direction that becomes "down" or "forward" depending on pitch)
        # Keep analytical fallback consistent with grasp reverse-geometry.
        STALK_LENGTH = GRASP_STALK_LENGTH_M
        PITCH_AXIS_DROP = 0.031  # drop from wrist_yaw to wrist_pitch axis

        # At pitch=0 the stalk points downward (−Z).
        # On Stretch, negative wrist_pitch tilts forward/outward.
        stalk_forward = -STALK_LENGTH * math.sin(wrist_pitch)  # horizontal (outward)
        stalk_down = STALK_LENGTH * math.cos(wrist_pitch)     # vertical (downward)

        # Wrist yaw rotates the horizontal component.
        # yaw=0 means directly along the arm axis.
        # The forward component projects onto the arm direction by cos(yaw).
        reach_from_stalk = stalk_forward * math.cos(wrist_yaw)

        # Lateral: perpendicular to arm axis in horizontal plane.
        # On Stretch, negative yaw swings the gripper tip toward +X (left
        # of arm when facing outward). This is opposite to the standard
        # sin(yaw) direction, so we negate it.
        lateral_from_stalk = stalk_forward * math.sin(wrist_yaw)
        lateral = WRIST_YAW_OFFSET_RIGHT - lateral_from_stalk

        # Total horizontal reach along arm direction
        reach = WRIST_YAW_OFFSET_FORWARD + reach_from_stalk

        # Total vertical drop
        drop = PITCH_AXIS_DROP + stalk_down

        return reach, drop, lateral

    def _lookup_gripper_offset_from_arm(self):
        """Look up the actual 3D offset from arm end (link_arm_l0) to
        gripper tip (link_grasp_center) using the live TF tree.

        link_arm_l0 is the outermost telescoping link — it moves with arm
        extension.  So the returned offset is purely the mechanical gripper
        displacement and does NOT include arm extension.

        Returns dict with:
            reach   – distance along arm axis (-Y in base_link), positive = outward
            lateral – offset perpendicular to arm axis in horizontal plane,
                      positive = toward +X (robot forward)
            drop    – distance below arm end, positive = below
        Or None if TF lookup fails.
        """
        # Try link_grasp_center first, then fall back to gripper body
        grasp_frame = None
        for candidate in ('link_grasp_center', 'link_gripper_s3_body'):
            try:
                self.ros_node.tf_buffer.lookup_transform(
                    'base_link', candidate,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
                grasp_frame = candidate
                break
            except Exception:
                continue

        if grasp_frame is None:
            print("  WARNING: Could not find gripper TF frame "
                  "(tried link_grasp_center, link_gripper_s3_body)")
            return None

        try:
            gc_tf = self.ros_node.tf_buffer.lookup_transform(
                'base_link', grasp_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
            arm_tf = self.ros_node.tf_buffer.lookup_transform(
                'base_link', 'link_arm_l0',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))

            dx = gc_tf.transform.translation.x - arm_tf.transform.translation.x
            dy = gc_tf.transform.translation.y - arm_tf.transform.translation.y
            dz = gc_tf.transform.translation.z - arm_tf.transform.translation.z

            # Arm extends along -Y in base_link.
            # dy is negative when gripper extends outward (-Y).
            reach = -dy       # positive = extending outward along arm axis
            lateral = dx      # positive = toward robot front (+X)
            drop = -dz        # positive = below arm level

            print(f"  TF gripper offset ({grasp_frame}): "
                  f"reach={reach*100:.1f}cm, lateral={lateral*100:.1f}cm, "
                  f"drop={drop*100:.1f}cm")
            print(f"    (raw dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f})")

            return {'reach': reach, 'lateral': lateral, 'drop': drop}

        except Exception as e:
            print(f"  WARNING: TF gripper lookup failed: {e}")
            return None

    def _compute_object_top_z(self, mask, padding_px=40):
        """Compute the maximum Z in base_link of the object and nearby region.

        Scans the SAM mask *and* a padded bounding-box around it to detect
        nearby tall obstacles the gripper must clear during approach.

        Uses vectorised numpy + a single TF lookup for speed.

        Returns the maximum Z coordinate (metres, in base_link), or None.
        """
        if self.depth_image is None or mask is None:
            return None
        if self.ros_node.camera_info is None:
            return None

        # Camera intrinsics
        fx = self.ros_node.camera_info.k[0]
        fy = self.ros_node.camera_info.k[4]
        cx_cam = self.ros_node.camera_info.k[2]
        cy_cam = self.ros_node.camera_info.k[5]
        H_orig = self.ros_node.camera_info.height

        # Camera → base_link transform (look up once)
        try:
            tf_msg = self.ros_node.tf_buffer.lookup_transform(
                'base_link', 'camera_color_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            print(f"  WARNING: Cannot compute object height: {e}")
            return None

        q = tf_msg.transform.rotation
        R = np.array([
            [1 - 2*(q.y**2 + q.z**2), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
            [2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x**2 + q.z**2), 2*(q.y*q.z - q.x*q.w)],
            [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x**2 + q.y**2)]
        ])
        t_vec = np.array([tf_msg.transform.translation.x,
                          tf_msg.transform.translation.y,
                          tf_msg.transform.translation.z])

        # Padded bounding box of the mask
        mask_ys, mask_xs = np.where(mask > 0)
        if len(mask_ys) == 0:
            return None

        h, w = self.depth_image.shape
        y_min = max(0, int(mask_ys.min()) - padding_px)
        y_max = min(h - 1, int(mask_ys.max()) + padding_px)
        x_min = max(0, int(mask_xs.min()) - padding_px)
        x_max = min(w - 1, int(mask_xs.max()) + padding_px)

        # Sample a grid inside the padded box (~20 steps per axis)
        stride = max(1, max(y_max - y_min, x_max - x_min) // 20)
        ys = np.arange(y_min, y_max + 1, stride)
        xs = np.arange(x_min, x_max + 1, stride)
        grid_ys, grid_xs = np.meshgrid(ys, xs, indexing='ij')
        grid_ys = grid_ys.ravel()
        grid_xs = grid_xs.ravel()

        # Depths at sampled points
        depths = self.depth_image[grid_ys, grid_xs].astype(np.float64)
        valid = (depths > 0.1) & (depths < 5.0)
        if not np.any(valid):
            return None

        grid_ys = grid_ys[valid]
        grid_xs = grid_xs[valid]
        depths = depths[valid]

        rotated_head = bool(getattr(self.ros_node, "head_image_rotated_90_cw", lambda: STRETCH_AI_ROTATE_HEAD_90_CW)())
        # Map display pixels back to camera pixels.
        if rotated_head:
            # Display is 90deg CW-rotated relative to camera frame.
            orig_col = grid_ys.astype(np.float64)
            orig_row = (H_orig - 1 - grid_xs).astype(np.float64)
        else:
            # Display matches camera frame orientation.
            orig_col = grid_xs.astype(np.float64)
            orig_row = grid_ys.astype(np.float64)

        # Pinhole → 3D in camera optical frame (vectorised)
        cam_x = (orig_col - cx_cam) * depths / fx
        cam_y = (orig_row - cy_cam) * depths / fy
        cam_z = depths

        # Transform all points to base_link in one shot
        cam_pts = np.stack([cam_x, cam_y, cam_z], axis=1)  # (N, 3)
        base_pts = (R @ cam_pts.T).T + t_vec                # (N, 3)

        max_z = float(base_pts[:, 2].max())
        print(f"  Object/region max height: {max_z:.3f}m "
              f"(scanned {len(depths)} points in {y_max-y_min}x{x_max-x_min}px region)")

        return max_z

    def _compute_grasp_orientation(self, mask, px, py):
        """Compute robust grasp orientation from a 3D top-surface rectangle.

        Pipeline:
          1) Sample mask pixels with valid depth and project to base_link.
          2) Keep only top-surface points (near max Z) to avoid tall side walls.
          3) Fit an oriented rectangle in XY using PCA extents.
          4) Force all rectangle corners onto one plane (mean top-surface Z).

        This gives a stable rectangle even for irregular tops and perspective.
        Returns (grasp_angle_rad, rect_info_dict) or (0.0, None) on failure.
        """
        import math

        if self.depth_image is None or self.ros_node.camera_info is None or mask is None:
            return 0.0, None

        mask_ys, mask_xs = np.where(mask > 0)
        if len(mask_ys) < 50:
            return 0.0, None

        # Subsample for speed while keeping coverage
        max_samples = 3000
        if len(mask_ys) > max_samples:
            step = int(np.ceil(len(mask_ys) / max_samples))
            mask_ys = mask_ys[::step]
            mask_xs = mask_xs[::step]

        # Camera intrinsics
        fx = self.ros_node.camera_info.k[0]
        fy = self.ros_node.camera_info.k[4]
        cx_cam = self.ros_node.camera_info.k[2]
        cy_cam = self.ros_node.camera_info.k[5]
        H_orig = self.ros_node.camera_info.height

        # Camera -> base transform (single lookup)
        try:
            tf_msg = self.ros_node.tf_buffer.lookup_transform(
                'base_link', 'camera_color_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            print(f"  WARNING: Cannot get camera→base TF for grasp orientation: {e}")
            return 0.0, None

        q = tf_msg.transform.rotation
        R = np.array([
            [1 - 2*(q.y**2 + q.z**2), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
            [2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x**2 + q.z**2), 2*(q.y*q.z - q.x*q.w)],
            [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x**2 + q.y**2)]
        ])
        t_vec = np.array([tf_msg.transform.translation.x,
                          tf_msg.transform.translation.y,
                          tf_msg.transform.translation.z])

        # Depth samples on mask
        depths = self.depth_image[mask_ys, mask_xs].astype(np.float64)
        valid = (depths > 0.1) & (depths < 5.0)
        if not np.any(valid):
            return 0.0, None
        mask_ys = mask_ys[valid]
        mask_xs = mask_xs[valid]
        depths = depths[valid]

        rotated_head = bool(getattr(self.ros_node, "head_image_rotated_90_cw", lambda: STRETCH_AI_ROTATE_HEAD_90_CW)())
        # Map display pixels back to camera image coordinates.
        if rotated_head:
            orig_col = mask_ys.astype(np.float64)
            orig_row = (H_orig - 1 - mask_xs).astype(np.float64)
        else:
            orig_col = mask_xs.astype(np.float64)
            orig_row = mask_ys.astype(np.float64)

        # Pinhole -> camera frame -> base_link (vectorized)
        cam_x = (orig_col - cx_cam) * depths / fx
        cam_y = (orig_row - cy_cam) * depths / fy
        cam_z = depths
        cam_pts = np.stack([cam_x, cam_y, cam_z], axis=1)
        base_pts = (R @ cam_pts.T).T + t_vec

        if len(base_pts) < 30:
            return 0.0, None

        # Top-surface extraction (avoid sides of tall objects)
        z_vals = base_pts[:, 2]
        z_max = float(z_vals.max())
        z_min = float(z_vals.min())
        z_span = max(0.0, z_max - z_min)
        top_band = min(0.05, max(0.015, 0.20 * z_span))  # 1.5cm .. 5cm
        top_sel = z_vals >= (z_max - top_band)
        top_pts = base_pts[top_sel]
        if len(top_pts) < 20:
            # Fallback: use top quartile by Z
            z_q = np.quantile(z_vals, 0.75)
            top_pts = base_pts[z_vals >= z_q]
        if len(top_pts) < 10:
            print("  WARNING: Not enough top-surface points for rectangle fit")
            return 0.0, None

        xy = top_pts[:, :2]
        mean_xy = xy.mean(axis=0)
        centered = xy - mean_xy

        # PCA in XY for robust oriented rectangle
        cov = (centered.T @ centered) / max(1, len(centered) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        order = np.argsort(eigvals)[::-1]
        u = eigvecs[:, order[0]]  # long-axis direction
        v = eigvecs[:, order[1]]  # narrow-axis direction

        proj_u = centered @ u
        proj_v = centered @ v
        u_min, u_max = float(proj_u.min()), float(proj_u.max())
        v_min, v_max = float(proj_v.min()), float(proj_v.max())
        long_len = u_max - u_min
        narrow_len = v_max - v_min

        if long_len < 1e-6:
            print("  WARNING: Long-axis projection too small for stable orientation")
            return 0.0, None

        # Ensure robust rectangle: if degenerate/triangle-like, thicken narrow side.
        min_aspect = 0.30
        if narrow_len < min_aspect * long_len:
            v_mid = 0.5 * (v_min + v_max)
            narrow_len = min_aspect * long_len
            v_min = v_mid - 0.5 * narrow_len
            v_max = v_mid + 0.5 * narrow_len
            print(f"  NOTE: Expanded narrow side to avoid degenerate rectangle "
                  f"(aspect<{min_aspect:.2f})")

        # Rectangle corners in XY (always a proper rectangle)
        corners_xy = np.array([
            mean_xy + u * u_max + v * v_max,
            mean_xy + u * u_max + v * v_min,
            mean_xy + u * u_min + v * v_min,
            mean_xy + u * u_min + v * v_max,
        ], dtype=np.float64)

        # Put all corners on one plane (mean top-surface height)
        z_plane = float(top_pts[:, 2].mean())
        corners_3d = np.column_stack([corners_xy, np.full(4, z_plane)])
        centroid_3d = np.array([mean_xy[0], mean_xy[1], z_plane], dtype=np.float64)
        centroid_xy = centroid_3d[:2]

        # Angle of long axis in base_link XY
        long_axis_angle = math.atan2(u[1], u[0])

        # Grasp yaw relative to arm direction (-Y)
        grasp_yaw = long_axis_angle + math.pi / 2.0
        while grasp_yaw > math.pi / 2.0:
            grasp_yaw -= math.pi
        while grasp_yaw < -math.pi / 2.0:
            grasp_yaw += math.pi

        # Axis endpoints for debug overlay
        long_dir = u / (np.linalg.norm(u) + 1e-9)
        narrow_dir = v / (np.linalg.norm(v) + 1e-9)
        long_end1 = np.array([centroid_xy[0] + long_dir[0] * long_len * 0.5,
                              centroid_xy[1] + long_dir[1] * long_len * 0.5,
                              z_plane], dtype=np.float64)
        long_end2 = np.array([centroid_xy[0] - long_dir[0] * long_len * 0.5,
                              centroid_xy[1] - long_dir[1] * long_len * 0.5,
                              z_plane], dtype=np.float64)
        narrow_end1 = np.array([centroid_xy[0] + narrow_dir[0] * narrow_len * 0.5,
                                centroid_xy[1] + narrow_dir[1] * narrow_len * 0.5,
                                z_plane], dtype=np.float64)
        narrow_end2 = np.array([centroid_xy[0] - narrow_dir[0] * narrow_len * 0.5,
                                centroid_xy[1] - narrow_dir[1] * narrow_len * 0.5,
                                z_plane], dtype=np.float64)

        # Reverse transform for overlay projection
        R_inv = R.T
        t_inv = -R_inv @ t_vec

        def base_to_image_pixel(pt_base):
            cam_pt = R_inv @ pt_base + t_inv
            if cam_pt[2] <= 0:
                return None
            orig_col = cam_pt[0] * fx / cam_pt[2] + cx_cam
            orig_row = cam_pt[1] * fy / cam_pt[2] + cy_cam
            if rotated_head:
                rot_px = int(H_orig - 1 - orig_row)
                rot_py = int(orig_col)
            else:
                rot_px = int(orig_col)
                rot_py = int(orig_row)
            return (rot_px, rot_py)

        axis_pixels = {}
        for name, pt in [('center', centroid_3d),
                         ('long1', long_end1), ('long2', long_end2),
                         ('narrow1', narrow_end1), ('narrow2', narrow_end2)]:
            pix = base_to_image_pixel(pt)
            if pix is not None:
                axis_pixels[name] = pix

        corner_pixels = []
        for corner in corners_3d:
            pix = base_to_image_pixel(corner)
            if pix is not None:
                corner_pixels.append((int(pix[0]), int(pix[1])))
        axis_pixels['corners'] = corner_pixels

        rect_info = {
            'center': (float(centroid_xy[0]), float(centroid_xy[1])),
            'size': (float(narrow_len), float(long_len)),  # (narrow, long) in metres
            'metric': True,
            'long_axis_angle_deg': math.degrees(long_axis_angle),
            'grasp_angle_deg': math.degrees(grasp_yaw),
            'axis_pixels': axis_pixels,
            'plane_z': z_plane,
            'top_z_max': z_max,
        }

        print("  3D grasp orientation (top-surface planar rectangle):")
        print(f"    Top points: {len(top_pts)}/{len(base_pts)} (z_max={z_max:.3f}, band={top_band:.3f})")
        print(f"    Long side: {math.degrees(long_axis_angle):.1f}° from +X, length={long_len*100:.1f}cm")
        print(f"    Narrow side: length={narrow_len*100:.1f}cm, plane_z={z_plane:.3f}m")
        print(f"    Grasp yaw: {math.degrees(grasp_yaw):.1f}° (wrist_yaw={grasp_yaw:.3f} rad)")

        return grasp_yaw, rect_info

    def _estimate_gripper_width(self, mask, px, py, depth, rect_info=None):
        """Estimate how wide the gripper should open based on object width at grasp point.

        Uses the SAM mask's narrow dimension (from minAreaRect) and the depth
        to convert from pixels to meters.

        Returns gripper aperture in meters, clamped to joint limits.
        """
        if self.ros_node.camera_info is None:
            return 0.5  # fallback: fully open

        # Get focal length for pixel-to-meter conversion
        fx = self.ros_node.camera_info.k[0]

        # Use rect narrow dimension if available
        if rect_info is not None and rect_info.get('metric'):
            # 3D PCA rect_info — size is already in metres
            object_width_m = rect_info['size'][0]
        elif rect_info is not None:
            # Legacy 2D rect_info — narrow dimension in pixels
            narrow_px = rect_info['size'][0]
            object_width_m = narrow_px * depth / fx
        else:
            # Fallback: measure mask width at the grasp row
            row = min(py, mask.shape[0] - 1)
            row_pixels = np.where(mask[row] > 0)[0]
            if len(row_pixels) > 0:
                narrow_px = row_pixels[-1] - row_pixels[0]
                object_width_m = narrow_px * depth / fx
            else:
                return 0.5  # fallback

        # Add a small margin (1cm each side) for clearance
        gripper_aperture = object_width_m + 0.02

        # Clamp to gripper joint limits
        grip_min = self.ros_node.JOINT_LIMITS[7][0]  # -0.1
        grip_max = self.ros_node.JOINT_LIMITS[7][1]  # 0.5501

        # The gripper joint value maps roughly to finger separation.
        # Stretch gripper: 0.0 = closed, ~0.55 = fully open (~12cm opening)
        # Map physical width to joint value: joint ≈ width_m / 0.22
        # (0.55 joint ≈ 0.12m opening, so 1 joint unit ≈ 0.22m)
        gripper_joint = max(grip_min, min(grip_max, gripper_aperture / 0.22))

        print(f"  Gripper width: object={object_width_m*100:.1f}cm, "
              f"aperture={gripper_aperture*100:.1f}cm, joint={gripper_joint:.3f}")

        return gripper_joint

    def _current_manip_joint6(self) -> list[float]:
        """Return current manipulation-space joint vector for arm_to().

        ordering: [base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll]
        """
        sensors = self.ros_node.get_sensor_snapshot()
        # Use bridge-tracked manipulation base_x first (from stretch_ai get_six_joints()).
        # Generic joint_state[0] ordering can differ and caused base_x restore drift.
        base_x = None
        try:
            with self.ros_node._lock:
                bx = float(self.ros_node._manip_base_x)
                if math.isfinite(bx):
                    base_x = bx
        except Exception:
            base_x = None
        if base_x is None:
            jp = sensors.get("observation.joint_state.position", [])
            if isinstance(jp, list) and len(jp) > 0:
                try:
                    bx = float(jp[0])
                    if math.isfinite(bx):
                        base_x = bx
                except (TypeError, ValueError):
                    base_x = None
        if base_x is None:
            base_x = 0.0
        q_actual = self.ros_node.get_actual_qpos()
        q = list(q_actual[:8]) + [0.0] * max(0, 8 - len(q_actual))
        return [
            base_x,
            float(q[1]),  # lift
            float(q[0]),  # arm extension
            float(q[2]),  # wrist_yaw
            float(q[3]),  # wrist_pitch
            float(q[4]),  # wrist_roll
        ]

    def _freeze_streaming_commands_to_current_state(self) -> None:
        """Avoid command-loop interference while running blocking scripted motions."""
        try:
            with self.ros_node._lock:
                cur = self.ros_node.actual_qpos
                if isinstance(cur, list) and len(cur) >= 10:
                    self.ros_node.qpos = list(cur[:10])
                    self.ros_node.published_qpos = list(cur[:10])
                    self.ros_node.qpos[8] = 0.0
                    self.ros_node.qpos[9] = 0.0
                    self.ros_node.published_qpos[8] = 0.0
                    self.ros_node.published_qpos[9] = 0.0
                    self.ros_node._base_linear_cmd = 0.0
                    self.ros_node._base_angular_cmd = 0.0
                    self.ros_node._needs_mode_retry = False
        except Exception:
            pass

    def _execute_approach_with_stretch_ai_ik(
        self,
        point_base,
        *,
        mode: str,
        grasp_yaw: float | None = None,
        gripper_width: float | None = None,
        preserve_existing_pre_action_state: bool = False,
    ) -> None:
        """v6 approach: use stretch_ai IK/open-loop planning instead of manual geometry."""
        import time

        def _wait_with_pause(duration_s: float) -> bool:
            duration_s = min(float(SCRIPT_STAGE_WAIT_CAP_S), max(0.0, float(duration_s)))
            end_t = time.time() + duration_s
            while time.time() < end_t:
                if self._is_abort_requested():
                    return False
                with self._action_lock:
                    st = self._action_state
                if st in ("paused", "awaiting_confirm"):
                    time.sleep(0.05)
                    continue
                time.sleep(min(0.05, max(0.0, end_t - time.time())))
            return not self._is_abort_requested()

        def _wait_until_running() -> bool:
            while True:
                if self._is_abort_requested():
                    return False
                with self._action_lock:
                    st = self._action_state
                if st == "running":
                    return True
                time.sleep(0.05)

        def _abort_and_return() -> None:
            if self._consume_skip_to_next_goal_request():
                print("Action aborted by user. Skipping to next queued goal...")
                self._set_action_state("idle")
                return
            print("Action aborted by user. Returning to start...")
            self._set_action_state("idle")
            if self._pre_action_state is not None:
                self.return_to_start()

        # Save current state so return_to_start() can restore it.
        if (not preserve_existing_pre_action_state) or (self._pre_action_state is None):
            manip0 = self._current_manip_joint6()
            base_x0 = float(manip0[0]) if isinstance(manip0, list) and len(manip0) >= 1 else 0.0
            wrist_roll0 = float(manip0[5]) if isinstance(manip0, list) and len(manip0) >= 6 else float(self.ros_node.qpos[4])
            base_pose0 = self.ros_node.get_measured_base_pose_xytheta()
            if not (isinstance(base_pose0, list) and len(base_pose0) >= 3):
                base_pose0 = [0.0, 0.0, 0.0]
            self._pre_action_state = {
                "arm_ext": self.ros_node.qpos[0],
                "lift": self.ros_node.qpos[1],
                "wrist_yaw": self.ros_node.qpos[2],
                "wrist_pitch": self.ros_node.qpos[3],
                "wrist_roll": wrist_roll0,
                "base_x": base_x0,
                "base_pose_xytheta_start": [float(base_pose0[0]), float(base_pose0[1]), float(base_pose0[2])],
                "gripper": self.ros_node.qpos[7],
                "head_pan": self.ros_node.qpos[5],
                "head_tilt": self.ros_node.qpos[6],
                "rotation_applied": 0.0,
            }
            yaw0 = self._get_current_base_yaw()
            if yaw0 is not None:
                self._pre_action_state["base_yaw_start"] = float(yaw0)

        target_base_xyz = (
            float(point_base.point.x),
            float(point_base.point.y),
            float(point_base.point.z),
        )
        base_pose = self.ros_node.get_measured_base_pose_xytheta()
        if not (isinstance(base_pose, list) and len(base_pose) >= 3):
            base_pose = [0.0, 0.0, 0.0]
        target_world_xyz = self._base_point_to_odom_xyz(target_base_xyz, base_pose)

        print(f"[IK pipeline] mode={mode} target_base={target_base_xyz} target_world={target_world_xyz}")
        self._freeze_streaming_commands_to_current_state()

        # Safety pre-step: ensure minimum lift before any IK planning/execution.
        # This is intentionally done before plan_open_loop_grasp().
        min_lift_m = 0.90
        precheck_joint = self._current_manip_joint6()
        if isinstance(precheck_joint, list) and len(precheck_joint) >= 6:
            current_lift = float(precheck_joint[1])
            if current_lift < float(min_lift_m):
                self._set_status(
                    f"IK pipeline: raising lift to {min_lift_m:.2f}m before planning...",
                    "QLabel { color: blue; font-size: 10px; }",
                )
                precheck_joint = [float(v) for v in precheck_joint[:6]]
                precheck_joint[1] = float(
                    np.clip(
                        float(min_lift_m),
                        float(self.ros_node.JOINT_LIMITS[1][0]),
                        float(self.ros_node.JOINT_LIMITS[1][1]),
                    )
                )
                if not self.ros_node.execute_arm_to(
                    precheck_joint[:6],
                    gripper=None,
                    blocking=True,
                    timeout_s=8.0,
                    reliable=False,
                ):
                    self._set_status("IK pipeline: pre-lift to 0.90m failed", "QLabel { color: red; font-size: 10px; }")
                    _abort_and_return()
                    return
                if not _wait_with_pause(0.2):
                    _abort_and_return()
                    return

        # Step 1: plan IK targets directly from the clicked world-frame point.
        # No manual pre-rotation/pre-lift: execute exactly what IK returns.
        self._set_status("IK pipeline: planning pregrasp/grasp targets...", "QLabel { color: blue; font-size: 10px; }")
        plan = self.ros_node.plan_open_loop_grasp(
            target_world_xyz,
            pregrasp_distance=IK_PREGRASP_DISTANCE_M if mode == "grasp" else 0.20,
            lift_distance=IK_LIFT_DISTANCE_M,
            wrist_yaw_target=(None if grasp_yaw is None else float(grasp_yaw)),
            wrist_pitch_target=float(np.deg2rad(GRASP_PITCH_DEG)),
            timeout_s=35.0,
        )
        if isinstance(plan, dict):
            err0 = str(plan.get("error", ""))
            if (not plan.get("ok", False)) and ("Timeout waiting for worker response to 'plan_open_loop_grasp'" in err0):
                self._set_status("IK pipeline: planner timeout; retrying...", "QLabel { color: orange; font-size: 10px; }")
                plan = self.ros_node.plan_open_loop_grasp(
                    target_world_xyz,
                    pregrasp_distance=IK_PREGRASP_DISTANCE_M if mode == "grasp" else 0.20,
                    lift_distance=IK_LIFT_DISTANCE_M,
                    wrist_yaw_target=(None if grasp_yaw is None else float(grasp_yaw)),
                    wrist_pitch_target=float(np.deg2rad(GRASP_PITCH_DEG)),
                    timeout_s=70.0,
                )
        if not isinstance(plan, dict) or not plan.get("ok", False):
            err = "IK planning failed" if not isinstance(plan, dict) else str(plan.get("error", "IK planning failed"))
            self._set_status(f"IK pipeline: {err}", "QLabel { color: red; font-size: 10px; }")
            _abort_and_return()
            return

        grasp_joint = plan.get("grasp_joint")
        if not (isinstance(grasp_joint, list) and len(grasp_joint) >= 6):
            self._set_status("IK pipeline: invalid grasp target", "QLabel { color: red; font-size: 10px; }")
            _abort_and_return()
            return
        # No separate pregrasp joint from planner.
        # Move to final grasp geometry immediately, but keep lift +10cm above target.
        target_lift = float(grasp_joint[1])
        target_lift_plus_margin = float(
            np.clip(
                float(target_lift) + 0.10,
                float(self.ros_node.JOINT_LIMITS[1][0]),
                float(self.ros_node.JOINT_LIMITS[1][1]),
            )
        )
        approach_joint = [float(v) for v in grasp_joint[:6]]
        approach_joint[0] = float(
            np.clip(
                float(approach_joint[0]) + float(IK_GRASP_BASE_X_OFFSET_M),
                float(MANIP_BASE_X_LIMITS[0]),
                float(MANIP_BASE_X_LIMITS[1]),
            )
        )
        # Reach-specific two-stage lift behavior:
        # 1) approach with lift held at 0.9m while base_x/arm settle
        # 2) then lower to (target_lift + 0.1m)
        # If (target_lift + 0.1m) is already >= 0.9m, skip stage (1).
        reach_lift_stage1 = float(
            np.clip(
                0.90,
                float(self.ros_node.JOINT_LIMITS[1][0]),
                float(self.ros_node.JOINT_LIMITS[1][1]),
            )
        )
        if mode == "reach" and target_lift_plus_margin < reach_lift_stage1:
            approach_joint[1] = float(reach_lift_stage1)
        else:
            approach_joint[1] = float(target_lift_plus_margin)

        # Step 2: move to grasp pose with lift hold and wait for user verification.
        self._set_status("IK pipeline: moving to grasp pose (+10cm lift hold)...", "QLabel { color: blue; font-size: 10px; }")
        if mode == "grasp":
            if isinstance(gripper_width, (int, float)) and np.isfinite(float(gripper_width)):
                gripper_open = float(
                    np.clip(
                        float(gripper_width),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
            else:
                gripper_open = float(self.ros_node.JOINT_LIMITS[7][1])
        else:
            gripper_open = None

        if not self.ros_node.execute_arm_to(
            approach_joint[:6],
            gripper=gripper_open,
            blocking=True,
            timeout_s=8.0,
            reliable=False,
        ):
            self._set_status("IK pipeline: approach move failed", "QLabel { color: red; font-size: 10px; }")
            _abort_and_return()
            return
        if not _wait_with_pause(0.3):
            _abort_and_return()
            return

        if mode == "reach":
            # If we approached at 0.9m, lower to target_lift+0.1m only after
            # base_x/arm have reached the approach target.
            if target_lift_plus_margin < reach_lift_stage1:
                self._set_status(
                    "Reach: lowering lift from 0.90m to target+0.10m...",
                    "QLabel { color: blue; font-size: 10px; }",
                )
                reach_lower_joint = [float(v) for v in approach_joint[:6]]
                reach_lower_joint[1] = float(target_lift_plus_margin)
                if not self.ros_node.execute_arm_to(
                    reach_lower_joint[:6],
                    gripper=None,
                    blocking=True,
                    timeout_s=8.0,
                    reliable=False,
                ):
                    self._set_status("IK pipeline: reach lift lower failed", "QLabel { color: red; font-size: 10px; }")
                    _abort_and_return()
                    return
                if not _wait_with_pause(0.3):
                    _abort_and_return()
                    return
            # Ensure subsequent manual controls (e.g., gripper buttons) start from
            # the measured post-reach pose and do not replay stale command targets.
            self.ros_node.sync_command_targets_to_actual()
            self._set_return_enabled(True)
            self._set_status("Reach completed (IK approach).", "QLabel { color: green; font-size: 10px; }")
            return

        # Sync command targets so manual tweaks during pause start from live robot state.
        self.ros_node.sync_command_targets_to_actual()
        self._set_status("At approach hold (+10cm). Press Continue to lower lift and grasp.", "QLabel { color: orange; font-size: 10px; }")
        self._set_action_state("awaiting_confirm")
        if not _wait_until_running():
            _abort_and_return()
            return

        # Step 3: Continue lowers only lift to target, keeping the rest as currently reached.
        self._set_status("IK pipeline: lowering lift to target grasp...", "QLabel { color: blue; font-size: 10px; }")
        lower_joint = self._current_manip_joint6()
        if not (isinstance(lower_joint, list) and len(lower_joint) >= 6):
            lower_joint = [float(v) for v in approach_joint[:6]]
        lower_joint = [float(v) for v in lower_joint[:6]]
        lower_joint[1] = float(
            np.clip(
                float(target_lift),
                float(self.ros_node.JOINT_LIMITS[1][0]),
                float(self.ros_node.JOINT_LIMITS[1][1]),
            )
        )
        if not self.ros_node.execute_arm_to(lower_joint[:6], gripper=gripper_open, blocking=True, timeout_s=8.0, reliable=False):
            self._set_status("IK pipeline: lift lower failed", "QLabel { color: red; font-size: 10px; }")
            _abort_and_return()
            return
        if not _wait_with_pause(0.3):
            _abort_and_return()
            return

        # Partial close to avoid hard motor overload:
        # close around object width by reducing from open target by 2-5cm equivalent.
        gripper_delta_joint = float(IK_GRIPPER_CLOSE_DELTA_M) / 0.22
        gripper_closed = float(
            np.clip(
                float(gripper_open) - float(gripper_delta_joint),
                float(IK_GRIPPER_CLOSE_MIN_JOINT),
                float(self.ros_node.JOINT_LIMITS[7][1]),
            )
        )
        gripper_closed = float(min(float(gripper_open), float(gripper_closed)))
        print(
            f"[IK gripper] open={gripper_open:+.3f}, close={gripper_closed:+.3f}, "
            f"delta_m={IK_GRIPPER_CLOSE_DELTA_M:.3f}"
        )
        if not self.ros_node.execute_arm_to(lower_joint[:6], gripper=gripper_closed, blocking=True, timeout_s=6.0, reliable=False):
            self._set_status("IK pipeline: gripper close failed", "QLabel { color: red; font-size: 10px; }")
            _abort_and_return()
            return
        if not _wait_with_pause(0.3):
            _abort_and_return()
            return

        # Pause after grasp close for user verification before lift.
        self._set_return_enabled(True)
        self.ros_node.sync_command_targets_to_actual()
        self._set_status("Grasp closed. Verify hold, then press Continue to lift.", "QLabel { color: orange; font-size: 10px; }")
        self._set_action_state("awaiting_post_grasp")
        if not _wait_until_running():
            _abort_and_return()
            return

        # Lift after grasp from *current* pose only (do not reuse planner base_x).
        lift_after_joint = self._current_manip_joint6()
        if not (isinstance(lift_after_joint, list) and len(lift_after_joint) >= 6):
            lift_after_joint = [float(v) for v in lower_joint[:6]]
        lift_after_joint = [float(v) for v in lift_after_joint[:6]]
        lift_after_joint[1] = float(
            np.clip(
                float(lift_after_joint[1]) + float(IK_LIFT_DISTANCE_M),
                float(self.ros_node.JOINT_LIMITS[1][0]),
                float(self.ros_node.JOINT_LIMITS[1][1]),
            )
        )
        gripper_lift = self._get_manual_gripper_target(fallback=gripper_closed)
        if gripper_lift is None:
            gripper_lift = gripper_closed
        self._set_manual_gripper_override(gripper_lift)
        self.ros_node.execute_arm_to(
            lift_after_joint[:6],
            gripper=gripper_lift,
            blocking=True,
            timeout_s=6.0,
            reliable=False,
        )
        _wait_with_pause(0.3)

        self._set_return_enabled(True)
        self._set_action_state("idle")
        self._set_status("Grasp completed and lifted. Use Return when ready.", "QLabel { color: green; font-size: 10px; }")

    def _execute_approach(self, point_base, mode='reach', height_clearance=REACH_HEIGHT_CLEARANCE,
                          grasp_yaw=None, gripper_width=None, object_top_z=None,
                          grasp_mask=None, long_axis_angle=None,
                          preserve_existing_pre_action_state=False):
        """Execute reach/grasp via the v6 IK pipeline.

        The legacy manual geometry path was removed to keep v6 deterministic and
        maintainable. All scripted approach motions should use stretch_ai IK RPCs.
        """
        # Keep signature compatibility with existing v5-style callers.
        _ = (height_clearance, object_top_z, grasp_mask, long_axis_angle)

        if not USE_STRETCH_AI_IK_GRASP_PIPELINE:
            raise RuntimeError(
                "Manual approach pipeline has been removed from v6. "
                "Enable USE_STRETCH_AI_IK_GRASP_PIPELINE."
            )

        return self._execute_approach_with_stretch_ai_ik(
            point_base,
            mode=str(mode),
            grasp_yaw=(None if grasp_yaw is None else float(grasp_yaw)),
            gripper_width=(None if gripper_width is None else float(gripper_width)),
            preserve_existing_pre_action_state=bool(preserve_existing_pre_action_state),
        )

    def reach_object(self):
        """Move arm above selected object (no gripper action)"""
        if not self._begin_action('reach'):
            self._set_status("Another action is already running/paused",
                             "QLabel { color: orange; font-size: 10px; }")
            return

        if self.selected_segment is None or self.head_rgb is None:
            self._set_action_state('idle')
            self._set_status("No object selected", "QLabel { color: red; }")
            return

        cx, cy = self.selected_segment['center']
        point_base, depth = self._get_3d_point_at_pixel(cx, cy)
        if point_base is None:
            self._set_action_state('idle')
            self._set_status("Failed to get 3D position", "QLabel { color: red; }")
            return

        # Compute object top height for collision avoidance
        object_top_z = None
        mask = self.selected_segment.get('mask')
        if mask is not None:
            object_top_z = self._compute_object_top_z(mask)

        print(f"\n{'='*60}")
        print(f"REACH (10cm above object)")
        if object_top_z is not None:
            print(f"  Object top Z: {object_top_z:.3f}m (base_link)")
        print(f"{'='*60}")

        self._set_status("Reaching above object...", "QLabel { color: blue; font-size: 10px; }")

        def run():
            try:
                self._execute_approach(point_base, mode='reach', height_clearance=REACH_HEIGHT_CLEARANCE,
                                       object_top_z=object_top_z)
            except Exception as e:
                print(f"Reach error: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"Reach failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
            finally:
                with self._action_lock:
                    st = self._action_state
                if st == 'running':
                    self._set_action_state('idle')

        from threading import Thread
        Thread(target=run, daemon=True).start()

    def _pixel_from_event(self, event):
        """Convert a mouse event position to image pixel coordinates."""
        if self.head_rgb is None:
            return None, None
        px = int(event.pos().x() * self.head_rgb.shape[1] / self.head_display.width())
        py = int(event.pos().y() * self.head_rgb.shape[0] / self.head_display.height())
        return px, py

    def _center_on_pixel(self, px, py):
        """Center camera on a pixel in the rotated image."""
        if self.head_rgb is None:
            return
        img_height, img_width = self.head_rgb.shape[:2]
        offset_x = (px - img_width // 2) / img_width
        offset_y = (py - img_height // 2) / img_height
        pan_adj = -offset_x * 0.2
        tilt_adj = offset_y * 0.2
        print(f"Centering camera on pixel ({px}, {py}): pan={pan_adj:.3f}, tilt={tilt_adj:.3f}")
        self.robot_controller.adjust_control('head_pan', pan_adj)
        self.robot_controller.adjust_control('head_tilt', tilt_adj)
        self.status_label.setText("Centering camera...")
        self.status_label.setStyleSheet("QLabel { color: blue; }")

    def _find_segment_at_pixel(self, px, py):
        if not self.segments:
            return None
        for seg in self.segments:
            if py < seg['mask'].shape[0] and px < seg['mask'].shape[1] and seg['mask'][py, px] > 0:
                return seg
        return None

    def _point_from_xyz(self, xyz):
        point = PointStamped()
        point.header.frame_id = 'base_link'
        point.header.stamp = self.ros_node.get_clock().now().to_msg()
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        return point

    @staticmethod
    def _base_point_to_odom_xyz(point_xyz, base_pose_xytheta):
        """Convert point from base_link frame into odom/world XY using base pose."""
        px, py, pz = [float(v) for v in point_xyz]
        bx, by, bt = [float(v) for v in base_pose_xytheta]
        c = math.cos(bt)
        s = math.sin(bt)
        ox = bx + c * px - s * py
        oy = by + s * px + c * py
        return (float(ox), float(oy), float(pz))

    @staticmethod
    def _odom_point_to_base_xyz(point_xyz, base_pose_xytheta):
        """Convert point from odom/world XY into current base_link frame."""
        ox, oy, oz = [float(v) for v in point_xyz]
        bx, by, bt = [float(v) for v in base_pose_xytheta]
        dx = ox - bx
        dy = oy - by
        c = math.cos(bt)
        s = math.sin(bt)
        px = c * dx + s * dy
        py = -s * dx + c * dy
        return (float(px), float(py), float(oz))

    def _resolve_goal_point_xyz_for_current_base(self, goal):
        """Resolve queued goal point for current base pose.

        For queued goals, use odom-anchored coordinates when available so that
        sequential goals are relative to current heading/pose (e.g., z-y, not z-x).
        """
        fallback_xyz = goal.get("point_xyz")
        if not (isinstance(fallback_xyz, (list, tuple)) and len(fallback_xyz) >= 3):
            return None
        fallback_xyz = (float(fallback_xyz[0]), float(fallback_xyz[1]), float(fallback_xyz[2]))

        odom_xyz = goal.get("point_odom_xyz")
        if not (isinstance(odom_xyz, (list, tuple)) and len(odom_xyz) >= 3):
            return fallback_xyz

        base_pose = self.ros_node.get_measured_base_pose_xytheta()
        if not (isinstance(base_pose, (list, tuple)) and len(base_pose) >= 3):
            return fallback_xyz

        try:
            return self._odom_point_to_base_xyz(
                (float(odom_xyz[0]), float(odom_xyz[1]), float(odom_xyz[2])),
                (float(base_pose[0]), float(base_pose[1]), float(base_pose[2])),
            )
        except Exception:
            return fallback_xyz

    def _prepare_reach_goal_from_pixel(self, px, py):
        point_base, depth = self._get_3d_point_at_pixel(px, py)
        if point_base is None:
            return None, "No valid depth at clicked point"

        object_top_z = None
        seg = self._find_segment_at_pixel(px, py)
        if seg is not None:
            object_top_z = self._compute_object_top_z(seg['mask'])

        point_xyz = (float(point_base.point.x), float(point_base.point.y), float(point_base.point.z))
        base_pose = self.ros_node.get_measured_base_pose_xytheta()
        point_odom_xyz = None
        if isinstance(base_pose, (list, tuple)) and len(base_pose) >= 3:
            point_odom_xyz = self._base_point_to_odom_xyz(
                point_xyz,
                (float(base_pose[0]), float(base_pose[1]), float(base_pose[2])),
            )

        goal = {
            "kind": "reach",
            "px": int(px),
            "py": int(py),
            "depth": float(depth),
            "point_xyz": point_xyz,
            "point_odom_xyz": point_odom_xyz,
            "object_top_z": None if object_top_z is None else float(object_top_z),
            "created_time": float(time.time()),
        }
        return goal, None

    def _prepare_grasp_goal_from_pixel(self, px, py, segment=None):
        point_base, depth = self._get_3d_point_at_pixel(px, py)
        if point_base is None:
            return None, "No valid depth at clicked point"
        if depth < 0.1 or depth > 2.0:
            return None, f"Object too far or invalid depth: {depth:.2f}m"

        mask = None
        if segment is not None:
            mask = segment['mask']
        else:
            seg = self._find_segment_at_pixel(px, py)
            if seg is not None:
                mask = seg['mask']
        if mask is None and self.selected_segment is not None and 'mask' in self.selected_segment:
            sel_mask = self.selected_segment['mask']
            if py < sel_mask.shape[0] and px < sel_mask.shape[1]:
                mask = sel_mask

        if mask is None:
            return None, "No segment mask at clicked point"

        gripper_width = None
        rect_info = None
        object_top_z = None
        long_axis_angle = None
        grasp_yaw, rect_info = self._compute_grasp_orientation(mask, px, py)
        gripper_width = self._estimate_gripper_width(mask, px, py, depth, rect_info)
        object_top_z = self._compute_object_top_z(mask)
        if rect_info is not None and rect_info.get('top_z_max') is not None:
            top_from_grasp_fit = float(rect_info['top_z_max'])
            object_top_z = top_from_grasp_fit if object_top_z is None else max(float(object_top_z), top_from_grasp_fit)
        if rect_info is not None:
            import math as _m
            long_axis_angle = _m.radians(rect_info['long_axis_angle_deg'])

        if object_top_z is None:
            return None, "Could not estimate object top surface"

        point_xyz = (float(point_base.point.x), float(point_base.point.y), float(point_base.point.z))
        base_pose = self.ros_node.get_measured_base_pose_xytheta()
        point_odom_xyz = None
        if isinstance(base_pose, (list, tuple)) and len(base_pose) >= 3:
            point_odom_xyz = self._base_point_to_odom_xyz(
                point_xyz,
                (float(base_pose[0]), float(base_pose[1]), float(base_pose[2])),
            )

        goal = {
            "kind": "grasp",
            "px": int(px),
            "py": int(py),
            "depth": float(depth),
            "point_xyz": point_xyz,
            "point_odom_xyz": point_odom_xyz,
            "object_top_z": float(object_top_z),
            "grasp_yaw": None if grasp_yaw is None else float(grasp_yaw),
            "gripper_width": None if gripper_width is None else float(gripper_width),
            "long_axis_angle": None if long_axis_angle is None else float(long_axis_angle),
            "grasp_mask": np.array(mask, copy=True),
            "grasp_debug_info": None if rect_info is None else dict(rect_info),
            "created_time": float(time.time()),
        }
        return goal, None

    def _store_queued_goal(self, goal):
        if goal is None:
            return
        self.queued_goals[goal["kind"]] = goal
        self._reset_goal_sequence_progress()
        self._set_status(
            f"Queued {goal['kind']} goal at ({goal['px']}, {goal['py']})",
            "QLabel { color: #1e88e5; font-size: 10px; }"
        )

    def add_reach_goal_at_pixel(self, px, py):
        with self._action_lock:
            if self._action_state != 'idle':
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        goal, err = self._prepare_reach_goal_from_pixel(px, py)
        if err:
            self._set_status(err, "QLabel { color: red; font-size: 10px; }")
            return
        self._store_queued_goal(goal)

    def add_grasp_goal_at_pixel(self, px, py, segment=None):
        with self._action_lock:
            if self._action_state != 'idle':
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        goal, err = self._prepare_grasp_goal_from_pixel(px, py, segment=segment)
        if err:
            self._set_status(err, "QLabel { color: red; font-size: 10px; }")
            return
        self._store_queued_goal(goal)

    def _start_prepared_goal(self, goal, preserve_existing_pre_action_state=False):
        if goal is None:
            return False
        if not self._begin_action(goal["kind"]):
            self._set_status("Another action is already running/paused", "QLabel { color: orange; font-size: 10px; }")
            return False

        resolved_xyz = self._resolve_goal_point_xyz_for_current_base(goal)
        if resolved_xyz is None:
            self._set_status("Invalid queued goal point", "QLabel { color: red; font-size: 10px; }")
            self._set_action_state('idle')
            return False
        point_base = self._point_from_xyz(resolved_xyz)
        kind = goal["kind"]
        if kind == "grasp":
            self._set_status("Executing queued grasp sequence...", "QLabel { color: blue; font-size: 10px; }")
            if isinstance(goal.get("grasp_debug_info"), dict):
                self._grasp_debug_info = dict(goal["grasp_debug_info"])
            else:
                self._grasp_debug_info = None
        else:
            self._set_status("Executing queued reach sequence...", "QLabel { color: blue; font-size: 10px; }")
            self._grasp_debug_info = None

        def run():
            try:
                kwargs = dict(
                    point_base=point_base,
                    mode=kind,
                    object_top_z=goal.get("object_top_z"),
                    preserve_existing_pre_action_state=bool(preserve_existing_pre_action_state),
                )
                if kind == "reach":
                    kwargs["height_clearance"] = REACH_HEIGHT_CLEARANCE
                else:
                    kwargs["grasp_yaw"] = goal.get("grasp_yaw")
                    kwargs["gripper_width"] = goal.get("gripper_width")
                    kwargs["grasp_mask"] = goal.get("grasp_mask")
                    kwargs["long_axis_angle"] = goal.get("long_axis_angle")
                self._execute_approach(**kwargs)
            except Exception as e:
                print(f"{kind.capitalize()} error: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"{kind.capitalize()} failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
            finally:
                launch_next = False
                if self._deferred_next_goal_start and self._goal_sequence_has_next():
                    self._deferred_next_goal_start = False
                    launch_next = True
                with self._action_lock:
                    st = self._action_state
                if st == 'running':
                    self._set_action_state('idle')
                self._update_goal_queue_label()
                self._update_next_goal_button_state()
                if launch_next:
                    self._start_next_queued_goal()

        from threading import Thread
        Thread(target=run, daemon=True).start()
        return True

    def _start_next_queued_goal(self):
        goals = self._goal_sequence_order()
        if self.queued_goal_cursor >= len(goals):
            self._set_status("No queued goal remaining", "QLabel { color: gray; font-size: 10px; }")
            self._update_next_goal_button_state()
            return False
        goal = goals[self.queued_goal_cursor]
        preserve = bool(self.queued_sequence_started and self._pre_action_state is not None)
        if preserve:
            self._sync_pre_action_rotation_from_odom()
        if not self._start_prepared_goal(goal, preserve_existing_pre_action_state=preserve):
            return False
        self.queued_sequence_started = True
        self.queued_goal_cursor += 1
        self._update_goal_queue_label()
        self._update_next_goal_button_state()
        return True

    def go_to_next_goal(self):
        """Run next queued goal, or skip current scripted action and continue sequence."""
        goals = self._goal_sequence_order()
        if not goals:
            self._set_status("No queued goals. Use right-click: Add Grasp Goal / Add Reach Goal.",
                             "QLabel { color: orange; font-size: 10px; }")
            return
        with self._action_lock:
            st = self._action_state

        # If a scripted action is active and another queued goal exists, abort current
        # action and continue with the next queued goal.
        if st in ('running', 'paused', 'awaiting_confirm') and self._goal_sequence_has_next():
            self._deferred_next_goal_start = True
            self._skip_to_next_goal_requested = True
            with self._action_lock:
                self._action_abort_requested = True
            self._set_action_state('running')  # release paused/confirm wait loops
            self._set_status("Skipping current action. Will move to next queued goal...",
                             "QLabel { color: orange; font-size: 10px; }")
            self._update_next_goal_button_state()
            return

        if st != 'idle':
            self._set_status("Wait for current action to finish or pause it first",
                             "QLabel { color: orange; font-size: 10px; }")
            return

        if not self._start_next_queued_goal():
            self._update_next_goal_button_state()

    def _reach_to_pixel(self, px, py):
        """Reach 10cm above the 3D point at the given pixel."""
        if not self._begin_action('reach'):
            self._set_status("Another action is already running/paused",
                             "QLabel { color: orange; font-size: 10px; }")
            return

        point_base, depth = self._get_3d_point_at_pixel(px, py)
        if point_base is None:
            self._set_action_state('idle')
            self._set_status("No valid depth at clicked point", "QLabel { color: red; }")
            return

        # Find segment mask at this pixel for collision-aware height
        object_top_z = None
        if self.segments:
            for seg in self.segments:
                if py < seg['mask'].shape[0] and px < seg['mask'].shape[1]:
                    if seg['mask'][py, px] > 0:
                        object_top_z = self._compute_object_top_z(seg['mask'])
                        break

        print(f"\n{'='*60}")
        print(f"REACH TO PIXEL ({px}, {py})")
        if object_top_z is not None:
            print(f"  Object top Z: {object_top_z:.3f}m (base_link)")
        print(f"{'='*60}")
        self._set_status("Reaching to point...", "QLabel { color: blue; font-size: 10px; }")

        def run():
            try:
                self._execute_approach(point_base, mode='reach', height_clearance=REACH_HEIGHT_CLEARANCE,
                                       object_top_z=object_top_z)
            except Exception as e:
                print(f"Reach error: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"Reach failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
            finally:
                with self._action_lock:
                    st = self._action_state
                if st == 'running':
                    self._set_action_state('idle')

        from threading import Thread
        Thread(target=run, daemon=True).start()

    def _grasp_at_pixel(self, px, py, segment=None):
        """Grasp the object at the given pixel with orientation + width estimation.
        segment: optional segment dict with 'mask'. If None, looks up from self.segments."""
        if not self._begin_action('grasp'):
            self._set_status("Another action is already running/paused",
                             "QLabel { color: orange; font-size: 10px; }")
            return

        point_base, depth = self._get_3d_point_at_pixel(px, py)
        if point_base is None:
            self._set_action_state('idle')
            self._set_status("No valid depth at clicked point", "QLabel { color: red; }")
            return
        if depth < 0.1 or depth > 2.0:
            self._set_action_state('idle')
            self._set_status(f"Object too far or invalid depth: {depth:.2f}m", "QLabel { color: red; }")
            return

        # Find the segment mask at this pixel
        mask = None
        if segment is not None:
            mask = segment['mask']
        elif self.segments:
            for seg in self.segments:
                if py < seg['mask'].shape[0] and px < seg['mask'].shape[1]:
                    if seg['mask'][py, px] > 0:
                        mask = seg['mask']
                        break
        # Fallback: use currently selected segment mask if pixel lookup missed
        # due rounding/component split, so grasp keeps top-surface safety.
        if mask is None and self.selected_segment is not None and 'mask' in self.selected_segment:
            sel_mask = self.selected_segment['mask']
            if py < sel_mask.shape[0] and px < sel_mask.shape[1]:
                mask = sel_mask

        # Compute optimal grasp orientation and gripper width
        gripper_width = None
        rect_info = None
        object_top_z = None
        long_axis_angle = None
        grasp_yaw = None
        if mask is not None:
            grasp_yaw, rect_info = self._compute_grasp_orientation(mask, px, py)
            gripper_width = self._estimate_gripper_width(mask, px, py, depth, rect_info)
            object_top_z = self._compute_object_top_z(mask)
            if rect_info is not None and rect_info.get('top_z_max') is not None:
                top_from_grasp_fit = float(rect_info['top_z_max'])
                if object_top_z is None:
                    object_top_z = top_from_grasp_fit
                else:
                    object_top_z = max(float(object_top_z), top_from_grasp_fit)
            # Store for debug overlay drawing
            self._grasp_debug_info = rect_info
            # Extract long axis angle (radians) for reverse IK approach
            if rect_info is not None:
                import math as _m
                long_axis_angle = _m.radians(rect_info['long_axis_angle_deg'])
                print(f"  Long axis angle: {rect_info['long_axis_angle_deg']:.1f}° from +X in base_link")

        # Safety: avoid grasping using only raw click depth (can be below object/table).
        if mask is None or object_top_z is None:
            self._set_action_state('idle')
            self._set_status(
                "Could not estimate object top surface. Re-segment and reselect before grasp.",
                "QLabel { color: red; font-size: 10px; }"
            )
            print("  ABORT grasp: missing mask/top-surface estimate; raw click depth is unsafe for final Z.")
            return

        print(f"\n{'='*60}")
        print(f"GRASP AT PIXEL ({px}, {py}) — orientation-aware")
        if object_top_z is not None:
            print(f"  Object top Z: {object_top_z:.3f}m (base_link)")
        print(f"{'='*60}")
        self._set_status("Executing smart grasp sequence...", "QLabel { color: blue; font-size: 10px; }")

        def run():
            try:
                self._execute_approach(point_base, mode='grasp',
                                       grasp_yaw=grasp_yaw,
                                       gripper_width=gripper_width,
                                       object_top_z=object_top_z,
                                       grasp_mask=mask,
                                       long_axis_angle=long_axis_angle)
            except Exception as e:
                print(f"Grasp error: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"Grasp failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
            finally:
                with self._action_lock:
                    st = self._action_state
                # Keep awaiting_post_grasp state until user returns.
                if st == 'running':
                    self._set_action_state('idle')

        from threading import Thread
        Thread(target=run, daemon=True).start()

    def _get_current_base_yaw(self):
        """Yaw from current odom orientation (radians), or None if unavailable."""
        if self.ros_node.odom is None:
            return None
        q = self.ros_node.odom.pose.pose.orientation
        import math
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    def _sync_pre_action_rotation_from_odom(self):
        """Update saved rotation_applied from odom yaw if start yaw is available."""
        if self._pre_action_state is None:
            return
        if 'base_yaw_start' not in self._pre_action_state:
            return
        yaw_now = self._get_current_base_yaw()
        if yaw_now is None:
            return
        import math
        yaw_start = float(self._pre_action_state['base_yaw_start'])
        dyaw = math.atan2(math.sin(yaw_now - yaw_start), math.cos(yaw_now - yaw_start))
        self._pre_action_state['rotation_applied'] = float(dyaw)

    def _consume_skip_to_next_goal_request(self):
        if self._skip_to_next_goal_requested:
            self._skip_to_next_goal_requested = False
            return True
        return False

    def return_to_start(self):
        """Return arm and base to the position saved before the last reach/grasp."""
        with self._action_lock:
            st = self._action_state

        # If an action is in progress, request abort and let action thread
        # trigger the return safely.
        if st in ('running', 'paused', 'awaiting_confirm', 'awaiting_post_grasp'):
            with self._action_lock:
                self._action_abort_requested = True
            self._set_action_state('running')  # release pause/confirm waits
            self._set_status("Abort requested... returning to start",
                             "QLabel { color: orange; font-size: 10px; }")
            self._set_return_enabled(False)
            return

        if self._pre_action_state is None:
            self._set_status("No saved position to return to", "QLabel { color: red; }")
            return

        self._sync_pre_action_rotation_from_odom()
        state = self._pre_action_state
        self._set_return_enabled(False)
        self._set_status("Returning to start position...", "QLabel { color: blue; font-size: 10px; }")

        print(f"\n{'='*60}")
        print(f"RETURN TO START")
        print(f"{'='*60}")

        def run():
            try:
                import time
                import math

                def _stage_sleep(duration_s):
                    time.sleep(min(float(SCRIPT_STAGE_WAIT_CAP_S), max(0.0, float(duration_s))))

                rotation = state['rotation_applied']
                saved_base_pose = state.get('base_pose_xytheta_start')
                LIFT_MAX = self.ros_node.JOINT_LIMITS[1][1]
                RETRACT_EXT = 0.0
                saved_base_x = float(state.get('base_x', 0.0))
                saved_wrist_yaw = float(state.get('wrist_yaw', 0.0))
                saved_wrist_pitch = float(state.get('wrist_pitch', 0.0))
                saved_wrist_roll = float(state.get('wrist_roll', 0.0))
                saved_lift = float(state.get('lift', 0.7))
                saved_arm = float(state.get('arm_ext', 0.0))
                # Preserve user-adjusted gripper target (if any) across return flow.
                target_q_before = self.ros_node.get_target_qpos()
                if isinstance(target_q_before, list) and len(target_q_before) >= 8:
                    preserved_gripper_target = float(target_q_before[7])
                else:
                    preserved_gripper_target = None

                # Prevent background command stream from reopening gripper or overwriting wrist/base_x.
                self._freeze_streaming_commands_to_current_state()
                if preserved_gripper_target is not None:
                    with self.ros_node._lock:
                        if self.ros_node.qpos is not None and len(self.ros_node.qpos) >= 8:
                            self.ros_node.qpos[7] = float(
                                np.clip(
                                    preserved_gripper_target,
                                    float(self.ros_node.JOINT_LIMITS[7][0]),
                                    float(self.ros_node.JOINT_LIMITS[7][1]),
                                )
                            )
                        if self.ros_node.published_qpos is not None and len(self.ros_node.published_qpos) >= 8:
                            self.ros_node.published_qpos[7] = float(self.ros_node.qpos[7])

                # Step 1: Lift to max first for safe retraction/rotation
                print(f"Lifting to max height ({LIFT_MAX:.3f}m) before return...")
                cur_joint = self._current_manip_joint6()
                if isinstance(cur_joint, list) and len(cur_joint) >= 6:
                    cur_joint[1] = float(LIFT_MAX)
                    self.ros_node.execute_arm_to(
                        cur_joint[:6],
                        gripper=None,
                        blocking=True,
                        timeout_s=8.0,
                        reliable=False,
                    )
                _stage_sleep(2.0)

                # Step 2: Retract arm
                print(f"Retracting arm to {RETRACT_EXT:.3f}m...")
                cur_joint = self._current_manip_joint6()
                if isinstance(cur_joint, list) and len(cur_joint) >= 6:
                    cur_joint[2] = float(RETRACT_EXT)
                    self.ros_node.execute_arm_to(
                        cur_joint[:6],
                        gripper=None,
                        blocking=True,
                        timeout_s=8.0,
                        reliable=False,
                    )
                _stage_sleep(2.0)

                # Step 3: Restore full base odom pose if available. This covers
                # manual base adjustments made during/after scripted actions.
                restored_base_pose = False
                if isinstance(saved_base_pose, list) and len(saved_base_pose) >= 3:
                    cur_pose = self.ros_node.get_measured_base_pose_xytheta()
                    if isinstance(cur_pose, list) and len(cur_pose) >= 3:
                        cx, cy, ct = float(cur_pose[0]), float(cur_pose[1]), float(cur_pose[2])
                        sx, sy, st = float(saved_base_pose[0]), float(saved_base_pose[1]), float(saved_base_pose[2])
                        wx = sx - cx
                        wy = sy - cy
                        dx_rel = math.cos(ct) * wx + math.sin(ct) * wy
                        dy_rel = -math.sin(ct) * wx + math.cos(ct) * wy
                        dtheta_rel = self.ros_node._wrap_angle(st - ct)
                        if (abs(dx_rel) > 0.01) or (abs(dy_rel) > 0.01) or (abs(dtheta_rel) > 0.02):
                            print(
                                "Restoring base pose "
                                f"(dx={dx_rel:+.3f}m, dy={dy_rel:+.3f}m, dtheta={math.degrees(dtheta_rel):+.1f}deg)..."
                            )
                            ok_pose = self.ros_node.move_base_relative(
                                dx=float(dx_rel),
                                dy=float(dy_rel),
                                dtheta=float(dtheta_rel),
                                blocking=True,
                                timeout_s=max(3.0, 4.0 + 8.0 * (abs(dx_rel) + abs(dy_rel) + abs(dtheta_rel))),
                            )
                            if not ok_pose:
                                print("  WARNING: bridge move_base_relative full-pose restore failed")
                            _stage_sleep(0.8)
                        restored_base_pose = True

                # Fallback for old saved states without full base pose.
                if (not restored_base_pose) and abs(rotation) > 0.02:
                    print(f"Rotating base back ({math.degrees(-rotation):.1f}°)...")
                    ok_rot = self.ros_node.move_base_relative(
                        dx=0.0,
                        dy=0.0,
                        dtheta=float(-rotation),
                        blocking=True,
                        timeout_s=max(2.0, 2.0 + 6.0 * abs(rotation)),
                    )
                    if not ok_rot:
                        print("  WARNING: bridge move_base_relative rotation failed during return")
                    _stage_sleep(max(1.0, abs(rotation) * 3.0))

                # Step 4: Restore base_x + wrist orientation (keep gripper closed)
                print(
                    "Restoring base_x/wrist "
                    f"(base_x={saved_base_x:.3f}, yaw={saved_wrist_yaw:.3f}, "
                    f"pitch={saved_wrist_pitch:.3f}, roll={saved_wrist_roll:.3f})"
                )
                restore_joint = [
                    float(saved_base_x),
                    float(LIFT_MAX),
                    float(RETRACT_EXT),
                    float(saved_wrist_yaw),
                    float(saved_wrist_pitch),
                    float(saved_wrist_roll),
                ]
                self.ros_node.execute_arm_to(
                    restore_joint[:6],
                    gripper=None,
                    blocking=True,
                    timeout_s=10.0,
                    reliable=False,
                )
                _stage_sleep(0.5)

                # Step 5: Restore head pan/tilt
                print(f"Restoring head pan={state['head_pan']:.3f}, tilt={state['head_tilt']:.3f}")
                self.ros_node.execute_arm_to(
                    restore_joint[:6],
                    gripper=None,
                    head=[float(state['head_pan']), float(state['head_tilt'])],
                    blocking=True,
                    timeout_s=8.0,
                    reliable=False,
                )
                _stage_sleep(1.0)

                # Step 6: Return to saved start arm/lift/base_x and keep closed gripper.
                print(
                    "Restoring saved arm/lift/base_x "
                    f"(base_x={saved_base_x:.3f}, lift={saved_lift:.3f}, arm={saved_arm:.3f})..."
                )
                final_joint = [
                    float(saved_base_x),
                    float(saved_lift),
                    float(saved_arm),
                    float(saved_wrist_yaw),
                    float(saved_wrist_pitch),
                    float(saved_wrist_roll),
                ]
                self.ros_node.execute_arm_to(
                    final_joint[:6],
                    gripper=None,
                    head=[float(state['head_pan']), float(state['head_tilt'])],
                    blocking=True,
                    timeout_s=10.0,
                    reliable=False,
                )
                _stage_sleep(1.5)

                self._pre_action_state = None
                self.queued_sequence_started = False
                self.ros_node.sync_command_targets_to_actual()
                print("Return completed!")
                self._set_status("Returned to start position", "QLabel { color: green; font-size: 10px; }")
                self._set_action_state('idle')
                self._update_goal_queue_label()
                self._update_next_goal_button_state()

            except Exception as e:
                print(f"Return error: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"Return failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
                self._set_return_enabled(True)
                self._set_action_state('idle')

        from threading import Thread
        Thread(target=run, daemon=True).start()

    def closeEvent(self, event):
        """Handle window close"""
        print("Closing application...", flush=True)
        if self.is_recording_demo:
            self.stop_demo_recording()
        self.control_timer.stop()
        self.robot_controller.stop()
        event.accept()


class BridgeGoalQueueUI(RobotTeleopUI):
    """Bridge wrapper for queued-goal UX: explicit execute, remove, clear, abort."""

    # Defensive compatibility methods in case of inheritance/path skew.
    def _set_manual_gripper_override(self, value: float | None) -> None:
        setter = getattr(super(), "_set_manual_gripper_override", None)
        if callable(setter):
            setter(value)
            return
        if value is None or not isinstance(value, (int, float)):
            self._manual_gripper_override = None
            return
        try:
            v = float(value)
            if not math.isfinite(v):
                self._manual_gripper_override = None
                return
        except (TypeError, ValueError):
            self._manual_gripper_override = None
            return
        lo, hi = self.ros_node.JOINT_LIMITS[7]
        self._manual_gripper_override = float(np.clip(v, lo, hi))

    def _get_manual_gripper_target(self, fallback: float | None = None) -> float | None:
        getter = getattr(super(), "_get_manual_gripper_target", None)
        if callable(getter):
            return getter(fallback)
        if self._manual_gripper_override is not None:
            return float(self._manual_gripper_override)
        target = self.ros_node.get_target_qpos()
        if isinstance(target, list) and len(target) >= 8:
            try:
                v = float(target[7])
                if math.isfinite(v):
                    lo, hi = self.ros_node.JOINT_LIMITS[7]
                    return float(np.clip(v, lo, hi))
            except (TypeError, ValueError):
                pass
        actual = self.ros_node.get_actual_qpos()
        if isinstance(actual, list) and len(actual) >= 8:
            try:
                v = float(actual[7])
                if math.isfinite(v):
                    lo, hi = self.ros_node.JOINT_LIMITS[7]
                    return float(np.clip(v, lo, hi))
            except (TypeError, ValueError):
                pass
        return None if fallback is None else float(fallback)

    def adjust_gripper_step(self, direction):
        grip_min, grip_max = self.ros_node.JOINT_LIMITS[7]
        current = None
        candidates = [
            getattr(self, "_get_manual_gripper_target", None),
            getattr(super(), "_get_manual_gripper_target", None),
        ]
        for getter in candidates:
            if callable(getter):
                try:
                    current = getter(float(self.ros_node.JOINT_LIMITS[7][1]))
                    break
                except Exception:
                    current = None
        if current is None:
            current = self.ros_node.get_target_qpos()
            if isinstance(current, list) and len(current) >= 8:
                try:
                    current = float(current[7])
                except (TypeError, ValueError):
                    current = float(self.ros_node.JOINT_LIMITS[7][1])
            else:
                current = float(self.ros_node.JOINT_LIMITS[7][1])

        delta = self.gripper_step if direction > 0 else -self.gripper_step
        target = max(grip_min, min(grip_max, float(current) + delta))
        self.set_gripper(target)
        self.status_label.setText(
            f"Gripper -> {target:.3f} (step {self.gripper_step:.3f})"
        )
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")

    def create_object_list(self):
        widget = super().create_object_list()
        layout = widget.layout()
        if layout is None:
            return widget

        if hasattr(self, "next_goal_button"):
            self.next_goal_button.setText("Next Goal (Manual)")
            self.next_goal_button.setToolTip(
                "Manually run/skip one queued goal. Use Execute Goals for full queued run."
            )

        self.execute_goals_button = QPushButton("Execute Goals")
        self.execute_goals_button.setMinimumHeight(35)
        self.execute_goals_button.clicked.connect(self.execute_all_queued_goals)
        layout.addWidget(self.execute_goals_button)

        self.abort_goals_button = QPushButton("Abort + Clear Goals")
        self.abort_goals_button.setMinimumHeight(35)
        self.abort_goals_button.clicked.connect(self.abort_and_clear_goals)
        layout.addWidget(self.abort_goals_button)

        self.goal_list_widget = QListWidget()
        self.goal_list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.goal_list_widget.setMinimumHeight(80)
        self.goal_list_widget.setMaximumHeight(130)
        self.goal_list_widget.itemSelectionChanged.connect(self._on_goal_selection_changed)
        layout.addWidget(self.goal_list_widget)

        row = QHBoxLayout()
        self.remove_goal_button = QPushButton("Remove Selected Goal")
        self.remove_goal_button.clicked.connect(self.remove_selected_goal)
        row.addWidget(self.remove_goal_button)
        self.clear_goals_button = QPushButton("Clear Goals")
        self.clear_goals_button.clicked.connect(self.clear_queued_goals)
        row.addWidget(self.clear_goals_button)
        layout.addLayout(row)

        self._update_goal_queue_label()
        self._update_next_goal_button_state()
        self._on_goal_selection_changed()
        return widget

    def _pixel_from_event(self, event):
        """Map click position on scaled QLabel to image pixel (handles letterboxing)."""
        if self.head_rgb is None or not hasattr(self, "head_display"):
            return None, None

        disp = self.head_display
        img_h, img_w = self.head_rgb.shape[:2]
        if img_h <= 0 or img_w <= 0:
            return None, None
        w = max(1, int(disp.width()))
        h = max(1, int(disp.height()))

        scale = min(float(w) / float(img_w), float(h) / float(img_h))
        shown_w = float(img_w) * scale
        shown_h = float(img_h) * scale
        x_off = (float(w) - shown_w) * 0.5
        y_off = (float(h) - shown_h) * 0.5

        ex = float(event.pos().x())
        ey = float(event.pos().y())
        if ex < x_off or ex >= (x_off + shown_w) or ey < y_off or ey >= (y_off + shown_h):
            return None, None

        px = int((ex - x_off) / scale)
        py = int((ey - y_off) / scale)
        px = int(np.clip(px, 0, img_w - 1))
        py = int(np.clip(py, 0, img_h - 1))
        return px, py

    def _on_goal_selection_changed(self):
        if not hasattr(self, "remove_goal_button"):
            return
        has_sel = hasattr(self, "goal_list_widget") and (self.goal_list_widget.currentItem() is not None)
        self.remove_goal_button.setEnabled(bool(has_sel))

    def _update_goal_queue_label(self):
        super()._update_goal_queue_label()
        if not hasattr(self, "goal_list_widget"):
            return
        self.goal_list_widget.clear()
        goals = self._goal_sequence_order()
        for idx, goal in enumerate(goals):
            kind = str(goal.get("kind", "?"))
            px = goal.get("px")
            py = goal.get("py")
            marker = "-> " if idx == self.queued_goal_cursor and self._goal_sequence_has_next() else "   "
            item = QListWidgetItem(f"{marker}{idx + 1}. {kind} ({px}, {py})")
            item.setData(Qt.ItemDataRole.UserRole, kind)
            self.goal_list_widget.addItem(item)
        self._on_goal_selection_changed()

    def _update_next_goal_button_state(self):
        super()._update_next_goal_button_state()
        goals = self._goal_sequence_order()
        has_goals = len(goals) > 0
        with self._action_lock:
            state = self._action_state
        active = state in ("running", "paused", "awaiting_confirm", "awaiting_post_grasp")
        idle = state == "idle"
        if hasattr(self, "execute_goals_button"):
            self.execute_goals_button.setEnabled(bool(has_goals and idle))
        if hasattr(self, "clear_goals_button"):
            self.clear_goals_button.setEnabled(bool(has_goals and idle))
        if hasattr(self, "abort_goals_button"):
            self.abort_goals_button.setEnabled(bool(has_goals or active))

    def remove_selected_goal(self):
        if not hasattr(self, "goal_list_widget"):
            return
        item = self.goal_list_widget.currentItem()
        if item is None:
            return
        kind = item.data(Qt.ItemDataRole.UserRole)
        if kind not in ("grasp", "reach"):
            return
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Remove goals only while idle", "QLabel { color: orange; font-size: 10px; }")
                return
        self.queued_goals[kind] = None
        self._reset_goal_sequence_progress()
        self._set_status(f"Removed queued {kind} goal", "QLabel { color: #1e88e5; font-size: 10px; }")

    def clear_queued_goals(self):
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Clear goals only while idle", "QLabel { color: orange; font-size: 10px; }")
                return
        self.queued_goals = {"grasp": None, "reach": None}
        self._reset_goal_sequence_progress()
        self._set_status("Cleared queued goals", "QLabel { color: gray; font-size: 10px; }")

    def execute_all_queued_goals(self):
        goals = self._goal_sequence_order()
        if not goals:
            self._set_status(
                "No queued goals. Right-click image and add goals first.",
                "QLabel { color: orange; font-size: 10px; }",
            )
            return
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Wait for current action to finish", "QLabel { color: orange; font-size: 10px; }")
                return
        self._reset_goal_sequence_progress()
        self._deferred_next_goal_start = True
        if not self._start_next_queued_goal():
            self._deferred_next_goal_start = False
            self._set_status("Failed to start goal execution", "QLabel { color: red; font-size: 10px; }")

    def abort_and_clear_goals(self):
        with self._action_lock:
            state = self._action_state
            self._action_abort_requested = True
        self._deferred_next_goal_start = False
        self._skip_to_next_goal_requested = False
        self.queued_goals = {"grasp": None, "reach": None}
        self._reset_goal_sequence_progress()
        if state in ("paused", "awaiting_confirm", "awaiting_post_grasp"):
            self._set_action_state("running")
        if state == "idle":
            self._set_status("Cleared queued goals", "QLabel { color: gray; font-size: 10px; }")
        else:
            self._set_status(
                "Abort requested. Cleared queued goals.",
                "QLabel { color: orange; font-size: 10px; }",
            )


def main() -> None:
    bridge = StretchAIDemoBridge()
    try:
        bridge.connect(timeout_s=STRETCH_AI_CONNECT_TIMEOUT_S)
    except Exception as exc:
        bridge.close()
        raise SystemExit(f"stretch_ai bridge connection error: {exc}") from exc

    app = QApplication(sys.argv)
    ui = BridgeGoalQueueUI(bridge)
    ui.show()

    rc = 1
    try:
        rc = app.exec()
    finally:
        try:
            bridge.publish_hold_stop()
        except Exception:
            pass
        bridge.close()

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
