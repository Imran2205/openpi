#!/usr/bin/env python3
"""
Robot Teleoperation UI with SAM Segmentation (ROS2 Version)
Features:
- RGB and Depth camera feeds from ROS topics
- Robot control buttons with ROS publishers
- SAM-based object segmentation
- Interactive object selection and manipulation
- Advanced grasp planning with IK
- Thread-safe UI updates
"""

import sys
import os
import json
import queue
from pathlib import Path

# Fix Qt plugin conflict between OpenCV and PyQt6
# This MUST be set before importing cv2
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QPushButton, QListWidget,
                              QGridLayout, QGroupBox, QListWidgetItem, QScrollArea,
                              QSizePolicy, QMenu, QSlider, QLineEdit, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QAction
from ultralytics import SAM
import colorsys
import time
import threading

# When True, head camera pans opposite to base rotation during
# grasp/reach actions so the scene stays roughly in view.
COMPENSATE_HEAD_ON_ROTATE = True

# How far above the object (meters) the gripper stops in reach mode.
REACH_HEIGHT_CLEARANCE = 0.20

# Downward grasp line angle (degrees from horizontal plane).
# 0 = horizontal, 90 = straight down.
# The grasp reverse-geometry computes wrist target by tracing a line of
# this angle from the grasp point within the principal-axis vertical plane.
GRASP_PITCH_DEG = 40.0

# Optional empirical calibration (meters) for residual bias after geometry.
# Keep zero unless tuning on hardware.
GRASP_LATERAL_TRIM_M =0.03   # + => compensate left bias by rotating right
GRASP_REACH_TRIM_M = 0.025    # + => extend slightly farther (2-3cm calibration)
GRASP_CLOSE_EXTRA_M = 0.015   # extra close after width-based command
GRASP_STALK_LENGTH_M = 0.2716  # wrist_pitch axis -> grasp center (URDF FK, ~0.27163m)
GRASP_RESIDUAL_ROT_GAIN = 0.60  # damp residual base correction to avoid overshoot
GRASP_RESIDUAL_ROT_MAX_DEG = 8.0  # safety cap for one residual correction step
GRASP_REACH_CORR_GAIN = 0.8
GRASP_REACH_CORR_MAX_STEP_M = 0.04
GRASP_REACH_CORR_THRESH_M = 0.008
GRASP_REACH_CORR_ITERS = 2
GRASP_PRELOWER_VERIFY_TIP_MARGIN_M = 0.03  # tip hover above grasp surface for user verification
GRASP_TIP_Z_MARGIN_M = 0.08  # final grasp tip stays at least this much above estimated top surface
GRASP_TARGET_Z_OFFSET_M = 0.03  # fixed added height above clicked target_z for grasp safety
DEMO_RECORD_FPS = 10  # record images + synchronized telemetry at ~10 Hz (best-effort)
DEMO_RECORD_QUEUE_MAX = 8

# Per-channel maximum step used when smoothing /joint_pose_cmd.
# Values are chosen to match the minimum values exposed by UI speed sliders.
# qpos order:
# [arm_extension, lift, wrist_yaw, wrist_pitch, wrist_roll,
#  head_pan, head_tilt, gripper, base_linear, base_angular]
COMMAND_SMOOTH_STEP_SIZES = [
    0.005,  # arm_extension
    0.005,  # lift
    0.020,  # wrist_yaw
    0.020,  # wrist_pitch
    0.020,  # wrist_roll
    0.020,  # head_pan
    0.020,  # head_tilt
    0.005,  # gripper
    0.020,  # base linear velocity
    0.030,  # base angular velocity (base min speed * 1.5)
]
DEFAULT_COMMAND_SMOOTH_DELAY_S = 0.05
# Cap long scripted stage waits in reach/grasp/return sequences.
SCRIPT_STAGE_WAIT_CAP_S = 0.5

# SAM post-processing thresholds for connected-component instance splitting.
# Very small components are often fragmented mask artifacts and not graspable.
SAM_CC_MIN_AREA_PX = 600
SAM_CC_MIN_WIDTH_PX = 18
SAM_CC_MIN_HEIGHT_PX = 18


def _to_jsonable(obj):
    """Convert numpy/ROS-friendly values to JSON-serializable Python types."""
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
    """Episode recorder with LeRobot-like directory and naming conventions.

    Layout (per dataset root):
      - meta/info.json
      - meta/episodes.jsonl
      - meta/tasks.jsonl
      - data/chunk-XXX/episode_XXXXXX.(parquet|jsonl)
      - images/chunk-XXX/observation.images.<cam>/episode_XXXXXX/frame_XXXXXX.png
      - depth/chunk-XXX/observation.depth.<cam>/episode_XXXXXX/frame_XXXXXX.png
      - prompts/chunk-XXX/episode_XXXXXX.txt

    If pyarrow is unavailable, episode tabular data is written as JSONL fallback.
    """

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
        self.writers = {}  # legacy unused in v3 recorder (frame PNGs only)
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

    def _create_writer(self, key: str, path: Path, frame):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(self.target_fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")
        self.writers[key] = writer
        return writer

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

        # Base directories
        meta_dir = self._ensure(root / "meta")
        data_dir = self._ensure(root / "data" / self.chunk_name)
        images_dir = self._ensure(root / "images" / self.chunk_name)
        prompts_dir = self._ensure(root / "prompts" / self.chunk_name)
        depth_dir = self._ensure(root / "depth" / self.chunk_name)

        # Camera/depth streams
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
        # Throttle to requested recording frequency (LeRobot-style fixed fps)
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

        for writer in self.writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self.writers = {}
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

        # Meta files
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

        # Reset queue for next episode
        self._sample_queue = queue.Queue(maxsize=self.queue_maxsize)
        return summary

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState, CameraInfo
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, TransformStamped, PointStamped
from sensor_msgs.msg import Imu, BatteryState, MagneticField
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

# Try to import grasp planner (may need adaptation for ROS)
try:
    from grasp_planner import GraspPlanner
    HAS_GRASP_PLANNER = True
except ImportError:
    print("Warning: grasp_planner not found. Grasp planning will be disabled.")
    HAS_GRASP_PLANNER = False


class CameraNode(Node):
    """Dedicated node for camera subscriptions only.
    Kept separate from the control node so publishers/timers/TF
    don't interfere with large image message delivery."""

    def __init__(self):
        super().__init__('camera_node')
        self.bridge = CvBridge()
        self.head_rgb = None
        self.head_depth = None
        self.wrist_rgb = None
        self.wrist_depth = None
        self.camera_info = None
        self.wrist_camera_info = None

        self.head_rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.head_rgb_callback, 1)
        self.head_depth_sub = self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw', self.head_depth_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)
        self.wrist_rgb_sub = self.create_subscription(
            Image, '/gripper_camera/color/image_rect_raw', self.wrist_rgb_callback, 1)
        self.wrist_depth_sub = self.create_subscription(
            Image, '/gripper_camera/aligned_depth_to_color/image_raw', self.wrist_depth_callback, 1)
        self.wrist_camera_info_sub = self.create_subscription(
            CameraInfo, '/gripper_camera/aligned_depth_to_color/camera_info', self.wrist_camera_info_callback, 10)
        self.get_logger().info('Camera node initialized')

    def head_rgb_callback(self, msg):
        try:
            # Force copy — CvBridge returns a VIEW of the msg buffer which
            # the DDS middleware may reuse/overwrite before cv2.rotate runs
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').copy()
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            self.head_rgb = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error (Head RGB): {e}')

    def head_depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv_image.dtype == np.uint16:
                cv_image = cv_image.astype(np.float32) / 1000.0
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            # print(cv_image.shape)
            self.head_depth = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error (Depth): {e}')

    def wrist_rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.wrist_rgb = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error (Wrist): {e}')

    def wrist_depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv_image.dtype == np.uint16:
                cv_image = cv_image.astype(np.float32) / 1000.0
            self.wrist_depth = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error (Wrist Depth): {e}')

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info('Camera info received')

    def wrist_camera_info_callback(self, msg):
        if self.wrist_camera_info is None:
            self.wrist_camera_info = msg
            self.get_logger().info('Wrist camera info received')


class RobotROSNode(Node):
    """ROS2 node for robot control (no camera subscriptions)"""

    def __init__(self, camera_node):
        super().__init__('robot_teleop_node')

        self.camera_node = camera_node

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # TF2 for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_states = None
        self.joint_states_initialized = False
        self.imu_mobile_base = None
        self.imu_wrist = None
        self.camera_accel_imu = None
        self.camera_gyro_imu = None
        self.mag_mobile_base = None
        self.battery_state = None
        self.odom = None
        self.actual_qpos = None  # Live measured qpos from joint_states (+ odom for base velocities)

        # Control state (matches the example controller)
        # [arm_extension, lift, wrist_yaw, wrist_pitch, wrist_roll,
        #  head_pan, head_tilt, gripper, base_trans_vel, base_rot_vel]
        self.qpos = None  # Will be initialized from joint states
        # Actual command stream published to /joint_pose_cmd after smoothing.
        self.published_qpos = None
        self.command_smooth_step_sizes = list(COMMAND_SMOOTH_STEP_SIZES)
        self.command_smooth_delay_s = float(DEFAULT_COMMAND_SMOOTH_DELAY_S)

        # Joint limits from example
        self.JOINT_LIMITS = [
            (0.00, 0.51),      # [0] arm extension
            (0.08, 1.05),      # [1] lift
            (-2.6, 2.6),       # [2] wrist_yaw
            (-1.0, 1.57),      # [3] wrist_pitch
            (-1.57, 1.57),     # [4] wrist_roll
            (-1.57, 1.57),     # [5] head_pan
            (-1.0, 1.0),       # [6] head_tilt
            (-0.1, 0.5501),    # [7] gripper
            (-2.0, 2.0),       # [8] base_translate (velocity)
            (-5.00, 5.00),     # [9] base_rotate (velocity)
        ]

        # Publishers
        self.joint_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_pose_cmd',
            10
        )
        # cmd_vel not used in position mode - robot controlled via /joint_pose_cmd

        # Joint states subscriber
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/stretch/joint_states',
            self.joint_states_callback,
            10
        )
        self.imu_mobile_base_sub = self.create_subscription(
            Imu,
            '/imu_mobile_base',
            self.imu_mobile_base_callback,
            20
        )
        self.imu_wrist_sub = self.create_subscription(
            Imu,
            '/imu_wrist',
            self.imu_wrist_callback,
            20
        )
        self.camera_accel_imu_sub = self.create_subscription(
            Imu,
            '/camera/accel/sample',
            self.camera_accel_imu_callback,
            20
        )
        self.camera_gyro_imu_sub = self.create_subscription(
            Imu,
            '/camera/gyro/sample',
            self.camera_gyro_imu_callback,
            20
        )
        self.mag_mobile_base_sub = self.create_subscription(
            MagneticField,
            '/magnetometer_mobile_base',
            self.mag_mobile_base_callback,
            20
        )
        self.battery_sub = self.create_subscription(
            BatteryState,
            '/battery',
            self.battery_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            20
        )

        # Publish timer also drives smoothing steps.
        self.publish_timer = self.create_timer(self.command_smooth_delay_s, self.publish_commands)

        self.get_logger().info('Robot ROS Node initialized')

    def joint_states_callback(self, msg):
        """Callback for joint states"""
        self.joint_states = msg
        measured_qpos = self._extract_measured_qpos(msg)
        if measured_qpos is not None:
            self.actual_qpos = measured_qpos

        # Initialize qpos from current joint states on first callback
        if not self.joint_states_initialized and self.qpos is None:
            try:
                if measured_qpos is None:
                    raise ValueError("measured_qpos is unavailable")
                # Initialize control array with current measured positions.
                # Base channels are command velocities and start at zero.
                self.qpos = [
                    measured_qpos[0],  # [0]
                    measured_qpos[1],  # [1]
                    measured_qpos[2],  # [2]
                    measured_qpos[3],  # [3]
                    measured_qpos[4],  # [4]
                    measured_qpos[5],  # [5]
                    measured_qpos[6],  # [6]
                    measured_qpos[7],  # [7]
                    0.0,            # [8] base_trans_vel
                    0.0,            # [9] base_rot_vel
                ]

                self.joint_states_initialized = True
                self.published_qpos = list(self.qpos)
                self.get_logger().info(f'Initialized joint positions: {self.qpos[:8]}')

            except (ValueError, IndexError) as e:
                self.get_logger().warn(f'Could not initialize joint positions: {e}')
                # Fallback to safe defaults
                self.qpos = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]
                self.actual_qpos = list(self.qpos)
                self.published_qpos = list(self.qpos)

    def imu_mobile_base_callback(self, msg):
        self.imu_mobile_base = msg

    def imu_wrist_callback(self, msg):
        self.imu_wrist = msg

    def camera_accel_imu_callback(self, msg):
        self.camera_accel_imu = msg

    def camera_gyro_imu_callback(self, msg):
        self.camera_gyro_imu = msg

    def mag_mobile_base_callback(self, msg):
        self.mag_mobile_base = msg

    def battery_callback(self, msg):
        self.battery_state = msg

    def odom_callback(self, msg):
        self.odom = msg
        if self.actual_qpos is not None and len(self.actual_qpos) >= 10:
            # Keep measured base velocities fresh in the measured qpos vector.
            self.actual_qpos[8] = float(msg.twist.twist.linear.x)
            self.actual_qpos[9] = float(msg.twist.twist.angular.z)

    def _extract_measured_qpos(self, msg):
        """Build measured qpos from joint state + odom velocity channels."""
        try:
            arm_lift = msg.position[msg.name.index('joint_lift')]
            arm_extension = 4 * msg.position[msg.name.index('joint_arm_l0')]
            wrist_yaw = msg.position[msg.name.index('joint_wrist_yaw')]
            wrist_pitch = msg.position[msg.name.index('joint_wrist_pitch')]
            wrist_roll = msg.position[msg.name.index('joint_wrist_roll')]
            head_pan = msg.position[msg.name.index('joint_head_pan')]
            head_tilt = msg.position[msg.name.index('joint_head_tilt')]
            gripper_pos = msg.position[msg.name.index('joint_gripper_finger_left')]
            base_trans_vel = 0.0
            base_rot_vel = 0.0
            if self.odom is not None:
                base_trans_vel = float(self.odom.twist.twist.linear.x)
                base_rot_vel = float(self.odom.twist.twist.angular.z)
            return [
                float(arm_extension),
                float(arm_lift),
                float(wrist_yaw),
                float(wrist_pitch),
                float(wrist_roll),
                float(head_pan),
                float(head_tilt),
                float(gripper_pos),
                float(base_trans_vel),
                float(base_rot_vel),
            ]
        except (ValueError, IndexError, TypeError):
            return None

    def get_actual_qpos(self):
        """Latest measured qpos for recording."""
        if self.actual_qpos is not None:
            return list(self.actual_qpos)
        if self.qpos is not None:
            return list(self.qpos)
        return []

    def get_published_qpos(self):
        """Latest smoothed qpos command published to /joint_pose_cmd."""
        if self.published_qpos is not None:
            return list(self.published_qpos)
        if self.qpos is not None:
            return list(self.qpos)
        return []

    @property
    def camera_info(self):
        return self.camera_node.camera_info

    def pixel_to_3d_point(self, pixel_x, pixel_y, depth):
        """
        Convert pixel coordinates and depth to 3D point in camera frame
        """
        if self.camera_info is None:
            self.get_logger().warn('Camera info not available yet')
            return None

        # Get camera intrinsics
        fx = self.camera_info.k[0]  # focal length x
        fy = self.camera_info.k[4]  # focal length y
        cx = self.camera_info.k[2]  # principal point x
        cy = self.camera_info.k[5]  # principal point y

        # Reverse the 90° CW rotation applied to the image.
        # cv2.ROTATE_90_CLOCKWISE: rotated[row_r][col_r] = orig[H-1-col_r][row_r]
        # So: orig_col = row_r = pixel_y,  orig_row = H-1-col_r = H-1-pixel_x
        # where H = original image height (camera_info.height)
        original_x = pixel_y                                       # column in original
        original_y = self.camera_info.height - 1 - pixel_x         # row in original

        # Convert pixel to 3D point in camera optical frame
        # Using pinhole camera model: X = (u - cx) * Z / fx
        x = (original_x - cx) * depth / fx
        y = (original_y - cy) * depth / fy
        z = depth

        # Create PointStamped message in camera frame
        point = PointStamped()
        point.header.frame_id = 'camera_color_optical_frame'
        point.header.stamp = self.get_clock().now().to_msg()
        # Convert to Python float (ROS messages don't accept numpy types)
        point.point.x = float(x)
        point.point.y = float(y)
        point.point.z = float(z)

        return point

    def transform_point_to_base(self, point_stamped):
        """
        Transform a point from camera frame to base_link frame

        Args:
            point_stamped: PointStamped in camera_color_optical_frame

        Returns:
            PointStamped in base_link frame, or None if transform fails
        """
        try:
            # Lookup transform from camera frame to base_link
            # Use Time(0) to get the latest available transform
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                point_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # DEBUG: Print transform details
            self.get_logger().info(
                f"Transform from '{point_stamped.header.frame_id}' to 'base_link':\n"
                f"  Translation: ({transform.transform.translation.x:.3f}, "
                f"{transform.transform.translation.y:.3f}, {transform.transform.translation.z:.3f})\n"
                f"  Rotation (quaternion): ({transform.transform.rotation.x:.3f}, "
                f"{transform.transform.rotation.y:.3f}, {transform.transform.rotation.z:.3f}, "
                f"{transform.transform.rotation.w:.3f})"
            )

            # Transform the point
            point_in_base = tf2_geometry_msgs.do_transform_point(point_stamped, transform)

            self.get_logger().info(
                f"Point before transform (camera frame): ({point_stamped.point.x:.3f}, "
                f"{point_stamped.point.y:.3f}, {point_stamped.point.z:.3f})\n"
                f"Point after transform (base_link): ({point_in_base.point.x:.3f}, "
                f"{point_in_base.point.y:.3f}, {point_in_base.point.z:.3f})"
            )

            return point_in_base

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF transform error: {e}')
            self.get_logger().error(
                f"If you see 'frame does not exist' errors, try these alternative frames:\n"
                f"  - camera_link\n"
                f"  - camera_depth_optical_frame\n"
                f"  - camera_color_frame\n"
                f"Run 'ros2 run tf2_tools view_frames' to see all available frames"
            )
            return None

    def publish_commands(self):
        """Publish control commands at regular intervals"""
        # Don't publish until we have initialized from joint states
        if self.qpos is None:
            return

        if self.published_qpos is None or len(self.published_qpos) != len(self.qpos):
            self.published_qpos = list(self.qpos)

        # Smooth target qpos into smaller increments before publishing.
        for i, target in enumerate(self.qpos):
            current = float(self.published_qpos[i])
            target = float(target)
            step_limit = (
                float(self.command_smooth_step_sizes[i])
                if i < len(self.command_smooth_step_sizes)
                else 0.0
            )
            if step_limit <= 0.0:
                self.published_qpos[i] = target
                continue
            delta = target - current
            if abs(delta) <= step_limit:
                self.published_qpos[i] = target
            else:
                self.published_qpos[i] = current + (step_limit if delta > 0.0 else -step_limit)

        # Publish joint commands
        msg = Float64MultiArray()
        msg.data = self.published_qpos.copy()
        self.joint_pub.publish(msg)

        # Base velocity is included in qpos[8] and qpos[9] and sent via /joint_pose_cmd

    def set_control(self, control_name, value):
        """Set control value by name"""
        if self.qpos is None:
            return

        # Map control names to qpos indices
        control_map = {
            'arm_extension': 0,
            'arm_lift': 1,
            'wrist_yaw': 2,
            'wrist_pitch': 3,
            'wrist_roll': 4,
            'head_pan': 5,
            'head_tilt': 6,
            'gripper': 7,
            'base_linear': 8,
            'base_angular': 9,
        }

        if control_name in control_map:
            idx = control_map[control_name]
            # Clip to limits
            new_value = np.clip(
                value,
                self.JOINT_LIMITS[idx][0],
                self.JOINT_LIMITS[idx][1]
            )
            self.qpos[idx] = new_value
            self.get_logger().info(f'Set {control_name} to {new_value:.3f}', throttle_duration_sec=0.5)

    def adjust_control(self, control_name, delta):
        """Adjust control value by delta"""
        if self.qpos is None:
            return

        control_map = {
            'arm_extension': 0,
            'arm_lift': 1,
            'wrist_yaw': 2,
            'wrist_pitch': 3,
            'wrist_roll': 4,
            'head_pan': 5,
            'head_tilt': 6,
            'gripper': 7,
            'base_linear': 8,
            'base_angular': 9,
        }

        if control_name in control_map:
            idx = control_map[control_name]
            new_value = self.qpos[idx] + delta
            # Clip to limits
            new_value = np.clip(
                new_value,
                self.JOINT_LIMITS[idx][0],
                self.JOINT_LIMITS[idx][1]
            )
            self.qpos[idx] = new_value
            self.get_logger().info(f'Adjust {control_name} by {delta:.3f} to {new_value:.3f}', throttle_duration_sec=0.5)

    def set_command_smoothing_delay(self, delay_s):
        """Update smoothing/publish loop period."""
        delay_s = float(np.clip(delay_s, 0.01, 0.5))
        if abs(delay_s - self.command_smooth_delay_s) < 1e-6:
            return
        self.command_smooth_delay_s = delay_s
        try:
            self.publish_timer.cancel()
        except Exception:
            pass
        self.publish_timer = self.create_timer(self.command_smooth_delay_s, self.publish_commands)
        self.get_logger().info(
            f"Command smoothing delay set to {self.command_smooth_delay_s:.3f}s",
            throttle_duration_sec=0.5,
        )

    def get_images(self):
        """Get current camera images from the camera node"""
        return (
            self.camera_node.head_rgb,
            self.camera_node.wrist_rgb,
            self.camera_node.head_depth,
            self.camera_node.wrist_depth,
        )

    def _imu_to_dict(self, msg):
        if msg is None:
            return None
        return {
            "orientation": [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            "angular_velocity": [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            "linear_acceleration": [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
        }

    def _mag_to_dict(self, msg):
        if msg is None:
            return None
        return {
            "magnetic_field": [msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z],
        }

    def _battery_to_dict(self, msg):
        if msg is None:
            return None
        return {
            "voltage": msg.voltage,
            "current": msg.current,
            "charge": msg.charge,
            "capacity": msg.capacity,
            "percentage": msg.percentage,
            "power_supply_status": int(msg.power_supply_status),
            "power_supply_health": int(msg.power_supply_health),
            "power_supply_technology": int(msg.power_supply_technology),
            "present": bool(msg.present),
        }

    def _odom_to_dict(self, msg):
        if msg is None:
            return None
        return {
            "position": [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            "orientation": [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ],
            "linear_velocity": [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            "angular_velocity": [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
        }

    def _camera_info_to_dict(self, msg):
        if msg is None:
            return None
        return {
            "width": int(msg.width),
            "height": int(msg.height),
            "k": list(msg.k),
            "d": list(msg.d),
            "r": list(msg.r),
            "p": list(msg.p),
            "distortion_model": msg.distortion_model,
        }

    def get_sensor_snapshot(self):
        """Snapshot of robot telemetry for recording."""
        sensors = {
            "observation.qpos_full": list(self.qpos) if self.qpos is not None else [],
            "observation.qpos_actual": list(self.actual_qpos) if self.actual_qpos is not None else [],
            "observation.joint_state.name": [],
            "observation.joint_state.position": [],
            "observation.joint_state.velocity": [],
            "observation.joint_state.effort": [],
            "observation.imu.mobile_base": self._imu_to_dict(self.imu_mobile_base),
            "observation.imu.wrist": self._imu_to_dict(self.imu_wrist),
            "observation.imu.camera_accel": self._imu_to_dict(self.camera_accel_imu),
            "observation.imu.camera_gyro": self._imu_to_dict(self.camera_gyro_imu),
            "observation.magnetometer.mobile_base": self._mag_to_dict(self.mag_mobile_base),
            "observation.battery": self._battery_to_dict(self.battery_state),
            "observation.odom": self._odom_to_dict(self.odom),
            "observation.camera_info.head": self._camera_info_to_dict(self.camera_node.camera_info),
            "observation.camera_info.wrist": self._camera_info_to_dict(self.camera_node.wrist_camera_info),
        }
        if self.joint_states is not None:
            sensors["observation.joint_state.name"] = list(self.joint_states.name)
            sensors["observation.joint_state.position"] = list(self.joint_states.position)
            sensors["observation.joint_state.velocity"] = list(self.joint_states.velocity)
            sensors["observation.joint_state.effort"] = list(self.joint_states.effort)
        return sensors

    def is_ready(self):
        """Check if robot is ready to receive commands"""
        return self.qpos is not None and self.joint_states_initialized


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
                self.sam_model = SAM("sam_b.pt")
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
        self.linear_speed = 0.2        # m/s for base translation
        self.angular_speed = 0.3       # rad/s for base rotation
        self.arm_speed = 0.05          # m or rad increment per update
        self.head_speed = 0.15         # rad increment per update
        self.wrist_speed = 0.15        # rad increment per update
        self.gripper_step = 0.03       # joint increment per click (open/close)
        self.command_smoothing_delay = DEFAULT_COMMAND_SMOOTH_DELAY_S

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
        content_layout.addWidget(control_widget, stretch=1)

        main_layout.addLayout(content_layout, stretch=1)

        # Status bar at bottom (fixed height)
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("QLabel { padding: 5px; background-color: #2c3e50; color: white; font-size: 11px; }")
        self.fps_label.setMaximumHeight(30)
        main_layout.addWidget(self.fps_label)

    def create_camera_widget(self):
        """Create camera feed widget"""
        widget = QGroupBox("Camera Feeds (ROS2)")
        layout = QVBoxLayout()

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

    def create_control_widget(self):
        """Create control panel widget"""
        widget = QWidget()
        widget.setMinimumWidth(300)  # Minimum width for controls
        widget.setMaximumWidth(500)  # Maximum width to prevent controls from getting too wide
        layout = QVBoxLayout()

        # Robot controls
        robot_controls = self.create_robot_controls()
        layout.addWidget(robot_controls)

        # Object list
        object_list = self.create_object_list()
        layout.addWidget(object_list, stretch=1)

        widget.setLayout(layout)
        return widget

    def _create_speed_slider(self, min_val, max_val, default_val, callback):
        """Create a compact speed slider row: [Slow] --slider-- [Fast] value_label."""
        row = QHBoxLayout()
        row.setSpacing(4)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        # Map default_val to slider position
        pos = int((default_val - min_val) / (max_val - min_val) * 100)
        slider.setValue(pos)
        slider.setFixedHeight(20)

        val_label = QLabel(f"{default_val:.3f}")
        val_label.setFixedWidth(45)
        val_label.setStyleSheet("QLabel { font-size: 10px; }")

        def on_change(v):
            speed = min_val + (max_val - min_val) * v / 100.0
            val_label.setText(f"{speed:.3f}")
            callback(speed)

        slider.valueChanged.connect(on_change)

        row.addWidget(QLabel("Spd"))
        row.addWidget(slider)
        row.addWidget(val_label)
        return row

    def _create_delay_slider(self, min_val, max_val, default_val, callback):
        """Create compact slider row for smoothing loop delay."""
        row = QHBoxLayout()
        row.setSpacing(4)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        pos = int((default_val - min_val) / max(1e-9, (max_val - min_val)) * 100)
        slider.setValue(max(0, min(100, pos)))
        slider.setFixedHeight(20)

        val_label = QLabel(f"{default_val:.3f}s")
        val_label.setFixedWidth(55)
        val_label.setStyleSheet("QLabel { font-size: 10px; }")

        def on_change(v):
            delay_s = min_val + (max_val - min_val) * v / 100.0
            val_label.setText(f"{delay_s:.3f}s")
            callback(delay_s)

        slider.valueChanged.connect(on_change)

        row.addWidget(QLabel("Delay"))
        row.addWidget(slider)
        row.addWidget(val_label)
        return row

    def create_robot_controls(self):
        """Create robot control buttons"""
        widget = QGroupBox("Robot Controls (ROS2)")
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

        def set_base_speed(v):
            self.linear_speed = v
            self.angular_speed = v * 1.5

        base_layout.addLayout(
            self._create_speed_slider(0.02, 0.50, self.linear_speed, set_base_speed), 2, 0, 1, 3)

        base_group.setLayout(base_layout)
        layout.addWidget(base_group)

        # Command smoothing controls (applies to all published qpos channels).
        smooth_group = QGroupBox("Command Smoothing")
        smooth_layout = QVBoxLayout()
        smooth_layout.setSpacing(3)

        def set_command_smoothing_delay(v):
            self.command_smoothing_delay = v
            self.ros_node.set_command_smoothing_delay(v)

        smooth_layout.addLayout(
            self._create_delay_slider(0.01, 0.20, self.command_smoothing_delay, set_command_smoothing_delay)
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
        """Pause/continue active action, or return after post-grasp hold."""
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
            # After grasp+lift hold, Continue means return to start.
            self.return_to_start()

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
        self.robot_controller.set_control('gripper', value)

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
        state = list(actual_qpos[:8]) if actual_qpos else []
        action = list(actual_qpos) if actual_qpos else []
        return {
            "timestamp": time.time(),
            "head_rgb": self.head_rgb if self.head_rgb is not None else None,
            "wrist_rgb": self.wrist_rgb if self.wrist_rgb is not None else None,
            "head_depth": self.depth_image if self.depth_image is not None else None,
            "wrist_depth": self.wrist_depth if self.wrist_depth is not None else None,
            "state": state,
            "action": action,
            "action_command": command_qpos,
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
        try:
            current = float(self.ros_node.qpos[7])
        except Exception:
            current = 0.0

        delta = self.gripper_step if direction > 0 else -self.gripper_step
        target = max(grip_min, min(grip_max, current + delta))
        self.robot_controller.set_control('gripper', target)
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
        status_color = "#27ae60" if self.ros_node.is_ready() else "#f39c12"
        self.fps_label.setText(f"FPS: {fps:.1f} | Robot: {robot_status} | Head: {head_shape} | Wrist: {wrist_shape}")
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

        # Reverse 90° CW rotation to get original camera pixel coords
        orig_col = grid_ys.astype(np.float64)
        orig_row = (H_orig - 1 - grid_xs).astype(np.float64)

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

        # Reverse 90° CW display rotation back to camera image coordinates
        orig_col = mask_ys.astype(np.float64)
        orig_row = (H_orig - 1 - mask_xs).astype(np.float64)

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
            rot_px = int(H_orig - 1 - orig_row)
            rot_py = int(orig_col)
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

    def _execute_approach(self, point_base, mode='reach', height_clearance=REACH_HEIGHT_CLEARANCE,
                          grasp_yaw=None, gripper_width=None, object_top_z=None,
                          grasp_mask=None, long_axis_angle=None,
                          preserve_existing_pre_action_state=False):
        """Shared approach logic for grasp and reach.

        Uses TF to look up the actual gripper tip position relative to the
        arm end, accounting for all wrist joint angles.  This ensures the
        gripper tip lands exactly at the target regardless of wrist orientation.

        object_top_z: maximum Z (in base_link) of the object surface.
                      The approach height is raised if needed so the gripper
                      body clears the object top during extension.

        Grasp sequence:
          1. Open gripper
          2. Base rotation to face object
          3. Adjust wrist yaw to object's principal axis (compensating for base rotation)
          4. Set grasp pitch (GRASP_PITCH_DEG) + wait for TF
          5. TF lookup for gripper offsets (with final yaw + pitch)
          6. Lift arm to approach height (clears object top)
          7. Extend arm to object distance
          8. Lower to grasp height
          9. Close gripper
         10. Lift + retract + rotate back

        Reach sequence:
          1. TF lookup (current wrist pose)
          2. Compute base rotation
          3. Rotate base
          4. Lift + extend
        """
        import time
        import math

        def _wait_with_pause(duration):
            """Sleep with pause/abort support."""
            # Shorten staged action waits now that command publishing is smoothed.
            duration = min(float(SCRIPT_STAGE_WAIT_CAP_S), max(0.0, float(duration)))
            end_t = time.time() + duration
            while time.time() < end_t:
                if self._is_abort_requested():
                    return False
                with self._action_lock:
                    st = self._action_state
                if st in ('paused', 'awaiting_confirm'):
                    time.sleep(0.05)
                    continue
                time.sleep(min(0.05, end_t - time.time()))
            return not self._is_abort_requested()

        def _wait_until_running():
            """Block until state returns to running or abort is requested."""
            while True:
                if self._is_abort_requested():
                    return False
                with self._action_lock:
                    st = self._action_state
                if st == 'running':
                    return True
                time.sleep(0.05)

        def _abort_and_return():
            """Abort current action and return to start pose."""
            if self._consume_skip_to_next_goal_request():
                print("Action aborted by user. Skipping to next queued goal...")
                self._set_action_state('idle')
                return
            print("Action aborted by user. Returning to start...")
            self._set_action_state('idle')
            if self._pre_action_state is not None:
                self.return_to_start()

        # Save current state so return_to_start() can restore it
        if (not preserve_existing_pre_action_state) or (self._pre_action_state is None):
            self._pre_action_state = {
                'arm_ext': self.ros_node.qpos[0],
                'lift': self.ros_node.qpos[1],
                'wrist_yaw': self.ros_node.qpos[2],
                'wrist_pitch': self.ros_node.qpos[3],
                'head_pan': self.ros_node.qpos[5],
                'head_tilt': self.ros_node.qpos[6],
                'rotation_applied': 0.0,  # updated after base rotate
            }
            yaw0 = self._get_current_base_yaw()
            if yaw0 is not None:
                self._pre_action_state['base_yaw_start'] = float(yaw0)

        target_x = point_base.point.x
        target_y = point_base.point.y
        target_z = point_base.point.z

        print(f"  Target in base_link: x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}")

        LIFT_MAX = self.ros_node.JOINT_LIMITS[1][1]  # 1.05
        LIFT_MIN = self.ros_node.JOINT_LIMITS[1][0]  # 0.08

        # --- Helper: look up arm pivot (link_arm_l0) in base_link ---
        def _get_arm_pivot():
            """Returns (arm_x, arm_y, arm_z, lift_to_z_offset) from TF."""
            _current_lift = self.ros_node.qpos[1]
            _current_ext = self.ros_node.qpos[0]
            _ax, _ay, _az, _lz_off = 0.0, 0.0, 0.0, 0.0
            try:
                arm_tf = self.ros_node.tf_buffer.lookup_transform(
                    'base_link', 'link_arm_l0',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
                _ax = arm_tf.transform.translation.x
                _ay = arm_tf.transform.translation.y
                _az = arm_tf.transform.translation.z
                _lz_off = _az - _current_lift
                print(f"  Arm pivot: x={_ax:.3f}, y={_ay:.3f}, z={_az:.3f}")
                print(f"  Lift-to-Z offset: {_lz_off:.3f}m "
                      f"(lift_joint={_current_lift:.3f}, arm_z={_az:.3f})")
            except Exception as e:
                print(f"  WARNING: Could not look up arm TF: {e}")
            return _ax, _ay, _az, _lz_off, _current_lift, _current_ext

        # --- Helper: look up gripper offsets via TF (with fallback) ---
        def _get_gripper_offsets():
            """Returns (GRIPPER_REACH, GRIPPER_LATERAL, GRIPPER_DROP)."""
            _GRIPPER_LATERAL = 0.0
            gripper_offset = self._lookup_gripper_offset_from_arm()
            if gripper_offset is not None:
                _GRIPPER_REACH = gripper_offset['reach']
                _GRIPPER_LATERAL = gripper_offset['lateral']
                _GRIPPER_DROP = gripper_offset['drop']
            else:
                _planned_yaw = self.ros_node.qpos[2]
                _planned_pitch = self.ros_node.qpos[3]
                _planned_roll = self.ros_node.qpos[4]
                _GRIPPER_REACH, _GRIPPER_DROP, _GRIPPER_LATERAL = self._compute_gripper_reach_drop(
                    _planned_yaw, _planned_pitch, _planned_roll)
                print(f"  (using analytical fallback)")
            print(f"  Gripper offsets: reach={_GRIPPER_REACH*100:.1f}cm, "
                  f"lateral={_GRIPPER_LATERAL*100:.1f}cm, drop={_GRIPPER_DROP*100:.1f}cm")
            return _GRIPPER_REACH, _GRIPPER_LATERAL, _GRIPPER_DROP

        # --- Helper: look up offset from arm end to any link frame ---
        def _get_link_offset_from_arm(frame_name):
            """Returns dict(reach, lateral, drop) for frame_name relative to link_arm_l0."""
            try:
                link_tf = self.ros_node.tf_buffer.lookup_transform(
                    'base_link', frame_name,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
                arm_tf = self.ros_node.tf_buffer.lookup_transform(
                    'base_link', 'link_arm_l0',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
                dx = link_tf.transform.translation.x - arm_tf.transform.translation.x
                dy = link_tf.transform.translation.y - arm_tf.transform.translation.y
                dz = link_tf.transform.translation.z - arm_tf.transform.translation.z
                return {'reach': -dy, 'lateral': dx, 'drop': -dz}
            except Exception as e:
                print(f"  WARNING: TF lookup failed for {frame_name}: {e}")
                return None

        # --- Helper: compute base rotation angle ---
        def _compute_base_rotation(arm_pivot_x, arm_pivot_y, gripper_lateral, dist):
            _dx = target_x - arm_pivot_x
            _dy = target_y - arm_pivot_y
            _angle = math.atan2(_dx, -_dy)
            # Lateral correction: the gripper tip may be offset from the arm
            # axis due to wrist yaw.  Adjust so the tip — not the arm — aims
            # at the target.
            if abs(gripper_lateral) > 0.001 and dist > 0.05:
                _lat_corr = math.atan2(gripper_lateral, dist)
                _angle -= _lat_corr
                print(f"  Lateral correction: {math.degrees(_lat_corr):.1f}° "
                      f"(gripper {gripper_lateral*100:.1f}cm off-axis)")
            return _angle

        # --- Helper: lift/extension ---
        def _lift_for_arm_z(desired_arm_z, lz_off):
            return max(LIFT_MIN, min(LIFT_MAX, desired_arm_z - lz_off))

        def _get_tip_z():
            """Current gripper tip Z in base_link from TF (link_grasp_center)."""
            try:
                tip_tf = self.ros_node.tf_buffer.lookup_transform(
                    'base_link', 'link_grasp_center',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
                return float(tip_tf.transform.translation.z)
            except Exception as e:
                print(f"  WARNING: Could not look up tip TF: {e}")
                return None

        def _lift_for_tip_z(desired_tip_z, fallback_drop, lz_off):
            """Compute lift joint command to place tip at desired_tip_z.

            Uses live TF tip Z for closed-loop mapping. Falls back to arm-Z model
            if tip TF is unavailable.
            """
            cur_lift = float(self.ros_node.qpos[1])
            cur_tip_z = _get_tip_z()
            if cur_tip_z is None:
                return _lift_for_arm_z(desired_tip_z + fallback_drop, lz_off)
            delta = desired_tip_z - cur_tip_z
            return max(LIFT_MIN, min(LIFT_MAX, cur_lift + delta))

        def _ext_for_distance(gap, cur_ext, gripper_reach):
            return max(0.0, min(0.5, cur_ext + gap - gripper_reach))

        # Track cumulative base rotation for head compensation
        _total_base_rotation = [0.0]  # mutable list so nested fn can modify

        # --- Helper: rotate base ---
        def _rotate_base(rotation_needed):
            if abs(rotation_needed) > 0.02:
                print(f"Rotating base ({math.degrees(rotation_needed):.1f}°)...")
                self.ros_node.qpos[9] = rotation_needed
                time.sleep(0.1)
                self.ros_node.qpos[9] = 0.0
                self._pre_action_state['rotation_applied'] = (
                    self._pre_action_state.get('rotation_applied', 0.0) + rotation_needed)
                _total_base_rotation[0] += rotation_needed
                # Compensate head pan: set to initial_pan - total_rotation
                # so the camera keeps looking at the same world direction
                initial_pan = self._pre_action_state['head_pan']
                target_pan = initial_pan - _total_base_rotation[0]
                print(f"  Head pan compensation: {target_pan:.3f} "
                      f"(initial={initial_pan:.3f}, total_rot={_total_base_rotation[0]:.3f})")
                self.robot_controller.set_control('head_pan', target_pan)
                if not _wait_with_pause(max(1.0, abs(rotation_needed) * 3.0)):
                    return False
            return True

        # ================================================================
        #  STEP 0 (shared): move to standardized pre-approach pose
        #  - retract arm fully
        #  - set wrist pitch from GRASP_PITCH_DEG
        #  - set wrist yaw to left joint limit
        # ================================================================
        if mode in ('reach', 'grasp'):
            arm_min, _arm_max = self.ros_node.JOINT_LIMITS[0]
            yaw_min, yaw_max = self.ros_node.JOINT_LIMITS[2]
            pitch_min, pitch_max = self.ros_node.JOINT_LIMITS[3]

            # GRASP_PITCH_DEG is from horizontal; convert to robot wrist_pitch convention.
            pitch_from_vertical = 90.0 - GRASP_PITCH_DEG
            prep_pitch_rad = -math.radians(pitch_from_vertical)
            prep_pitch_rad = max(pitch_min, min(pitch_max, prep_pitch_rad))

            # "Left limit" corresponds to +yaw on this UI/robot setup.
            yaw_left_limit = yaw_max

            print(f"\n--- Step 0: Pre-approach reset pose ---")
            print(f"  Retract arm -> {arm_min:.3f}, pitch -> {math.degrees(prep_pitch_rad):.1f}°, "
                  f"yaw(left limit) -> {math.degrees(yaw_left_limit):.1f}°")

            self.ros_node.qpos[0] = arm_min
            self.ros_node.qpos[3] = prep_pitch_rad
            self.ros_node.qpos[2] = yaw_left_limit
            if not _wait_with_pause(3.0):
                _abort_and_return()
                return

        # ================================================================
        #  REACH MODE — current wrist pose, no yaw/pitch changes
        # ================================================================
        if mode == 'reach':
            # TF lookup with current wrist angles
            GRIPPER_REACH, GRIPPER_LATERAL, GRIPPER_DROP = _get_gripper_offsets()
            arm_pivot_x, arm_pivot_y, arm_pivot_z, lift_to_z_offset, current_lift, current_ext = _get_arm_pivot()

            dx = target_x - arm_pivot_x
            dy = target_y - arm_pivot_y
            distance_to_object = math.sqrt(dx**2 + dy**2)

            rotation_needed = _compute_base_rotation(
                arm_pivot_x, arm_pivot_y, GRIPPER_LATERAL, distance_to_object)
            print(f"  Angle from arm: {math.degrees(rotation_needed):.1f}°")
            print(f"  Distance: {distance_to_object:.3f}m")

            if not _rotate_base(rotation_needed):
                _abort_and_return()
                return

            # Update target coordinates after base rotation
            if abs(rotation_needed) > 0.02:
                cos_r = math.cos(rotation_needed)
                sin_r = math.sin(rotation_needed)
                new_tx = target_x * cos_r + target_y * sin_r
                new_ty = -target_x * sin_r + target_y * cos_r
                target_x = new_tx
                target_y = new_ty

            # Collision-aware reach height
            # Use full requested clearance over object top (not just +3cm),
            # otherwise clicks on lower/deeper pixels can collide with table.
            TABLE_SAFETY = 0.04
            if object_top_z is not None:
                min_safe_arm_z = object_top_z + GRIPPER_DROP + height_clearance + TABLE_SAFETY
                print(f"  Object top: {object_top_z:.3f}m → min safe arm_z: {min_safe_arm_z:.3f}m")
            else:
                min_safe_arm_z = target_z + GRIPPER_DROP + height_clearance + TABLE_SAFETY

            desired_arm_z = max(
                target_z + GRIPPER_DROP + height_clearance + TABLE_SAFETY,
                min_safe_arm_z,
            )
            target_lift = _lift_for_arm_z(desired_arm_z, lift_to_z_offset)
            actual_arm_z = target_lift + lift_to_z_offset
            actual_clearance = actual_arm_z - GRIPPER_DROP - target_z
            print(f"Raising arm (desired clearance={height_clearance*100:.0f}cm, "
                  f"actual={actual_clearance*100:.0f}cm, lift={target_lift:.3f})...")
            if actual_clearance < height_clearance - 0.01:
                print(f"  WARNING: lift clamped to {LIFT_MAX}m, clearance reduced to {actual_clearance*100:.0f}cm")
            self.ros_node.qpos[1] = target_lift
            if not _wait_with_pause(1.5):
                _abort_and_return()
                return

            target_ext = _ext_for_distance(distance_to_object, current_ext, GRIPPER_REACH)
            print(f"Extending arm to {target_ext:.3f}m (cur={current_ext:.3f}, "
                  f"dist={distance_to_object:.3f}, reach={GRIPPER_REACH:.3f})...")
            self.ros_node.qpos[0] = target_ext
            if not _wait_with_pause(2.0):
                _abort_and_return()
                return

            print("Reach completed!")
            self._set_return_enabled(True)
            self._set_status("Reach completed!", "QLabel { color: green; font-size: 10px; }")

        # ================================================================
        #  GRASP MODE — reverse computation from grasp pose
        # ================================================================
        elif mode == 'grasp':
            # GRASP_PITCH_DEG = angle from horizontal plane
            # Robot wrist_pitch: 0 = straight down, negative = tilt forward
            # Angle from vertical = 90 - GRASP_PITCH_DEG
            pitch_from_vertical = 90.0 - GRASP_PITCH_DEG
            grasp_pitch_rad = -math.radians(pitch_from_vertical)
            # Use object-top-safe grasp surface when available, so grasp tip
            # does not scrape/collide with table due depth noise.
            GRASP_TIP_Z_MARGIN = GRASP_TIP_Z_MARGIN_M
            target_based_surface_z = target_z + GRASP_TARGET_Z_OFFSET_M
            if object_top_z is not None:
                grasp_surface_z = max(
                    target_based_surface_z,
                    object_top_z + GRASP_TIP_Z_MARGIN
                )
            else:
                grasp_surface_z = target_based_surface_z

            # ============================================================
            # PHASE 1: REVERSE COMPUTATION
            #   grasp_3d + principal_axis + pitch → wrist_position → base_angle
            #   yaw is computed LAST, after the arm is in position.
            # ============================================================
            print(f"\n--- Phase 1: Reverse computation from grasp pose ---")
            print(f"  Grasp point (tip target): ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")
            if object_top_z is not None:
                print(f"  Object top Z: {object_top_z:.3f} -> using grasp_surface_z={grasp_surface_z:.3f}")
            else:
                print(f"  No object_top_z; using target_z + fixed offset "
                      f"({GRASP_TARGET_Z_OFFSET_M:.3f}) -> grasp_surface_z={grasp_surface_z:.3f}")
            print(f"  Pitch: {GRASP_PITCH_DEG:.1f}° from horizontal ({pitch_from_vertical:.1f}° from vertical)")

            # (a) Stalk geometry — line from grasp point to wrist target
            #     in the vertical plane of the long axis, at GRASP_PITCH_DEG
            #     above horizontal. Length = STALK_LENGTH.
            # Geometry to the arm-connected wrist joint (wrist_yaw):
            #   wrist_yaw -> wrist_pitch: forward 0.019m, down 0.031m
            #   wrist_pitch -> grasp_center: STALK_LENGTH at grasp pitch
            STALK_LENGTH = GRASP_STALK_LENGTH_M   # wrist_pitch axis to grasp_center
            WRIST_YAW_TO_PITCH_FWD = 0.019
            PITCH_AXIS_DROP = 0.031
            pitch_horiz_rad = math.radians(GRASP_PITCH_DEG)
            stalk_horizontal = (
                STALK_LENGTH * math.cos(pitch_horiz_rad) + WRIST_YAW_TO_PITCH_FWD
            )  # XY distance from grasp point to wrist_yaw target
            stalk_vertical = STALK_LENGTH * math.sin(pitch_horiz_rad) + PITCH_AXIS_DROP  # Z distance
            print(f"  Stalk: horiz={stalk_horizontal*100:.1f}cm, vert={stalk_vertical*100:.1f}cm")

            # (b) Principal axis direction → wrist target position
            #     Project long axis to XY plane. From the grasp point,
            #     draw line along approach direction (toward robot) with
            #     horizontal length = stalk_horizontal, going upward by
            #     stalk_vertical. Endpoint = wrist target.
            arm_pivot_x, arm_pivot_y, arm_pivot_z, lift_to_z_offset, current_lift, current_ext = _get_arm_pivot()
            planned_yaw_target = None

            def _heading_target_from_wrist(wx, wy, px, py, lateral_off):
                """Convert desired wrist point to a heading point for base rotation.

                Wrist yaw pivot has a fixed lateral offset from arm line. For low-angle
                grasps this creates a circle-like side motion around wrist pivot; if we
                ignore it, base heading can be biased. We compensate by shifting the
                heading target along the arm-left normal.
                """
                vx = wx - px
                vy = wy - py
                d = math.hypot(vx, vy)
                if d < 1e-6:
                    return wx, wy
                # Left normal of arm heading vector in XY
                nx = -vy / d
                ny = vx / d
                return wx - lateral_off * nx, wy - lateral_off * ny

            def _yaw_lateral_scale(yaw_rad):
                """Scale lateral heading compensation by yaw magnitude.

                Needed compensation is strongest when yaw is near 0 (arm-aligned),
                and much smaller when yaw approaches +/-90 deg.
                """
                # |cos(yaw)|: 1.0 at 0 deg, 0.0 at 90 deg
                return abs(math.cos(yaw_rad))

            wrist_axis_offset_now = _get_link_offset_from_arm('link_wrist_yaw')
            if wrist_axis_offset_now is None:
                wrist_axis_offset_now = _get_link_offset_from_arm('link_wrist_pitch')
            nominal_wrist_reach = wrist_axis_offset_now['reach'] if wrist_axis_offset_now is not None else 0.031
            nominal_wrist_lateral = wrist_axis_offset_now['lateral'] if wrist_axis_offset_now is not None else 0.083
            base_lateral_for_heading = nominal_wrist_lateral + GRASP_LATERAL_TRIM_M
            print(f"  Wrist nominal offsets: reach={nominal_wrist_reach:.3f}, "
                  f"lateral={nominal_wrist_lateral:.3f}, "
                  f"heading_lateral_base(with trim)={base_lateral_for_heading:.3f}")

            if long_axis_angle is not None:
                # Two possible directions along the projected principal axis.
                # Evaluate both and choose the one that minimizes
                # base rotation and arm extension (as requested), while
                # preferring the toward-pivot direction and feasible yaw.
                dir1_x = math.cos(long_axis_angle)
                dir1_y = math.sin(long_axis_angle)
                candidates = [
                    (dir1_x, dir1_y),
                    (-dir1_x, -dir1_y),
                ]

                to_pivot_x = arm_pivot_x - target_x
                to_pivot_y = arm_pivot_y - target_y
                yaw_min, yaw_max = self.ros_node.JOINT_LIMITS[2]
                best = None

                def _clamp(v, vmin, vmax):
                    return max(vmin, min(vmax, v))

                def _yaw_violation(v):
                    if v < yaw_min:
                        return yaw_min - v
                    if v > yaw_max:
                        return v - yaw_max
                    return 0.0

                def _ang_diff(a, b):
                    return math.atan2(math.sin(a - b), math.cos(a - b))

                print(f"  Principal axis (XY): {math.degrees(long_axis_angle):.1f}° from +X")
                for i, (cx, cy) in enumerate(candidates, start=1):
                    wx = target_x + stalk_horizontal * cx
                    wy = target_y + stalk_horizontal * cy
                    # Use compensated heading point for base rotation planning
                    hx, hy = _heading_target_from_wrist(
                        wx, wy, arm_pivot_x, arm_pivot_y, base_lateral_for_heading
                    )
                    dx_c = hx - arm_pivot_x
                    dy_c = hy - arm_pivot_y
                    dist_c = math.sqrt((wx - arm_pivot_x)**2 + (wy - arm_pivot_y)**2)
                    rot_c = math.atan2(dx_c, -dy_c)
                    dot_c = cx * to_pivot_x + cy * to_pivot_y
                    pred_ext_c = max(0.0, min(0.5, current_ext + dist_c - nominal_wrist_reach))

                    # Predicted yaw branches in rotated frame for this side candidate.
                    sx = target_x - wx
                    sy = target_y - wy
                    cos_c = math.cos(rot_c)
                    sin_c = math.sin(rot_c)
                    sx_r = sx * cos_c + sy * sin_c
                    sy_r = -sx * sin_c + sy * cos_c
                    yaw_c1 = math.atan2(sx_r, -sy_r)
                    yaw_c2 = math.atan2(-sx_r, -sy_r)

                    rot_norm = abs(rot_c) / max(1e-6, math.radians(90.0))
                    ext_norm = abs(pred_ext_c - current_ext) / 0.5
                    toward_penalty = 0.2 if dot_c < 0.0 else 0.0
                    move_score = rot_norm + 0.8 * ext_norm + toward_penalty

                    # Choose between yaw branches with hard priority:
                    # (1) feasible yaw in limits, (2) less yaw violation,
                    # then (3) less movement (base/stretch), then alignment.
                    dnorm = math.hypot(sx_r, sy_r)
                    if dnorm > 1e-6:
                        ux = sx_r / dnorm
                        uy = sy_r / dnorm
                    else:
                        ux, uy = 0.0, -1.0

                    for b_idx, yaw_raw in enumerate((yaw_c1, yaw_c2), start=1):
                        yaw_cmd = _clamp(yaw_raw, yaw_min, yaw_max)
                        viol = _yaw_violation(yaw_raw)
                        feasible_flag = 0 if viol < 1e-6 else 1
                        px = math.sin(yaw_cmd)
                        py = -math.cos(yaw_cmd)
                        align = ux * px + uy * py
                        score_tuple = (
                            feasible_flag,            # prefer feasible yaw first
                            viol,                     # then least violation
                            move_score,               # then least base/stretch effort
                            1.0 - align,              # then better vector alignment
                            abs(_ang_diff(yaw_cmd, 0.0)),  # tie-break: smaller |yaw|
                        )
                        print(f"    cand{i}.b{b_idx}: dir=({cx:.3f},{cy:.3f}) dot={dot_c:.3f} "
                              f"rot={math.degrees(rot_c):.1f}° "
                              f"ext={pred_ext_c:.3f} "
                              f"yaw_raw={math.degrees(yaw_raw):.1f}° "
                              f"yaw_cmd={math.degrees(yaw_cmd):.1f}° "
                              f"viol={math.degrees(viol):.1f}° score={score_tuple}")
                        if best is None or score_tuple < best[0]:
                            best = (score_tuple, cx, cy, wx, wy, yaw_cmd, yaw_raw, i, b_idx)

                _, approach_dir_x, approach_dir_y, wrist_x, wrist_y, planned_yaw_target, planned_yaw_raw, sel_i, sel_b = best
                print(f"  Selected approach dir: ({approach_dir_x:.3f}, {approach_dir_y:.3f}) "
                      f"from cand{sel_i}.b{sel_b}, planned_yaw={math.degrees(planned_yaw_target):.1f}° "
                      f"(raw={math.degrees(planned_yaw_raw):.1f}°)")
            else:
                # No orientation info — put wrist directly toward arm pivot
                dx_raw = arm_pivot_x - target_x
                dy_raw = arm_pivot_y - target_y
                dist_raw = math.sqrt(dx_raw**2 + dy_raw**2)
                if dist_raw > 0.01:
                    approach_dir_x = dx_raw / dist_raw
                    approach_dir_y = dy_raw / dist_raw
                else:
                    approach_dir_x, approach_dir_y = 0.0, 1.0
                wrist_x = target_x + stalk_horizontal * approach_dir_x
                wrist_y = target_y + stalk_horizontal * approach_dir_y

            wrist_z = grasp_surface_z + stalk_vertical
            print(f"  → Wrist target: ({wrist_x:.3f}, {wrist_y:.3f}, {wrist_z:.3f})")

            # (c) Base rotation: aim arm at wrist target
            dx_w = wrist_x - arm_pivot_x
            dy_w = wrist_y - arm_pivot_y
            distance_to_wrist = math.sqrt(dx_w**2 + dy_w**2)
            yaw_for_heading = planned_yaw_target if planned_yaw_target is not None else 0.0
            lateral_for_heading = base_lateral_for_heading * _yaw_lateral_scale(yaw_for_heading)
            head_tx, head_ty = _heading_target_from_wrist(
                wrist_x, wrist_y, arm_pivot_x, arm_pivot_y, lateral_for_heading
            )
            rotation_needed = math.atan2(head_tx - arm_pivot_x, -(head_ty - arm_pivot_y))
            print(f"  → Base rotation: {math.degrees(rotation_needed):.1f}°")
            print(f"  → Distance to wrist: {distance_to_wrist:.3f}m")
            print(f"  → Heading target (lateral compensated): ({head_tx:.3f}, {head_ty:.3f}), "
                  f"yaw={math.degrees(yaw_for_heading):.1f}°, "
                  f"lateral_used={lateral_for_heading:.3f}")

            # ============================================================
            # PHASE 2: SEQUENTIAL EXECUTION (wait for each step)
            # ============================================================
            print(f"\n--- Phase 2: Sequential execution ---")

            # --- Step 1: Open gripper ---
            open_width = gripper_width if gripper_width is not None else 0.5
            approach_width = min(self.ros_node.JOINT_LIMITS[7][1], open_width * 1.3)
            print(f"Step 1: Opening gripper ({approach_width:.3f})...")
            self.robot_controller.set_control('gripper', approach_width)
            if not _wait_with_pause(1.0):
                _abort_and_return()
                return

            # --- Step 2: Set pitch ---
            print(f"Step 2: Setting pitch to {GRASP_PITCH_DEG:.1f}° forward...")
            self.ros_node.qpos[3] = grasp_pitch_rad
            if not _wait_with_pause(1.5):
                _abort_and_return()
                return

            # --- Step 3: Rotate base ---
            print(f"Step 3: Rotating base {math.degrees(rotation_needed):.1f}°...")
            if not _rotate_base(rotation_needed):
                _abort_and_return()
                return
            if not _wait_with_pause(0.5):
                _abort_and_return()
                return

            # Transform target & wrist coords to rotated frame
            if abs(rotation_needed) > 0.02:
                cos_r = math.cos(rotation_needed)
                sin_r = math.sin(rotation_needed)
                new_tx = target_x * cos_r + target_y * sin_r
                new_ty = -target_x * sin_r + target_y * cos_r
                new_wx = wrist_x * cos_r + wrist_y * sin_r
                new_wy = -wrist_x * sin_r + wrist_y * cos_r
                target_x, target_y = new_tx, new_ty
                wrist_x, wrist_y = new_wx, new_wy
                print(f"  Rotated target: ({target_x:.3f}, {target_y:.3f})")
                print(f"  Rotated wrist:  ({wrist_x:.3f}, {wrist_y:.3f})")

            # --- Step 4: TF lookup (with pitch set, before yaw) ---
            print(f"Step 4: TF lookup for offsets...")
            GRIPPER_REACH, GRIPPER_LATERAL, GRIPPER_DROP = _get_gripper_offsets()
            arm_pivot_x, arm_pivot_y, arm_pivot_z, lift_to_z_offset, current_lift, current_ext = _get_arm_pivot()

            # --- Step 4b: Residual base alignment using live TF ---
            # Compensates open-loop rotate_by errors and frame mismatch.
            wrist_axis_live = _get_link_offset_from_arm('link_wrist_yaw')
            if wrist_axis_live is None:
                wrist_axis_live = _get_link_offset_from_arm('link_wrist_pitch')
            live_lateral = wrist_axis_live['lateral'] if wrist_axis_live is not None else nominal_wrist_lateral
            live_heading_lateral = (live_lateral + GRASP_LATERAL_TRIM_M) * _yaw_lateral_scale(yaw_for_heading)
            align_tx, align_ty = _heading_target_from_wrist(
                wrist_x, wrist_y, arm_pivot_x, arm_pivot_y, live_heading_lateral
            )
            dx_align = align_tx - arm_pivot_x
            dy_align = align_ty - arm_pivot_y
            residual_rot_raw = math.atan2(dx_align, -dy_align)
            residual_rot = residual_rot_raw * GRASP_RESIDUAL_ROT_GAIN
            max_residual = math.radians(GRASP_RESIDUAL_ROT_MAX_DEG)
            residual_rot = max(-max_residual, min(max_residual, residual_rot))
            if abs(residual_rot) > 0.01:
                print(f"Step 4b: Residual base correction raw={math.degrees(residual_rot_raw):.2f}°, "
                      f"applied={math.degrees(residual_rot):.2f}° "
                      f"(gain={GRASP_RESIDUAL_ROT_GAIN:.2f}, "
                      f"live_heading_lateral={live_heading_lateral:.3f})...")
                if not _rotate_base(residual_rot):
                    _abort_and_return()
                    return
                rotation_needed += residual_rot
                if not _wait_with_pause(0.3):
                    _abort_and_return()
                    return
                # Keep target/wrist in the currently-rotated frame
                cos_r2 = math.cos(residual_rot)
                sin_r2 = math.sin(residual_rot)
                new_tx2 = target_x * cos_r2 + target_y * sin_r2
                new_ty2 = -target_x * sin_r2 + target_y * cos_r2
                new_wx2 = wrist_x * cos_r2 + wrist_y * sin_r2
                new_wy2 = -wrist_x * sin_r2 + wrist_y * cos_r2
                target_x, target_y = new_tx2, new_ty2
                wrist_x, wrist_y = new_wx2, new_wy2
                GRIPPER_REACH, GRIPPER_LATERAL, GRIPPER_DROP = _get_gripper_offsets()
                arm_pivot_x, arm_pivot_y, arm_pivot_z, lift_to_z_offset, current_lift, current_ext = _get_arm_pivot()
                print(f"  Residual corrected target: ({target_x:.3f}, {target_y:.3f})")
                print(f"  Residual corrected wrist:  ({wrist_x:.3f}, {wrist_y:.3f})")

            # Distance from arm pivot to wrist target.
            # IMPORTANT: arm extension moves only along arm axis (-Y in rotated frame),
            # so use projected reach on -Y for extension, not Euclidean XY distance.
            dx_w2 = wrist_x - arm_pivot_x
            dy_w2 = wrist_y - arm_pivot_y
            distance_to_wrist = math.sqrt(dx_w2**2 + dy_w2**2)
            desired_wrist_reach = -(wrist_y - arm_pivot_y)  # projection on arm (-Y) axis
            # Reach from arm pivot to wrist_yaw axis (the intended endpoint
            # of reverse geometry line).
            wrist_axis_offset = _get_link_offset_from_arm('link_wrist_yaw')
            if wrist_axis_offset is None:
                wrist_axis_offset = _get_link_offset_from_arm('link_wrist_pitch')
            if wrist_axis_offset is not None:
                wrist_horiz_reach = wrist_axis_offset['reach']
                print(f"  Wrist axis offset TF: reach={wrist_horiz_reach:.3f}, "
                      f"lateral={wrist_axis_offset['lateral']:.3f}, drop={wrist_axis_offset['drop']:.3f}")
            else:
                # Fallback: infer from current gripper reach minus modeled stalk horizontal
                wrist_horiz_reach = GRIPPER_REACH - stalk_horizontal
                print(f"  Wrist axis offset fallback from gripper: {wrist_horiz_reach:.3f}")
            print(f"  Distance to wrist: {distance_to_wrist:.3f}m")
            print(f"  Wrist horiz reach (arm→wrist): {wrist_horiz_reach:.3f}m")
            print(f"  Wrist relative to pivot: dx={dx_w2:.3f}, dy={dy_w2:.3f}, "
                  f"desired_reach(-Y)={desired_wrist_reach:.3f}")

            # --- Step 5: Lift arm above object ---
            # User-requested order: always lift to max first, then approach.
            approach_lift = LIFT_MAX
            print(f"Step 5: Lifting arm to MAX first (lift={approach_lift:.3f})...")
            self.ros_node.qpos[1] = approach_lift
            if not _wait_with_pause(5.0):
                _abort_and_return()
                return

            # --- Step 6: Extend arm to wrist target ---
            # Use arm-axis projection (desired_wrist_reach) so lateral/base-rotation
            # error does not cause over-extension.
            target_ext = max(
                0.0, min(0.5, current_ext + desired_wrist_reach - wrist_horiz_reach + GRASP_REACH_TRIM_M)
            )
            print(f"Step 6: Extending arm to {target_ext:.3f}m "
                  f"(desired_reach={desired_wrist_reach:.3f}, "
                  f"wrist_reach={wrist_horiz_reach:.3f}, cur_ext={current_ext:.3f}, "
                  f"reach_trim={GRASP_REACH_TRIM_M:.3f})...")
            self.ros_node.qpos[0] = target_ext
            if not _wait_with_pause(5.0):
                _abort_and_return()
                return

            # --- Step 6b: Closed-loop wrist reach correction in rotated frame ---
            # Similar to base residual correction: compare desired vs actual in
            # the post-rotation frame, then apply damped incremental correction.
            for i in range(max(1, GRASP_REACH_CORR_ITERS)):
                arm_pivot_x, arm_pivot_y, arm_pivot_z, lift_to_z_offset, current_lift, current_ext = _get_arm_pivot()
                wrist_axis_offset = _get_link_offset_from_arm('link_wrist_yaw')
                if wrist_axis_offset is None:
                    wrist_axis_offset = _get_link_offset_from_arm('link_wrist_pitch')
                if wrist_axis_offset is None:
                    print("Step 6b: No wrist-axis TF available for closed-loop reach correction.")
                    break

                desired_reach_i = -(wrist_y - arm_pivot_y) + GRASP_REACH_TRIM_M
                actual_reach_i = wrist_axis_offset['reach']
                reach_err_i = desired_reach_i - actual_reach_i
                lateral_des_i = wrist_x - arm_pivot_x
                lateral_err_i = lateral_des_i - wrist_axis_offset['lateral']

                print(f"Step 6b[{i+1}/{GRASP_REACH_CORR_ITERS}]: "
                      f"desired_reach={desired_reach_i:.3f}, actual_reach={actual_reach_i:.3f}, "
                      f"reach_err={reach_err_i:+.3f}, lateral_err={lateral_err_i:+.3f}")

                if abs(reach_err_i) < GRASP_REACH_CORR_THRESH_M:
                    print("  Reach correction within threshold.")
                    break

                ext_step = max(
                    -GRASP_REACH_CORR_MAX_STEP_M,
                    min(GRASP_REACH_CORR_MAX_STEP_M, GRASP_REACH_CORR_GAIN * reach_err_i)
                )
                refined_ext = max(0.0, min(0.5, current_ext + ext_step))
                if abs(refined_ext - current_ext) < 0.003:
                    print("  Reach correction step too small after clamp; stopping refinement.")
                    break

                print(f"  Applying reach correction: ext {current_ext:.3f} -> {refined_ext:.3f} "
                      f"(step={ext_step:+.3f})")
                self.ros_node.qpos[0] = refined_ext
                if not _wait_with_pause(1.0):
                    _abort_and_return()
                    return

            # --- Step 7: Compute & set yaw (arm is now in position) ---
            # After base rotation, the arm direction is -Y = (0, -1) in
            # the rotated frame.  The stalk (wrist → grasp) projected to
            # XY gives the direction the gripper needs to point.
            #
            # The user's method:
            #   - Take the line from wrist target to grasp point on XY plane
            #   - Compute the angle between this line and the arm direction (-Y)
            #   - That angle = wrist yaw
            stalk_dx = target_x - wrist_x   # wrist → grasp, X component
            stalk_dy = target_y - wrist_y   # wrist → grasp, Y component
            # Arm direction in rotated frame = (0, -1)
            # Angle of stalk from -Y. There is a sign ambiguity in practice.
            # Resolve it by choosing the yaw whose forward model aligns best
            # with desired stalk direction (wrist -> grasp) on XY.
            yaw_a = math.atan2(stalk_dx, -stalk_dy)
            yaw_b = math.atan2(-stalk_dx, -stalk_dy)
            yaw_min, yaw_max = self.ros_node.JOINT_LIMITS[2]

            def _clamp(v, vmin, vmax):
                return max(vmin, min(vmax, v))

            def _ang_diff(a, b):
                return math.atan2(math.sin(a - b), math.cos(a - b))

            cand_a = _clamp(yaw_a, yaw_min, yaw_max)
            cand_b = _clamp(yaw_b, yaw_min, yaw_max)

            # If planning stage found a feasible target branch, follow it.
            # This allows higher base/stretch when needed to avoid yaw limits.
            if planned_yaw_target is not None:
                da = abs(_ang_diff(cand_a, planned_yaw_target))
                db = abs(_ang_diff(cand_b, planned_yaw_target))
                choose_a = da <= db
                score_a = -da
                score_b = -db
            else:
                dnorm = math.hypot(stalk_dx, stalk_dy)
                if dnorm < 1e-6:
                    score_a = score_b = 0.0
                else:
                    ux = stalk_dx / dnorm
                    uy = stalk_dy / dnorm
                    # Forward model direction for each yaw candidate
                    pax = math.sin(cand_a)
                    pay = -math.cos(cand_a)
                    pbx = math.sin(cand_b)
                    pby = -math.cos(cand_b)
                    score_a = ux * pax + uy * pay
                    score_b = ux * pbx + uy * pby
                choose_a = score_a >= score_b
            clamped_yaw = cand_a if choose_a else cand_b
            computed_yaw = yaw_a if choose_a else yaw_b
            print(f"Step 7: Setting yaw to {math.degrees(clamped_yaw):.1f}° "
                  f"(computed={math.degrees(computed_yaw):.1f}°, "
                  f"align_a={score_a:.3f}, align_b={score_b:.3f}, "
                  f"stalk dir: dx={stalk_dx:.3f}, dy={stalk_dy:.3f})...")
            self.ros_node.qpos[2] = clamped_yaw
            if not _wait_with_pause(1.5):
                _abort_and_return()
                return

            # --- Step 8: Re-lookup TF with final yaw+pitch for accurate drop ---
            print(f"Step 8: TF re-lookup with final orientation...")
            GRIPPER_REACH, GRIPPER_LATERAL, GRIPPER_DROP = _get_gripper_offsets()

            # Tip-Z targets (use live tip TF for lift mapping, not only GRIPPER_DROP model).
            desired_final_tip_z = grasp_surface_z
            if object_top_z is not None:
                desired_final_tip_z = max(desired_final_tip_z, object_top_z + GRASP_TIP_Z_MARGIN)
            cur_tip_z = _get_tip_z()
            if cur_tip_z is not None:
                print(f"  Tip Z now: {cur_tip_z:.3f}")
            print(f"  Desired final tip Z: {desired_final_tip_z:.3f}")

            # --- Step 8.5: Hover slightly above grasp, then confirm ---
            requested_hover = GRASP_PRELOWER_VERIFY_TIP_MARGIN_M
            desired_hover_tip_z = desired_final_tip_z + requested_hover
            prelower_lift = _lift_for_tip_z(
                desired_hover_tip_z, fallback_drop=GRIPPER_DROP, lz_off=lift_to_z_offset
            )
            print(f"Step 8.5: Moving to pre-grasp hover (lift={prelower_lift:.3f}, "
                  f"tip_margin={GRASP_PRELOWER_VERIFY_TIP_MARGIN_M*100:.0f}cm)...")
            self.ros_node.qpos[1] = prelower_lift
            if not _wait_with_pause(2.0):
                _abort_and_return()
                return
            hover_tip_z = _get_tip_z()
            if hover_tip_z is not None:
                print(f"  Hover tip z={hover_tip_z:.3f} (target={desired_hover_tip_z:.3f})")

            self._set_action_state('awaiting_confirm')
            self._set_return_enabled(True)
            self._set_status(
                "Hover alignment check. Press Continue to lower for grasp or Return to abort.",
                "QLabel { color: orange; font-size: 10px; }"
            )
            if not _wait_until_running():
                _abort_and_return()
                return

            # --- Step 9: Lower to grasp height ---
            grasp_lift = _lift_for_tip_z(
                desired_final_tip_z, fallback_drop=GRIPPER_DROP, lz_off=lift_to_z_offset
            )
            if object_top_z is not None:
                print(f"  Final tip target={desired_final_tip_z:.3f}; required >= "
                      f"{(object_top_z + GRASP_TIP_Z_MARGIN):.3f}")
            else:
                print(f"  Final tip target={desired_final_tip_z:.3f} (no object_top_z)")
            print(f"Step 9: Lowering to grasp (lift={grasp_lift:.3f})...")
            self.ros_node.qpos[1] = grasp_lift
            if not _wait_with_pause(5.0):
                _abort_and_return()
                return
            final_tip_z = _get_tip_z()
            if final_tip_z is not None:
                print(f"  Final tip z={final_tip_z:.3f}")

            # --- User confirmation before grasp ---
            self._set_action_state('awaiting_confirm')
            self._set_return_enabled(True)
            self._set_status(
                "Please verify grasp position. Press Continue to grasp or Return to abort.",
                "QLabel { color: orange; font-size: 10px; }"
            )
            if not _wait_until_running():
                _abort_and_return()
                return

            # --- Step 10: Close gripper ---
            close_to = 0.0
            if gripper_width is not None:
                close_to = max(0.0, gripper_width * 0.5)
            grip_min, grip_max = self.ros_node.JOINT_LIMITS[7]
            close_to_extra = max(grip_min, min(grip_max, close_to - GRASP_CLOSE_EXTRA_M))
            print(f"Step 10: Closing gripper ({close_to_extra:.3f}) "
                  f"[base={close_to:.3f}, extra={GRASP_CLOSE_EXTRA_M:.3f}]...")
            self.robot_controller.set_control('gripper', close_to_extra)
            if not _wait_with_pause(5.5):
                _abort_and_return()
                return

            # --- User confirmation after grasp close, before lift ---
            self._set_action_state('awaiting_confirm')
            self._set_return_enabled(True)
            self._set_status(
                "Grasp closed. Verify grip. Press Continue to lift or Return to abort.",
                "QLabel { color: orange; font-size: 10px; }"
            )
            if not _wait_until_running():
                _abort_and_return()
                return

            # --- Step 11: Lift to max ---
            print(f"Step 11: Lifting to max ({LIFT_MAX}m)...")
            self.ros_node.qpos[1] = LIFT_MAX
            if not _wait_with_pause(3.5):
                _abort_and_return()
                return

            print("Grasp completed and lifted.")
            self._set_return_enabled(True)
            if self._goal_sequence_has_next():
                self._set_status("Grasp completed and lifted. Press Go To Next Goal or Return.",
                                 "QLabel { color: green; font-size: 10px; }")
            else:
                self._set_status("Grasp completed and lifted. Press Return to go back.",
                                 "QLabel { color: green; font-size: 10px; }")
            self._set_action_state('idle')
            self._update_next_goal_button_state()
            return

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

    def _prepare_reach_goal_from_pixel(self, px, py):
        point_base, depth = self._get_3d_point_at_pixel(px, py)
        if point_base is None:
            return None, "No valid depth at clicked point"

        object_top_z = None
        seg = self._find_segment_at_pixel(px, py)
        if seg is not None:
            object_top_z = self._compute_object_top_z(seg['mask'])

        goal = {
            "kind": "reach",
            "px": int(px),
            "py": int(py),
            "depth": float(depth),
            "point_xyz": (float(point_base.point.x), float(point_base.point.y), float(point_base.point.z)),
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
        _grasp_yaw, rect_info = self._compute_grasp_orientation(mask, px, py)
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

        goal = {
            "kind": "grasp",
            "px": int(px),
            "py": int(py),
            "depth": float(depth),
            "point_xyz": (float(point_base.point.x), float(point_base.point.y), float(point_base.point.z)),
            "object_top_z": float(object_top_z),
            "gripper_width": None if gripper_width is None else float(gripper_width),
            "long_axis_angle": None if long_axis_angle is None else float(long_axis_angle),
            "grasp_mask": np.array(mask, copy=True),
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

        point_base = self._point_from_xyz(goal["point_xyz"])
        kind = goal["kind"]
        if kind == "grasp":
            self._set_status("Executing queued grasp sequence...", "QLabel { color: blue; font-size: 10px; }")
        else:
            self._set_status("Executing queued reach sequence...", "QLabel { color: blue; font-size: 10px; }")

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
        if mask is not None:
            _grasp_yaw, rect_info = self._compute_grasp_orientation(mask, px, py)
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
        if st in ('running', 'paused', 'awaiting_confirm'):
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
                LIFT_MAX = self.ros_node.JOINT_LIMITS[1][1]
                RETRACT_EXT = 0.0

                # Step 1: Lift to max first for safe retraction/rotation
                print(f"Lifting to max height ({LIFT_MAX:.3f}m) before return...")
                self.ros_node.qpos[1] = LIFT_MAX
                _stage_sleep(2.0)

                # Step 2: Retract arm
                print(f"Retracting arm to {RETRACT_EXT:.3f}m...")
                self.ros_node.qpos[0] = RETRACT_EXT
                _stage_sleep(2.0)

                # Step 3: Rotate base back (if it wasn't already undone by grasp)
                if abs(rotation) > 0.02:
                    print(f"Rotating base back ({math.degrees(-rotation):.1f}°)...")
                    self.ros_node.qpos[9] = -rotation
                    time.sleep(0.1)
                    self.ros_node.qpos[9] = 0.0
                    _stage_sleep(max(1.0, abs(rotation) * 3.0))

                # Step 4: Restore wrist yaw and pitch
                if 'wrist_yaw' in state:
                    print(f"Restoring wrist yaw={state['wrist_yaw']:.3f}")
                    self.ros_node.qpos[2] = state['wrist_yaw']
                if 'wrist_pitch' in state:
                    print(f"Restoring wrist pitch={state['wrist_pitch']:.3f}")
                    self.ros_node.qpos[3] = state['wrist_pitch']
                _stage_sleep(0.5)

                # Step 5: Restore head pan/tilt
                print(f"Restoring head pan={state['head_pan']:.3f}, tilt={state['head_tilt']:.3f}")
                self.ros_node.qpos[5] = state['head_pan']
                self.ros_node.qpos[6] = state['head_tilt']
                _stage_sleep(1.0)

                # Step 6: Set arm to ready height (0.7m)
                READY_HEIGHT = 0.7
                print(f"Setting arm to ready height ({READY_HEIGHT}m)...")
                self.ros_node.qpos[1] = READY_HEIGHT
                _stage_sleep(1.5)

                self._pre_action_state = None
                self.queued_sequence_started = False
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


def main():
    """Main function"""
    print("="*60)
    print("Robot Teleoperation UI with SAM (ROS2 Version)")
    print("="*60)

    # Initialize ROS2
    rclpy.init()

    # Create camera node (dedicated, no other activity)
    camera_node = CameraNode()

    # Create robot control node (publishers, TF, timers — no camera subs)
    ros_node = RobotROSNode(camera_node)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create UI
    ui = RobotTeleopUI(ros_node)
    ui.show()

    print("\nUI Started!", flush=True)
    print("- Waiting for camera feeds from ROS topics", flush=True)
    print("- Use control buttons to move robot", flush=True)
    print("- Click 'Run SAM Segmentation' to detect objects", flush=True)
    print("- Select objects and use action buttons", flush=True)

    import threading

    # Camera node gets its own executor thread
    camera_executor = MultiThreadedExecutor()
    camera_executor.add_node(camera_node)

    def spin_camera():
        try:
            camera_executor.spin()
        except Exception as e:
            print(f"Camera executor error: {e}")

    camera_thread = threading.Thread(target=spin_camera, daemon=True)
    camera_thread.start()

    # Control node gets its own executor thread
    control_executor = MultiThreadedExecutor()
    control_executor.add_node(ros_node)

    def spin_control():
        try:
            control_executor.spin()
        except Exception as e:
            print(f"Control executor error: {e}")

    control_thread = threading.Thread(target=spin_control, daemon=True)
    control_thread.start()

    # Run Qt application
    exit_code = app.exec()

    # Cleanup
    print("\nCleaning up...")
    ui.robot_controller.stop()
    camera_node.destroy_node()
    ros_node.destroy_node()
    rclpy.shutdown()
    print("Done!")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
