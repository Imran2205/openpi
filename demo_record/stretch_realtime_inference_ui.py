#!/usr/bin/env python3
"""Simple real-time inference UI for Stretch.

Features:
- Subscribes to head/wrist RGB and joint feedback from ROS2.
- Loads a trained OpenPI checkpoint and runs prompt-conditioned inference.
- Displays camera feeds, measured qpos, and predicted actions.
- Sends predicted actions to /joint_pose_cmd.
- Abort button immediately publishes a hold/stop command.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import threading
import time
from pathlib import Path
from typing import Any

# Must be set before importing cv2/PyQt on some systems.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import jax.numpy as jnp
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

import openpi.models.model as _model
from openpi.policies import policy as _policy
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class StretchOutputs(transforms.DataTransformFn):
    """Inference output parser for Stretch actions."""

    action_dim: int = 10

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


def create_stretch_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: Path | str,
    *,
    action_dim: int = 10,
    default_prompt: str | None = None,
) -> _policy.Policy:
    """Create a trained policy, but keep Stretch output dimensions (10)."""
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        pytorch_device = "cuda"
        try:
            import torch

            if not torch.cuda.is_available():
                pytorch_device = "cpu"
        except Exception:
            pytorch_device = "cpu"
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
        pytorch_device = None

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("data_config.asset_id is required to load norm stats")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *data_config.repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            StretchOutputs(action_dim=action_dim),
            *data_config.repack_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device,
    )


class StretchROSNode(Node):
    """ROS bridge for camera, qpos feedback and command publishing."""

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

    def __init__(self):
        super().__init__("stretch_realtime_inference_ui")
        self.bridge = CvBridge()

        self.head_rgb: np.ndarray | None = None
        self.wrist_rgb: np.ndarray | None = None
        self.joint_states: JointState | None = None
        self.odom: Odometry | None = None
        self.actual_qpos: list[float] | None = None

        self.create_subscription(Image, "/camera/color/image_raw", self._head_rgb_cb, 1)
        self.create_subscription(Image, "/gripper_camera/color/image_rect_raw", self._wrist_rgb_cb, 1)
        self.create_subscription(JointState, "/stretch/joint_states", self._joint_cb, 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self.joint_pub = self.create_publisher(Float64MultiArray, "/joint_pose_cmd", 10)

    def _head_rgb_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8").copy()
            self.head_rgb = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        except CvBridgeError:
            return

    def _wrist_rgb_cb(self, msg: Image):
        try:
            self.wrist_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except CvBridgeError:
            return

    def _odom_cb(self, msg: Odometry):
        self.odom = msg
        if self.actual_qpos is not None and len(self.actual_qpos) >= 10:
            self.actual_qpos[8] = float(msg.twist.twist.linear.x)
            self.actual_qpos[9] = float(msg.twist.twist.angular.z)

    def _joint_cb(self, msg: JointState):
        self.joint_states = msg
        measured = self._extract_qpos(msg)
        if measured is not None:
            self.actual_qpos = measured

    def _extract_qpos(self, msg: JointState) -> list[float] | None:
        try:
            arm_lift = msg.position[msg.name.index("joint_lift")]
            arm_extension = 4.0 * msg.position[msg.name.index("joint_arm_l0")]
            wrist_yaw = msg.position[msg.name.index("joint_wrist_yaw")]
            wrist_pitch = msg.position[msg.name.index("joint_wrist_pitch")]
            wrist_roll = msg.position[msg.name.index("joint_wrist_roll")]
            head_pan = msg.position[msg.name.index("joint_head_pan")]
            head_tilt = msg.position[msg.name.index("joint_head_tilt")]
            gripper = msg.position[msg.name.index("joint_gripper_finger_left")]
            base_lin = float(self.odom.twist.twist.linear.x) if self.odom is not None else 0.0
            base_ang = float(self.odom.twist.twist.angular.z) if self.odom is not None else 0.0
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
        except Exception:
            return None

    def get_actual_qpos(self) -> list[float]:
        if self.actual_qpos is not None:
            return list(self.actual_qpos)
        return [0.0] * 10

    def publish_joint_pose_cmd(self, qpos_cmd: list[float]) -> None:
        msg = Float64MultiArray()
        msg.data = qpos_cmd
        self.joint_pub.publish(msg)

    def publish_hold_stop(self) -> None:
        q = self.get_actual_qpos()
        q[8] = 0.0
        q[9] = 0.0
        for _ in range(3):
            self.publish_joint_pose_cmd(q)
            time.sleep(0.02)


class InferenceUI(QMainWindow):
    status_signal = pyqtSignal(str)
    predicted_signal = pyqtSignal(str)
    qpos_signal = pyqtSignal(str)

    def __init__(self, ros_node: StretchROSNode, policy: _policy.Policy, step_hz: float):
        super().__init__()
        self.ros_node = ros_node
        self.policy = policy
        self.step_hz = max(1e-3, float(step_hz))

        self._abort = threading.Event()
        self._worker: threading.Thread | None = None

        self._init_ui()

        self.status_signal.connect(self.status_label.setText)
        self.predicted_signal.connect(self.predicted_text.setPlainText)
        self.qpos_signal.connect(self.qpos_text.setPlainText)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)

    def _init_ui(self):
        self.setWindowTitle("Stretch Real-time Policy UI")
        self.resize(1300, 820)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        camera_group = QGroupBox("Cameras")
        cam_layout = QGridLayout(camera_group)
        self.head_label = QLabel("Waiting for head camera...")
        self.head_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.head_label.setStyleSheet("background: #000; color: #fff;")
        self.wrist_label = QLabel("Waiting for wrist camera...")
        self.wrist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wrist_label.setStyleSheet("background: #111; color: #ddd;")
        cam_layout.addWidget(self.head_label, 0, 0)
        cam_layout.addWidget(self.wrist_label, 0, 1)
        layout.addWidget(camera_group, stretch=3)

        control_group = QGroupBox("Policy Control")
        ctl = QVBoxLayout(control_group)

        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Prompt:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g. pick up the red object")
        prompt_row.addWidget(self.prompt_input)
        ctl.addLayout(prompt_row)

        btn_row = QHBoxLayout()
        self.execute_btn = QPushButton("Execute")
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setStyleSheet("QPushButton { background: #d32f2f; color: white; }")
        self.execute_btn.clicked.connect(self._on_execute)
        self.abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self.execute_btn)
        btn_row.addWidget(self.abort_btn)
        ctl.addLayout(btn_row)

        info_row = QHBoxLayout()
        self.qpos_text = QTextEdit()
        self.qpos_text.setReadOnly(True)
        self.qpos_text.setPlaceholderText("Measured qpos")
        self.predicted_text = QTextEdit()
        self.predicted_text.setReadOnly(True)
        self.predicted_text.setPlaceholderText("Latest predicted action")
        info_row.addWidget(self.qpos_text)
        info_row.addWidget(self.predicted_text)
        ctl.addLayout(info_row)

        self.status_label = QLabel("Idle")
        ctl.addWidget(self.status_label)

        layout.addWidget(control_group, stretch=2)

    def _to_pixmap(self, rgb: np.ndarray, target_w: int = 620, target_h: int = 360) -> QPixmap:
        img = np.ascontiguousarray(rgb)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _refresh(self):
        head = self.ros_node.head_rgb
        if head is not None:
            self.head_label.setPixmap(self._to_pixmap(head))

        wrist = self.ros_node.wrist_rgb
        if wrist is not None:
            self.wrist_label.setPixmap(self._to_pixmap(wrist))

        q = self.ros_node.get_actual_qpos()
        self.qpos_signal.emit("\n".join([f"[{i}] {v:+.5f}" for i, v in enumerate(q)]))

    def _build_policy_obs(self, prompt: str) -> dict[str, Any] | None:
        head = self.ros_node.head_rgb
        if head is None:
            return None

        wrist = self.ros_node.wrist_rgb
        if wrist is None:
            wrist = np.zeros_like(head)

        head_224 = cv2.resize(head, (224, 224), interpolation=cv2.INTER_AREA)
        wrist_224 = cv2.resize(wrist, (224, 224), interpolation=cv2.INTER_AREA)

        q = self.ros_node.get_actual_qpos()
        state = np.asarray(q[:8], dtype=np.float32)

        return {
            "image": head_224,
            "wrist_image": wrist_224,
            "state": state,
            "prompt": prompt,
        }

    def _action_to_robot_command(self, action: np.ndarray) -> list[float]:
        pred = np.asarray(action, dtype=np.float32).reshape(-1)
        padded = np.zeros(10, dtype=np.float32)
        n = min(10, pred.shape[0])
        padded[:n] = pred[:n]

        actual = np.asarray(self.ros_node.get_actual_qpos(), dtype=np.float32)
        if actual.shape[0] < 10:
            tmp = np.zeros(10, dtype=np.float32)
            tmp[: actual.shape[0]] = actual
            actual = tmp

        cmd = actual.copy()
        # First 8 channels are trained as delta actions in your stretch delta dataset.
        cmd[:8] = actual[:8] + padded[:8]
        # Base channels are command velocities.
        cmd[8] = padded[8]
        cmd[9] = padded[9]

        for i, (lo, hi) in enumerate(self.ros_node.JOINT_LIMITS):
            cmd[i] = np.clip(cmd[i], lo, hi)
        return [float(v) for v in cmd]

    def _on_execute(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.status_signal.emit("Prompt is required")
            return

        if self._worker is not None and self._worker.is_alive():
            self.status_signal.emit("Already executing")
            return

        self._abort.clear()
        self._worker = threading.Thread(target=self._run_policy_loop, args=(prompt,), daemon=True)
        self._worker.start()

    def _on_abort(self):
        self._abort.set()
        self.ros_node.publish_hold_stop()
        self.status_signal.emit("ABORTED: published hold/stop")

    def _run_policy_loop(self, prompt: str):
        self.status_signal.emit("Executing...")
        dt = 1.0 / self.step_hz

        try:
            while not self._abort.is_set():
                obs = self._build_policy_obs(prompt)
                if obs is None:
                    self.status_signal.emit("Waiting for camera frames...")
                    time.sleep(0.1)
                    continue

                result = self.policy.infer(obs)
                actions = np.asarray(result["actions"])
                if actions.ndim == 1:
                    actions = actions[None, :]

                first = actions[0]
                self.predicted_signal.emit(
                    "model action[0]:\n"
                    + np.array2string(first, precision=5, suppress_small=False)
                    + f"\nshape={actions.shape}"
                )

                for a in actions:
                    if self._abort.is_set():
                        break
                    cmd = self._action_to_robot_command(a)
                    self.ros_node.publish_joint_pose_cmd(cmd)

                    end_t = time.time() + dt
                    while time.time() < end_t:
                        if self._abort.is_set():
                            break
                        time.sleep(0.005)

            self.ros_node.publish_hold_stop()
            self.status_signal.emit("Stopped")
        except Exception as e:
            self.ros_node.publish_hold_stop()
            self.status_signal.emit(f"Execution error: {e}")

    def closeEvent(self, event):
        self._abort.set()
        self.ros_node.publish_hold_stop()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.timer.stop()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stretch realtime inference UI")
    p.add_argument("--config", default="pi05_stretch_low_mem_finetune")
    p.add_argument(
        "--checkpoint",
        default="/data/ibk5106/robotics/policies/openpi/checkpoints/pi05_stretch_low_mem_finetune/stretch3_experiment/3000",
    )
    p.add_argument("--action-dim", type=int, default=10)
    p.add_argument("--step-hz", type=float, default=6.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rclpy.init()
    ros_node = StretchROSNode()

    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cfg = _config.get_config(args.config)
    policy = create_stretch_policy(cfg, args.checkpoint, action_dim=args.action_dim)

    app = QApplication([])
    ui = InferenceUI(ros_node, policy, step_hz=args.step_hz)
    ui.show()
    rc = app.exec()

    ros_node.publish_hold_stop()
    ros_node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
