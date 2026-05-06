#!/usr/bin/env python3
"""Bridge-command teleop + ROS-topic synchronized demo recorder.

This variant uses:
- ROS2 topics (`rclpy`) for all observations (images + robot state)
- stretch_ai bridge worker only for robot commands / IK execution

Recorded samples are timestamped with ROS message header time (head RGB when
available), and state rows are aligned to that reference stamp.
"""

from __future__ import annotations

import asyncio
import base64
import colorsys
import copy
from collections import deque
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np
try:
    from flask import Flask, jsonify, request
except Exception as exc:  # pragma: no cover - environment dependent
    Flask = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    request = None  # type: ignore[assignment]
    _FLASK_IMPORT_ERROR: Exception | None = exc
else:
    _FLASK_IMPORT_ERROR = None
from ultralytics import SAM
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QEvent
from PyQt6.QtCore import QThread
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import QSize
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtWidgets import QFrame
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtWidgets import QGroupBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QInputDialog
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QListWidget
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QMenu
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QProgressBar
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QTableWidget
from PyQt6.QtWidgets import QTableWidgetItem
from PyQt6.QtWidgets import QSlider
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import QTreeWidget
from PyQt6.QtWidgets import QTreeWidgetItem
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtGui import QAction
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QImage
from PyQt6.QtGui import QKeySequence
from PyQt6.QtGui import QPixmap
from PyQt6.QtGui import QShortcut


def adjust_gamma(image: np.ndarray, gamma: float = 1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


_BLE_IMPORT_ERROR: Exception | None = None
try:
    from bleak import BleakClient
    from bleak import BleakScanner
except Exception as exc:  # pragma: no cover - environment dependent
    BleakClient = None  # type: ignore[assignment]
    BleakScanner = None  # type: ignore[assignment]
    _BLE_IMPORT_ERROR = exc

_RCLPY_IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import JointState
    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import Imu
    from sensor_msgs.msg import BatteryState
    from sensor_msgs.msg import MagneticField
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge
    from cv_bridge import CvBridgeError
    from tf2_ros import Buffer
    from tf2_ros import TransformListener
except Exception as exc:  # pragma: no cover - environment dependent
    _RCLPY_IMPORT_ERROR = exc

    class _DummyRclpy:
        class time:
            class Time:
                def __init__(self, *args, **kwargs):
                    pass

        class duration:
            class Duration:
                def __init__(self, *, seconds: float = 0.0):
                    self.seconds = float(seconds)

    rclpy = _DummyRclpy()  # type: ignore[assignment]
    Node = object  # type: ignore[assignment]
    MultiThreadedExecutor = None  # type: ignore[assignment]
    qos_profile_sensor_data = None  # type: ignore[assignment]
    Image = object  # type: ignore[assignment]
    JointState = object  # type: ignore[assignment]
    CameraInfo = object  # type: ignore[assignment]
    Imu = object  # type: ignore[assignment]
    BatteryState = object  # type: ignore[assignment]
    MagneticField = object  # type: ignore[assignment]
    Odometry = object  # type: ignore[assignment]
    CvBridge = object  # type: ignore[assignment]
    CvBridgeError = Exception  # type: ignore[assignment]
    Buffer = object  # type: ignore[assignment]
    TransformListener = object  # type: ignore[assignment]


###############################################################################
# Runtime configuration
###############################################################################
ROBOT_CONFIGS_DIR = (Path(__file__).resolve().parent / "configs").resolve()
ROBOT_CONFIG_NAME = str(os.environ.get("OPENPI_ROBOT_CONFIG", "stretch3")).strip() or "stretch3"
ROBOT_CONFIG_PATH = (ROBOT_CONFIGS_DIR / f"{ROBOT_CONFIG_NAME}.json").resolve()
DEFAULT_ROBOT_URDF_PATH = (
    "/home/ibk5106/ament_ws/src/stretch_ros2/stretch_description/urdf/exported_urdf/stretch.urdf"
)
# Do not parse URDF on import/startup; parse when embodiment is selected.
ENABLE_STARTUP_URDF_PARSE = False
DEFAULT_ROS_JOINT_STATE_NAMES: list[str] = []
DEFAULT_CONTROL_LIMITS = [
    (0.00, 0.51),     # arm_extension (derived from joint_arm_l0 * 4.0)
    (0.08, 1.05),     # arm_lift
    (-1.2, 2.2),      # wrist_yaw
    (-1.57, 1.57),    # wrist_pitch
    (-1.57, 1.57),    # wrist_roll
    (-1.57, 1.57),    # head_pan
    (-1.0, 1.0),      # head_tilt
    (-0.1, 0.5501),   # gripper
    (-2.0, 2.0),      # base_linear command step
    (-5.0, 5.0),      # base_angular command step
]
URDF_BASE_TRANSLATION_NAMES = ["joint_mobile_base_translation", "joint_base_x", "base_x_joint"]
URDF_BASE_ROTATION_NAMES = ["joint_mobile_base_rotation", "joint_base_theta", "base_theta_joint"]
BASE_CONTROLLABLE_KEYS = ["base_x", "base_theta"]
BASE_ODOM_STATE_KEYS = ["base_x", "base_y", "base_theta"]
CONTROLLABLE_STATE_JOINT_GROUPS: list[list[str]] = [
    ["joint_lift", "lift"],
    ["joint_arm_l0", "arm"],
    ["joint_wrist_yaw", "wrist_yaw"],
    ["joint_wrist_pitch", "wrist_pitch"],
    ["joint_wrist_roll", "wrist_roll"],
    ["joint_head_pan", "head_pan"],
    ["joint_head_tilt", "head_tilt"],
    ["joint_gripper_finger_left", "gripper_finger_left", "joint_gripper_finger_right", "gripper_finger_right"],
]


def _normalize_urdf_paths(raw: Any) -> list[str]:
    out: list[str] = []
    if isinstance(raw, list):
        for v in raw:
            s = str(v).strip()
            if s:
                out.append(s)
    elif isinstance(raw, (str, Path)):
        s = str(raw).strip()
        if s:
            out.append(s)
    deduped: list[str] = []
    seen: set[str] = set()
    for s in out:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def _normalize_joint_state_names(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in raw:
        s = str(v).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _normalize_base_key_list(raw: Any, allowed: list[str]) -> list[str]:
    names = _normalize_joint_state_names(raw)
    allowed_set = set(str(v) for v in allowed)
    out: list[str] = []
    for name in names:
        s = str(name).strip()
        if s in allowed_set and s not in out:
            out.append(s)
    return out


def _normalize_joint_state_pairs(joint_names_raw: Any, joint_positions_raw: Any) -> tuple[list[str], list[float]]:
    names: list[str] = []
    positions: list[float] = []
    if not isinstance(joint_names_raw, list) or not isinstance(joint_positions_raw, list):
        return names, positions
    seen: set[str] = set()
    n = min(len(joint_names_raw), len(joint_positions_raw))
    for i in range(n):
        name = str(joint_names_raw[i]).strip()
        if not name or name in seen:
            continue
        try:
            value = float(joint_positions_raw[i])
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        names.append(name)
        positions.append(value)
        seen.add(name)
    return names, positions


def _normalize_joint_limits_map(raw: Any) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        name = str(k).strip()
        if not name or not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            lo = float(v[0])
            hi = float(v[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        if lo > hi:
            lo, hi = hi, lo
        out[name] = (float(lo), float(hi))
    return out


def _load_urdf_joint_metadata(urdf_paths: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "joint_order": [],
        "joint_units": {},
        "joint_limits": {},
        "errors": [],
        "source_by_joint": {},
    }
    order: list[str] = []
    units: dict[str, str] = {}
    limits: dict[str, tuple[float, float]] = {}
    source_by_joint: dict[str, str] = {}

    paths = _normalize_urdf_paths(urdf_paths)
    if not paths:
        out["errors"] = ["empty urdf_paths"]
        return out

    for raw in paths:
        try:
            p = Path(str(raw)).expanduser().resolve()
        except Exception:
            out["errors"].append(f"invalid URDF path: {raw}")
            continue
        if not p.exists():
            out["errors"].append(f"URDF not found: {p}")
            continue

        try:
            root = ET.parse(str(p)).getroot()
        except Exception as exc:
            out["errors"].append(f"URDF parse error ({p}): {exc}")
            continue

        for joint in root.findall("joint"):
            name = str(joint.get("name", "")).strip()
            jtype = str(joint.get("type", "")).strip().lower()
            if not name:
                continue
            if jtype not in ("prismatic", "revolute", "continuous"):
                # Ignore fixed/passive joints for control metadata.
                continue
            if name not in source_by_joint:
                source_by_joint[name] = str(p)
            if name not in order:
                order.append(name)

            if jtype == "prismatic":
                units[name] = "m"
            else:
                units[name] = "rad"

            lim = joint.find("limit")
            lo: float | None = None
            hi: float | None = None
            if lim is not None:
                lo_raw = lim.get("lower")
                hi_raw = lim.get("upper")
                try:
                    if lo_raw is not None:
                        lo = float(lo_raw)
                    if hi_raw is not None:
                        hi = float(hi_raw)
                except Exception:
                    lo = None
                    hi = None
            if lo is None or hi is None:
                if jtype == "continuous":
                    lo, hi = -math.pi, math.pi
            if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
                if lo > hi:
                    lo, hi = hi, lo
                limits[name] = (float(lo), float(hi))

    out["joint_order"] = order
    out["joint_units"] = units
    out["joint_limits"] = limits
    out["source_by_joint"] = source_by_joint
    return out


def _merge_joint_state_names(*lists: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for seq in lists:
        for v in seq:
            s = str(v).strip()
            if s and s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _urdf_merged_state_joint_names(urdf_order: list[str], ros_joint_names_with_state: list[str]) -> list[str]:
    merged = _merge_joint_state_names(urdf_order, ros_joint_names_with_state)
    ros_set = set(_normalize_joint_state_names(ros_joint_names_with_state))
    out: list[str] = []
    for name in merged:
        if name in ros_set:
            out.append(name)
    return out


def _derive_controllable_joint_names(urdf_order: list[str], ros_joint_names_with_state: list[str]) -> list[str]:
    ordered_state = _urdf_merged_state_joint_names(urdf_order, ros_joint_names_with_state)
    alias_to_group: dict[str, int] = {}
    for gi, aliases in enumerate(CONTROLLABLE_STATE_JOINT_GROUPS):
        for alias in aliases:
            s = str(alias).strip()
            if s:
                alias_to_group[s] = int(gi)
    out: list[str] = []
    picked_groups: set[int] = set()
    for name in ordered_state:
        gi = alias_to_group.get(str(name))
        if gi is None or gi in picked_groups:
            continue
        out.append(str(name))
        picked_groups.add(int(gi))
    return out


def _base_name_from_urdf(order: list[str], candidates: list[str], fallback: str) -> str:
    for name in candidates:
        if name in order:
            return str(name)
    return str(fallback)


def _derive_base_controllable_names(urdf_order: list[str]) -> list[str]:
    tx_name = _base_name_from_urdf(urdf_order, URDF_BASE_TRANSLATION_NAMES, "")
    rot_name = _base_name_from_urdf(urdf_order, URDF_BASE_ROTATION_NAMES, "")
    out: list[str] = []
    if str(tx_name).strip():
        out.append("base_x")
    if str(rot_name).strip():
        out.append("base_theta")
    if len(out) == 0:
        out = list(BASE_CONTROLLABLE_KEYS)
    return out


def _derive_base_state_names(*, has_odom_state: bool) -> list[str]:
    return list(BASE_ODOM_STATE_KEYS) if bool(has_odom_state) else []


def _ordered_joint_state_names(
    urdf_order: list[str],
    ros_joint_names: list[str],
    configured_joint_names: list[str],
) -> list[str]:
    merged_nonbase = _merge_joint_state_names(configured_joint_names, urdf_order, ros_joint_names)
    base_x_name = _base_name_from_urdf(merged_nonbase, URDF_BASE_TRANSLATION_NAMES, "joint_base_x")
    base_theta_name = _base_name_from_urdf(merged_nonbase, URDF_BASE_ROTATION_NAMES, "joint_base_theta")
    base_first = [base_x_name, "joint_base_y", base_theta_name]
    out = _merge_joint_state_names(base_first, merged_nonbase)
    return out


def _derive_control_limits_from_joint_limits(joint_limits: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    out = [tuple(v) for v in DEFAULT_CONTROL_LIMITS]

    def _pick(name: str, *, min_span: float = 1e-6) -> tuple[float, float] | None:
        v = joint_limits.get(name)
        if v is None:
            return None
        lo = float(v[0])
        hi = float(v[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return None
        if abs(hi - lo) <= float(min_span):
            return None
        return (lo, hi)

    jl_lift = _pick("joint_lift")
    if jl_lift is not None:
        out[1] = jl_lift

    jl_arm = _pick("joint_arm_l0")
    if jl_arm is not None:
        out[0] = (float(jl_arm[0]) * 4.0, float(jl_arm[1]) * 4.0)

    for idx, name in (
        (2, "joint_wrist_yaw"),
        (3, "joint_wrist_pitch"),
        (4, "joint_wrist_roll"),
        (5, "joint_head_pan"),
        (6, "joint_head_tilt"),
        (7, "joint_gripper_finger_left"),
    ):
        v = _pick(name)
        if v is not None:
            out[idx] = v

    for name in URDF_BASE_TRANSLATION_NAMES:
        v = _pick(name)
        if v is not None:
            out[8] = v
            break
    for name in URDF_BASE_ROTATION_NAMES:
        v = _pick(name)
        if v is not None:
            out[9] = v
            break
    return [(float(lo), float(hi)) for lo, hi in out]


def _persist_robot_runtime_config(config_path: Path, updates: dict[str, Any]) -> None:
    try:
        existing: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing = loaded
        merged = dict(existing)
        merged.update(updates)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=True)
            f.write("\n")
    except Exception as exc:
        print(f"[robot_config] failed to persist {config_path.name}: {exc}", file=sys.stderr)


def _load_robot_runtime_config(config_dir: Path, config_name: str) -> tuple[dict[str, Any], Path]:
    cfg = {
        "robot_name": str(config_name),
        "robot_ip": "192.168.1.7",
        "urdf_paths": [],
        "joint_state_names": [],
        "controllable_joint_names": [],
        "base_controllable_names": [],
        "base_state_names": [],
        "joint_limits": {},
        "control_limits": [],
        "ui_step_defaults": [],
    }
    p = (config_dir / f"{config_name}.json").resolve()
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                ip = str(loaded.get("robot_ip", cfg["robot_ip"])).strip()
                if ip:
                    cfg["robot_ip"] = ip
                urdf_paths = _normalize_urdf_paths(loaded.get("urdf_paths"))
                if len(urdf_paths) == 0:
                    # Backward-compat: migrate single urdf_path.
                    legacy = str(loaded.get("urdf_path", "")).strip()
                    if legacy:
                        urdf_paths = [legacy]
                if len(urdf_paths) > 0:
                    cfg["urdf_paths"] = urdf_paths
                cfg["joint_state_names"] = _normalize_joint_state_names(
                    loaded.get("joint_state_names", loaded.get("joint_list", cfg["joint_state_names"]))
                )
                cfg["controllable_joint_names"] = _normalize_joint_state_names(
                    loaded.get("controllable_joint_names", cfg["controllable_joint_names"])
                )
                cfg["base_controllable_names"] = _normalize_base_key_list(
                    loaded.get("base_controllable_names", cfg["base_controllable_names"]),
                    BASE_CONTROLLABLE_KEYS,
                )
                cfg["base_state_names"] = _normalize_base_key_list(
                    loaded.get("base_state_names", cfg["base_state_names"]),
                    BASE_ODOM_STATE_KEYS,
                )
                cfg["joint_limits"] = _normalize_joint_limits_map(loaded.get("joint_limits"))
                control_limits = loaded.get("control_limits")
                if isinstance(control_limits, list) and len(control_limits) >= 10:
                    parsed: list[list[float]] = []
                    for pair in control_limits[:10]:
                        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                            try:
                                lo = float(pair[0])
                                hi = float(pair[1])
                                if lo > hi:
                                    lo, hi = hi, lo
                                parsed.append([lo, hi])
                            except Exception:
                                parsed.append(list(DEFAULT_CONTROL_LIMITS[len(parsed)]))
                        else:
                            parsed.append(list(DEFAULT_CONTROL_LIMITS[len(parsed)]))
                    cfg["control_limits"] = parsed
                ui_step_defaults = loaded.get("ui_step_defaults")
                if isinstance(ui_step_defaults, list):
                    cfg["ui_step_defaults"] = [float(v) for v in ui_step_defaults[:7] if isinstance(v, (int, float))]
                rn = str(loaded.get("robot_name", cfg["robot_name"])).strip()
                if rn:
                    cfg["robot_name"] = rn
    except Exception as exc:
        print(f"[robot_config] failed to load {config_name}.json: {exc}", file=sys.stderr)
    return cfg, p


def _embodiment_label_to_config_name(label: str) -> str:
    s = str(label).strip()
    if s in EMBODIMENT_CONFIG_NAME_BY_LABEL:
        return str(EMBODIMENT_CONFIG_NAME_BY_LABEL[s])
    low = s.lower()
    if low in {k.lower(): v for k, v in EMBODIMENT_CONFIG_NAME_BY_LABEL.items()}:
        # keep deterministic mapping even for case-only differences
        for k, v in EMBODIMENT_CONFIG_NAME_BY_LABEL.items():
            if str(k).lower() == low:
                return str(v)
    slug = []
    prev_us = False
    for ch in low:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            slug.append(ch)
            prev_us = False
        elif not prev_us:
            slug.append("_")
            prev_us = True
    cfg = "".join(slug).strip("_")
    return str(cfg or ROBOT_CONFIG_NAME)


ROBOT_RUNTIME_CONFIG, ROBOT_CONFIG_PATH = _load_robot_runtime_config(ROBOT_CONFIGS_DIR, ROBOT_CONFIG_NAME)
ROBOT_URDF_PATHS = _normalize_urdf_paths(ROBOT_RUNTIME_CONFIG.get("urdf_paths", []))
if bool(ENABLE_STARTUP_URDF_PARSE):
    if len(ROBOT_URDF_PATHS) == 0:
        ROBOT_URDF_PATHS = [str(DEFAULT_ROBOT_URDF_PATH)]
    ROBOT_URDF_META = _load_urdf_joint_metadata(ROBOT_URDF_PATHS)
else:
    ROBOT_URDF_META = {
        "joint_order": [],
        "joint_units": {},
        "joint_limits": {},
        "errors": [],
        "source_by_joint": {},
    }
ROBOT_URDF_JOINT_ORDER = list(ROBOT_URDF_META.get("joint_order", []))
ROBOT_URDF_JOINT_UNITS = dict(ROBOT_URDF_META.get("joint_units", {}))
ROBOT_URDF_JOINT_LIMITS = dict(ROBOT_URDF_META.get("joint_limits", {}))
_cfg_joint_limits = _normalize_joint_limits_map(ROBOT_RUNTIME_CONFIG.get("joint_limits", {}))
ROBOT_JOINT_LIMITS_BY_NAME = dict(ROBOT_URDF_JOINT_LIMITS)
ROBOT_JOINT_LIMITS_BY_NAME.update(_cfg_joint_limits)
ROBOT_JOINT_STATE_NAMES = _normalize_joint_state_names(ROBOT_RUNTIME_CONFIG.get("joint_state_names", []))
ROBOT_CONTROLLABLE_JOINT_NAMES = _normalize_joint_state_names(
    ROBOT_RUNTIME_CONFIG.get("controllable_joint_names", [])
)
if len(ROBOT_CONTROLLABLE_JOINT_NAMES) == 0:
    ROBOT_CONTROLLABLE_JOINT_NAMES = _derive_controllable_joint_names(
        ROBOT_URDF_JOINT_ORDER,
        ROBOT_JOINT_STATE_NAMES,
    )
ROBOT_BASE_CONTROLLABLE_NAMES = _normalize_base_key_list(
    ROBOT_RUNTIME_CONFIG.get("base_controllable_names", []),
    BASE_CONTROLLABLE_KEYS,
)
if len(ROBOT_BASE_CONTROLLABLE_NAMES) == 0:
    ROBOT_BASE_CONTROLLABLE_NAMES = _derive_base_controllable_names(ROBOT_URDF_JOINT_ORDER)
ROBOT_BASE_STATE_NAMES = _normalize_base_key_list(
    ROBOT_RUNTIME_CONFIG.get("base_state_names", []),
    BASE_ODOM_STATE_KEYS,
)
_cfg_control_limits = ROBOT_RUNTIME_CONFIG.get("control_limits", [])
if isinstance(_cfg_control_limits, list) and len(_cfg_control_limits) >= 10:
    ROBOT_CONTROL_LIMITS = [(float(v[0]), float(v[1])) for v in _cfg_control_limits[:10]]
else:
    ROBOT_CONTROL_LIMITS = _derive_control_limits_from_joint_limits(ROBOT_JOINT_LIMITS_BY_NAME)
if bool(ENABLE_STARTUP_URDF_PARSE) and isinstance(ROBOT_URDF_META.get("errors"), list) and len(ROBOT_URDF_META.get("errors", [])) > 0:
    for err in ROBOT_URDF_META.get("errors", []):
        print(f"[robot_config] urdf parse warning: {err}", file=sys.stderr)
print(
    f"[robot_config] robot={ROBOT_RUNTIME_CONFIG.get('robot_name')} ip={ROBOT_RUNTIME_CONFIG.get('robot_ip')} "
    f"urdf_files={len(ROBOT_URDF_PATHS)} startup_urdf_parse={bool(ENABLE_STARTUP_URDF_PARSE)} "
    f"joints_resolved={len(ROBOT_JOINT_STATE_NAMES)}",
    flush=True,
)
print(
    f"[robot_config] base_translation_name={_base_name_from_urdf(ROBOT_URDF_JOINT_ORDER, URDF_BASE_TRANSLATION_NAMES, 'joint_base_x')} "
    f"base_rotation_name={_base_name_from_urdf(ROBOT_URDF_JOINT_ORDER, URDF_BASE_ROTATION_NAMES, 'joint_base_theta')}",
    flush=True,
)

STRETCH_AI_REPO = "/home/ibk5106/Desktop/Projects/stretch_ai"
STRETCH_AI_ENV_NAME = "stretch_ai"
STRETCH_AI_WORKER_LAUNCHER = "mamba"
STRETCH_AI_WORKER_PYTHON = "python"

STRETCH_AI_ROBOT_IP = str(ROBOT_RUNTIME_CONFIG.get("robot_ip", "192.168.1.7"))
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
STRETCH_AI_DEFAULT_OBS_SOURCE = "ros_topic"
FORCE_ROS_TOPIC_OBSERVATION = True
COMMAND_HISTORY_MAXLEN = 4096

# ROS2 topic observation configuration.
ROS_TOPICS_ROTATE_HEAD_90_CW = True
ROS_TOPICS_CONNECT_TIMEOUT_S = 8.0

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
DEMO_RECORD_QUEUE_MAX = 20
DEFAULT_DATASET_ROOT = str((Path.cwd() / "demo_record/stretch_recordings_v9_simple_alltask").resolve())
DEMO_RECORD_RGB_DEFAULT_FORMAT = "jpg"
DEMO_RECORD_RGB_JPEG_QUALITY = 90

# Task/prompt library for recording UI.
TASK_PROMPT_LIBRARY_JSON = "task_prompt_library.json"
TASK_DROPDOWN_OPTIONS = [
    "Reach to object",
    "Grasp object",
    "Place object",
    "Pick and place",
    "Lift object",
    "Move object to target location",
    "Stack objects",
    "Unstack objects",
    "Sort objects by category, color, or size",
    "Push object",
    "Pull object",
    "Slide object",
    "Rotate or reorient object",
    "Insert object into slot or hole",
    "Wipe or clean surface",
    "Pouring from one container to another",
]
TASK_DEFAULT_NAME = "Pick and place"
TASK_DEFAULT_PROMPTS = {
    "Reach to object": "Reach the gripper tip to the target object.",
    "Grasp object": "Grasp the target object safely.",
    "Place object": "Place the grasped object at the marked target point.",
    "Pick and place": "Pick up the blue block and put that in the box.",
    "Lift object": "Lift the object vertically after grasping.",
    "Move object to target location": "Move the grasped object to the target location.",
    "Stack objects": "Pick up an object and stack it on top of another object.",
    "Unstack objects": "Remove the top object from a stack and place it aside.",
    "Sort objects by category, color, or size": "Sort objects by color into separate regions.",
    "Push object": "Push the object to the marked target area.",
    "Pull object": "Pull the object toward the robot.",
    "Slide object": "Slide the object along the table to the target.",
    "Rotate or reorient object": "Rotate the object to match the target orientation.",
    "Insert object into slot or hole": "Insert the object into the slot or hole.",
    "Wipe or clean surface": "Wipe the marked surface area.",
    "Pouring from one container to another": "Pour from one container into the target container.",
}
AUTO_LOOP_COMPLEX_TASKS = {
    "Stack objects",
    "Unstack objects",
    "Sort objects by category, color, or size",
    "Insert object into slot or hole",
    "Wipe or clean surface",
    "Pouring from one container to another",
}
AUTO_LOOP_SIMPLE_SEQUENCE_GOAL_KINDS = {
    "grasp",
    "reach",
    "place_object",
    "release",
    "drag",
    "drag_curve",
    "lift_delta",
    "stretch_delta",
    "translate_delta",
}
EMBODIMENT_OPTIONS = ["Stretch 3", "Viper x", "UR5", "Franka panda"]
EMBODIMENT_DEFAULT = "Stretch 3"
EMBODIMENT_PLACEHOLDER_TEXT = "Select Embodiment"
PRIMARY_VIEW_PLACEHOLDER_TEXT = "Select Primary View"
EMBODIMENT_CONFIG_NAME_BY_LABEL = {
    "Stretch 3": "stretch3",
    "Viper x": "viper_x",
    "UR5": "ur5",
    "Franka panda": "franka_panda",
}
RGB_SOURCE_OPTIONS = [
    ("Head Camera", "head"),
    ("Wrist Camera", "wrist"),
]

# External 3-encoder input device integration (direct BLE)
DEVICE_BLE_NAME = "ESP32C3-IMU"
DEVICE_BLE_CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_BLE_RETRY_SLEEP_S = 0.5
DEVICE_POLL_HZ = 50.0
DEVICE_ENCODER_DEADBAND = 1e-4
DEVICE_IMU_YAW_GAIN = 1.0
DEVICE_IMU_YAW_MAX_STEP_RAD = 0.10
DEVICE_DEBUG_PRINT = False
DEVICE_DEBUG_PRINT_MAX = 0
DEVICE_BASE_STEP_M = 0.12
DEVICE_LIFT_STEP_M = 0.02
DEVICE_ARM_STEP_M = 0.02
DEVICE_GRIPPER_STEP = 0.02
DEVICE_GRIPPER_TOGGLE_OPEN_JOINT = 0.15
DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT = -0.15
# Device precision-level step mapping ranges (same ranges as UI sliders).
DEVICE_PREC_LEVELS = 5  # prec 0..4
DEVICE_BASE_STEP_MIN_M = 0.005
DEVICE_BASE_STEP_MAX_M = 0.20
DEVICE_ARM_STEP_MIN = 0.005
DEVICE_ARM_STEP_MAX = 0.10
DEVICE_GRIPPER_STEP_MIN = 0.005
DEVICE_GRIPPER_STEP_MAX = 0.10

# Command smoothing (same values as v5)
COMMAND_SMOOTH_STEP_SIZES = [
    0.020,
    0.020,
    0.020,
    0.020,
    0.020,
    0.020,
    0.020,
    0.005,
    0.020,
    0.030,
]

DEFAULT_COMMAND_SMOOTH_DELAY_S = 0.02
UI_REFRESH_MS = 100
DEFAULT_BASE_ROTATE_STEP_DEG = 0.2
DEFAULT_BASE_ROTATE_STEP_DELAY_S = 0.10
# Main v9 UI step-size defaults (single place to edit).
# Order:
# [base_linear_m, base_rotate_deg, arm_step, head_step, wrist_step, gripper_step, smoothing_delay_s]
UI_STEP_SLIDER_DEFAULTS = [
    0.03,
    float(DEFAULT_BASE_ROTATE_STEP_DEG),
    0.005,
    0.02,
    0.02,
    0.02,
    float(DEFAULT_COMMAND_SMOOTH_DELAY_S),
]
UI_STEP_IDX_BASE_LINEAR = 0
UI_STEP_IDX_BASE_ROTATE_DEG = 1
UI_STEP_IDX_ARM = 2
UI_STEP_IDX_HEAD = 3
UI_STEP_IDX_WRIST = 4
UI_STEP_IDX_GRIPPER = 5
UI_STEP_IDX_SMOOTH_DELAY = 6

# Queue execution confirmation policy:
# keep explicit user confirmation pauses in queued grasp/reach flows so users
# can fine-tune pose before contact/release.
QUEUE_REQUIRE_GRASP_CONFIRM = True


def _derive_ui_step_defaults(
    *,
    control_limits: list[tuple[float, float]],
    configured_defaults: Any,
) -> list[float]:
    if isinstance(configured_defaults, list) and len(configured_defaults) >= 7:
        out: list[float] = []
        for i in range(7):
            try:
                out.append(float(configured_defaults[i]))
            except Exception:
                out.append(float(UI_STEP_SLIDER_DEFAULTS[i]))
        return out

    def _range_of(idx: int) -> float:
        try:
            lo, hi = control_limits[idx]
            r = float(abs(float(hi) - float(lo)))
            if math.isfinite(r) and r > 1e-9:
                return r
        except Exception:
            pass
        lo0, hi0 = DEFAULT_CONTROL_LIMITS[idx]
        return float(abs(float(hi0) - float(lo0)))

    # Preserve old default ratios against command/joint ranges.
    old_ranges = [float(abs(float(hi) - float(lo))) for lo, hi in DEFAULT_CONTROL_LIMITS]
    old_defaults = [float(v) for v in UI_STEP_SLIDER_DEFAULTS]

    base_lin = old_defaults[UI_STEP_IDX_BASE_LINEAR] * (_range_of(8) / max(1e-9, old_ranges[8]))
    base_rot_deg = math.degrees(
        math.radians(old_defaults[UI_STEP_IDX_BASE_ROTATE_DEG]) * (_range_of(9) / max(1e-9, old_ranges[9]))
    )
    arm_range = min(_range_of(0), _range_of(1))
    arm_ref = min(old_ranges[0], old_ranges[1])
    arm_step = old_defaults[UI_STEP_IDX_ARM] * (arm_range / max(1e-9, arm_ref))
    head_range = min(_range_of(5), _range_of(6))
    head_ref = min(old_ranges[5], old_ranges[6])
    head_step = old_defaults[UI_STEP_IDX_HEAD] * (head_range / max(1e-9, head_ref))
    wrist_range = min(_range_of(2), _range_of(3), _range_of(4))
    wrist_ref = min(old_ranges[2], old_ranges[3], old_ranges[4])
    wrist_step = old_defaults[UI_STEP_IDX_WRIST] * (wrist_range / max(1e-9, wrist_ref))
    gripper_step = old_defaults[UI_STEP_IDX_GRIPPER] * (_range_of(7) / max(1e-9, old_ranges[7]))

    out = [
        float(np.clip(base_lin, DEVICE_BASE_STEP_MIN_M, DEVICE_BASE_STEP_MAX_M)),
        float(np.clip(base_rot_deg, 0.2, 25.0)),
        float(np.clip(arm_step, DEVICE_ARM_STEP_MIN, DEVICE_ARM_STEP_MAX)),
        float(np.clip(head_step, 0.01, 0.30)),
        float(np.clip(wrist_step, 0.01, 0.30)),
        float(np.clip(gripper_step, DEVICE_GRIPPER_STEP_MIN, DEVICE_GRIPPER_STEP_MAX)),
        float(UI_STEP_SLIDER_DEFAULTS[UI_STEP_IDX_SMOOTH_DELAY]),
    ]
    return out


ROBOT_UI_STEP_DEFAULTS = _derive_ui_step_defaults(
    control_limits=ROBOT_CONTROL_LIMITS,
    configured_defaults=ROBOT_RUNTIME_CONFIG.get("ui_step_defaults", []),
)

# NOTE:
# Do not persist robot runtime metadata at module import/startup.
# Config JSON should only be updated after explicit embodiment selection.

UI_CAMERA_DISPLAY_SCALE = 1.00
HEAD_DISPLAY_CROP_BOTTOM_FRAC = 0.20

# In bridge arm_to mode, base motion is available as manipulation base_x only.
MANIP_BASE_X_LIMITS = (-1.35, 1.35)

# v5 behavior constants (kept for full feature parity with robot_teleop_ui_ros_v5.py).
COMPENSATE_HEAD_ON_ROTATE = True
REACH_HEIGHT_CLEARANCE = 0.20
GRASP_PITCH_DEG = -40.0
# Fixed wrist-pitch target for "Grasp and Rotate" mode (radians).
GRASP_ROTATE_FORCE_PITCH_RAD = -1.55
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
GRASP_TARGET_Z_OFFSET_M = 0.015
# Hard cap on explicit scripted inter-stage waits. Keep this small so
# sequential joints start right after the previous move completes.
SCRIPT_STAGE_WAIT_CAP_S = 0.08

# Action completion tolerances/timeouts (UI side; easy to tune).
ACTION_BASE_X_SETTLE_TOL_M = 0.025
ACTION_BASE_X_SETTLE_TIMEOUT_S = 6.0
RETURN_BASE_X_RESTORE_TOL_M = 0.025
RETURN_BASE_X_RESTORE_ATTEMPTS = 1
RETURN_FINAL_NAV_BASE_X_REFINE_TOL_M = 0.025
EXECUTE_ARM_TO_INTERMEDIATE_TIMEOUT_MIN_S = 0.6
EXECUTE_ARM_TO_INTERMEDIATE_TIMEOUT_MAX_S = 1.5
EXECUTE_ARM_TO_REFRESH_TIMEOUT_LAST_S = 0.20
EXECUTE_ARM_TO_REFRESH_TIMEOUT_INTERMEDIATE_S = 0.08
ACTION_MOVE_TIMEOUT_SHORT_S = 6.0
ACTION_MOVE_TIMEOUT_DEFAULT_S = 8.0
ACTION_MOVE_TIMEOUT_LONG_S = 10.0
ACTION_MOVE_TIMEOUT_HOME_S = 12.0
# Hold duration for grasp-only / lift-only auto-loop type1 episodes.
AUTO_LOOP_GRASP_HOLD_S = 3.0
# Worker-side motion tuning (passed from UI at connect time).
WORKER_TUNE_NAV_BLOCKING_POS_TOL_M = 0.05
WORKER_TUNE_NAV_BLOCKING_YAW_TOL_DEG = 3.5
WORKER_TUNE_ARM_TO_TOL_BASE_X_M = 0.04
WORKER_TUNE_ARM_TO_TOL_LIFT_M = 0.03
WORKER_TUNE_ARM_TO_TOL_ARM_M = 0.03
WORKER_TUNE_ARM_TO_TOL_WRIST_RAD = 0.10
WORKER_TUNE_ARM_TO_TOL_GRIPPER = 0.03
WORKER_TUNE_ARM_TO_TOL_HEAD_RAD = 0.10
WORKER_TUNE_ARM_TO_MODE_WAIT_TIMEOUT_S = 1.5
WORKER_TUNE_ARM_TO_STALLED_CHECK_MIN_S = 0.6
WORKER_TUNE_ARM_TO_STALLED_CHECK_MAX_S = 2.0
WORKER_TUNE_ARM_TO_STALLED_CHECK_SCALE = 0.30
WORKER_TUNE_SET_RPC_TIMEOUT_S = 4.0


SAM_CC_MIN_AREA_PX = 600
SAM_CC_MIN_WIDTH_PX = 18
SAM_CC_MIN_HEIGHT_PX = 18
# UI toggle: keep manual grasp-region panel hidden for now.
MANUAL_GRASP_REGION_UI_VISIBLE = False
# Extra width padding for column 3 content panel.
# Increased slightly to avoid right-edge clipping of goal/source controls.
THIRD_COLUMN_EXTRA_WIDTH_PX = 60
# Scale factor for the third column width (set <1.0 to make it narrower).
# Restored to the previous (wider) value to prevent content cropping.
THIRD_COLUMN_WIDTH_SCALE = 0.85
# Compact UI button heights.
UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX = 26
UI_THIRD_COLUMN_BUTTON_HEIGHT_PX = 26
UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX = 24
UI_BASE_MIDDLE_ARROW_BUTTON_HEIGHT_PX = 12

# v6: use stretch_ai IK/open-loop planning for reach/grasp instead of custom geometry.
USE_STRETCH_AI_IK_GRASP_PIPELINE = True
IK_PREGRASP_DISTANCE_M = 0.10
IK_LIFT_DISTANCE_M = 0.20
IK_SAFE_LIFT_M = 0.95
# Return-stage safety lift kept independent from startup/default lift.
RETURN_SAFE_LIFT_M = 0.95
# Calibration offset applied to IK grasp base_x before execution.
# Positive pushes farther forward; negative pulls back.
IK_GRASP_BASE_X_OFFSET_M = 0.0
# Gripper partial-close tuning for grasp:
# close target = open_target - (IK_GRIPPER_CLOSE_DELTA_M / 0.22)
# where 0.22m per joint-unit comes from existing width->joint mapping.
IK_GRIPPER_CLOSE_DELTA_M = 0.035  # tune between 0.02 .. 0.05 (2-5 cm)
IK_GRIPPER_CLOSE_MIN_JOINT = -0.02  # prevent hard full-close motor load
# Optional close target mode for grasp:
# when True, final close target uses
#   min(estimated_object_width_joint, DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT)
# when False, keep existing delta-based close behavior.
IK_GRIPPER_CLOSE_USE_MIN_OBJECT_WIDTH_AND_DEVICE_CLOSE = True
RETURN_USE_NAV_BASE_POSE_CORRECTION = False
AUTO_SEND_DEFAULT_POSE_ON_CONNECT = True
DEFAULT_INIT_LIFT_M = 0.95
# Startup command target (non-base joints) shown in the UI command table.
# Order: [arm_extension, arm_lift, wrist_yaw, wrist_pitch, wrist_roll, head_pan, head_tilt, gripper]
# DEFAULT_INIT_CMD_QPOS8 = [0.0, 0.95, -1.2, -0.535, 0.0, -0.92, -0.66, 0.22505]
# DEFAULT_INIT_CMD_QPOS8 = [0.0, 0.85, 0.28, -0.6981, 0.0, -1.56, -0.75, 0.22505]
DEFAULT_INIT_CMD_QPOS8 = [0.0, 0.85, 0.28, -0.6981, 0.0, -1.5708, -0.7854, 0.15]
# DEFAULT_INIT_CMD_QPOS8 = [0.01, 0.78, 0.0, -1.5, 0.0, -1.5707963267948966, -0.7853981633974483, 0.0]
ALLOW_COMMAND_SYNC_FROM_STATE = False
# Wrist camera-view parked pitch used outside active approach motion.
CAMERA_VIEW_PARK_PITCH_RAD = -0.6981 # -1.5
# Lift parked value used with camera-view parked pitch.
CAMERA_VIEW_PARK_LIFT_M = 0.85 # 0.75
# If False, skip the extra park move after startup/return flows.
# Useful when DEFAULT_INIT_CMD_QPOS8 already matches CAMERA_VIEW_PARK_* targets.
ENABLE_CAMERA_VIEW_PARK_MOVE = False
# Clip IK wrist-yaw target around startup yaw to simplify demonstrations.
IK_WRIST_YAW_CLIP_AROUND_INIT_DEG = 10.0
# Auto-loop replay augmentation:
# randomize bring-back placement into grasp location by perturbing joint targets.
# Set either value to 0.0 to disable that axis jitter.
AUTO_LOOP_GRASP_PLACE_JITTER_BASE_X_M = 0.07  # +/- 3 cm on manip base_x
AUTO_LOOP_GRASP_PLACE_JITTER_ARM_M = 0.12     # +/- 5 cm on arm extension
# Auto-loop pickup variation (for recovery-style data):
# Before final exact pickup, optionally visit an intentionally off-target pre-approach.
# This helps generate "corrective" demonstrations where the gripper is slightly off.
AUTO_LOOP_PICK_VARIATION_ENABLED = True
# Sampling weights (normalized internally; set all but one to 0 to force that mode).
AUTO_LOOP_PICK_VARIATION_W_DIRECT = 0.35
AUTO_LOOP_PICK_VARIATION_W_OVERSHOOT = 0.20
AUTO_LOOP_PICK_VARIATION_W_SIDE = 0.20
AUTO_LOOP_PICK_VARIATION_W_SHORT = 0.25
# Magnitudes for each mode.
AUTO_LOOP_PICK_VARIATION_OVERSHOOT_ARM_M = 0.12
AUTO_LOOP_PICK_VARIATION_OVERSHOOT_BASE_X_M = 0.02
AUTO_LOOP_PICK_VARIATION_SHORT_ARM_M = 0.12
AUTO_LOOP_PICK_VARIATION_SHORT_BASE_X_M = 0.02
AUTO_LOOP_PICK_VARIATION_SIDE_BASE_X_M = 0.03
AUTO_LOOP_PICK_VARIATION_SIDE_YAW_DEG = 8.0
# Ensure random variation moves stay above contact height.
AUTO_LOOP_PICK_VARIATION_MIN_CLEARANCE_M = 0.02

# v7: geometry-aware grasp surface classification
# If visible-surface normal is mostly horizontal (small |normal.z|), treat as
# vertical-face grasp (door handle / knob / cabinet pull / side grasp).
VERTICAL_SURFACE_NORMAL_Z_MAX = 0.45
HORIZONTAL_SURFACE_NORMAL_Z_MIN = 0.75
VERTICAL_OBJECT_HEIGHT_MIN_M = 0.15
VERTICAL_OBJECT_XY_SPAN_MAX_M = 0.12
# For vertical-face grasps: pause at reach target - standoff instead of lift + margin.
IK_REACH_STANDOFF_M = 0.10
# Keep direct single-click grasp behavior aligned with queued grasp behavior.
# Queued grasp currently does not pass geometry-class strategy into IK stage
# selection, so default this off to avoid direct-only vertical misclassification.
DIRECT_GRASP_USE_GEOMETRY_STRATEGY = False

# Head-view drag operation tuning.
DRAG_POINT_Z_SAFETY_M = 0.03
DRAG_MIN_PIXEL_LENGTH_PX = 6
DRAG_PLAN_TIMEOUT_S = 35.0
DRAG_NEAREST_VALID_DEPTH_RADIUS_PX = 30
DRAG_CURVE_MIN_CAPTURE_POINTS = 4
DRAG_CURVE_FIT_SAMPLES = 64
DRAG_CURVE_EXEC_WAYPOINTS = 10
# Curve execution smoothing:
# stream intermediate waypoints without waiting for full settle, and block only
# on the final waypoint of each forward/return pass.
DRAG_CURVE_STREAM_NONBLOCKING_INTERMEDIATE = False
DRAG_CURVE_STREAM_INTER_WAYPOINT_DELAY_S = 0.0
# When chaining drag/curve immediately after grasp(no-lift), keep arm
# extension from retracting too much at drag start.
# NOTE: if this is 0.0, drag directions that require slight retract can fail.
DRAG_CHAINED_MAX_INITIAL_RETRACT_M = 0.05
# Extra settle delay between scripted curve type2 reset steps for placement precision.
CURVE_RESET_INTER_STEP_SETTLE_S = 0.015
# Settle delay between linear drag type2 (joint-pair) return steps.
LINEAR_RETURN_INTER_STEP_SETTLE_S = 0.02


class PointStamped:
    """Minimal geometry_msgs.msg.PointStamped-compatible container."""

    def __init__(self):
        self.header = types.SimpleNamespace(frame_id="", stamp=None)
        self.point = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)


class EncoderDeviceHttpBridge:
    """Receives IMU+encoder packets directly over BLE and stores latest samples."""

    def __init__(
        self,
        *,
        device_name: str = DEVICE_BLE_NAME,
        characteristic_uuid: str = DEVICE_BLE_CHARACTERISTIC_UUID,
    ):
        self.device_name = str(device_name)
        self.characteristic_uuid = str(characteristic_uuid)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._last_error: str | None = None
        self._require_calibration = True
        self._q_ref_xyzw: np.ndarray | None = None
        s2 = float(math.sqrt(0.5))
        # Equivalent to scipy: R.from_euler('yx', [90, 0], degrees=True)
        self._mount_q_xyzw = np.array([0.0, s2, 0.0, s2], dtype=np.float64)

    def _safe_put(self, payload: dict[str, Any]) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    @staticmethod
    def _quat_mul(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = [float(v) for v in a_xyzw]
        bx, by, bz, bw = [float(v) for v in b_xyzw]
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_inv(q_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = [float(v) for v in q_xyzw]
        n2 = x * x + y * y + z * z + w * w
        if n2 <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return np.array([-x / n2, -y / n2, -z / n2, w / n2], dtype=np.float64)

    @staticmethod
    def _quat_norm(q_xyzw: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(q_xyzw))
        if n <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q_xyzw / n

    def _parse_notify_line(self, line: str) -> dict[str, Any] | None:
        txt = str(line).strip()
        if not txt:
            return None
        parts = [p.strip() for p in txt.split(",")]
        # print(f"[device] notify line parts: {parts}")
        if len(parts) < 5:
            return None
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            w = float(parts[4])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(w)):
            return None

        q_raw = np.array([x, y, z, w], dtype=np.float64)
        q_cur = self._quat_mul(q_raw, self._mount_q_xyzw)

        calibrate = self._require_calibration
        if len(parts) > 16:
            try:
                calibrate = calibrate or (int(float(parts[16])) == 1)
            except (TypeError, ValueError):
                pass
        if calibrate:
            self._q_ref_xyzw = q_cur.copy()
            self._require_calibration = False

        if self._q_ref_xyzw is not None:
            q_corr = self._quat_mul(self._quat_inv(self._q_ref_xyzw), q_cur)
        else:
            q_corr = q_cur
        q_corr = self._quat_norm(q_corr)

        return {
            "quat": [
                f"{float(q_corr[0]):.6f}",
                f"{float(q_corr[1]):.6f}",
                f"{float(q_corr[2]):.6f}",
                f"{float(q_corr[3]):.6f}",
            ],
            "raw": parts[5:],
            "line": txt,
            "parts": parts,
        }

    async def _ble_loop(self) -> None:
        if BleakClient is None or BleakScanner is None:
            self._last_error = f"bleak import failed: {_BLE_IMPORT_ERROR}"
            return

        while not self._stop_event.is_set():
            try:
                device = await BleakScanner.find_device_by_filter(
                    lambda d, ad: bool(d and d.name and self.device_name in d.name)
                )
            except Exception as exc:
                self._last_error = f"BLE scan failed: {exc}"
                await asyncio.sleep(float(DEVICE_BLE_RETRY_SLEEP_S))
                continue

            if not device:
                self._last_error = f"BLE device not found: {self.device_name}"
                await asyncio.sleep(float(DEVICE_BLE_RETRY_SLEEP_S))
                continue

            try:
                async with BleakClient(device) as client:
                    self._last_error = None

                    def handle_notify(_sender: Any, data: Any) -> None:
                        
                        try:
                            text = data.decode("utf-8").strip() # bytes(data).decode("utf-8", errors="ignore").replace("\x00", "")
                        except Exception:
                            return
                        # print(f"[device] notify: {text}")
                        for line in text.splitlines():
                            payload = self._parse_notify_line(line)
                            # print(f"[device] parsed payload: {payload}")
                            if isinstance(payload, dict):
                                self._safe_put(payload)

                    await client.start_notify(self.characteristic_uuid, handle_notify)
                    while not self._stop_event.is_set():
                        await asyncio.sleep(0.1)
                    try:
                        await client.stop_notify(self.characteristic_uuid)
                    except Exception:
                        pass
            except Exception as exc:
                self._last_error = f"BLE connection failed: {exc}"
                await asyncio.sleep(float(DEVICE_BLE_RETRY_SLEEP_S))

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._ble_loop())
        except Exception as exc:
            self._last_error = str(exc)
        finally:
            try:
                loop.stop()
                loop.close()
            except Exception:
                pass

    def start(self) -> bool:
        if self._started:
            return True
        if BleakClient is None or BleakScanner is None:
            self._last_error = f"bleak import failed: {_BLE_IMPORT_ERROR}"
            return False
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
            name="encoder_ble_bridge",
        )
        self._worker_thread.start()
        self._started = True
        return True

    def stop(self) -> None:
        self._stop_event.set()
        t = self._worker_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.5)
        self._worker_thread = None
        self._started = False

    def get_latest(self) -> dict[str, Any] | None:
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def drain(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while True:
            try:
                item = self._queue.get_nowait()
                # print(item)
            except queue.Empty:
                break
            if isinstance(item, dict):
                items.append(item)
        return items

    @property
    def last_error(self) -> str | None:
        return self._last_error


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
        self.rgb_image_format = str(DEMO_RECORD_RGB_DEFAULT_FORMAT).lower()
        self.rgb_jpeg_quality = int(DEMO_RECORD_RGB_JPEG_QUALITY)

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

    def start(
        self,
        dataset_root: str,
        prompt: str,
        *,
        rgb_image_format: str = DEMO_RECORD_RGB_DEFAULT_FORMAT,
        rgb_jpeg_quality: int = DEMO_RECORD_RGB_JPEG_QUALITY,
    ):
        root = Path(dataset_root).expanduser().resolve()
        self.root = root
        self.prompt = (prompt or "").strip()
        fmt = str(rgb_image_format or DEMO_RECORD_RGB_DEFAULT_FORMAT).strip().lower()
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in {"jpg", "png"}:
            fmt = str(DEMO_RECORD_RGB_DEFAULT_FORMAT).lower()
        self.rgb_image_format = fmt
        q = int(rgb_jpeg_quality) if isinstance(rgb_jpeg_quality, (int, float)) else int(DEMO_RECORD_RGB_JPEG_QUALITY)
        self.rgb_jpeg_quality = int(max(60, min(100, q)))
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
            # Depth preview images are intentionally disabled to reduce recorder I/O load.
            # Keep JSON keys, but write them as null in rows.
            # "head_depth_preview_frames": self._ensure(images_dir / "observation.images.head_depth" / self.episode_name),
            # "wrist_depth_preview_frames": self._ensure(images_dir / "observation.images.wrist_depth" / self.episode_name),
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
        rgb_ext = ".jpg" if self.rgb_image_format == "jpg" else ".png"

        head_rgb_png_rel = None
        if head_rgb is not None:
            head_rgb_path = self.paths["head_rgb_frames"] / f"frame_{self.frame_index:06d}{rgb_ext}"
            head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
            if self.rgb_image_format == "jpg":
                ok = cv2.imwrite(
                    str(head_rgb_path),
                    head_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.rgb_jpeg_quality)],
                )
            else:
                ok = cv2.imwrite(str(head_rgb_path), head_bgr)
            if ok:
                head_rgb_png_rel = self._rel(head_rgb_path)

        wrist_rgb_png_rel = None
        if wrist_rgb is not None:
            wrist_rgb_path = self.paths["wrist_rgb_frames"] / f"frame_{self.frame_index:06d}{rgb_ext}"
            wrist_rgb = adjust_gamma(wrist_rgb, 2.5)
            wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
            if self.rgb_image_format == "jpg":
                ok = cv2.imwrite(
                    str(wrist_rgb_path),
                    wrist_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.rgb_jpeg_quality)],
                )
            else:
                ok = cv2.imwrite(str(wrist_rgb_path), wrist_bgr)
            if ok:
                wrist_rgb_png_rel = self._rel(wrist_rgb_path)

        head_depth_png_rel = None
        head_depth_preview_png_rel = None
        if head_depth is not None:
            # Depth preview generation/write disabled for faster recording.
            # head_depth_rgb = self._depth_preview_rgb(head_depth)
            # head_depth_preview_png = self.paths["head_depth_preview_frames"] / f"frame_{self.frame_index:06d}.png"
            # cv2.imwrite(str(head_depth_preview_png), cv2.cvtColor(head_depth_rgb, cv2.COLOR_RGB2BGR))
            # head_depth_preview_png_rel = self._rel(head_depth_preview_png)
            depth_mm = np.clip(np.array(head_depth, dtype=np.float32) * 1000.0, 0, 65535).astype(np.uint16)
            depth_png = self.paths["head_depth_frames"] / f"frame_{self.frame_index:06d}.png"
            cv2.imwrite(str(depth_png), depth_mm)
            head_depth_png_rel = self._rel(depth_png)

        wrist_depth_png_rel = None
        wrist_depth_preview_png_rel = None
        if wrist_depth is not None:
            # Depth preview generation/write disabled for faster recording.
            # wrist_depth_rgb = self._depth_preview_rgb(wrist_depth)
            # wrist_depth_preview_png = self.paths["wrist_depth_preview_frames"] / f"frame_{self.frame_index:06d}.png"
            # cv2.imwrite(str(wrist_depth_preview_png), cv2.cvtColor(wrist_depth_rgb, cv2.COLOR_RGB2BGR))
            # wrist_depth_preview_png_rel = self._rel(wrist_depth_preview_png)
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

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    @staticmethod
    def _safe_rmtree(path: Path) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _prune_empty_parent(self, start_path: Path) -> None:
        stop_at = Path(self.root) if self.root is not None else None
        cur = start_path
        while True:
            if stop_at is not None and cur == stop_at:
                break
            try:
                cur.rmdir()
            except OSError:
                break
            except Exception:
                break
            parent = cur.parent
            if parent == cur:
                break
            cur = parent

    def _discard_episode_files(self) -> None:
        data_file = self.paths.get("data_file_jsonl")
        prompt_file = self.paths.get("prompt_file")
        if isinstance(data_file, Path):
            self._safe_unlink(data_file)
            self._prune_empty_parent(data_file.parent)
        if isinstance(prompt_file, Path):
            self._safe_unlink(prompt_file)
            self._prune_empty_parent(prompt_file.parent)

        for key in ("head_rgb_frames", "wrist_rgb_frames", "head_depth_frames", "wrist_depth_frames"):
            p = self.paths.get(key)
            if isinstance(p, Path):
                self._safe_rmtree(p)
                self._prune_empty_parent(p.parent)

    def stop(self, *, discard: bool = False):
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

        if discard:
            self._discard_episode_files()
            summary = {
                "episode_index": int(self.episode_index),
                "num_frames": int(self.frame_index),
                "data_format": self.data_format,
                "dataset_root": str(self.root),
                "task": self.prompt,
                "dropped_frames": int(self._dropped_frames),
                "discarded": True,
            }
            self._sample_queue = queue.Queue(maxsize=self.queue_maxsize)
            return summary

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
                # Preview depth image streams are disabled.
                "head_depth": None,
                "wrist_depth": None,
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
            "rgb_storage": f"{self.rgb_image_format}_frames",
            "depth_preview_storage": "disabled",
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
        return self._bridge.lookup_transform(target_frame, source_frame, *args, **kwargs)


class RosTopicObservationNode(Node):
    """ROS2 subscriber node for observation-only data path."""

    def __init__(self):
        super().__init__("stretch_obs_v8_2mode")
        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._prefer_cv_bridge = True
        self._last_cb_error_t = 0.0

        self.head_rgb: np.ndarray | None = None
        self.wrist_rgb: np.ndarray | None = None
        self.head_depth: np.ndarray | None = None
        self.wrist_depth: np.ndarray | None = None
        self.head_info: dict[str, Any] | None = None
        self.wrist_info: dict[str, Any] | None = None

        self.actual_qpos: list[float] | None = None
        self.base_pose_xytheta: list[float] | None = None
        self.joint_state_name: list[str] = []
        self.joint_state_position: list[float] = []
        self.joint_state_velocity: list[float] = []
        self.joint_state_effort: list[float] = []
        self.imu_mobile: dict[str, Any] | None = None
        self.imu_wrist: dict[str, Any] | None = None
        self.imu_cam_accel: dict[str, Any] | None = None
        self.imu_cam_gyro: dict[str, Any] | None = None
        self.mag_mobile: dict[str, Any] | None = None
        self.battery: dict[str, Any] | None = None
        self.odom: dict[str, Any] | None = None
        self._base_lin = 0.0
        self._base_ang = 0.0
        self._stamp_ns: dict[str, int | None] = {
            "head_rgb": None,
            "wrist_rgb": None,
            "head_depth": None,
            "wrist_depth": None,
            "joint_state": None,
            "odom": None,
            "head_info": None,
            "wrist_info": None,
        }
        self._recv_wall_ns: dict[str, int | None] = {k: None for k in self._stamp_ns}
        self._ros_minus_wall_ns: int | None = None
        self._joint_hist: deque[tuple[int, list[float]]] = deque(maxlen=512)
        self._base_pose_hist: deque[tuple[int, list[float]]] = deque(maxlen=512)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = qos_profile_sensor_data
        self.create_subscription(Image, HEAD_RGB_TOPIC, self._head_rgb_cb, sensor_qos)
        self.create_subscription(Image, WRIST_RGB_TOPIC, self._wrist_rgb_cb, sensor_qos)
        self.create_subscription(Image, HEAD_DEPTH_TOPIC, self._head_depth_cb, sensor_qos)
        self.create_subscription(Image, WRIST_DEPTH_TOPIC, self._wrist_depth_cb, sensor_qos)
        self.create_subscription(CameraInfo, HEAD_CAMERA_INFO_TOPIC, self._head_info_cb, sensor_qos)
        self.create_subscription(CameraInfo, WRIST_CAMERA_INFO_TOPIC, self._wrist_info_cb, sensor_qos)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_cb, sensor_qos)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, sensor_qos)
        self.create_subscription(Imu, IMU_MOBILE_BASE_TOPIC, self._imu_mobile_cb, sensor_qos)
        self.create_subscription(Imu, IMU_WRIST_TOPIC, self._imu_wrist_cb, sensor_qos)
        self.create_subscription(Imu, IMU_CAMERA_ACCEL_TOPIC, self._imu_cam_accel_cb, sensor_qos)
        self.create_subscription(Imu, IMU_CAMERA_GYRO_TOPIC, self._imu_cam_gyro_cb, sensor_qos)
        self.create_subscription(MagneticField, MAGNETOMETER_TOPIC, self._mag_cb, sensor_qos)
        self.create_subscription(BatteryState, BATTERY_TOPIC, self._battery_cb, sensor_qos)

    def _log_cb_error(self, cb_name: str, exc: Exception) -> None:
        now = time.time()
        if now - self._last_cb_error_t < 1.0:
            return
        self._last_cb_error_t = now
        print(f"[ros_obs] callback {cb_name} error: {exc}", file=sys.stderr, flush=True)

    @staticmethod
    def _camera_info_dict(msg: CameraInfo) -> dict[str, Any]:
        return {
            "width": int(msg.width),
            "height": int(msg.height),
            "k": [float(v) for v in msg.k],
            "d": [float(v) for v in msg.d],
            "r": [float(v) for v in msg.r],
            "p": [float(v) for v in msg.p],
            "distortion_model": str(msg.distortion_model),
        }

    @staticmethod
    def _imu_to_dict(msg: Imu) -> dict[str, Any]:
        return {
            "orientation": [
                float(msg.orientation.x),
                float(msg.orientation.y),
                float(msg.orientation.z),
                float(msg.orientation.w),
            ],
            "angular_velocity": [
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ],
            "linear_acceleration": [
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
            ],
        }

    @staticmethod
    def _mag_to_dict(msg: MagneticField) -> dict[str, Any]:
        return {
            "magnetic_field": [
                float(msg.magnetic_field.x),
                float(msg.magnetic_field.y),
                float(msg.magnetic_field.z),
            ]
        }

    @staticmethod
    def _battery_to_dict(msg: BatteryState) -> dict[str, Any]:
        return {
            "voltage": float(msg.voltage),
            "current": float(msg.current),
            "charge": float(msg.charge),
            "capacity": float(msg.capacity),
            "percentage": float(msg.percentage),
            "power_supply_status": int(msg.power_supply_status),
            "power_supply_health": int(msg.power_supply_health),
            "power_supply_technology": int(msg.power_supply_technology),
            "present": bool(msg.present),
        }

    @staticmethod
    def _msg_stamp_ns(msg: Any) -> int | None:
        try:
            hdr = getattr(msg, "header", None)
            st = getattr(hdr, "stamp", None)
            sec = int(getattr(st, "sec"))
            nsec = int(getattr(st, "nanosec"))
            if sec < 0 or nsec < 0:
                return None
            return sec * 1_000_000_000 + nsec
        except Exception:
            return None

    def _update_stream_stamp(self, key: str, msg: Any, *, recv_wall_ns: int | None = None) -> int | None:
        stamp_ns = self._msg_stamp_ns(msg)
        if recv_wall_ns is None:
            recv_wall_ns = time.time_ns()
        with self._lock:
            self._stamp_ns[key] = stamp_ns
            self._recv_wall_ns[key] = int(recv_wall_ns)
            if stamp_ns is not None:
                self._ros_minus_wall_ns = int(stamp_ns - int(recv_wall_ns))
        return stamp_ns

    @staticmethod
    def _history_at_or_before(
        history: deque[tuple[int, list[float]]],
        ref_stamp_ns: int | None,
    ) -> tuple[list[float] | None, int | None]:
        if ref_stamp_ns is None or len(history) == 0:
            if len(history) == 0:
                return None, None
            ts, val = history[-1]
            return list(val), int(ts)
        ref = int(ref_stamp_ns)
        for ts, val in reversed(history):
            if int(ts) <= ref:
                return list(val), int(ts)
        ts, val = history[0]
        return list(val), int(ts)

    @staticmethod
    def _extract_qpos(msg: JointState, *, base_lin: float, base_ang: float) -> list[float] | None:
        try:
            idx = {str(name): i for i, name in enumerate(msg.name)}

            def _pick(names: list[str]) -> float:
                for n in names:
                    j = idx.get(n)
                    if isinstance(j, int) and 0 <= j < len(msg.position):
                        return float(msg.position[j])
                raise KeyError(str(names))

            arm_lift = _pick(["joint_lift", "lift"])
            arm_l0 = _pick(["joint_arm_l0", "arm"])
            arm_extension = float(4.0 * arm_l0)
            wrist_yaw = _pick(["joint_wrist_yaw", "wrist_yaw"])
            wrist_pitch = _pick(["joint_wrist_pitch", "wrist_pitch"])
            wrist_roll = _pick(["joint_wrist_roll", "wrist_roll"])
            head_pan = _pick(["joint_head_pan", "head_pan"])
            head_tilt = _pick(["joint_head_tilt", "head_tilt"])
            gripper = _pick(["joint_gripper_finger_left", "gripper_finger_left", "joint_gripper_finger_right"])
        except Exception:
            return None

        return [
            arm_extension,
            arm_lift,
            wrist_yaw,
            wrist_pitch,
            wrist_roll,
            head_pan,
            head_tilt,
            gripper,
            float(base_lin),
            float(base_ang),
        ]

    @staticmethod
    def _decode_raw_image_rgb(msg: Image) -> np.ndarray | None:
        try:
            width = int(msg.width)
            height = int(msg.height)
            step = int(msg.step)
            encoding = str(msg.encoding).lower()
            is_big = int(msg.is_bigendian)
            _ = is_big
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        try:
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        except Exception:
            return None

        def _reshape(bytes_per_pixel: int) -> np.ndarray | None:
            row_stride = step if step > 0 else width * bytes_per_pixel
            expected = row_stride * height
            if buf.size < expected:
                return None
            raw = buf[:expected].reshape(height, row_stride)
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
    def _decode_raw_depth_m(msg: Image) -> np.ndarray | None:
        try:
            width = int(msg.width)
            height = int(msg.height)
            step = int(msg.step)
            encoding = str(msg.encoding).lower()
            is_big = int(msg.is_bigendian)
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        try:
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        except Exception:
            return None

        if encoding in {"16uc1", "mono16"}:
            bpp = 2
            row_stride = step if step > 0 else width * bpp
            expected = row_stride * height
            if buf.size < expected:
                return None
            raw = buf[:expected].reshape(height, row_stride)
            pix = raw[:, : width * bpp]
            dtype = np.dtype(">u2") if is_big else np.dtype("<u2")
            depth_u16 = pix.view(dtype).reshape(height, width)
            return depth_u16.astype(np.float32) / 1000.0

        if encoding in {"32fc1"}:
            bpp = 4
            row_stride = step if step > 0 else width * bpp
            expected = row_stride * height
            if buf.size < expected:
                return None
            raw = buf[:expected].reshape(height, row_stride)
            pix = raw[:, : width * bpp]
            dtype = np.dtype(">f4") if is_big else np.dtype("<f4")
            depth_f32 = pix.view(dtype).reshape(height, width).astype(np.float32)
            depth_f32[~np.isfinite(depth_f32)] = 0.0
            return depth_f32

        return None

    def _head_rgb_cb(self, msg: Image) -> None:
        recv_wall_ns = time.time_ns()
        rgb = None
        if self._prefer_cv_bridge:
            try:
                rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8").copy()
            except Exception as exc:
                self._prefer_cv_bridge = False
                self._log_cb_error("head_rgb(cv_bridge)", exc)
        if rgb is None:
            rgb = self._decode_raw_image_rgb(msg)
        if rgb is None:
            return
        if ROS_TOPICS_ROTATE_HEAD_90_CW:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        self._update_stream_stamp("head_rgb", msg, recv_wall_ns=recv_wall_ns)
        with self._lock:
            self.head_rgb = rgb

    def _wrist_rgb_cb(self, msg: Image) -> None:
        recv_wall_ns = time.time_ns()
        rgb = None
        if self._prefer_cv_bridge:
            try:
                rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8").copy()
            except Exception as exc:
                self._prefer_cv_bridge = False
                self._log_cb_error("wrist_rgb(cv_bridge)", exc)
        if rgb is None:
            rgb = self._decode_raw_image_rgb(msg)
        if rgb is None:
            return
        self._update_stream_stamp("wrist_rgb", msg, recv_wall_ns=recv_wall_ns)
        with self._lock:
            self.wrist_rgb = rgb

    def _head_depth_cb(self, msg: Image) -> None:
        recv_wall_ns = time.time_ns()
        depth = None
        if self._prefer_cv_bridge:
            try:
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if depth is not None:
                    depth = np.asarray(depth)
                    if depth.dtype == np.uint16:
                        depth = depth.astype(np.float32) / 1000.0
                    else:
                        depth = depth.astype(np.float32)
            except Exception as exc:
                self._prefer_cv_bridge = False
                self._log_cb_error("head_depth(cv_bridge)", exc)
        if depth is None:
            depth = self._decode_raw_depth_m(msg)
        if depth is None:
            return
        if ROS_TOPICS_ROTATE_HEAD_90_CW:
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        self._update_stream_stamp("head_depth", msg, recv_wall_ns=recv_wall_ns)
        with self._lock:
            self.head_depth = depth

    def _wrist_depth_cb(self, msg: Image) -> None:
        recv_wall_ns = time.time_ns()
        depth = None
        if self._prefer_cv_bridge:
            try:
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if depth is not None:
                    depth = np.asarray(depth)
                    if depth.dtype == np.uint16:
                        depth = depth.astype(np.float32) / 1000.0
                    else:
                        depth = depth.astype(np.float32)
            except Exception as exc:
                self._prefer_cv_bridge = False
                self._log_cb_error("wrist_depth(cv_bridge)", exc)
        if depth is None:
            depth = self._decode_raw_depth_m(msg)
        if depth is None:
            return
        self._update_stream_stamp("wrist_depth", msg, recv_wall_ns=recv_wall_ns)
        with self._lock:
            self.wrist_depth = depth

    def _head_info_cb(self, msg: CameraInfo) -> None:
        try:
            info = self._camera_info_dict(msg)
            self._update_stream_stamp("head_info", msg)
            with self._lock:
                self.head_info = info
        except Exception as exc:
            self._log_cb_error("head_camera_info", exc)

    def _wrist_info_cb(self, msg: CameraInfo) -> None:
        try:
            info = self._camera_info_dict(msg)
            self._update_stream_stamp("wrist_info", msg)
            with self._lock:
                self.wrist_info = info
        except Exception as exc:
            self._log_cb_error("wrist_camera_info", exc)

    def _joint_cb(self, msg: JointState) -> None:
        try:
            stamp_ns = self._update_stream_stamp("joint_state", msg)
            with self._lock:
                base_lin = float(self._base_lin)
                base_ang = float(self._base_ang)
            measured = self._extract_qpos(msg, base_lin=base_lin, base_ang=base_ang)
            with self._lock:
                self.joint_state_name = [str(v) for v in msg.name]
                self.joint_state_position = [float(v) for v in msg.position]
                self.joint_state_velocity = [float(v) for v in msg.velocity]
                self.joint_state_effort = [float(v) for v in msg.effort]
                if measured is not None:
                    self.actual_qpos = measured
                    if stamp_ns is not None:
                        self._joint_hist.append((int(stamp_ns), list(measured)))
        except Exception as exc:
            self._log_cb_error("joint_state", exc)

    def _odom_cb(self, msg: Odometry) -> None:
        try:
            stamp_ns = self._update_stream_stamp("odom", msg)
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            z = float(msg.pose.pose.position.z)
            qx = float(msg.pose.pose.orientation.x)
            qy = float(msg.pose.pose.orientation.y)
            qz = float(msg.pose.pose.orientation.z)
            qw = float(msg.pose.pose.orientation.w)
            lin_x = float(msg.twist.twist.linear.x)
            lin_y = float(msg.twist.twist.linear.y)
            lin_z = float(msg.twist.twist.linear.z)
            ang_x = float(msg.twist.twist.angular.x)
            ang_y = float(msg.twist.twist.angular.y)
            ang_z = float(msg.twist.twist.angular.z)
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            theta = float(np.arctan2(siny_cosp, cosy_cosp))
            odom = {
                "position": [x, y, z],
                "orientation": [qx, qy, qz, qw],
                "linear_velocity": [lin_x, lin_y, lin_z],
                "angular_velocity": [ang_x, ang_y, ang_z],
            }
            with self._lock:
                self.base_pose_xytheta = [x, y, theta]
                self._base_lin = lin_x
                self._base_ang = ang_z
                self.odom = odom
                if self.actual_qpos is not None and len(self.actual_qpos) >= 10:
                    self.actual_qpos[8] = lin_x
                    self.actual_qpos[9] = ang_z
                if stamp_ns is not None:
                    self._base_pose_hist.append((int(stamp_ns), [x, y, theta]))
        except Exception as exc:
            self._log_cb_error("odom", exc)

    def _imu_mobile_cb(self, msg: Imu) -> None:
        try:
            with self._lock:
                self.imu_mobile = self._imu_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("imu_mobile", exc)

    def _imu_wrist_cb(self, msg: Imu) -> None:
        try:
            with self._lock:
                self.imu_wrist = self._imu_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("imu_wrist", exc)

    def _imu_cam_accel_cb(self, msg: Imu) -> None:
        try:
            with self._lock:
                self.imu_cam_accel = self._imu_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("imu_cam_accel", exc)

    def _imu_cam_gyro_cb(self, msg: Imu) -> None:
        try:
            with self._lock:
                self.imu_cam_gyro = self._imu_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("imu_cam_gyro", exc)

    def _mag_cb(self, msg: MagneticField) -> None:
        try:
            with self._lock:
                self.mag_mobile = self._mag_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("mag_mobile", exc)

    def _battery_cb(self, msg: BatteryState) -> None:
        try:
            with self._lock:
                self.battery = self._battery_to_dict(msg)
        except Exception as exc:
            self._log_cb_error("battery", exc)

    def get_snapshot(self):
        with self._lock:
            hr = None if self.head_rgb is None else self.head_rgb.copy()
            wr = None if self.wrist_rgb is None else self.wrist_rgb.copy()
            hd = None if self.head_depth is None else self.head_depth.copy()
            wd = None if self.wrist_depth is None else self.wrist_depth.copy()
            hi = None if self.head_info is None else dict(self.head_info)
            wi = None if self.wrist_info is None else dict(self.wrist_info)
        return hr, wr, hd, wd, hi, wi

    def get_observation_snapshot(self):
        with self._lock:
            return {
                "actual_qpos": None if self.actual_qpos is None else list(self.actual_qpos),
                "base_pose_xytheta": None if self.base_pose_xytheta is None else list(self.base_pose_xytheta),
                "joint_state_name": list(self.joint_state_name),
                "joint_state_position": list(self.joint_state_position),
                "joint_state_velocity": list(self.joint_state_velocity),
                "joint_state_effort": list(self.joint_state_effort),
                "imu_mobile": None if self.imu_mobile is None else dict(self.imu_mobile),
                "imu_wrist": None if self.imu_wrist is None else dict(self.imu_wrist),
                "imu_cam_accel": None if self.imu_cam_accel is None else dict(self.imu_cam_accel),
                "imu_cam_gyro": None if self.imu_cam_gyro is None else dict(self.imu_cam_gyro),
                "mag_mobile": None if self.mag_mobile is None else dict(self.mag_mobile),
                "battery": None if self.battery is None else dict(self.battery),
                "odom": None if self.odom is None else dict(self.odom),
                "stamp_ns": dict(self._stamp_ns),
                "recv_wall_ns": dict(self._recv_wall_ns),
                "ros_minus_wall_ns": self._ros_minus_wall_ns,
            }

    def get_timing_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stamp_ns": dict(self._stamp_ns),
                "recv_wall_ns": dict(self._recv_wall_ns),
                "ros_minus_wall_ns": self._ros_minus_wall_ns,
            }

    def get_aligned_observation_snapshot(self, reference_stamp_ns: int | None = None) -> dict[str, Any]:
        with self._lock:
            if reference_stamp_ns is None:
                reference_stamp_ns = (
                    self._stamp_ns.get("head_rgb")
                    or self._stamp_ns.get("head_depth")
                    or self._stamp_ns.get("wrist_rgb")
                    or self._stamp_ns.get("wrist_depth")
                )

            aligned_qpos, joint_stamp_ns = self._history_at_or_before(self._joint_hist, reference_stamp_ns)
            aligned_pose, odom_stamp_ns = self._history_at_or_before(self._base_pose_hist, reference_stamp_ns)
            if aligned_qpos is None:
                aligned_qpos = None if self.actual_qpos is None else list(self.actual_qpos)
            if aligned_pose is None:
                aligned_pose = None if self.base_pose_xytheta is None else list(self.base_pose_xytheta)

            return {
                "reference_stamp_ns": reference_stamp_ns,
                "aligned_joint_stamp_ns": joint_stamp_ns,
                "aligned_odom_stamp_ns": odom_stamp_ns,
                "actual_qpos": aligned_qpos,
                "base_pose_xytheta": aligned_pose,
            }


class RosTopicObservationClient:
    """Threaded ROS2 observation client used when source is `ros_topic`."""

    def __init__(self):
        if _RCLPY_IMPORT_ERROR is not None:
            raise RuntimeError(f"ROS2 imports unavailable: {_RCLPY_IMPORT_ERROR}")
        self._node: RosTopicObservationNode | None = None
        self._executor: MultiThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return bool(self._connected and self._node is not None)

    def connect(self, timeout_s: float = ROS_TOPICS_CONNECT_TIMEOUT_S) -> None:
        if self.connected:
            return
        if not hasattr(rclpy, "ok") or not rclpy.ok():
            rclpy.init(args=None)

        self._node = RosTopicObservationNode()
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if self._node is None:
                break
            hr, _, _, _, _, _ = self._node.get_snapshot()
            obs = self._node.get_observation_snapshot()
            aq = obs.get("actual_qpos")
            if hr is not None and isinstance(aq, list) and len(aq) > 0:
                self._connected = True
                return
            time.sleep(0.05)
        self.close()
        raise RuntimeError("Timed out waiting for ROS2 topic observations")

    def get_snapshot(self):
        if self._node is None:
            return None, None, None, None, None, None
        return self._node.get_snapshot()

    def get_observation_snapshot(self) -> dict[str, Any]:
        if self._node is None:
            return {}
        return self._node.get_observation_snapshot()

    def get_timing_snapshot(self) -> dict[str, Any]:
        if self._node is None:
            return {}
        return self._node.get_timing_snapshot()

    def get_aligned_observation_snapshot(self, reference_stamp_ns: int | None = None) -> dict[str, Any]:
        if self._node is None:
            return {}
        return self._node.get_aligned_observation_snapshot(reference_stamp_ns=reference_stamp_ns)

    def lookup_transform(self, target_frame: str, source_frame: str, timeout_s: float = 1.0):
        if self._node is None:
            raise RuntimeError("ROS2 observation client not connected")
        return self._node.tf_buffer.lookup_transform(
            str(target_frame),
            str(source_frame),
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=float(timeout_s)),
        )

    def get_clock(self):
        if self._node is None:
            return _DummyClock()
        return self._node.get_clock()

    def close(self) -> None:
        self._connected = False
        executor = self._executor
        node = self._node
        spin_thread = self._spin_thread
        self._executor = None
        self._node = None
        self._spin_thread = None

        if executor is not None:
            try:
                executor.shutdown(timeout_sec=1.0)
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=2.0)


class StretchAIDemoBridge:
    JOINT_LIMITS = list(ROBOT_CONTROL_LIMITS)

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

    JOINT_STATE_NAMES = list(ROBOT_JOINT_STATE_NAMES)
    CONTROLLABLE_JOINT_NAMES = list(ROBOT_CONTROLLABLE_JOINT_NAMES)
    BASE_CONTROLLABLE_NAMES = list(ROBOT_BASE_CONTROLLABLE_NAMES)
    BASE_STATE_NAMES = list(ROBOT_BASE_STATE_NAMES)

    def __init__(self):
        self._lock = threading.Lock()
        self.JOINT_LIMITS = [tuple(v) for v in ROBOT_CONTROL_LIMITS]
        self.JOINT_STATE_NAMES = list(ROBOT_JOINT_STATE_NAMES)
        self.CONTROLLABLE_JOINT_NAMES = list(ROBOT_CONTROLLABLE_JOINT_NAMES)
        self.BASE_CONTROLLABLE_NAMES = list(ROBOT_BASE_CONTROLLABLE_NAMES)
        self.BASE_STATE_NAMES = list(ROBOT_BASE_STATE_NAMES)
        self.JOINT_LIMITS_BY_NAME = dict(ROBOT_JOINT_LIMITS_BY_NAME)
        self.JOINT_UNITS_BY_NAME = dict(ROBOT_URDF_JOINT_UNITS)
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
        self._base_y0: float | None = None
        self.command_base_pose_xytheta: list[float] | None = [0.0, 0.0, 0.0]
        self.command_base_pose_last_wall_time: float | None = None
        self._command_latched_qpos10: list[float] = [0.0] * 10
        self._command_latched_pose_xytheta: list[float] = [0.0, 0.0, 0.0]
        self._command_latched_manip_base_x: float = 0.0
        self._did_initialize_default_pose = False

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
        self._image_source: str = str(STRETCH_AI_DEFAULT_OBS_SOURCE)
        self._ros_obs_client: RosTopicObservationClient | None = None
        self._next_ros_connect_t: float = 0.0
        self._command_history: deque[dict[str, Any]] = deque(maxlen=COMMAND_HISTORY_MAXLEN)

    def _latest_ros_observation_stamp_ns(self) -> int | None:
        client = self._ros_obs_client
        if client is None or not client.connected:
            return None
        try:
            timing = client.get_timing_snapshot()
        except Exception:
            return None
        stamp_map = timing.get("stamp_ns") if isinstance(timing.get("stamp_ns"), dict) else {}
        latest = None
        for key in ("joint_state", "odom", "head_rgb", "head_depth", "wrist_rgb", "wrist_depth"):
            v = stamp_map.get(key)
            if isinstance(v, int):
                latest = int(v) if latest is None else max(int(latest), int(v))
        return latest

    def _refresh_state_from_ros_topic(
        self,
        *,
        require_newer_than_ns: int | None = None,
        timeout_s: float = 0.0,
    ) -> bool:
        """Pull latest ROS-topic observation into cached state.

        If `require_newer_than_ns` is provided, poll until a newer ROS stamp arrives
        or timeout. This reduces stale-state usage after sending chunked commands.
        """
        client = self._ros_obs_client
        if client is None or not client.connected:
            return False
        deadline = time.time() + max(0.0, float(timeout_s))
        while True:
            try:
                obs = client.get_observation_snapshot()
            except Exception:
                return False
            stamp_map = obs.get("stamp_ns") if isinstance(obs.get("stamp_ns"), dict) else {}
            latest_stamp = None
            for key in ("joint_state", "odom", "head_rgb", "head_depth", "wrist_rgb", "wrist_depth"):
                v = stamp_map.get(key)
                if isinstance(v, int):
                    latest_stamp = int(v) if latest_stamp is None else max(int(latest_stamp), int(v))
            if (
                require_newer_than_ns is not None
                and isinstance(latest_stamp, int)
                and int(latest_stamp) <= int(require_newer_than_ns)
                and time.time() < deadline
            ):
                time.sleep(0.01)
                continue
            try:
                with self._lock:
                    aq = obs.get("actual_qpos")
                    if isinstance(aq, list) and len(aq) > 0:
                        self.actual_qpos = [float(v) for v in aq[:10]] + [0.0] * max(0, 10 - len(aq))
                    pose = obs.get("base_pose_xytheta")
                    if isinstance(pose, list) and len(pose) >= 3:
                        self._base_x = float(pose[0])
                        self._base_y = float(pose[1])
                        if self._base_y0 is None:
                            self._base_y0 = float(pose[1])
                        self._base_theta = float(pose[2])
                    # Keep command pose latched to sent-command history.
                    # Only initialize once from measurements at startup.
                    if self.command_base_pose_xytheta is None:
                        self.command_base_pose_xytheta = [
                            self._base_x,
                            self._base_y0 if self._base_y0 is not None else self._base_y,
                            self._base_theta,
                        ]
                    if self.actual_qpos is not None and len(self.actual_qpos) >= 10:
                        if self.qpos is None:
                            self.qpos = list(self.actual_qpos)
                            self.qpos[8] = 0.0
                            self.qpos[9] = 0.0
                        if self.published_qpos is None:
                            self.published_qpos = list(self.qpos)
                return True
            except Exception:
                return False

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

    def _get_ros_minus_wall_ns(self) -> int | None:
        client = self._ros_obs_client
        if client is None or not client.connected:
            return None
        try:
            timing = client.get_timing_snapshot()
        except Exception:
            return None
        val = timing.get("ros_minus_wall_ns")
        return int(val) if isinstance(val, int) else None

    @staticmethod
    def _mid_joint_limit(limit_pair: tuple[float, float]) -> float:
        lo, hi = float(limit_pair[0]), float(limit_pair[1])
        return 0.5 * (lo + hi)

    def _default_command_qpos10(self) -> list[float]:
        q8 = list(DEFAULT_INIT_CMD_QPOS8) if isinstance(DEFAULT_INIT_CMD_QPOS8, (list, tuple)) else []
        if len(q8) < 8:
            q8 = q8 + [0.0] * (8 - len(q8))
        return [
            float(np.clip(float(q8[0]), self.JOINT_LIMITS[0][0], self.JOINT_LIMITS[0][1])),
            float(np.clip(float(q8[1]), self.JOINT_LIMITS[1][0], self.JOINT_LIMITS[1][1])),
            float(np.clip(float(q8[2]), self.JOINT_LIMITS[2][0], self.JOINT_LIMITS[2][1])),
            float(np.clip(float(q8[3]), self.JOINT_LIMITS[3][0], self.JOINT_LIMITS[3][1])),
            float(np.clip(float(q8[4]), self.JOINT_LIMITS[4][0], self.JOINT_LIMITS[4][1])),
            float(np.clip(float(q8[5]), self.JOINT_LIMITS[5][0], self.JOINT_LIMITS[5][1])),
            float(np.clip(float(q8[6]), self.JOINT_LIMITS[6][0], self.JOINT_LIMITS[6][1])),
            float(np.clip(float(q8[7]), self.JOINT_LIMITS[7][0], self.JOINT_LIMITS[7][1])),
            0.0,
            0.0,
        ]

    def _initialize_default_command_pose(self, *, send_to_robot: bool = True) -> None:
        if self._did_initialize_default_pose:
            return
        q_default = self._default_command_qpos10()
        pose_default = self.get_measured_base_pose_xytheta()
        if not (isinstance(pose_default, list) and len(pose_default) >= 3):
            pose_default = [0.0, 0.0, 0.0]
        pose_default = [float(pose_default[0]), float(pose_default[1]), float(pose_default[2])]
        with self._lock:
            self.qpos = list(q_default)
            self.published_qpos = list(q_default)
            self.command_base_pose_xytheta = list(pose_default)
            self.command_base_pose_last_wall_time = time.time()
            self._manip_base_x = 0.0
            if self._base_y0 is None:
                self._base_y0 = float(pose_default[1])
            self._base_linear_cmd = 0.0
            self._base_angular_cmd = 0.0
            self._needs_mode_retry = False
            self._command_latched_qpos10 = list(q_default)
            self._command_latched_pose_xytheta = list(pose_default)
            self._command_latched_manip_base_x = 0.0
        self._record_command_event(
            "init_default_target",
            qpos10=q_default,
            command_pose_xytheta=pose_default,
            manip_base_x_cmd=0.0,
        )

        if bool(send_to_robot):
            joint6 = [
                0.0,               # manipulation base_x
                q_default[1],      # lift
                q_default[0],      # arm extension
                q_default[2],      # wrist_yaw
                q_default[3],      # wrist_pitch
                q_default[4],      # wrist_roll
            ]
            ok = self.execute_arm_to(
                joint6,
                gripper=q_default[7],
                head=[q_default[5], q_default[6]],
                blocking=True,
                timeout_s=12.0,
                reliable=False,
            )
            if not ok:
                print(
                    "[stretch_ai_bridge] default init pose command failed; "
                    "continuing with local command targets.",
                    file=sys.stderr,
                )
        self._did_initialize_default_pose = True

    def move_to_startup_home_pose(self, *, timeout_s: float = ACTION_MOVE_TIMEOUT_HOME_S) -> bool:
        """Move robot to the same startup/default joint target used at UI load."""
        q_default = self._default_command_qpos10()
        pose_default = self.get_measured_base_pose_xytheta()
        if not (isinstance(pose_default, list) and len(pose_default) >= 3):
            pose_default = [0.0, 0.0, 0.0]
        pose_default = [float(pose_default[0]), float(pose_default[1]), float(pose_default[2])]

        with self._lock:
            self.qpos = list(q_default)
            self.published_qpos = list(q_default)
            self.command_base_pose_xytheta = list(pose_default)
            self.command_base_pose_last_wall_time = time.time()
            self._manip_base_x = 0.0
            if self._base_y0 is None:
                self._base_y0 = float(pose_default[1])
            self._base_linear_cmd = 0.0
            self._base_angular_cmd = 0.0
            self._needs_mode_retry = False
            self._command_latched_qpos10 = list(q_default)
            self._command_latched_pose_xytheta = list(pose_default)
            self._command_latched_manip_base_x = 0.0

        self._record_command_event(
            "manual_home_target",
            qpos10=q_default,
            command_pose_xytheta=pose_default,
            manip_base_x_cmd=0.0,
        )

        joint6 = [
            0.0,               # manipulation base_x
            q_default[1],      # lift
            q_default[0],      # arm extension
            q_default[2],      # wrist_yaw
            q_default[3],      # wrist_pitch
            q_default[4],      # wrist_roll
        ]
        ok = self.execute_arm_to(
            joint6,
            gripper=q_default[7],
            head=[q_default[5], q_default[6]],
            blocking=True,
            timeout_s=float(timeout_s),
            reliable=False,
        )
        return bool(ok)

    def _record_command_event(
        self,
        reason: str,
        *,
        qpos10: list[float] | None = None,
        command_pose_xytheta: list[float] | None = None,
        manip_base_x_cmd: float | None = None,
        wall_time_ns: int | None = None,
    ) -> None:
        with self._lock:
            if isinstance(qpos10, list) and len(qpos10) > 0:
                q = [float(v) for v in qpos10[:10]]
            else:
                q = list(self._command_latched_qpos10)
            if len(q) < 10:
                q = q + [0.0] * (10 - len(q))
            q[8] = 0.0
            q[9] = 0.0

            if isinstance(command_pose_xytheta, list) and len(command_pose_xytheta) >= 3:
                pose = [
                    float(command_pose_xytheta[0]),
                    float(command_pose_xytheta[1]),
                    float(command_pose_xytheta[2]),
                ]
            else:
                pose = list(self._command_latched_pose_xytheta)
            if isinstance(manip_base_x_cmd, (int, float)):
                manip_bx_cmd = float(manip_base_x_cmd)
            else:
                manip_bx_cmd = float(self._command_latched_manip_base_x)
            self._command_latched_qpos10 = [float(v) for v in q[:10]]
            self._command_latched_pose_xytheta = [float(pose[0]), float(pose[1]), float(pose[2])]
            self._command_latched_manip_base_x = float(manip_bx_cmd)
        wall_ns = int(time.time_ns()) if wall_time_ns is None else int(wall_time_ns)
        ros_minus_wall_ns = self._get_ros_minus_wall_ns()
        ros_est_ns = int(wall_ns + ros_minus_wall_ns) if ros_minus_wall_ns is not None else None
        action11 = [float(v) for v in q[:8]] + [float(pose[0]), float(pose[1]), float(pose[2])]
        self._command_history.append(
            {
                "reason": str(reason),
                "wall_time_ns": wall_ns,
                "ros_time_ns_est": ros_est_ns,
                "qpos10": [float(v) for v in q[:10]],
                "command_pose_xytheta": [float(pose[0]), float(pose[1]), float(pose[2])],
                "manip_base_x_cmd": float(manip_bx_cmd),
                "action_command11": action11,
            }
        )

    def _select_command_event(self, reference_stamp_ns: int | None = None) -> dict[str, Any] | None:
        if len(self._command_history) == 0:
            return None
        # Keep action_command as a latched signal: update when command is sent,
        # hold that value until the next command is sent.
        # This avoids apparent lag introduced by cross-clock stamp matching.
        return dict(self._command_history[-1])

    def _worker_script_path(self) -> Path:
        return Path(__file__).with_name("stretch_ai_bridge_worker.py")

    def _worker_cmd(self) -> tuple[list[str], dict[str, str]]:
        worker_path = self._worker_script_path()
        if not worker_path.exists():
            raise RuntimeError(f"Missing worker script: {worker_path}")

        cmd = [STRETCH_AI_WORKER_LAUNCHER, "run", "-n", STRETCH_AI_ENV_NAME, STRETCH_AI_WORKER_PYTHON, "-u"]
        cmd.append(str(worker_path))
        cmd += ["--jpeg-quality", str(STRETCH_AI_WORKER_JPEG_QUALITY)]
        cmd += ["--obs-source", "bridge"]
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

    @staticmethod
    def _worker_motion_tuning_params() -> dict[str, float]:
        return {
            "nav_blocking_pos_tol_m": float(WORKER_TUNE_NAV_BLOCKING_POS_TOL_M),
            "nav_blocking_yaw_tol_deg": float(WORKER_TUNE_NAV_BLOCKING_YAW_TOL_DEG),
            "arm_to_tol_base_x_m": float(WORKER_TUNE_ARM_TO_TOL_BASE_X_M),
            "arm_to_tol_lift_m": float(WORKER_TUNE_ARM_TO_TOL_LIFT_M),
            "arm_to_tol_arm_m": float(WORKER_TUNE_ARM_TO_TOL_ARM_M),
            "arm_to_tol_wrist_rad": float(WORKER_TUNE_ARM_TO_TOL_WRIST_RAD),
            "arm_to_tol_gripper": float(WORKER_TUNE_ARM_TO_TOL_GRIPPER),
            "arm_to_tol_head_rad": float(WORKER_TUNE_ARM_TO_TOL_HEAD_RAD),
            "arm_to_mode_wait_timeout_s": float(WORKER_TUNE_ARM_TO_MODE_WAIT_TIMEOUT_S),
            "arm_to_stalled_check_min_s": float(WORKER_TUNE_ARM_TO_STALLED_CHECK_MIN_S),
            "arm_to_stalled_check_max_s": float(WORKER_TUNE_ARM_TO_STALLED_CHECK_MAX_S),
            "arm_to_stalled_check_scale": float(WORKER_TUNE_ARM_TO_STALLED_CHECK_SCALE),
        }

    def _push_worker_motion_tuning(self) -> None:
        rpc = self._rpc
        if rpc is None:
            return
        params = self._worker_motion_tuning_params()
        try:
            rpc.request(
                "set_motion_tuning",
                params,
                timeout_s=max(float(WORKER_TUNE_SET_RPC_TIMEOUT_S), float(STRETCH_AI_RPC_TIMEOUT_S)),
            )
        except Exception as exc:
            now = time.time()
            if now - self._last_cmd_error_t > 2.0:
                self._last_cmd_error_t = now
                print(f"[stretch_ai_bridge] set_motion_tuning failed: {exc}", file=sys.stderr)

    def _refresh_joint_model_from_ros_topics(self) -> None:
        client = self._ros_obs_client
        if client is None or not client.connected:
            return
        try:
            obs = client.get_observation_snapshot()
        except Exception:
            return
        ros_joint_names, _ros_joint_positions = _normalize_joint_state_pairs(
            obs.get("joint_state_name"),
            obs.get("joint_state_position"),
        )
        merged_state_names = _urdf_merged_state_joint_names(
            ROBOT_URDF_JOINT_ORDER,
            ros_joint_names,
        )
        if len(merged_state_names) == 0:
            with self._lock:
                merged_state_names = list(self.JOINT_STATE_NAMES)
        controllable_names = _derive_controllable_joint_names(
            ROBOT_URDF_JOINT_ORDER,
            merged_state_names,
        )
        stamp_map = obs.get("stamp_ns") if isinstance(obs.get("stamp_ns"), dict) else {}
        odom_stamp = stamp_map.get("odom") if isinstance(stamp_map, dict) else None
        has_odom_state = isinstance(odom_stamp, int) and int(odom_stamp) > 0
        base_controllable_names = _derive_base_controllable_names(ROBOT_URDF_JOINT_ORDER)
        base_state_names = _derive_base_state_names(has_odom_state=has_odom_state)
        with self._lock:
            self.JOINT_STATE_NAMES = list(merged_state_names)
            self.CONTROLLABLE_JOINT_NAMES = list(controllable_names)
            self.BASE_CONTROLLABLE_NAMES = list(base_controllable_names)
            self.BASE_STATE_NAMES = list(base_state_names)
            self.JOINT_LIMITS_BY_NAME = dict(ROBOT_JOINT_LIMITS_BY_NAME)
            self.JOINT_UNITS_BY_NAME = dict(ROBOT_URDF_JOINT_UNITS)
            limits_by_name = dict(self.JOINT_LIMITS_BY_NAME)

        # Preserve config-populated URDF limits/control limits when available.
        # This avoids overwriting with fallback/default ranges on ROS refresh.
        existing_cfg, existing_cfg_path = _load_robot_runtime_config(ROBOT_CONFIGS_DIR, ROBOT_CONFIG_NAME)
        existing_joint_limits = _normalize_joint_limits_map(existing_cfg.get("joint_limits", {}))
        if len(existing_joint_limits) > 0:
            persisted_joint_limits = {
                str(k): [float(v[0]), float(v[1])]
                for k, v in existing_joint_limits.items()
            }
        else:
            persisted_joint_limits: dict[str, list[float]] = {}
            for name in merged_state_names:
                lim = limits_by_name.get(name)
                if lim is None:
                    continue
                persisted_joint_limits[name] = [float(lim[0]), float(lim[1])]

        existing_control_limits_raw = existing_cfg.get("control_limits", [])
        control_limits_to_persist: list[list[float]] = []
        if isinstance(existing_control_limits_raw, list) and len(existing_control_limits_raw) >= 10:
            for pair in existing_control_limits_raw[:10]:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    try:
                        lo = float(pair[0])
                        hi = float(pair[1])
                        if lo > hi:
                            lo, hi = hi, lo
                        control_limits_to_persist.append([float(lo), float(hi)])
                    except Exception:
                        control_limits_to_persist.append(list(DEFAULT_CONTROL_LIMITS[len(control_limits_to_persist)]))
                else:
                    control_limits_to_persist.append(list(DEFAULT_CONTROL_LIMITS[len(control_limits_to_persist)]))
        else:
            control_limits_to_persist = [[float(lo), float(hi)] for lo, hi in self.JOINT_LIMITS]

        existing_ui_steps_raw = existing_cfg.get("ui_step_defaults", [])
        if isinstance(existing_ui_steps_raw, list) and len(existing_ui_steps_raw) >= 7:
            ui_steps_to_persist = []
            for i in range(7):
                try:
                    ui_steps_to_persist.append(float(existing_ui_steps_raw[i]))
                except Exception:
                    ui_steps_to_persist.append(float(ROBOT_UI_STEP_DEFAULTS[i]))
        else:
            ui_steps_to_persist = [float(v) for v in ROBOT_UI_STEP_DEFAULTS]

        existing_urdf_paths = _normalize_urdf_paths(existing_cfg.get("urdf_paths", []))
        _persist_robot_runtime_config(
            existing_cfg_path,
            {
                "robot_name": str(ROBOT_RUNTIME_CONFIG.get("robot_name", ROBOT_CONFIG_NAME)),
                "robot_ip": str(ROBOT_RUNTIME_CONFIG.get("robot_ip", "192.168.1.7")),
                "urdf_paths": list(existing_urdf_paths),
                "joint_state_names": list(merged_state_names),
                "controllable_joint_names": list(controllable_names),
                "base_controllable_names": list(base_controllable_names),
                "base_state_names": list(base_state_names),
                "joint_limits": persisted_joint_limits,
                "control_limits": control_limits_to_persist,
                "ui_step_defaults": ui_steps_to_persist,
            },
        )

    def get_joint_state_names(self) -> list[str]:
        with self._lock:
            return list(self.JOINT_STATE_NAMES)

    def get_controllable_joint_names(self) -> list[str]:
        with self._lock:
            return list(self.CONTROLLABLE_JOINT_NAMES)

    def get_base_controllable_names(self) -> list[str]:
        with self._lock:
            return list(self.BASE_CONTROLLABLE_NAMES)

    def get_base_state_names(self) -> list[str]:
        with self._lock:
            return list(self.BASE_STATE_NAMES)

    def get_joint_limits_by_name(self) -> dict[str, tuple[float, float]]:
        with self._lock:
            return dict(self.JOINT_LIMITS_BY_NAME)

    def get_joint_units_by_name(self) -> dict[str, str]:
        with self._lock:
            return dict(self.JOINT_UNITS_BY_NAME)

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
        self._push_worker_motion_tuning()

        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self._command_thread.start()

        # This variant enforces ROS-topic observations for synchronized recording.
        if FORCE_ROS_TOPIC_OBSERVATION:
            with self._lock:
                self._image_source = "ros_topic"
        ok, err = self._try_connect_ros_topics(timeout_s=min(float(timeout_s), ROS_TOPICS_CONNECT_TIMEOUT_S))
        if not ok:
            raise RuntimeError(f"ros_topic observation connect failed: {err}")

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            source = self.get_image_source()
            if source == "ros_topic":
                client = self._ros_obs_client
                if client is not None and client.connected:
                    hr, _, _, _, _, _ = client.get_snapshot()
                    obs = client.get_observation_snapshot()
                    aq = obs.get("actual_qpos")
                    if hr is not None and isinstance(aq, list) and len(aq) > 0:
                        self._refresh_joint_model_from_ros_topics()
                        self._initialize_default_command_pose(
                            send_to_robot=bool(AUTO_SEND_DEFAULT_POSE_ON_CONNECT)
                        )
                        return
            else:
                with self._lock:
                    if self.actual_qpos is not None and self.head_rgb is not None:
                        return
            time.sleep(0.05)
        raise RuntimeError(
            f"stretch_ai worker connected, but no observations arrived before timeout "
            f"(source={self.get_image_source()})"
        )

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
                    if self._base_y0 is None:
                        self._base_y0 = float(by)
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

                # Base control path:
                # - linear: send absolute base_x target x2 = observed x1 + dx
                # - angular: keep relative rotate action
                if base_active:
                    if abs(dx) > 1e-6 and abs(dtheta) <= 1e-6:
                        ok = self._send_manual_base_x_absolute_step(
                            step_dx=dx,
                            timeout_s=max(1.5, 2.0 + 4.0 * abs(dx)),
                        )
                    elif abs(dtheta) > 1e-6 and abs(dx) <= 1e-6:
                        ok = self._send_manual_base_theta_absolute_step(
                            step_dtheta=dtheta,
                            timeout_s=max(1.5, 2.0 + 4.0 * abs(dtheta)),
                        )
                    else:
                        # Fallback for combined inputs (rare from current UI bindings).
                        ok = self.move_base_relative(
                            dx=dx,
                            dy=0.0,
                            dtheta=dtheta,
                            blocking=False,
                            timeout_s=max(1.5, 2.0 + 4.0 * (abs(dx) + abs(dtheta))),
                        )
                    if not ok:
                        t_now = time.time()
                        if t_now - self._last_base_step_error_t > 2.0:
                            self._last_base_step_error_t = t_now
                            print("[stretch_ai_bridge] base manual step command failed", file=sys.stderr)

                pending = self.has_pending_command()
                if pending:
                    self.publish_commands(force=False)
            except Exception:
                pass
            time.sleep(max(0.01, float(self.command_smooth_delay_s)))

    def _send_manual_base_x_absolute_step(self, *, step_dx: float, timeout_s: float = 2.0) -> bool:
        """Manual base button semantics via execute_arm_to manipulation base_x target."""
        try:
            with self._lock:
                # execute_arm_to base_x is manipulation-space base_x.
                # Use latched command base_x so post-sync steps start from synced cmd state.
                obs_manip_x = float(self._command_latched_manip_base_x)
                target_manip_base_x = float(
                    np.clip(
                        obs_manip_x + float(step_dx),
                        float(MANIP_BASE_X_LIMITS[0]),
                        float(MANIP_BASE_X_LIMITS[1]),
                    )
                )
                applied_step_dx = float(target_manip_base_x - obs_manip_x)
                # Saturated at manip-base limit: keep command unchanged and skip send.
                if abs(applied_step_dx) <= 1e-9:
                    return True

                # Keep command-pose tracking in world x/y/theta space for table/recording.
                if isinstance(self.command_base_pose_xytheta, list) and len(self.command_base_pose_xytheta) >= 3:
                    cmd_x_origin = float(self.command_base_pose_xytheta[0])
                    cmd_y = float(self.command_base_pose_xytheta[1])
                    cmd_theta = float(self.command_base_pose_xytheta[2])
                else:
                    cmd_x_origin = float(self._base_x)
                    cmd_y = float(self._base_y0 if self._base_y0 is not None else self._base_y)
                    cmd_theta = float(self._base_theta)
                cmd_x_world = float(cmd_x_origin + applied_step_dx)
                self.command_base_pose_xytheta = [cmd_x_world, cmd_y, cmd_theta]
                self.command_base_pose_last_wall_time = time.time()

                if isinstance(self.qpos, list) and len(self.qpos) >= 8:
                    q_src = list(self.qpos[:8])
                elif isinstance(self.actual_qpos, list) and len(self.actual_qpos) >= 8:
                    q_src = list(self.actual_qpos[:8])
                else:
                    q_src = [0.0] * 8

                lift = float(np.clip(float(q_src[1]), self.JOINT_LIMITS[1][0], self.JOINT_LIMITS[1][1]))
                arm = float(np.clip(float(q_src[0]), self.JOINT_LIMITS[0][0], self.JOINT_LIMITS[0][1]))
                wrist_yaw = float(np.clip(float(q_src[2]), self.JOINT_LIMITS[2][0], self.JOINT_LIMITS[2][1]))
                wrist_pitch = float(np.clip(float(q_src[3]), self.JOINT_LIMITS[3][0], self.JOINT_LIMITS[3][1]))
                wrist_roll = float(np.clip(float(q_src[4]), self.JOINT_LIMITS[4][0], self.JOINT_LIMITS[4][1]))
                head_pan = float(np.clip(float(q_src[5]), self.JOINT_LIMITS[5][0], self.JOINT_LIMITS[5][1]))
                head_tilt = float(np.clip(float(q_src[6]), self.JOINT_LIMITS[6][0], self.JOINT_LIMITS[6][1]))
                gripper = float(np.clip(float(q_src[7]), self.JOINT_LIMITS[7][0], self.JOINT_LIMITS[7][1]))

            return self.execute_arm_to(
                [target_manip_base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll],
                gripper=gripper,
                head=[head_pan, head_tilt],
                blocking=False,
                timeout_s=float(timeout_s),
                reliable=False,
            )
        except Exception:
            return False

    def _send_manual_base_theta_absolute_step(self, *, step_dtheta: float, timeout_s: float = 2.0) -> bool:
        """Manual rotate semantics: theta2 = observed theta1 + step."""
        try:
            with self._lock:
                obs_x = float(self._base_x)
                obs_y = float(self._base_y0 if self._base_y0 is not None else self._base_y)
                obs_theta = float(self._base_theta)
                target_theta = self._wrap_angle(obs_theta + float(step_dtheta))
                # Seed command-pose from current observation so move_base_relative
                # records an absolute target for this click.
                self.command_base_pose_xytheta = [obs_x, obs_y, obs_theta]
                self.command_base_pose_last_wall_time = time.time()
            rel_theta = self._wrap_angle(target_theta - obs_theta)
            return self.move_base_relative(
                dx=0.0,
                dy=0.0,
                dtheta=float(rel_theta),
                blocking=False,
                timeout_s=float(timeout_s),
            )
        except Exception:
            return False

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
        pre_stamp_ns = self._latest_ros_observation_stamp_ns()

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

        # Record immediately when command is sent so action_command leads
        # subsequent observations until the next command update.
        self._record_command_event("execute_qpos_cmd", qpos10=cmd)
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
        finally:
            self._refresh_state_from_ros_topic(
                require_newer_than_ns=pre_stamp_ns,
                timeout_s=0.20,
            )

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
        pre_stamp_ns = self._latest_ros_observation_stamp_ns()
        q_event = None
        pose_event = None
        with self._lock:
            if self.command_base_pose_xytheta is None:
                self.command_base_pose_xytheta = [float(self._base_x), float(self._base_y), float(self._base_theta)]
            x, y, theta = [float(v) for v in self.command_base_pose_xytheta]
            if abs(float(dy)) <= 1e-12 and abs(float(dtheta)) <= 1e-12:
                # For x-only commands, keep y/theta latched in action_command.
                x += float(dx)
            else:
                x += float(dx) * np.cos(theta) - float(dy) * np.sin(theta)
                y += float(dx) * np.sin(theta) + float(dy) * np.cos(theta)
                theta = self._wrap_angle(theta + float(dtheta))
            self.command_base_pose_xytheta = [x, y, theta]
            self.command_base_pose_last_wall_time = time.time()
            pose_event = [x, y, theta]
            q_now = list(self._command_latched_qpos10)
            q_event = [float(v) for v in q_now[:10]]
            if len(q_event) < 10:
                q_event += [0.0] * (10 - len(q_event))
            q_event[8] = 0.0
            q_event[9] = 0.0

        self._record_command_event(
            "move_base_relative",
            qpos10=q_event,
            command_pose_xytheta=pose_event,
        )
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
            ok = bool(isinstance(res, dict) and res.get("ok", False))
            return ok
        except Exception as exc:
            now = time.time()
            if now - self._last_cmd_error_t > 2.0:
                self._last_cmd_error_t = now
                print(f"[stretch_ai_bridge] move_base_relative error: {exc}", file=sys.stderr)
            return False
        finally:
            if bool(blocking):
                self._refresh_state_from_ros_topic(
                    require_newer_than_ns=pre_stamp_ns,
                    timeout_s=0.60,
                )
            else:
                self._refresh_state_from_ros_topic(timeout_s=0.0)

    def rotate_base_relative(self, theta_rad: float, timeout_s: float = 10.0) -> bool:
        """Rotate base by relative yaw angle using bridge xyt navigation action."""
        rpc = self._rpc
        if rpc is None:
            return False
        pre_stamp_ns = self._latest_ros_observation_stamp_ns()
        q_event = None
        pose_event = None
        try:
            # Ensure streaming velocity channels are zero before one-shot rotate.
            with self._lock:
                if self.qpos is not None and len(self.qpos) >= 10:
                    self.qpos[8] = 0.0
                    self.qpos[9] = 0.0
                if self.published_qpos is not None and len(self.published_qpos) >= 10:
                    self.published_qpos[8] = 0.0
                    self.published_qpos[9] = 0.0
                if self.command_base_pose_xytheta is None:
                    self.command_base_pose_xytheta = [float(self._base_x), float(self._base_y), float(self._base_theta)]
                x, y, theta = [float(v) for v in self.command_base_pose_xytheta]
                theta = self._wrap_angle(theta + float(theta_rad))
                self.command_base_pose_xytheta = [x, y, theta]
                self.command_base_pose_last_wall_time = time.time()
                pose_event = [x, y, theta]
                if self._base_y0 is not None:
                    pose_event[1] = float(self._base_y0)
                q_now = list(self._command_latched_qpos10)
                q_event = [float(v) for v in q_now[:10]]
                if len(q_event) < 10:
                    q_event += [0.0] * (10 - len(q_event))
                q_event[8] = 0.0
                q_event[9] = 0.0

            self._record_command_event(
                "rotate_base_relative",
                qpos10=q_event,
                command_pose_xytheta=pose_event,
            )
            res = rpc.request(
                "rotate_base_relative",
                {"theta_rad": float(theta_rad), "timeout_s": float(timeout_s)},
                timeout_s=max(5.0, float(timeout_s) + 3.0),
            )
            # print(res)
            with self._lock:
                self._last_exec_result = dict(res) if isinstance(res, dict) else {"result": res}
            ok = bool(isinstance(res, dict) and res.get("ok", False))
            return ok
        except Exception as exc:
            now = time.time()
            if now - self._last_cmd_error_t > 2.0:
                self._last_cmd_error_t = now
                print(f"[stretch_ai_bridge] rotate_base_relative error: {exc}", file=sys.stderr)
            return False
        finally:
            # rotate_base_relative is blocking by design
            self._refresh_state_from_ros_topic(
                require_newer_than_ns=pre_stamp_ns,
                timeout_s=0.60,
            )

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
        pre_stamp_ns = self._latest_ros_observation_stamp_ns()
        vec_cmd = np.asarray(joint6, dtype=np.float32).reshape(-1)
        if vec_cmd.shape[0] < 6:
            return False
        vec_cmd = vec_cmd[:6].copy()
        grip_cmd = None if gripper is None else float(gripper)
        head_cmd = None if head is None else [float(head[0]), float(head[1])]

        q_event = None
        pose_event = None
        with self._lock:
            q_now = list(self._command_latched_qpos10)
            q_event = [float(v) for v in q_now[:10]]
            if len(q_event) < 10:
                q_event += [0.0] * (10 - len(q_event))
            q_event[0] = float(np.clip(float(vec_cmd[2]), self.JOINT_LIMITS[0][0], self.JOINT_LIMITS[0][1]))
            q_event[1] = float(np.clip(float(vec_cmd[1]), self.JOINT_LIMITS[1][0], self.JOINT_LIMITS[1][1]))
            q_event[2] = float(np.clip(float(vec_cmd[3]), self.JOINT_LIMITS[2][0], self.JOINT_LIMITS[2][1]))
            q_event[3] = float(np.clip(float(vec_cmd[4]), self.JOINT_LIMITS[3][0], self.JOINT_LIMITS[3][1]))
            q_event[4] = float(np.clip(float(vec_cmd[5]), self.JOINT_LIMITS[4][0], self.JOINT_LIMITS[4][1]))
            if grip_cmd is not None:
                q_event[7] = float(np.clip(float(grip_cmd), self.JOINT_LIMITS[7][0], self.JOINT_LIMITS[7][1]))
            if isinstance(head_cmd, list) and len(head_cmd) >= 2:
                q_event[5] = float(np.clip(float(head_cmd[0]), self.JOINT_LIMITS[5][0], self.JOINT_LIMITS[5][1]))
                q_event[6] = float(np.clip(float(head_cmd[1]), self.JOINT_LIMITS[6][0], self.JOINT_LIMITS[6][1]))
            else:
                # Always send explicit current head target to avoid hidden worker
                # defaults nudging pan/tilt on first arm_to command.
                head_cmd = [float(q_event[5]), float(q_event[6])]
            q_event[8] = 0.0
            q_event[9] = 0.0
            pose_event = (
                list(self.command_base_pose_xytheta)
                if self.command_base_pose_xytheta is not None
                else [float(self._base_x), float(self._base_y), float(self._base_theta)]
            )
            # IK arm_to base_x is manipulation-space x. Reflect its delta into
            # command-pose x for action_command unless caller already applied it.
            prev_manip_cmd = float(self._command_latched_manip_base_x)
            delta_manip_cmd = float(vec_cmd[0]) - prev_manip_cmd
            prev_pose = (
                list(self._command_latched_pose_xytheta)
                if isinstance(self._command_latched_pose_xytheta, list) and len(self._command_latched_pose_xytheta) >= 3
                else list(pose_event)
            )
            if abs(delta_manip_cmd) > 1e-9:
                current_shift = float(pose_event[0]) - float(prev_pose[0])
                if abs(current_shift - delta_manip_cmd) > 1e-4:
                    pose_event[0] = float(prev_pose[0] + delta_manip_cmd)
                    self.command_base_pose_xytheta = [
                        float(pose_event[0]),
                        float(pose_event[1]),
                        float(pose_event[2]),
                    ]
                    self.command_base_pose_last_wall_time = time.time()

        self._record_command_event(
            "execute_arm_to",
            qpos10=q_event,
            command_pose_xytheta=pose_event,
            manip_base_x_cmd=float(vec_cmd[0]),
        )
        try:
            res = rpc.request(
                "execute_arm_to",
                {
                    "joint": [float(v) for v in vec_cmd.tolist()],
                    "gripper": grip_cmd,
                    "head": head_cmd,
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
                    self._manip_base_x = float(np.clip(float(vec_cmd[0]), MANIP_BASE_X_LIMITS[0], MANIP_BASE_X_LIMITS[1]))
                    self.qpos[0] = float(np.clip(float(vec_cmd[2]), self.JOINT_LIMITS[0][0], self.JOINT_LIMITS[0][1]))  # arm_extension
                    self.qpos[1] = float(np.clip(float(vec_cmd[1]), self.JOINT_LIMITS[1][0], self.JOINT_LIMITS[1][1]))  # lift
                    self.qpos[2] = float(np.clip(float(vec_cmd[3]), self.JOINT_LIMITS[2][0], self.JOINT_LIMITS[2][1]))  # wrist_yaw
                    self.qpos[3] = float(np.clip(float(vec_cmd[4]), self.JOINT_LIMITS[3][0], self.JOINT_LIMITS[3][1]))  # wrist_pitch
                    self.qpos[4] = float(np.clip(float(vec_cmd[5]), self.JOINT_LIMITS[4][0], self.JOINT_LIMITS[4][1]))  # wrist_roll
                    if grip_cmd is not None:
                        self.qpos[7] = float(np.clip(float(grip_cmd), self.JOINT_LIMITS[7][0], self.JOINT_LIMITS[7][1]))
                    if isinstance(head_cmd, list) and len(head_cmd) >= 2:
                        self.qpos[5] = float(np.clip(float(head_cmd[0]), self.JOINT_LIMITS[5][0], self.JOINT_LIMITS[5][1]))
                        self.qpos[6] = float(np.clip(float(head_cmd[1]), self.JOINT_LIMITS[6][0], self.JOINT_LIMITS[6][1]))
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
        finally:
            if bool(blocking):
                self._refresh_state_from_ros_topic(
                    require_newer_than_ns=pre_stamp_ns,
                    timeout_s=0.60,
                )
            else:
                self._refresh_state_from_ros_topic(timeout_s=0.0)

    def sync_command_targets_to_actual(self) -> bool:
        """Optionally reset local command targets to measured robot state."""
        if not bool(ALLOW_COMMAND_SYNC_FROM_STATE):
            with self._lock:
                self._base_linear_cmd = 0.0
                self._base_angular_cmd = 0.0
                self._needs_mode_retry = False
            return True

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

    def sync_base_command_pose_to_observation(self) -> bool:
        """Set command base pose x/y/theta to latest observed base pose."""
        # Pull latest snapshots before syncing command targets.
        self._refresh_state_from_ros_topic(timeout_s=0.2)
        self._refresh_manip_base_x_from_worker()
        measured = self.get_measured_base_pose_xytheta()
        if not (isinstance(measured, list) and len(measured) >= 3):
            return False
        pose = [float(measured[0]), float(measured[1]), float(measured[2])]
        with self._lock:
            self.command_base_pose_xytheta = list(pose)
            self.command_base_pose_last_wall_time = time.time()
            q_event = list(self._command_latched_qpos10[:10]) if isinstance(self._command_latched_qpos10, list) else [0.0] * 10
            if len(q_event) < 10:
                q_event += [0.0] * (10 - len(q_event))
            q_event[8] = 0.0
            q_event[9] = 0.0
            manip_cmd = float(self._manip_base_x)
        self._record_command_event(
            "sync_base_pose_to_obs",
            qpos10=q_event,
            command_pose_xytheta=pose,
            manip_base_x_cmd=manip_cmd,
        )
        return True

    def sync_nonbase_command_joints_to_observation(self) -> bool:
        """Set command joints [0..7] to latest observed robot joints [0..7]."""
        self._refresh_state_from_ros_topic(timeout_s=0.2)
        self._refresh_manip_base_x_from_worker()
        actual = self.get_actual_qpos()
        if not (isinstance(actual, list) and len(actual) >= 8):
            return False
        with self._lock:
            if self.qpos is None or len(self.qpos) < 10:
                self.qpos = [0.0] * 10
            if self.published_qpos is None or len(self.published_qpos) < 10:
                self.published_qpos = [0.0] * 10
            for i in range(8):
                lo, hi = self.JOINT_LIMITS[i]
                v = float(np.clip(float(actual[i]), float(lo), float(hi)))
                self.qpos[i] = v
                self.published_qpos[i] = v
            self.qpos[8] = 0.0
            self.qpos[9] = 0.0
            self.published_qpos[8] = 0.0
            self.published_qpos[9] = 0.0
            self._base_linear_cmd = 0.0
            self._base_angular_cmd = 0.0
            self._needs_mode_retry = False

            pose = (
                list(self.command_base_pose_xytheta)
                if isinstance(self.command_base_pose_xytheta, list) and len(self.command_base_pose_xytheta) >= 3
                else [float(self._base_x), float(self._base_y), float(self._base_theta)]
            )
            q_event = [float(v) for v in self.qpos[:10]]
            manip_cmd = float(self._manip_base_x)
        self._record_command_event(
            "sync_nonbase_joints_to_obs",
            qpos10=q_event,
            command_pose_xytheta=[float(pose[0]), float(pose[1]), float(pose[2])],
            manip_base_x_cmd=manip_cmd,
        )
        return True

    def _refresh_manip_base_x_from_worker(self) -> bool:
        """Best-effort refresh of manipulation base_x from worker observe()."""
        rpc = self._rpc
        if rpc is None:
            return False
        try:
            obs = rpc.request("observe", {}, timeout_s=max(1.0, 0.5 * STRETCH_AI_RPC_TIMEOUT_S))
        except Exception:
            return False
        val = obs.get("manip_base_x") if isinstance(obs, dict) else None
        if not isinstance(val, (int, float)):
            return False
        with self._lock:
            self._manip_base_x = float(val)
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
        self.base_rotate_step_rad = float(np.deg2rad(np.clip(float(step_deg), 5.8, 45.0)))

    def set_base_rotate_step_delay(self, delay_s: float) -> None:
        self.base_rotate_step_delay_s = float(np.clip(float(delay_s), 0.0, 0.5))

    def _ensure_ros_topics_connected(self, timeout_s: float = ROS_TOPICS_CONNECT_TIMEOUT_S) -> None:
        if _RCLPY_IMPORT_ERROR is not None:
            raise RuntimeError(f"ROS2 observation imports unavailable: {_RCLPY_IMPORT_ERROR}")
        client = self._ros_obs_client
        if client is None:
            client = RosTopicObservationClient()
            self._ros_obs_client = client
        if not client.connected:
            client.connect(timeout_s=float(timeout_s))

    def _try_connect_ros_topics(self, timeout_s: float = 1.0) -> tuple[bool, str | None]:
        now = time.time()
        client = self._ros_obs_client
        if client is not None and client.connected:
            return True, None
        if now < float(self._next_ros_connect_t):
            return False, "ROS topic reconnect backoff active"
        try:
            self._ensure_ros_topics_connected(timeout_s=float(timeout_s))
            self._next_ros_connect_t = 0.0
            return True, None
        except Exception as exc:
            self._next_ros_connect_t = now + 2.0
            return False, str(exc)

    def set_image_source(self, source: str) -> dict[str, Any]:
        if FORCE_ROS_TOPIC_OBSERVATION:
            with self._lock:
                self._image_source = "ros_topic"
            ok, err = self._try_connect_ros_topics(timeout_s=min(3.0, ROS_TOPICS_CONNECT_TIMEOUT_S))
            if not ok:
                return {"ok": False, "source": "ros_topic", "forced": True, "error": err}
            return {"ok": True, "source": "ros_topic", "forced": True}

        src = str(source).strip().lower()
        if src in {"ros", "ros_topic", "ros_topics", "topic", "topics"}:
            src = "ros_topic"
        elif src in {"bridge", "worker"}:
            src = "bridge"
        else:
            return {"ok": False, "error": f"Unknown image source {source!r}"}

        with self._lock:
            self._image_source = src

        if src == "ros_topic":
            ok, err = self._try_connect_ros_topics(timeout_s=min(3.0, ROS_TOPICS_CONNECT_TIMEOUT_S))
            if not ok:
                return {"ok": False, "source": src, "error": err}
        return {"ok": True, "source": src}

    def get_image_source(self) -> str:
        if FORCE_ROS_TOPIC_OBSERVATION:
            return "ros_topic"
        with self._lock:
            return str(self._image_source)

    def head_image_rotated_90_cw(self) -> bool:
        return bool(ROS_TOPICS_ROTATE_HEAD_90_CW)

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
        if source == "ros_topic":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_obs_client
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
        if source == "ros_topic":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_obs_client
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
        if source == "ros_topic":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_obs_client
            if client is not None and client.connected:
                obs = client.get_observation_snapshot()
                pose = obs.get("base_pose_xytheta")
                if isinstance(pose, list) and len(pose) >= 3:
                    return [float(pose[0]), float(pose[1]), float(pose[2])]
            # Do not fall back to zeros: that can corrupt queued goal world anchoring.
            with self._lock:
                return [float(self._base_x), float(self._base_y), float(self._base_theta)]

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
        if source == "ros_topic":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_obs_client
            if client is not None and client.connected:
                _, _, _, _, head_info, _ = client.get_snapshot()
                if head_info is not None:
                    return _CameraInfoCompat(head_info)
            return None

        with self._lock:
            raw = dict(self._camera_info_head) if self._camera_info_head is not None else None
        return _CameraInfoCompat(raw) if raw is not None else None

    def get_clock(self):
        source = self.get_image_source()
        if source == "ros_topic":
            client = self._ros_obs_client
            if client is not None and client.connected:
                return client.get_clock()
        return self._clock

    def lookup_transform(self, target_frame: str, source_frame: str, *args, **kwargs):
        source = self.get_image_source()
        if source == "ros_topic":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_obs_client
            if client is None or not client.connected:
                raise RuntimeError("ROS2 topic observation client is not connected")
            timeout_s = 1.0
            timeout_obj = kwargs.get("timeout")
            if timeout_obj is not None:
                try:
                    if hasattr(timeout_obj, "nanoseconds"):
                        timeout_s = max(0.05, float(timeout_obj.nanoseconds) / 1e9)
                    elif hasattr(timeout_obj, "seconds"):
                        timeout_s = max(0.05, float(timeout_obj.seconds))
                    elif isinstance(timeout_obj, (int, float)):
                        timeout_s = max(0.05, float(timeout_obj))
                except Exception:
                    timeout_s = 1.0
            return client.lookup_transform(target_frame, source_frame, timeout_s=timeout_s)

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

    def get_aligned_record_components(self) -> dict[str, Any]:
        """Return ROS-time aligned observation/command components for one record row."""
        self._try_connect_ros_topics(timeout_s=0.5)
        client = self._ros_obs_client
        if client is None or not client.connected:
            raise RuntimeError("ROS topic observation client is not connected")

        head_rgb, wrist_rgb, head_depth, wrist_depth, _, _ = client.get_snapshot()
        timing = client.get_timing_snapshot()
        stamp_map = timing.get("stamp_ns") if isinstance(timing.get("stamp_ns"), dict) else {}
        ref_stamp_ns = (
            stamp_map.get("head_rgb")
            or stamp_map.get("head_depth")
            or stamp_map.get("wrist_rgb")
            or stamp_map.get("wrist_depth")
        )
        aligned_obs = client.get_aligned_observation_snapshot(reference_stamp_ns=ref_stamp_ns)

        actual = aligned_obs.get("actual_qpos")
        if not isinstance(actual, list):
            actual = self.get_actual_qpos()
        actual10 = [float(v) for v in actual[:10]] + [0.0] * max(0, 10 - len(actual))

        pose = aligned_obs.get("base_pose_xytheta")
        if not (isinstance(pose, list) and len(pose) >= 3):
            pose = self.get_measured_base_pose_xytheta()
        measured_pose = [float(pose[0]), float(pose[1]), float(pose[2])]

        cmd_event = self._select_command_event(ref_stamp_ns)
        if cmd_event is not None:
            cmd_qpos = [float(v) for v in cmd_event.get("qpos10", [])[:10]]
            cmd_qpos += [0.0] * max(0, 10 - len(cmd_qpos))
            cmd_pose = cmd_event.get("command_pose_xytheta")
            if not (isinstance(cmd_pose, list) and len(cmd_pose) >= 3):
                cmd_pose = self.get_command_base_pose_xytheta()
            command_pose = [float(cmd_pose[0]), float(cmd_pose[1]), float(cmd_pose[2])]
            manip_base_x_cmd = cmd_event.get("manip_base_x_cmd")
            if isinstance(manip_base_x_cmd, (int, float)):
                manip_base_x_cmd = float(manip_base_x_cmd)
            else:
                manip_base_x_cmd = float(self._manip_base_x)
        else:
            cmd_raw = self.get_published_qpos()
            cmd_qpos = [float(v) for v in cmd_raw[:10]] + [0.0] * max(0, 10 - len(cmd_raw))
            cpose = self.get_command_base_pose_xytheta()
            command_pose = [float(cpose[0]), float(cpose[1]), float(cpose[2])]
            manip_base_x_cmd = float(self._manip_base_x)

        return {
            "timestamp": (float(ref_stamp_ns) / 1e9) if isinstance(ref_stamp_ns, int) else time.time(),
            "reference_stamp_ns": ref_stamp_ns if isinstance(ref_stamp_ns, int) else None,
            "aligned_joint_stamp_ns": aligned_obs.get("aligned_joint_stamp_ns"),
            "aligned_odom_stamp_ns": aligned_obs.get("aligned_odom_stamp_ns"),
            "stamp_ns_map": stamp_map if isinstance(stamp_map, dict) else {},
            "actual_qpos10": actual10,
            "measured_pose_xytheta": measured_pose,
            "command_qpos10": cmd_qpos,
            "command_pose_xytheta": command_pose,
            "command_manip_base_x": manip_base_x_cmd,
            "head_rgb": head_rgb,
            "wrist_rgb": wrist_rgb,
            "head_depth": head_depth,
            "wrist_depth": wrist_depth,
            "command_event": cmd_event,
        }

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

        if source == "ros_topic":
            self._try_connect_ros_topics(timeout_s=0.5)
            client = self._ros_obs_client
            obs = client.get_observation_snapshot() if client is not None and client.connected else {}
            actual = obs.get("actual_qpos")
            if not isinstance(actual, list):
                actual = []
            base_pose = obs.get("base_pose_xytheta")
            if not (isinstance(base_pose, list) and len(base_pose) >= 3):
                base_pose = [0.0, 0.0, 0.0]
            joint_names, jp = _normalize_joint_state_pairs(
                obs.get("joint_state_name"),
                obs.get("joint_state_position"),
            )
            jv = obs.get("joint_state_velocity")
            if not isinstance(jv, list):
                jv = []
            je = obs.get("joint_state_effort")
            if not isinstance(je, list):
                je = []
            imu_mobile = obs.get("imu_mobile")
            imu_wrist = obs.get("imu_wrist")
            imu_cam_accel = obs.get("imu_cam_accel")
            imu_cam_gyro = obs.get("imu_cam_gyro")
            mag_mobile = obs.get("mag_mobile")
            battery = obs.get("battery")
            odom = obs.get("odom")
            if not isinstance(odom, dict):
                odom = {
                    "position": [base_pose[0], base_pose[1], 0.0],
                    "orientation": [0.0, 0.0, np.sin(base_pose[2] / 2.0), np.cos(base_pose[2] / 2.0)],
                    "linear_velocity": [float(actual[8]) if len(actual) > 8 else 0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, float(actual[9]) if len(actual) > 9 else 0.0],
                }
            _, _, _, _, head_info, wrist_info = (
                client.get_snapshot() if client is not None and client.connected else (None, None, None, None, None, None)
            )
            stamp_ns_map = obs.get("stamp_ns") if isinstance(obs.get("stamp_ns"), dict) else {}
            recv_wall_ns_map = obs.get("recv_wall_ns") if isinstance(obs.get("recv_wall_ns"), dict) else {}
            ros_minus_wall_ns = obs.get("ros_minus_wall_ns")
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
            stamp_ns_map = {}
            recv_wall_ns_map = {}
            ros_minus_wall_ns = None

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
            "observation.sync.stamp_ns": stamp_ns_map,
            "observation.sync.recv_wall_ns": recv_wall_ns_map,
            "observation.sync.ros_minus_wall_ns": ros_minus_wall_ns,
        }

    def is_ready(self):
        actual = self.get_actual_qpos()
        with self._lock:
            return self.qpos is not None and len(actual) > 0

    def close(self) -> None:
        client = self._ros_obs_client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            self._ros_obs_client = None

        self._stop_event.set()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        if self._command_thread is not None and self._command_thread.is_alive():
            self._command_thread.join(timeout=2.0)
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
        self.record_rgb_format = str(DEMO_RECORD_RGB_DEFAULT_FORMAT).lower()
        self.record_rgb_jpeg_quality = int(DEMO_RECORD_RGB_JPEG_QUALITY)
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
        self.setWindowTitle("UI4LFD")
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

        rec_layout.addWidget(QLabel("RGB Format"), 2, 0)
        self.record_rgb_format_combo = QComboBox()
        self.record_rgb_format_combo.addItems(["jpg", "png"])
        self.record_rgb_format_combo.setCurrentText(self.record_rgb_format)
        self.record_rgb_format_combo.currentTextChanged.connect(self.on_record_rgb_format_changed)
        rec_layout.addWidget(self.record_rgb_format_combo, 2, 1, 1, 2)

        self.record_toggle_button = QPushButton("Record")
        self.record_toggle_button.clicked.connect(self.toggle_demo_recording)
        rec_layout.addWidget(self.record_toggle_button, 3, 0, 1, 3)

        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        rec_layout.addWidget(self.status_label, 4, 0, 1, 3)

        self.fps_label = QLabel("FPS: --")
        rec_layout.addWidget(self.fps_label, 5, 0, 1, 3)
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
        btn_left.setEnabled(False)
        btn_right.setEnabled(False)

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
                min_val=5.8,
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

    def on_record_rgb_format_changed(self, text):
        fmt = str(text or DEMO_RECORD_RGB_DEFAULT_FORMAT).strip().lower()
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in {"jpg", "png"}:
            fmt = str(DEMO_RECORD_RGB_DEFAULT_FORMAT).lower()
        self.record_rgb_format = fmt

    def browse_record_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select LeRobot Dataset Folder", self.dataset_root)
        if folder:
            self.dataset_root = folder
            self.record_folder_input.setText(folder)

    def _record_dataset_subdir(self, kind: str) -> str:
        base = Path(self.dataset_root or str(Path.cwd())).expanduser().resolve()
        if str(kind) == "type2":
            return str((base / "type2").resolve())
        return str(base)

    def _build_record_sample(self):
        sensors = self.ros_node.get_sensor_snapshot()
        aligned: dict[str, Any] | None = None
        if hasattr(self.ros_node, "get_aligned_record_components"):
            try:
                aligned = self.ros_node.get_aligned_record_components()
            except Exception:
                aligned = None

        if isinstance(aligned, dict):
            actual_qpos = list(aligned.get("actual_qpos10") or [])
            command_qpos = list(aligned.get("command_qpos10") or [])
            measured_pose = list(aligned.get("measured_pose_xytheta") or [0.0, 0.0, 0.0])
            command_pose = list(aligned.get("command_pose_xytheta") or measured_pose)
            sample_ts = float(aligned.get("timestamp", time.time()))
            head_rgb = aligned.get("head_rgb")
            wrist_rgb = aligned.get("wrist_rgb")
            head_depth = aligned.get("head_depth")
            wrist_depth = aligned.get("wrist_depth")
            sensors["observation.sync.reference_stamp_ns"] = aligned.get("reference_stamp_ns")
            sensors["observation.sync.aligned_joint_stamp_ns"] = aligned.get("aligned_joint_stamp_ns")
            sensors["observation.sync.aligned_odom_stamp_ns"] = aligned.get("aligned_odom_stamp_ns")
            sensors["observation.sync.topic_stamp_ns"] = aligned.get("stamp_ns_map", {})
            cmd_event = aligned.get("command_event")
            if isinstance(cmd_event, dict):
                sensors["action_command.sent_wall_time_ns"] = cmd_event.get("wall_time_ns")
                sensors["action_command.sent_ros_time_ns_est"] = cmd_event.get("ros_time_ns_est")
                sensors["action_command.source"] = cmd_event.get("reason")
            sensors["action_command.manip_base_x"] = aligned.get("command_manip_base_x")
        else:
            actual_qpos = self.ros_node.get_actual_qpos()
            command_qpos = self.ros_node.get_published_qpos()
            measured_pose = self.ros_node.get_measured_base_pose_xytheta() or [0.0, 0.0, 0.0]
            command_pose = self.ros_node.get_command_base_pose_xytheta() or list(measured_pose)
            sample_ts = time.time()
            head_rgb = self.head_rgb if self.head_rgb is not None else None
            wrist_rgb = self.wrist_rgb if self.wrist_rgb is not None else None
            head_depth = self.depth_image if self.depth_image is not None else None
            wrist_depth = self.wrist_depth if self.wrist_depth is not None else None

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
            "timestamp": sample_ts,
            "head_rgb": head_rgb,
            "wrist_rgb": wrist_rgb,
            "head_depth": head_depth,
            "wrist_depth": wrist_depth,
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
            self._set_prompt_input_text(prompt)
        try:
            self.demo_recorder.start(
                self._record_dataset_subdir("type1"),
                prompt,
                rgb_image_format=self.record_rgb_format,
                rgb_jpeg_quality=self.record_rgb_jpeg_quality,
            )
            self.is_recording_demo = True
            self.record_toggle_button.setText("Stop Recording")
            self.status_label.setText(
                f"Recording demo: {prompt} (fps={self.demo_recorder.target_fps:.1f}, rgb={self.record_rgb_format})"
            )
        except Exception as exc:
            self.status_label.setText(f"Record start failed: {exc}")

    def stop_demo_recording(self):
        if not self.is_recording_demo:
            return
        try:
            # Freeze recording at stop-click time (do not capture frames while popup is open).
            self.is_recording_demo = False
            self.record_toggle_button.setText("Record")
            choice = QMessageBox.question(
                self,
                "Stop Recording",
                "Save this episode?\n\nYes: save episode\nNo: discard episode and delete recorded files",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            discard = choice == QMessageBox.StandardButton.No
            summary = self.demo_recorder.stop(discard=discard)
            if discard:
                ep = summary.get("episode_index", "?") if isinstance(summary, dict) else "?"
                self.status_label.setText(f"Discarded demo ep {ep} (deleted recorded files)")
            else:
                sync_ok = self._sync_base_cmd_from_observation(update_status=False)
                if summary is None:
                    self.status_label.setText(
                        f"Recording stopped (base cmd sync: {'ok' if sync_ok else 'failed'})"
                    )
                else:
                    self.status_label.setText(
                        f"Saved demo ep {summary['episode_index']} ({summary['num_frames']} frames, "
                        f"dropped={summary.get('dropped_frames', 0)}, "
                        f"base_sync={'ok' if sync_ok else 'failed'})"
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
    ui_loop_record_review_signal = pyqtSignal()
    JOINT_TABLE_NAMES = ["base_x", "base_y", "base_theta"]

    def __init__(self, ros_node):
        super().__init__()

        self.ros_node = ros_node
        self._robot_runtime_started = False
        self._robot_runtime_starting = False
        self._joint_limits_by_name = {}
        self._joint_units_by_name = {}
        if hasattr(self.ros_node, "get_joint_limits_by_name"):
            try:
                self._joint_limits_by_name = dict(self.ros_node.get_joint_limits_by_name())
            except Exception:
                self._joint_limits_by_name = {}
        if hasattr(self.ros_node, "get_joint_units_by_name"):
            try:
                self._joint_units_by_name = dict(self.ros_node.get_joint_units_by_name())
            except Exception:
                self._joint_units_by_name = {}
        self.JOINT_TABLE_NAMES = self._build_joint_table_names()
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
        self._action_state = 'idle'  # idle|running|paused|awaiting_confirm|awaiting_post_grasp|awaiting_post_reach_release
        self._action_mode = None
        self._action_abort_requested = False
        self._manual_gripper_override: float | None = None
        self._last_grasp_target_lift_m: float | None = None
        # Queued workflow goals planned from the same frame.
        self.queued_goals = {
            "grasp": None,
            "reach": None,
            "place_object": None,
            "release": None,
            "drag": None,
            "drag_curve": None,
            "lift_delta": None,
            "stretch_delta": None,
            "translate_delta": None,
        }
        # True execution queue in insertion order.
        self.queued_goal_sequence: list[dict[str, Any]] = []
        self.queued_goal_cursor = 0
        self.queued_sequence_started = False
        self._run_all_queued_goals = False
        self._deferred_next_goal_start = False
        self._skip_to_next_goal_requested = False
        # Per-run replay policy for queued drag/curve goals.
        self._queued_drag_repeat_count = 1
        self._queued_drag_return_to_start = False
        # Auto-loop replay state (learn from first manual trial, then alternate).
        self._auto_loop_requested_rounds = 0
        self._auto_rounds_left = 0
        self._auto_loop_running = False
        self._auto_loop_abort = False
        self._auto_loop_mode = "pick_place"  # "pick_place" | "goal_sequence"
        self._auto_sequence_replay_active = False
        self._auto_first_trial_pending = False
        self._auto_start_after_return = False
        self._auto_capture_enabled = False
        self._auto_pose_home: dict[str, Any] | None = None
        self._auto_pose_grasp: dict[str, Any] | None = None
        self._auto_pose_reach: dict[str, Any] | None = None
        self._auto_pose_place_object: dict[str, Any] | None = None
        self._auto_pose_release: dict[str, Any] | None = None
        self._auto_pose_grasp_target: dict[str, Any] | None = None
        self._auto_pose_reach_target: dict[str, Any] | None = None
        self._auto_pose_place_object_target: dict[str, Any] | None = None
        self._auto_initial_base_x: float | None = None
        self._auto_gripper_open: float | None = None
        self._auto_gripper_closed: float | None = None
        # v8 manual region-based grasp annotation state.
        self.manual_grasp_regions: list[dict[str, Any]] = []
        self._manual_region_next_id = 1
        self._manual_selected_region_id: int | None = None
        self._manual_draw_mode: str | None = None  # None | "draw_rect" | "pick_points"
        self._manual_dragging_rect = False
        self._manual_rect_start_px: tuple[int, int] | None = None
        self._manual_rect_live_px: tuple[int, int] | None = None
        # Head-view drag operation annotation state.
        self._drag_draw_mode = False
        self._drag_mode_kind = "line"  # "line" | "curve"
        self._drag_dragging = False
        self._drag_start_px: tuple[int, int] | None = None
        self._drag_live_px: tuple[int, int] | None = None
        self._drag_path_points: list[tuple[int, int]] = []
        # Modifier + wheel shortcuts (accumulate partial wheel deltas to notch steps).
        self._wheel_notch_accum: dict[str, float] = {
            "base_x": 0.0,
            "arm_lift": 0.0,
            "arm_extension": 0.0,
        }
        self._wheel_mod_latch = {
            "ctrl": False,
            "alt": False,
            "shift": False,
        }
        self.dataset_root = str((Path.cwd() / "demo_record/stretch_recordings_v9_simple_alltas").resolve())
        self.record_prompt = ""
        self.selected_task_name = str(TASK_DEFAULT_NAME)
        self.selected_embodiment = ""
        self._task_prompt_library_path = (Path(__file__).resolve().parent / TASK_PROMPT_LIBRARY_JSON).resolve()
        self._task_prompt_library: dict[str, list[str]] = {}
        self._loading_task_ui = False
        self.record_rgb_format = str(DEMO_RECORD_RGB_DEFAULT_FORMAT).lower()
        self.record_rgb_jpeg_quality = int(DEMO_RECORD_RGB_JPEG_QUALITY)
        self.is_recording_demo = False
        self.demo_recorder = LeRobotStyleRecorder(robot_type="stretch3", target_fps=DEMO_RECORD_FPS)
        # Auto-loop recording session (type1/type2 segmented episodes).
        self._loop_record_session_active = False
        self._loop_record_expected_rows = 0
        self._loop_record_prompt = ""
        self._loop_record_type_roots: dict[str, str] = {}
        self._loop_record_current: dict[str, Any] | None = None
        self._loop_record_entries: list[dict[str, Any]] = []
        # Auto-loop recording enable toggle (UI): default ON.
        self._loop_record_armed = True
        self._loop_record_stop_requested = False
        self._non_auto_record_finalize_watch_armed = False

        # Control parameters (increased for better responsiveness)
        self._ui_step_defaults = [float(v) for v in ROBOT_UI_STEP_DEFAULTS]
        self.linear_speed = float(self._ui_step_defaults[UI_STEP_IDX_BASE_LINEAR])       # m per base command step
        # Rotation step used by base rotate buttons (kept in rad for command path).
        self.base_angle_step_deg = float(self._ui_step_defaults[UI_STEP_IDX_BASE_ROTATE_DEG])
        self.angular_speed = float(np.deg2rad(self.base_angle_step_deg))
        self.arm_speed = float(self._ui_step_defaults[UI_STEP_IDX_ARM])         # m or rad increment per update
        self.head_speed = float(self._ui_step_defaults[UI_STEP_IDX_HEAD])       # rad increment per update
        self.wrist_speed = float(self._ui_step_defaults[UI_STEP_IDX_WRIST])     # rad increment per update
        self.gripper_step = float(self._ui_step_defaults[UI_STEP_IDX_GRIPPER])  # joint increment per click (open/close)
        self.command_smoothing_delay = float(self._ui_step_defaults[UI_STEP_IDX_SMOOTH_DELAY])

        # Grasp planner (will be adapted for ROS)
        # For now, we'll implement basic grasping behavior
        self.grasp_planner_available = True  # Always show grasp button
        print("Note: Grasp planner will use basic approach behavior")

        # Setup robot controller
        self.robot_controller = RobotController(ros_node)
        self.robot_controller.on_images_updated = self.on_images_updated
        self.robot_controller.on_error = self.on_error
        self.robot_controller.on_fps_updated = self.on_fps_updated
        self.device_bridge = EncoderDeviceHttpBridge()
        self._device_prev_yaw_rad: float | None = None
        self._device_prev_enc: tuple[float, float, float] | None = None
        self._device_prec_level: int | None = None
        self.device_yaw_reverse = True
        self._device_debug_print_count = 0
        self._device_b1_active = False
        self._device_b1_yaw_ref: float | None = None
        self._device_b1_wrist_ref: float | None = None
        self._device_b1_prev_pressed = False
        self._device_b1_open_next = True
        self._head_max_w = 0
        self._head_max_h = 0
        self._wrist_max_w = 0
        self._wrist_max_h = 0
        self._device_input_ok = False

        # Setup segmentation thread
        self.seg_thread = SegmentationThread()
        self.seg_thread.segmentation_complete.connect(self.on_segmentation_complete)
        self.seg_thread.segmentation_error.connect(self.on_error)
        self.seg_thread.model_loading.connect(self.on_model_loading)

        # Task/prompt library state for recording controls.
        self._load_task_prompt_library()

        # Setup UI
        self.init_ui()
        # Thread-safe UI update signals (worker threads -> main Qt thread)
        self.ui_status_signal.connect(self._apply_status_update)
        self.ui_return_enabled_signal.connect(self.return_button.setEnabled)
        self.ui_action_state_signal.connect(self._apply_action_state_ui)
        self.ui_loop_record_review_signal.connect(self._show_auto_loop_record_review_dialog)
        self._update_goal_queue_label()
        self._update_next_goal_button_state()
        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.installEventFilter(self)

        # Setup timer for update loop (runs in main thread)
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.robot_controller.step)

        self.device_timer = QTimer()
        self.device_timer.timeout.connect(self._poll_device_control)

        print("UI initialized successfully", flush=True)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("UI4LFD")
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

        # Column 1: task row + camera feeds
        task_top_widget = self.create_task_top_widget()
        camera_widget = self.create_camera_widget()
        col1_widget = QWidget()
        col1_layout = QVBoxLayout(col1_widget)
        col1_layout.setContentsMargins(0, 0, 0, 0)
        col1_layout.setSpacing(5)
        col1_layout.addWidget(task_top_widget, stretch=0)
        col1_layout.addWidget(camera_widget, stretch=0)
        col1_layout.addStretch(1)
        camera_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        col1_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        col1_w = int(
            max(
                camera_widget.sizeHint().width(),
                camera_widget.minimumSizeHint().width(),
                task_top_widget.sizeHint().width(),
            )
        )
        col1_widget.setFixedWidth(col1_w)
        content_layout.addWidget(col1_widget)

        # Column 2: robot controls
        middle_scroll = QScrollArea()
        middle_scroll.setWidgetResizable(False)
        middle_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        middle_container = QWidget()
        middle_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        middle_layout = QVBoxLayout(middle_container)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(6)
        self.middle_empty_placeholder = QWidget()
        self.middle_empty_placeholder.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        middle_layout.addWidget(self.middle_empty_placeholder)
        self.robot_controls_widget = self.create_robot_controls()
        middle_layout.addWidget(self.robot_controls_widget)
        middle_layout.addStretch(1)
        middle_scroll.setWidget(middle_container)
        self.middle_scroll = middle_scroll
        middle_container.adjustSize()
        middle_w_raw = int(
            max(
                430,
                middle_container.sizeHint().width()
                + middle_scroll.frameWidth() * 2
                + middle_scroll.verticalScrollBar().sizeHint().width()
                + 8,
            )
        )
        # Reduce middle column width by an additional ~7% from current sizing.
        # (previous factor 1.08 -> 1.08 * 0.93 = 1.0044)
        middle_w = int(max(430, round(float(middle_w_raw) * 1.0044)))
        middle_scroll.setFixedWidth(middle_w)
        content_layout.addWidget(middle_scroll)

        # Column 3: existing right panel (detected objects / recording / auto loop)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.create_object_list())
        right_layout.addStretch(1)
        right_scroll.setWidget(right_container)
        right_container.adjustSize()
        right_w_raw = int(
            max(
                1,
                right_container.sizeHint().width()
                + right_scroll.frameWidth() * 2
                + int(THIRD_COLUMN_EXTRA_WIDTH_PX),
            )
        )
        right_w = int(max(1, round(float(right_w_raw) * float(THIRD_COLUMN_WIDTH_SCALE))))
        right_scroll.setFixedWidth(right_w)
        content_layout.addWidget(right_scroll)
        content_layout.addStretch(1)

        # Keep columns at content-fit widths and avoid clipping on startup.
        try:
            required_w = int(col1_w + middle_w + right_w + (2 * content_layout.spacing()) + 40)
            self.setMinimumWidth(max(1000, required_w))
            if self.width() < required_w:
                self.resize(required_w, self.height())
        except Exception:
            pass

        main_layout.addLayout(content_layout, stretch=1)

        # Status bar at bottom (fixed height)
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("QLabel { padding: 5px; background-color: #2c3e50; color: white; font-size: 11px; }")
        self.fps_label.setMaximumHeight(30)
        main_layout.addWidget(self.fps_label)

        # Now that prompt widgets exist, sync prompt history with top task selection.
        try:
            self.selected_task_name = str(self.task_combo.currentText() or TASK_DEFAULT_NAME)
            emb_data = self.embodiment_combo.currentData() if hasattr(self, "embodiment_combo") else None
            self.selected_embodiment = str(emb_data) if isinstance(emb_data, str) else ""
            self._refresh_prompt_history_for_task(self.selected_task_name, update_prompt_box=True)
        except Exception:
            pass
        self._apply_embodiment_ui_state()
        self.update_camera_displays()

    def create_task_top_widget(self):
        """Global top-row controls: embodiment + primary camera."""
        widget = QGroupBox("")
        layout = QHBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Robot Embodiment"))
        self.embodiment_combo = QComboBox()
        self.embodiment_combo.addItem(EMBODIMENT_PLACEHOLDER_TEXT, None)
        for emb_name in EMBODIMENT_OPTIONS:
            self.embodiment_combo.addItem(str(emb_name), str(emb_name))
        self.embodiment_combo.setCurrentIndex(0)
        self.embodiment_combo.currentIndexChanged.connect(self._on_embodiment_changed)
        layout.addWidget(self.embodiment_combo, 1)

        layout.addWidget(QLabel("Primary View"))
        self.primary_camera_combo = QComboBox()
        self.primary_camera_combo.addItem(PRIMARY_VIEW_PLACEHOLDER_TEXT, None)
        for source_name, source_key in RGB_SOURCE_OPTIONS:
            self.primary_camera_combo.addItem(source_name, source_key)
        self.primary_camera_combo.setCurrentIndex(0)
        self.primary_camera_combo.setEnabled(False)
        self.primary_camera_combo.currentIndexChanged.connect(self._on_primary_view_changed)
        layout.addWidget(self.primary_camera_combo, 1)
        layout.addStretch(1)

        widget.setLayout(layout)
        return widget

    def create_camera_widget(self):
        """Create camera feed widget"""
        widget = QGroupBox("")
        layout = QVBoxLayout()
        self.image_source_status = QLabel("ROS2 Topics (fixed)")
        self.image_source_status.setStyleSheet("QLabel { color: #1e88e5; font-size: 10px; }")
        self.image_source_status.setVisible(False)

        # Primary camera display
        self.head_container = QWidget()
        self.head_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        head_container_layout = QHBoxLayout(self.head_container)
        head_container_layout.setContentsMargins(0, 0, 0, 0)
        head_container_layout.addStretch()
        self.head_display = QLabel("No view selected")
        self.head_display.setScaledContents(False)
        self.head_display.setStyleSheet("QLabel { background-color: black; color: white; }")
        self.head_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.head_display.mousePressEvent = self._on_head_mouse_press
        self.head_display.mouseMoveEvent = self._on_head_mouse_move
        self.head_display.mouseReleaseEvent = self._on_head_mouse_release
        self.head_display.setFixedSize(
            int(round(640 * float(UI_CAMERA_DISPLAY_SCALE))),
            int(round(480 * float(UI_CAMERA_DISPLAY_SCALE))),
        )
        head_container_layout.addWidget(self.head_display)
        head_container_layout.addStretch()
        self.head_container.setFixedSize(self.head_display.size())
        layout.addWidget(self.head_container, stretch=6, alignment=Qt.AlignmentFlag.AlignHCenter)
        # Keep secondary views bottom-aligned when window height grows.
        self.camera_bottom_spacer = QWidget()
        self.camera_bottom_spacer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.camera_bottom_spacer, stretch=1)

        # Separator between primary camera and bottom camera row.
        top_sep = QFrame()
        top_sep.setFrameShape(QFrame.Shape.HLine)
        top_sep.setFrameShadow(QFrame.Shadow.Plain)
        top_sep.setStyleSheet("QFrame { color: #9e9e9e; background-color: #9e9e9e; min-height: 1px; max-height: 1px; }")
        layout.addWidget(top_sep)

        # Secondary RGB source panels: one column per available source.
        self.secondary_rgb_combos: list[QComboBox] = []
        self.secondary_rgb_displays: list[QLabel] = []
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(6)
        for idx, (_name, default_key) in enumerate(RGB_SOURCE_OPTIONS):
            if idx > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setFrameShadow(QFrame.Shadow.Plain)
                sep.setStyleSheet("QFrame { color: #9e9e9e; background-color: #9e9e9e; min-width: 1px; max-width: 1px; }")
                bottom_row.addWidget(sep)
            col_widget = QWidget()
            col_layout = QVBoxLayout(col_widget)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(4)
            combo = QComboBox()
            for source_name, source_key in RGB_SOURCE_OPTIONS:
                combo.addItem(source_name, source_key)
            default_i = max(0, combo.findData(default_key))
            combo.setCurrentIndex(default_i)
            combo.currentIndexChanged.connect(lambda _i: self.update_camera_displays())
            col_layout.addWidget(combo)
            disp = QLabel("No RGB feed")
            disp.setScaledContents(False)
            disp.setStyleSheet("QLabel { background-color: #1a1a1a; color: gray; }")
            disp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            disp.setFixedSize(
                int(round(320 * float(UI_CAMERA_DISPLAY_SCALE))),
                int(round(240 * float(UI_CAMERA_DISPLAY_SCALE))),
            )
            col_layout.addWidget(disp)
            bottom_row.addWidget(col_widget)
            self.secondary_rgb_combos.append(combo)
            self.secondary_rgb_displays.append(disp)
        self.secondary_rgb_panel = QWidget()
        self.secondary_rgb_panel.setLayout(bottom_row)
        layout.addWidget(self.secondary_rgb_panel, stretch=0)

        # Segmentation button is placed under object list (right column).
        self.segment_button = QPushButton("Segment Objects")
        self.segment_button.clicked.connect(self.run_segmentation)
        self.segment_button.setMinimumHeight(UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX)
        self.segment_button.setMaximumHeight(UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX)

        widget.setLayout(layout)
        return widget

    def on_image_source_changed(self, _index: int):
        source = "ros_topic"
        try:
            result = None
            if hasattr(self.ros_node, "set_image_source"):
                result = self.ros_node.set_image_source(source)
            self.image_source = "ros_topic"
            if hasattr(self, "image_source_status"):
                self.image_source_status.setText("ROS2 Topics (fixed)")
                if isinstance(result, dict) and not result.get("ok", False):
                    self.image_source_status.setStyleSheet("QLabel { color: #ef6c00; font-size: 10px; }")
                else:
                    self.image_source_status.setStyleSheet("QLabel { color: #1e88e5; font-size: 10px; }")
            if hasattr(self, "status_label"):
                if isinstance(result, dict) and not result.get("ok", False):
                    self.status_label.setText(
                        f"Observation source fixed to ros_topic, waiting for data: "
                        f"{result.get('error', 'connect pending')}"
                    )
                    self.status_label.setStyleSheet("QLabel { color: #ef6c00; font-size: 10px; }")
                else:
                    self.status_label.setText("Observation source fixed to ros_topic")
                    self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")
        except Exception as exc:
            if hasattr(self, "image_source_status"):
                self.image_source_status.setText(f"Using: {self.image_source}")
                self.image_source_status.setStyleSheet("QLabel { color: #d32f2f; font-size: 10px; }")
            if hasattr(self, "status_label"):
                self.status_label.setText(f"Observation source switch failed: {exc}")
                self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")
            print(f"[observation_source] switch error: {exc}", file=sys.stderr)

    def _has_selected_embodiment(self) -> bool:
        if not hasattr(self, "embodiment_combo"):
            return False
        try:
            val = self.embodiment_combo.currentData()
        except Exception:
            return False
        return isinstance(val, str) and len(val.strip()) > 0

    def _has_selected_primary_view(self) -> bool:
        if not hasattr(self, "primary_camera_combo"):
            return False
        try:
            val = self.primary_camera_combo.currentData()
        except Exception:
            return False
        return isinstance(val, str) and len(val.strip()) > 0

    def _clear_primary_camera_display(self, text: str = "No view selected") -> None:
        if not hasattr(self, "head_display"):
            return
        self.head_display.setPixmap(QPixmap())
        self.head_display.setText(str(text))

    def _clear_secondary_camera_displays(self) -> None:
        if not hasattr(self, "secondary_rgb_displays"):
            return
        for disp in self.secondary_rgb_displays:
            if disp is None:
                continue
            disp.setPixmap(QPixmap())
            disp.setText("No view")

    def _run_startup_move_then_park_async(self) -> None:
        def _startup_move_then_park():
            time.sleep(1.0)
            try:
                with self._action_lock:
                    if self._action_state != "idle":
                        return
                self._set_status(
                    "Startup: moving to init pose, then parking pitch/lift for camera view...",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                self._move_to_startup_pose_for_action(timeout_s=12.0)
                self._park_camera_view_pose(timeout_s=10.0)
            except Exception as exc:
                print(f"[startup_pitch_park] {exc}")

        from threading import Thread
        Thread(target=_startup_move_then_park, daemon=True).start()

    def _rebuild_robot_controls_panel(self) -> None:
        if not hasattr(self, "middle_scroll"):
            return
        container = self.middle_scroll.widget()
        if container is None:
            return
        layout = container.layout()
        if layout is None:
            return
        old = getattr(self, "robot_controls_widget", None)
        if isinstance(old, QWidget):
            layout.removeWidget(old)
            old.deleteLater()
        self.JOINT_TABLE_NAMES = self._build_joint_table_names()
        self.robot_controls_widget = self.create_robot_controls()
        # Keep placeholder as first row; controls in second row.
        layout.insertWidget(1, self.robot_controls_widget)
        has_embodiment = bool(self._has_selected_embodiment())
        self.robot_controls_widget.setVisible(has_embodiment)
        if hasattr(self, "middle_empty_placeholder"):
            self.middle_empty_placeholder.setVisible(not has_embodiment)
        container.adjustSize()

    def _start_robot_runtime_if_needed(self) -> None:
        if bool(self._robot_runtime_started) or bool(self._robot_runtime_starting):
            return
        self._robot_runtime_starting = True
        try:
            self._set_status(
                "Starting robot runtime for selected embodiment...",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            self.ros_node.connect(timeout_s=STRETCH_AI_CONNECT_TIMEOUT_S)
            self.ros_node.set_command_smoothing_delay(self.command_smoothing_delay)
            if hasattr(self.ros_node, "set_base_rotate_step_deg"):
                self.ros_node.set_base_rotate_step_deg(self.base_angle_step_deg)
            if hasattr(self.ros_node, "set_base_rotate_step_delay"):
                self.ros_node.set_base_rotate_step_delay(self.command_smoothing_delay)

            if not bool(self._device_input_ok):
                self._device_input_ok = bool(self.device_bridge.start())
                if not self._device_input_ok:
                    print(
                        f"[device] BLE input disabled: {self.device_bridge.last_error}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[device] BLE input enabled for '{DEVICE_BLE_NAME}' "
                        f"(char={DEVICE_BLE_CHARACTERISTIC_UUID})"
                    )

            if hasattr(self, "control_timer") and not self.control_timer.isActive():
                self.control_timer.start(100)  # ~10 FPS (gives GIL time to camera thread)
            if bool(self._device_input_ok) and hasattr(self, "device_timer") and not self.device_timer.isActive():
                self.device_timer.start(max(10, int(round(1000.0 / max(1e-3, float(DEVICE_POLL_HZ))))))

            self._robot_runtime_started = True
            self._rebuild_robot_controls_panel()
            self._run_startup_move_then_park_async()
            self._set_status(
                "Robot runtime started.",
                "QLabel { color: green; font-size: 10px; }",
            )
        except Exception:
            self._robot_runtime_started = False
            try:
                self.ros_node.close()
            except Exception:
                pass
            raise
        finally:
            self._robot_runtime_starting = False

    def _update_selected_embodiment_config_from_urdf(self) -> None:
        if not self._has_selected_embodiment():
            return
        if not hasattr(self, "embodiment_combo"):
            return
        emb_label = str(self.embodiment_combo.currentText() or "").strip()
        if not emb_label:
            return
        config_name = _embodiment_label_to_config_name(emb_label)
        cfg, cfg_path = _load_robot_runtime_config(ROBOT_CONFIGS_DIR, config_name)
        raw_loaded: dict[str, Any] = {}
        try:
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                if isinstance(parsed, dict):
                    raw_loaded = dict(parsed)
        except Exception:
            raw_loaded = {}
        urdf_paths = _normalize_urdf_paths(raw_loaded.get("urdf_paths", []))
        if len(urdf_paths) == 0:
            legacy = str(raw_loaded.get("urdf_path", "")).strip()
            if legacy:
                urdf_paths = [legacy]
        if len(urdf_paths) == 0:
            self._set_status(
                f"{config_name}: no urdf_paths configured; skipped URDF joint extraction",
                "QLabel { color: orange; font-size: 10px; }",
            )
            return
        urdf_meta = _load_urdf_joint_metadata(urdf_paths)
        urdf_joint_order = _normalize_joint_state_names(urdf_meta.get("joint_order", []))
        urdf_joint_limits = _normalize_joint_limits_map(urdf_meta.get("joint_limits", {}))
        existing_joint_state_names = _normalize_joint_state_names(cfg.get("joint_state_names", []))
        existing_controllable_joint_names = _normalize_joint_state_names(cfg.get("controllable_joint_names", []))
        existing_base_controllable_names = _normalize_base_key_list(
            cfg.get("base_controllable_names", []),
            BASE_CONTROLLABLE_KEYS,
        )
        existing_base_state_names = _normalize_base_key_list(cfg.get("base_state_names", []), BASE_ODOM_STATE_KEYS)
        existing_joint_limits = _normalize_joint_limits_map(cfg.get("joint_limits", {}))
        existing_control_limits_raw = cfg.get("control_limits", [])
        existing_control_limits: list[tuple[float, float]] = []
        if isinstance(existing_control_limits_raw, list) and len(existing_control_limits_raw) >= 10:
            for pair in existing_control_limits_raw[:10]:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    try:
                        lo = float(pair[0])
                        hi = float(pair[1])
                        if lo > hi:
                            lo, hi = hi, lo
                        existing_control_limits.append((float(lo), float(hi)))
                    except Exception:
                        existing_control_limits.append(tuple(DEFAULT_CONTROL_LIMITS[len(existing_control_limits)]))
                else:
                    existing_control_limits.append(tuple(DEFAULT_CONTROL_LIMITS[len(existing_control_limits)]))
        if len(existing_control_limits) < 10:
            existing_control_limits = [tuple(v) for v in DEFAULT_CONTROL_LIMITS]

        have_urdf_joints = len(urdf_joint_order) > 0
        have_urdf_limits = len(urdf_joint_limits) > 0

        if have_urdf_joints:
            joint_state_names_out = list(urdf_joint_order)
            controllable_joint_names = _derive_controllable_joint_names(urdf_joint_order, urdf_joint_order)
            base_controllable_names = _derive_base_controllable_names(urdf_joint_order)
        else:
            joint_state_names_out = list(existing_joint_state_names)
            controllable_joint_names = list(existing_controllable_joint_names)
            base_controllable_names = (
                list(existing_base_controllable_names)
                if len(existing_base_controllable_names) > 0
                else list(BASE_CONTROLLABLE_KEYS)
            )
        base_state_names = list(existing_base_state_names)

        if have_urdf_limits:
            limits_source = urdf_joint_limits
            limit_name_order = urdf_joint_order if have_urdf_joints else list(urdf_joint_limits.keys())
            control_limits_out = _derive_control_limits_from_joint_limits(urdf_joint_limits)
        else:
            limits_source = existing_joint_limits
            limit_name_order = joint_state_names_out
            control_limits_out = (
                list(existing_control_limits)
                if len(existing_control_limits) >= 10
                else [tuple(v) for v in DEFAULT_CONTROL_LIMITS]
            )
        persisted_joint_limits: dict[str, list[float]] = {}
        for name in limit_name_order:
            lim = limits_source.get(str(name))
            if lim is None:
                continue
            persisted_joint_limits[str(name)] = [float(lim[0]), float(lim[1])]
        _persist_robot_runtime_config(
            cfg_path,
            {
                "robot_name": str(cfg.get("robot_name", config_name)),
                "robot_ip": str(cfg.get("robot_ip", "")),
                "urdf_paths": list(urdf_paths),
                "joint_state_names": list(joint_state_names_out),
                "controllable_joint_names": list(controllable_joint_names),
                "base_controllable_names": list(base_controllable_names),
                "base_state_names": list(base_state_names),
                "joint_limits": persisted_joint_limits,
                "control_limits": [[float(lo), float(hi)] for lo, hi in control_limits_out],
            },
        )
        err_list = urdf_meta.get("errors", [])
        if isinstance(err_list, list) and len(err_list) > 0:
            self._set_status(
                f"{config_name}: URDF parsed with warnings; joints={len(joint_state_names_out)} limits={len(persisted_joint_limits)}",
                "QLabel { color: #ef6c00; font-size: 10px; }",
            )
            for err in err_list:
                print(f"[embodiment_config] {config_name}: {err}", file=sys.stderr)
        else:
            self._set_status(
                f"{config_name}: updated from URDF (joints={len(joint_state_names_out)}, limits={len(persisted_joint_limits)})",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )

    def _apply_embodiment_ui_state(self) -> None:
        has_embodiment = bool(self._has_selected_embodiment())
        if has_embodiment and hasattr(self, "embodiment_combo"):
            emb_data = self.embodiment_combo.currentData()
            self.selected_embodiment = str(emb_data) if isinstance(emb_data, str) else ""
        else:
            self.selected_embodiment = ""

        if hasattr(self, "primary_camera_combo"):
            if not has_embodiment:
                prev_block = self.primary_camera_combo.blockSignals(True)
                try:
                    self.primary_camera_combo.setCurrentIndex(0)
                finally:
                    self.primary_camera_combo.blockSignals(prev_block)
            self.primary_camera_combo.setEnabled(has_embodiment)

        if hasattr(self, "robot_controls_widget"):
            self.robot_controls_widget.setVisible(has_embodiment)
        if hasattr(self, "middle_empty_placeholder"):
            self.middle_empty_placeholder.setVisible(not has_embodiment)
        if hasattr(self, "secondary_rgb_panel"):
            self.secondary_rgb_panel.setVisible(has_embodiment)

        if not has_embodiment:
            self._clear_primary_camera_display("No view selected")
            self._clear_secondary_camera_displays()

    def create_control_widget(self):
        """Create two-column right side: controls (middle) + detected objects/actions (right)."""
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Middle column: robot controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(False)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_container = QWidget()
        left_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.create_robot_controls())
        left_layout.addStretch(1)
        left_scroll.setWidget(left_container)

        # Right-most column: detected objects + action flow + recording widgets
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(False)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addStretch(1)
        right_layout.addWidget(self.create_object_list())
        right_scroll.setWidget(right_container)

        # Fit each column to content width (no stretch, no clipping).
        left_container.adjustSize()
        right_container.adjustSize()
        left_w = int(
            max(
                430,
                left_container.sizeHint().width()
                + left_scroll.frameWidth() * 2
                + left_scroll.verticalScrollBar().sizeHint().width()
                + 8,
            )
        )
        right_w_raw = int(
            max(
                560,
                right_container.sizeHint().width()
                + right_scroll.frameWidth() * 2
                + right_scroll.verticalScrollBar().sizeHint().width()
                + 8,
            )
        )
        right_w = int(max(1, round(float(right_w_raw) * float(THIRD_COLUMN_WIDTH_SCALE))))
        left_scroll.setFixedWidth(left_w)
        right_scroll.setFixedWidth(right_w)

        layout.addWidget(left_scroll, stretch=1)
        layout.addWidget(right_scroll, stretch=1)
        total_w = int(
            left_w
            + right_w
            + layout.spacing()
            + layout.contentsMargins().left()
            + layout.contentsMargins().right()
        )
        widget.setFixedWidth(total_w)
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

    def _create_cm_std_slider(
        self,
        label_text: str,
        default_m: float,
        callback_m,
        *,
        min_cm: float = 0.0,
        max_cm: float = 20.0,
    ):
        """Create a slider row that displays centimeters and returns meters."""
        row = QHBoxLayout()
        row.setSpacing(4)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(round(min_cm * 10.0)), int(round(max_cm * 10.0)))  # 0.1 cm steps
        slider.setFixedHeight(20)

        cm_default = float(np.clip(float(default_m) * 100.0, float(min_cm), float(max_cm)))
        slider.setValue(int(round(cm_default * 10.0)))

        val_label = QLabel(f"{cm_default:.1f}cm")
        val_label.setFixedWidth(60)
        val_label.setStyleSheet("QLabel { font-size: 10px; }")

        def on_change(v):
            cm = float(v) / 10.0
            m = cm / 100.0
            val_label.setText(f"{cm:.1f}cm")
            callback_m(float(m))

        slider.valueChanged.connect(on_change)

        row.addWidget(QLabel(label_text))
        row.addWidget(slider)
        row.addWidget(val_label)
        return row

    def create_robot_controls(self):
        """Create robot control buttons"""
        widget = QGroupBox("")
        layout = QVBoxLayout()
        available_controls = self._resolve_available_manual_controls()

        header_row = QHBoxLayout()
        header_label = QLabel("Robot Control")
        header_label.setStyleSheet("QLabel { font-weight: 600; }")
        header_row.addWidget(header_label)
        header_row.addStretch()
        self.robot_control_settings_button = QPushButton("⚙")
        self.robot_control_settings_button.setToolTip("Open joint step-size settings")
        self.robot_control_settings_button.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        self.robot_control_settings_button.setMaximumWidth(32)
        self.robot_control_settings_button.clicked.connect(self._open_robot_control_settings_popup)
        header_row.addWidget(self.robot_control_settings_button)
        layout.addLayout(header_row)

        # Base controls
        base_group = QGroupBox("Base (X: Ctrl+Mouse Wheel)")
        base_layout = QGridLayout()
        base_layout.setHorizontalSpacing(3)
        base_layout.setVerticalSpacing(2)

        btn_forward = QPushButton("↑")
        btn_forward.setToolTip("Move Forward")
        btn_forward.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_forward.pressed.connect(lambda: self.start_control('base_linear', self.linear_speed))
        btn_forward.released.connect(lambda: self.stop_control('base_linear'))
        base_layout.addWidget(btn_forward, 0, 0)

        btn_left = QPushButton("←")
        btn_left.setToolTip("Rotate Left")
        btn_left.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_left.pressed.connect(
            lambda: self.start_control('base_angular', float(np.deg2rad(self.base_angle_step_deg)))
        )
        btn_left.released.connect(lambda: self.stop_control('base_angular'))
        btn_left.setEnabled(False)
        base_layout.addWidget(btn_left, 0, 2)

        btn_backward = QPushButton("↓")
        btn_backward.setToolTip("Move Backward")
        btn_backward.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_backward.pressed.connect(lambda: self.start_control('base_linear', -self.linear_speed))
        btn_backward.released.connect(lambda: self.stop_control('base_linear'))
        base_layout.addWidget(btn_backward, 0, 1)

        btn_right = QPushButton("→")
        btn_right.setToolTip("Rotate Right")
        btn_right.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_right.pressed.connect(
            lambda: self.start_control('base_angular', -float(np.deg2rad(self.base_angle_step_deg)))
        )
        btn_right.released.connect(lambda: self.stop_control('base_angular'))
        btn_right.setEnabled(False)
        base_layout.addWidget(btn_right, 0, 3)

        move_cm_label = QLabel("Move (cm)")
        move_cm_label.setVisible(False)
        base_layout.addWidget(move_cm_label, 1, 0)
        self.base_distance_cm_input = QLineEdit("0")
        self.base_distance_cm_input.setPlaceholderText("+100 / -100")
        self.base_distance_cm_input.setVisible(False)
        move_dist_btn = QPushButton("Move Distance")
        move_dist_btn.setVisible(False)

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
        base_layout.addWidget(self.base_distance_cm_input, 1, 1)
        base_layout.addWidget(move_dist_btn, 1, 2)

        base_linear_available = ("base_linear" in available_controls)
        btn_forward.setEnabled(base_linear_available)
        btn_backward.setEnabled(base_linear_available)
        # Keep UI behavior unchanged: base rotation buttons remain hidden/disabled in this panel.
        btn_left.setEnabled(False)
        btn_right.setEnabled(False)

        base_group.setLayout(base_layout)
        base_group.setVisible(base_linear_available or ("base_angular" in available_controls))
        layout.addWidget(base_group)

        # Arm controls
        arm_group = QGroupBox("Arm (Lift: Alt+Mouse Wheel, Ext: Shift+Mouse Wheel)")
        arm_layout = QGridLayout()
        arm_layout.setSpacing(3)

        btn_lift_up = QPushButton("Lift ↑")
        btn_lift_up.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_lift_up.pressed.connect(lambda: self.start_control_incremental('arm_lift', self.arm_speed))
        btn_lift_up.released.connect(lambda: self.stop_control('arm_lift'))
        arm_layout.addWidget(btn_lift_up, 0, 0)

        btn_lift_down = QPushButton("Lift ↓")
        btn_lift_down.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_lift_down.pressed.connect(lambda: self.start_control_incremental('arm_lift', -self.arm_speed))
        btn_lift_down.released.connect(lambda: self.stop_control('arm_lift'))
        arm_layout.addWidget(btn_lift_down, 0, 1)

        btn_extend = QPushButton("Extend →")
        btn_extend.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_extend.pressed.connect(lambda: self.start_control_incremental('arm_extension', self.arm_speed))
        btn_extend.released.connect(lambda: self.stop_control('arm_extension'))
        arm_layout.addWidget(btn_extend, 0, 2)

        btn_retract = QPushButton("Retract ←")
        btn_retract.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_retract.pressed.connect(lambda: self.start_control_incremental('arm_extension', -self.arm_speed))
        btn_retract.released.connect(lambda: self.stop_control('arm_extension'))
        arm_layout.addWidget(btn_retract, 0, 3)

        arm_lift_available = ("arm_lift" in available_controls)
        arm_ext_available = ("arm_extension" in available_controls)
        btn_lift_up.setEnabled(arm_lift_available)
        btn_lift_down.setEnabled(arm_lift_available)
        btn_extend.setEnabled(arm_ext_available)
        btn_retract.setEnabled(arm_ext_available)

        arm_group.setLayout(arm_layout)
        arm_group.setVisible(arm_lift_available or arm_ext_available)
        layout.addWidget(arm_group)

        # Head controls
        head_group = QGroupBox("Head")
        head_layout = QGridLayout()
        head_layout.setSpacing(3)

        btn_pan_left = QPushButton("Pan ←")
        btn_pan_left.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_pan_left.pressed.connect(lambda: self.start_control_incremental('head_pan', self.head_speed))
        btn_pan_left.released.connect(lambda: self.stop_control('head_pan'))
        head_layout.addWidget(btn_pan_left, 0, 0)

        btn_pan_right = QPushButton("Pan →")
        btn_pan_right.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_pan_right.pressed.connect(lambda: self.start_control_incremental('head_pan', -self.head_speed))
        btn_pan_right.released.connect(lambda: self.stop_control('head_pan'))
        head_layout.addWidget(btn_pan_right, 0, 1)

        btn_tilt_up = QPushButton("Tilt ↑")
        btn_tilt_up.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_tilt_up.pressed.connect(lambda: self.start_control_incremental('head_tilt', self.head_speed))
        btn_tilt_up.released.connect(lambda: self.stop_control('head_tilt'))
        head_layout.addWidget(btn_tilt_up, 0, 2)

        btn_tilt_down = QPushButton("Tilt ↓")
        btn_tilt_down.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_tilt_down.pressed.connect(lambda: self.start_control_incremental('head_tilt', -self.head_speed))
        btn_tilt_down.released.connect(lambda: self.stop_control('head_tilt'))
        head_layout.addWidget(btn_tilt_down, 0, 3)

        head_pan_available = ("head_pan" in available_controls)
        head_tilt_available = ("head_tilt" in available_controls)
        btn_pan_left.setEnabled(head_pan_available)
        btn_pan_right.setEnabled(head_pan_available)
        btn_tilt_up.setEnabled(head_tilt_available)
        btn_tilt_down.setEnabled(head_tilt_available)

        head_group.setLayout(head_layout)
        head_group.setVisible(head_pan_available or head_tilt_available)
        layout.addWidget(head_group)

        # Wrist controls
        wrist_group = QGroupBox("Wrist")
        wrist_layout = QGridLayout()
        wrist_layout.setSpacing(3)

        btn_roll_left = QPushButton("Roll ↶")
        btn_roll_left.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_roll_left.pressed.connect(lambda: self.start_control_incremental('wrist_roll', -self.wrist_speed))
        btn_roll_left.released.connect(lambda: self.stop_control('wrist_roll'))
        wrist_layout.addWidget(btn_roll_left, 0, 0)

        btn_roll_right = QPushButton("Roll ↷")
        btn_roll_right.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_roll_right.pressed.connect(lambda: self.start_control_incremental('wrist_roll', self.wrist_speed))
        btn_roll_right.released.connect(lambda: self.stop_control('wrist_roll'))
        wrist_layout.addWidget(btn_roll_right, 1, 0)

        btn_pitch_down = QPushButton("Pitch ↓")
        btn_pitch_down.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_pitch_down.pressed.connect(lambda: self.start_control_incremental('wrist_pitch', -self.wrist_speed))
        btn_pitch_down.released.connect(lambda: self.stop_control('wrist_pitch'))
        wrist_layout.addWidget(btn_pitch_down, 0, 1)

        btn_pitch_up = QPushButton("Pitch ↑")
        btn_pitch_up.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_pitch_up.pressed.connect(lambda: self.start_control_incremental('wrist_pitch', self.wrist_speed))
        btn_pitch_up.released.connect(lambda: self.stop_control('wrist_pitch'))
        wrist_layout.addWidget(btn_pitch_up, 1, 1)

        btn_yaw_left = QPushButton("Yaw ←")
        btn_yaw_left.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_yaw_left.pressed.connect(lambda: self.start_control_incremental('wrist_yaw', self.wrist_speed))
        btn_yaw_left.released.connect(lambda: self.stop_control('wrist_yaw'))
        wrist_layout.addWidget(btn_yaw_left, 0, 2)

        btn_yaw_right = QPushButton("Yaw →")
        btn_yaw_right.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_yaw_right.pressed.connect(lambda: self.start_control_incremental('wrist_yaw', -self.wrist_speed))
        btn_yaw_right.released.connect(lambda: self.stop_control('wrist_yaw'))
        wrist_layout.addWidget(btn_yaw_right, 1, 2)

        wrist_roll_available = ("wrist_roll" in available_controls)
        wrist_pitch_available = ("wrist_pitch" in available_controls)
        wrist_yaw_available = ("wrist_yaw" in available_controls)
        btn_roll_left.setEnabled(wrist_roll_available)
        btn_roll_right.setEnabled(wrist_roll_available)
        btn_pitch_down.setEnabled(wrist_pitch_available)
        btn_pitch_up.setEnabled(wrist_pitch_available)
        btn_yaw_left.setEnabled(wrist_yaw_available)
        btn_yaw_right.setEnabled(wrist_yaw_available)

        wrist_group.setLayout(wrist_layout)
        wrist_group.setVisible(wrist_roll_available or wrist_pitch_available or wrist_yaw_available)
        layout.addWidget(wrist_group)

        # Gripper controls
        gripper_group = QGroupBox("Gripper")
        gripper_layout = QGridLayout()
        gripper_layout.setSpacing(3)

        btn_gripper_open = QPushButton("Open +")
        btn_gripper_open.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_gripper_open.setToolTip("Open gripper by one step")
        btn_gripper_open.clicked.connect(lambda: self.adjust_gripper_step(+1))
        gripper_layout.addWidget(btn_gripper_open, 0, 1)

        btn_gripper_close = QPushButton("Close -")
        btn_gripper_close.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_gripper_close.setToolTip("Close gripper by one step")
        btn_gripper_close.clicked.connect(lambda: self.adjust_gripper_step(-1))
        gripper_layout.addWidget(btn_gripper_close, 0, 2)

        btn_gripper_open_full = QPushButton("Open Full")
        btn_gripper_open_full.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_gripper_open_full.setToolTip("Open gripper to maximum limit")
        btn_gripper_open_full.clicked.connect(lambda: self.set_gripper(self.ros_node.JOINT_LIMITS[7][1]))
        gripper_layout.addWidget(btn_gripper_open_full, 0, 0)

        btn_gripper_close_full = QPushButton("Close Full")
        btn_gripper_close_full.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        btn_gripper_close_full.setToolTip("Close gripper to minimum limit")
        btn_gripper_close_full.clicked.connect(lambda: self.set_gripper(self.ros_node.JOINT_LIMITS[7][0]))
        gripper_layout.addWidget(btn_gripper_close_full, 0, 3)

        gripper_available = ("gripper" in available_controls)
        btn_gripper_open.setEnabled(gripper_available)
        btn_gripper_close.setEnabled(gripper_available)
        btn_gripper_open_full.setEnabled(gripper_available)
        btn_gripper_close_full.setEnabled(gripper_available)

        gripper_group.setLayout(gripper_layout)
        gripper_group.setVisible(gripper_available)
        layout.addWidget(gripper_group)

        # Device precision indicator (from incoming device packet raw[4], range 0..4).
        self.device_prec_label = QLabel("Device Prec: --")
        self.device_prec_label.setStyleSheet("QLabel { color: #555; font-size: 10px; }")
        self.device_prec_label.setVisible(False)
        layout.addWidget(self.device_prec_label)

        # Yaw sign mode toggle for device-driven wrist yaw.
        self.device_yaw_mode_button = QPushButton()
        self.device_yaw_mode_button.setMinimumHeight(UI_ROBOT_CONTROL_BUTTON_HEIGHT_PX)
        self.device_yaw_mode_button.clicked.connect(self._toggle_device_yaw_mode)
        self._update_device_yaw_mode_button()
        self.device_yaw_mode_button.setVisible(False)
        layout.addWidget(self.device_yaw_mode_button)

        # Keep table in control column; size it so all rows are visible without table scrolling.
        layout.addWidget(self.create_joint_state_panel())

        widget.setLayout(layout)
        return widget

    def _open_robot_control_settings_popup(self):
        dlg = QDialog(self)
        self._robot_control_settings_dialog = dlg
        dlg.setWindowTitle("Robot Control Settings")
        dlg.setModal(False)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setSpacing(6)

        old_ranges = [float(abs(float(hi) - float(lo))) for lo, hi in DEFAULT_CONTROL_LIMITS]
        cur_ranges = [float(abs(float(hi) - float(lo))) for lo, hi in self.ros_node.JOINT_LIMITS]

        def _scaled_bounds(idx: int, old_min: float, old_max: float) -> tuple[float, float]:
            ratio_min = float(old_min) / max(1e-9, old_ranges[idx])
            ratio_max = float(old_max) / max(1e-9, old_ranges[idx])
            mn = max(1e-5, ratio_min * cur_ranges[idx])
            mx = max(mn * 1.25, ratio_max * cur_ranges[idx])
            return float(mn), float(mx)

        base_lin_min, base_lin_max = _scaled_bounds(8, DEVICE_BASE_STEP_MIN_M, DEVICE_BASE_STEP_MAX_M)
        arm_ref_range = min(old_ranges[0], old_ranges[1])
        arm_cur_range = min(cur_ranges[0], cur_ranges[1])
        arm_min = max(1e-5, (DEVICE_ARM_STEP_MIN / max(1e-9, arm_ref_range)) * arm_cur_range)
        arm_max = max(arm_min * 1.25, (DEVICE_ARM_STEP_MAX / max(1e-9, arm_ref_range)) * arm_cur_range)
        head_ref_range = min(old_ranges[5], old_ranges[6])
        head_cur_range = min(cur_ranges[5], cur_ranges[6])
        head_min = max(1e-5, (0.02 / max(1e-9, head_ref_range)) * head_cur_range)
        head_max = max(head_min * 1.25, (0.30 / max(1e-9, head_ref_range)) * head_cur_range)
        wrist_ref_range = min(old_ranges[2], old_ranges[3], old_ranges[4])
        wrist_cur_range = min(cur_ranges[2], cur_ranges[3], cur_ranges[4])
        wrist_min = max(1e-5, (0.02 / max(1e-9, wrist_ref_range)) * wrist_cur_range)
        wrist_max = max(wrist_min * 1.25, (0.30 / max(1e-9, wrist_ref_range)) * wrist_cur_range)
        gripper_min, gripper_max = _scaled_bounds(7, DEVICE_GRIPPER_STEP_MIN, DEVICE_GRIPPER_STEP_MAX)

        def set_base_linear_step(v):
            self.linear_speed = float(v)

        def set_base_angle_step_deg(v):
            self.base_angle_step_deg = float(v)
            self.angular_speed = float(np.deg2rad(self.base_angle_step_deg))
            if hasattr(self.ros_node, "set_base_rotate_step_deg"):
                self.ros_node.set_base_rotate_step_deg(self.base_angle_step_deg)

        dlg_layout.addLayout(
            self._create_speed_slider(
                base_lin_min,
                base_lin_max,
                self.linear_speed,
                set_base_linear_step,
                label_text="Base linear step",
            )
        )
        dlg_layout.addLayout(
            self._create_speed_slider(
                5.8,
                20.0,
                self.base_angle_step_deg,
                set_base_angle_step_deg,
                label_text="Base rotation step (deg)",
            )
        )
        dlg_layout.addLayout(
            self._create_speed_slider(
                arm_min,
                arm_max,
                self.arm_speed,
                lambda v: setattr(self, "arm_speed", float(v)),
                label_text="Arm step",
            )
        )
        dlg_layout.addLayout(
            self._create_speed_slider(
                head_min,
                head_max,
                self.head_speed,
                lambda v: setattr(self, "head_speed", float(v)),
                label_text="Head step",
            )
        )
        dlg_layout.addLayout(
            self._create_speed_slider(
                wrist_min,
                wrist_max,
                self.wrist_speed,
                lambda v: setattr(self, "wrist_speed", float(v)),
                label_text="Wrist step",
            )
        )
        dlg_layout.addLayout(
            self._create_speed_slider(
                gripper_min,
                gripper_max,
                self.gripper_step,
                lambda v: setattr(self, "gripper_step", float(v)),
                label_text="Gripper step",
            )
        )

        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX)
        close_btn.clicked.connect(dlg.close)
        dlg_layout.addWidget(close_btn)

        dlg.resize(560, 320)
        dlg.show()

    @staticmethod
    def _joint_name_is_base_alias(name: str) -> bool:
        s = str(name).strip()
        if not s:
            return False
        if s in {"base_x", "base_y", "base_theta", "joint_base_x", "joint_base_y", "joint_base_theta"}:
            return True
        if s in URDF_BASE_TRANSLATION_NAMES:
            return True
        if s in URDF_BASE_ROTATION_NAMES:
            return True
        return False

    def _get_ros_state_availability(self) -> tuple[list[str], bool, list[str], list[str], list[str]]:
        if not bool(getattr(self, "_robot_runtime_started", False)):
            return [], False, [], [], []
        try:
            sensors = self.ros_node.get_sensor_snapshot()
        except Exception:
            return [], False, [], list(BASE_CONTROLLABLE_KEYS), []
        joint_names, _joint_pos = _normalize_joint_state_pairs(
            sensors.get("observation.joint_state.name"),
            sensors.get("observation.joint_state.position"),
        )
        joint_names = _urdf_merged_state_joint_names(ROBOT_URDF_JOINT_ORDER, joint_names)
        controllable_names: list[str] = []
        getter = getattr(self.ros_node, "get_controllable_joint_names", None)
        if callable(getter):
            try:
                controllable_names = _normalize_joint_state_names(getter())
            except Exception:
                controllable_names = []
        if len(controllable_names) == 0:
            controllable_names = _derive_controllable_joint_names(ROBOT_URDF_JOINT_ORDER, joint_names)
        state_set = set(joint_names)
        controllable_names = [str(n) for n in controllable_names if str(n) in state_set]
        base_controllable_names: list[str] = []
        base_controllable_getter = getattr(self.ros_node, "get_base_controllable_names", None)
        if callable(base_controllable_getter):
            try:
                base_controllable_names = _normalize_base_key_list(
                    base_controllable_getter(),
                    BASE_CONTROLLABLE_KEYS,
                )
            except Exception:
                base_controllable_names = []
        if len(base_controllable_names) == 0:
            base_controllable_names = _derive_base_controllable_names(ROBOT_URDF_JOINT_ORDER)
        stamp_map = sensors.get("observation.sync.stamp_ns")
        has_odom_state = False
        if isinstance(stamp_map, dict):
            odom_stamp = stamp_map.get("odom")
            has_odom_state = isinstance(odom_stamp, int) and int(odom_stamp) > 0
        base_state_names: list[str] = []
        base_state_getter = getattr(self.ros_node, "get_base_state_names", None)
        if callable(base_state_getter):
            try:
                base_state_names = _normalize_base_key_list(
                    base_state_getter(),
                    BASE_ODOM_STATE_KEYS,
                )
            except Exception:
                base_state_names = []
        if bool(has_odom_state):
            base_state_names = _derive_base_state_names(has_odom_state=True)
        else:
            base_state_names = []
        return joint_names, bool(has_odom_state), controllable_names, base_controllable_names, base_state_names

    def _resolve_available_manual_controls(self) -> set[str]:
        (
            _state_joint_names,
            has_odom_state,
            controllable_joint_names,
            base_controllable_names,
            _base_state_names,
        ) = self._get_ros_state_availability()
        names = set(controllable_joint_names)
        base_ctrl = set(base_controllable_names)

        def _has_any(candidates: list[str]) -> bool:
            return any((str(c) in names) for c in candidates)

        available: set[str] = set()
        if has_odom_state and ("base_x" in base_ctrl):
            available.add("base_linear")
        if has_odom_state and ("base_theta" in base_ctrl):
            available.add("base_angular")
        if _has_any(["joint_lift", "lift"]):
            available.add("arm_lift")
        if _has_any(["joint_arm_l0", "arm"]):
            available.add("arm_extension")
        if _has_any(["joint_head_pan", "head_pan"]):
            available.add("head_pan")
        if _has_any(["joint_head_tilt", "head_tilt"]):
            available.add("head_tilt")
        if _has_any(["joint_wrist_roll", "wrist_roll"]):
            available.add("wrist_roll")
        if _has_any(["joint_wrist_pitch", "wrist_pitch"]):
            available.add("wrist_pitch")
        if _has_any(["joint_wrist_yaw", "wrist_yaw"]):
            available.add("wrist_yaw")
        if _has_any(["joint_gripper_finger_left", "gripper_finger_left", "joint_gripper_finger_right"]):
            available.add("gripper")
        return available

    def _build_joint_table_names(self) -> list[str]:
        (
            _state_joint_names,
            has_odom_state,
            controllable_joint_names,
            _base_controllable_names,
            base_state_names,
        ) = self._get_ros_state_availability()
        ordered: list[str] = []
        if has_odom_state:
            ordered.extend([k for k in base_state_names if k in BASE_ODOM_STATE_KEYS])
        seen: set[str] = set(ordered)
        for name in controllable_joint_names:
            s = str(name).strip()
            if not s or self._joint_name_is_base_alias(s):
                continue
            if s not in seen:
                ordered.append(s)
                seen.add(s)
        return ordered

    def _joint_table_display_name(self, key: str) -> str:
        s = str(key)
        if s == "base_x":
            return "base_x (m)"
        if s == "base_y":
            return "base_y (m)"
        if s == "base_theta":
            return "base_theta (rad)"
        # Gripper finger rows: show no unit suffix in table label.
        if s in {"joint_gripper_finger_left", "joint_gripper_finger_right"}:
            return s
        unit = self._joint_units_by_name.get(s)
        if unit not in ("m", "rad"):
            # Fallback display units for common Stretch joints when runtime
            # config/URDF metadata does not carry unit text.
            if s in {
                "joint_lift",
                "joint_arm_l0",
                "joint_arm_l1",
                "joint_arm_l2",
                "joint_arm_l3",
                "wrist_extension",
            }:
                unit = "m"
            elif s in {
                "joint_wrist_yaw",
                "joint_wrist_pitch",
                "joint_wrist_roll",
                "joint_head_pan",
                "joint_head_tilt",
                "joint_mobile_base_rotation",
                "joint_base_theta",
            }:
                unit = "rad"
        if unit in ("m", "rad"):
            return f"{s} ({unit})"
        return s

    def create_joint_state_panel(self):
        """Create realtime joint command/state panel."""
        state_group = QGroupBox("Joint State")
        state_layout = QVBoxLayout()

        self.joint_state_table = QTableWidget(len(self.JOINT_TABLE_NAMES), 4)
        self.joint_state_table.setHorizontalHeaderLabels(["", "Joint Name", "Action", "Observation"])
        self.joint_state_table.verticalHeader().setVisible(False)
        self.joint_state_table.setAlternatingRowColors(True)
        self.joint_state_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.joint_state_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.joint_state_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.joint_state_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.joint_state_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.joint_state_table.setColumnWidth(0, 28)
        self.joint_state_table.horizontalHeader().setStretchLastSection(False)
        display_names = [self._joint_table_display_name(name) for name in self.JOINT_TABLE_NAMES]
        fm = self.joint_state_table.fontMetrics()
        joint_name_w = max((fm.horizontalAdvance(str(n)) for n in display_names), default=140) + 24
        action_w = max(100, fm.horizontalAdvance("Action") + 24)
        obs_w = max(130, fm.horizontalAdvance("Observation") + 24)
        self.joint_state_table.setColumnWidth(1, int(joint_name_w))
        self.joint_state_table.setColumnWidth(2, int(action_w))
        self.joint_state_table.setColumnWidth(3, int(obs_w))
        table_w = int(28 + joint_name_w + action_w + obs_w + 22)
        self.joint_state_table.setMinimumWidth(max(430, table_w))
        self.joint_sync_checkboxes: list[QCheckBox] = []
        for row, name in enumerate(display_names):
            key = str(self.JOINT_TABLE_NAMES[row]) if row < len(self.JOINT_TABLE_NAMES) else ""
            cb = QCheckBox()
            cb.setChecked(key in {"base_x", "base_y", "base_theta"})
            cb_container = QWidget()
            cb_layout = QHBoxLayout(cb_container)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cb_layout.addWidget(cb)
            self.joint_state_table.setCellWidget(row, 0, cb_container)
            self.joint_sync_checkboxes.append(cb)
            self.joint_state_table.setItem(row, 1, QTableWidgetItem(name))
            self.joint_state_table.setItem(row, 2, QTableWidgetItem("--"))
            self.joint_state_table.setItem(row, 3, QTableWidgetItem("--"))

        self.joint_sync_all_checkbox = QCheckBox(self.joint_state_table.horizontalHeader())
        self.joint_sync_all_checkbox.setChecked(False)
        self.joint_sync_all_checkbox.stateChanged.connect(self._on_joint_sync_all_toggled)
        self.joint_sync_all_checkbox.setToolTip("Check/uncheck all joint rows")
        self.joint_sync_all_checkbox.show()
        hdr = self.joint_state_table.horizontalHeader()
        hdr.sectionResized.connect(self._position_joint_sync_all_checkbox)
        hdr.geometriesChanged.connect(self._position_joint_sync_all_checkbox)
        hdr.sectionMoved.connect(self._position_joint_sync_all_checkbox)
        self._position_joint_sync_all_checkbox()

        row_h = self.joint_state_table.verticalHeader().defaultSectionSize()
        hdr_h = self.joint_state_table.horizontalHeader().height()
        frame = self.joint_state_table.frameWidth()
        table_h = int(hdr_h + (row_h * len(self.JOINT_TABLE_NAMES)) + (2 * frame) + 2)
        self.joint_state_table.setFixedHeight(table_h)
        state_layout.addWidget(self.joint_state_table)

        self.sync_joints_cmd_button = QPushButton("Sync Selected Joints")
        self.sync_joints_cmd_button.clicked.connect(self._on_sync_selected_joints_clicked)
        state_layout.addWidget(self.sync_joints_cmd_button)

        state_group.setLayout(state_layout)
        return state_group

    def _position_joint_sync_all_checkbox(self):
        if not (hasattr(self, "joint_state_table") and hasattr(self, "joint_sync_all_checkbox")):
            return
        header = self.joint_state_table.horizontalHeader()
        cb = self.joint_sync_all_checkbox
        w = int(header.sectionSize(0))
        h = int(header.height())
        hint = cb.sizeHint()
        x = max(0, (w - int(hint.width())) // 2)
        y = max(0, (h - int(hint.height())) // 2)
        cb.move(x, y)

    def _sync_base_cmd_from_observation(self, *, update_status: bool = True) -> bool:
        ok = False
        try:
            if hasattr(self.ros_node, "sync_base_command_pose_to_observation"):
                ok = bool(self.ros_node.sync_base_command_pose_to_observation())
        except Exception as exc:
            ok = False
            if update_status and hasattr(self, "status_label"):
                self.status_label.setText(f"Sync base cmd failed: {exc}")
                self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")
        if update_status and ok and hasattr(self, "status_label"):
            self.status_label.setText("Synced command base x/y/theta from observation")
            self.status_label.setStyleSheet("QLabel { color: #1e88e5; font-size: 10px; }")
        elif update_status and (not ok) and hasattr(self, "status_label"):
            self.status_label.setText("Sync base cmd failed: no valid observation")
            self.status_label.setStyleSheet("QLabel { color: orange; font-size: 10px; }")
        if hasattr(self, "_update_joint_state_table"):
            try:
                self._update_joint_state_table()
            except Exception:
                pass
        return ok

    def _on_sync_base_cmd_clicked(self):
        self._sync_base_cmd_from_observation(update_status=True)

    def _on_joint_sync_all_toggled(self, state: int):
        checked = bool(state)
        if not hasattr(self, "joint_sync_checkboxes"):
            return
        for cb in self.joint_sync_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

    def _selected_joint_sync_rows(self) -> list[int]:
        rows: list[int] = []
        if not hasattr(self, "joint_sync_checkboxes"):
            return rows
        for idx, cb in enumerate(self.joint_sync_checkboxes):
            if bool(cb.isChecked()):
                rows.append(int(idx))
        return rows

    def _on_sync_selected_joints_clicked(self):
        rows = self._selected_joint_sync_rows()
        if not rows:
            self._set_status("Sync: select at least one joint row", "QLabel { color: orange; font-size: 10px; }")
            return

        selected_keys: list[str] = []
        for r in rows:
            idx = int(r)
            if 0 <= idx < len(self.JOINT_TABLE_NAMES):
                selected_keys.append(str(self.JOINT_TABLE_NAMES[idx]))
        base_keys = {"base_x", "base_y", "base_theta"}
        sync_base = any((k in base_keys) for k in selected_keys)
        sync_nonbase = any((k not in base_keys) for k in selected_keys)

        ok_nonbase = True
        ok_base = True
        if sync_nonbase:
            try:
                ok_nonbase = bool(self.ros_node.sync_nonbase_command_joints_to_observation())
            except Exception:
                ok_nonbase = False
        if sync_base:
            ok_base = bool(self._sync_base_cmd_from_observation(update_status=False))

        self._update_joint_state_table()
        if ok_nonbase and ok_base:
            self._set_status("Sync complete", "QLabel { color: #1e88e5; font-size: 10px; }")
        else:
            self._set_status(
                f"Sync partial/failed (joints={'ok' if ok_nonbase else 'fail'}, base={'ok' if ok_base else 'fail'})",
                "QLabel { color: orange; font-size: 10px; }",
            )

    def _on_sync_joints_cmd_clicked(self):
        ok = False
        try:
            if hasattr(self.ros_node, "sync_nonbase_command_joints_to_observation"):
                ok = bool(self.ros_node.sync_nonbase_command_joints_to_observation())
        except Exception as exc:
            ok = False
            if hasattr(self, "status_label"):
                self.status_label.setText(f"Sync joints cmd failed: {exc}")
                self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")
        if ok and hasattr(self, "status_label"):
            self.status_label.setText("Synced non-base command joints from observation")
            self.status_label.setStyleSheet("QLabel { color: #1e88e5; font-size: 10px; }")
        elif (not ok) and hasattr(self, "status_label"):
            self.status_label.setText("Sync joints cmd failed: no valid observation")
            self.status_label.setStyleSheet("QLabel { color: orange; font-size: 10px; }")
        self._update_joint_state_table()

    def _on_home_pose_clicked(self):
        self._set_status("Sending robot to startup home pose...", "QLabel { color: #1e88e5; font-size: 10px; }")

        def run():
            try:
                ok = False
                if hasattr(self.ros_node, "move_to_startup_home_pose"):
                    ok = bool(self.ros_node.move_to_startup_home_pose(timeout_s=14.0))
                if ok:
                    try:
                        if isinstance(DEFAULT_INIT_CMD_QPOS8, (list, tuple)) and len(DEFAULT_INIT_CMD_QPOS8) >= 8:
                            self._set_manual_gripper_override(float(DEFAULT_INIT_CMD_QPOS8[7]))
                    except Exception:
                        pass
                    self._set_status("Home pose reached.", "QLabel { color: green; font-size: 10px; }")
                else:
                    self._set_status("Home pose command failed.", "QLabel { color: red; font-size: 10px; }")
            except Exception as exc:
                self._set_status(f"Home pose failed: {exc}", "QLabel { color: red; font-size: 10px; }")
            finally:
                try:
                    self._update_joint_state_table()
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()

    def _update_joint_state_table(self):
        if not hasattr(self, "joint_state_table"):
            return
        try:
            sensors = self.ros_node.get_sensor_snapshot()
        except Exception:
            return

        cmd_q = sensors.get("observation.qpos_published")
        obs_q = sensors.get("observation.qpos_actual")
        cmd_pose = sensors.get("observation.command_base_pose_xytheta")
        obs_pose = sensors.get("observation.base_pose_xytheta")
        obs_joint_names = sensors.get("observation.joint_state.name")
        obs_joint_pos = sensors.get("observation.joint_state.position")

        if not isinstance(cmd_q, list):
            cmd_q = []
        if not isinstance(obs_q, list):
            obs_q = []
        if not (isinstance(cmd_pose, list) and len(cmd_pose) >= 3):
            cmd_pose = [0.0, 0.0, 0.0]
        if not (isinstance(obs_pose, list) and len(obs_pose) >= 3):
            obs_pose = [0.0, 0.0, 0.0]
        if not isinstance(obs_joint_names, list):
            obs_joint_names = []
        if not isinstance(obs_joint_pos, list):
            obs_joint_pos = []

        cmd10 = [float(v) for v in cmd_q[:10]] + [0.0] * max(0, 10 - len(cmd_q))
        obs10 = [float(v) for v in obs_q[:10]] + [0.0] * max(0, 10 - len(obs_q))
        obs_by_name: dict[str, float] = {}
        n_obs = min(len(obs_joint_names), len(obs_joint_pos))
        for i in range(n_obs):
            try:
                k = str(obs_joint_names[i]).strip()
                if not k:
                    continue
                v = float(obs_joint_pos[i])
                if math.isfinite(v):
                    obs_by_name[k] = v
            except Exception:
                continue

        cmd_known: dict[str, float] = {
            "joint_lift": float(cmd10[1]),
            "joint_arm_l0": float(cmd10[0]) / 4.0,
            "joint_wrist_yaw": float(cmd10[2]),
            "joint_wrist_pitch": float(cmd10[3]),
            "joint_wrist_roll": float(cmd10[4]),
            "joint_head_pan": float(cmd10[5]),
            "joint_head_tilt": float(cmd10[6]),
            "joint_gripper_finger_left": float(cmd10[7]),
            "joint_mobile_base_translation": float(cmd_pose[0]),
            "joint_mobile_base_rotation": float(cmd_pose[2]),
            "joint_base_x": float(cmd_pose[0]),
            "joint_base_theta": float(cmd_pose[2]),
        }
        obs_known: dict[str, float] = {
            "joint_lift": float(obs10[1]),
            "joint_arm_l0": float(obs10[0]) / 4.0,
            "joint_wrist_yaw": float(obs10[2]),
            "joint_wrist_pitch": float(obs10[3]),
            "joint_wrist_roll": float(obs10[4]),
            "joint_head_pan": float(obs10[5]),
            "joint_head_tilt": float(obs10[6]),
            "joint_gripper_finger_left": float(obs10[7]),
            "joint_mobile_base_translation": float(obs_pose[0]),
            "joint_mobile_base_rotation": float(obs_pose[2]),
            "joint_base_x": float(obs_pose[0]),
            "joint_base_theta": float(obs_pose[2]),
        }

        for row, key in enumerate(self.JOINT_TABLE_NAMES):
            cmd_text = "--"
            obs_text = "--"
            if key == "base_x":
                cmd_text = f"{float(cmd_pose[0]):+.5f}"
                obs_text = f"{float(obs_pose[0]):+.5f}"
            elif key == "base_y":
                cmd_text = f"{float(cmd_pose[1]):+.5f}"
                obs_text = f"{float(obs_pose[1]):+.5f}"
            elif key == "base_theta":
                cmd_text = f"{float(cmd_pose[2]):+.5f}"
                obs_text = f"{float(obs_pose[2]):+.5f}"
            else:
                cmd_v = cmd_known.get(key)
                obs_v = obs_known.get(key, obs_by_name.get(key))
                # Display-only normalization:
                # table row "joint_arm_l0" is shown as full extension (x4)
                # while internal handling remains unchanged.
                if key == "joint_arm_l0":
                    if isinstance(cmd_v, (int, float)) and math.isfinite(float(cmd_v)):
                        cmd_v = float(cmd_v) * 4.0
                    if isinstance(obs_v, (int, float)) and math.isfinite(float(obs_v)):
                        obs_v = float(obs_v) * 4.0
                if isinstance(cmd_v, (int, float)) and math.isfinite(float(cmd_v)):
                    cmd_text = f"{float(cmd_v):+.5f}"
                if isinstance(obs_v, (int, float)) and math.isfinite(float(obs_v)):
                    obs_text = f"{float(obs_v):+.5f}"
            item_cmd = self.joint_state_table.item(row, 2)
            item_obs = self.joint_state_table.item(row, 3)
            if item_cmd is not None:
                item_cmd.setText(cmd_text)
            if item_obs is not None:
                item_obs.setText(obs_text)

    def create_object_list(self):
        """Create object list widget"""
        widget = QGroupBox("")
        layout = QVBoxLayout()
        top_controls = QWidget()
        self._variation_controls_widget = top_controls
        top_controls_layout = QHBoxLayout(top_controls)
        top_controls_layout.setContentsMargins(0, 0, 0, 0)
        top_controls_layout.setSpacing(6)

        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(4)
        left_col.addWidget(self._create_user_profile_group())

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)
        right_col.addWidget(self._create_randomize_src_group())
        right_col.addWidget(self._create_latency_group())

        top_controls_layout.addLayout(left_col, 1)
        top_controls_layout.addLayout(right_col, 1)
        layout.addWidget(top_controls)

        # List widget (scrollable)
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.on_object_selected)
        self.object_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.object_list.setMinimumHeight(84)
        self.object_list.setMaximumHeight(145)
        # Keep segmentation trigger above object list.
        if hasattr(self, "segment_button"):
            layout.addWidget(self.segment_button)
        layout.addWidget(self.object_list, stretch=0)

        # Action buttons
        btn_layout = QHBoxLayout()

        self.center_button = QPushButton("Center")
        self.center_button.setToolTip("Center camera on selected object")
        self.center_button.clicked.connect(self.center_camera_on_object)
        self.center_button.setEnabled(False)
        self.center_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.center_button.setVisible(False)
        btn_layout.addWidget(self.center_button)

        self.reach_button = QPushButton("Reach")
        self.reach_button.setToolTip("Move arm above selected object (10cm clearance)")
        self.reach_button.clicked.connect(self.reach_object)
        self.reach_button.setEnabled(False)
        self.reach_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.reach_button.setVisible(False)
        btn_layout.addWidget(self.reach_button)

        self.grasp_button = QPushButton("Grasp")
        self.grasp_button.setToolTip("Move to and grasp selected object")
        self.grasp_button.clicked.connect(self.grasp_object)
        self.grasp_button.setEnabled(False)
        self.grasp_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.grasp_button.setVisible(False)
        btn_layout.addWidget(self.grasp_button)

        # Keep these buttons for internal logic, but hide from UI.
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(0)
        layout.addLayout(btn_layout)

        # Return/Pause side-by-side row.
        self.return_pause_widget = QWidget()
        action_row = QHBoxLayout(self.return_pause_widget)
        action_row.setContentsMargins(0, 0, 0, 0)
        self.return_button = QPushButton("Return to Home")
        self.return_button.setToolTip("Return arm and base to position before last reach/grasp")
        self.return_button.clicked.connect(self.return_to_start)
        self.return_button.setEnabled(False)
        self.return_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        action_row.addWidget(self.return_button)
        self.return_shortcut = QShortcut(QKeySequence("Tab"), self)
        self.return_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.return_shortcut.activated.connect(self._on_return_shortcut)

        self.play_pause_button = QPushButton("Pause")
        self.play_pause_button.setToolTip("Pause running action / Continue paused action")
        self.play_pause_button.clicked.connect(self.on_play_pause_clicked)
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        action_row.addWidget(self.play_pause_button)
        layout.addWidget(self.return_pause_widget)
        self.space_pause_shortcut = QShortcut(QKeySequence("Space"), self)
        self.space_pause_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.space_pause_shortcut.activated.connect(self._on_space_pause_shortcut)

        self.next_goal_button = QPushButton("Go To Next Goal")
        self.next_goal_button.setToolTip("Execute next queued goal (or skip current action and move to next queued goal)")
        self.next_goal_button.clicked.connect(self.go_to_next_goal)
        self.next_goal_button.setEnabled(False)
        self.next_goal_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.next_goal_button.setVisible(False)
        layout.addWidget(self.next_goal_button)

        self.goal_queue_label = QLabel("Queued goals: (none)")
        self.goal_queue_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        self.goal_queue_label.setWordWrap(True)
        self.goal_queue_label.setVisible(False)
        layout.addWidget(self.goal_queue_label)

        # Manual region-based grasp workflow (v8).
        manual_group = QGroupBox("Manual Grasp Regions")
        manual_layout = QVBoxLayout()
        manual_layout.setSpacing(4)

        self.manual_region_tree = QTreeWidget()
        self.manual_region_tree.setHeaderLabels(["Region / Grasp Points"])
        self.manual_region_tree.setMinimumHeight(120)
        self.manual_region_tree.setMaximumHeight(220)
        self.manual_region_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.manual_region_tree.itemSelectionChanged.connect(self._on_manual_region_tree_selection_changed)
        manual_layout.addWidget(self.manual_region_tree)

        manual_btn_row1 = QHBoxLayout()
        self.manual_compute_button = QPushButton("Compute Grasp")
        self.manual_compute_button.setToolTip("Compute grasp orientation from selected region + points")
        self.manual_compute_button.clicked.connect(self.compute_selected_manual_region_grasp)
        manual_btn_row1.addWidget(self.manual_compute_button)

        self.manual_align_button = QPushButton("Align Gripper")
        self.manual_align_button.setToolTip("Align wrist yaw/pitch only; no reach/base move")
        self.manual_align_button.clicked.connect(self.align_gripper_to_selected_manual_region)
        manual_btn_row1.addWidget(self.manual_align_button)
        manual_layout.addLayout(manual_btn_row1)

        manual_btn_row2 = QHBoxLayout()
        self.manual_execute_button = QPushButton("Execute Region Grasp")
        self.manual_execute_button.setToolTip("Run IK grasp execution for selected region")
        self.manual_execute_button.clicked.connect(self.execute_selected_manual_region_grasp)
        manual_btn_row2.addWidget(self.manual_execute_button)

        self.manual_delete_button = QPushButton("Delete Region")
        self.manual_delete_button.setToolTip("Delete selected manual region entry")
        self.manual_delete_button.clicked.connect(self.delete_selected_manual_region)
        manual_btn_row2.addWidget(self.manual_delete_button)
        manual_layout.addLayout(manual_btn_row2)

        manual_hint = QLabel(
            "Right-click image -> Draw Grasp Rectangle. Then click 2 grip-tip points inside it."
        )
        manual_hint.setWordWrap(True)
        manual_hint.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        manual_layout.addWidget(manual_hint)
        manual_group.setLayout(manual_layout)
        if bool(MANUAL_GRASP_REGION_UI_VISIBLE):
            layout.addWidget(manual_group)
        else:
            manual_group.setVisible(False)

        # Demonstration recording controls
        record_group = QGroupBox("Task Description")
        record_layout = QGridLayout()
        record_layout.setSpacing(4)

        record_layout.addWidget(QLabel("Task"), 0, 0)
        self.task_combo = QComboBox()
        self.task_combo.addItems(TASK_DROPDOWN_OPTIONS)
        default_task_idx = max(0, self.task_combo.findText(str(TASK_DEFAULT_NAME)))
        self.task_combo.setCurrentIndex(default_task_idx)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        record_layout.addWidget(self.task_combo, 0, 1, 1, 3)

        self.add_task_button = QPushButton("Add Task")
        self.add_task_button.setToolTip("Load latest saved prompt for selected task into prompt box")
        self.add_task_button.clicked.connect(self._on_add_task_clicked)
        self.add_task_button.setMinimumHeight(UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX)
        record_layout.addWidget(self.add_task_button, 0, 4)

        record_layout.addWidget(QLabel("Saved Prompts"), 1, 0)
        self.task_prompt_history_combo = QComboBox()
        self.task_prompt_history_combo.currentIndexChanged.connect(self._on_saved_prompt_selected)
        record_layout.addWidget(self.task_prompt_history_combo, 1, 1, 1, 4)

        record_layout.addWidget(QLabel("Prompt"), 2, 0)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("e.g. pick up the red block and place it in the tray")
        self.prompt_input.setFixedHeight(56)
        self.prompt_input.textChanged.connect(self.on_prompt_changed)
        record_layout.addWidget(self.prompt_input, 2, 1, 1, 4)

        self.save_task_prompt_button = QPushButton("Save Prompt")
        self.save_task_prompt_button.setToolTip("Save current prompt under selected task")
        self.save_task_prompt_button.clicked.connect(self._on_save_prompt_for_task_clicked)
        self.save_task_prompt_button.setMinimumHeight(UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX)
        record_layout.addWidget(self.save_task_prompt_button, 3, 1, 1, 4)

        record_layout.addWidget(QLabel("Folder"), 4, 0)
        self.record_folder_input = QLineEdit(self.dataset_root)
        self.record_folder_input.textChanged.connect(self.on_record_folder_changed)
        record_layout.addWidget(self.record_folder_input, 4, 1, 1, 3)

        self.browse_record_folder_button = QPushButton("Browse")
        self.browse_record_folder_button.clicked.connect(self.browse_record_folder)
        self.browse_record_folder_button.setMinimumHeight(UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX)
        record_layout.addWidget(self.browse_record_folder_button, 4, 4)

        record_layout.addWidget(QLabel("RGB Format"), 5, 0)
        self.record_rgb_format_combo = QComboBox()
        self.record_rgb_format_combo.addItems(["jpg", "png"])
        self.record_rgb_format_combo.setCurrentText(self.record_rgb_format)
        self.record_rgb_format_combo.currentTextChanged.connect(self.on_record_rgb_format_changed)
        record_layout.addWidget(self.record_rgb_format_combo, 5, 1, 1, 4)

        self.record_toggle_button = QPushButton("Record")
        self.record_toggle_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.record_toggle_button.setToolTip("Start/stop recording demonstration in LeRobot-style layout")
        self.record_toggle_button.clicked.connect(self.toggle_demo_recording)
        self.record_toggle_button.setVisible(False)
        record_layout.addWidget(self.record_toggle_button, 6, 0, 1, 5)
        self.record_shortcut_r = QShortcut(QKeySequence("R"), self)
        self.record_shortcut_r.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.record_shortcut_r.activated.connect(self._on_record_shortcut)
        self.execute_goals_shortcut = QShortcut(QKeySequence("E"), self)
        self.execute_goals_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.execute_goals_shortcut.activated.connect(self._on_execute_goals_shortcut)

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

    def _create_user_profile_group(self):
        group = QGroupBox("User Variation")
        gl = QVBoxLayout()
        gl.setSpacing(4)

        def set_delay(v):
            self.command_smoothing_delay = float(v)
            self.ros_node.set_command_smoothing_delay(float(v))
            if hasattr(self.ros_node, "set_base_rotate_step_delay"):
                self.ros_node.set_base_rotate_step_delay(float(v))

        gl.addLayout(
            self._create_speed_slider(
                0.005,
                0.20,
                self.command_smoothing_delay,
                set_delay,
                reverse=False,
                value_fmt="{:.3f}s",
                label_text="Speed",
            )
        )

        def set_arm_std(m):
            globals()["AUTO_LOOP_PICK_VARIATION_OVERSHOOT_ARM_M"] = float(m)
            globals()["AUTO_LOOP_PICK_VARIATION_SHORT_ARM_M"] = float(m)

        def set_base_x_std(m):
            globals()["AUTO_LOOP_PICK_VARIATION_OVERSHOOT_BASE_X_M"] = float(m)
            globals()["AUTO_LOOP_PICK_VARIATION_SHORT_BASE_X_M"] = float(m)

        gl.addLayout(
            self._create_cm_std_slider(
                "Along X (std)",
                float(AUTO_LOOP_PICK_VARIATION_OVERSHOOT_BASE_X_M),
                set_base_x_std,
            )
        )
        gl.addLayout(
            self._create_cm_std_slider(
                "Along Y (std)",
                float(AUTO_LOOP_PICK_VARIATION_OVERSHOOT_ARM_M),
                set_arm_std,
            )
        )
        group.setLayout(gl)
        return group

    def _create_randomize_src_group(self):
        group = QGroupBox("Source Variation")
        gl = QVBoxLayout()
        gl.setSpacing(4)

        def set_base_x_std(m):
            globals()["AUTO_LOOP_GRASP_PLACE_JITTER_BASE_X_M"] = float(m)

        def set_arm_std(m):
            globals()["AUTO_LOOP_GRASP_PLACE_JITTER_ARM_M"] = float(m)

        gl.addLayout(
            self._create_cm_std_slider(
                "Along X (std)",
                float(AUTO_LOOP_GRASP_PLACE_JITTER_BASE_X_M),
                set_base_x_std,
            )
        )
        gl.addLayout(
            self._create_cm_std_slider(
                "Along Y (std)",
                float(AUTO_LOOP_GRASP_PLACE_JITTER_ARM_M),
                set_arm_std,
            )
        )
        group.setLayout(gl)
        return group

    def _create_latency_group(self):
        widget = QWidget()
        gl = QVBoxLayout(widget)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(2)

        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(QLabel("System Latency"))

        self.latency_slider = QSlider(Qt.Orientation.Horizontal)
        self.latency_slider.setRange(0, 4)
        self.latency_slider.setSingleStep(1)
        self.latency_slider.setPageStep(1)
        self.latency_slider.setTickInterval(1)
        self.latency_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.latency_slider.setFixedHeight(22)
        speed_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        self.latency_slider.setValue(0)

        self.latency_value_label = QLabel("1x")
        self.latency_value_label.setFixedWidth(38)
        self.latency_value_label.setStyleSheet("QLabel { font-size: 10px; }")

        def on_latency_change(idx: int):
            i = int(np.clip(int(idx), 0, len(speed_values) - 1))
            speed = float(speed_values[i])
            self.latency_value_label.setText(f"{speed:g}x")
            base_delay = float(self._ui_step_defaults[UI_STEP_IDX_SMOOTH_DELAY])
            delay_s = float(np.clip(base_delay / max(1e-6, speed), 0.005, 0.20))
            self.command_smoothing_delay = delay_s
            self.ros_node.set_command_smoothing_delay(delay_s)
            if hasattr(self.ros_node, "set_base_rotate_step_delay"):
                self.ros_node.set_base_rotate_step_delay(delay_s)

        self.latency_slider.valueChanged.connect(on_latency_change)

        row.addWidget(self.latency_slider, 1)
        row.addWidget(self.latency_value_label)
        gl.addLayout(row)
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
        elif state == 'awaiting_post_reach_release':
            self.play_pause_button.setEnabled(True)
            self.play_pause_button.setText("Continue")
            self.return_button.setEnabled(False)
        self._update_next_goal_button_state()

    def _rebuild_queued_goal_map_from_sequence(self) -> None:
        keys = (
            "grasp",
            "reach",
            "place_object",
            "release",
            "drag",
            "drag_curve",
            "lift_delta",
            "stretch_delta",
            "translate_delta",
        )
        self.queued_goals = {k: None for k in keys}
        for g in list(getattr(self, "queued_goal_sequence", [])):
            if not isinstance(g, dict):
                continue
            kind = str(g.get("kind", ""))
            if kind in self.queued_goals:
                self.queued_goals[kind] = g

    def _goal_sequence_order(self):
        seq = getattr(self, "queued_goal_sequence", None)
        if isinstance(seq, list):
            return [g for g in seq if isinstance(g, dict)]
        # Backward-compatible fallback (should be empty in normal flow).
        goals = []
        for k in ("grasp", "reach", "place_object", "release", "drag", "drag_curve", "lift_delta", "stretch_delta", "translate_delta"):
            g = self.queued_goals.get(k)
            if isinstance(g, dict):
                goals.append(g)
        return goals

    def _goal_sequence_has_next(self):
        return self.queued_goal_cursor < len(self._goal_sequence_order())

    def _update_goal_queue_label(self):
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            QTimer.singleShot(0, self._update_goal_queue_label)
            return
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
            kind = str(g.get("kind", "?"))
            if kind == "drag":
                sx = g.get("px")
                sy = g.get("py")
                ex = g.get("end_px")
                ey = g.get("end_py")
                labels.append(f"{prefix}{idx+1}. drag ({sx},{sy}) -> ({ex},{ey})")
            elif kind == "grasp":
                px = g.get("px")
                py = g.get("py")
                rotate_deg = g.get("grasp_rotate_deg")
                is_rotate = isinstance(rotate_deg, (int, float)) and abs(float(rotate_deg)) > 1e-6
                is_precise = bool(g.get("precise_grasp", False))
                if bool(is_precise):
                    labels.append(f"{prefix}{idx+1}. grasp precise ({px},{py})")
                elif bool(g.get("post_grasp_lift", True)):
                    labels.append(f"{prefix}{idx+1}. grasp+lift ({px},{py})")
                elif bool(is_rotate):
                    labels.append(f"{prefix}{idx+1}. grasp+rotate({float(rotate_deg):+.1f}deg) ({px},{py})")
                else:
                    labels.append(f"{prefix}{idx+1}. grasp ({px},{py})")
            elif kind == "drag_curve":
                pts = g.get("path_px")
                npts = len(pts) if isinstance(pts, list) else 0
                sx = g.get("px")
                sy = g.get("py")
                ex = g.get("end_px")
                ey = g.get("end_py")
                no_adj = bool(g.get("no_height_adjustment", False))
                if no_adj:
                    htxt = "keep_lift"
                else:
                    h_cm = float(g.get("surface_height_offset_cm", float(g.get("surface_height_offset_m", 0.0)) * 100.0))
                    htxt = f"h={h_cm:+.1f}cm"
                labels.append(f"{prefix}{idx+1}. drag_curve ({sx},{sy}) -> ({ex},{ey}), n={npts}, {htxt}")
            elif kind == "release":
                src = str(g.get("source_goal_kind", "?"))
                labels.append(f"{prefix}{idx+1}. release (prev end: {src})")
            elif kind in ("lift_delta", "stretch_delta", "translate_delta"):
                delta_cm = float(g.get("delta_cm", float(g.get("delta_m", 0.0)) * 100.0))
                axis = {
                    "lift_delta": "lift",
                    "stretch_delta": "stretch",
                    "translate_delta": "translate",
                }.get(kind, kind)
                labels.append(f"{prefix}{idx+1}. {axis} {delta_cm:+.2f} cm")
            else:
                px = g.get("px")
                py = g.get("py")
                if kind == "reach" and bool(g.get("precise_place", False)):
                    labels.append(f"{prefix}{idx+1}. place precise ({px},{py})")
                else:
                    labels.append(f"{prefix}{idx+1}. {kind} ({px},{py})")
        self.goal_queue_label.setText("Queued goals:\n" + "\n".join(labels))
        self.goal_queue_label.setStyleSheet("QLabel { color: #555; font-size: 10px; }")

    def _update_next_goal_button_state(self):
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            QTimer.singleShot(0, self._update_next_goal_button_state)
            return
        if not hasattr(self, "next_goal_button"):
            return
        with self._action_lock:
            st = self._action_state
        enabled = False
        if self._goal_sequence_has_next():
            # Allow starting next goal from idle, or skipping to next during an active/paused action.
            enabled = st in ('idle', 'running', 'paused', 'awaiting_confirm', 'awaiting_post_reach_release')
        self.next_goal_button.setEnabled(enabled)

    def _reset_goal_sequence_progress(self):
        self.queued_goal_cursor = 0
        self.queued_sequence_started = False
        self._run_all_queued_goals = False
        self._deferred_next_goal_start = False
        self._skip_to_next_goal_requested = False
        self._queued_drag_repeat_count = 1
        self._queued_drag_return_to_start = False
        self._update_goal_queue_label()
        self._update_next_goal_button_state()

    def _apply_status_update(self, text, style):
        self.status_label.setText(text)
        if style:
            self.status_label.setStyleSheet(style)

    def _set_status(self, text, style=None):
        msg = str(text)
        print(f"[ui_status] {msg}", flush=True)
        self.ui_status_signal.emit(msg, style or "")

    def _update_device_yaw_mode_button(self) -> None:
        if not hasattr(self, "device_yaw_mode_button"):
            return
        if bool(getattr(self, "device_yaw_reverse", False)):
            self.device_yaw_mode_button.setText("Yaw Mode: Reverse")
            self.device_yaw_mode_button.setToolTip("Device yaw sign is inverted before sending")
        else:
            self.device_yaw_mode_button.setText("Yaw Mode: Normal")
            self.device_yaw_mode_button.setToolTip("Device yaw sign is sent directly")

    def _toggle_device_yaw_mode(self) -> None:
        self.device_yaw_reverse = not bool(getattr(self, "device_yaw_reverse", False))
        self._update_device_yaw_mode_button()
        mode = "reverse" if self.device_yaw_reverse else "normal"
        self._set_status(f"Device yaw mode: {mode}", "QLabel { color: #1e88e5; font-size: 10px; }")

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
        self._set_return_enabled(False)
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

        if state in ('paused', 'awaiting_confirm', 'awaiting_post_reach_release'):
            self._set_action_state('running')
            self.status_label.setText("Resuming action...")
            self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")
            return

        if state == 'awaiting_post_grasp':
            self._set_action_state('running')
            self.status_label.setText("Resuming action...")
            self.status_label.setStyleSheet("QLabel { color: blue; font-size: 10px; }")

    def _on_space_pause_shortcut(self) -> None:
        if not hasattr(self, "play_pause_button"):
            return
        if not self.play_pause_button.isEnabled():
            return
        self.on_play_pause_clicked()

    def _on_return_shortcut(self) -> None:
        if self._text_input_has_focus():
            return
        if not hasattr(self, "return_button"):
            return
        if not self.return_button.isEnabled():
            return
        self.return_to_start()

    def _text_input_has_focus(self) -> bool:
        fw = QApplication.focusWidget()
        return isinstance(fw, QLineEdit)

    def _on_record_shortcut(self) -> None:
        if self._text_input_has_focus():
            return
        if hasattr(self, "record_toggle_button") and self.record_toggle_button.isEnabled():
            self.toggle_demo_recording()

    def _on_execute_goals_shortcut(self) -> None:
        if self._text_input_has_focus():
            return
        if hasattr(self, "execute_goals_button") and self.execute_goals_button.isEnabled():
            self.execute_all_queued_goals()

    @staticmethod
    def _wheel_modifier_mode(mods: Qt.KeyboardModifiers) -> str | None:
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        meta = bool(mods & Qt.KeyboardModifier.MetaModifier)
        if meta:
            return None
        # Alt gets priority and tolerates AltGr-style Ctrl+Alt combos.
        if alt and (not shift):
            return "arm_lift"
        if ctrl and (not alt) and (not shift):
            return "base_x"
        if shift and (not alt) and (not ctrl):
            return "arm_extension"
        return None

    def _wheel_modifier_mode_from_event(self, event) -> str | None:
        mods_evt = event.modifiers()
        mods_kbd = QApplication.keyboardModifiers()
        ctrl = bool(mods_evt & Qt.KeyboardModifier.ControlModifier) or bool(mods_kbd & Qt.KeyboardModifier.ControlModifier) or bool(self._wheel_mod_latch.get("ctrl", False))
        alt = bool(mods_evt & Qt.KeyboardModifier.AltModifier) or bool(mods_kbd & Qt.KeyboardModifier.AltModifier) or bool(self._wheel_mod_latch.get("alt", False))
        shift = bool(mods_evt & Qt.KeyboardModifier.ShiftModifier) or bool(mods_kbd & Qt.KeyboardModifier.ShiftModifier) or bool(self._wheel_mod_latch.get("shift", False))
        meta = bool(mods_evt & Qt.KeyboardModifier.MetaModifier) or bool(mods_kbd & Qt.KeyboardModifier.MetaModifier)
        if meta:
            return None
        # Alt gets priority and tolerates Ctrl+Alt (AltGr-like).
        if alt and (not shift):
            return "arm_lift"
        if ctrl and (not alt) and (not shift):
            return "base_x"
        if shift and (not alt) and (not ctrl):
            return "arm_extension"
        return None

    def _apply_wheel_shortcut_step(self, mode: str, direction: int) -> bool:
        sign = 1 if int(direction) > 0 else -1
        if mode == "base_x":
            step_m = float(sign) * float(self.linear_speed)
            timeout_s = max(1.5, 2.0 + 4.0 * abs(float(step_m)))
            fn = getattr(self.ros_node, "_send_manual_base_x_absolute_step", None)
            if callable(fn):
                try:
                    return bool(fn(step_dx=float(step_m), timeout_s=float(timeout_s)))
                except Exception:
                    pass
            try:
                return bool(
                    self.ros_node.move_base_relative(
                        dx=float(step_m),
                        dy=0.0,
                        dtheta=0.0,
                        blocking=False,
                        timeout_s=float(timeout_s),
                    )
                )
            except Exception:
                return False

        if mode not in ("arm_lift", "arm_extension"):
            return False
        delta = float(sign) * float(self.arm_speed)
        if abs(float(delta)) <= 0.0:
            return False
        # Keep parity with manual incremental controls: clear base hold before non-base increments.
        try:
            if self.ros_node.is_base_command_active():
                self.ros_node.stop_base()
        except Exception:
            pass
        try:
            self.robot_controller.adjust_control(mode, float(delta))
            try:
                self.ros_node.publish_commands(force=False)
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _handle_modifier_wheel_event(self, event) -> bool:
        mode = self._wheel_modifier_mode_from_event(event)
        if mode is None:
            return False

        required_control = {
            "base_x": "base_linear",
            "arm_lift": "arm_lift",
            "arm_extension": "arm_extension",
        }.get(mode, "")
        if required_control and (required_control not in self._resolve_available_manual_controls()):
            # Availability probe can be stale/missing on some ROS state snapshots.
            # Fall back to low-level control map before giving up.
            control_map = getattr(self.ros_node, "CONTROL_MAP", {})
            if not (isinstance(control_map, dict) and required_control in control_map):
                event.accept()
                return True

        delta_y = int(event.angleDelta().y())
        if delta_y == 0:
            # Some stacks report wheel under modifiers on x-axis.
            delta_y = int(event.angleDelta().x())
        if delta_y == 0:
            # Trackpad fallback: treat each directional wheel event as one notch.
            pdy = int(event.pixelDelta().y())
            if pdy == 0:
                pdy = int(event.pixelDelta().x())
            if pdy > 0:
                delta_y = 120
            elif pdy < 0:
                delta_y = -120
            else:
                event.accept()
                return True

        self._wheel_notch_accum[mode] = float(self._wheel_notch_accum.get(mode, 0.0) + float(delta_y))
        notch = 120.0
        ticks = 0
        while self._wheel_notch_accum[mode] >= notch:
            ticks += 1
            self._wheel_notch_accum[mode] -= notch
        while self._wheel_notch_accum[mode] <= -notch:
            ticks -= 1
            self._wheel_notch_accum[mode] += notch

        if ticks != 0:
            direction = 1 if ticks > 0 else -1
            for _ in range(abs(int(ticks))):
                self._apply_wheel_shortcut_step(mode, direction)

        event.accept()
        return True

    def eventFilter(self, obj, event):
        try:
            if event is not None and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                down = bool(event.type() == QEvent.Type.KeyPress)
                k = event.key()
                if k in (Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Shift):
                    if k == Qt.Key.Key_Control:
                        self._wheel_mod_latch["ctrl"] = down
                    elif k == Qt.Key.Key_Alt:
                        self._wheel_mod_latch["alt"] = down
                    elif k == Qt.Key.Key_Shift:
                        self._wheel_mod_latch["shift"] = down
            if event is not None and event.type() == QEvent.Type.Wheel:
                if self._handle_modifier_wheel_event(event):
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    @staticmethod
    def _quat_to_yaw_rad(quat: list[Any] | tuple[Any, ...] | None) -> float | None:
        if not isinstance(quat, (list, tuple)) or len(quat) < 4:
            return None
        try:
            x = float(quat[0])
            y = float(quat[1])
            z = float(quat[2])
            w = float(quat[3])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(w)):
            return None
        # ZYX yaw from quaternion.
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _wrap_angle_rad(a: float) -> float:
        return float(math.atan2(math.sin(a), math.cos(a)))

    @staticmethod
    def _build_prec_step_dict(min_val: float, max_val: float) -> dict[int, float]:
        """Map prec levels 0..4 to 5 linearly spaced step sizes (max -> min)."""
        lo = float(min_val)
        hi = float(min_val) + (float(max_val) - float(min_val)) / 3.0
        vals = np.linspace(hi, lo, int(DEVICE_PREC_LEVELS))
        return {idx: float(vals[idx]) for idx in range(int(DEVICE_PREC_LEVELS))}

    def _device_step_from_prec(self, *, prec_level: int, min_val: float, max_val: float) -> float:
        mp = self._build_prec_step_dict(min_val=min_val, max_val=max_val)
        p = int(np.clip(int(prec_level), 0, int(DEVICE_PREC_LEVELS) - 1))
        return float(mp[p])

    def _poll_device_control(self) -> None:
        if not getattr(self, "_device_input_ok", False):
            return
        packets = self.device_bridge.drain()
        if len(packets) == 0:
            return
        dead = float(DEVICE_ENCODER_DEADBAND)
        for payload in packets:
            raw = payload.get("raw")
            if not isinstance(raw, list) or len(raw) < 10:
                continue
            try:
                enc1 = float(raw[6])  # encoder-1: base fwd/back
                enc2 = float(raw[7])  # encoder-2: lift
                enc3 = float(raw[8])  # encoder-3: arm (or gripper when button-1 held)
                b1 = bool(int(float(raw[9])))  # button-1: gripper mode
                b2 = bool(int(float(raw[10])))  # button-2: wrist yaw mode
                precision = max(0.0, float(raw[4]))
                mode = int(float(raw[3]))
            except (TypeError, ValueError, IndexError):
                continue
            if not (
                math.isfinite(enc1) and math.isfinite(enc2) and math.isfinite(enc3) and math.isfinite(precision)
            ):
                continue

            if bool(DEVICE_DEBUG_PRINT) and self._device_debug_print_count < int(DEVICE_DEBUG_PRINT_MAX):
                print(
                    "[device] decoded raw "
                    f"mode={mode} prec={precision:.3f} "
                    f"enc1={enc1:+.3f} enc2={enc2:+.3f} enc3={enc3:+.3f} "
                    f"b1={int(b1)} b2={int(b2)}"
                )
                self._device_debug_print_count += 1

            # Precision selection requested:
            # rem = prec % 2 -> {0,1}; map 0->level 3, 1->level 4
            prec_raw = int(np.clip(int(round(float(precision))), 0, int(DEVICE_PREC_LEVELS) - 1))
            prec_mod = int(prec_raw % 2)
            prec_level = 3 if prec_mod == 0 else 4
            self._device_prec_level = int(prec_mod)
            if hasattr(self, "device_prec_label"):
                self.device_prec_label.setText(f"Device Prec: {prec_mod} (lvl {prec_level})")

            # Map each joint step from slider-range [min,max] via linspace with inverse precision mapping.
            base_step_size = self._device_step_from_prec(
                prec_level=prec_level,
                min_val=float(DEVICE_BASE_STEP_MIN_M),
                max_val=float(DEVICE_BASE_STEP_MAX_M),
            )
            arm_step_size = self._device_step_from_prec(
                prec_level=prec_level,
                min_val=float(DEVICE_ARM_STEP_MIN),
                max_val=float(DEVICE_ARM_STEP_MAX),
            )
            # Apply only when encoder values change; each changed sample contributes:
            #   new_cmd = current_cmd + encoder_value * step
            changed1 = True
            changed2 = True
            changed3 = True

            any_joint_cmd = False

            # Encoder-1: base forward/backward (manual absolute-base-x path as in v9).
            if changed1 and abs(enc1) > dead:
                base_step = -enc1 * base_step_size
                self.ros_node._send_manual_base_x_absolute_step(
                    step_dx=float(base_step),
                    timeout_s=max(1.5, 2.0 + 4.0 * abs(float(base_step))),
                )

            # Encoder-2: lift (cmd_action += delta).
            if changed2 and abs(enc2) > dead:
                delta_lift = -enc2 * arm_step_size
                cur = self.ros_node.get_target_qpos()
                if isinstance(cur, list) and len(cur) >= 2:
                    lo, hi = self.ros_node.JOINT_LIMITS[1]
                    tgt = float(np.clip(float(cur[1]) + float(delta_lift), lo, hi))
                    self.robot_controller.set_control("arm_lift", tgt)
                    any_joint_cmd = True

            # Encoder-3: always controls arm extension.
            if changed3 and abs(enc3) > dead:
                delta_arm = -enc3 * arm_step_size
                cur = self.ros_node.get_target_qpos()
                if isinstance(cur, list) and len(cur) >= 1:
                    lo, hi = self.ros_node.JOINT_LIMITS[0]
                    tgt = float(np.clip(float(cur[0]) + float(delta_arm), lo, hi))
                    self.robot_controller.set_control("arm_extension", tgt)
                    any_joint_cmd = True

            # Button-1: edge-triggered gripper toggle (independent of encoder-3).
            b1_rising = bool(b1) and (not bool(self._device_b1_prev_pressed))
            self._device_b1_prev_pressed = bool(b1)
            if b1_rising:
                grip_open = float(
                    np.clip(
                        float(DEVICE_GRIPPER_TOGGLE_OPEN_JOINT),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                grip_close = float(
                    np.clip(
                        float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                target = grip_open if bool(self._device_b1_open_next) else grip_close
                self._device_b1_open_next = not bool(self._device_b1_open_next)
                self.set_gripper(float(target))
                any_joint_cmd = True

            # Yaw from device is applied around current wrist yaw baseline:
            # target_yaw = baseline_wrist_yaw + device_yaw (optionally sign-reversed).
            yaw_now = self._quat_to_yaw_rad(payload.get("quat"))
            # print(yaw_now)
            if yaw_now is None:
                self._device_b1_active = False
                continue
            if False: # b1
                if not bool(self._device_b1_active):
                    self._device_b1_active = True
                    q_target = self.ros_node.get_target_qpos()
                    wrist_ref = 0.0
                    if isinstance(q_target, list) and len(q_target) >= 3:
                        try:
                            wrist_ref = float(q_target[2])
                        except (TypeError, ValueError):
                            wrist_ref = 0.0
                    self._device_b1_wrist_ref = wrist_ref

                yaw_cmd = -float(yaw_now) if bool(getattr(self, "device_yaw_reverse", False)) else float(yaw_now)
                lo_yaw, hi_yaw = self.ros_node.JOINT_LIMITS[2]
                base_yaw = float(self._device_b1_wrist_ref) if self._device_b1_wrist_ref is not None else 0.0
                target_yaw = float(np.clip(float(base_yaw + yaw_cmd), float(lo_yaw), float(hi_yaw)))
                self.robot_controller.set_control("wrist_yaw", target_yaw)
                any_joint_cmd = True
            else:
                self._device_b1_active = False
                self._device_b1_wrist_ref = None

            if any_joint_cmd:
                try:
                    self.ros_node.publish_commands(force=False)
                except Exception:
                    pass

    def start_control(self, control_name, value):
        """Start continuous control (velocity-based)"""
        self.robot_controller.set_control(control_name, value)

    def start_control_incremental(self, control_name, delta):
        """Start incremental control (position-based)"""
        # Safety: if base command is still latched, clear it before non-base holds.
        if control_name not in ("base_linear", "base_angular"):
            try:
                if self.ros_node.is_base_command_active():
                    self.ros_node.stop_base()
            except Exception:
                pass
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

    def _default_task_prompt_library(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for task in TASK_DROPDOWN_OPTIONS:
            default_prompt = str(TASK_DEFAULT_PROMPTS.get(task, task)).strip()
            out[str(task)] = [default_prompt] if default_prompt else [str(task)]
        return out

    def _load_task_prompt_library(self) -> None:
        lib = self._default_task_prompt_library()
        path = self._task_prompt_library_path
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    raw_tasks = raw.get("tasks", raw)
                    if isinstance(raw_tasks, dict):
                        for task in TASK_DROPDOWN_OPTIONS:
                            vals = raw_tasks.get(task)
                            prompts: list[str] = []
                            if isinstance(vals, list):
                                for p in vals:
                                    s = str(p).strip()
                                    if s:
                                        prompts.append(s)
                            elif isinstance(vals, str):
                                s = vals.strip()
                                if s:
                                    prompts.append(s)
                            if prompts:
                                dedup: list[str] = []
                                seen: set[str] = set()
                                for p in prompts:
                                    if p not in seen:
                                        seen.add(p)
                                        dedup.append(p)
                                lib[str(task)] = dedup
        except Exception as exc:
            print(f"[task_prompt_library] load failed: {exc}")
        self._task_prompt_library = lib
        self._save_task_prompt_library()

    def _save_task_prompt_library(self) -> None:
        path = self._task_prompt_library_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "tasks": self._task_prompt_library,
                "task_order": TASK_DROPDOWN_OPTIONS,
                "updated_at": float(time.time()),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            print(f"[task_prompt_library] save failed: {exc}")

    def _prompts_for_task(self, task_name: str) -> list[str]:
        task = str(task_name)
        prompts = self._task_prompt_library.get(task, [])
        if isinstance(prompts, list):
            out = [str(p).strip() for p in prompts if str(p).strip()]
            if out:
                return out
        default_prompt = str(TASK_DEFAULT_PROMPTS.get(task, task)).strip()
        return [default_prompt] if default_prompt else [task]

    def _set_prompt_input_text(self, text: str) -> None:
        if not hasattr(self, "prompt_input"):
            return
        try:
            if hasattr(self.prompt_input, "setPlainText"):
                self.prompt_input.setPlainText(str(text))
            else:
                self.prompt_input.setText(str(text))
        except Exception:
            pass

    def _get_prompt_input_text(self) -> str:
        if not hasattr(self, "prompt_input"):
            return ""
        try:
            if hasattr(self.prompt_input, "toPlainText"):
                return str(self.prompt_input.toPlainText() or "")
            if hasattr(self.prompt_input, "text"):
                return str(self.prompt_input.text() or "")
        except Exception:
            pass
        return ""

    def _refresh_prompt_history_for_task(self, task_name: str, *, update_prompt_box: bool = True) -> None:
        if not hasattr(self, "task_prompt_history_combo"):
            return
        prompts = self._prompts_for_task(task_name)
        self._loading_task_ui = True
        prev = self.task_prompt_history_combo.blockSignals(True)
        self.task_prompt_history_combo.clear()
        for idx, prompt in enumerate(prompts):
            label = prompt if len(prompt) <= 120 else (prompt[:117] + "...")
            self.task_prompt_history_combo.addItem(f"{idx + 1}. {label}", prompt)
        self.task_prompt_history_combo.setCurrentIndex(len(prompts) - 1)
        self.task_prompt_history_combo.blockSignals(prev)
        self._loading_task_ui = False

        latest = str(prompts[-1])
        if update_prompt_box and hasattr(self, "prompt_input"):
            self._loading_task_ui = True
            self._set_prompt_input_text(latest)
            self._loading_task_ui = False
        self.record_prompt = latest

    def _on_task_changed(self, _index: int) -> None:
        if not hasattr(self, "task_combo"):
            return
        task = str(self.task_combo.currentText() or TASK_DEFAULT_NAME)
        self.selected_task_name = task
        self._refresh_prompt_history_for_task(task, update_prompt_box=True)

    def _on_saved_prompt_selected(self, _index: int) -> None:
        if self._loading_task_ui:
            return
        if not hasattr(self, "task_prompt_history_combo"):
            return
        prompt = str(self.task_prompt_history_combo.currentData() or "").strip()
        if not prompt:
            return
        self._loading_task_ui = True
        if hasattr(self, "prompt_input"):
            self._set_prompt_input_text(prompt)
        self._loading_task_ui = False
        self.record_prompt = prompt

    def _on_add_task_clicked(self) -> None:
        task = str(getattr(self, "selected_task_name", TASK_DEFAULT_NAME))
        prompts = self._prompts_for_task(task)
        latest = str(prompts[-1])
        self._loading_task_ui = True
        self._set_prompt_input_text(latest)
        self._loading_task_ui = False
        self.record_prompt = latest
        self._set_status(
            f"Loaded latest prompt for task '{task}'",
            "QLabel { color: #1e88e5; font-size: 10px; }",
        )

    def _on_save_prompt_for_task_clicked(self) -> None:
        if not hasattr(self, "task_combo"):
            return
        task = str(self.task_combo.currentText() or TASK_DEFAULT_NAME)
        prompt = str(self._get_prompt_input_text() or "").strip()
        if not prompt:
            self._set_status("Prompt is empty; nothing saved.", "QLabel { color: orange; font-size: 10px; }")
            return

        prompts = self._prompts_for_task(task)
        prompts = [p for p in prompts if p != prompt]
        prompts.append(prompt)  # latest prompt at end
        self._task_prompt_library[str(task)] = prompts
        self._save_task_prompt_library()
        self.selected_task_name = task
        self._refresh_prompt_history_for_task(task, update_prompt_box=True)
        self._set_status(
            f"Saved prompt under task '{task}'",
            "QLabel { color: #1e88e5; font-size: 10px; }",
        )

    def _on_embodiment_changed(self, _index: int) -> None:
        if not hasattr(self, "embodiment_combo"):
            return
        self._apply_embodiment_ui_state()
        if self._has_selected_embodiment():
            try:
                self._update_selected_embodiment_config_from_urdf()
                self._start_robot_runtime_if_needed()
            except Exception as exc:
                self._set_status(
                    f"Embodiment initialization failed: {exc}",
                    "QLabel { color: red; font-size: 10px; }",
                )
        self.update_camera_displays()

    def _on_primary_view_changed(self, _index: int) -> None:
        if not self._has_selected_embodiment():
            self._clear_primary_camera_display("No view selected")
            return
        self.update_camera_displays()

    def on_prompt_changed(self, text=None):
        if text is None:
            text = self._get_prompt_input_text()
        if self._loading_task_ui:
            self.record_prompt = str(text)
            return
        self.record_prompt = text

    def on_record_folder_changed(self, text):
        self.dataset_root = text.strip()

    def on_record_rgb_format_changed(self, text):
        fmt = str(text or DEMO_RECORD_RGB_DEFAULT_FORMAT).strip().lower()
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in {"jpg", "png"}:
            fmt = str(DEMO_RECORD_RGB_DEFAULT_FORMAT).lower()
        self.record_rgb_format = fmt

    def browse_record_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select LeRobot Dataset Folder",
            self.dataset_root or str(Path.cwd()),
        )
        if folder:
            self.dataset_root = folder
            self.record_folder_input.setText(folder)

    def _record_dataset_subdir(self, kind: str) -> str:
        base = Path(self.dataset_root or str(Path.cwd())).expanduser().resolve()
        if str(kind) == "type2":
            return str((base / "type2").resolve())
        return str(base)

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    @staticmethod
    def _safe_rmtree(path: Path) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _prune_empty_parent(self, start_path: Path, *, stop_at: Path) -> None:
        cur = start_path
        stop_root = Path(stop_at).resolve()
        while True:
            try:
                cur_resolved = cur.resolve()
            except Exception:
                cur_resolved = cur
            if cur_resolved == stop_root:
                break
            try:
                cur.rmdir()
            except OSError:
                break
            except Exception:
                break
            parent = cur.parent
            if parent == cur:
                break
            cur = parent

    def _delete_episode_artifacts(self, dataset_root: str, episode_index: int) -> None:
        root = Path(dataset_root).expanduser().resolve()
        ep = int(episode_index)
        chunk = f"chunk-{ep // 1000:03d}"
        episode_name = f"episode_{ep:06d}"
        data_file = root / "data" / chunk / f"{episode_name}.jsonl"
        prompt_file = root / "prompts" / chunk / f"{episode_name}.txt"
        self._safe_unlink(data_file)
        self._safe_unlink(prompt_file)
        self._safe_rmtree(root / "images" / chunk / "observation.images.head_rgb" / episode_name)
        self._safe_rmtree(root / "images" / chunk / "observation.images.wrist_rgb" / episode_name)
        self._safe_rmtree(root / "depth" / chunk / "observation.depth.head" / episode_name)
        self._safe_rmtree(root / "depth" / chunk / "observation.depth.wrist" / episode_name)
        self._prune_empty_parent(data_file.parent, stop_at=root)
        self._prune_empty_parent(prompt_file.parent, stop_at=root)
        self._prune_empty_parent((root / "images" / chunk / "observation.images.head_rgb"), stop_at=root)
        self._prune_empty_parent((root / "images" / chunk / "observation.images.wrist_rgb"), stop_at=root)
        self._prune_empty_parent((root / "depth" / chunk / "observation.depth.head"), stop_at=root)
        self._prune_empty_parent((root / "depth" / chunk / "observation.depth.wrist"), stop_at=root)

    def _rebuild_dataset_meta(self, dataset_root: str) -> None:
        root = Path(dataset_root).expanduser().resolve()
        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = meta_dir / "episodes.jsonl"
        tasks_path = meta_dir / "tasks.jsonl"
        info_path = meta_dir / "info.json"

        recs: list[dict[str, Any]] = []
        if episodes_path.exists():
            try:
                with open(episodes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(rec, dict):
                            recs.append(rec)
            except Exception:
                recs = []

        kept: list[dict[str, Any]] = []
        for rec in recs:
            data_rel = rec.get("data_path")
            if not isinstance(data_rel, str) or len(data_rel.strip()) == 0:
                continue
            data_abs = root / data_rel
            if data_abs.exists():
                kept.append(rec)

        kept.sort(key=lambda x: int(x.get("episode_index", -1)))

        with open(episodes_path, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(_to_jsonable(rec), ensure_ascii=True) + "\n")

        with open(tasks_path, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(
                    json.dumps(
                        {
                            "episode_index": int(rec.get("episode_index", -1)),
                            "task": str(rec.get("task", "")),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

        info: dict[str, Any] = {}
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
        total_episodes = int(len(kept))
        total_frames = int(sum(int(rec.get("length", 0)) for rec in kept))
        info.update(
            {
                "dataset_type": "lerobot_style",
                "codebase_version": str(info.get("codebase_version", "v2.1_style")),
                "robot_type": str(info.get("robot_type", "stretch3")),
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "fps": float(info.get("fps", DEMO_RECORD_FPS)),
                "rgb_storage": str(info.get("rgb_storage", f"{self.record_rgb_format}_frames")),
                "depth_preview_storage": "disabled",
                "tabular_format": "jsonl",
                "features": info.get(
                    "features",
                    [
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
                ),
            }
        )
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def _is_auto_loop_record_toggle_enabled(self) -> bool:
        return bool(getattr(self, "_loop_record_armed", False))

    def _reset_auto_loop_record_session_state(self) -> None:
        self._loop_record_session_active = False
        self._loop_record_expected_rows = 0
        self._loop_record_prompt = ""
        self._loop_record_type_roots = {}
        self._loop_record_current = None
        self._loop_record_entries = []
        self._loop_record_stop_requested = False
        self._non_auto_record_finalize_watch_armed = False

    def _stop_auto_loop_record_segment(self, *, save: bool) -> dict[str, Any] | None:
        cur = self._loop_record_current
        if not isinstance(cur, dict):
            return None
        rec = cur.get("recorder")
        if not isinstance(rec, LeRobotStyleRecorder):
            self._loop_record_current = None
            return None
        summary = rec.stop(discard=not bool(save))
        entry: dict[str, Any] | None = None
        if bool(save) and isinstance(summary, dict) and not bool(summary.get("discarded", False)):
            entry = {
                "kind": str(cur.get("kind", "")),
                "round_idx": int(cur.get("round_idx", 0)),
                "dataset_root": str(summary.get("dataset_root", "")),
                "episode_index": int(summary.get("episode_index", -1)),
                "num_frames": int(summary.get("num_frames", 0)),
                "dropped_frames": int(summary.get("dropped_frames", 0)),
            }
            self._loop_record_entries.append(entry)
        self._loop_record_current = None
        return entry

    def _start_auto_loop_record_segment(self, *, kind: str, round_idx: int) -> bool:
        if not self._loop_record_session_active:
            return True
        target_root = self._loop_record_type_roots.get(str(kind))
        if not isinstance(target_root, str) or len(target_root.strip()) == 0:
            self._set_status(
                f"Auto-loop recording start failed: unknown kind '{kind}'",
                "QLabel { color: red; font-size: 10px; }",
            )
            return False
        if self._loop_record_current is not None:
            self._stop_auto_loop_record_segment(save=True)
        try:
            rec = LeRobotStyleRecorder(robot_type="stretch3", target_fps=DEMO_RECORD_FPS)
            rec.start(
                target_root,
                self._loop_record_prompt,
                rgb_image_format=self.record_rgb_format,
                rgb_jpeg_quality=self.record_rgb_jpeg_quality,
            )
            self._loop_record_current = {
                "kind": str(kind),
                "round_idx": int(round_idx),
                "recorder": rec,
            }
            return True
        except Exception as exc:
            self._set_status(
                f"Auto-loop recording start failed ({kind} r{round_idx}): {exc}",
                "QLabel { color: red; font-size: 10px; }",
            )
            self._loop_record_current = None
            return False

    def _start_auto_loop_record_session(self, repeats_after_first: int) -> bool:
        if not self._is_auto_loop_record_toggle_enabled():
            self._reset_auto_loop_record_session_state()
            return True
        if self._loop_record_session_active:
            self._stop_auto_loop_record_segment(save=False)
            self._reset_auto_loop_record_session_state()

        base_root = Path(self.dataset_root or str(Path.cwd())).expanduser().resolve()
        type1_root = str(base_root.resolve())
        type2_root = str((base_root / "type2").resolve())
        try:
            Path(type1_root).mkdir(parents=True, exist_ok=True)
            Path(type2_root).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._set_status(
                f"Auto-loop recording setup failed: {exc}",
                "QLabel { color: red; font-size: 10px; }",
            )
            return False

        prompt = (self.record_prompt or "").strip()
        if not prompt:
            prompt = "unspecified_task"
            self._set_prompt_input_text(prompt)
        self._loop_record_session_active = True
        self._loop_record_expected_rows = int(max(1, int(repeats_after_first) + 1))
        self._loop_record_prompt = str(prompt)
        self._loop_record_type_roots = {"type1": type1_root, "type2": type2_root}
        self._loop_record_entries = []
        self._loop_record_current = None
        self._loop_record_stop_requested = False
        if not self._start_auto_loop_record_segment(kind="type1", round_idx=0):
            self._reset_auto_loop_record_session_state()
            return False
        self._set_status(
            f"Auto-loop recording ON: type1->{base_root}, type2->{Path(type2_root)}",
            "QLabel { color: #1e88e5; font-size: 10px; }",
        )
        return True

    def _start_non_autoloop_record_session(self) -> bool:
        """Record exactly one queued run (human-corrected first trial) and review it."""
        if not self._is_auto_loop_record_toggle_enabled():
            return True
        if self._loop_record_session_active:
            self._stop_auto_loop_record_segment(save=False)
            self._reset_auto_loop_record_session_state()
        # One row/session only: first trial recording.
        ok = bool(self._start_auto_loop_record_session(0))
        if ok:
            self._set_status(
                "First-trial recording ON for this queued run.",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
        return ok

    def _finalize_non_autoloop_record_session(self) -> None:
        """Finalize one-shot queued-run recording and prompt review table."""
        if not self._loop_record_session_active:
            return
        if bool(self._auto_start_after_return) or bool(self._auto_loop_running) or bool(self._auto_first_trial_pending):
            return
        self._stop_auto_loop_record_segment(save=True)
        if len(self._loop_record_entries) > 0:
            self.ui_loop_record_review_signal.emit()
        else:
            self._update_auto_loop_record_button_ui()
            self._reset_auto_loop_record_session_state()

    def _start_non_autoloop_record_finalize_watcher(self) -> None:
        if not self._loop_record_session_active:
            return
        self._non_auto_record_finalize_watch_armed = True

        def _watch() -> None:
            while bool(getattr(self, "_non_auto_record_finalize_watch_armed", False)):
                if bool(self._auto_loop_running) or bool(self._auto_start_after_return):
                    self._non_auto_record_finalize_watch_armed = False
                    return
                with self._action_lock:
                    st = self._action_state
                done = bool(
                    st == "idle"
                    and (not bool(self._run_all_queued_goals))
                    and (not bool(self._deferred_next_goal_start))
                )
                if done:
                    self._non_auto_record_finalize_watch_armed = False
                    QTimer.singleShot(0, self._finalize_non_autoloop_record_session)
                    return
                time.sleep(0.10)

        from threading import Thread
        Thread(target=_watch, daemon=True).start()

    def _abort_auto_loop_record_session(self, *, keep_existing_files: bool = False) -> None:
        self._stop_auto_loop_record_segment(save=False)
        if not bool(keep_existing_files):
            for entry in list(self._loop_record_entries):
                try:
                    self._delete_episode_artifacts(
                        str(entry.get("dataset_root", "")),
                        int(entry.get("episode_index", -1)),
                    )
                except Exception:
                    pass
        for p in self._loop_record_type_roots.values():
            try:
                self._rebuild_dataset_meta(p)
            except Exception:
                pass
        self._reset_auto_loop_record_session_state()

    def _show_auto_loop_record_review_dialog(self) -> None:
        if not self._loop_record_session_active:
            return
        entries = list(self._loop_record_entries)
        if len(entries) == 0:
            self._reset_auto_loop_record_session_state()
            return

        max_row_seen = 0
        for entry in entries:
            try:
                max_row_seen = max(max_row_seen, int(entry.get("round_idx", 0)))
            except Exception:
                continue
        rows = int(max(1, max_row_seen + 1))
        by_slot: dict[tuple[int, str], dict[str, Any]] = {}
        for entry in entries:
            try:
                k = str(entry.get("kind", ""))
                r = int(entry.get("round_idx", 0))
            except Exception:
                continue
            by_slot[(r, k)] = entry

        dialog = QDialog(self)
        dialog.setWindowTitle("Auto Loop Recording Review")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)
        info = QLabel(
            "Select which rounds to keep.\n"
            "Type 1: pick/place forward phase.\n"
            "Type 2: bring-back + return phase."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        table = QTableWidget(rows, 3)
        table.setHorizontalHeaderLabels(["Round", "Type 1", "Type 2"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 100)
        layout.addWidget(table)

        checkbox_cells: dict[tuple[int, str], tuple[QCheckBox, dict[str, Any] | None]] = {}
        for row_idx in range(rows):
            table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx)))
            for kind, col_idx in (("type1", 1), ("type2", 2)):
                entry = by_slot.get((row_idx, kind))
                cell = QWidget()
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cb = QCheckBox()
                cb.setChecked(entry is not None)
                cb.setEnabled(entry is not None)
                cell_layout.addWidget(cb)
                table.setCellWidget(row_idx, col_idx, cell)
                checkbox_cells[(row_idx, kind)] = (cb, entry)

        controls = QHBoxLayout()
        t1_all = QPushButton("Check All Type 1")
        t1_none = QPushButton("Uncheck All Type 1")
        t2_all = QPushButton("Check All Type 2")
        t2_none = QPushButton("Uncheck All Type 2")
        controls.addWidget(t1_all)
        controls.addWidget(t1_none)
        controls.addWidget(t2_all)
        controls.addWidget(t2_none)
        layout.addLayout(controls)

        save_btn = QPushButton("Save")
        save_btn.setMinimumHeight(34)
        layout.addWidget(save_btn)

        def _set_column(kind: str, checked: bool) -> None:
            for row_idx in range(rows):
                cb, entry = checkbox_cells[(row_idx, kind)]
                if entry is not None:
                    cb.setChecked(bool(checked))

        t1_all.clicked.connect(lambda: _set_column("type1", True))
        t1_none.clicked.connect(lambda: _set_column("type1", False))
        t2_all.clicked.connect(lambda: _set_column("type2", True))
        t2_none.clicked.connect(lambda: _set_column("type2", False))

        finalized = {"done": False}

        def _finalize() -> None:
            if bool(finalized["done"]):
                return
            finalized["done"] = True
            kept = 0
            dropped = 0
            for row_idx in range(rows):
                for kind in ("type1", "type2"):
                    cb, entry = checkbox_cells[(row_idx, kind)]
                    if entry is None:
                        continue
                    if cb.isChecked():
                        kept += 1
                    else:
                        dropped += 1
                        try:
                            self._delete_episode_artifacts(
                                str(entry.get("dataset_root", "")),
                                int(entry.get("episode_index", -1)),
                            )
                        except Exception:
                            pass
            for p in self._loop_record_type_roots.values():
                try:
                    self._rebuild_dataset_meta(p)
                except Exception:
                    pass
            self._set_status(
                f"Auto-loop recording saved: kept={kept}, removed={dropped}",
                "QLabel { color: green; font-size: 10px; }",
            )
            self._update_auto_loop_record_button_ui()
            self._reset_auto_loop_record_session_state()
            dialog.accept()

        save_btn.clicked.connect(_finalize)
        dialog.rejected.connect(_finalize)
        dialog.exec()

    def _build_record_sample(self):
        sensors = self.ros_node.get_sensor_snapshot()
        aligned: dict[str, Any] | None = None
        if hasattr(self.ros_node, "get_aligned_record_components"):
            try:
                aligned = self.ros_node.get_aligned_record_components()
            except Exception:
                aligned = None

        if isinstance(aligned, dict):
            actual_qpos = list(aligned.get("actual_qpos10") or [])
            command_qpos = list(aligned.get("command_qpos10") or [])
            measured_pose = list(aligned.get("measured_pose_xytheta") or [0.0, 0.0, 0.0])
            command_pose = list(aligned.get("command_pose_xytheta") or measured_pose)
            sample_ts = float(aligned.get("timestamp", time.time()))
            head_rgb = aligned.get("head_rgb")
            wrist_rgb = aligned.get("wrist_rgb")
            head_depth = aligned.get("head_depth")
            wrist_depth = aligned.get("wrist_depth")
            sensors["observation.sync.reference_stamp_ns"] = aligned.get("reference_stamp_ns")
            sensors["observation.sync.aligned_joint_stamp_ns"] = aligned.get("aligned_joint_stamp_ns")
            sensors["observation.sync.aligned_odom_stamp_ns"] = aligned.get("aligned_odom_stamp_ns")
            sensors["observation.sync.topic_stamp_ns"] = aligned.get("stamp_ns_map", {})
            cmd_event = aligned.get("command_event")
            if isinstance(cmd_event, dict):
                sensors["action_command.sent_wall_time_ns"] = cmd_event.get("wall_time_ns")
                sensors["action_command.sent_ros_time_ns_est"] = cmd_event.get("ros_time_ns_est")
                sensors["action_command.source"] = cmd_event.get("reason")
            sensors["action_command.manip_base_x"] = aligned.get("command_manip_base_x")
        else:
            actual_qpos = self.ros_node.get_actual_qpos()
            command_qpos = self.ros_node.get_published_qpos()
            measured_pose = self.ros_node.get_measured_base_pose_xytheta() or [0.0, 0.0, 0.0]
            command_pose = self.ros_node.get_command_base_pose_xytheta() or list(measured_pose)
            sample_ts = time.time()
            head_rgb = self.head_rgb if self.head_rgb is not None else None
            wrist_rgb = self.wrist_rgb if self.wrist_rgb is not None else None
            head_depth = self.depth_image if self.depth_image is not None else None
            wrist_depth = self.wrist_depth if self.wrist_depth is not None else None

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
            "timestamp": sample_ts,
            "head_rgb": head_rgb,
            "wrist_rgb": wrist_rgb,
            "head_depth": head_depth,
            "wrist_depth": wrist_depth,
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
            self._set_prompt_input_text(prompt)
        try:
            self.demo_recorder.start(
                self._record_dataset_subdir("type1"),
                prompt,
                rgb_image_format=self.record_rgb_format,
                rgb_jpeg_quality=self.record_rgb_jpeg_quality,
            )
            self.is_recording_demo = True
            self.record_toggle_button.setText("Stop Recording")
            self.status_label.setText(
                f"Recording demo: {prompt} (fps={self.demo_recorder.target_fps:.1f}, rgb={self.record_rgb_format})"
            )
            self.status_label.setStyleSheet("QLabel { color: #1e88e5; font-size: 10px; }")
        except Exception as e:
            self.status_label.setText(f"Record start failed: {e}")
            self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")

    def stop_demo_recording(self):
        if not self.is_recording_demo:
            return
        try:
            # Freeze recording at stop-click time (do not capture frames while popup is open).
            self.is_recording_demo = False
            self.record_toggle_button.setText("Record")
            choice = QMessageBox.question(
                self,
                "Stop Recording",
                "Save this episode?\n\nYes: save episode\nNo: discard episode and delete recorded files",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            discard = choice == QMessageBox.StandardButton.No
            summary = self.demo_recorder.stop(discard=discard)
            if discard:
                ep = summary.get("episode_index", "?") if isinstance(summary, dict) else "?"
                self.status_label.setText(
                    f"Discarded demo ep {ep} (deleted recorded files)"
                )
                self.status_label.setStyleSheet("QLabel { color: #ff9800; font-size: 10px; }")
            else:
                sync_ok = self._sync_base_cmd_from_observation(update_status=False)
                if summary is None:
                    self.status_label.setText(
                        f"Recording stopped (base cmd sync: {'ok' if sync_ok else 'failed'})"
                    )
                    self.status_label.setStyleSheet("QLabel { color: green; font-size: 10px; }")
                else:
                    self.status_label.setText(
                        f"Saved demo ep {summary['episode_index']} "
                        f"({summary['num_frames']} frames, dropped={summary.get('dropped_frames', 0)}, "
                        f"base_sync={'ok' if sync_ok else 'failed'})"
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
        self._update_joint_state_table()
        needs_record_sample = bool(self.is_recording_demo) or (
            bool(self._loop_record_session_active) and isinstance(self._loop_record_current, dict)
        )
        sample = None
        if needs_record_sample:
            try:
                sample = self._build_record_sample()
            except Exception as e:
                print(f"Recording sample build error: {e}")
                sample = None
        if self.is_recording_demo and isinstance(sample, dict):
            try:
                self.demo_recorder.record_step(sample)
            except Exception as e:
                print(f"Recording step error: {e}")
                self.stop_demo_recording()
                self.status_label.setText(f"Recording stopped due error: {e}")
                self.status_label.setStyleSheet("QLabel { color: red; font-size: 10px; }")
        if bool(self._loop_record_session_active) and isinstance(self._loop_record_current, dict) and isinstance(sample, dict):
            try:
                rec = self._loop_record_current.get("recorder")
                if isinstance(rec, LeRobotStyleRecorder):
                    rec.record_step(sample)
            except Exception as e:
                print(f"Auto-loop recording step error: {e}")
                self._abort_auto_loop_record_session(keep_existing_files=False)
                self._set_status(
                    f"Auto-loop recording disabled due error: {e}",
                    "QLabel { color: red; font-size: 10px; }",
                )
        self._update_auto_loop_progress_ui()
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
        if not self._has_selected_embodiment():
            self._clear_primary_camera_display("No view selected")
            self._clear_secondary_camera_displays()
            return

        head_raw = self.head_rgb.copy() if self.head_rgb is not None else None
        head_display = None
        if head_raw is not None:
            head_display = head_raw.copy()
            # Apply mask overlay if available and using head camera for segmentation
            if self.mask_overlay is not None and self.use_head_for_segmentation:
                head_display = cv2.addWeighted(head_display, 0.7, self.mask_overlay, 0.3, 0)

            # Draw 3D grasp axes and bounding box for debugging
            if self._grasp_debug_info is not None and 'axis_pixels' in self._grasp_debug_info:
                ap = self._grasp_debug_info['axis_pixels']
                if 'corners' in ap and len(ap['corners']) >= 3:
                    corners = ap['corners']
                    for i in range(len(corners)):
                        cv2.line(head_display, corners[i], corners[(i + 1) % len(corners)], (255, 255, 0), 1, cv2.LINE_AA)
                if 'long1' in ap and 'long2' in ap:
                    cv2.line(head_display, ap['long1'], ap['long2'], (0, 255, 0), 2, cv2.LINE_AA)
                if 'narrow1' in ap and 'narrow2' in ap:
                    cv2.line(head_display, ap['narrow1'], ap['narrow2'], (0, 0, 255), 2, cv2.LINE_AA)
                if 'center' in ap:
                    cv2.circle(head_display, ap['center'], 5, (0, 255, 255), -1)
                if 'long1' in ap:
                    cv2.putText(head_display, "long", ap['long1'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if 'narrow1' in ap:
                    cv2.putText(head_display, "grasp", ap['narrow1'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw manual region rectangles and selected grip-tip points (v8).
            if self.manual_grasp_regions:
                for region in self.manual_grasp_regions:
                    rid = int(region.get("id", -1))
                    x0, y0, x1, y1 = self._manual_region_rect_norm(region)
                    is_sel = (self._manual_selected_region_id is not None and rid == int(self._manual_selected_region_id))
                    rect_color = (0, 255, 255) if is_sel else (255, 0, 255)
                    rect_thick = 2 if is_sel else 1
                    cv2.rectangle(head_display, (x0, y0), (x1, y1), rect_color, rect_thick)
                    cv2.putText(head_display, f"R{rid}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rect_color, 1, cv2.LINE_AA)
                    points = list(region.get("points", []))
                    for i, p in enumerate(points, start=1):
                        px, py = int(p[0]), int(p[1])
                        p_color = (0, 255, 0) if i == 1 else (255, 200, 0)
                        cv2.circle(head_display, (px, py), 4, p_color, -1)
                        cv2.putText(head_display, f"P{i}", (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, p_color, 1, cv2.LINE_AA)

            if (
                self._manual_draw_mode == "draw_rect"
                and self._manual_dragging_rect
                and self._manual_rect_start_px is not None
                and self._manual_rect_live_px is not None
            ):
                x0, y0 = self._manual_rect_start_px
                x1, y1 = self._manual_rect_live_px
                cv2.rectangle(head_display, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
                cv2.putText(
                    head_display,
                    "drag region",
                    (int(min(x0, x1)), int(max(12, min(y0, y1) - 4))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if (
                bool(self._drag_draw_mode)
                and bool(self._drag_dragging)
                and self._drag_start_px is not None
                and self._drag_live_px is not None
            ):
                sx, sy = self._drag_start_px
                ex, ey = self._drag_live_px
                if str(getattr(self, "_drag_mode_kind", "line")) == "curve":
                    draw_pts = list(getattr(self, "_drag_path_points", []))
                    draw_pts.append((int(ex), int(ey)))
                    smooth_pts = self._fit_smooth_curve_pixels(draw_pts, samples=int(DRAG_CURVE_FIT_SAMPLES))
                    if len(smooth_pts) >= 2:
                        arr = np.array(smooth_pts, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(head_display, [arr], False, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(head_display, (int(sx), int(sy)), 4, (0, 255, 255), -1)
                    cv2.circle(head_display, (int(ex), int(ey)), 4, (255, 200, 0), -1)
                    cv2.putText(
                        head_display,
                        "curve",
                        (int(min(sx, ex)), int(max(12, min(sy, ey) - 4))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.arrowedLine(head_display, (int(sx), int(sy)), (int(ex), int(ey)), (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.12)
                    cv2.circle(head_display, (int(sx), int(sy)), 4, (0, 255, 255), -1)
                    cv2.circle(head_display, (int(ex), int(ey)), 4, (255, 200, 0), -1)
                    cv2.putText(
                        head_display,
                        "drag",
                        (int(min(sx, ex)), int(max(12, min(sy, ey) - 4))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        wrist_display = self.wrist_rgb.copy() if self.wrist_rgb is not None else None

        # Primary camera selection drives top display.
        primary_key = None
        if hasattr(self, "primary_camera_combo"):
            try:
                key = self.primary_camera_combo.currentData()
                if isinstance(key, str) and key in {"head", "wrist"}:
                    primary_key = key
            except Exception:
                primary_key = None
        primary_img = None
        if str(primary_key) == "head":
            primary_img = head_display
        elif str(primary_key) == "wrist":
            primary_img = wrist_display

        if primary_img is not None:
            primary_vis = primary_img.copy()
            if str(primary_key) == "head":
                crop_frac = float(np.clip(float(HEAD_DISPLAY_CROP_BOTTOM_FRAC), 0.0, 0.95))
                if crop_frac > 1e-6:
                    h0 = int(primary_vis.shape[0])
                    crop_h = max(1, int(round(float(h0) * (1.0 - crop_frac))))
                    primary_vis = primary_vis[:crop_h, :]

            head_pixmap = self.numpy_to_pixmap(primary_vis)
            if not head_pixmap.isNull():
                self._head_max_w = max(int(self._head_max_w), int(head_pixmap.width()))
                self._head_max_h = max(int(self._head_max_h), int(head_pixmap.height()))
                head_w = max(1, int(round(float(self._head_max_w) * float(UI_CAMERA_DISPLAY_SCALE))))
                head_h = max(1, int(round(float(self._head_max_h) * float(UI_CAMERA_DISPLAY_SCALE))))
                head_pixmap = head_pixmap.scaled(
                    head_w,
                    head_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.head_display.setFixedSize(int(head_pixmap.width()), int(head_pixmap.height()))
                self.head_container.setFixedSize(self.head_display.size())
                self.head_display.setText("")
                self.head_display.setPixmap(head_pixmap)
        else:
            if self._has_selected_primary_view():
                self._clear_primary_camera_display("No RGB feed")
            else:
                self._clear_primary_camera_display("Select Primary View")

        # Secondary RGB columns.
        if hasattr(self, "secondary_rgb_displays") and hasattr(self, "secondary_rgb_combos"):
            for combo, disp in zip(self.secondary_rgb_combos, self.secondary_rgb_displays):
                source_key = combo.currentData() if combo is not None else None
                source_key = str(source_key) if isinstance(source_key, str) else "head"
                # Bottom views should remain clean RGB (no segmentation/annotation overlay).
                src_img = head_raw if source_key == "head" else wrist_display
                if src_img is None:
                    disp.setText("No RGB feed")
                    disp.setPixmap(QPixmap())
                    continue
                show_img = src_img.copy()
                if source_key == "head":
                    crop_frac = float(np.clip(float(HEAD_DISPLAY_CROP_BOTTOM_FRAC), 0.0, 0.95))
                    if crop_frac > 1e-6:
                        h0 = int(show_img.shape[0])
                        crop_h = max(1, int(round(float(h0) * (1.0 - crop_frac))))
                        show_img = show_img[:crop_h, :]
                pix = self.numpy_to_pixmap(show_img)
                if pix.isNull():
                    disp.setText("No RGB feed")
                    disp.setPixmap(QPixmap())
                    continue
                pix = pix.scaled(
                    max(1, disp.width()),
                    max(1, disp.height()),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                disp.setPixmap(pix)

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

        self.status_label.setText("Segmentation updated")
        self.status_label.setStyleSheet("QLabel { color: green; }")
        self.segment_button.setEnabled(True)
        self.segment_button.setText("Segment Objects")

        # Update display
        self.update_camera_displays()

    def on_error(self, error_msg):
        """Handle error messages"""
        print(f"Error: {error_msg}")
        self.status_label.setText(error_msg)
        self.status_label.setStyleSheet("QLabel { color: red; }")
        if not self.segment_button.isEnabled():
            self.segment_button.setEnabled(True)
            self.segment_button.setText("Segment Objects")

    def on_object_selected(self, item):
        """Handle object selection from list"""
        idx = item.data(Qt.ItemDataRole.UserRole)
        self.selected_segment = self.segments[idx]
        self.center_button.setEnabled(True)
        self.reach_button.setEnabled(True)
        self.grasp_button.setEnabled(True)
        print(f"Selected object {idx}: center={self.selected_segment['center']}, depth={self.selected_segment['depth']:.3f}m")

    def _manual_region_by_id(self, region_id: int | None):
        if region_id is None:
            return None
        for region in self.manual_grasp_regions:
            if int(region.get("id", -1)) == int(region_id):
                return region
        return None

    @staticmethod
    def _manual_region_rect_norm(region: dict[str, Any]):
        x0, y0, x1, y1 = [int(v) for v in region.get("rect", (0, 0, 0, 0))]
        xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
        ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
        return xa, ya, xb, yb

    def _manual_refresh_region_tree(self):
        if not hasattr(self, "manual_region_tree"):
            return
        tree = self.manual_region_tree
        tree.blockSignals(True)
        tree.clear()
        select_item = None
        for region in self.manual_grasp_regions:
            rid = int(region["id"])
            x0, y0, x1, y1 = self._manual_region_rect_norm(region)
            root = QTreeWidgetItem([f"R{rid}: ({x0},{y0}) -> ({x1},{y1})"])
            root.setData(0, Qt.ItemDataRole.UserRole, rid)
            points = list(region.get("points", []))
            for i, p in enumerate(points, start=1):
                child = QTreeWidgetItem([f"P{i}: ({int(p[0])},{int(p[1])})"])
                child.setData(0, Qt.ItemDataRole.UserRole, rid)
                root.addChild(child)
            comp = region.get("computed")
            if isinstance(comp, dict):
                yaw_deg = np.degrees(float(comp.get("grasp_yaw", 0.0)))
                grip = float(comp.get("gripper_width", 0.0))
                c = QTreeWidgetItem([f"grasp: yaw={yaw_deg:.1f}deg, width={grip:.3f}"])
                c.setData(0, Qt.ItemDataRole.UserRole, rid)
                root.addChild(c)
            tree.addTopLevelItem(root)
            root.setExpanded(True)
            if self._manual_selected_region_id is not None and rid == int(self._manual_selected_region_id):
                select_item = root
        tree.blockSignals(False)
        if select_item is not None:
            tree.setCurrentItem(select_item)

    def _on_manual_region_tree_selection_changed(self):
        if not hasattr(self, "manual_region_tree"):
            return
        item = self.manual_region_tree.currentItem()
        if item is None:
            self._manual_selected_region_id = None
            self.update_camera_displays()
            return
        rid = item.data(0, Qt.ItemDataRole.UserRole)
        if rid is None:
            parent = item.parent()
            rid = parent.data(0, Qt.ItemDataRole.UserRole) if parent is not None else None
        self._manual_selected_region_id = int(rid) if rid is not None else None
        self.update_camera_displays()

    def _start_manual_region_draw(self):
        if self.head_rgb is None:
            self._set_status("Draw Region: no head image", "QLabel { color: red; }")
            return
        self._manual_draw_mode = "draw_rect"
        self._manual_dragging_rect = False
        self._manual_rect_start_px = None
        self._manual_rect_live_px = None
        self._set_status("Draw Region: drag left mouse on image to define rectangle", "QLabel { color: #1e88e5; font-size: 10px; }")

    def _start_drag_operation_draw(self):
        if self.head_rgb is None or self.depth_image is None:
            self._set_status("Drag: missing head RGB/depth", "QLabel { color: red; }")
            return
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Drag: wait until current action is idle", "QLabel { color: orange; }")
                return
        # Cancel manual-region drawing mode while drag mode is active.
        self._manual_draw_mode = None
        self._manual_dragging_rect = False
        self._manual_rect_start_px = None
        self._manual_rect_live_px = None

        self._drag_draw_mode = True
        self._drag_mode_kind = "line"
        self._drag_dragging = False
        self._drag_start_px = None
        self._drag_live_px = None
        self._drag_path_points = []
        self._set_status("Drag mode: hold left mouse and drag start->end on head view", "QLabel { color: #1e88e5; font-size: 10px; }")

    def _start_curved_trajectory_draw(self):
        if self.head_rgb is None or self.depth_image is None:
            self._set_status("Curve: missing head RGB/depth", "QLabel { color: red; }")
            return
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Curve: wait until current action is idle", "QLabel { color: orange; }")
                return
        self._manual_draw_mode = None
        self._manual_dragging_rect = False
        self._manual_rect_start_px = None
        self._manual_rect_live_px = None

        self._drag_draw_mode = True
        self._drag_mode_kind = "curve"
        self._drag_dragging = False
        self._drag_start_px = None
        self._drag_live_px = None
        self._drag_path_points = []
        self._set_status(
            "Curve mode: hold left mouse and draw curved path on head view",
            "QLabel { color: #1e88e5; font-size: 10px; }",
        )

    def _polyline_length_px(self, points: list[tuple[int, int]]) -> float:
        if not isinstance(points, list) or len(points) < 2:
            return 0.0
        arr = np.array(points, dtype=np.float32)
        seg = np.diff(arr, axis=0)
        if seg.size == 0:
            return 0.0
        return float(np.sum(np.linalg.norm(seg, axis=1)))

    def _fit_smooth_curve_pixels(self, points: list[tuple[int, int]], samples: int = 64) -> list[tuple[int, int]]:
        if not isinstance(points, list) or len(points) < 2:
            return []
        arr = np.array([(float(p[0]), float(p[1])) for p in points], dtype=np.float64)
        if arr.shape[0] < 2:
            return []
        d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(d)))
        total = float(s[-1])
        if total <= 1e-6:
            p0 = arr[0]
            return [(int(round(float(p0[0]))), int(round(float(p0[1]))))]

        t = s / total
        deg = int(max(1, min(3, int(arr.shape[0]) - 1)))
        try:
            cx = np.polyfit(t, arr[:, 0], deg)
            cy = np.polyfit(t, arr[:, 1], deg)
        except Exception:
            return [(int(round(float(x))), int(round(float(y)))) for x, y in arr.tolist()]

        n = int(max(8, samples))
        ts = np.linspace(0.0, 1.0, n)
        xs = np.polyval(cx, ts)
        ys = np.polyval(cy, ts)
        out: list[tuple[int, int]] = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            p = (int(round(float(x))), int(round(float(y))))
            if len(out) == 0 or out[-1] != p:
                out.append(p)
        return out

    def _snapshot_pre_action_state_for_return(self):
        """Capture current state so Return can restore position after drag."""
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

    def _plan_drag_joint_for_point_base(
        self,
        point_base: PointStamped,
        *,
        wrist_yaw: float,
        wrist_pitch: float,
        wrist_roll: float,
        base_pose_for_conversion: list[float] | tuple[float, float, float] | None = None,
    ) -> list[float] | None:
        base_pose = base_pose_for_conversion
        if not (isinstance(base_pose, (list, tuple)) and len(base_pose) >= 3):
            base_pose = self.ros_node.get_measured_base_pose_xytheta()
        if not (isinstance(base_pose, (list, tuple)) and len(base_pose) >= 3):
            base_pose = [0.0, 0.0, 0.0]
        target_base_xyz = (
            float(point_base.point.x),
            float(point_base.point.y),
            float(point_base.point.z),
        )
        target_world_xyz = self._base_point_to_odom_xyz(target_base_xyz, base_pose)
        plan = self.ros_node.plan_open_loop_grasp(
            target_world_xyz,
            pregrasp_distance=0.0,
            lift_distance=0.0,
            wrist_yaw_target=float(wrist_yaw),
            wrist_pitch_target=float(wrist_pitch),
            wrist_roll_target=float(wrist_roll),
            timeout_s=float(DRAG_PLAN_TIMEOUT_S),
        )
        if isinstance(plan, dict):
            err0 = str(plan.get("error", ""))
            if (not plan.get("ok", False)) and ("Timeout waiting for worker response to 'plan_open_loop_grasp'" in err0):
                plan = self.ros_node.plan_open_loop_grasp(
                    target_world_xyz,
                    pregrasp_distance=0.0,
                    lift_distance=0.0,
                    wrist_yaw_target=float(wrist_yaw),
                    wrist_pitch_target=float(wrist_pitch),
                    wrist_roll_target=float(wrist_roll),
                    timeout_s=float(max(2.0 * DRAG_PLAN_TIMEOUT_S, 50.0)),
                )
        if not (isinstance(plan, dict) and plan.get("ok", False)):
            return None
        joint = plan.get("grasp_joint")
        if not (isinstance(joint, list) and len(joint) >= 6):
            return None
        out = [float(v) for v in joint[:6]]
        out[0] = float(np.clip(out[0], float(MANIP_BASE_X_LIMITS[0]), float(MANIP_BASE_X_LIMITS[1])))
        out[1] = float(np.clip(out[1], float(self.ros_node.JOINT_LIMITS[1][0]), float(self.ros_node.JOINT_LIMITS[1][1])))
        out[2] = float(np.clip(out[2], float(self.ros_node.JOINT_LIMITS[0][0]), float(self.ros_node.JOINT_LIMITS[0][1])))
        out[3] = float(np.clip(out[3], float(self.ros_node.JOINT_LIMITS[2][0]), float(self.ros_node.JOINT_LIMITS[2][1])))
        out[4] = float(np.clip(out[4], float(self.ros_node.JOINT_LIMITS[3][0]), float(self.ros_node.JOINT_LIMITS[3][1])))
        out[5] = float(np.clip(out[5], float(self.ros_node.JOINT_LIMITS[4][0]), float(self.ros_node.JOINT_LIMITS[4][1])))
        return out

    def _execute_drag_operation_between_pixels(
        self,
        start_px: tuple[int, int],
        end_px: tuple[int, int],
        *,
        repeat_count: int = 1,
        return_along_path: bool = False,
        force_keep_current_lift: bool = False,
        fallback_path_points_base: list[tuple[float, float, float]] | None = None,
    ) -> bool:
        return self._execute_drag_operation_with_pixel_path(
            [(int(start_px[0]), int(start_px[1])), (int(end_px[0]), int(end_px[1]))],
            path_kind="line",
            repeat_count=int(repeat_count),
            return_along_path=bool(return_along_path),
            force_keep_current_lift=bool(force_keep_current_lift),
            fallback_path_points_base=fallback_path_points_base,
        )

    def _execute_drag_operation_with_pixel_path(
        self,
        path_pixels: list[tuple[int, int]],
        *,
        path_kind: str = "line",
        repeat_count: int = 1,
        return_along_path: bool = False,
        surface_height_offset_m: float = 0.0,
        no_height_adjustment: bool = False,
        keep_current_lift_when_no_adjust: bool = False,
        force_keep_current_lift: bool = False,
        fallback_path_points_base: list[tuple[float, float, float]] | None = None,
    ) -> bool:
        if not self._begin_action("drag"):
            self._set_status("Another action is already running/paused", "QLabel { color: orange; font-size: 10px; }")
            return False
        if not isinstance(path_pixels, list) or len(path_pixels) < 2:
            self._set_action_state("idle")
            self._set_status("Drag path is invalid", "QLabel { color: red; }")
            return False
        try:
            repeat_n = int(repeat_count)
        except Exception:
            repeat_n = 1
        repeat_n = int(max(1, repeat_n))
        return_path = bool(return_along_path and repeat_n > 1)
        z_extra = float(surface_height_offset_m)
        if not math.isfinite(z_extra):
            z_extra = 0.0
        use_no_height_adjust = bool(no_height_adjustment)
        mode_label = "Curve" if str(path_kind) == "curve" else "Drag"
        if return_path:
            self._set_status(
                f"{mode_label}: computing path in 3D (passes={repeat_n}, return-to-start enabled)...",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
        else:
            self._set_status(
                f"{mode_label}: computing path in 3D...",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )

        def run():
            drag_success = False
            try:
                self._snapshot_pre_action_state_for_return()
                self._freeze_streaming_commands_to_current_state()

                waypoint_pixels = list(path_pixels)
                if str(path_kind) == "curve" and len(waypoint_pixels) > int(DRAG_CURVE_EXEC_WAYPOINTS):
                    idx = np.linspace(0, len(waypoint_pixels) - 1, int(DRAG_CURVE_EXEC_WAYPOINTS))
                    waypoint_pixels = [waypoint_pixels[int(round(float(i)))] for i in idx.tolist()]
                    dedup: list[tuple[int, int]] = []
                    for p in waypoint_pixels:
                        if len(dedup) == 0 or dedup[-1] != p:
                            dedup.append(p)
                    waypoint_pixels = dedup

                fallback_points: list[tuple[float, float, float]] | None = None
                if isinstance(fallback_path_points_base, list):
                    parsed: list[tuple[float, float, float]] = []
                    for xyz in fallback_path_points_base:
                        if not (isinstance(xyz, (list, tuple)) and len(xyz) >= 3):
                            continue
                        try:
                            x = float(xyz[0])
                            y = float(xyz[1])
                            z = float(xyz[2])
                        except Exception:
                            continue
                        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                            continue
                        parsed.append((x, y, z))
                    if len(parsed) >= 2:
                        fallback_points = parsed

                path_points: list[PointStamped] = []
                wp_count = len(waypoint_pixels)
                for wp_idx, wp in enumerate(waypoint_pixels):
                    pref_dir = None
                    if wp_count >= 2:
                        if wp_idx == 0:
                            nx = int(waypoint_pixels[1][0]) - int(waypoint_pixels[0][0])
                            ny = int(waypoint_pixels[1][1]) - int(waypoint_pixels[0][1])
                            pref_dir = (float(nx), float(ny))
                        elif wp_idx == (wp_count - 1):
                            nx = int(waypoint_pixels[-1][0]) - int(waypoint_pixels[-2][0])
                            ny = int(waypoint_pixels[-1][1]) - int(waypoint_pixels[-2][1])
                            pref_dir = (float(nx), float(ny))
                        else:
                            nx = int(waypoint_pixels[wp_idx + 1][0]) - int(waypoint_pixels[wp_idx - 1][0])
                            ny = int(waypoint_pixels[wp_idx + 1][1]) - int(waypoint_pixels[wp_idx - 1][1])
                            pref_dir = (float(nx), float(ny))
                    p, _, _px_res, _py_res = self._get_3d_point_nearest_valid_depth(
                        int(wp[0]),
                        int(wp[1]),
                        search_radius_px=int(DRAG_NEAREST_VALID_DEPTH_RADIUS_PX),
                        preferred_dir_px=pref_dir,
                        require_forward=bool(wp_idx > 0),
                    )
                    if p is None and fallback_points is not None:
                        # Use stored fallback path points (captured when goal was queued)
                        # when live depth becomes invalid after grasp occlusions.
                        if wp_count <= 1:
                            map_idx = 0
                        else:
                            map_idx = int(
                                round(
                                    float(wp_idx) * float(len(fallback_points) - 1) / float(wp_count - 1)
                                )
                            )
                        map_idx = int(np.clip(map_idx, 0, len(fallback_points) - 1))
                        p = self._point_from_xyz(fallback_points[map_idx])
                    if p is None:
                        if str(path_kind) == "curve":
                            print(
                                f"[curve drag] skipping waypoint with invalid depth "
                                f"{int(wp_idx) + 1}/{int(wp_count)}"
                            )
                            continue
                        # Start/end depth must exist; otherwise trajectory target is ambiguous.
                        if wp_idx == 0:
                            raise RuntimeError("invalid depth at drag start point")
                        if wp_idx == (wp_count - 1):
                            raise RuntimeError("invalid depth at drag end point")
                        continue
                    p.point.z = float(p.point.z) + float(DRAG_POINT_Z_SAFETY_M)
                    if not use_no_height_adjust:
                        p.point.z = float(p.point.z) + float(z_extra)
                    path_points.append(p)
                if len(path_points) < 2:
                    raise RuntimeError("invalid depth along drag path")
                p_start = path_points[0]
                # Anchor all waypoint base->odom conversion to one fixed base pose
                # captured before any drag motion. Without this, after moving to
                # drag-start pose the remaining points can be reinterpreted with a
                # shifted base frame and flip extension direction.
                drag_ref_base_pose = self.ros_node.get_measured_base_pose_xytheta()
                if not (isinstance(drag_ref_base_pose, (list, tuple)) and len(drag_ref_base_pose) >= 3):
                    drag_ref_base_pose = [0.0, 0.0, 0.0]
                else:
                    drag_ref_base_pose = [
                        float(drag_ref_base_pose[0]),
                        float(drag_ref_base_pose[1]),
                        float(drag_ref_base_pose[2]),
                    ]

                cur_joint = self._current_manip_joint6()
                if not (isinstance(cur_joint, list) and len(cur_joint) >= 6):
                    raise RuntimeError("current joint state unavailable")
                cur_joint = [float(v) for v in cur_joint[:6]]
                wrist_yaw = float(cur_joint[3])
                wrist_pitch = float(cur_joint[4])
                wrist_roll = float(cur_joint[5])
                # Prefer live commanded wrist orientation so manual yaw/pitch/roll
                # edits during grasp carry into waypoint IK for curve/drag.
                q_target_live = self.ros_node.get_target_qpos()
                if isinstance(q_target_live, list) and len(q_target_live) >= 5:
                    try:
                        y_live = float(q_target_live[2])
                        p_live = float(q_target_live[3])
                        r_live = float(q_target_live[4])
                        if math.isfinite(y_live):
                            wrist_yaw = float(
                                np.clip(
                                    float(y_live),
                                    float(self.ros_node.JOINT_LIMITS[2][0]),
                                    float(self.ros_node.JOINT_LIMITS[2][1]),
                                )
                            )
                        if math.isfinite(p_live):
                            wrist_pitch = float(
                                np.clip(
                                    float(p_live),
                                    float(self.ros_node.JOINT_LIMITS[3][0]),
                                    float(self.ros_node.JOINT_LIMITS[3][1]),
                                )
                            )
                        if math.isfinite(r_live):
                            wrist_roll = float(
                                np.clip(
                                    float(r_live),
                                    float(self.ros_node.JOINT_LIMITS[4][0]),
                                    float(self.ros_node.JOINT_LIMITS[4][1]),
                                )
                            )
                    except Exception:
                        pass
                fixed_drag_lift = None
                if bool(force_keep_current_lift):
                    fixed_drag_lift = float(cur_joint[1])
                    fixed_drag_lift = float(
                        np.clip(
                            float(fixed_drag_lift),
                            float(self.ros_node.JOINT_LIMITS[1][0]),
                            float(self.ros_node.JOINT_LIMITS[1][1]),
                        )
                    )
                elif use_no_height_adjust:
                    if bool(keep_current_lift_when_no_adjust):
                        fixed_drag_lift = float(cur_joint[1])
                    else:
                        fixed_drag_lift = float(IK_SAFE_LIFT_M)
                    fixed_drag_lift = float(
                        np.clip(
                            float(fixed_drag_lift),
                            float(self.ros_node.JOINT_LIMITS[1][0]),
                            float(self.ros_node.JOINT_LIMITS[1][1]),
                        )
                    )

                start_joint = self._plan_drag_joint_for_point_base(
                    p_start,
                    wrist_yaw=wrist_yaw,
                    wrist_pitch=wrist_pitch,
                    wrist_roll=wrist_roll,
                    base_pose_for_conversion=drag_ref_base_pose,
                )
                if not (isinstance(start_joint, list) and len(start_joint) >= 6):
                    raise RuntimeError("failed to plan drag start point")
                preserve_curve_contact_start = bool(str(path_kind) == "curve" and bool(force_keep_current_lift))
                if bool(preserve_curve_contact_start):
                    start_joint = [float(v) for v in cur_joint[:6]]
                # Always use IK for curve waypoints (no non-IK shortcut).
                relative_curve_follow = False
                if fixed_drag_lift is not None:
                    start_joint[1] = float(fixed_drag_lift)
                if bool(force_keep_current_lift):
                    # No initial arm retraction when chained from grasp(no-lift) for curve.
                    if str(path_kind) == "curve":
                        max_retract = 0.0
                    else:
                        max_retract = float(max(0.0, float(DRAG_CHAINED_MAX_INITIAL_RETRACT_M)))
                    min_arm = float(cur_joint[2]) - float(max_retract)
                    start_joint[2] = float(
                        np.clip(
                            max(float(start_joint[2]), float(min_arm)),
                            float(self.ros_node.JOINT_LIMITS[0][0]),
                            float(self.ros_node.JOINT_LIMITS[0][1]),
                        )
                    )
                # Move above start then lower, except chained curve replay where
                # we are already at contact start after grasp.
                if (not bool(relative_curve_follow)) and (not bool(preserve_curve_contact_start)):
                    safe_joint = [float(v) for v in start_joint[:6]]
                    if bool(force_keep_current_lift) and (fixed_drag_lift is not None):
                        safe_lift = float(fixed_drag_lift)
                    else:
                        safe_lift = float(
                            np.clip(
                                max(float(IK_SAFE_LIFT_M), float(cur_joint[1]), float(start_joint[1]) + float(IK_REACH_STANDOFF_M)),
                                float(self.ros_node.JOINT_LIMITS[1][0]),
                                float(self.ros_node.JOINT_LIMITS[1][1]),
                            )
                        )
                    safe_joint[1] = safe_lift
                    self._set_status("Drag: moving above start point...", "QLabel { color: blue; font-size: 10px; }")
                    if not self._execute_arm_to_chunked(
                        safe_joint[:6],
                        gripper=None,
                        timeout_s=float(ACTION_MOVE_TIMEOUT_DEFAULT_S),
                        reliable=False,
                    ):
                        raise RuntimeError("failed moving above drag start")

                    self._set_status("Drag: lowering to drag height...", "QLabel { color: blue; font-size: 10px; }")
                    if not self._execute_arm_to_chunked(
                        start_joint[:6],
                        gripper=None,
                        timeout_s=float(ACTION_MOVE_TIMEOUT_DEFAULT_S),
                        reliable=False,
                    ):
                        raise RuntimeError("failed lowering to drag start")
                else:
                    self._set_status(
                        "Curve: chained start pose detected; following path without extra lift/retract.",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )

                def _follow_path_points(target_points: list[PointStamped], *, phase_text: str) -> None:
                    total_hops = len(target_points)
                    moved_hops = 0
                    allow_skip = bool(str(path_kind) == "curve")
                    for hop_idx, p_tgt in enumerate(target_points, start=1):
                        if self._is_abort_requested():
                            raise RuntimeError("drag operation aborted")
                        self._set_status(
                            f"{phase_text} {hop_idx}/{total_hops}...",
                            "QLabel { color: blue; font-size: 10px; }",
                        )
                        tgt_joint = self._plan_drag_joint_for_point_base(
                            p_tgt,
                            wrist_yaw=wrist_yaw,
                            wrist_pitch=wrist_pitch,
                            wrist_roll=wrist_roll,
                            base_pose_for_conversion=drag_ref_base_pose,
                        )
                        if not (isinstance(tgt_joint, list) and len(tgt_joint) >= 6):
                            if bool(allow_skip):
                                print(
                                    f"[curve drag] skipping unplannable waypoint "
                                    f"{hop_idx}/{total_hops}"
                                )
                                continue
                            raise RuntimeError(f"failed dragging waypoint {hop_idx}/{total_hops}")
                        drag_joint = [float(v) for v in tgt_joint[:6]]
                        drag_joint[1] = float(start_joint[1])  # keep contact height across path.
                        if not self._execute_arm_to_chunked(
                            drag_joint[:6],
                            gripper=None,
                            timeout_s=float(ACTION_MOVE_TIMEOUT_LONG_S),
                            reliable=False,
                        ):
                            if bool(allow_skip):
                                print(
                                    f"[curve drag] skipping failed waypoint move "
                                    f"{hop_idx}/{total_hops}"
                                )
                                continue
                            raise RuntimeError(f"failed dragging waypoint {hop_idx}/{total_hops}")
                        moved_hops += 1
                    if total_hops > 0 and moved_hops <= 0:
                        raise RuntimeError(f"no valid waypoint move during '{phase_text}'")

                for pass_idx in range(1, int(repeat_n) + 1):
                    if self._is_abort_requested():
                        raise RuntimeError("drag operation aborted")
                    _follow_path_points(
                        path_points[1:],
                        phase_text=f"{mode_label}: forward {pass_idx}/{repeat_n}",
                    )
                    if return_path and pass_idx < int(repeat_n):
                        _follow_path_points(
                            list(reversed(path_points[:-1])),
                            phase_text=f"{mode_label}: return {pass_idx}/{repeat_n}",
                        )

                drag_success = True
                self._set_action_state("idle")
                self._set_return_enabled(True)
                if str(path_kind) == "curve":
                    if int(repeat_n) > 1:
                        self._set_status(
                            f"Curved drag complete ({repeat_n} forward pass(es)). Press Return when needed.",
                            "QLabel { color: green; font-size: 10px; }",
                        )
                    else:
                        self._set_status(
                            "Curved drag complete. Press Return when needed.",
                            "QLabel { color: green; font-size: 10px; }",
                        )
                else:
                    if int(repeat_n) > 1:
                        self._set_status(
                            f"Drag operation complete ({repeat_n} forward pass(es)). Press Return when needed.",
                            "QLabel { color: green; font-size: 10px; }",
                        )
                    else:
                        self._set_status(
                            "Drag operation complete. Press Return when needed.",
                            "QLabel { color: green; font-size: 10px; }",
                        )
            except Exception as exc:
                self._set_status(f"Drag failed: {exc}", "QLabel { color: red; font-size: 10px; }")
                # Keep Return available after drag failure so operator can recover.
                self._set_return_enabled(True)
            finally:
                launch_next = False
                if self._run_all_queued_goals:
                    if bool(drag_success):
                        if self._goal_sequence_has_next():
                            launch_next = True
                        else:
                            self._run_all_queued_goals = False
                    else:
                        # Stop queued sequence on drag failure to avoid wrong-place release.
                        self._run_all_queued_goals = False
                if self._deferred_next_goal_start:
                    if self._goal_sequence_has_next():
                        self._deferred_next_goal_start = False
                        launch_next = True
                    else:
                        self._deferred_next_goal_start = False
                with self._action_lock:
                    st = self._action_state
                if st == "running":
                    self._set_action_state("idle")
                self._update_goal_queue_label()
                self._update_next_goal_button_state()
                if launch_next:
                    self._start_next_queued_goal()

        threading.Thread(target=run, daemon=True).start()
        return True

    def _on_head_mouse_press(self, event):
        if self.head_rgb is None:
            return
        if event.button() == Qt.MouseButton.LeftButton and bool(self._drag_draw_mode):
            px, py = self._pixel_from_event(event)
            if px is None:
                return
            self._drag_dragging = True
            self._drag_start_px = (int(px), int(py))
            self._drag_live_px = (int(px), int(py))
            self._drag_path_points = [(int(px), int(py))]
            self.update_camera_displays()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._manual_draw_mode == "draw_rect":
            px, py = self._pixel_from_event(event)
            if px is None:
                return
            self._manual_dragging_rect = True
            self._manual_rect_start_px = (int(px), int(py))
            self._manual_rect_live_px = (int(px), int(py))
            self.update_camera_displays()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._manual_draw_mode == "pick_points":
            px, py = self._pixel_from_event(event)
            if px is None:
                return
            region = self._manual_region_by_id(self._manual_selected_region_id)
            if region is None:
                self._set_status("Pick Points: select a region first", "QLabel { color: orange; }")
                return
            points = list(region.get("points", []))
            if len(points) >= 2:
                self._set_status("Region already has 2 points. Compute grasp or redraw.", "QLabel { color: orange; }")
                return
            points.append((int(px), int(py)))
            region["points"] = points
            region["computed"] = None
            self._manual_refresh_region_tree()
            self.update_camera_displays()
            if len(points) < 2:
                self._set_status("Pick second grip-tip point", "QLabel { color: #1e88e5; font-size: 10px; }")
            else:
                self._manual_draw_mode = None
                self._set_status("2 points captured. Click Compute Grasp.", "QLabel { color: green; font-size: 10px; }")
            return
        self.on_image_click(event)

    def _on_head_mouse_move(self, event):
        if bool(self._drag_draw_mode) and bool(self._drag_dragging):
            px, py = self._pixel_from_event(event)
            if px is None:
                return
            self._drag_live_px = (int(px), int(py))
            if str(getattr(self, "_drag_mode_kind", "line")) == "curve":
                p = (int(px), int(py))
                if len(self._drag_path_points) == 0 or self._drag_path_points[-1] != p:
                    self._drag_path_points.append(p)
            self.update_camera_displays()
            return
        if not self._manual_dragging_rect or self._manual_draw_mode != "draw_rect":
            return
        px, py = self._pixel_from_event(event)
        if px is None:
            return
        self._manual_rect_live_px = (int(px), int(py))
        self.update_camera_displays()

    def _on_head_mouse_release(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if bool(self._drag_draw_mode):
            if not bool(self._drag_dragging) or self._drag_start_px is None:
                self._drag_draw_mode = False
                self._drag_mode_kind = "line"
                self._drag_dragging = False
                self._drag_start_px = None
                self._drag_live_px = None
                self._drag_path_points = []
                self.update_camera_displays()
                return
            px, py = self._pixel_from_event(event)
            if px is None:
                self._drag_draw_mode = False
                self._drag_mode_kind = "line"
                self._drag_dragging = False
                self._drag_start_px = None
                self._drag_live_px = None
                self._drag_path_points = []
                self.update_camera_displays()
                return
            self._drag_live_px = (int(px), int(py))
            sx, sy = self._drag_start_px
            ex, ey = self._drag_live_px
            draw_kind = str(getattr(self, "_drag_mode_kind", "line"))
            path_pts = list(getattr(self, "_drag_path_points", []))
            if len(path_pts) == 0 or path_pts[-1] != (int(ex), int(ey)):
                path_pts.append((int(ex), int(ey)))
            self._drag_draw_mode = False
            self._drag_mode_kind = "line"
            self._drag_dragging = False
            self._drag_start_px = None
            self._drag_live_px = None
            self._drag_path_points = []
            self.update_camera_displays()
            if draw_kind == "curve":
                smooth_pts = self._fit_smooth_curve_pixels(path_pts, samples=int(DRAG_CURVE_FIT_SAMPLES))
                if len(smooth_pts) < int(DRAG_CURVE_MIN_CAPTURE_POINTS) or self._polyline_length_px(smooth_pts) < float(DRAG_MIN_PIXEL_LENGTH_PX):
                    self._set_status("Curve: path too short", "QLabel { color: orange; }")
                    return
                self.add_curved_drag_goal_from_pixels(smooth_pts)
            else:
                if max(abs(int(ex) - int(sx)), abs(int(ey) - int(sy))) < int(DRAG_MIN_PIXEL_LENGTH_PX):
                    self._set_status("Drag: line too short", "QLabel { color: orange; }")
                    return
                self.add_drag_goal_between_pixels((int(sx), int(sy)), (int(ex), int(ey)))
            return
        if not self._manual_dragging_rect or self._manual_draw_mode != "draw_rect":
            return
        px, py = self._pixel_from_event(event)
        if px is None:
            self._manual_dragging_rect = False
            self._manual_rect_start_px = None
            self._manual_rect_live_px = None
            return
        self._manual_rect_live_px = (int(px), int(py))
        x0, y0 = self._manual_rect_start_px if self._manual_rect_start_px is not None else (int(px), int(py))
        x1, y1 = self._manual_rect_live_px
        xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
        ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
        self._manual_dragging_rect = False
        self._manual_rect_start_px = None
        self._manual_rect_live_px = None
        if abs(xb - xa) < 6 or abs(yb - ya) < 6:
            self._set_status("Draw Region: rectangle too small", "QLabel { color: orange; }")
            self.update_camera_displays()
            return
        rid = int(self._manual_region_next_id)
        self._manual_region_next_id += 1
        self.manual_grasp_regions.append(
            {
                "id": rid,
                "rect": (int(xa), int(ya), int(xb), int(yb)),
                "points": [],
                "computed": None,
            }
        )
        self._manual_selected_region_id = rid
        self._manual_draw_mode = "pick_points"
        self._manual_refresh_region_tree()
        self.update_camera_displays()
        self._set_status("Region created. Click 2 grip-tip points inside the box.", "QLabel { color: #1e88e5; font-size: 10px; }")

    def _selected_manual_region(self):
        region = self._manual_region_by_id(self._manual_selected_region_id)
        if region is None and self.manual_grasp_regions:
            region = self.manual_grasp_regions[-1]
            self._manual_selected_region_id = int(region["id"])
        return region

    def _mask_from_manual_region(self, region):
        if self.head_rgb is None:
            return None
        h, w = self.head_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = self._manual_region_rect_norm(region)
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 <= x0 or y1 <= y0:
            return None
        mask[y0:y1 + 1, x0:x1 + 1] = 1
        return mask

    def compute_selected_manual_region_grasp(self):
        region = self._selected_manual_region()
        if region is None:
            self._set_status("Compute Grasp: no region selected", "QLabel { color: orange; }")
            return
        mask = self._mask_from_manual_region(region)
        if mask is None:
            self._set_status("Compute Grasp: invalid region mask", "QLabel { color: red; }")
            return

        x0, y0, x1, y1 = self._manual_region_rect_norm(region)
        cx = int(round((x0 + x1) * 0.5))
        cy = int(round((y0 + y1) * 0.5))
        point_base, depth = self._get_3d_point_at_pixel(cx, cy)
        if point_base is None or depth is None:
            self._set_status("Compute Grasp: invalid depth at region center", "QLabel { color: red; }")
            return

        shape_info = self._analyze_segment_geometry(mask)
        grasp_yaw, rect_info = self._compute_grasp_orientation(mask, cx, cy)

        points = list(region.get("points", []))
        gripper_width = None
        if len(points) >= 2:
            p1, p2 = points[0], points[1]
            pb1, _ = self._get_3d_point_at_pixel(int(p1[0]), int(p1[1]))
            pb2, _ = self._get_3d_point_at_pixel(int(p2[0]), int(p2[1]))
            if pb1 is not None and pb2 is not None:
                dx = float(pb2.point.x - pb1.point.x)
                dy = float(pb2.point.y - pb1.point.y)
                dz = float(pb2.point.z - pb1.point.z)
                if np.hypot(dx, dy) > 1e-4:
                    axis_angle = float(np.arctan2(dy, dx))
                    grasp_yaw = self._resolve_wrist_yaw_candidate(axis_angle + float(np.pi / 2.0))
                width_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                grip_aperture = float(width_m + 0.02)
                gripper_width = float(
                    np.clip(
                        grip_aperture / 0.22,
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )

        if gripper_width is None:
            gripper_width = self._estimate_gripper_width(mask, cx, cy, depth, rect_info)

        object_top_z = self._compute_object_top_z(mask)
        if rect_info is not None and rect_info.get("top_z_max") is not None:
            top_from_fit = float(rect_info["top_z_max"])
            object_top_z = top_from_fit if object_top_z is None else max(float(object_top_z), top_from_fit)

        region["computed"] = {
            "target_base_xyz": [float(point_base.point.x), float(point_base.point.y), float(point_base.point.z)],
            "grasp_yaw": float(grasp_yaw),
            "gripper_width": float(gripper_width),
            "object_top_z": None if object_top_z is None else float(object_top_z),
            "shape_info": shape_info,
            "rect_info": rect_info,
        }
        self._grasp_debug_info = rect_info
        self._manual_refresh_region_tree()
        self.update_camera_displays()
        self._set_status(
            f"Region grasp computed: yaw={np.degrees(float(grasp_yaw)):.1f}deg, grip={float(gripper_width):.3f}",
            "QLabel { color: #1e88e5; font-size: 10px; }",
        )

    def align_gripper_to_selected_manual_region(self):
        region = self._selected_manual_region()
        if region is None or not isinstance(region.get("computed"), dict):
            self._set_status("Align Gripper: compute grasp first", "QLabel { color: orange; }")
            return
        comp = region["computed"]
        joint = self._current_manip_joint6()
        if not (isinstance(joint, list) and len(joint) >= 6):
            self._set_status("Align Gripper: current joint state unavailable", "QLabel { color: red; }")
            return
        joint = [float(v) for v in joint[:6]]
        joint[3] = float(comp["grasp_yaw"])
        joint[4] = float(np.deg2rad(GRASP_PITCH_DEG))
        ok = self._execute_arm_to_chunked(joint[:6], gripper=None, timeout_s=6.0, reliable=False)
        if ok:
            self._set_status("Gripper aligned to computed orientation", "QLabel { color: green; font-size: 10px; }")
        else:
            self._set_status("Align Gripper failed", "QLabel { color: red; font-size: 10px; }")

    def execute_selected_manual_region_grasp(self):
        region = self._selected_manual_region()
        if region is None or not isinstance(region.get("computed"), dict):
            self._set_status("Execute Region Grasp: compute grasp first", "QLabel { color: orange; }")
            return
        if not self._begin_action('grasp'):
            self._set_status("Another action is already running/paused", "QLabel { color: orange; font-size: 10px; }")
            return

        comp = region["computed"]
        target_xyz = comp.get("target_base_xyz")
        if not (isinstance(target_xyz, list) and len(target_xyz) >= 3):
            self._set_action_state('idle')
            self._set_status("Execute Region Grasp: invalid target xyz", "QLabel { color: red; }")
            return

        # Use the latest manual wrist orientation at execution time, so any
        # user adjustments after Align Gripper are honored by IK planning.
        exec_wrist_yaw = float(comp.get("grasp_yaw", 0.0))
        exec_wrist_pitch = None
        exec_wrist_roll = None
        joint_now = self._current_manip_joint6()
        if isinstance(joint_now, list) and len(joint_now) >= 6:
            exec_wrist_yaw = float(joint_now[3])
            exec_wrist_pitch = float(joint_now[4])
            exec_wrist_roll = float(joint_now[5])

        mask = self._mask_from_manual_region(region)
        if mask is None:
            self._set_action_state('idle')
            self._set_status("Execute Region Grasp: invalid region mask", "QLabel { color: red; }")
            return
        self._grasp_debug_info = comp.get("rect_info")
        self.update_camera_displays()

        point_base = self._point_from_xyz(target_xyz)
        gripper_width = float(comp.get("gripper_width", self.ros_node.JOINT_LIMITS[7][1]))
        object_top_z = comp.get("object_top_z")
        shape_info = comp.get("shape_info")

        self._set_status(
            "Executing manual region grasp (using current wrist orientation)...",
            "QLabel { color: blue; font-size: 10px; }",
        )

        def run():
            try:
                self._execute_approach(
                    point_base,
                    mode='grasp',
                    grasp_yaw=exec_wrist_yaw,
                    gripper_width=gripper_width,
                    object_top_z=object_top_z,
                    grasp_mask=mask,
                    long_axis_angle=None,
                    wrist_pitch_target=exec_wrist_pitch,
                    wrist_roll_target=exec_wrist_roll,
                    grasp_shape_info=shape_info,
                )
            except Exception as e:
                print(f"Manual region grasp error: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"Manual region grasp failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
            finally:
                with self._action_lock:
                    st = self._action_state
                if st == 'running':
                    self._set_action_state('idle')

        from threading import Thread
        Thread(target=run, daemon=True).start()

    def delete_selected_manual_region(self):
        region = self._selected_manual_region()
        if region is None:
            self._set_status("Delete Region: no region selected", "QLabel { color: orange; }")
            return
        rid = int(region["id"])
        self.manual_grasp_regions = [r for r in self.manual_grasp_regions if int(r.get("id", -1)) != rid]
        if self._manual_selected_region_id == rid:
            self._manual_selected_region_id = None
        self._manual_draw_mode = None
        self._manual_refresh_region_tree()
        self.update_camera_displays()
        self._set_status(f"Deleted region R{rid}", "QLabel { color: gray; font-size: 10px; }")

    def on_image_click(self, event):
        """Handle click on head RGB image — left-click selects segment, right-click shows context menu"""
        if self.head_rgb is None:
            return
        if hasattr(self, "primary_camera_combo"):
            try:
                key = self.primary_camera_combo.currentData()
                if isinstance(key, str) and key != "head":
                    self._set_status(
                        "Interactive goals use Head Camera. Set Primary camera to Head Camera.",
                        "QLabel { color: orange; font-size: 10px; }",
                    )
                    return
            except Exception:
                pass

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
            # Right-click: show context menu with goal-add operations.
            clicked_on_segment = False
            clicked_segment = None
            if self.segments:
                for i, seg in enumerate(self.segments):
                    if py < seg['mask'].shape[0] and px < seg['mask'].shape[1]:
                        if seg['mask'][py, px] > 0:
                            clicked_on_segment = True
                            clicked_segment = seg
                            break

            menu = QMenu(self)

            add_grasp_point_action = QAction("Add Grasp Point", self)
            add_grasp_point_action.triggered.connect(
                lambda: self._add_grasp_point_with_mode_prompt(
                    int(px),
                    int(py),
                    segment=(clicked_segment if bool(clicked_on_segment) else None),
                )
            )
            menu.addAction(add_grasp_point_action)

            add_place_point_action = QAction("Add Place Point", self)
            add_place_point_action.triggered.connect(
                lambda: self._add_place_point_with_mode_prompt(int(px), int(py))
            )
            menu.addAction(add_place_point_action)

            see_grasp_action = QAction("See Grasp", self)
            if clicked_on_segment:
                see_grasp_action.triggered.connect(
                    lambda: self._preview_grasp_at_pixel(px, py, segment=clicked_segment)
                )
            else:
                see_grasp_action.triggered.connect(
                    lambda: self._preview_grasp_at_pixel(px, py)
                )
            menu.addAction(see_grasp_action)

            stack_here_action = QAction("Stack Here", self)
            stack_here_action.triggered.connect(lambda: self._stack_here_at_pixel(px, py))
            menu.addAction(stack_here_action)

            add_drag_operation_action = QAction("Add Drag Operation", self)
            add_drag_operation_action.triggered.connect(self._start_drag_operation_draw)
            menu.addAction(add_drag_operation_action)

            add_curved_action = QAction("Add Cruved Trajectory", self)
            add_curved_action.triggered.connect(lambda: self._add_cruved_trajectory_at_pixel(px, py))
            menu.addAction(add_curved_action)

            menu.addSeparator()
            draw_region_action = QAction("Draw Grasp Rectangle", self)
            draw_region_action.triggered.connect(self._start_manual_region_draw)
            menu.addAction(draw_region_action)

            menu.exec(self.head_display.mapToGlobal(event.pos()))

    def _add_grasp_point_with_mode_prompt(self, px: int, py: int, *, segment=None) -> None:
        """Prompt grasp mode (with/without lift) and add queued grasp point."""
        items = ["Grasp + Lift", "Grasp (No Lift)", "Grasp Precise", "Grasp and Rotate"]
        choice, ok = QInputDialog.getItem(
            self,
            "Select Grasp Type",
            "Choose grasp behavior:",
            items,
            0,
            False,
        )
        if not bool(ok):
            self._set_status("Add Grasp Point canceled", "QLabel { color: gray; font-size: 10px; }")
            return
        selected = str(choice).strip()
        if selected == "Grasp and Rotate":
            deg, ok_deg = QInputDialog.getDouble(
                self,
                "Rotate Wrist Yaw",
                "Rotation amount (degrees, +/-):",
                30.0,
                -360.0,
                360.0,
                1,
            )
            if not bool(ok_deg):
                self._set_status("Add Grasp+Rotate canceled", "QLabel { color: gray; font-size: 10px; }")
                return
            self.add_grasp_goal_at_pixel(
                int(px),
                int(py),
                segment=segment,
                post_grasp_lift=False,
                grasp_rotate_deg=float(deg),
                wrist_pitch_target=float(GRASP_ROTATE_FORCE_PITCH_RAD),
            )
            return
        if selected == "Grasp Precise":
            self.add_grasp_goal_at_pixel(
                int(px),
                int(py),
                segment=segment,
                post_grasp_lift=True,
                precise_grasp=True,
            )
            return
        use_lift = selected == "Grasp + Lift"
        self.add_grasp_goal_at_pixel(int(px), int(py), segment=segment, post_grasp_lift=bool(use_lift))

    def _add_place_point_with_mode_prompt(self, px: int, py: int) -> None:
        """Prompt place mode (place/release) and add corresponding goal."""
        items = ["Place", "Place Precise", "Place Object", "Release"]
        choice, ok = QInputDialog.getItem(
            self,
            "Select Place Type",
            "Choose place behavior:",
            items,
            0,
            False,
        )
        if not bool(ok):
            self._set_status("Add Place Point canceled", "QLabel { color: gray; font-size: 10px; }")
            return
        selected = str(choice).strip()
        if selected == "Release":
            self.add_release_goal_from_previous_goal_end()
            return
        if selected == "Place Precise":
            self.add_reach_goal_at_pixel(int(px), int(py), precise_place=True)
            return
        if selected == "Place Object":
            self.add_place_object_goal_at_pixel(int(px), int(py))
            return
        self.add_reach_goal_at_pixel(int(px), int(py))

    def _stack_here_at_pixel(self, px: int, py: int) -> None:
        """Temporary stack-here behavior: queue as a place point target."""
        self.add_reach_goal_at_pixel(int(px), int(py))
        self._set_status("Added: Stack Here", "QLabel { color: #1e88e5; font-size: 10px; }")

    def _add_cruved_trajectory_at_pixel(self, px: int, py: int) -> None:
        """Start interactive curved-trajectory drawing mode."""
        _ = (int(px), int(py))
        self._start_curved_trajectory_draw()

    def _prompt_curve_height_options(self) -> tuple[bool, float, bool]:
        """Prompt curve execution height options.

        Returns: (accepted, height_cm, no_height_adjustment)
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Curve Height Options")
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Height above drawn surface (cm, +/-):"))
        spin = QDoubleSpinBox(dlg)
        spin.setRange(-200.0, 200.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setValue(0.0)
        spin.setSuffix(" cm")
        layout.addWidget(spin)

        no_adj_cb = QCheckBox("No height adjustment (keep lifted surface)", dlg)
        no_adj_cb.setChecked(False)
        no_adj_cb.setToolTip(
            "Checked: ignore height input. In sequence after grasp, keep current lifted level. "
            "Curve-only start uses safe lift."
        )
        layout.addWidget(no_adj_cb)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = QPushButton("OK", dlg)
        cancel_btn = QPushButton("Cancel", dlg)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        accepted = bool(dlg.exec() == QDialog.DialogCode.Accepted)
        return accepted, float(spin.value()), bool(no_adj_cb.isChecked())

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

    def _get_3d_point_nearest_valid_depth(
        self,
        px: int,
        py: int,
        *,
        search_radius_px: int = 0,
        preferred_dir_px: tuple[float, float] | None = None,
        require_forward: bool = False,
    ):
        """Get 3D point at (px,py), or nearest valid-depth pixel within radius.

        Returns (point_base, depth, resolved_px, resolved_py). If no valid depth
        is found in the radius, returns (None, None, None, None).
        """
        try:
            px_i = int(px)
            py_i = int(py)
        except Exception:
            return None, None, None, None

        p0, d0 = self._get_3d_point_at_pixel(px_i, py_i)
        if p0 is not None:
            return p0, d0, int(px_i), int(py_i)

        r = int(max(0, int(search_radius_px)))
        if r <= 0 or self.depth_image is None:
            return None, None, None, None

        h, w = self.depth_image.shape[:2]
        if not (0 <= px_i < w and 0 <= py_i < h):
            return None, None, None, None

        x0 = int(max(0, px_i - r))
        x1 = int(min(w - 1, px_i + r))
        y0 = int(max(0, py_i - r))
        y1 = int(min(h - 1, py_i + r))
        if x1 < x0 or y1 < y0:
            return None, None, None, None

        region = self.depth_image[y0 : y1 + 1, x0 : x1 + 1]
        valid = np.isfinite(region) & (region > 0.1) & (region < 5.0)
        if not np.any(valid):
            return None, None, None, None

        ys, xs = np.nonzero(valid)
        gxs = xs.astype(np.int32) + int(x0)
        gys = ys.astype(np.int32) + int(y0)
        dx = gxs.astype(np.float32) - float(px_i)
        dy = gys.astype(np.float32) - float(py_i)
        dist2 = dx * dx + dy * dy
        within = dist2 <= float(r * r)
        if not np.any(within):
            return None, None, None, None

        gxs = gxs[within]
        gys = gys[within]
        dist2 = dist2[within]

        # Optional directional gating: prefer (or require) candidates in the
        # same 2D pixel direction as the intended path tangent.
        if isinstance(preferred_dir_px, (list, tuple)) and len(preferred_dir_px) >= 2:
            try:
                dirx = float(preferred_dir_px[0])
                diry = float(preferred_dir_px[1])
            except Exception:
                dirx = 0.0
                diry = 0.0
            if math.isfinite(dirx) and math.isfinite(diry):
                nrm = float(math.hypot(dirx, diry))
                if nrm > 1e-6:
                    vx = gxs.astype(np.float32) - float(px_i)
                    vy = gys.astype(np.float32) - float(py_i)
                    dots = vx * float(dirx) + vy * float(diry)
                    forward = dots >= 0.0
                    if np.any(forward):
                        gxs = gxs[forward]
                        gys = gys[forward]
                        dist2 = dist2[forward]
                    elif bool(require_forward):
                        return None, None, None, None

        order = np.argsort(dist2, kind="stable")
        for oi in order.tolist():
            cx = int(gxs[int(oi)])
            cy = int(gys[int(oi)])
            p, d = self._get_3d_point_at_pixel(cx, cy)
            if p is not None:
                return p, d, cx, cy
        return None, None, None, None

    def _build_fallback_base_path_from_pixels(
        self,
        pixels: list[tuple[int, int]],
        *,
        search_radius_px: int,
    ) -> list[tuple[float, float, float]]:
        """Build dense fallback base-frame path for queued drag/curve execution.

        Uses nearest-valid-depth lookup per pixel and fills missing points from
        neighboring valid samples so index-based remapping stays stable.
        """
        if not isinstance(pixels, list) or len(pixels) < 2:
            return []

        vals: list[tuple[float, float, float] | None] = []
        has_valid = False
        valid_count = 0
        npx = len(pixels)
        for i, pxy in enumerate(pixels):
            if not (isinstance(pxy, (list, tuple)) and len(pxy) >= 2):
                vals.append(None)
                continue
            try:
                px_i = int(pxy[0])
                py_i = int(pxy[1])
            except Exception:
                vals.append(None)
                continue
            pref_dir = None
            if npx >= 2:
                if i == 0:
                    nx = int(pixels[1][0]) - int(pixels[0][0])
                    ny = int(pixels[1][1]) - int(pixels[0][1])
                    pref_dir = (float(nx), float(ny))
                elif i == (npx - 1):
                    nx = int(pixels[-1][0]) - int(pixels[-2][0])
                    ny = int(pixels[-1][1]) - int(pixels[-2][1])
                    pref_dir = (float(nx), float(ny))
                else:
                    nx = int(pixels[i + 1][0]) - int(pixels[i - 1][0])
                    ny = int(pixels[i + 1][1]) - int(pixels[i - 1][1])
                    pref_dir = (float(nx), float(ny))
            p, _, _rx, _ry = self._get_3d_point_nearest_valid_depth(
                int(px_i),
                int(py_i),
                search_radius_px=int(search_radius_px),
                preferred_dir_px=pref_dir,
                require_forward=bool(i > 0),
            )
            if p is None:
                vals.append(None)
                continue
            xyz = (float(p.point.x), float(p.point.y), float(p.point.z))
            vals.append(xyz)
            has_valid = True
            valid_count += 1

        if not has_valid:
            return []
        if int(valid_count) < 2:
            return []

        # Forward-fill gaps.
        last_valid: tuple[float, float, float] | None = None
        for i in range(len(vals)):
            if vals[i] is None:
                if last_valid is not None:
                    vals[i] = last_valid
            else:
                last_valid = vals[i]

        # Backward-fill leading gaps.
        next_valid: tuple[float, float, float] | None = None
        for i in range(len(vals) - 1, -1, -1):
            if vals[i] is None:
                if next_valid is not None:
                    vals[i] = next_valid
            else:
                next_valid = vals[i]

        out: list[tuple[float, float, float]] = []
        for v in vals:
            if v is None:
                continue
            out.append((float(v[0]), float(v[1]), float(v[2])))
        if len(out) >= 2:
            dx = float(out[-1][0]) - float(out[0][0])
            dy = float(out[-1][1]) - float(out[0][1])
            dz = float(out[-1][2]) - float(out[0][2])
            if (dx * dx + dy * dy + dz * dz) < 1e-8:
                return []
        return out

    def _resolve_wrist_yaw_candidate(self, desired_yaw: float) -> float:
        """Resolve desired wrist yaw with +/-pi-equivalent fallback inside joint limits."""
        import math

        desired = float(math.atan2(math.sin(float(desired_yaw)), math.cos(float(desired_yaw))))
        wrist_lo, wrist_hi = float(self.ros_node.JOINT_LIMITS[2][0]), float(self.ros_node.JOINT_LIMITS[2][1])

        current_wrist_yaw = 0.0
        try:
            q_now = self.ros_node.get_actual_qpos()
            if isinstance(q_now, list) and len(q_now) > 2:
                cand = float(q_now[2])
                if math.isfinite(cand):
                    current_wrist_yaw = cand
        except Exception:
            current_wrist_yaw = 0.0

        candidates: list[float] = []
        for k in range(-3, 4):
            y = float(desired + float(k) * math.pi)
            if wrist_lo <= y <= wrist_hi:
                candidates.append(y)
        if candidates:
            return min(
                candidates,
                key=lambda y: abs(math.atan2(math.sin(y - current_wrist_yaw), math.cos(y - current_wrist_yaw))),
            )
        return float(np.clip(desired, wrist_lo, wrist_hi))

    def _mask_to_base_points(self, mask, max_samples: int = 5000):
        """Project mask pixels with valid depth into base_link point cloud."""
        if self.depth_image is None or self.ros_node.camera_info is None or mask is None:
            return None

        mask_ys, mask_xs = np.where(mask > 0)
        if len(mask_ys) < 30:
            return None

        if len(mask_ys) > int(max_samples):
            step = int(np.ceil(len(mask_ys) / float(max_samples)))
            mask_ys = mask_ys[::step]
            mask_xs = mask_xs[::step]

        depths = self.depth_image[mask_ys, mask_xs].astype(np.float64)
        valid = (depths > 0.1) & (depths < 5.0)
        if not np.any(valid):
            return None
        mask_ys = mask_ys[valid]
        mask_xs = mask_xs[valid]
        depths = depths[valid]

        fx = self.ros_node.camera_info.k[0]
        fy = self.ros_node.camera_info.k[4]
        cx_cam = self.ros_node.camera_info.k[2]
        cy_cam = self.ros_node.camera_info.k[5]
        H_orig = self.ros_node.camera_info.height

        try:
            tf_msg = self.ros_node.tf_buffer.lookup_transform(
                "base_link",
                "camera_color_optical_frame",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except Exception:
            return None

        q = tf_msg.transform.rotation
        R = np.array([
            [1 - 2 * (q.y**2 + q.z**2), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w)],
            [2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x**2 + q.z**2), 2 * (q.y * q.z - q.x * q.w)],
            [2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x**2 + q.y**2)],
        ])
        t_vec = np.array(
            [
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z,
            ]
        )

        rotated_head = bool(getattr(self.ros_node, "head_image_rotated_90_cw", lambda: STRETCH_AI_ROTATE_HEAD_90_CW)())
        if rotated_head:
            orig_col = mask_ys.astype(np.float64)
            orig_row = (H_orig - 1 - mask_xs).astype(np.float64)
        else:
            orig_col = mask_xs.astype(np.float64)
            orig_row = mask_ys.astype(np.float64)

        cam_x = (orig_col - cx_cam) * depths / fx
        cam_y = (orig_row - cy_cam) * depths / fy
        cam_z = depths
        cam_pts = np.stack([cam_x, cam_y, cam_z], axis=1)
        base_pts = (R @ cam_pts.T).T + t_vec
        if len(base_pts) < 20:
            return None
        return base_pts

    def _analyze_segment_geometry(self, mask):
        """Analyze visible 3D surface geometry and choose grasp approach strategy."""
        import math

        base_pts = self._mask_to_base_points(mask, max_samples=5000)
        if base_pts is None or len(base_pts) < 30:
            return None

        pts = np.asarray(base_pts, dtype=np.float64).reshape(-1, 3)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = (centered.T @ centered) / max(1, len(centered) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        axis_major = eigvecs[:, 0]
        axis_mid = eigvecs[:, 1]
        axis_minor = eigvecs[:, 2]  # approx visible-surface normal
        if axis_minor[2] < 0.0:
            axis_minor = -axis_minor

        proj_major = centered @ axis_major
        proj_mid = centered @ axis_mid
        proj_minor = centered @ axis_minor
        span_major = float(np.quantile(proj_major, 0.95) - np.quantile(proj_major, 0.05))
        span_mid = float(np.quantile(proj_mid, 0.95) - np.quantile(proj_mid, 0.05))
        span_minor = float(np.quantile(proj_minor, 0.95) - np.quantile(proj_minor, 0.05))

        z_vals = pts[:, 2]
        z_span = float(np.quantile(z_vals, 0.95) - np.quantile(z_vals, 0.05))
        xy = pts[:, :2]
        x_span = float(np.quantile(xy[:, 0], 0.95) - np.quantile(xy[:, 0], 0.05))
        y_span = float(np.quantile(xy[:, 1], 0.95) - np.quantile(xy[:, 1], 0.05))
        xy_span = float(np.hypot(x_span, y_span))

        normal_z_abs = float(abs(axis_minor[2]))
        is_vertical_face = normal_z_abs <= float(VERTICAL_SURFACE_NORMAL_Z_MAX)
        is_horizontal_top = normal_z_abs >= float(HORIZONTAL_SURFACE_NORMAL_Z_MIN)
        is_tall_slender = (z_span >= float(VERTICAL_OBJECT_HEIGHT_MIN_M)) and (xy_span <= float(VERTICAL_OBJECT_XY_SPAN_MAX_M))

        if is_vertical_face or is_tall_slender:
            geometry_class = "vertical_like"
            approach_strategy = "reach_standoff"
        elif is_horizontal_top:
            geometry_class = "horizontal_top_like"
            approach_strategy = "lift_standoff"
        else:
            geometry_class = "mixed_surface"
            approach_strategy = "lift_standoff"

        # Use the most horizontal principal axis among (major, mid) for yaw inference.
        horiz_major = float(np.linalg.norm(axis_major[:2]))
        horiz_mid = float(np.linalg.norm(axis_mid[:2]))
        axis_for_yaw = axis_major if horiz_major >= horiz_mid else axis_mid
        axis_xy_norm = float(np.linalg.norm(axis_for_yaw[:2]))
        if axis_xy_norm > 1e-6:
            axis_angle_xy = float(math.atan2(float(axis_for_yaw[1]), float(axis_for_yaw[0])))
        else:
            axis_angle_xy = None

        info = {
            "ok": True,
            "centroid": [float(v) for v in centroid.tolist()],
            "axis_major": [float(v) for v in axis_major.tolist()],
            "axis_mid": [float(v) for v in axis_mid.tolist()],
            "axis_minor": [float(v) for v in axis_minor.tolist()],
            "span_major": span_major,
            "span_mid": span_mid,
            "span_minor": span_minor,
            "z_span": z_span,
            "xy_span": xy_span,
            "normal_z_abs": normal_z_abs,
            "is_vertical_face": bool(is_vertical_face),
            "is_horizontal_top": bool(is_horizontal_top),
            "is_tall_slender": bool(is_tall_slender),
            "geometry_class": geometry_class,
            "approach_strategy": approach_strategy,
            "axis_angle_xy_rad": axis_angle_xy,
        }

        print(
            "[geometry] "
            f"class={geometry_class} strategy={approach_strategy} "
            f"normal|z|={normal_z_abs:.3f} z_span={z_span:.3f} xy_span={xy_span:.3f} "
            f"span_major={span_major:.3f} span_mid={span_mid:.3f} span_minor={span_minor:.3f}"
        )
        return info

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

        # Grasp yaw relative to arm direction (-Y), with wrist-limit-aware
        # equivalent-angle fallback (yaw and yaw +/- pi are equivalent for
        # parallel-jaw grasp alignment).
        desired_yaw = long_axis_angle + math.pi / 2.0
        desired_yaw = math.atan2(math.sin(desired_yaw), math.cos(desired_yaw))
        wrist_lo, wrist_hi = float(self.ros_node.JOINT_LIMITS[2][0]), float(self.ros_node.JOINT_LIMITS[2][1])

        current_wrist_yaw = 0.0
        try:
            q_now = self.ros_node.get_actual_qpos()
            if isinstance(q_now, list) and len(q_now) > 2:
                wrist_yaw_now = float(q_now[2])
                if math.isfinite(wrist_yaw_now):
                    current_wrist_yaw = wrist_yaw_now
        except Exception:
            current_wrist_yaw = 0.0

        # Build equivalent candidates separated by pi and keep only feasible ones.
        candidates: list[float] = []
        for k in range(-3, 4):
            y = float(desired_yaw + float(k) * math.pi)
            if wrist_lo <= y <= wrist_hi:
                candidates.append(y)

        used_equivalent_fallback = False
        used_hard_clip = False
        if candidates:
            # Prefer smallest motion from current wrist yaw for smoother execution.
            grasp_yaw = min(
                candidates,
                key=lambda y: abs(math.atan2(math.sin(y - current_wrist_yaw), math.cos(y - current_wrist_yaw))),
            )
            if (desired_yaw < wrist_lo) or (desired_yaw > wrist_hi):
                used_equivalent_fallback = True
        else:
            # If no equivalent representation fits, hard-clip as last resort.
            grasp_yaw = float(np.clip(desired_yaw, wrist_lo, wrist_hi))
            used_hard_clip = True

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
        if used_equivalent_fallback:
            print(
                "    NOTE: desired yaw was outside wrist limits; using equivalent +/-180deg yaw "
                f"within [{wrist_lo:.3f}, {wrist_hi:.3f}]"
            )
        elif used_hard_clip:
            print(
                "    WARNING: no equivalent yaw was inside wrist limits; hard-clipped to "
                f"[{wrist_lo:.3f}, {wrist_hi:.3f}]"
            )

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

    def _default_init_q8(self) -> list[float]:
        q8 = list(DEFAULT_INIT_CMD_QPOS8) if isinstance(DEFAULT_INIT_CMD_QPOS8, (list, tuple)) else []
        if len(q8) < 8:
            q8 = q8 + [0.0] * (8 - len(q8))
        return [
            float(np.clip(float(q8[0]), float(self.ros_node.JOINT_LIMITS[0][0]), float(self.ros_node.JOINT_LIMITS[0][1]))),
            float(np.clip(float(q8[1]), float(self.ros_node.JOINT_LIMITS[1][0]), float(self.ros_node.JOINT_LIMITS[1][1]))),
            float(np.clip(float(q8[2]), float(self.ros_node.JOINT_LIMITS[2][0]), float(self.ros_node.JOINT_LIMITS[2][1]))),
            float(np.clip(float(q8[3]), float(self.ros_node.JOINT_LIMITS[3][0]), float(self.ros_node.JOINT_LIMITS[3][1]))),
            float(np.clip(float(q8[4]), float(self.ros_node.JOINT_LIMITS[4][0]), float(self.ros_node.JOINT_LIMITS[4][1]))),
            float(np.clip(float(q8[5]), float(self.ros_node.JOINT_LIMITS[5][0]), float(self.ros_node.JOINT_LIMITS[5][1]))),
            float(np.clip(float(q8[6]), float(self.ros_node.JOINT_LIMITS[6][0]), float(self.ros_node.JOINT_LIMITS[6][1]))),
            float(np.clip(float(q8[7]), float(self.ros_node.JOINT_LIMITS[7][0]), float(self.ros_node.JOINT_LIMITS[7][1]))),
        ]

    def _default_init_joint6(self) -> list[float]:
        q8 = self._default_init_q8()
        return [
            0.0,         # manip base_x
            float(q8[1]),
            float(q8[0]),
            float(q8[2]),
            float(q8[3]),
            float(q8[4]),
        ]

    def _clip_ik_wrist_yaw_around_init(self, yaw_rad: float) -> float:
        q8 = self._default_init_q8()
        init_yaw = float(q8[2])
        band = float(np.deg2rad(abs(float(IK_WRIST_YAW_CLIP_AROUND_INIT_DEG))))
        lo = float(max(float(self.ros_node.JOINT_LIMITS[2][0]), init_yaw - band))
        hi = float(min(float(self.ros_node.JOINT_LIMITS[2][1]), init_yaw + band))
        if hi < lo:
            lo, hi = hi, lo
        return float(np.clip(float(yaw_rad), lo, hi))

    def _move_to_startup_pose_for_action(
        self,
        *,
        timeout_s: float = 12.0,
        keep_current_base_x: bool = True,
        base_x_override: float | None = None,
        retract_before_lower: bool = False,
        gripper_override: float | None = None,
    ) -> bool:
        q8 = self._default_init_q8()
        init_joint6 = self._default_init_joint6()
        cur = self._current_manip_joint6()
        if not (isinstance(cur, list) and len(cur) >= 6):
            cur = list(init_joint6)
        cur = [float(v) for v in cur[:6]]
        if isinstance(base_x_override, (int, float)) and math.isfinite(float(base_x_override)):
            target_base_x = float(
                np.clip(
                    float(base_x_override),
                    float(MANIP_BASE_X_LIMITS[0]),
                    float(MANIP_BASE_X_LIMITS[1]),
                )
            )
        else:
            target_base_x = float(cur[0]) if bool(keep_current_base_x) else float(init_joint6[0])
        head_cmd = [float(q8[5]), float(q8[6])]
        grip_cmd = float(q8[7])
        if isinstance(gripper_override, (int, float)) and math.isfinite(float(gripper_override)):
            grip_cmd = float(
                np.clip(
                    float(gripper_override),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )

        # If we are already at startup pose targets, skip redundant startup commands.
        # This avoids extra "no-op" delays before grasp planning starts.
        if not bool(retract_before_lower):
            try:
                target_joint6 = [float(v) for v in init_joint6[:6]]
                target_joint6[0] = float(target_base_x)
                joint_tols = [
                    float(WORKER_TUNE_ARM_TO_TOL_BASE_X_M),
                    float(WORKER_TUNE_ARM_TO_TOL_LIFT_M),
                    float(WORKER_TUNE_ARM_TO_TOL_ARM_M),
                    float(WORKER_TUNE_ARM_TO_TOL_WRIST_RAD),
                    float(WORKER_TUNE_ARM_TO_TOL_WRIST_RAD),
                    float(WORKER_TUNE_ARM_TO_TOL_WRIST_RAD),
                ]
                near_joint6 = all(
                    abs(float(cur[i]) - float(target_joint6[i])) <= max(1e-6, float(joint_tols[i]))
                    for i in range(6)
                )

                q_chk = self.ros_node.get_target_qpos()
                if not (isinstance(q_chk, list) and len(q_chk) >= 8):
                    q_chk = self.ros_node.get_actual_qpos()
                near_head_gripper = False
                if isinstance(q_chk, list) and len(q_chk) >= 8:
                    near_head_gripper = (
                        abs(float(q_chk[5]) - float(head_cmd[0])) <= max(1e-6, float(WORKER_TUNE_ARM_TO_TOL_HEAD_RAD))
                        and abs(float(q_chk[6]) - float(head_cmd[1])) <= max(1e-6, float(WORKER_TUNE_ARM_TO_TOL_HEAD_RAD))
                        and abs(float(q_chk[7]) - float(grip_cmd)) <= max(1e-6, float(WORKER_TUNE_ARM_TO_TOL_GRIPPER))
                    )

                if bool(near_joint6 and near_head_gripper):
                    return True
            except Exception:
                pass

        if bool(retract_before_lower):
            # Return-safe sequence:
            # 1) keep/raise lift to a safe level and retract arm first
            # 2) restore wrist pitch
            # 3) lower to init lift and apply full init-list pose
            transit_lift = float(
                np.clip(
                    max(float(cur[1]), float(init_joint6[1]), float(RETURN_SAFE_LIFT_M)),
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )
            step1 = list(cur)
            step1[0] = float(target_base_x)
            step1[1] = float(transit_lift)
            step1[2] = float(init_joint6[2])
            if not self._execute_arm_to_chunked(
                step1,
                gripper=grip_cmd,
                head=head_cmd,
                timeout_s=max(2.0, float(timeout_s) * 0.4),
                reliable=False,
            ):
                return False

            step2 = list(step1)
            step2[0] = float(target_base_x)
            step2[4] = float(init_joint6[4])
            if not self._execute_arm_to_chunked(
                step2,
                gripper=grip_cmd,
                head=head_cmd,
                timeout_s=max(2.0, float(timeout_s) * 0.4),
                reliable=False,
            ):
                return False

            step3 = list(init_joint6)
            step3[0] = float(target_base_x)
            return bool(
                self._execute_arm_to_chunked(
                    step3,
                    gripper=grip_cmd,
                    head=head_cmd,
                    timeout_s=max(2.0, float(timeout_s) * 0.4),
                    reliable=False,
                )
            )

        # Unpark sequence at round start:
        # 1) lift from park (0.75) to init-list value
        # 2) pitch from park (-1.5) to init-list value
        # 3) apply full init-list pose (other joints)
        step1 = list(cur)
        step1[0] = float(target_base_x)
        step1[1] = float(init_joint6[1])
        if not self._execute_arm_to_chunked(
            step1,
            gripper=grip_cmd,
            head=head_cmd,
            timeout_s=max(2.0, float(timeout_s) * 0.4),
            reliable=False,
        ):
            return False

        step2 = list(step1)
        step2[0] = float(target_base_x)
        step2[4] = float(init_joint6[4])
        if not self._execute_arm_to_chunked(
            step2,
            gripper=grip_cmd,
            head=head_cmd,
            timeout_s=max(2.0, float(timeout_s) * 0.4),
            reliable=False,
        ):
            return False

        step3 = list(init_joint6)
        step3[0] = float(target_base_x)
        return bool(
            self._execute_arm_to_chunked(
                step3,
                gripper=grip_cmd,
                head=head_cmd,
                timeout_s=max(2.0, float(timeout_s) * 0.4),
                reliable=False,
            )
        )

    def _park_pitch_for_camera_view(self, *, timeout_s: float = ACTION_MOVE_TIMEOUT_DEFAULT_S) -> bool:
        cur = self._current_manip_joint6()
        if not (isinstance(cur, list) and len(cur) >= 6):
            cur = self._default_init_joint6()
        cur = [float(v) for v in cur[:6]]
        cur[4] = float(
            np.clip(
                float(CAMERA_VIEW_PARK_PITCH_RAD),
                float(self.ros_node.JOINT_LIMITS[3][0]),
                float(self.ros_node.JOINT_LIMITS[3][1]),
            )
        )
        q8 = self._default_init_q8()
        g = self._get_manual_gripper_target(fallback=float(q8[7]))
        if g is None:
            g = float(q8[7])
        return bool(
            self._execute_arm_to_chunked(
                cur,
                gripper=float(g),
                head=[float(q8[5]), float(q8[6])],
                timeout_s=float(timeout_s),
                reliable=False,
            )
        )

    def _park_lift_for_camera_view(self, *, timeout_s: float = ACTION_MOVE_TIMEOUT_DEFAULT_S) -> bool:
        cur = self._current_manip_joint6()
        if not (isinstance(cur, list) and len(cur) >= 6):
            cur = self._default_init_joint6()
        cur = [float(v) for v in cur[:6]]
        cur[1] = float(
            np.clip(
                float(CAMERA_VIEW_PARK_LIFT_M),
                float(self.ros_node.JOINT_LIMITS[1][0]),
                float(self.ros_node.JOINT_LIMITS[1][1]),
            )
        )
        q8 = self._default_init_q8()
        g = self._get_manual_gripper_target(fallback=float(q8[7]))
        if g is None:
            g = float(q8[7])
        return bool(
            self._execute_arm_to_chunked(
                cur,
                gripper=float(g),
                head=[float(q8[5]), float(q8[6])],
                timeout_s=float(timeout_s),
                reliable=False,
            )
        )

    def _park_camera_view_pose(self, *, timeout_s: float = ACTION_MOVE_TIMEOUT_LONG_S) -> bool:
        if not bool(ENABLE_CAMERA_VIEW_PARK_MOVE):
            print("[park] camera-view park move skipped (ENABLE_CAMERA_VIEW_PARK_MOVE=False).")
            return True
        # Park order requested:
        # 1) pitch -> CAMERA_VIEW_PARK_PITCH_RAD
        # 2) lift -> CAMERA_VIEW_PARK_LIFT_M
        ok1 = self._park_pitch_for_camera_view(timeout_s=max(2.0, float(timeout_s) * 0.5))
        if not ok1:
            return False
        ok2 = self._park_lift_for_camera_view(timeout_s=max(2.0, float(timeout_s) * 0.5))
        return bool(ok2)

    def _execute_arm_to_chunked(
        self,
        target_joint6: list[float],
        *,
        gripper: float | None = None,
        head: list[float] | None = None,
        timeout_s: float = ACTION_MOVE_TIMEOUT_DEFAULT_S,
        reliable: bool = False,
        block_until_reached: bool = True,
        refresh_after_send: bool = True,
        inter_step_delay_s: float | None = None,
    ) -> bool:
        """Execute arm_to in small steps using the same delay as manual speed slider."""
        target = np.asarray(target_joint6, dtype=np.float32).reshape(-1)
        if target.shape[0] < 6:
            return False
        target = target[:6].copy()

        # Clip target into bridge/robot limits.
        target[0] = float(np.clip(target[0], float(MANIP_BASE_X_LIMITS[0]), float(MANIP_BASE_X_LIMITS[1])))
        target[1] = float(np.clip(target[1], float(self.ros_node.JOINT_LIMITS[1][0]), float(self.ros_node.JOINT_LIMITS[1][1])))
        target[2] = float(np.clip(target[2], float(self.ros_node.JOINT_LIMITS[0][0]), float(self.ros_node.JOINT_LIMITS[0][1])))
        target[3] = float(np.clip(target[3], float(self.ros_node.JOINT_LIMITS[2][0]), float(self.ros_node.JOINT_LIMITS[2][1])))
        target[4] = float(np.clip(target[4], float(self.ros_node.JOINT_LIMITS[3][0]), float(self.ros_node.JOINT_LIMITS[3][1])))
        target[5] = float(np.clip(target[5], float(self.ros_node.JOINT_LIMITS[4][0]), float(self.ros_node.JOINT_LIMITS[4][1])))

        current = self._current_manip_joint6()
        if not (isinstance(current, list) and len(current) >= 6):
            current = [float(v) for v in target.tolist()]
        current = np.asarray(current[:6], dtype=np.float32).reshape(-1)
        if current.shape[0] < 6 or not np.all(np.isfinite(current)):
            current = target.copy()

        grip_cmd = None
        grip_hold_cmd = None
        grip_start_cmd = None
        run_gripper_ramp_after_motion = False
        if gripper is not None:
            grip_cmd = float(
                np.clip(
                    float(gripper),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            g_now = None
            q_target_now = self.ros_node.get_target_qpos()
            if isinstance(q_target_now, list) and len(q_target_now) >= 8:
                try:
                    g_now = float(q_target_now[7])
                    if not math.isfinite(g_now):
                        g_now = None
                except Exception:
                    g_now = None
            if not (isinstance(g_now, (int, float)) and math.isfinite(float(g_now))):
                q_actual_now = self.ros_node.get_actual_qpos()
                if isinstance(q_actual_now, list) and len(q_actual_now) >= 8:
                    try:
                        g_now = float(q_actual_now[7])
                        if not math.isfinite(g_now):
                            g_now = None
                    except Exception:
                        g_now = None
            if not (isinstance(g_now, (int, float)) and math.isfinite(float(g_now))):
                try:
                    g_now = self._get_manual_gripper_target(fallback=None)
                except Exception:
                    g_now = None
            if not (isinstance(g_now, (int, float)) and math.isfinite(float(g_now))):
                g_now = float(grip_cmd)
            grip_start_cmd = float(
                np.clip(
                    float(g_now),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            grip_hold_cmd = float(grip_start_cmd)
            grip_delta = abs(float(grip_cmd) - float(grip_start_cmd))
            grip_tol = max(1e-4, float(abs(getattr(self, "gripper_step", DEVICE_GRIPPER_STEP))) * 0.25)
            run_gripper_ramp_after_motion = bool(grip_delta > grip_tol)

        head_cmd = None
        if isinstance(head, list) and len(head) >= 2:
            head_cmd = [
                float(np.clip(float(head[0]), float(self.ros_node.JOINT_LIMITS[5][0]), float(self.ros_node.JOINT_LIMITS[5][1]))),
                float(np.clip(float(head[1]), float(self.ros_node.JOINT_LIMITS[6][0]), float(self.ros_node.JOINT_LIMITS[6][1]))),
            ]

        # Match manual stepping semantics: same per-group step sizes and same delay slider.
        arm_step = max(0.001, abs(float(self.arm_speed)))
        wrist_step = max(0.002, abs(float(self.wrist_speed)))
        # Base_x is sent directly (no chunking); only lift/arm/wrist are chunked.
        # joint6 ordering: [base_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll]
        step_limits_nonbase = np.asarray([arm_step, arm_step, wrist_step, wrist_step, wrist_step], dtype=np.float32)
        delta_nonbase = target[1:] - current[1:]
        ratios = np.abs(delta_nonbase) / np.maximum(step_limits_nonbase, 1e-6)
        n_steps = int(max(1, int(np.ceil(float(np.max(ratios))))))
        if inter_step_delay_s is None:
            delay_s = float(np.clip(float(self.command_smoothing_delay), 0.01, 0.5))
        else:
            delay_s = max(0.0, float(inter_step_delay_s))

        for i in range(1, n_steps + 1):
            if self._is_abort_requested():
                with self._action_lock:
                    st = self._action_state
                # During explicit return-to-start, state is idle but abort flag may
                # still be latched from the prior action abort request.
                if st != "idle":
                    return False
            u = float(i) / float(n_steps)
            # Ease-in/ease-out profile:
            # small step at start, larger in middle, small near end.
            alpha = 0.5 - 0.5 * math.cos(math.pi * u)
            step_joint = target.copy()
            if n_steps > 1:
                step_joint[1:] = (current[1:] + delta_nonbase * alpha).astype(np.float32)
            # Directly command IK base_x target each send (no base_x chunking).
            step_joint[0] = float(target[0])
            is_last = i == n_steps
            # If gripper must ramp after arm motion, force final arm step blocking so
            # gripper starts only after other joints have reached the target.
            should_block = bool(is_last and (bool(block_until_reached) or bool(run_gripper_ramp_after_motion)))
            pre_stamp_ns = None
            if bool(refresh_after_send) and hasattr(self.ros_node, "_latest_ros_observation_stamp_ns"):
                try:
                    pre_stamp_ns = self.ros_node._latest_ros_observation_stamp_ns()
                except Exception:
                    pre_stamp_ns = None
            ok = self.ros_node.execute_arm_to(
                step_joint[:6].tolist(),
                gripper=(grip_hold_cmd if bool(run_gripper_ramp_after_motion) else grip_cmd),
                head=head_cmd,
                # Stream intermediate chunks quickly; only final chunk blocks/checks reached.
                blocking=bool(should_block),
                timeout_s=(
                    float(timeout_s)
                    if should_block
                    else max(
                        float(EXECUTE_ARM_TO_INTERMEDIATE_TIMEOUT_MIN_S),
                        min(float(EXECUTE_ARM_TO_INTERMEDIATE_TIMEOUT_MAX_S), float(timeout_s)),
                    )
                ),
                reliable=bool(reliable),
            )
            if not ok:
                return False
            if bool(refresh_after_send) and hasattr(self.ros_node, "_refresh_state_from_ros_topic"):
                try:
                    self.ros_node._refresh_state_from_ros_topic(
                        require_newer_than_ns=pre_stamp_ns,
                        timeout_s=(
                            float(EXECUTE_ARM_TO_REFRESH_TIMEOUT_LAST_S)
                            if should_block
                            else float(EXECUTE_ARM_TO_REFRESH_TIMEOUT_INTERMEDIATE_S)
                        ),
                    )
                except Exception:
                    pass
            if i < n_steps and delay_s > 0.0:
                time.sleep(delay_s)

        if bool(run_gripper_ramp_after_motion) and isinstance(grip_cmd, float) and isinstance(grip_start_cmd, float):
            if self._is_abort_requested():
                with self._action_lock:
                    st = self._action_state
                if st != "idle":
                    return False
            grip_step = float(abs(getattr(self, "gripper_step", DEVICE_GRIPPER_STEP)))
            grip_step = max(1e-4, grip_step)
            grip_delta = float(grip_cmd - grip_start_cmd)
            n_grip_steps = int(max(1, int(math.ceil(abs(float(grip_delta)) / float(grip_step)))))
            for gi in range(1, n_grip_steps + 1):
                if self._is_abort_requested():
                    with self._action_lock:
                        st = self._action_state
                    if st != "idle":
                        return False
                frac = float(gi) / float(n_grip_steps)
                g_step = float(
                    np.clip(
                        float(grip_start_cmd + grip_delta * frac),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                is_last_grip = bool(gi == n_grip_steps)
                pre_stamp_ns = None
                if bool(refresh_after_send) and hasattr(self.ros_node, "_latest_ros_observation_stamp_ns"):
                    try:
                        pre_stamp_ns = self.ros_node._latest_ros_observation_stamp_ns()
                    except Exception:
                        pre_stamp_ns = None
                ok = self.ros_node.execute_arm_to(
                    [float(v) for v in target[:6].tolist()],
                    gripper=float(g_step),
                    head=head_cmd,
                    blocking=bool(is_last_grip and bool(block_until_reached)),
                    timeout_s=(
                        float(timeout_s)
                        if bool(is_last_grip and bool(block_until_reached))
                        else max(
                            float(EXECUTE_ARM_TO_INTERMEDIATE_TIMEOUT_MIN_S),
                            min(float(EXECUTE_ARM_TO_INTERMEDIATE_TIMEOUT_MAX_S), float(timeout_s)),
                        )
                    ),
                    reliable=bool(reliable),
                )
                if not ok:
                    return False
                if bool(refresh_after_send) and hasattr(self.ros_node, "_refresh_state_from_ros_topic"):
                    try:
                        self.ros_node._refresh_state_from_ros_topic(
                            require_newer_than_ns=pre_stamp_ns,
                            timeout_s=(
                                float(EXECUTE_ARM_TO_REFRESH_TIMEOUT_LAST_S)
                                if bool(is_last_grip and bool(block_until_reached))
                                else float(EXECUTE_ARM_TO_REFRESH_TIMEOUT_INTERMEDIATE_S)
                            ),
                        )
                    except Exception:
                        pass
                if (not is_last_grip) and delay_s > 0.0:
                    time.sleep(delay_s)
        if isinstance(grip_cmd, float):
            # Keep UI-side command memory aligned so subsequent arm stages do not
            # restart gripper ramps from stale cached values.
            self._set_manual_gripper_override(float(grip_cmd))
        return True

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
        wrist_pitch_target: float | None = None,
        wrist_roll_target: float | None = None,
        grasp_shape_info: dict[str, Any] | None = None,
        preserve_existing_pre_action_state: bool = False,
        target_world_xyz_override: tuple[float, float, float] | list[float] | None = None,
        preplanned_grasp_joint6: list[float] | tuple[float, ...] | np.ndarray | None = None,
        post_grasp_lift: bool = True,
        grasp_rotate_deg: float | None = None,
        precise_grasp: bool = False,
        precise_place: bool = False,
    ) -> None:
        """v6 approach: use stretch_ai IK/open-loop planning instead of manual geometry."""
        import time
        mode_s = str(mode)
        mode_is_grasp = bool(mode_s == "grasp")
        mode_is_place_object = bool(mode_s == "place_object")
        mode_is_reach_like = bool(mode_s in ("reach", "place_object"))
        mode_is_precise_grasp = bool(mode_is_grasp and bool(precise_grasp))
        mode_is_precise_place = bool(mode_is_reach_like and bool(precise_place))
        rotate_after_grasp_deg = (
            float(grasp_rotate_deg)
            if isinstance(grasp_rotate_deg, (int, float)) and math.isfinite(float(grasp_rotate_deg))
            else 0.0
        )
        force_pitch_target_rad = (
            float(wrist_pitch_target)
            if isinstance(wrist_pitch_target, (int, float)) and math.isfinite(float(wrist_pitch_target))
            else None
        )
        if isinstance(force_pitch_target_rad, float):
            requested_pitch_rad = float(force_pitch_target_rad)
            force_pitch_target_rad = float(
                np.clip(
                    float(requested_pitch_rad),
                    float(self.ros_node.JOINT_LIMITS[3][0]),
                    float(self.ros_node.JOINT_LIMITS[3][1]),
                )
            )
            if abs(float(force_pitch_target_rad) - float(requested_pitch_rad)) > 1e-6:
                print(
                    "[IK pitch clip] "
                    f"requested={math.degrees(requested_pitch_rad):+.1f}deg -> "
                    f"clipped={math.degrees(force_pitch_target_rad):+.1f}deg "
                    f"(limits=[{math.degrees(float(self.ros_node.JOINT_LIMITS[3][0])):+.1f}, "
                    f"{math.degrees(float(self.ros_node.JOINT_LIMITS[3][1])):+.1f}] deg)"
                )
        mode_is_grasp_rotate = bool(
            bool(mode_is_grasp)
            and (not bool(post_grasp_lift))
            and abs(float(rotate_after_grasp_deg)) > 1e-6
        )
        place_object_close_joint = float(
            np.clip(
                float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT),
                float(self.ros_node.JOINT_LIMITS[7][0]),
                float(self.ros_node.JOINT_LIMITS[7][1]),
            )
        )
        # Optional fast path for queued grasp. By default we keep confirmation
        # pauses so users can adjust before grasp/release.
        auto_replay_fast_mode = bool(
            bool(getattr(self, "_auto_sequence_replay_active", False))
            and bool(getattr(self, "_auto_loop_running", False))
            and (not bool(getattr(self, "_auto_first_trial_pending", False)))
        )
        capture_first_trial_pose = bool(
            bool(getattr(self, "_auto_capture_enabled", False))
            or (
                bool(getattr(self, "_auto_loop_mode", "") == "goal_sequence")
                and bool(getattr(self, "_auto_first_trial_pending", False))
            )
        )
        fast_grasp_queue_mode = bool(
            bool(mode_is_grasp)
            and (
                bool(auto_replay_fast_mode)
                or (
                    bool(getattr(self, "_run_all_queued_goals", False))
                    and (not bool(getattr(self, "_auto_first_trial_pending", False)))
                    and (not bool(QUEUE_REQUIRE_GRASP_CONFIRM))
                )
            )
        )
        if bool(mode_is_precise_grasp) or bool(mode_is_precise_place):
            auto_replay_fast_mode = False
            fast_grasp_queue_mode = False

        def _wait_with_pause(duration_s: float) -> bool:
            duration_s = min(float(SCRIPT_STAGE_WAIT_CAP_S), max(0.0, float(duration_s)))
            end_t = time.time() + duration_s
            while time.time() < end_t:
                if self._is_abort_requested():
                    return False
                with self._action_lock:
                    st = self._action_state
                if st in ("paused", "awaiting_confirm", "awaiting_post_grasp", "awaiting_post_reach_release"):
                    time.sleep(0.05)
                    continue
                time.sleep(min(0.05, max(0.0, end_t - time.time())))
            return not self._is_abort_requested()

        def _stage_wait(duration_s: float) -> bool:
            if bool(fast_grasp_queue_mode):
                return not self._is_abort_requested()
            return _wait_with_pause(duration_s)

        def _wait_until_running() -> bool:
            while True:
                if self._is_abort_requested():
                    return False
                with self._action_lock:
                    st = self._action_state
                if st == "running":
                    return True
                time.sleep(0.05)

        def _get_live_robot_gripper_value(fallback: float | None = None) -> float:
            lo = float(self.ros_node.JOINT_LIMITS[7][0])
            hi = float(self.ros_node.JOINT_LIMITS[7][1])

            # Prefer target command at Continue time (what operator set),
            # then measured actual, and only then manual override cache.
            target = self.ros_node.get_target_qpos()
            if isinstance(target, list) and len(target) >= 8:
                try:
                    v = float(target[7])
                    if math.isfinite(v):
                        return float(np.clip(v, lo, hi))
                except (TypeError, ValueError):
                    pass

            actual = self.ros_node.get_actual_qpos()
            if isinstance(actual, list) and len(actual) >= 8:
                try:
                    v = float(actual[7])
                    if math.isfinite(v):
                        return float(np.clip(v, lo, hi))
                except (TypeError, ValueError):
                    pass

            g_manual = self._get_manual_gripper_target(fallback=None)
            if isinstance(g_manual, (int, float)) and math.isfinite(float(g_manual)):
                return float(np.clip(float(g_manual), lo, hi))

            if not isinstance(fallback, (int, float)) or (not math.isfinite(float(fallback))):
                fallback = float(hi)
            return float(np.clip(float(fallback), lo, hi))

        def _wait_for_base_x_target(
            target_base_x: float,
            tol_m: float = ACTION_BASE_X_SETTLE_TOL_M,
            timeout_s: float = ACTION_BASE_X_SETTLE_TIMEOUT_S,
        ) -> bool:
            """Wait until manipulation base_x reaches target (within tolerance)."""
            end_t = time.time() + max(0.1, float(timeout_s))
            target = float(target_base_x)
            tol = abs(float(tol_m))
            while time.time() < end_t:
                if self._is_abort_requested():
                    return False
                with self._action_lock:
                    st = self._action_state
                if st in ("paused", "awaiting_confirm", "awaiting_post_grasp", "awaiting_post_reach_release"):
                    time.sleep(0.05)
                    continue
                cur = self._current_manip_joint6()
                if isinstance(cur, list) and len(cur) >= 1:
                    try:
                        cur_x = float(cur[0])
                        if math.isfinite(cur_x) and abs(cur_x - target) <= tol:
                            return True
                    except (TypeError, ValueError):
                        pass
                time.sleep(0.05)
            return False

        def _abort_and_return() -> None:
            if self._consume_skip_to_next_goal_request():
                print("Action aborted by user. Skipping to next queued goal...")
                self._set_action_state("idle")
                return
            user_abort_requested = self._is_abort_requested()
            self._set_action_state("idle")

            # If user explicitly requested abort/return while action was active,
            # keep existing behavior and auto-return to saved start.
            if user_abort_requested:
                print("Action aborted by user. Returning to start...")
                if self._pre_action_state is not None:
                    self.return_to_start()
                return

            # Failure path: hold current pose, do not auto-return.
            # Keep Return enabled so user can decide when to go back.
            print("Action failed. Holding current pose; use Return to go back.")
            self.ros_node.sync_command_targets_to_actual()
            self._set_return_enabled(True)
            self._set_status(
                "Action failed. Holding position. Adjust manually or press Return.",
                "QLabel { color: orange; font-size: 10px; }",
            )

        # For a new action start, first un-park from camera-view pitch by restoring
        # the configured startup/default joint pose.
        if (not preserve_existing_pre_action_state) or (self._pre_action_state is None):
            self._set_status(
                "IK pipeline: moving to startup action pose before planning...",
                "QLabel { color: blue; font-size: 10px; }",
            )
            startup_gripper_override = None
            if bool(mode_is_place_object):
                # Place-object should remain closed until the target release.
                startup_gripper_override = float(place_object_close_joint)
            if not self._move_to_startup_pose_for_action(
                timeout_s=12.0,
                gripper_override=startup_gripper_override,
            ):
                self._set_status("IK pipeline: startup pose move failed", "QLabel { color: red; font-size: 10px; }")
                _abort_and_return()
                return
            if not _stage_wait(0.25):
                _abort_and_return()
                return

        if isinstance(grasp_yaw, (int, float)) and math.isfinite(float(grasp_yaw)):
            yaw_in = float(grasp_yaw)
            yaw_clipped = self._clip_ik_wrist_yaw_around_init(yaw_in)
            if abs(float(yaw_clipped) - float(yaw_in)) > 1e-6:
                print(
                    "[IK yaw clip] "
                    f"input={math.degrees(yaw_in):+.1f}deg -> clipped={math.degrees(yaw_clipped):+.1f}deg "
                    f"(init±{IK_WRIST_YAW_CLIP_AROUND_INIT_DEG:.1f}deg)"
                )
            grasp_yaw = float(yaw_clipped)

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
        target_world_xyz = None
        if isinstance(target_world_xyz_override, (list, tuple)) and len(target_world_xyz_override) >= 3:
            try:
                tx = float(target_world_xyz_override[0])
                ty = float(target_world_xyz_override[1])
                tz = float(target_world_xyz_override[2])
                if math.isfinite(tx) and math.isfinite(ty) and math.isfinite(tz):
                    target_world_xyz = (tx, ty, tz)
            except Exception:
                target_world_xyz = None
        if target_world_xyz is None:
            base_pose = self.ros_node.get_measured_base_pose_xytheta()
            if not (isinstance(base_pose, list) and len(base_pose) >= 3):
                base_pose = [0.0, 0.0, 0.0]
            target_world_xyz = self._base_point_to_odom_xyz(target_base_xyz, base_pose)

        print(f"[IK pipeline] mode={mode} target_base={target_base_xyz} target_world={target_world_xyz}")
        self._freeze_streaming_commands_to_current_state()

        # Grasp+rotate mode requirement: enforce wrist pitch target first
        # (blocking) before any other scripted approach movement.
        if bool(mode_is_grasp_rotate) and isinstance(force_pitch_target_rad, float):
            pre_align_joint = self._current_manip_joint6()
            if not (isinstance(pre_align_joint, list) and len(pre_align_joint) >= 6):
                pre_align_joint = self._default_init_joint6()
            pre_align_joint = [float(v) for v in pre_align_joint[:6]]
            pitch_err = float(force_pitch_target_rad) - float(pre_align_joint[4])
            if abs(float(pitch_err)) > float(np.deg2rad(0.5)):
                pre_align_joint[4] = float(force_pitch_target_rad)
                self._set_status(
                    f"Grasp+rotate: pre-align wrist pitch to {float(math.degrees(force_pitch_target_rad)):+.1f}deg...",
                    "QLabel { color: blue; font-size: 10px; }",
                )
                if not self._execute_arm_to_chunked(
                    pre_align_joint[:6],
                    gripper=None,
                    timeout_s=8.0,
                    reliable=True,
                ):
                    self._set_status(
                        "IK pipeline: wrist-pitch pre-align failed",
                        "QLabel { color: red; font-size: 10px; }",
                    )
                    _abort_and_return()
                    return
                if not _stage_wait(0.1):
                    _abort_and_return()
                    return

        # Safety pre-step: ensure minimum lift before any IK planning/execution.
        # This is intentionally done before plan_open_loop_grasp().
        min_lift_m = float(IK_SAFE_LIFT_M)
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
                if not self._execute_arm_to_chunked(
                    precheck_joint[:6],
                    gripper=None,
                    timeout_s=8.0,
                    reliable=False,
                ):
                    self._set_status(
                        f"IK pipeline: pre-lift to {min_lift_m:.2f}m failed",
                        "QLabel { color: red; font-size: 10px; }",
                    )
                    _abort_and_return()
                    return
                if not _stage_wait(0.2):
                    _abort_and_return()
                    return

        # Step 1: plan IK targets directly from the clicked world-frame point.
        # No manual pre-rotation/pre-lift: execute exactly what IK returns.
        grasp_joint = None
        if bool(mode_is_reach_like) and preplanned_grasp_joint6 is not None:
            try:
                pre = np.asarray(preplanned_grasp_joint6, dtype=np.float32).reshape(-1)
                if pre.shape[0] >= 6 and np.all(np.isfinite(pre[:6])):
                    grasp_joint = [float(v) for v in pre[:6].tolist()]
            except Exception:
                grasp_joint = None
        if grasp_joint is not None:
            self._set_status(
                "IK pipeline: using precomputed reach target...",
                "QLabel { color: blue; font-size: 10px; }",
            )
        else:
            self._set_status("IK pipeline: planning pregrasp/grasp targets...", "QLabel { color: blue; font-size: 10px; }")
            plan = self.ros_node.plan_open_loop_grasp(
                target_world_xyz,
                pregrasp_distance=IK_PREGRASP_DISTANCE_M if bool(mode_is_grasp) else 0.20,
                lift_distance=IK_LIFT_DISTANCE_M,
                wrist_yaw_target=(None if grasp_yaw is None else float(grasp_yaw)),
                wrist_pitch_target=(
                    float(force_pitch_target_rad)
                    if force_pitch_target_rad is not None
                    else float(np.deg2rad(GRASP_PITCH_DEG))
                ),
                wrist_roll_target=(None if wrist_roll_target is None else float(wrist_roll_target)),
                timeout_s=35.0,
            )
            if isinstance(plan, dict):
                err0 = str(plan.get("error", ""))
                if (not plan.get("ok", False)) and ("Timeout waiting for worker response to 'plan_open_loop_grasp'" in err0):
                    self._set_status("IK pipeline: planner timeout; retrying...", "QLabel { color: orange; font-size: 10px; }")
                    plan = self.ros_node.plan_open_loop_grasp(
                        target_world_xyz,
                        pregrasp_distance=IK_PREGRASP_DISTANCE_M if bool(mode_is_grasp) else 0.20,
                        lift_distance=IK_LIFT_DISTANCE_M,
                        wrist_yaw_target=(None if grasp_yaw is None else float(grasp_yaw)),
                        wrist_pitch_target=(
                            float(force_pitch_target_rad)
                            if force_pitch_target_rad is not None
                            else float(np.deg2rad(GRASP_PITCH_DEG))
                        ),
                        wrist_roll_target=(None if wrist_roll_target is None else float(wrist_roll_target)),
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
        if bool(mode_is_grasp) and math.isfinite(float(target_lift)):
            self._last_grasp_target_lift_m = float(target_lift)
        target_lift_plus_margin = float(
            np.clip(
                # max(float(IK_SAFE_LIFT_M), float(target_lift) + IK_REACH_STANDOFF_M),
                float(target_lift) + IK_REACH_STANDOFF_M,
                float(self.ros_node.JOINT_LIMITS[1][0]),
                float(self.ros_node.JOINT_LIMITS[1][1]),
            )
        )
        approach_joint = [float(v) for v in grasp_joint[:6]]
        # For forced-pitch grasp modes (e.g., grasp+rotate), do not let planner
        # output drift wrist pitch during approach transit.
        if bool(mode_is_grasp) and isinstance(force_pitch_target_rad, float):
            approach_joint[4] = float(
                np.clip(
                    float(force_pitch_target_rad),
                    float(self.ros_node.JOINT_LIMITS[3][0]),
                    float(self.ros_node.JOINT_LIMITS[3][1]),
                )
            )
        # Apply calibration offset only for grasp mode.
        # Reach mode should use IK base_x directly.
        base_x_cmd = float(approach_joint[0])
        if bool(mode_is_grasp):
            base_x_cmd = float(base_x_cmd + float(IK_GRASP_BASE_X_OFFSET_M))
        approach_joint[0] = float(
            np.clip(
                float(base_x_cmd),
                float(MANIP_BASE_X_LIMITS[0]),
                float(MANIP_BASE_X_LIMITS[1]),
            )
        )
        grasp_strategy = "lift_standoff"
        if bool(mode_is_grasp) and isinstance(grasp_shape_info, dict):
            grasp_strategy = str(grasp_shape_info.get("approach_strategy", "lift_standoff"))

        # Reach-specific two-stage lift behavior:
        # 1) approach with lift held >=0.9m while base_x/arm settle
        # 2) only then adjust to (target_lift + 0.1m)
        reach_lift_current = float(IK_SAFE_LIFT_M)
        cur_joint6_for_reach = self._current_manip_joint6()
        if isinstance(cur_joint6_for_reach, list) and len(cur_joint6_for_reach) >= 2:
            try:
                reach_lift_current = float(cur_joint6_for_reach[1])
            except (TypeError, ValueError):
                reach_lift_current = float(IK_SAFE_LIFT_M)
        reach_lift_stage1 = float(
            np.clip(
                max(float(IK_SAFE_LIFT_M), float(reach_lift_current)),
                # float(reach_lift_current),
                float(self.ros_node.JOINT_LIMITS[1][0]),
                float(self.ros_node.JOINT_LIMITS[1][1]),
            )
        )
        if bool(mode_is_reach_like):
            approach_joint[1] = float(reach_lift_stage1)
        elif bool(mode_is_grasp) and grasp_strategy == "reach_standoff":
            # Vertical-face style grasp: keep target lift and pause at reach-10cm.
            approach_joint[1] = float(
                np.clip(
                    # max(float(IK_SAFE_LIFT_M), float(target_lift)),
                    float(target_lift),
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )
            approach_joint[2] = float(
                np.clip(
                    float(grasp_joint[2]) - float(IK_REACH_STANDOFF_M),
                    float(self.ros_node.JOINT_LIMITS[0][0]),
                    float(self.ros_node.JOINT_LIMITS[0][1]),
                )
            )
        else:
            approach_joint[1] = float(target_lift_plus_margin)

        # Step 2: move to grasp pose with lift hold and wait for user verification.
        if bool(mode_is_grasp) and grasp_strategy == "reach_standoff":
            self._set_status(
                f"IK pipeline: moving to reach standoff (-{IK_REACH_STANDOFF_M*100:.0f}cm)...",
                "QLabel { color: blue; font-size: 10px; }",
            )
        else:
            self._set_status("IK pipeline: moving to grasp pose (+10cm lift hold)...", "QLabel { color: blue; font-size: 10px; }")
        gripper_width_joint = None
        if bool(mode_is_grasp):
            if isinstance(gripper_width, (int, float)) and np.isfinite(float(gripper_width)):
                gripper_width_joint = float(
                    np.clip(
                        float(gripper_width),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                gripper_open = float(gripper_width_joint)
            else:
                gripper_open = float(self.ros_node.JOINT_LIMITS[7][1])
            # Initialize manual override tracking from planner open width.
            self._set_manual_gripper_override(gripper_open)
        elif bool(mode_is_place_object):
            # Keep closed while approaching place target.
            gripper_open = float(place_object_close_joint)
            self._set_manual_gripper_override(gripper_open)
        else:
            gripper_open = None

        if not self._execute_arm_to_chunked(
            approach_joint[:6],
            gripper=gripper_open,
            timeout_s=8.0,
            reliable=False,
        ):
            self._set_status("IK pipeline: approach move failed", "QLabel { color: red; font-size: 10px; }")
            _abort_and_return()
            return
        if not _stage_wait(0.3):
            _abort_and_return()
            return

        if bool(mode_is_reach_like):
            stage2_base_x = float(approach_joint[0])
            # Stage-2: wait until base_x settles, then adjust lift to target_lift+0.1m.
            self._set_status(
                "Reach: waiting for base_x to reach target before lowering lift...",
                "QLabel { color: blue; font-size: 10px; }",
            )
            if not _wait_for_base_x_target(
                float(stage2_base_x),
                tol_m=float(ACTION_BASE_X_SETTLE_TOL_M),
                timeout_s=float(ACTION_BASE_X_SETTLE_TIMEOUT_S),
            ):
                self._set_status(
                    "IK pipeline: base_x did not settle at target; skipping lift-down",
                    "QLabel { color: red; font-size: 10px; }",
                )
                _abort_and_return()
                return
            self._set_status(
                f"Reach: adjusting lift from {reach_lift_stage1:.2f}m to target+0.10m...",
                "QLabel { color: blue; font-size: 10px; }",
            )
            # Do not replay planned base_x here. Only adjust lift from the *current*
            # pose so manual base edits are preserved.
            reach_lower_joint = self._current_manip_joint6()
            if not (isinstance(reach_lower_joint, list) and len(reach_lower_joint) >= 6):
                reach_lower_joint = [float(v) for v in approach_joint[:6]]
            reach_lower_joint = [float(v) for v in reach_lower_joint[:6]]
            reach_lower_joint[1] = float(target_lift_plus_margin)
            reach_stage2_gripper = float(place_object_close_joint) if bool(mode_is_place_object) else None
            if not self._execute_arm_to_chunked(
                reach_lower_joint[:6],
                gripper=reach_stage2_gripper,
                timeout_s=8.0,
                reliable=False,
            ):
                self._set_status("IK pipeline: reach lift adjust failed", "QLabel { color: red; font-size: 10px; }")
                _abort_and_return()
                return
            if not _wait_with_pause(0.3):
                _abort_and_return()
                return
            # Hold at reach until user confirms placement tweaks, then Continue will:
            # 1) move lift to (grasp_lift + 0.5cm), 2) open gripper, 3) enable Return.
            self.ros_node.sync_command_targets_to_actual()
            require_post_reach_continue = not bool(auto_replay_fast_mode)
            self._set_return_enabled(False)
            if bool(require_post_reach_continue):
                if bool(mode_is_precise_place):
                    self._set_status(
                        "Place precise hold ready. Place manually, then press Continue.",
                        "QLabel { color: orange; font-size: 10px; }",
                    )
                else:
                    self._set_status(
                        "Reach hold ready. Adjust manually, then press Continue to release.",
                        "QLabel { color: orange; font-size: 10px; }",
                    )
                self._set_action_state("awaiting_post_reach_release")
                if not _wait_until_running():
                    _abort_and_return()
                    return
            else:
                self._set_status(
                    "Auto replay: continuing from reach hold to release.",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )

            if bool(mode_is_precise_place):
                # Precise place mode: user manually performs final placement/release
                # during the hold. Continue should skip scripted release and proceed
                # to normal post-place flow (direct return in single-goal use).
                self.ros_node.sync_command_targets_to_actual()
                if capture_first_trial_pose:
                    snap_reach = self._capture_current_pose_snapshot("reach_final_precise")
                    if isinstance(snap_reach, dict):
                        tgt_g = None
                        act_g = None
                        man_g = None
                        q_t = self.ros_node.get_target_qpos()
                        if isinstance(q_t, list) and len(q_t) >= 8:
                            try:
                                vv = float(q_t[7])
                                if math.isfinite(vv):
                                    tgt_g = float(vv)
                            except Exception:
                                tgt_g = None
                        q_a = self.ros_node.get_actual_qpos()
                        if isinstance(q_a, list) and len(q_a) >= 8:
                            try:
                                vv = float(q_a[7])
                                if math.isfinite(vv):
                                    act_g = float(vv)
                            except Exception:
                                act_g = None
                        try:
                            vv = self._get_manual_gripper_target(fallback=None)
                            if isinstance(vv, (int, float)) and math.isfinite(float(vv)):
                                man_g = float(vv)
                        except Exception:
                            man_g = None
                        precise_open = _get_live_robot_gripper_value(
                            fallback=float(snap_reach.get("gripper", float(self.ros_node.JOINT_LIMITS[7][1])))
                        )
                        snap_reach["gripper"] = float(precise_open)
                        print(
                            "[precise_place] learned open at Continue: "
                            f"{float(precise_open):+.4f} "
                            f"(target={tgt_g}, actual={act_g}, manual={man_g})"
                        )
                        self._auto_pose_reach = dict(snap_reach)
                        self._auto_pose_reach_target = dict(snap_reach)
                        self._auto_gripper_open = float(precise_open)
                self._set_action_state("idle")
                self._set_return_enabled(True)
                if not bool(self._goal_sequence_has_next()):
                    self._set_status(
                        "Place precise complete. Returning home...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    self.return_to_start()
                else:
                    self._set_status(
                        "Place precise complete. Continuing queued goals...",
                        "QLabel { color: green; font-size: 10px; }",
                    )
                return

            release_lift_base = None
            release_lift_src = "reach_target"
            # Prefer the lift used by the most recent grasp stage.
            if isinstance(self._last_grasp_target_lift_m, (int, float)) and math.isfinite(float(self._last_grasp_target_lift_m)):
                release_lift_base = float(self._last_grasp_target_lift_m)
                release_lift_src = "last_grasp_target_lift"
            elif release_lift_base is None and isinstance(self._auto_pose_grasp_target, dict):
                j6g = self._auto_pose_grasp_target.get("joint6")
                if isinstance(j6g, list) and len(j6g) >= 2:
                    try:
                        release_lift_base = float(j6g[1])
                        release_lift_src = "auto_pose_grasp_target"
                    except (TypeError, ValueError):
                        release_lift_base = None
            elif release_lift_base is None and isinstance(self._auto_pose_grasp, dict):
                j6g = self._auto_pose_grasp.get("joint6")
                if isinstance(j6g, list) and len(j6g) >= 2:
                    try:
                        release_lift_base = float(j6g[1])
                        release_lift_src = "auto_pose_grasp"
                    except (TypeError, ValueError):
                        release_lift_base = None
            elif release_lift_base is None and isinstance(preplanned_grasp_joint6, (list, tuple, np.ndarray)):
                try:
                    pre = np.asarray(preplanned_grasp_joint6, dtype=np.float32).reshape(-1)
                    if pre.shape[0] >= 2 and np.all(np.isfinite(pre[:2])):
                        release_lift_base = float(pre[1])
                        release_lift_src = "preplanned_grasp_joint6"
                except Exception:
                    release_lift_base = None
            if release_lift_base is None:
                release_lift_base = float(target_lift)
                release_lift_src = "reach_target"

            release_lift_target = float(
                np.clip(
                    float(release_lift_base) + 0.0025,
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )
            self._set_status(
                f"Reach release: setting lift to grasp_lift+0.5cm ({release_lift_target:.3f}m)...",
                "QLabel { color: blue; font-size: 10px; }",
            )
            print(
                "[reach_release_lift] "
                f"source={release_lift_src} base={release_lift_base:.3f} target={release_lift_target:.3f}"
            )
            release_joint = self._current_manip_joint6()
            if not (isinstance(release_joint, list) and len(release_joint) >= 6):
                release_joint = [float(v) for v in reach_lower_joint[:6]]
            release_joint = [float(v) for v in release_joint[:6]]
            release_joint[1] = float(release_lift_target)
            if not self._execute_arm_to_chunked(
                release_joint[:6],
                gripper=None,
                timeout_s=8.0,
                reliable=False,
            ):
                self._set_status("Reach release: lift adjust failed", "QLabel { color: red; font-size: 10px; }")
                _abort_and_return()
                return
            if not _wait_with_pause(0.2):
                _abort_and_return()
                return

            gripper_release_open = float(
                np.clip(
                    float(DEVICE_GRIPPER_TOGGLE_OPEN_JOINT),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            self._set_status("Reach release: opening gripper...", "QLabel { color: blue; font-size: 10px; }")
            if not self._execute_arm_to_chunked(
                release_joint[:6],
                gripper=gripper_release_open,
                timeout_s=6.0,
                reliable=False,
            ):
                self._set_status("Reach release: gripper open failed", "QLabel { color: red; font-size: 10px; }")
                _abort_and_return()
                return
            self._set_manual_gripper_override(gripper_release_open)
            if not _wait_with_pause(0.2):
                _abort_and_return()
                return

            if capture_first_trial_pose:
                self._auto_pose_reach_target = {
                    "label": "reach_target",
                    "joint6": [float(v) for v in release_joint[:6]],
                    "gripper": float(gripper_release_open),
                    "head": [float(self.ros_node.qpos[5]), float(self.ros_node.qpos[6])],
                    "base_pose_xytheta": self.ros_node.get_measured_base_pose_xytheta() or [0.0, 0.0, 0.0],
                    "captured_at": float(time.time()),
                }

            # Ensure subsequent manual controls/return start from the measured release pose.
            self.ros_node.sync_command_targets_to_actual()
            if self._auto_capture_enabled:
                snap_reach = self._capture_current_pose_snapshot("reach_final")
                if isinstance(snap_reach, dict):
                    self._auto_pose_reach = snap_reach
                    self._auto_gripper_open = float(snap_reach["gripper"])
                    print(
                        "[auto_loop] captured reach final pose: "
                        f"joint6={self._auto_pose_reach['joint6']}, grip={self._auto_pose_reach['gripper']:.3f}"
                    )

            manual_single_reach_like_first_trial = bool(
                bool(getattr(self, "_auto_loop_mode", "") == "goal_sequence")
                and bool(getattr(self, "_auto_first_trial_pending", False))
                and bool(getattr(self, "_auto_start_after_return", False))
                and (not bool(getattr(self, "_auto_loop_running", False)))
                and bool(getattr(self, "_run_all_queued_goals", False))
                and (not bool(self._goal_sequence_has_next()))
            )
            if bool(manual_single_reach_like_first_trial):
                goals_now = self._goal_sequence_order() if hasattr(self, "_goal_sequence_order") else []
                manual_single_reach_like_first_trial = bool(
                    isinstance(goals_now, list)
                    and len(goals_now) == 1
                    and isinstance(goals_now[0], dict)
                    and str(goals_now[0].get("kind", "")) in ("reach", "place_object")
                )

            if bool(manual_single_reach_like_first_trial):
                self._set_status(
                    f"Reach done. Holding {AUTO_LOOP_GRASP_HOLD_S:.1f}s for manual episode...",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not _wait_with_pause(float(AUTO_LOOP_GRASP_HOLD_S)):
                    _abort_and_return()
                    return
                # End manual type1 segment here so return-home is excluded.
                should_close_manual_segment_now = bool(
                    bool(getattr(self, "_loop_record_session_active", False))
                    and isinstance(getattr(self, "_loop_record_current", None), dict)
                )
                if bool(should_close_manual_segment_now):
                    self._stop_auto_loop_record_segment(save=True)
                self._set_return_enabled(True)
                self._set_action_state("idle")
                self._set_status(
                    "Reach first trial complete. Use Return to start auto replay.",
                    "QLabel { color: green; font-size: 10px; }",
                )
                return

            self._set_return_enabled(True)
            self._set_action_state("idle")
            self._set_status("Reach release complete. Use Return when ready.", "QLabel { color: green; font-size: 10px; }")
            return

        # Sync command targets so manual tweaks during pause start from live robot state.
        self.ros_node.sync_command_targets_to_actual()
        if bool(fast_grasp_queue_mode):
            self._set_status(
                "Fast queued grasp: auto-continuing to lower and close gripper.",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
        else:
            if bool(mode_is_precise_grasp):
                self._set_status(
                    "Grasp precise hold ready. Grasp manually, then press Continue.",
                    "QLabel { color: orange; font-size: 10px; }",
                )
            elif grasp_strategy == "reach_standoff":
                self._set_status(
                    f"At reach standoff (-{IK_REACH_STANDOFF_M*100:.0f}cm). Press Continue to advance and grasp.",
                    "QLabel { color: orange; font-size: 10px; }",
                )
            else:
                self._set_status("At approach hold (+10cm). Press Continue to lower lift and grasp.", "QLabel { color: orange; font-size: 10px; }")
            self._set_action_state("awaiting_confirm")
            if not _wait_until_running():
                _abort_and_return()
                return

        # Step 3:
        # - horizontal/top objects: lower lift to target grasp
        # - vertical-like objects: keep lift and advance arm reach from standoff
        lower_joint = self._current_manip_joint6()
        if not (isinstance(lower_joint, list) and len(lower_joint) >= 6):
            lower_joint = [float(v) for v in approach_joint[:6]]
        lower_joint = [float(v) for v in lower_joint[:6]]
        # Respect live manual wrist edits made during grasp pause/hold by taking the
        # latest commanded wrist orientation from target qpos (if available).
        if bool(mode_is_grasp):
            q_target_live = self.ros_node.get_target_qpos()
            if isinstance(q_target_live, list) and len(q_target_live) >= 5:
                try:
                    yaw_live = float(q_target_live[2])
                    pitch_live = float(q_target_live[3])
                    roll_live = float(q_target_live[4])
                    if math.isfinite(yaw_live):
                        lower_joint[3] = float(
                            np.clip(
                                float(yaw_live),
                                float(self.ros_node.JOINT_LIMITS[2][0]),
                                float(self.ros_node.JOINT_LIMITS[2][1]),
                            )
                        )
                    if (force_pitch_target_rad is None) and math.isfinite(pitch_live):
                        lower_joint[4] = float(
                            np.clip(
                                float(pitch_live),
                                float(self.ros_node.JOINT_LIMITS[3][0]),
                                float(self.ros_node.JOINT_LIMITS[3][1]),
                            )
                        )
                    if math.isfinite(roll_live):
                        lower_joint[5] = float(
                            np.clip(
                                float(roll_live),
                                float(self.ros_node.JOINT_LIMITS[4][0]),
                                float(self.ros_node.JOINT_LIMITS[4][1]),
                            )
                        )
                except Exception:
                    pass
        if isinstance(force_pitch_target_rad, float) and bool(mode_is_grasp):
            lower_joint[4] = float(force_pitch_target_rad)
        allow_return_after_grasp = not bool(self._run_all_queued_goals and self._goal_sequence_has_next())
        gripper_preclose = gripper_open
        if bool(mode_is_grasp):
            # Respect any manual gripper edits made during pause before Continue.
            g_manual = self._get_manual_gripper_target(fallback=gripper_open)
            if g_manual is None:
                g_manual = gripper_open
            gripper_preclose = float(
                np.clip(
                    float(g_manual),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            self._set_manual_gripper_override(gripper_preclose)
            if capture_first_trial_pose:
                self._auto_pose_grasp_target = {
                    "label": "grasp_target",
                    "joint6": [float(v) for v in lower_joint[:6]],
                    "gripper": float(gripper_preclose),
                    "head": [float(self.ros_node.qpos[5]), float(self.ros_node.qpos[6])],
                    "base_pose_xytheta": self.ros_node.get_measured_base_pose_xytheta() or [0.0, 0.0, 0.0],
                    "captured_at": float(time.time()),
                }

        if bool(mode_is_precise_grasp):
            # Precise grasp: user performs final lower+close manually at hold pose.
            self._set_status(
                "Grasp precise: manual grasp accepted. Continuing post-grasp flow...",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            live_joint = self._current_manip_joint6()
            if isinstance(live_joint, list) and len(live_joint) >= 6:
                lower_joint = [float(v) for v in live_joint[:6]]
            tgt_g = None
            act_g = None
            man_g = None
            q_t = self.ros_node.get_target_qpos()
            if isinstance(q_t, list) and len(q_t) >= 8:
                try:
                    vv = float(q_t[7])
                    if math.isfinite(vv):
                        tgt_g = float(vv)
                except Exception:
                    tgt_g = None
            q_a = self.ros_node.get_actual_qpos()
            if isinstance(q_a, list) and len(q_a) >= 8:
                try:
                    vv = float(q_a[7])
                    if math.isfinite(vv):
                        act_g = float(vv)
                except Exception:
                    act_g = None
            try:
                vv = self._get_manual_gripper_target(fallback=None)
                if isinstance(vv, (int, float)) and math.isfinite(float(vv)):
                    man_g = float(vv)
            except Exception:
                man_g = None
            g_live = _get_live_robot_gripper_value(fallback=float(self.ros_node.qpos[7]))
            gripper_closed = float(
                np.clip(
                    float(g_live),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            self._set_manual_gripper_override(float(gripper_closed))
            self._auto_gripper_closed = float(gripper_closed)
            print(
                "[precise_grasp] learned close at Continue: "
                f"{float(gripper_closed):+.4f} "
                f"(target={tgt_g}, actual={act_g}, manual={man_g})"
            )
            if capture_first_trial_pose and isinstance(self._auto_pose_grasp_target, dict):
                self._auto_pose_grasp_target["gripper"] = float(gripper_closed)
            self._device_b1_open_next = True
            self.ros_node.sync_command_targets_to_actual()
            require_post_grasp_continue = False
        else:
            if grasp_strategy == "reach_standoff":
                self._set_status("IK pipeline: advancing reach to target grasp...", "QLabel { color: blue; font-size: 10px; }")
                lower_joint[1] = float(
                    np.clip(
                        float(target_lift),
                        float(self.ros_node.JOINT_LIMITS[1][0]),
                        float(self.ros_node.JOINT_LIMITS[1][1]),
                    )
                )
                lower_joint[2] = float(
                    np.clip(
                        float(grasp_joint[2]),
                        float(self.ros_node.JOINT_LIMITS[0][0]),
                        float(self.ros_node.JOINT_LIMITS[0][1]),
                    )
                )
            else:
                self._set_status("IK pipeline: lowering lift to target grasp...", "QLabel { color: blue; font-size: 10px; }")
                lower_joint[1] = float(
                    np.clip(
                        float(target_lift),
                        float(self.ros_node.JOINT_LIMITS[1][0]),
                        float(self.ros_node.JOINT_LIMITS[1][1]),
                    )
                )

            if not self._execute_arm_to_chunked(
                lower_joint[:6],
                gripper=gripper_preclose,
                timeout_s=8.0,
                reliable=False,
            ):
                self._set_status("IK pipeline: lift lower failed", "QLabel { color: red; font-size: 10px; }")
                _abort_and_return()
                return
            if not _stage_wait(0.3):
                _abort_and_return()
                return

            # Final close target:
            # - default: delta-based close around object width
            # - optional: min(object-width-joint, DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT)
            if (
                bool(IK_GRIPPER_CLOSE_USE_MIN_OBJECT_WIDTH_AND_DEVICE_CLOSE)
                and bool(mode_is_grasp)
                and isinstance(gripper_width_joint, (int, float))
                and math.isfinite(float(gripper_width_joint))
            ):
                device_close_joint = float(
                    np.clip(
                        float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                gripper_closed = float(
                    np.clip(
                        min(float(gripper_width_joint), float(device_close_joint)),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                close_mode = (
                    "min(obj_width,device_close)"
                    f" obj={float(gripper_width_joint):+.3f} device={float(device_close_joint):+.3f}"
                )
            else:
                gripper_delta_joint = float(IK_GRIPPER_CLOSE_DELTA_M) / 0.22
                gripper_closed = float(
                    np.clip(
                        float(gripper_preclose) - float(gripper_delta_joint),
                        float(IK_GRIPPER_CLOSE_MIN_JOINT),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                close_mode = f"delta_m={IK_GRIPPER_CLOSE_DELTA_M:.3f}"
            gripper_closed = float(min(float(gripper_preclose), float(gripper_closed)))
            print(
                f"[IK gripper] open={gripper_preclose:+.3f}, close={gripper_closed:+.3f}, "
                f"mode={close_mode}"
            )
            if not self._execute_arm_to_chunked(
                lower_joint[:6],
                gripper=gripper_closed,
                timeout_s=6.0,
                reliable=False,
            ):
                self._set_status("IK pipeline: gripper close failed", "QLabel { color: red; font-size: 10px; }")
                _abort_and_return()
                return
            # Mark B1 as pressed-state after scripted grasp close so device toggle
            # logic starts from a safe latched state.
            self._device_b1_open_next = True
            self._auto_gripper_closed = float(gripper_closed)
            if not _stage_wait(0.3):
                _abort_and_return()
                return

            require_post_grasp_continue = bool(
                (not bool(fast_grasp_queue_mode))
                and (not bool(auto_replay_fast_mode))
            )

        # Pause after grasp close for user verification before lift.
        self._set_manual_gripper_override(gripper_closed)
        self._set_return_enabled(bool(allow_return_after_grasp))
        self.ros_node.sync_command_targets_to_actual()
        if bool(mode_is_precise_grasp):
            if bool(post_grasp_lift):
                self._set_status(
                    "Grasp precise confirmed. Continuing to lift.",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
            else:
                self._set_status(
                    "Grasp precise confirmed. Continuing (no lift).",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
        elif bool(fast_grasp_queue_mode):
            if bool(post_grasp_lift):
                self._set_status(
                    "Fast queued grasp: auto-lifting after close.",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
            else:
                self._set_status(
                    "Fast queued grasp: finishing after close (no lift).",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
        else:
            if (not bool(auto_replay_fast_mode)) and bool(post_grasp_lift):
                self._set_status("Grasp closed. Verify hold, then press Continue to lift.", "QLabel { color: orange; font-size: 10px; }")
            elif not bool(auto_replay_fast_mode):
                self._set_status("Grasp closed. Verify hold, then press Continue.", "QLabel { color: orange; font-size: 10px; }")
            else:
                self._set_status(
                    "Grasp closed. No-lift mode: continuing to next queued goal.",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
        if bool(require_post_grasp_continue):
            self._set_action_state("awaiting_post_grasp")
            if not _wait_until_running():
                _abort_and_return()
                return

        # Capture the live user-adjusted grasp target after the post-grasp pause.
        # This preserves manual base_x / arm / wrist edits for all later replays.
        if capture_first_trial_pose:
            snap_target_live = self._capture_current_pose_snapshot("grasp_target_final_adjusted")
            if isinstance(snap_target_live, dict):
                if bool(mode_is_precise_grasp) and isinstance(self._auto_gripper_closed, (int, float)):
                    snap_target_live["gripper"] = float(
                        np.clip(
                            float(self._auto_gripper_closed),
                            float(self.ros_node.JOINT_LIMITS[7][0]),
                            float(self.ros_node.JOINT_LIMITS[7][1]),
                        )
                    )
                self._auto_pose_grasp_target = dict(snap_target_live)
                print(
                    "[auto_loop] captured grasp target (adjusted): "
                    f"joint6={self._auto_pose_grasp_target['joint6']}, "
                    f"grip={self._auto_pose_grasp_target['gripper']:.3f}"
                )

        if not bool(post_grasp_lift):
            self._set_return_enabled(bool(allow_return_after_grasp))
            self.ros_node.sync_command_targets_to_actual()
            if capture_first_trial_pose:
                snap_grasp = self._capture_current_pose_snapshot("grasp_final_no_lift")
                if isinstance(snap_grasp, dict):
                    if bool(mode_is_precise_grasp) and isinstance(self._auto_gripper_closed, (int, float)):
                        snap_grasp["gripper"] = float(
                            np.clip(
                                float(self._auto_gripper_closed),
                                float(self.ros_node.JOINT_LIMITS[7][0]),
                                float(self.ros_node.JOINT_LIMITS[7][1]),
                            )
                        )
                    self._auto_pose_grasp = snap_grasp
                    self._auto_pose_grasp_target = dict(snap_grasp)
                    print(
                        "[auto_loop] captured grasp final pose (no lift): "
                        f"joint6={self._auto_pose_grasp['joint6']}, grip={self._auto_pose_grasp['gripper']:.3f}"
                    )
            # Optional grasp+rotate extension (same as no-lift grasp plus wrist yaw rotate).
            if bool(mode_is_grasp_rotate):
                rot_rad = float(np.deg2rad(float(rotate_after_grasp_deg)))
                rot_joint = self._current_manip_joint6()
                if not (isinstance(rot_joint, list) and len(rot_joint) >= 6):
                    j6_snap = self._auto_pose_grasp.get("joint6") if isinstance(self._auto_pose_grasp, dict) else None
                    if isinstance(j6_snap, list) and len(j6_snap) >= 6:
                        try:
                            rot_joint = [float(v) for v in j6_snap[:6]]
                        except Exception:
                            rot_joint = None
                if not (isinstance(rot_joint, list) and len(rot_joint) >= 6):
                    rot_joint = [float(v) for v in lower_joint[:6]]
                rot_joint = [float(v) for v in rot_joint[:6]]
                if isinstance(force_pitch_target_rad, float):
                    rot_joint[4] = float(force_pitch_target_rad)
                rot_joint[3] = float(
                    np.clip(
                        float(rot_joint[3]) + float(rot_rad),
                        float(self.ros_node.JOINT_LIMITS[2][0]),
                        float(self.ros_node.JOINT_LIMITS[2][1]),
                    )
                )
                self._set_status(
                    f"Grasp+rotate: rotating wrist yaw by {float(rotate_after_grasp_deg):+.1f}deg...",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not self._execute_arm_to_chunked(
                    rot_joint[:6],
                    gripper=float(gripper_closed),
                    timeout_s=8.0,
                    reliable=True,
                ):
                    _abort_and_return()
                    return
                self._set_manual_gripper_override(float(gripper_closed))
                self.ros_node.sync_command_targets_to_actual()
            hold_manual_no_lift = bool(
                bool(getattr(self, "_auto_first_trial_pending", False))
                and bool(getattr(self, "_auto_start_after_return", False))
                and (not bool(getattr(self, "_auto_loop_running", False)))
            )
            if bool(hold_manual_no_lift):
                hold_label = (
                    "Grasp+rotate done"
                    if bool(mode_is_grasp_rotate)
                    else "Grasp closed (no lift)"
                )
                self._set_status(
                    f"{hold_label}. Holding {AUTO_LOOP_GRASP_HOLD_S:.1f}s for manual episode...",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not _wait_with_pause(float(AUTO_LOOP_GRASP_HOLD_S)):
                    _abort_and_return()
                    return
                # For auto-loop first manual trial with a single no-lift grasp goal,
                # close the manual type1 segment now (right after close+hold), so
                # return-home motion is not included in this episode.
                should_close_manual_segment_now = bool(
                    bool(getattr(self, "_auto_loop_mode", "") == "goal_sequence")
                    and bool(getattr(self, "_auto_first_trial_pending", False))
                    and bool(getattr(self, "_auto_start_after_return", False))
                    and (not bool(getattr(self, "_auto_loop_running", False)))
                    and bool(getattr(self, "_run_all_queued_goals", False))
                    and (not bool(self._goal_sequence_has_next()))
                    and bool(getattr(self, "_loop_record_session_active", False))
                    and isinstance(getattr(self, "_loop_record_current", None), dict)
                )
                if bool(should_close_manual_segment_now):
                    self._stop_auto_loop_record_segment(save=True)
                if bool(mode_is_grasp_rotate):
                    # Reset (outside episode): rotate back, open, then wait for Return.
                    back_joint = self._current_manip_joint6()
                    if not (isinstance(back_joint, list) and len(back_joint) >= 6):
                        j6_snap = self._auto_pose_grasp.get("joint6") if isinstance(self._auto_pose_grasp, dict) else None
                        if isinstance(j6_snap, list) and len(j6_snap) >= 6:
                            try:
                                back_joint = [float(v) for v in j6_snap[:6]]
                            except Exception:
                                back_joint = None
                    if not (isinstance(back_joint, list) and len(back_joint) >= 6):
                        back_joint = [float(v) for v in lower_joint[:6]]
                    back_joint = [float(v) for v in back_joint[:6]]
                    if isinstance(force_pitch_target_rad, float):
                        back_joint[4] = float(force_pitch_target_rad)
                    back_joint[3] = float(
                        np.clip(
                            float(back_joint[3]) - float(np.deg2rad(float(rotate_after_grasp_deg))),
                            float(self.ros_node.JOINT_LIMITS[2][0]),
                            float(self.ros_node.JOINT_LIMITS[2][1]),
                        )
                    )
                    self._set_status(
                        "Grasp+rotate reset: rotating back...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not self._execute_arm_to_chunked(
                        back_joint[:6],
                        gripper=float(gripper_closed),
                        timeout_s=8.0,
                        reliable=True,
                    ):
                        _abort_and_return()
                        return
                    if not _stage_wait(0.1):
                        _abort_and_return()
                        return
                    self._set_status(
                        "Grasp+rotate reset: opening gripper...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    g_open_reset = float(
                        np.clip(
                            float(DEVICE_GRIPPER_TOGGLE_OPEN_JOINT),
                            float(self.ros_node.JOINT_LIMITS[7][0]),
                            float(self.ros_node.JOINT_LIMITS[7][1]),
                        )
                    )
                    if not self._execute_arm_to_chunked(
                        back_joint[:6],
                        gripper=float(g_open_reset),
                        timeout_s=6.0,
                        reliable=True,
                    ):
                        _abort_and_return()
                        return
                    self._set_manual_gripper_override(float(g_open_reset))
                    self.ros_node.sync_command_targets_to_actual()
                    self._skip_next_auto_goal_pose_capture_kind = "grasp"
                    self._set_action_state("idle")
                    self._set_return_enabled(True)
                    self._set_status(
                        "Grasp+rotate first trial complete. Object released. Use Return to start auto replay.",
                        "QLabel { color: green; font-size: 10px; }",
                    )
                    return
            elif bool(mode_is_grasp_rotate):
                # Keep consistent behavior for one-off non-loop runs: hold after rotate.
                if not _wait_with_pause(float(AUTO_LOOP_GRASP_HOLD_S)):
                    _abort_and_return()
                    return
            self._set_action_state("idle")
            if bool(allow_return_after_grasp):
                if bool(mode_is_grasp_rotate):
                    self._set_status("Grasp+rotate completed. Use Return when ready.", "QLabel { color: green; font-size: 10px; }")
                else:
                    self._set_status("Grasp completed (no lift). Use Return when ready.", "QLabel { color: green; font-size: 10px; }")
            else:
                if bool(mode_is_grasp_rotate):
                    self._set_status("Grasp+rotate completed. Continuing queued goals...", "QLabel { color: green; font-size: 10px; }")
                else:
                    self._set_status("Grasp completed (no lift). Continuing queued goals...", "QLabel { color: green; font-size: 10px; }")
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
        self._execute_arm_to_chunked(
            lift_after_joint[:6],
            gripper=gripper_lift,
            timeout_s=6.0,
            reliable=False,
        )
        _stage_wait(0.3)

        self._set_return_enabled(bool(allow_return_after_grasp))
        if capture_first_trial_pose:
            snap_grasp = self._capture_current_pose_snapshot("grasp_final")
            if isinstance(snap_grasp, dict):
                self._auto_pose_grasp = snap_grasp
                if not isinstance(self._auto_pose_grasp_target, dict):
                    self._auto_pose_grasp_target = dict(snap_grasp)
                print(
                    "[auto_loop] captured grasp final pose: "
                    f"joint6={self._auto_pose_grasp['joint6']}, grip={self._auto_pose_grasp['gripper']:.3f}"
                )

        manual_single_lift_first_trial = bool(
            bool(post_grasp_lift)
            and bool(getattr(self, "_auto_loop_mode", "") == "goal_sequence")
            and bool(getattr(self, "_auto_first_trial_pending", False))
            and bool(getattr(self, "_auto_start_after_return", False))
            and (not bool(getattr(self, "_auto_loop_running", False)))
            and bool(getattr(self, "_run_all_queued_goals", False))
            and (not bool(self._goal_sequence_has_next()))
        )
        if bool(manual_single_lift_first_trial):
            # Keep Return disabled during scripted put-back.
            self._set_return_enabled(False)
            self._set_status(
                f"Grasp+lift done. Holding {AUTO_LOOP_GRASP_HOLD_S:.1f}s for manual episode...",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            if not _wait_with_pause(float(AUTO_LOOP_GRASP_HOLD_S)):
                _abort_and_return()
                return

            # End manual type1 episode at lift-hold (exclude put-back + return).
            should_close_manual_segment_now = bool(
                bool(getattr(self, "_loop_record_session_active", False))
                and isinstance(getattr(self, "_loop_record_current", None), dict)
            )
            if bool(should_close_manual_segment_now):
                self._stop_auto_loop_record_segment(save=True)

            # Put object back: lower to grasp target height, then open gripper.
            target_snap = self._auto_pose_grasp_target if isinstance(self._auto_pose_grasp_target, dict) else None
            target_joint6 = None
            if isinstance(target_snap, dict):
                j6t = target_snap.get("joint6")
                if isinstance(j6t, list) and len(j6t) >= 6:
                    try:
                        target_joint6 = [float(v) for v in j6t[:6]]
                    except Exception:
                        target_joint6 = None
            if not (isinstance(target_joint6, list) and len(target_joint6) >= 6):
                target_joint6 = [float(v) for v in lower_joint[:6]]

            head_cmd = None
            if isinstance(target_snap, dict):
                h = target_snap.get("head")
                if isinstance(h, list) and len(h) >= 2:
                    try:
                        head_cmd = [float(h[0]), float(h[1])]
                    except Exception:
                        head_cmd = None

            grip_close_hold = self._get_manual_gripper_target(fallback=gripper_lift)
            if grip_close_hold is None:
                grip_close_hold = gripper_lift
            grip_close_hold = float(
                np.clip(
                    float(grip_close_hold),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            self._set_status(
                "Grasp+lift reset: lowering to grasp height...",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            if not self._execute_arm_to_chunked(
                target_joint6[:6],
                gripper=float(grip_close_hold),
                head=head_cmd,
                timeout_s=8.0,
                reliable=True,
            ):
                _abort_and_return()
                return
            if not _stage_wait(0.15):
                _abort_and_return()
                return

            grip_open_putback = float(
                np.clip(
                    float(DEVICE_GRIPPER_TOGGLE_OPEN_JOINT),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            self._set_status(
                "Grasp+lift reset: releasing object...",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            if not self._execute_arm_to_chunked(
                target_joint6[:6],
                gripper=float(grip_open_putback),
                head=head_cmd,
                timeout_s=6.0,
                reliable=True,
            ):
                _abort_and_return()
                return
            self._set_manual_gripper_override(float(grip_open_putback))
            self.ros_node.sync_command_targets_to_actual()
            if not _stage_wait(0.15):
                _abort_and_return()
                return

            self._set_return_enabled(True)
            self._set_action_state("idle")
            self._set_status(
                "Grasp+lift first trial complete. Object released. Use Return to start auto replay.",
                "QLabel { color: green; font-size: 10px; }",
            )
            # Keep lifted grasp snapshot for replay; do not overwrite it at
            # generic post-action capture with this post-release pose.
            self._skip_next_auto_goal_pose_capture_kind = "grasp"
            return

        self._set_action_state("idle")
        if bool(allow_return_after_grasp):
            self._set_status("Grasp completed and lifted. Use Return when ready.", "QLabel { color: green; font-size: 10px; }")
        else:
            self._set_status("Grasp completed and lifted. Continuing queued goals...", "QLabel { color: green; font-size: 10px; }")

    def _execute_approach(self, point_base, mode='reach', height_clearance=REACH_HEIGHT_CLEARANCE,
                          grasp_yaw=None, gripper_width=None, object_top_z=None,
                          grasp_mask=None, long_axis_angle=None,
                          wrist_pitch_target=None, wrist_roll_target=None,
                          grasp_shape_info=None,
                          preserve_existing_pre_action_state=False,
                          target_world_xyz_override=None,
                          preplanned_grasp_joint6=None,
                          post_grasp_lift=True,
                          grasp_rotate_deg=None,
                          precise_grasp=False,
                          precise_place=False):
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
            wrist_pitch_target=(None if wrist_pitch_target is None else float(wrist_pitch_target)),
            wrist_roll_target=(None if wrist_roll_target is None else float(wrist_roll_target)),
            grasp_shape_info=(grasp_shape_info if isinstance(grasp_shape_info, dict) else None),
            preserve_existing_pre_action_state=bool(preserve_existing_pre_action_state),
            target_world_xyz_override=target_world_xyz_override,
            preplanned_grasp_joint6=preplanned_grasp_joint6,
            post_grasp_lift=bool(post_grasp_lift),
            grasp_rotate_deg=(None if grasp_rotate_deg is None else float(grasp_rotate_deg)),
            precise_grasp=bool(precise_grasp),
            precise_place=bool(precise_place),
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
        full_h, full_w = self.head_rgb.shape[:2]
        shown_h = full_h
        crop_frac = float(np.clip(float(HEAD_DISPLAY_CROP_BOTTOM_FRAC), 0.0, 0.95))
        if crop_frac > 1e-6:
            shown_h = max(1, int(round(float(full_h) * (1.0 - crop_frac))))
        px = int(event.pos().x() * float(full_w) / max(1.0, float(self.head_display.width())))
        py = int(event.pos().y() * float(shown_h) / max(1.0, float(self.head_display.height())))
        px = int(np.clip(px, 0, full_w - 1))
        py = int(np.clip(py, 0, shown_h - 1))
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

    def _prepare_reach_goal_from_pixel(self, px, py, *, precise_place: bool = False):
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
            "precise_place": bool(precise_place),
            "created_time": float(time.time()),
        }
        return goal, None

    def _prepare_place_object_goal_from_pixel(self, px, py):
        """Copy of reach-goal preparation for dedicated place-object goal kind."""
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
            "kind": "place_object",
            "px": int(px),
            "py": int(py),
            "depth": float(depth),
            "point_xyz": point_xyz,
            "point_odom_xyz": point_odom_xyz,
            "object_top_z": None if object_top_z is None else float(object_top_z),
            "created_time": float(time.time()),
        }
        return goal, None

    def _prepare_release_goal_from_previous_goal_end(self):
        goals = self._goal_sequence_order()
        if len(goals) == 0:
            return None, "Add at least one goal before adding Release"
        prev = goals[-1]
        if not isinstance(prev, dict):
            return None, "Previous goal is invalid"

        prev_kind = str(prev.get("kind", ""))
        goal = {
            "kind": "release",
            "use_previous_goal_end": True,
            "source_goal_kind": prev_kind,
            "created_time": float(time.time()),
        }

        # For UI display only.
        if prev_kind in ("drag", "drag_curve"):
            ex = prev.get("end_px")
            ey = prev.get("end_py")
            if isinstance(ex, (int, float)) and isinstance(ey, (int, float)):
                goal["px"] = int(ex)
                goal["py"] = int(ey)
        else:
            px = prev.get("px")
            py = prev.get("py")
            if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                goal["px"] = int(px)
                goal["py"] = int(py)

        # Optional fallback absolute target if previous goal has one.
        pxyz = prev.get("point_xyz")
        if isinstance(pxyz, (list, tuple)) and len(pxyz) >= 3:
            try:
                goal["point_xyz"] = (float(pxyz[0]), float(pxyz[1]), float(pxyz[2]))
            except Exception:
                pass
        podom = prev.get("point_odom_xyz")
        if isinstance(podom, (list, tuple)) and len(podom) >= 3:
            try:
                goal["point_odom_xyz"] = (float(podom[0]), float(podom[1]), float(podom[2]))
            except Exception:
                pass
        return goal, None

    def _prepare_grasp_goal_from_pixel(
        self,
        px,
        py,
        segment=None,
        *,
        post_grasp_lift: bool = True,
        grasp_rotate_deg: float | None = None,
        wrist_pitch_target: float | None = None,
        precise_grasp: bool = False,
    ):
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
        if isinstance(grasp_yaw, (int, float)) and math.isfinite(float(grasp_yaw)):
            grasp_yaw = float(self._clip_ik_wrist_yaw_around_init(float(grasp_yaw)))
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

        point_x = float(point_base.point.x)
        point_y = float(point_base.point.y)
        point_z = float(point_base.point.z)
        if isinstance(rect_info, dict):
            center_xy = rect_info.get("center")
            if isinstance(center_xy, (list, tuple)) and len(center_xy) >= 2:
                try:
                    cx = float(center_xy[0])
                    cy = float(center_xy[1])
                    if math.isfinite(cx) and math.isfinite(cy):
                        point_x = cx
                        point_y = cy
                except (TypeError, ValueError):
                    pass
            plane_z = rect_info.get("plane_z")
            if isinstance(plane_z, (int, float)):
                pz = float(plane_z)
                if math.isfinite(pz):
                    point_z = pz
        if object_top_z is not None and isinstance(object_top_z, (int, float)):
            oz = float(object_top_z)
            if math.isfinite(oz):
                point_z = float(oz - float(GRASP_TARGET_Z_OFFSET_M))
        point_xyz = (float(point_x), float(point_y), float(point_z))
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
            "post_grasp_lift": bool(post_grasp_lift),
            "precise_grasp": bool(precise_grasp),
            "grasp_rotate_deg": (
                None
                if not isinstance(grasp_rotate_deg, (int, float))
                else float(grasp_rotate_deg)
            ),
            "wrist_pitch_target": (
                None
                if not isinstance(wrist_pitch_target, (int, float))
                else float(wrist_pitch_target)
            ),
            "grasp_mask": np.array(mask, copy=True),
            "grasp_debug_info": None if rect_info is None else dict(rect_info),
            "created_time": float(time.time()),
        }
        return goal, None

    def _store_queued_goal(self, goal):
        if goal is None:
            return
        if not isinstance(goal, dict):
            return
        kind = str(goal.get("kind", ""))
        self.queued_goal_sequence.append(dict(goal))
        if kind in self.queued_goals:
            self.queued_goals[kind] = self.queued_goal_sequence[-1]
        self._reset_goal_sequence_progress()
        if kind in ("lift_delta", "stretch_delta", "translate_delta"):
            delta_cm = float(goal.get("delta_cm", float(goal.get("delta_m", 0.0)) * 100.0))
            axis = {
                "lift_delta": "lift",
                "stretch_delta": "stretch",
                "translate_delta": "translate",
            }.get(kind, kind)
            msg = f"Queued {axis} {delta_cm:+.2f} cm"
        elif kind == "release":
            src = str(goal.get("source_goal_kind", "?"))
            msg = f"Queued release at previous goal end (from {src})"
        elif kind == "grasp":
            rotate_deg = goal.get("grasp_rotate_deg")
            is_rotate = isinstance(rotate_deg, (int, float)) and abs(float(rotate_deg)) > 1e-6
            is_precise = bool(goal.get("precise_grasp", False))
            if bool(is_precise):
                msg = f"Queued grasp precise goal at ({goal.get('px')}, {goal.get('py')})"
            elif bool(goal.get("post_grasp_lift", True)):
                msg = f"Queued grasp+lift goal at ({goal.get('px')}, {goal.get('py')})"
            elif bool(is_rotate):
                msg = (
                    f"Queued grasp+rotate goal ({float(rotate_deg):+.1f}deg) "
                    f"at ({goal.get('px')}, {goal.get('py')})"
                )
            else:
                msg = f"Queued grasp goal at ({goal.get('px')}, {goal.get('py')})"
        elif kind == "reach":
            if bool(goal.get("precise_place", False)):
                msg = f"Queued place precise goal at ({goal.get('px')}, {goal.get('py')})"
            else:
                msg = f"Queued reach goal at ({goal.get('px')}, {goal.get('py')})"
        elif kind == "place_object":
            msg = f"Queued place_object goal at ({goal.get('px')}, {goal.get('py')})"
        else:
            msg = f"Queued {kind} goal at ({goal.get('px')}, {goal.get('py')})"
        self._set_status(
            msg,
            "QLabel { color: #1e88e5; font-size: 10px; }"
        )

    def _precompute_followup_reach_from_grasp_goal(self, grasp_goal: dict[str, Any]) -> None:
        """Precompute queued reach IK target before executing queued grasp.

        This keeps grasp->reach sequence deterministic by planning the reach
        target (with grasp yaw hint) from the same pre-grasp snapshot.
        """
        reach_goal = None
        try:
            goals = self._goal_sequence_order()
            gidx = next((i for i, g in enumerate(goals) if g is grasp_goal), -1)
            if gidx >= 0:
                for g in goals[gidx + 1 :]:
                    if isinstance(g, dict) and str(g.get("kind", "")) in ("reach", "place_object"):
                        reach_goal = g
                        break
        except Exception:
            reach_goal = None
        if not isinstance(reach_goal, dict):
            reach_goal = self.queued_goals.get("reach")
        if not isinstance(reach_goal, dict):
            reach_goal = self.queued_goals.get("place_object")
        if not isinstance(reach_goal, dict) or str(reach_goal.get("kind", "")) != "reach":
            if not isinstance(reach_goal, dict) or str(reach_goal.get("kind", "")) != "place_object":
                return

        target_world = reach_goal.get("point_odom_xyz")
        if not (isinstance(target_world, (list, tuple)) and len(target_world) >= 3):
            resolved_xyz = self._resolve_goal_point_xyz_for_current_base(reach_goal)
            if not (isinstance(resolved_xyz, (list, tuple)) and len(resolved_xyz) >= 3):
                return
            base_pose = self.ros_node.get_measured_base_pose_xytheta()
            if not (isinstance(base_pose, (list, tuple)) and len(base_pose) >= 3):
                return
            target_world = self._base_point_to_odom_xyz(
                (float(resolved_xyz[0]), float(resolved_xyz[1]), float(resolved_xyz[2])),
                (float(base_pose[0]), float(base_pose[1]), float(base_pose[2])),
            )

        grasp_yaw = grasp_goal.get("grasp_yaw")
        yaw_hint = float(grasp_yaw) if isinstance(grasp_yaw, (int, float)) else None
        if isinstance(yaw_hint, float) and math.isfinite(yaw_hint):
            yaw_hint = float(self._clip_ik_wrist_yaw_around_init(float(yaw_hint)))
        self._set_status(
            "Sequence: precomputing reach target from current frame...",
            "QLabel { color: blue; font-size: 10px; }",
        )
        plan = self.ros_node.plan_open_loop_grasp(
            [float(target_world[0]), float(target_world[1]), float(target_world[2])],
            pregrasp_distance=0.20,
            lift_distance=IK_LIFT_DISTANCE_M,
            wrist_yaw_target=yaw_hint,
            wrist_pitch_target=float(np.deg2rad(GRASP_PITCH_DEG)),
            wrist_roll_target=None,
            timeout_s=25.0,
        )
        if not (isinstance(plan, dict) and bool(plan.get("ok", False))):
            err = None
            if isinstance(plan, dict):
                err = plan.get("error")
            print(f"[goal_sequence] reach precompute skipped: {err if err else 'planner failed'}")
            return
        grasp_joint = plan.get("grasp_joint")
        if not (isinstance(grasp_joint, list) and len(grasp_joint) >= 6):
            print("[goal_sequence] reach precompute skipped: invalid grasp_joint")
            return

        reach_goal["_precomputed_grasp_joint6"] = [float(v) for v in grasp_joint[:6]]
        if yaw_hint is not None:
            reach_goal["_precomputed_yaw_hint"] = float(yaw_hint)
        reach_goal["_precomputed_world_xyz"] = [
            float(target_world[0]),
            float(target_world[1]),
            float(target_world[2]),
        ]
        print(
            "[goal_sequence] precomputed reach from grasp frame: "
            f"target_world=({float(target_world[0]):+.3f}, {float(target_world[1]):+.3f}, {float(target_world[2]):+.3f}) "
            f"joint6={reach_goal['_precomputed_grasp_joint6']}"
        )

    def add_reach_goal_at_pixel(self, px, py, *, precise_place: bool = False):
        with self._action_lock:
            if self._action_state != 'idle':
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        goal, err = self._prepare_reach_goal_from_pixel(px, py, precise_place=bool(precise_place))
        if err:
            self._set_status(err, "QLabel { color: red; font-size: 10px; }")
            return
        self._store_queued_goal(goal)

    def add_place_object_goal_at_pixel(self, px, py):
        with self._action_lock:
            if self._action_state != 'idle':
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        goal, err = self._prepare_place_object_goal_from_pixel(px, py)
        if err:
            self._set_status(err, "QLabel { color: red; font-size: 10px; }")
            return
        self._store_queued_goal(goal)

    def add_release_goal_from_previous_goal_end(self):
        with self._action_lock:
            if self._action_state != 'idle':
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        goal, err = self._prepare_release_goal_from_previous_goal_end()
        if err:
            self._set_status(err, "QLabel { color: red; font-size: 10px; }")
            return
        self._store_queued_goal(goal)

    def add_release_goal_at_pixel(self, px, py):
        # Kept for backward compatibility: release no longer uses mouse depth.
        _ = (px, py)
        self.add_release_goal_from_previous_goal_end()

    def add_grasp_goal_at_pixel(
        self,
        px,
        py,
        segment=None,
        *,
        post_grasp_lift: bool = True,
        grasp_rotate_deg: float | None = None,
        wrist_pitch_target: float | None = None,
        precise_grasp: bool = False,
    ):
        with self._action_lock:
            if self._action_state != 'idle':
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        goal, err = self._prepare_grasp_goal_from_pixel(
            px,
            py,
            segment=segment,
            post_grasp_lift=bool(post_grasp_lift),
            grasp_rotate_deg=grasp_rotate_deg,
            wrist_pitch_target=wrist_pitch_target,
            precise_grasp=bool(precise_grasp),
        )
        if err:
            self._set_status(err, "QLabel { color: red; font-size: 10px; }")
            return
        self._store_queued_goal(goal)

    def add_drag_goal_between_pixels(self, start_px, end_px):
        with self._action_lock:
            if self._action_state != 'idle':
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        sx, sy = int(start_px[0]), int(start_px[1])
        ex, ey = int(end_px[0]), int(end_px[1])
        if max(abs(ex - sx), abs(ey - sy)) < int(DRAG_MIN_PIXEL_LENGTH_PX):
            self._set_status("Drag: line too short", "QLabel { color: orange; font-size: 10px; }")
            return
        goal = {
            "kind": "drag",
            "px": int(sx),
            "py": int(sy),
            "end_px": int(ex),
            "end_py": int(ey),
            "created_time": float(time.time()),
        }
        # Capture fallback 3D endpoints now (before grasp can occlude depth) so
        # queued drag can still execute when live endpoint depth is unavailable.
        try:
            fb = self._build_fallback_base_path_from_pixels(
                [(int(sx), int(sy)), (int(ex), int(ey))],
                search_radius_px=int(DRAG_NEAREST_VALID_DEPTH_RADIUS_PX),
            )
            if len(fb) >= 2:
                goal["path_base_xyz"] = [[float(x), float(y), float(z)] for (x, y, z) in fb]
        except Exception:
            pass
        self._store_queued_goal(goal)

    def add_curved_drag_goal_from_pixels(self, path_pixels):
        with self._action_lock:
            if self._action_state != 'idle':
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        if not isinstance(path_pixels, list) or len(path_pixels) < int(DRAG_CURVE_MIN_CAPTURE_POINTS):
            self._set_status("Curve: need a longer mouse path", "QLabel { color: orange; font-size: 10px; }")
            return
        norm_pts: list[tuple[int, int]] = []
        for p in path_pixels:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                continue
            try:
                px_i = int(p[0])
                py_i = int(p[1])
            except Exception:
                continue
            if len(norm_pts) == 0 or norm_pts[-1] != (px_i, py_i):
                norm_pts.append((px_i, py_i))
        if len(norm_pts) < 2 or self._polyline_length_px(norm_pts) < float(DRAG_MIN_PIXEL_LENGTH_PX):
            self._set_status("Curve: path too short", "QLabel { color: orange; }")
            return
        accepted, height_cm, no_height_adjust = self._prompt_curve_height_options()
        if not accepted:
            self._set_status("Curve add canceled", "QLabel { color: gray; font-size: 10px; }")
            return
        start = norm_pts[0]
        end = norm_pts[-1]
        goal = {
            "kind": "drag_curve",
            "px": int(start[0]),
            "py": int(start[1]),
            "end_px": int(end[0]),
            "end_py": int(end[1]),
            "path_px": [[int(x), int(y)] for (x, y) in norm_pts],
            "surface_height_offset_cm": float(height_cm),
            "surface_height_offset_m": float(height_cm / 100.0),
            "no_height_adjustment": bool(no_height_adjust),
            "created_time": float(time.time()),
        }
        # Capture fallback 3D path now (before grasp can occlude depth) so curve
        # replay still follows intermediate points even if depth drops later.
        try:
            fb = self._build_fallback_base_path_from_pixels(
                [(int(x), int(y)) for (x, y) in norm_pts],
                search_radius_px=int(DRAG_NEAREST_VALID_DEPTH_RADIUS_PX),
            )
            if len(fb) >= 2:
                goal["path_base_xyz"] = [[float(x), float(y), float(z)] for (x, y, z) in fb]
        except Exception:
            pass
        self._store_queued_goal(goal)

    def _prompt_and_add_delta_goal(self, *, kind: str, title: str, label: str) -> None:
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Add goals while idle (before starting sequence)", "QLabel { color: orange; font-size: 10px; }")
                return
        value_cm, ok = QInputDialog.getDouble(
            self,
            title,
            label,
            0.0,
            -200.0,
            200.0,
            2,
        )
        if not bool(ok):
            return
        delta_cm = float(value_cm)
        goal = {
            "kind": str(kind),
            "delta_cm": float(delta_cm),
            "delta_m": float(delta_cm / 100.0),
            "created_time": float(time.time()),
        }
        self._store_queued_goal(goal)

    def _execute_relative_delta_goal(self, kind: str, delta_m: float) -> bool:
        cur = self._current_manip_joint6()
        if not (isinstance(cur, list) and len(cur) >= 6):
            return False
        tgt = [float(v) for v in cur[:6]]

        if kind == "lift_delta":
            lo, hi = self.ros_node.JOINT_LIMITS[1]
            tgt[1] = float(np.clip(float(tgt[1]) + float(delta_m), float(lo), float(hi)))
        elif kind == "stretch_delta":
            lo, hi = self.ros_node.JOINT_LIMITS[0]
            tgt[2] = float(np.clip(float(tgt[2]) + float(delta_m), float(lo), float(hi)))
        elif kind == "translate_delta":
            lo, hi = MANIP_BASE_X_LIMITS
            tgt[0] = float(np.clip(float(tgt[0]) + float(delta_m), float(lo), float(hi)))
        else:
            return False

        q_target = self.ros_node.get_target_qpos()
        if not (isinstance(q_target, list) and len(q_target) >= 8):
            q_target = self.ros_node.get_actual_qpos()
        if not (isinstance(q_target, list) and len(q_target) >= 8):
            q_target = self._default_init_q8()

        g = self._get_manual_gripper_target(fallback=float(q_target[7]))
        if g is None:
            g = float(q_target[7])
        g = float(
            np.clip(
                float(g),
                float(self.ros_node.JOINT_LIMITS[7][0]),
                float(self.ros_node.JOINT_LIMITS[7][1]),
            )
        )
        head_cmd = [
            float(np.clip(float(q_target[5]), float(self.ros_node.JOINT_LIMITS[5][0]), float(self.ros_node.JOINT_LIMITS[5][1]))),
            float(np.clip(float(q_target[6]), float(self.ros_node.JOINT_LIMITS[6][0]), float(self.ros_node.JOINT_LIMITS[6][1]))),
        ]
        return bool(
            self._execute_arm_to_chunked(
                tgt[:6],
                gripper=float(g),
                head=head_cmd,
                timeout_s=float(ACTION_MOVE_TIMEOUT_DEFAULT_S),
                reliable=False,
            )
        )

    def _execute_release_goal(
        self,
        point_base: PointStamped | None,
        *,
        use_current_pose: bool = False,
    ) -> bool:
        """Direct release: go to target point at current arm height and open gripper."""
        cur = self._current_manip_joint6()
        if not (isinstance(cur, list) and len(cur) >= 6):
            return False
        cur = [float(v) for v in cur[:6]]

        q_target = self.ros_node.get_target_qpos()
        if not (isinstance(q_target, list) and len(q_target) >= 8):
            q_target = self.ros_node.get_actual_qpos()
        if not (isinstance(q_target, list) and len(q_target) >= 8):
            q_target = self._default_init_q8()

        grip_hold = self._get_manual_gripper_target(fallback=float(q_target[7]))
        if grip_hold is None:
            grip_hold = float(q_target[7])
        grip_hold = float(
            np.clip(
                float(grip_hold),
                float(self.ros_node.JOINT_LIMITS[7][0]),
                float(self.ros_node.JOINT_LIMITS[7][1]),
            )
        )
        head_cmd = [
            float(np.clip(float(q_target[5]), float(self.ros_node.JOINT_LIMITS[5][0]), float(self.ros_node.JOINT_LIMITS[5][1]))),
            float(np.clip(float(q_target[6]), float(self.ros_node.JOINT_LIMITS[6][0]), float(self.ros_node.JOINT_LIMITS[6][1]))),
        ]

        if bool(use_current_pose) or point_base is None:
            target_joint = [float(v) for v in cur[:6]]
        else:
            target_joint = self._plan_drag_joint_for_point_base(
                point_base,
                wrist_yaw=float(cur[3]),
                wrist_pitch=float(cur[4]),
                wrist_roll=float(cur[5]),
            )
            if not (isinstance(target_joint, list) and len(target_joint) >= 6):
                return False
            target_joint = [float(v) for v in target_joint[:6]]
            # Keep current vertical height and wrist orientation for release mode.
            target_joint[1] = float(np.clip(float(cur[1]), float(self.ros_node.JOINT_LIMITS[1][0]), float(self.ros_node.JOINT_LIMITS[1][1])))
            target_joint[3] = float(cur[3])
            target_joint[4] = float(cur[4])
            target_joint[5] = float(cur[5])

            if not self._execute_arm_to_chunked(
                target_joint[:6],
                gripper=float(grip_hold),
                head=head_cmd,
                timeout_s=float(ACTION_MOVE_TIMEOUT_LONG_S),
                reliable=False,
            ):
                return False

        grip_open = float(
            np.clip(
                float(DEVICE_GRIPPER_TOGGLE_OPEN_JOINT),
                float(self.ros_node.JOINT_LIMITS[7][0]),
                float(self.ros_node.JOINT_LIMITS[7][1]),
            )
        )
        if not self._execute_arm_to_chunked(
            target_joint[:6],
            gripper=float(grip_open),
            head=head_cmd,
            timeout_s=6.0,
            reliable=False,
        ):
            return False
        self._set_manual_gripper_override(float(grip_open))
        self.ros_node.sync_command_targets_to_actual()
        return True

    def _start_prepared_goal(self, goal, preserve_existing_pre_action_state=False):
        if goal is None:
            return False
        kind = str(goal.get("kind", ""))
        drag_repeat_count = 1
        drag_return_to_start = False
        chained_from_no_lift_grasp = False
        try:
            if bool(preserve_existing_pre_action_state):
                seq = self._goal_sequence_order()
                cur_idx = int(self.queued_goal_cursor)
                if 0 < cur_idx <= len(seq):
                    prev_goal = seq[cur_idx - 1]
                    chained_from_no_lift_grasp = bool(
                        isinstance(prev_goal, dict)
                        and str(prev_goal.get("kind", "")) == "grasp"
                        and (not bool(prev_goal.get("post_grasp_lift", True)))
                    )
        except Exception:
            chained_from_no_lift_grasp = False
        if bool(self._run_all_queued_goals):
            try:
                drag_repeat_count = int(max(1, int(getattr(self, "_queued_drag_repeat_count", 1))))
            except Exception:
                drag_repeat_count = 1
            drag_return_to_start = bool(getattr(self, "_queued_drag_return_to_start", False))
        if kind == "drag":
            sx = goal.get("px")
            sy = goal.get("py")
            ex = goal.get("end_px")
            ey = goal.get("end_py")
            if not all(isinstance(v, (int, float)) for v in (sx, sy, ex, ey)):
                self._set_status("Invalid queued drag goal", "QLabel { color: red; font-size: 10px; }")
                return False
            if int(drag_repeat_count) > 1 and bool(drag_return_to_start):
                self._set_status(
                    f"Executing queued drag operation ({drag_repeat_count} passes with return-to-start)...",
                    "QLabel { color: blue; font-size: 10px; }",
                )
            else:
                self._set_status("Executing queued drag operation...", "QLabel { color: blue; font-size: 10px; }")
            fallback_path = None
            path_base_xyz = goal.get("path_base_xyz")
            if isinstance(path_base_xyz, list) and len(path_base_xyz) >= 2:
                parsed_fb: list[tuple[float, float, float]] = []
                for xyz in path_base_xyz:
                    if not (isinstance(xyz, (list, tuple)) and len(xyz) >= 3):
                        continue
                    try:
                        parsed_fb.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
                    except Exception:
                        continue
                if len(parsed_fb) >= 2:
                    fallback_path = parsed_fb
            return bool(
                self._execute_drag_operation_between_pixels(
                    (int(sx), int(sy)),
                    (int(ex), int(ey)),
                    repeat_count=int(drag_repeat_count),
                    return_along_path=bool(drag_return_to_start),
                    force_keep_current_lift=bool(
                        bool(chained_from_no_lift_grasp) or bool(goal.get("_force_keep_current_lift", False))
                    ),
                    fallback_path_points_base=fallback_path,
                )
            )
        if kind == "drag_curve":
            pts = goal.get("path_px")
            if not isinstance(pts, list) or len(pts) < 2:
                self._set_status("Invalid queued curved trajectory", "QLabel { color: red; font-size: 10px; }")
                return False
            path: list[tuple[int, int]] = []
            for p in pts:
                if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                    continue
                try:
                    path.append((int(p[0]), int(p[1])))
                except Exception:
                    continue
            if len(path) < 2:
                self._set_status("Invalid queued curved trajectory", "QLabel { color: red; font-size: 10px; }")
                return False
            z_off_m = float(goal.get("surface_height_offset_m", 0.0))
            no_height_adjust = bool(goal.get("no_height_adjustment", False))
            keep_current_lift = bool(
                (no_height_adjust and bool(preserve_existing_pre_action_state))
                or bool(chained_from_no_lift_grasp)
            )
            if int(drag_repeat_count) > 1 and bool(drag_return_to_start):
                self._set_status(
                    f"Executing queued curved trajectory ({drag_repeat_count} passes with return-to-start)...",
                    "QLabel { color: blue; font-size: 10px; }",
                )
            else:
                self._set_status("Executing queued curved trajectory...", "QLabel { color: blue; font-size: 10px; }")
            fallback_path = None
            path_base_xyz = goal.get("path_base_xyz")
            if isinstance(path_base_xyz, list) and len(path_base_xyz) >= 2:
                parsed_fb: list[tuple[float, float, float]] = []
                for xyz in path_base_xyz:
                    if not (isinstance(xyz, (list, tuple)) and len(xyz) >= 3):
                        continue
                    try:
                        parsed_fb.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
                    except Exception:
                        continue
                if len(parsed_fb) >= 2:
                    fallback_path = parsed_fb
            return bool(
                self._execute_drag_operation_with_pixel_path(
                    path,
                    path_kind="curve",
                    repeat_count=int(drag_repeat_count),
                    return_along_path=bool(drag_return_to_start),
                    surface_height_offset_m=float(z_off_m),
                    no_height_adjustment=bool(no_height_adjust),
                    keep_current_lift_when_no_adjust=bool(keep_current_lift),
                    force_keep_current_lift=bool(
                        bool(chained_from_no_lift_grasp) or bool(goal.get("_force_keep_current_lift", False))
                    ),
                    fallback_path_points_base=fallback_path,
                )
            )
        if kind in ("lift_delta", "stretch_delta", "translate_delta"):
            if not self._begin_action(kind):
                self._set_status("Another action is already running/paused", "QLabel { color: orange; font-size: 10px; }")
                return False
            delta_m = float(goal.get("delta_m", 0.0))
            delta_cm = float(goal.get("delta_cm", float(delta_m) * 100.0))
            axis = {
                "lift_delta": "lift",
                "stretch_delta": "stretch",
                "translate_delta": "translate",
            }.get(kind, kind)
            self._set_status(
                f"Executing queued {axis} {delta_cm:+.2f} cm...",
                "QLabel { color: blue; font-size: 10px; }",
            )

            def run_delta():
                try:
                    ok = self._execute_relative_delta_goal(kind, delta_m)
                    if not ok:
                        raise RuntimeError(f"{axis} move failed")
                    self._set_status(
                        f"{axis.capitalize()} move complete ({delta_cm:+.2f} cm).",
                        "QLabel { color: green; font-size: 10px; }",
                    )
                except Exception as e:
                    print(f"{axis} delta error: {e}")
                    self._set_status(f"{axis.capitalize()} failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
                finally:
                    launch_next = False
                    if self._run_all_queued_goals:
                        if self._goal_sequence_has_next():
                            launch_next = True
                        else:
                            self._run_all_queued_goals = False
                    if self._deferred_next_goal_start:
                        if self._goal_sequence_has_next():
                            self._deferred_next_goal_start = False
                            launch_next = True
                        else:
                            self._deferred_next_goal_start = False
                    with self._action_lock:
                        st = self._action_state
                    if st == 'running':
                        self._set_action_state('idle')
                    self._update_goal_queue_label()
                    self._update_next_goal_button_state()
                    if launch_next:
                        self._start_next_queued_goal()

            from threading import Thread
            Thread(target=run_delta, daemon=True).start()
            return True
        if kind == "release":
            if not self._begin_action(kind):
                self._set_status("Another action is already running/paused", "QLabel { color: orange; font-size: 10px; }")
                return False
            use_prev_end = bool(goal.get("use_previous_goal_end", False))
            point_base = None
            if not bool(use_prev_end):
                resolved_xyz = self._resolve_goal_point_xyz_for_current_base(goal)
                if resolved_xyz is None:
                    self._set_status("Invalid queued release point", "QLabel { color: red; font-size: 10px; }")
                    self._set_action_state('idle')
                    return False
                point_base = self._point_from_xyz(resolved_xyz)
            if bool(use_prev_end):
                src = str(goal.get("source_goal_kind", "?"))
                self._set_status(
                    f"Executing queued release at previous goal end ({src})...",
                    "QLabel { color: blue; font-size: 10px; }",
                )
            else:
                self._set_status("Executing queued release (direct, keep current height)...", "QLabel { color: blue; font-size: 10px; }")

            def run_release():
                try:
                    ok = self._execute_release_goal(point_base, use_current_pose=bool(use_prev_end))
                    if not ok:
                        raise RuntimeError("release move failed")
                    if (not bool(self._auto_loop_running)) and bool(self._auto_first_trial_pending):
                        snap_rel = self._capture_current_pose_snapshot("release_final")
                        if isinstance(snap_rel, dict):
                            self._auto_pose_release = snap_rel
                            print(
                                "[auto_loop] captured release pose: "
                                f"joint6={self._auto_pose_release['joint6']}, "
                                f"grip={self._auto_pose_release['gripper']:.3f}"
                            )
                    allow_return = not bool(self._run_all_queued_goals and self._goal_sequence_has_next())
                    self._set_return_enabled(bool(allow_return))
                    if bool(allow_return):
                        self._set_status("Release complete. Use Return when ready.", "QLabel { color: green; font-size: 10px; }")
                    else:
                        self._set_status("Release complete. Continuing queued goals...", "QLabel { color: green; font-size: 10px; }")
                except Exception as e:
                    print(f"Release error: {e}")
                    self._set_status(f"Release failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
                finally:
                    launch_next = False
                    if self._run_all_queued_goals:
                        if self._goal_sequence_has_next():
                            launch_next = True
                        else:
                            self._run_all_queued_goals = False
                    if self._deferred_next_goal_start:
                        if self._goal_sequence_has_next():
                            self._deferred_next_goal_start = False
                            launch_next = True
                        else:
                            self._deferred_next_goal_start = False
                    with self._action_lock:
                        st = self._action_state
                    if st == 'running':
                        self._set_action_state('idle')
                    self._update_goal_queue_label()
                    self._update_next_goal_button_state()
                    if launch_next:
                        self._start_next_queued_goal()

            from threading import Thread
            Thread(target=run_release, daemon=True).start()
            return True
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
            rotate_deg = goal.get("grasp_rotate_deg")
            is_rotate = isinstance(rotate_deg, (int, float)) and abs(float(rotate_deg)) > 1e-6
            is_precise = bool(goal.get("precise_grasp", False))
            if bool(is_precise):
                self._set_status("Executing queued grasp precise sequence...", "QLabel { color: blue; font-size: 10px; }")
            elif bool(goal.get("post_grasp_lift", True)):
                self._set_status("Executing queued grasp+lift sequence...", "QLabel { color: blue; font-size: 10px; }")
            elif bool(is_rotate):
                self._set_status(
                    f"Executing queued grasp+rotate sequence ({float(rotate_deg):+.1f}deg)...",
                    "QLabel { color: blue; font-size: 10px; }",
                )
            else:
                self._set_status("Executing queued grasp sequence (no lift)...", "QLabel { color: blue; font-size: 10px; }")
            if isinstance(goal.get("grasp_debug_info"), dict):
                self._grasp_debug_info = dict(goal["grasp_debug_info"])
            else:
                self._grasp_debug_info = None
        elif kind == "reach":
            if bool(goal.get("precise_place", False)):
                self._set_status("Executing queued place precise sequence...", "QLabel { color: blue; font-size: 10px; }")
            else:
                self._set_status("Executing queued reach sequence...", "QLabel { color: blue; font-size: 10px; }")
            self._grasp_debug_info = None
        elif kind == "place_object":
            self._set_status("Executing queued place object sequence...", "QLabel { color: blue; font-size: 10px; }")
            self._grasp_debug_info = None
        else:
            self._set_status(f"Executing queued {kind} sequence...", "QLabel { color: blue; font-size: 10px; }")
            self._grasp_debug_info = None

        def run():
            try:
                if kind == "grasp":
                    try:
                        self._precompute_followup_reach_from_grasp_goal(goal)
                    except Exception as pre_exc:
                        print(f"[goal_sequence] reach precompute error: {pre_exc}")
                kwargs = dict(
                    point_base=point_base,
                    mode=("place_object" if kind == "place_object" else ("reach" if kind == "reach" else kind)),
                    object_top_z=goal.get("object_top_z"),
                    preserve_existing_pre_action_state=bool(preserve_existing_pre_action_state),
                )
                goal_odom_xyz = goal.get("point_odom_xyz")
                if isinstance(goal_odom_xyz, (list, tuple)) and len(goal_odom_xyz) >= 3:
                    kwargs["target_world_xyz_override"] = (
                        float(goal_odom_xyz[0]),
                        float(goal_odom_xyz[1]),
                        float(goal_odom_xyz[2]),
                    )
                if kind in ("reach", "place_object"):
                    kwargs["height_clearance"] = REACH_HEIGHT_CLEARANCE
                    pre_joint6 = goal.get("_precomputed_grasp_joint6")
                    if isinstance(pre_joint6, list) and len(pre_joint6) >= 6:
                        kwargs["preplanned_grasp_joint6"] = [float(v) for v in pre_joint6[:6]]
                    pre_yaw = goal.get("_precomputed_yaw_hint")
                    if isinstance(pre_yaw, (int, float)):
                        kwargs["grasp_yaw"] = float(pre_yaw)
                else:
                    kwargs["grasp_yaw"] = goal.get("grasp_yaw")
                    kwargs["gripper_width"] = goal.get("gripper_width")
                    kwargs["grasp_mask"] = goal.get("grasp_mask")
                    kwargs["long_axis_angle"] = goal.get("long_axis_angle")
                    kwargs["wrist_pitch_target"] = goal.get("wrist_pitch_target")
                    kwargs["post_grasp_lift"] = bool(goal.get("post_grasp_lift", True))
                    kwargs["grasp_rotate_deg"] = goal.get("grasp_rotate_deg")
                    kwargs["precise_grasp"] = bool(goal.get("precise_grasp", False))
                if kind in ("reach", "place_object"):
                    kwargs["precise_place"] = bool(goal.get("precise_place", False))
                self._execute_approach(**kwargs)
            except Exception as e:
                print(f"{kind.capitalize()} error: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"{kind.capitalize()} failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
            finally:
                launch_next = False
                if self._run_all_queued_goals:
                    if self._goal_sequence_has_next():
                        launch_next = True
                    else:
                        self._run_all_queued_goals = False
                if self._deferred_next_goal_start:
                    if self._goal_sequence_has_next():
                        self._deferred_next_goal_start = False
                        launch_next = True
                    else:
                        self._deferred_next_goal_start = False
                with self._action_lock:
                    st = self._action_state
                if st == 'running':
                    self._set_action_state('idle')
                # Capture learned pose from first/manual trial (or latest successful run).
                try:
                    self._maybe_capture_goal_pose_for_auto(kind)
                except Exception as cap_exc:
                    print(f"[auto_loop] pose capture failed for {kind}: {cap_exc}")
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
            self._run_all_queued_goals = False
            self._set_status("No queued goal remaining", "QLabel { color: gray; font-size: 10px; }")
            self._update_next_goal_button_state()
            return False
        goal = goals[self.queued_goal_cursor]
        preserve = bool(self.queued_sequence_started and self._pre_action_state is not None)
        if preserve:
            self._sync_pre_action_rotation_from_odom()
            try:
                dbg_base_x = float(self._pre_action_state.get("base_x", float("nan")))
                dbg_pose = self._pre_action_state.get("base_pose_xytheta_start")
                print(
                    "[goal_sequence] reusing saved return state "
                    f"for goal#{self.queued_goal_cursor + 1} "
                    f"(base_x={dbg_base_x:+.3f}, base_pose_start={dbg_pose})"
                )
            except Exception:
                pass
        else:
            print(
                "[goal_sequence] starting new return state "
                f"for goal#{self.queued_goal_cursor + 1}"
            )
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
            self._set_status("No queued goals. Use right-click to add goals (grasp/place/drag/curve).",
                             "QLabel { color: orange; font-size: 10px; }")
            return
        with self._action_lock:
            st = self._action_state

        # If a scripted action is active and another queued goal exists, abort current
        # action and continue with the next queued goal.
        if st in ('running', 'paused', 'awaiting_confirm', 'awaiting_post_reach_release') and self._goal_sequence_has_next():
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
        grasp_shape_info = None
        if mask is not None:
            grasp_shape_info = self._analyze_segment_geometry(mask)
            grasp_yaw, rect_info = self._compute_grasp_orientation(mask, px, py)
            # For non-horizontal/vertical-like surfaces, orientation from full 3D
            # surface PCA is more stable than top-surface rectangle orientation.
            if isinstance(grasp_shape_info, dict):
                axis_angle = grasp_shape_info.get("axis_angle_xy_rad")
                strategy = str(grasp_shape_info.get("approach_strategy", "lift_standoff"))
                if axis_angle is not None and strategy == "reach_standoff":
                    grasp_yaw = self._resolve_wrist_yaw_candidate(float(axis_angle) + float(np.pi / 2.0))
                    print(
                        "  Using geometry-PCA yaw for vertical-like object: "
                        f"axis={np.degrees(float(axis_angle)):.1f}deg, yaw={np.degrees(float(grasp_yaw)):.1f}deg"
                    )
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
            if isinstance(grasp_shape_info, dict):
                print(
                    "  Geometry class: "
                    f"{grasp_shape_info.get('geometry_class')} "
                    f"(strategy={grasp_shape_info.get('approach_strategy')})"
                )

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

        point_base_exec = point_base
        if isinstance(rect_info, dict):
            center_xy = rect_info.get("center")
            plane_z = rect_info.get("plane_z")
            target_x = float(point_base.point.x)
            target_y = float(point_base.point.y)
            target_z = float(point_base.point.z)
            if isinstance(center_xy, (list, tuple)) and len(center_xy) >= 2:
                try:
                    cx = float(center_xy[0])
                    cy = float(center_xy[1])
                    if math.isfinite(cx) and math.isfinite(cy):
                        target_x = cx
                        target_y = cy
                except (TypeError, ValueError):
                    pass
            if isinstance(plane_z, (int, float)) and math.isfinite(float(plane_z)):
                target_z = float(plane_z)
            if isinstance(object_top_z, (int, float)) and math.isfinite(float(object_top_z)):
                target_z = float(float(object_top_z) - float(GRASP_TARGET_Z_OFFSET_M))
            point_base_exec = self._point_from_xyz((target_x, target_y, target_z))
            print(
                f"  IK target from top-surface centroid (z-{GRASP_TARGET_Z_OFFSET_M*100:.0f}cm): "
                f"x={target_x:+.3f}, y={target_y:+.3f}, z={target_z:+.3f} (base_link)"
            )

        def run():
            try:
                shape_info_for_ik = (
                    grasp_shape_info if bool(DIRECT_GRASP_USE_GEOMETRY_STRATEGY) else None
                )
                self._execute_approach(point_base_exec, mode='grasp',
                                       grasp_yaw=grasp_yaw,
                                       gripper_width=gripper_width,
                                       object_top_z=object_top_z,
                                       grasp_mask=mask,
                                       long_axis_angle=long_axis_angle,
                                       grasp_shape_info=shape_info_for_ik)
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

    def _preview_grasp_at_pixel(self, px, py, segment=None):
        """Compute/draw grasp debug overlay at click point without robot motion."""
        point_base, depth = self._get_3d_point_at_pixel(px, py)
        if point_base is None or depth is None:
            self._set_status("See Grasp: no valid depth at clicked point", "QLabel { color: red; }")
            return

        mask = None
        if segment is not None and isinstance(segment, dict):
            mask = segment.get('mask')
        if mask is None and self.segments:
            for seg in self.segments:
                if py < seg['mask'].shape[0] and px < seg['mask'].shape[1] and seg['mask'][py, px] > 0:
                    mask = seg['mask']
                    break
        if mask is None and self.selected_segment is not None and 'mask' in self.selected_segment:
            sel_mask = self.selected_segment['mask']
            if py < sel_mask.shape[0] and px < sel_mask.shape[1]:
                mask = sel_mask

        if mask is None:
            self._set_status("See Grasp: click on a segmented object", "QLabel { color: orange; }")
            return

        grasp_shape_info = self._analyze_segment_geometry(mask)
        grasp_yaw, rect_info = self._compute_grasp_orientation(mask, px, py)
        if isinstance(grasp_shape_info, dict):
            axis_angle = grasp_shape_info.get("axis_angle_xy_rad")
            strategy = str(grasp_shape_info.get("approach_strategy", "lift_standoff"))
            if axis_angle is not None and strategy == "reach_standoff":
                grasp_yaw = self._resolve_wrist_yaw_candidate(float(axis_angle) + float(np.pi / 2.0))

        gripper_width = self._estimate_gripper_width(mask, px, py, depth, rect_info)
        object_top_z = self._compute_object_top_z(mask)
        self._grasp_debug_info = rect_info
        self.update_camera_displays()

        geom = ""
        if isinstance(grasp_shape_info, dict):
            geom = (
                f", class={grasp_shape_info.get('geometry_class')}"
                f", strategy={grasp_shape_info.get('approach_strategy')}"
            )
        msg = (
            "See Grasp: "
            f"yaw={np.degrees(float(grasp_yaw)):.1f}deg, "
            f"gripper={float(gripper_width):.3f}, "
            f"top_z={(float(object_top_z) if object_top_z is not None else float('nan')):.3f}m"
            f"{geom}"
        )
        print(msg)
        self._set_status(msg, "QLabel { color: #1e88e5; font-size: 10px; }")

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
        if st in ('running', 'paused', 'awaiting_confirm', 'awaiting_post_grasp', 'awaiting_post_reach_release'):
            with self._action_lock:
                self._action_abort_requested = True
            self._set_action_state('running')  # release pause/confirm waits
            self._set_status("Abort requested... returning to start",
                             "QLabel { color: orange; font-size: 10px; }")
            self._set_return_enabled(False)
            return

        if self._pre_action_state is None:
            # Fallback: if no per-action snapshot exists, still allow Return button
            # to send robot to the configured startup/home pose.
            self._set_return_enabled(False)
            self._set_status(
                "No saved action state. Returning to startup home pose...",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )

            def run_home_only():
                try:
                    ok = bool(
                        self._move_to_startup_pose_for_action(
                            timeout_s=12.0,
                            keep_current_base_x=False,
                            retract_before_lower=True,
                        )
                    )
                    if ok:
                        self.ros_node.sync_command_targets_to_actual()
                        self._set_status(
                            "Returned to startup home pose",
                            "QLabel { color: green; font-size: 10px; }",
                        )
                    else:
                        self._set_status(
                            "Return home failed (startup pose)",
                            "QLabel { color: red; font-size: 10px; }",
                        )
                except Exception as e:
                    self._set_status(f"Return home failed: {e}", "QLabel { color: red; font-size: 10px; }")
                finally:
                    self._set_return_enabled(True)
                    self._set_action_state('idle')

            from threading import Thread
            Thread(target=run_home_only, daemon=True).start()
            return

        # For auto-loop: user may have manually adjusted reach/place before pressing Return.
        self._maybe_capture_reach_pose_before_return_for_auto()
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

                def _restore_manip_base_x(
                    target_base_x: float,
                    attempts: int = RETURN_BASE_X_RESTORE_ATTEMPTS,
                    tol_m: float = RETURN_BASE_X_RESTORE_TOL_M,
                ) -> None:
                    """Best-effort manip base_x convergence with verification/retry."""
                    target = float(target_base_x)
                    for attempt in range(int(max(1, attempts))):
                        cur_joint = self._current_manip_joint6()
                        if not (isinstance(cur_joint, list) and len(cur_joint) >= 6):
                            return
                        cur_joint = [float(v) for v in cur_joint[:6]]
                        err = float(target - cur_joint[0])
                        if abs(err) <= float(tol_m):
                            return
                        print(
                            f"Refining base_x (attempt {attempt + 1}/{attempts}): "
                            f"current={cur_joint[0]:+.3f}, target={target:+.3f}, err={err:+.3f}"
                        )
                        cur_joint[0] = float(target)
                        self._execute_arm_to_chunked(
                            cur_joint[:6],
                            gripper=None,
                            timeout_s=8.0,
                            reliable=False,
                        )
                        _stage_sleep(0.25)

                def _ordered_return_lift_retract_base(
                    *,
                    move_base_home: bool,
                    gripper_override: float | None,
                    restore_home_pitch: bool = False,
                    note_prefix: str,
                ) -> bool:
                    """Blocking return order: lift -> retract -> (optional) base_x home."""
                    cur_joint = self._current_manip_joint6()
                    if not (isinstance(cur_joint, list) and len(cur_joint) >= 6):
                        cur_joint = self._default_init_joint6()
                    cur_joint = [float(v) for v in cur_joint[:6]]
                    g_cmd = self._get_manual_gripper_target(
                        fallback=float(self.ros_node.JOINT_LIMITS[7][1])
                    )
                    if isinstance(gripper_override, (int, float)) and math.isfinite(float(gripper_override)):
                        g_cmd = float(gripper_override)
                    if g_cmd is None:
                        g_cmd = float(self.ros_node.JOINT_LIMITS[7][1])
                    g_cmd = float(
                        np.clip(
                            float(g_cmd),
                            float(self.ros_node.JOINT_LIMITS[7][0]),
                            float(self.ros_node.JOINT_LIMITS[7][1]),
                        )
                    )

                    # Ensure gripper command is fully applied before any return motion.
                    self._set_status(
                        f"{note_prefix}: applying gripper command",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not self._execute_arm_to_chunked(
                        cur_joint[:6],
                        gripper=float(g_cmd),
                        timeout_s=6.0,
                        reliable=True,
                    ):
                        return False
                    _stage_sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

                    safe_lift = float(
                        np.clip(
                            max(float(RETURN_SAFE_LIFT_M), float(cur_joint[1])),
                            float(self.ros_node.JOINT_LIMITS[1][0]),
                            float(self.ros_node.JOINT_LIMITS[1][1]),
                        )
                    )
                    step_lift = [float(v) for v in cur_joint[:6]]
                    step_lift[1] = float(safe_lift)
                    self._set_status(
                        f"{note_prefix}: lift to safe height",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not self._execute_arm_to_chunked(
                        step_lift[:6],
                        gripper=float(g_cmd),
                        timeout_s=10.0,
                        reliable=True,
                    ):
                        return False
                    _stage_sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

                    step_retract = [float(v) for v in step_lift[:6]]
                    step_retract[2] = float(RETRACT_EXT)
                    self._set_status(
                        f"{note_prefix}: retract arm to 0",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not self._execute_arm_to_chunked(
                        step_retract[:6],
                        gripper=float(g_cmd),
                        timeout_s=10.0,
                        reliable=True,
                    ):
                        return False
                    _stage_sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))
                    step_after_retract = [float(v) for v in step_retract[:6]]
                    if bool(restore_home_pitch):
                        init_joint6 = self._default_init_joint6()
                        step_after_retract[4] = float(
                            np.clip(
                                float(init_joint6[4]),
                                float(self.ros_node.JOINT_LIMITS[3][0]),
                                float(self.ros_node.JOINT_LIMITS[3][1]),
                            )
                        )
                        self._set_status(
                            f"{note_prefix}: restore pitch home",
                            "QLabel { color: #1e88e5; font-size: 10px; }",
                        )
                        if not self._execute_arm_to_chunked(
                            step_after_retract[:6],
                            gripper=float(g_cmd),
                            timeout_s=10.0,
                            reliable=True,
                        ):
                            return False
                        _stage_sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

                    if bool(move_base_home):
                        home_joint6 = self._default_init_joint6()
                        home_base_x = float(home_joint6[0]) if isinstance(home_joint6, list) and len(home_joint6) >= 1 else 0.0
                        home_base_x = float(
                            np.clip(
                                float(home_base_x),
                                float(MANIP_BASE_X_LIMITS[0]),
                                float(MANIP_BASE_X_LIMITS[1]),
                            )
                        )
                        step_home = [float(v) for v in step_after_retract[:6]]
                        step_home[0] = float(home_base_x)
                        self._set_status(
                            f"{note_prefix}: base_x to home",
                            "QLabel { color: #1e88e5; font-size: 10px; }",
                        )
                        if not self._execute_arm_to_chunked(
                            step_home[:6],
                            gripper=float(g_cmd),
                            timeout_s=10.0,
                            reliable=True,
                        ):
                            return False
                    self._set_manual_gripper_override(float(g_cmd))
                    return True

                rotation = state['rotation_applied']
                saved_base_pose = state.get('base_pose_xytheta_start')
                LIFT_MAX = self.ros_node.JOINT_LIMITS[1][1]
                RETRACT_EXT = 0.0
                saved_base_x = float(state.get('base_x', 0.0))
                saved_wrist_yaw = float(state.get('wrist_yaw', 0.0))
                saved_wrist_pitch = float(state.get('wrist_pitch', 0.0))
                saved_wrist_roll = float(state.get('wrist_roll', 0.0))
                saved_lift = float(state.get('lift', 0.85))
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

                # Auto-loop handoff return:
                # after first manual trial, do not restore base_x/home here.
                # Do safe arm return (lift then retract), then immediately start auto replay.
                if bool(self._auto_start_after_return):
                    goals_now = self._goal_sequence_order() if hasattr(self, "_goal_sequence_order") else []
                    single_goal_kind = ""
                    if isinstance(goals_now, list) and len(goals_now) == 1 and isinstance(goals_now[0], dict):
                        single_goal_kind = str(goals_now[0].get("kind", ""))
                    single_grasp_rotate_handoff = bool(
                        str(getattr(self, "_auto_loop_mode", "")) == "goal_sequence"
                        and bool(getattr(self, "_auto_first_trial_pending", False))
                        and isinstance(goals_now, list)
                        and len(goals_now) == 1
                        and isinstance(goals_now[0], dict)
                        and str(single_goal_kind) == "grasp"
                        and (not bool(goals_now[0].get("post_grasp_lift", True)))
                        and isinstance(goals_now[0].get("grasp_rotate_deg"), (int, float))
                        and abs(float(goals_now[0].get("grasp_rotate_deg"))) > 1e-6
                    )
                    single_reach_handoff = bool(
                        str(getattr(self, "_auto_loop_mode", "")) == "goal_sequence"
                        and bool(getattr(self, "_auto_first_trial_pending", False))
                        and isinstance(goals_now, list)
                        and len(goals_now) == 1
                        and isinstance(goals_now[0], dict)
                        and str(single_goal_kind) in ("reach", "place_object")
                    )
                    if bool(single_reach_handoff):
                        print("Auto-loop return (single reach/place): ordered return including base_x home.")
                        if str(single_goal_kind) == "place_object":
                            g_ret = float(
                                np.clip(
                                    float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT),
                                    float(self.ros_node.JOINT_LIMITS[7][0]),
                                    float(self.ros_node.JOINT_LIMITS[7][1]),
                                )
                            )
                        else:
                            g_ret = self._get_manual_gripper_target(
                                fallback=float(self.ros_node.JOINT_LIMITS[7][1])
                            )
                            if g_ret is None:
                                g_ret = float(self.ros_node.JOINT_LIMITS[7][1])
                        if not _ordered_return_lift_retract_base(
                            move_base_home=True,
                            gripper_override=float(g_ret),
                            note_prefix="Auto-loop return",
                        ):
                            raise RuntimeError("ordered single-reach/place return failed")
                        _stage_sleep(0.10)
                        self._pre_action_state = None
                        self.queued_sequence_started = False
                        self.ros_node.sync_command_targets_to_actual()
                        self._set_status(
                            "Return complete: ordered return done for single-reach/place auto replay.",
                            "QLabel { color: green; font-size: 10px; }",
                        )
                        self._stop_auto_loop_record_segment(save=True)
                        self._set_action_state('idle')
                        self._update_goal_queue_label()
                        self._update_next_goal_button_state()
                        print(
                            "[auto_loop] return complete trigger: "
                            f"armed={self._auto_start_after_return} pending={self._auto_first_trial_pending} "
                            f"has_grasp={isinstance(self._auto_pose_grasp, dict)} "
                            f"has_reach={isinstance(self._auto_pose_reach, dict)} "
                            f"target_forward={self._auto_loop_requested_rounds}"
                        )
                        self._maybe_start_auto_loop_after_return()
                        return
                    print("Auto-loop return: ordered safe arm return (lift + retract); base_x unchanged.")
                    if not _ordered_return_lift_retract_base(
                        move_base_home=False,
                        gripper_override=None,
                        restore_home_pitch=bool(single_grasp_rotate_handoff),
                        note_prefix="Auto-loop return",
                    ):
                        raise RuntimeError("ordered auto-loop return failed")
                    _stage_sleep(0.10)
                    self._pre_action_state = None
                    self.queued_sequence_started = False
                    self.ros_node.sync_command_targets_to_actual()
                    self._set_status(
                        "Return complete: ordered safe arm return done for auto replay (base unchanged).",
                        "QLabel { color: green; font-size: 10px; }",
                    )
                    self._stop_auto_loop_record_segment(save=True)
                    self._set_action_state('idle')
                    self._update_goal_queue_label()
                    self._update_next_goal_button_state()
                    print(
                        "[auto_loop] return complete trigger: "
                        f"armed={self._auto_start_after_return} pending={self._auto_first_trial_pending} "
                        f"has_grasp={isinstance(self._auto_pose_grasp, dict)} "
                        f"has_reach={isinstance(self._auto_pose_reach, dict)} "
                        f"target_forward={self._auto_loop_requested_rounds}"
                    )
                    self._maybe_start_auto_loop_after_return()
                    return

                # Simplified return policy:
                # Ordered manual return:
                # - default: lift -> retract (base unchanged)
                # - single reach/place_object goal: lift -> retract -> base_x home
                goals_now = self._goal_sequence_order() if hasattr(self, "_goal_sequence_order") else []
                single_reach_like = bool(
                    isinstance(goals_now, list)
                    and len(goals_now) == 1
                    and isinstance(goals_now[0], dict)
                    and str(goals_now[0].get("kind", "")) in ("reach", "place_object")
                )
                single_grasp_rotate = bool(
                    isinstance(goals_now, list)
                    and len(goals_now) == 1
                    and isinstance(goals_now[0], dict)
                    and str(goals_now[0].get("kind", "")) == "grasp"
                    and (not bool(goals_now[0].get("post_grasp_lift", True)))
                    and isinstance(goals_now[0].get("grasp_rotate_deg"), (int, float))
                    and abs(float(goals_now[0].get("grasp_rotate_deg"))) > 1e-6
                )
                single_goal_kind = (
                    str(goals_now[0].get("kind", ""))
                    if bool(single_reach_like) and isinstance(goals_now, list) and len(goals_now) == 1 and isinstance(goals_now[0], dict)
                    else ""
                )
                manual_return_gripper = preserved_gripper_target
                if str(single_goal_kind) == "place_object":
                    # Single place-object task: close before return and keep closed at home.
                    manual_return_gripper = float(
                        np.clip(
                            float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT),
                            float(self.ros_node.JOINT_LIMITS[7][0]),
                            float(self.ros_node.JOINT_LIMITS[7][1]),
                        )
                    )
                print(
                    "Manual return: ordered sequence "
                    f"(lift -> retract{' -> base_x home' if single_reach_like else ''})."
                )
                if not _ordered_return_lift_retract_base(
                    move_base_home=bool(single_reach_like),
                    gripper_override=manual_return_gripper,
                    restore_home_pitch=bool(single_grasp_rotate),
                    note_prefix="Manual return",
                ):
                    raise RuntimeError("manual ordered return failed")
                _stage_sleep(0.10)
                self._pre_action_state = None
                self.queued_sequence_started = False
                self.ros_node.sync_command_targets_to_actual()
                if bool(single_reach_like):
                    print("Manual return completed. Base moved home (single reach/place).")
                else:
                    print("Manual return completed. Base unchanged.")
                self._set_status(
                    (
                        "Return complete: ordered return done "
                        "(lift -> retract -> base_x home)."
                        if bool(single_reach_like)
                        else "Return complete: ordered return done (lift -> retract). Base unchanged."
                    ),
                    "QLabel { color: green; font-size: 10px; }",
                )
                self._set_action_state('idle')
                self._update_goal_queue_label()
                self._update_next_goal_button_state()
                # Stop here: startup pose already restored; skip legacy extra
                # return stages that re-raise lift and add redundant motion.
                return

                # Step 3: Optional nav base pose correction.
                # Disabled by default because move_base_relative() can still rotate
                # due planner behavior. For deterministic linear return use manip base_x only.
                if bool(RETURN_USE_NAV_BASE_POSE_CORRECTION):
                    if isinstance(saved_base_pose, list) and len(saved_base_pose) >= 3:
                        sx, sy = float(saved_base_pose[0]), float(saved_base_pose[1])
                        for attempt in range(3):
                            cur_pose = self.ros_node.get_measured_base_pose_xytheta()
                            if not (isinstance(cur_pose, list) and len(cur_pose) >= 3):
                                break
                            cx, cy, ct = float(cur_pose[0]), float(cur_pose[1]), float(cur_pose[2])
                            wx = sx - cx
                            wy = sy - cy
                            dx_rel = math.cos(ct) * wx + math.sin(ct) * wy
                            if abs(dx_rel) <= float(RETURN_BASE_X_RESTORE_TOL_M):
                                break
                            print(
                                "Restoring base linear x "
                                f"(attempt {attempt + 1}/3, dx={dx_rel:+.3f}m)..."
                            )
                            ok_pose = self.ros_node.move_base_relative(
                                dx=float(dx_rel),
                                dy=0.0,
                                dtheta=0.0,
                                blocking=True,
                                timeout_s=max(2.0, 3.0 + 8.0 * abs(dx_rel)),
                            )
                            if not ok_pose:
                                print("  WARNING: bridge move_base_relative x-only restore failed")
                            _stage_sleep(0.60)
                else:
                    # User-preferred return behavior: one direct textbox-style
                    # base translation command (no chunk/retry loop).
                    if isinstance(saved_base_pose, list) and len(saved_base_pose) >= 3:
                        cur_pose = self.ros_node.get_measured_base_pose_xytheta()
                        if isinstance(cur_pose, list) and len(cur_pose) >= 3:
                            sx, sy = float(saved_base_pose[0]), float(saved_base_pose[1])
                            cx, cy, ct = float(cur_pose[0]), float(cur_pose[1]), float(cur_pose[2])
                            wx = sx - cx
                            wy = sy - cy
                            dx_rel = math.cos(ct) * wx + math.sin(ct) * wy
                            if abs(dx_rel) > float(RETURN_BASE_X_RESTORE_TOL_M):
                                print(
                                    "Return base single-shot "
                                    f"(dx={dx_rel:+.3f}m, textbox-style)"
                                )
                                ok_pose = self.ros_node.move_base_relative(
                                    dx=float(dx_rel),
                                    dy=0.0,
                                    dtheta=0.0,
                                    blocking=True,
                                    timeout_s=max(2.0, 3.0 + 8.0 * abs(dx_rel)),
                                )
                                if not ok_pose:
                                    print("  WARNING: single-shot base return command failed")
                                _stage_sleep(0.25)
                            else:
                                print("Return base single-shot skipped (already within 8mm).")
                    else:
                        print("Return base single-shot skipped (missing saved base pose).")

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
                self._execute_arm_to_chunked(
                    restore_joint[:6],
                    gripper=None,
                    timeout_s=10.0,
                    reliable=False,
                )
                _stage_sleep(0.5)

                # Step 5: Restore head pan/tilt
                print(f"Restoring head pan={state['head_pan']:.3f}, tilt={state['head_tilt']:.3f}")
                self._execute_arm_to_chunked(
                    restore_joint[:6],
                    gripper=None,
                    head=[float(state['head_pan']), float(state['head_tilt'])],
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
                self._execute_arm_to_chunked(
                    final_joint[:6],
                    gripper=None,
                    head=[float(state['head_pan']), float(state['head_tilt'])],
                    timeout_s=10.0,
                    reliable=False,
                )
                _stage_sleep(1.5)

                # Make manip base_x fully converge to start value, especially when
                # returning from pre-grasp pause/abort.
                _restore_manip_base_x(
                    saved_base_x,
                    attempts=int(RETURN_BASE_X_RESTORE_ATTEMPTS),
                    tol_m=float(RETURN_BASE_X_RESTORE_TOL_M),
                )

                # Final nav x refine (same toggle as Step-3).
                if bool(RETURN_USE_NAV_BASE_POSE_CORRECTION):
                    if isinstance(saved_base_pose, list) and len(saved_base_pose) >= 3:
                        cur_pose = self.ros_node.get_measured_base_pose_xytheta()
                        if isinstance(cur_pose, list) and len(cur_pose) >= 3:
                            cx, cy, ct = float(cur_pose[0]), float(cur_pose[1]), float(cur_pose[2])
                            sx, sy = float(saved_base_pose[0]), float(saved_base_pose[1])
                            wx = sx - cx
                            wy = sy - cy
                            dx_rel = math.cos(ct) * wx + math.sin(ct) * wy
                            if abs(dx_rel) > float(RETURN_FINAL_NAV_BASE_X_REFINE_TOL_M):
                                print(
                                    "Final base x refine "
                                    f"(dx={dx_rel:+.3f}m)"
                                )
                                self.ros_node.move_base_relative(
                                    dx=float(dx_rel),
                                    dy=0.0,
                                    dtheta=0.0,
                                    blocking=True,
                                    timeout_s=max(2.0, 3.0 + 6.0 * abs(dx_rel)),
                                )
                                _stage_sleep(0.25)

                # Final return state policy:
                # 1) restore configured startup/default pose
                # 2) then park pitch and lift for camera/depth visibility.
                q8_home = self._default_init_q8()
                home_joint6 = self._default_init_joint6()
                print("Return final: restoring configured startup pose before pitch park.")
                self._execute_arm_to_chunked(
                    home_joint6,
                    gripper=float(q8_home[7]),
                    head=[float(q8_home[5]), float(q8_home[6])],
                    timeout_s=12.0,
                    reliable=False,
                )
                _stage_sleep(0.25)
                print(
                    "Return final: parking camera view "
                    f"(pitch={CAMERA_VIEW_PARK_PITCH_RAD:+.3f} rad, lift={CAMERA_VIEW_PARK_LIFT_M:.3f} m)."
                )
                self._park_camera_view_pose(timeout_s=10.0)
                _stage_sleep(0.20)

                self._pre_action_state = None
                self.queued_sequence_started = False
                self.ros_node.sync_command_targets_to_actual()
                print("Return completed!")
                self._set_status("Returned to start position", "QLabel { color: green; font-size: 10px; }")
                self._set_action_state('idle')
                self._update_goal_queue_label()
                self._update_next_goal_button_state()
                print(
                    "[auto_loop] return complete trigger: "
                    f"armed={self._auto_start_after_return} pending={self._auto_first_trial_pending} "
                    f"has_grasp={isinstance(self._auto_pose_grasp, dict)} "
                    f"has_reach={isinstance(self._auto_pose_reach, dict)} "
                    f"target_forward={self._auto_loop_requested_rounds}"
                )
                self._maybe_start_auto_loop_after_return()

            except Exception as e:
                print(f"Return error: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"Return failed: {str(e)}", "QLabel { color: red; font-size: 10px; }")
                self._set_return_enabled(True)
                self._set_action_state('idle')

        from threading import Thread
        Thread(target=run, daemon=True).start()

    def _capture_current_pose_snapshot(self, label: str) -> dict[str, Any] | None:
        joint6 = self._current_manip_joint6()
        if not (isinstance(joint6, list) and len(joint6) >= 6):
            return None
        joint6 = [float(v) for v in joint6[:6]]

        q_target = self.ros_node.get_target_qpos()
        if not (isinstance(q_target, list) and len(q_target) >= 8):
            q_target = self.ros_node.get_actual_qpos()
        if not (isinstance(q_target, list) and len(q_target) >= 8):
            q_target = [0.0] * 8

        try:
            grip = self._get_manual_gripper_target(fallback=float(q_target[7]))
            if grip is None:
                grip = float(q_target[7])
        except Exception:
            grip = float(self.ros_node.JOINT_LIMITS[7][1])

        grip = float(
            np.clip(
                float(grip),
                float(self.ros_node.JOINT_LIMITS[7][0]),
                float(self.ros_node.JOINT_LIMITS[7][1]),
            )
        )
        head = [float(q_target[5]), float(q_target[6])]
        base_pose = self.ros_node.get_measured_base_pose_xytheta()
        if not (isinstance(base_pose, list) and len(base_pose) >= 3):
            base_pose = [0.0, 0.0, 0.0]

        return {
            "label": str(label),
            "joint6": joint6,
            "gripper": grip,
            "head": head,
            "base_pose_xytheta": [float(base_pose[0]), float(base_pose[1]), float(base_pose[2])],
            "captured_at": float(time.time()),
        }

    def _maybe_capture_goal_pose_for_auto(self, kind: str) -> None:
        if str(kind) not in ("grasp", "reach", "place_object"):
            return
        skip_kind = getattr(self, "_skip_next_auto_goal_pose_capture_kind", None)
        if isinstance(skip_kind, str) and str(skip_kind) == str(kind):
            self._skip_next_auto_goal_pose_capture_kind = None
            return
        # Keep first/manual-trial learned poses stable during auto replay.
        if bool(self._auto_loop_running) and (not bool(self._auto_first_trial_pending)):
            return
        snap_label = f"goal_{kind}_final"
        snap = self._capture_current_pose_snapshot(snap_label)
        if not isinstance(snap, dict):
            return
        if str(kind) == "grasp":
            self._auto_pose_grasp = snap
            if self._auto_gripper_closed is None:
                self._auto_gripper_closed = float(snap["gripper"])
            print(
                "[auto_loop] captured grasp pose: "
                f"joint6={self._auto_pose_grasp['joint6']}, grip={self._auto_pose_grasp['gripper']:.3f}"
            )
        elif str(kind) == "place_object":
            self._auto_pose_place_object = snap
            self._auto_pose_place_object_target = dict(snap)
            # Compatibility: current pick-place auto-loop still reads reach slots.
            self._auto_pose_reach = dict(snap)
            self._auto_pose_reach_target = dict(snap)
            self._auto_gripper_open = float(snap["gripper"])
            print(
                "[auto_loop] captured place_object pose: "
                f"joint6={self._auto_pose_place_object['joint6']}, grip={self._auto_pose_place_object['gripper']:.3f}"
            )
        else:
            self._auto_pose_reach = snap
            self._auto_pose_reach_target = dict(snap)
            self._auto_gripper_open = float(snap["gripper"])
            print(
                "[auto_loop] captured reach pose: "
                f"joint6={self._auto_pose_reach['joint6']}, grip={self._auto_pose_reach['gripper']:.3f}"
            )

    def _maybe_capture_reach_pose_before_return_for_auto(self) -> None:
        """Capture final user-corrected reach/place pose before return starts."""
        if not bool(self._auto_first_trial_pending):
            return
        goals = self._goal_sequence_order()
        single_place_object = bool(
            isinstance(goals, list)
            and len(goals) == 1
            and isinstance(goals[0], dict)
            and str(goals[0].get("kind", "")) == "place_object"
        )
        snap = self._capture_current_pose_snapshot("reach_before_return")
        if not isinstance(snap, dict):
            return
        if bool(single_place_object):
            self._auto_pose_place_object = snap
            self._auto_pose_place_object_target = dict(snap)
        else:
            self._auto_pose_reach = snap
            self._auto_pose_reach_target = dict(snap)
        self._auto_gripper_open = float(snap["gripper"])
        print(
            "[auto_loop] captured reach pose before return: "
            f"joint6={snap['joint6']}, grip={snap['gripper']:.3f}"
        )

    def start_auto_pick_place_loop(self, rounds: int) -> bool:
        """Arm automation: run current queued grasp->reach once, then repeat learned loop."""
        r = int(rounds)
        if r <= 0:
            self._set_status("Auto loop count must be >= 1", "QLabel { color: red; font-size: 10px; }")
            return False
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Auto loop can start only while idle", "QLabel { color: orange; font-size: 10px; }")
                return False
        grasp_goal = self.queued_goals.get("grasp")
        if not (
            isinstance(grasp_goal, dict)
            and bool(grasp_goal.get("post_grasp_lift", True))
            and isinstance(self.queued_goals.get("reach"), dict)
        ):
            self._set_status(
                "Auto loop requires queued goals: grasp+lift and reach",
                "QLabel { color: red; font-size: 10px; }",
            )
            return False
        if self._auto_loop_running:
            self._set_status("Auto loop already running", "QLabel { color: orange; font-size: 10px; }")
            return False

        self._auto_loop_requested_rounds = int(r)
        self._auto_rounds_left = int(r)
        self._update_auto_loop_progress_ui()
        self._auto_loop_abort = False
        self._auto_loop_running = False
        self._auto_loop_mode = "pick_place"
        self._auto_sequence_replay_active = False
        self._auto_first_trial_pending = True
        self._auto_start_after_return = True
        self._auto_capture_enabled = True
        self._auto_pose_home = self._capture_current_pose_snapshot("auto_home_before_first_trial")
        self._auto_pose_grasp = None
        self._auto_pose_reach = None
        self._auto_pose_release = None
        self._auto_pose_grasp_target = None
        self._auto_pose_reach_target = None
        self._auto_gripper_open = None
        self._auto_gripper_closed = None
        self._auto_initial_base_x = None
        if isinstance(self._auto_pose_home, dict):
            j6 = self._auto_pose_home.get("joint6")
            if isinstance(j6, list) and len(j6) >= 1:
                try:
                    bx = float(j6[0])
                    if math.isfinite(bx):
                        self._auto_initial_base_x = float(bx)
                except Exception:
                    self._auto_initial_base_x = None

        if not self._start_auto_loop_record_session(int(r)):
            self._auto_first_trial_pending = False
            self._auto_start_after_return = False
            return False

        self._set_status(
            f"Auto loop armed: first manual trial starts now (repeats after first trial={r}). "
            "After first trial, press Return once to begin auto replay.",
            "QLabel { color: #1e88e5; font-size: 10px; }",
        )
        self.execute_all_queued_goals()
        return True

    def stop_auto_pick_place_loop(self) -> None:
        self._auto_loop_abort = True
        self._auto_first_trial_pending = False
        self._auto_start_after_return = False
        self._auto_sequence_replay_active = False
        if self._loop_record_session_active:
            self._loop_record_stop_requested = True
            if not self._auto_loop_running:
                self._stop_auto_loop_record_segment(save=False)
                if len(self._loop_record_entries) > 0:
                    self.ui_loop_record_review_signal.emit()
                else:
                    self._update_auto_loop_record_button_ui()
                    self._reset_auto_loop_record_session_state()
        if self._auto_loop_running:
            self._set_status("Auto loop stop requested", "QLabel { color: orange; font-size: 10px; }")
        else:
            self._set_status("Auto loop disarmed", "QLabel { color: gray; font-size: 10px; }")
        self._update_auto_loop_progress_ui()

    def _maybe_start_auto_loop_after_return(self) -> None:
        if not bool(self._auto_start_after_return):
            print("[auto_loop] return completed: auto loop not armed; skipping replay.")
            return
        self._auto_first_trial_pending = False
        self._auto_start_after_return = False
        if not (isinstance(self._auto_pose_grasp, dict) and isinstance(self._auto_pose_reach, dict)):
            self._set_status(
                "Auto loop could not start: missing learned grasp/reach poses.",
                "QLabel { color: red; font-size: 10px; }",
            )
            self._auto_capture_enabled = False
            return
        if not isinstance(self._auto_pose_home, dict):
            self._auto_pose_home = self._capture_current_pose_snapshot("auto_home_fallback")

        self._auto_loop_running = True
        self._auto_capture_enabled = False
        self._update_auto_loop_progress_ui()
        from threading import Thread
        Thread(target=self._run_auto_pick_place_loop, daemon=True).start()

    def _run_auto_pick_place_loop(self) -> None:
        loop_success = False

        def _stage_wait(t_s: float) -> bool:
            end_t = time.time() + max(0.0, float(t_s))
            while time.time() < end_t:
                if self._auto_loop_abort:
                    return False
                time.sleep(0.02)
            return True

        def _clip_joint6(joint6: list[float]) -> list[float]:
            j = [float(v) for v in joint6[:6]]
            j[0] = float(np.clip(j[0], float(MANIP_BASE_X_LIMITS[0]), float(MANIP_BASE_X_LIMITS[1])))
            j[1] = float(np.clip(j[1], float(self.ros_node.JOINT_LIMITS[1][0]), float(self.ros_node.JOINT_LIMITS[1][1])))
            j[2] = float(np.clip(j[2], float(self.ros_node.JOINT_LIMITS[0][0]), float(self.ros_node.JOINT_LIMITS[0][1])))
            j[3] = float(np.clip(j[3], float(self.ros_node.JOINT_LIMITS[2][0]), float(self.ros_node.JOINT_LIMITS[2][1])))
            j[4] = float(np.clip(j[4], float(self.ros_node.JOINT_LIMITS[3][0]), float(self.ros_node.JOINT_LIMITS[3][1])))
            j[5] = float(np.clip(j[5], float(self.ros_node.JOINT_LIMITS[4][0]), float(self.ros_node.JOINT_LIMITS[4][1])))
            return j

        def _pose_joint6(
            pose: dict[str, Any],
            *,
            lift_override: float | None = None,
            yaw_offset_rad: float = 0.0,
            base_x_offset_m: float = 0.0,
            arm_offset_m: float = 0.0,
        ) -> list[float] | None:
            joint6 = pose.get("joint6")
            if not (isinstance(joint6, list) and len(joint6) >= 6):
                return None
            j = _clip_joint6([float(v) for v in joint6[:6]])
            if abs(float(yaw_offset_rad)) > 0.0:
                j[3] = float(
                    np.clip(
                        float(j[3]) + float(yaw_offset_rad),
                        float(self.ros_node.JOINT_LIMITS[2][0]),
                        float(self.ros_node.JOINT_LIMITS[2][1]),
                    )
                )
            if lift_override is not None:
                j[1] = float(
                    np.clip(
                        float(lift_override),
                        float(self.ros_node.JOINT_LIMITS[1][0]),
                        float(self.ros_node.JOINT_LIMITS[1][1]),
                    )
                )
            if abs(float(base_x_offset_m)) > 0.0:
                j[0] = float(
                    np.clip(
                        float(j[0]) + float(base_x_offset_m),
                        float(MANIP_BASE_X_LIMITS[0]),
                        float(MANIP_BASE_X_LIMITS[1]),
                    )
                )
            if abs(float(arm_offset_m)) > 0.0:
                j[2] = float(
                    np.clip(
                        float(j[2]) + float(arm_offset_m),
                        float(self.ros_node.JOINT_LIMITS[0][0]),
                        float(self.ros_node.JOINT_LIMITS[0][1]),
                    )
                )
            return j

        def _pose_lift(pose: dict[str, Any] | None) -> float | None:
            if not isinstance(pose, dict):
                return None
            joint6 = pose.get("joint6")
            if not (isinstance(joint6, list) and len(joint6) >= 2):
                return None
            try:
                lv = float(joint6[1])
            except Exception:
                return None
            if not math.isfinite(lv):
                return None
            return float(lv)

        def _compute_transit_lift() -> float:
            candidates: list[float] = [float(IK_SAFE_LIFT_M)]
            for p in (
                self._auto_pose_home,
                self._auto_pose_grasp,
                self._auto_pose_reach,
            ):
                lv = _pose_lift(p)
                if lv is not None:
                    candidates.append(float(lv))
            return float(
                np.clip(
                    max(candidates),
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )

        def _variation_safe_lift(target_pose: dict[str, Any], transit_lift: float) -> float:
            """Lift used for random pre-approach/pre-place so we don't hit/displace object."""
            target_lift = _pose_lift(target_pose)
            clearance = max(0.0, float(AUTO_LOOP_PICK_VARIATION_MIN_CLEARANCE_M))
            if target_lift is None:
                desired = float(transit_lift)
            else:
                desired = max(float(transit_lift), float(target_lift) + float(clearance))
            return float(
                np.clip(
                    desired,
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )

        def _move_pose(
            pose: dict[str, Any],
            *,
            gripper: float | None,
            note: str,
            timeout_s: float = 10.0,
            lift_override: float | None = None,
            yaw_offset_rad: float = 0.0,
            base_x_offset_m: float = 0.0,
            arm_offset_m: float = 0.0,
        ) -> bool:
            if self._auto_loop_abort:
                return False
            joint6 = _pose_joint6(
                pose,
                lift_override=lift_override,
                yaw_offset_rad=float(yaw_offset_rad),
                base_x_offset_m=float(base_x_offset_m),
                arm_offset_m=float(arm_offset_m),
            )
            if not (isinstance(joint6, list) and len(joint6) >= 6):
                return False
            head = pose.get("head")
            head_cmd = None
            if isinstance(head, list) and len(head) >= 2:
                head_cmd = [float(head[0]), float(head[1])]
            grip_cmd = gripper
            if grip_cmd is None:
                g = pose.get("gripper")
                if isinstance(g, (int, float)):
                    grip_cmd = float(g)
            self._set_status(note, "QLabel { color: #1e88e5; font-size: 10px; }")
            return bool(
                self._execute_arm_to_chunked(
                    [float(v) for v in joint6],
                    gripper=grip_cmd,
                    head=head_cmd,
                    timeout_s=float(timeout_s),
                    reliable=False,
                )
            )

        def _sample_grasp_place_jitter() -> tuple[float, float]:
            max_bx = max(0.0, float(AUTO_LOOP_GRASP_PLACE_JITTER_BASE_X_M))
            max_arm = max(0.0, float(AUTO_LOOP_GRASP_PLACE_JITTER_ARM_M))
            jitter_bx = float(np.random.uniform(-max_bx, +max_bx)) if max_bx > 0.0 else 0.0
            jitter_arm = float(np.random.uniform(-max_arm, +max_arm)) if max_arm > 0.0 else 0.0
            return jitter_bx, jitter_arm

        def _sample_pick_variation() -> dict[str, float | str]:
            """Random pre-approach mode used before exact pickup target."""
            if not bool(AUTO_LOOP_PICK_VARIATION_ENABLED):
                return {
                    "mode": "direct",
                    "base_x_offset_m": 0.0,
                    "arm_offset_m": 0.0,
                    "yaw_offset_rad": 0.0,
                }

            modes = ("direct", "overshoot", "side", "short")
            probs = np.array(
                [
                    float(AUTO_LOOP_PICK_VARIATION_W_DIRECT),
                    float(AUTO_LOOP_PICK_VARIATION_W_OVERSHOOT),
                    float(AUTO_LOOP_PICK_VARIATION_W_SIDE),
                    float(AUTO_LOOP_PICK_VARIATION_W_SHORT),
                ],
                dtype=np.float64,
            )
            probs = np.clip(probs, 0.0, None)
            if float(np.sum(probs)) <= 0.0:
                mode = "direct"
            else:
                probs = probs / float(np.sum(probs))
                mode = str(np.random.choice(modes, p=probs))

            base_off = 0.0
            arm_off = 0.0
            yaw_off = 0.0
            if mode == "overshoot":
                arm_off = +float(np.random.uniform(0.0, abs(float(AUTO_LOOP_PICK_VARIATION_OVERSHOOT_ARM_M))))
                base_off = +abs(float(AUTO_LOOP_PICK_VARIATION_OVERSHOOT_BASE_X_M))
            elif mode == "short":
                arm_off = -abs(float(AUTO_LOOP_PICK_VARIATION_SHORT_ARM_M))
                base_off = -abs(float(AUTO_LOOP_PICK_VARIATION_SHORT_BASE_X_M))
            elif mode == "side":
                s = -1.0 if float(np.random.rand()) < 0.5 else +1.0
                base_off = s * abs(float(AUTO_LOOP_PICK_VARIATION_SIDE_BASE_X_M))
                yaw_off = float(np.deg2rad(s * abs(float(AUTO_LOOP_PICK_VARIATION_SIDE_YAW_DEG))))

            return {
                "mode": mode,
                "base_x_offset_m": float(base_off),
                "arm_offset_m": float(arm_off),
                "yaw_offset_rad": float(yaw_off),
            }

        def _pose_with_offsets(pose: dict[str, Any], *, base_x_offset_m: float = 0.0, arm_offset_m: float = 0.0) -> dict[str, Any]:
            out = dict(pose)
            j = _pose_joint6(
                pose,
                base_x_offset_m=float(base_x_offset_m),
                arm_offset_m=float(arm_offset_m),
            )
            if isinstance(j, list) and len(j) >= 6:
                out["joint6"] = [float(v) for v in j[:6]]
            return out

        def _pick_from(
            lifted_pose: dict[str, Any],
            target_pose: dict[str, Any],
            label: str,
            *,
            transit_lift: float,
            g_open: float,
            g_close: float,
            use_variation: bool = False,
        ) -> bool:
            if not _move_pose(
                lifted_pose,
                gripper=float(g_open),
                note=f"Auto: {label} pick transit",
                timeout_s=10.0,
                lift_override=float(transit_lift),
            ):
                return False
            if bool(use_variation):
                pick_var = _sample_pick_variation()
                pick_mode = str(pick_var.get("mode", "direct"))
            else:
                pick_mode = "direct"
                pick_var = {"base_x_offset_m": 0.0, "arm_offset_m": 0.0, "yaw_offset_rad": 0.0}
            if pick_mode != "direct":
                var_bx = float(pick_var.get("base_x_offset_m", 0.0))
                var_arm = float(pick_var.get("arm_offset_m", 0.0))
                var_yaw = float(pick_var.get("yaw_offset_rad", 0.0))
                print(
                    "[auto_loop pick variation] "
                    f"{label}: mode={pick_mode} base_x={var_bx:+.3f}m arm={var_arm:+.3f}m "
                    f"yaw={np.degrees(var_yaw):+.1f}deg"
                )
                if not _move_pose(
                    target_pose,
                    gripper=float(g_open),
                    note=f"Auto: {label} pre-approach ({pick_mode})",
                    timeout_s=10.0,
                    lift_override=float(_variation_safe_lift(target_pose, float(transit_lift))),
                    yaw_offset_rad=float(var_yaw),
                    base_x_offset_m=float(var_bx),
                    arm_offset_m=float(var_arm),
                ):
                    return False
            if not _move_pose(
                target_pose,
                gripper=float(g_open),
                note=f"Auto: {label} pick lower",
                timeout_s=10.0,
            ):
                return False
            if not _stage_wait(0.12):
                return False
            if not _move_pose(
                target_pose,
                gripper=float(g_close),
                note=f"Auto: {label} close gripper",
                timeout_s=8.0,
            ):
                return False
            if not _stage_wait(0.12):
                return False
            return _move_pose(
                lifted_pose,
                gripper=float(g_close),
                note=f"Auto: {label} pick raise",
                timeout_s=10.0,
                lift_override=float(transit_lift),
            )

        def _place_to(
            lifted_pose: dict[str, Any],
            target_pose: dict[str, Any],
            label: str,
            *,
            transit_lift: float,
            g_open: float,
            g_close: float,
            base_x_offset_m: float = 0.0,
            arm_offset_m: float = 0.0,
            use_variation: bool = False,
        ) -> bool:
            if not _move_pose(
                lifted_pose,
                gripper=float(g_close),
                note=f"Auto: {label} place transit",
                timeout_s=10.0,
                lift_override=float(transit_lift),
                base_x_offset_m=float(base_x_offset_m),
                arm_offset_m=float(arm_offset_m),
            ):
                return False
            if bool(use_variation):
                place_var = _sample_pick_variation()
                place_mode = str(place_var.get("mode", "direct"))
            else:
                place_mode = "direct"
                place_var = {"base_x_offset_m": 0.0, "arm_offset_m": 0.0, "yaw_offset_rad": 0.0}
            if place_mode != "direct":
                var_bx = float(place_var.get("base_x_offset_m", 0.0))
                var_arm = float(place_var.get("arm_offset_m", 0.0))
                var_yaw = float(place_var.get("yaw_offset_rad", 0.0))
                print(
                    "[auto_loop place variation] "
                    f"{label}: mode={place_mode} base_x={var_bx:+.3f}m arm={var_arm:+.3f}m "
                    f"yaw={np.degrees(var_yaw):+.1f}deg"
                )
                if not _move_pose(
                    target_pose,
                    gripper=float(g_close),
                    note=f"Auto: {label} pre-place ({place_mode})",
                    timeout_s=10.0,
                    lift_override=float(_variation_safe_lift(target_pose, float(transit_lift))),
                    yaw_offset_rad=float(var_yaw),
                    base_x_offset_m=float(base_x_offset_m) + float(var_bx),
                    arm_offset_m=float(arm_offset_m) + float(var_arm),
                ):
                    return False
            if not _move_pose(
                target_pose,
                gripper=float(g_close),
                note=f"Auto: {label} place lower",
                timeout_s=10.0,
                base_x_offset_m=float(base_x_offset_m),
                arm_offset_m=float(arm_offset_m),
            ):
                return False
            if not _stage_wait(0.12):
                return False
            if not _move_pose(
                target_pose,
                gripper=float(g_open),
                note=f"Auto: {label} open gripper",
                timeout_s=8.0,
                base_x_offset_m=float(base_x_offset_m),
                arm_offset_m=float(arm_offset_m),
            ):
                return False
            if not _stage_wait(0.12):
                return False
            return _move_pose(
                lifted_pose,
                gripper=float(g_open),
                note=f"Auto: {label} place raise",
                timeout_s=10.0,
                lift_override=float(transit_lift),
                base_x_offset_m=float(base_x_offset_m),
                arm_offset_m=float(arm_offset_m),
            )

        def _restore_initial_base_x(*, g_open: float) -> bool:
            if self._auto_initial_base_x is None:
                return True
            cur = self._current_manip_joint6()
            if not (isinstance(cur, list) and len(cur) >= 6):
                return False
            tgt = [float(v) for v in cur[:6]]
            tgt[0] = float(
                np.clip(
                    float(self._auto_initial_base_x),
                    float(MANIP_BASE_X_LIMITS[0]),
                    float(MANIP_BASE_X_LIMITS[1]),
                )
            )
            self._set_status(
                f"Auto loop final: restoring initial base_x={tgt[0]:+.3f}",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            return bool(
                self._execute_arm_to_chunked(
                    tgt,
                    gripper=float(g_open),
                    timeout_s=10.0,
                    reliable=False,
                )
            )

        def _initial_base_x_value() -> float:
            if isinstance(self._auto_initial_base_x, (int, float)) and math.isfinite(float(self._auto_initial_base_x)):
                return float(
                    np.clip(
                        float(self._auto_initial_base_x),
                        float(MANIP_BASE_X_LIMITS[0]),
                        float(MANIP_BASE_X_LIMITS[1]),
                    )
                )
            return 0.0

        def _unpark_to_initial_action_pose() -> bool:
            bx = _initial_base_x_value()
            self._set_status(
                f"Auto: preparing action pose (base_x={bx:+.3f})",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            return bool(
                self._move_to_startup_pose_for_action(
                    timeout_s=12.0,
                    keep_current_base_x=False,
                    base_x_override=float(bx),
                    gripper_override=float(g_open),
                )
            )

        def _return_to_initial_parked_state(*, g_open: float) -> bool:
            bx = _initial_base_x_value()
            self._set_status(
                f"Auto: returning to initial state (base_x={bx:+.3f})",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            self._set_manual_gripper_override(float(g_open))
            if not _restore_initial_base_x(g_open=float(g_open)):
                return False
            if not self._move_to_startup_pose_for_action(
                timeout_s=12.0,
                keep_current_base_x=False,
                base_x_override=float(bx),
                retract_before_lower=True,
                gripper_override=float(g_open),
            ):
                return False
            return bool(self._park_camera_view_pose(timeout_s=10.0))

        try:
            if not self._begin_action("auto_loop"):
                self._auto_loop_running = False
                return
            repeats_after_first = int(max(1, self._auto_loop_requested_rounds))
            g_open = float(DEVICE_GRIPPER_TOGGLE_OPEN_JOINT)
            g_close = float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT)
            has_learned_open = bool(
                isinstance(self._auto_gripper_open, (int, float))
                and math.isfinite(float(self._auto_gripper_open))
            )
            has_learned_close = bool(
                isinstance(self._auto_gripper_closed, (int, float))
                and math.isfinite(float(self._auto_gripper_closed))
            )
            if bool(has_learned_open):
                g_open = float(self._auto_gripper_open)
            if bool(has_learned_close):
                g_close = float(self._auto_gripper_closed)
            g_open = float(
                np.clip(
                    float(g_open),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            g_close = float(
                np.clip(
                    float(g_close),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            if not bool(has_learned_close):
                g_close = float(min(float(g_close), float(g_open)))
            print(
                "[auto_pick_place gripper pair] "
                f"open={float(g_open):+.4f} close={float(g_close):+.4f} "
                f"(learned_open={has_learned_open} learned_close={has_learned_close})"
            )

            grasp_lifted = self._auto_pose_grasp if isinstance(self._auto_pose_grasp, dict) else None
            reach_lifted = self._auto_pose_reach if isinstance(self._auto_pose_reach, dict) else None
            grasp_target = self._auto_pose_grasp_target if isinstance(self._auto_pose_grasp_target, dict) else grasp_lifted
            reach_target = self._auto_pose_reach_target if isinstance(self._auto_pose_reach_target, dict) else reach_lifted
            if not (isinstance(grasp_lifted, dict) and isinstance(reach_lifted, dict)):
                raise RuntimeError("auto loop missing grasp/reach lifted poses")
            if not (isinstance(grasp_target, dict) and isinstance(reach_target, dict)):
                raise RuntimeError("auto loop missing grasp/reach target poses")
            # Keep first-round learned poses as immutable bases for jitter generation.
            base_grasp_lifted = dict(grasp_lifted)
            base_grasp_target = dict(grasp_target)
            base_reach_lifted = dict(reach_lifted)
            base_reach_target = dict(reach_target)
            # Current object-location poses (updated after each bring-back placement).
            current_grasp_lifted = dict(base_grasp_lifted)
            current_grasp_target = dict(base_grasp_target)
            current_reach_lifted = dict(base_reach_lifted)
            current_reach_target = dict(base_reach_target)

            transit_lift = _compute_transit_lift()
            self._set_status(
                "Auto loop started: "
                f"repeats after first={repeats_after_first}, transit_lift={transit_lift:.3f}m.",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            # Start first bring-back from the current post-return point.
            # (Do not force initial pose before this step.)

            # repeats_after_first means how many forward grasp->reach rounds run
            # after the first manual trial.
            for rep_idx in range(1, repeats_after_first + 1):
                if self._auto_loop_abort:
                    break
                self._auto_rounds_left = int(repeats_after_first - rep_idx + 1)

                # 1) Bring object back: reach -> grasp.
                if not self._start_auto_loop_record_segment(kind="type2", round_idx=int(rep_idx - 1)):
                    raise RuntimeError("auto-loop record failed to start type2 segment")
                self._set_status(
                    f"Auto reset {rep_idx}/{repeats_after_first}: bring back (reach -> grasp)",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not _pick_from(
                    current_reach_lifted,
                    current_reach_target,
                    "reach",
                    transit_lift=float(transit_lift),
                    g_open=float(g_open),
                    g_close=float(g_close),
                    use_variation=False,
                ):
                    raise RuntimeError("auto reset pick from reach failed")
                jitter_bx, jitter_arm = _sample_grasp_place_jitter()
                if abs(float(jitter_bx)) > 0.0 or abs(float(jitter_arm)) > 0.0:
                    print(
                        "[auto_loop jitter] "
                        f"reset_place_to_grasp base_x={jitter_bx:+.3f}m arm={jitter_arm:+.3f}m"
                    )
                jitter_grasp_lifted = _pose_with_offsets(
                    base_grasp_lifted,
                    base_x_offset_m=float(jitter_bx),
                    arm_offset_m=float(jitter_arm),
                )
                jitter_grasp_target = _pose_with_offsets(
                    base_grasp_target,
                    base_x_offset_m=float(jitter_bx),
                    arm_offset_m=float(jitter_arm),
                )
                if not _place_to(
                    jitter_grasp_lifted,
                    jitter_grasp_target,
                    "grasp",
                    transit_lift=float(transit_lift),
                    g_open=float(g_open),
                    g_close=float(g_close),
                    use_variation=False,
                ):
                    raise RuntimeError("auto reset place to grasp failed")
                current_grasp_lifted = dict(jitter_grasp_lifted)
                current_grasp_target = dict(jitter_grasp_target)
                if not _stage_wait(0.15):
                    raise RuntimeError("auto reset interrupted")

                # 2) Return robot to initial state (including initial base_x), then park.
                if not _return_to_initial_parked_state(g_open=float(g_open)):
                    raise RuntimeError("auto reset return-to-initial failed")
                if not _stage_wait(0.15):
                    raise RuntimeError("auto reset return interrupted")
                self._stop_auto_loop_record_segment(save=True)

                # 3) Start repeat forward round: grasp -> reach.
                if not self._start_auto_loop_record_segment(kind="type1", round_idx=int(rep_idx)):
                    raise RuntimeError("auto-loop record failed to start type1 segment")
                self._set_status(
                    f"Auto repeat {rep_idx}/{repeats_after_first}: grasp -> reach",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not _unpark_to_initial_action_pose():
                    raise RuntimeError("auto repeat pre-motion prepare failed")
                if not _pick_from(
                    current_grasp_lifted,
                    current_grasp_target,
                    "grasp",
                    transit_lift=float(transit_lift),
                    g_open=float(g_open),
                    g_close=float(g_close),
                    use_variation=True,
                ):
                    raise RuntimeError("auto repeat pick from grasp failed")
                if not _place_to(
                    current_reach_lifted,
                    current_reach_target,
                    "reach",
                    transit_lift=float(transit_lift),
                    g_open=float(g_open),
                    g_close=float(g_close),
                    use_variation=True,
                ):
                    raise RuntimeError("auto repeat place to reach failed")
                if not _stage_wait(0.15):
                    raise RuntimeError("auto repeat interrupted")
                self._stop_auto_loop_record_segment(save=True)

            if self._auto_loop_abort:
                self._set_status("Auto loop aborted", "QLabel { color: orange; font-size: 10px; }")
            else:
                # Final post-loop cleanup:
                # bring object back to initial grasp location and return robot initial state.
                if not self._start_auto_loop_record_segment(
                    kind="type2",
                    round_idx=int(repeats_after_first),
                ):
                    raise RuntimeError("auto-loop record failed to start final type2 segment")
                self._set_status(
                    "Auto final: bring back object (reach -> grasp), then return initial.",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not _pick_from(
                    current_reach_lifted,
                    current_reach_target,
                    "reach",
                    transit_lift=float(transit_lift),
                    g_open=float(g_open),
                    g_close=float(g_close),
                    use_variation=False,
                ):
                    raise RuntimeError("auto final pick from reach failed")
                jitter_bx, jitter_arm = _sample_grasp_place_jitter()
                if abs(float(jitter_bx)) > 0.0 or abs(float(jitter_arm)) > 0.0:
                    print(
                        "[auto_loop jitter] "
                        f"final_place_to_grasp base_x={jitter_bx:+.3f}m arm={jitter_arm:+.3f}m"
                    )
                jitter_grasp_lifted = _pose_with_offsets(
                    base_grasp_lifted,
                    base_x_offset_m=float(jitter_bx),
                    arm_offset_m=float(jitter_arm),
                )
                jitter_grasp_target = _pose_with_offsets(
                    base_grasp_target,
                    base_x_offset_m=float(jitter_bx),
                    arm_offset_m=float(jitter_arm),
                )
                if not _place_to(
                    jitter_grasp_lifted,
                    jitter_grasp_target,
                    "grasp",
                    transit_lift=float(transit_lift),
                    g_open=float(g_open),
                    g_close=float(g_close),
                    use_variation=False,
                ):
                    raise RuntimeError("auto final place to grasp failed")
                if not _return_to_initial_parked_state(g_open=float(g_open)):
                    raise RuntimeError("auto final return-to-initial failed")
                self._stop_auto_loop_record_segment(save=True)
                self._set_status("Auto loop completed", "QLabel { color: green; font-size: 10px; }")
                loop_success = True
        except Exception as exc:
            self._stop_auto_loop_record_segment(save=False)
            self._set_status(f"Auto loop failed: {exc}", "QLabel { color: red; font-size: 10px; }")
            print(f"[auto_loop] error: {exc}")
            traceback.print_exc()
        finally:
            if self._loop_record_session_active:
                if loop_success and not self._auto_loop_abort:
                    self.ui_loop_record_review_signal.emit()
                else:
                    self._stop_auto_loop_record_segment(save=False)
                    if len(self._loop_record_entries) > 0:
                        # Even on unexpected loop failure (e.g. connection drop),
                        # keep completed segments and let user decide via review table.
                        self.ui_loop_record_review_signal.emit()
                    elif self._auto_loop_abort or bool(self._loop_record_stop_requested):
                        self._update_auto_loop_record_button_ui()
                        self._reset_auto_loop_record_session_state()
                    else:
                        self._abort_auto_loop_record_session(keep_existing_files=False)
            self._auto_loop_running = False
            self._auto_loop_abort = False
            self._auto_rounds_left = 0
            self._auto_start_after_return = False
            self._auto_sequence_replay_active = False
            self._auto_loop_mode = "pick_place"
            self._set_action_state("idle")
            QTimer.singleShot(0, self._update_auto_loop_progress_ui)

    def closeEvent(self, event):
        """Handle window close"""
        print("Closing application...", flush=True)
        app_inst = QApplication.instance()
        if app_inst is not None:
            try:
                app_inst.removeEventFilter(self)
            except Exception:
                pass
        if self.is_recording_demo:
            self.stop_demo_recording()
        self.control_timer.stop()
        if hasattr(self, "device_timer"):
            self.device_timer.stop()
        if hasattr(self, "device_bridge"):
            try:
                self.device_bridge.stop()
            except Exception:
                pass
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

        # Reorder top controls: segment/object/task first; variation controls appear
        # just before loop settings.
        if hasattr(self, "_variation_controls_widget") and isinstance(self._variation_controls_widget, QWidget):
            layout.removeWidget(self._variation_controls_widget)
        if hasattr(self, "return_pause_widget") and isinstance(self.return_pause_widget, QWidget):
            layout.removeWidget(self.return_pause_widget)

        if hasattr(self, "next_goal_button"):
            self.next_goal_button.setText("Next Goal (Manual)")
            self.next_goal_button.setToolTip(
                "Manually run/skip one queued goal. Use Execute Goals for full queued run."
            )

        self.execute_goals_button = QPushButton("Execute Goals")
        self.execute_goals_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.execute_goals_button.clicked.connect(self.execute_all_queued_goals)
        self.execute_goals_button.setVisible(False)
        self.execute_goals_button.setEnabled(False)
        layout.addWidget(self.execute_goals_button)

        self.abort_goals_button = QPushButton("Abort + Clear Goals")
        self.abort_goals_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.abort_goals_button.clicked.connect(self.abort_and_clear_goals)
        self.abort_goals_button.setVisible(False)
        self.abort_goals_button.setEnabled(False)
        layout.addWidget(self.abort_goals_button)

        self.goal_list_widget = QListWidget()
        self.goal_list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.goal_list_widget.setMinimumHeight(55)
        self.goal_list_widget.setMaximumHeight(90)
        self.goal_list_widget.itemSelectionChanged.connect(self._on_goal_selection_changed)
        layout.addWidget(self.goal_list_widget)

        row = QHBoxLayout()
        self.remove_goal_button = QPushButton("Remove Selected Goal")
        self.remove_goal_button.setMinimumHeight(UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX)
        self.remove_goal_button.clicked.connect(self.remove_selected_goal)
        row.addWidget(self.remove_goal_button)
        self.clear_goals_button = QPushButton("Clear Goals")
        self.clear_goals_button.setMinimumHeight(UI_THIRD_COLUMN_COMPACT_BUTTON_HEIGHT_PX)
        self.clear_goals_button.clicked.connect(self.clear_queued_goals)
        row.addWidget(self.clear_goals_button)
        layout.addLayout(row)

        # Place variation controls directly before loop settings.
        if hasattr(self, "_variation_controls_widget") and isinstance(self._variation_controls_widget, QWidget):
            layout.addWidget(self._variation_controls_widget)

        auto_group = QGroupBox("Loop Settings")
        auto_layout = QVBoxLayout(auto_group)
        auto_layout.setContentsMargins(6, 6, 6, 6)
        auto_layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(QLabel("Number of Iterations"))
        self.auto_loop_count_input = QLineEdit("1")
        self.auto_loop_count_input.setToolTip(
            "Iterations for Start: grasp+reach auto-loop rounds, or drag/curve forward-pass repeats"
        )
        self.auto_loop_count_input.setMaximumWidth(90)
        self.auto_loop_count_input.textChanged.connect(lambda _t: self._update_auto_loop_progress_ui())
        top_row.addWidget(self.auto_loop_count_input)
        top_row.addSpacing(18)
        top_row.addStretch(1)
        top_row.addWidget(QLabel("Save to disk"))
        self.auto_loop_record_button = QCheckBox()
        self.auto_loop_record_button.setToolTip("Checked = save loop recording, unchecked = do not save.")
        self.auto_loop_record_button.toggled.connect(self._on_auto_loop_record_button_clicked)
        top_row.addWidget(self.auto_loop_record_button)
        auto_layout.addLayout(top_row)

        start_stop_row = QHBoxLayout()
        start_stop_row.setSpacing(6)
        self.auto_loop_start_button = QPushButton("Start")
        self.auto_loop_start_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.auto_loop_start_button.setToolTip(
            "Start queued goals. Drag/curve uses iteration repeats with reverse return; grasp+reach uses auto-loop."
        )
        self.auto_loop_start_button.clicked.connect(self._on_auto_loop_start_clicked)
        start_stop_row.addWidget(self.auto_loop_start_button, 2)
        self.auto_loop_stop_button = QPushButton("Stop")
        self.auto_loop_stop_button.setMinimumHeight(UI_THIRD_COLUMN_BUTTON_HEIGHT_PX)
        self.auto_loop_stop_button.clicked.connect(self._on_auto_loop_stop_clicked)
        start_stop_row.addWidget(self.auto_loop_stop_button, 1)
        auto_layout.addLayout(start_stop_row)

        self.auto_loop_progress_bar = QProgressBar()
        self.auto_loop_progress_bar.setMinimum(0)
        self.auto_loop_progress_bar.setMaximum(1)
        self.auto_loop_progress_bar.setValue(0)
        self.auto_loop_progress_bar.setTextVisible(True)
        self.auto_loop_progress_bar.setFormat("Progress: Iteration 0/1")
        self.auto_loop_progress_bar.setEnabled(False)
        auto_layout.addWidget(self.auto_loop_progress_bar)

        # Return/Pause controls should be below progress.
        if hasattr(self, "return_pause_widget") and isinstance(self.return_pause_widget, QWidget):
            auto_layout.addWidget(self.return_pause_widget)

        layout.addWidget(auto_group)

        self._update_goal_queue_label()
        self._update_next_goal_button_state()
        self._on_goal_selection_changed()
        self._update_auto_loop_record_button_ui()
        self._update_auto_loop_progress_ui()
        self._update_auto_loop_start_stop_button_ui()
        return widget

    def _update_auto_loop_record_button_ui(self) -> None:
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            QTimer.singleShot(0, self._update_auto_loop_record_button_ui)
            return
        if not hasattr(self, "auto_loop_record_button"):
            return
        armed = bool(getattr(self, "_loop_record_armed", False))
        prev = self.auto_loop_record_button.blockSignals(True)
        self.auto_loop_record_button.setChecked(armed)
        self.auto_loop_record_button.blockSignals(prev)
        self.auto_loop_record_button.setText("")

    def _get_auto_loop_display_total(self) -> int:
        if not hasattr(self, "auto_loop_count_input"):
            return 0
        try:
            text = str(self.auto_loop_count_input.text()).strip()
            return int(max(0, int(text)))
        except Exception:
            return 0

    def _update_auto_loop_progress_ui(self) -> None:
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            QTimer.singleShot(0, self._update_auto_loop_progress_ui)
            return
        if not hasattr(self, "auto_loop_progress_bar"):
            return
        total = int(max(0, int(getattr(self, "_auto_loop_requested_rounds", 0))))
        display_total = int(max(0, self._get_auto_loop_display_total()))
        left = int(max(0, int(getattr(self, "_auto_rounds_left", 0))))
        running = bool(getattr(self, "_auto_loop_running", False))

        self.auto_loop_progress_bar.setEnabled(bool(running))
        if display_total <= 0:
            self.auto_loop_progress_bar.setMinimum(0)
            self.auto_loop_progress_bar.setMaximum(1)
            self.auto_loop_progress_bar.setValue(0)
            self.auto_loop_progress_bar.setFormat("Progress: Iteration 0/0")
            self._update_auto_loop_start_stop_button_ui()
            return

        self.auto_loop_progress_bar.setMinimum(0)
        self.auto_loop_progress_bar.setMaximum(int(display_total))

        completed = int(max(0, min(total, total - left)))
        if running and left > 0 and total > 0:
            current_round = int(max(1, completed + 1))
            current_round = int(max(0, min(display_total, current_round)))
            self.auto_loop_progress_bar.setValue(current_round)
            self.auto_loop_progress_bar.setFormat(f"Progress: Iteration {current_round}/{display_total}")
        else:
            shown = int(max(0, min(display_total, completed)))
            self.auto_loop_progress_bar.setValue(shown)
            self.auto_loop_progress_bar.setFormat(f"Progress: Iteration {shown}/{display_total}")
        self._update_auto_loop_start_stop_button_ui()

    def _update_auto_loop_start_stop_button_ui(self) -> None:
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            QTimer.singleShot(0, self._update_auto_loop_start_stop_button_ui)
            return
        if hasattr(self, "auto_loop_start_button"):
            active = bool(
                (getattr(self, "_auto_loop_running", False) and (not getattr(self, "_auto_loop_abort", False)))
                or getattr(self, "_auto_start_after_return", False)
            )
            if active:
                self.auto_loop_start_button.setText("Pause")
                self.auto_loop_start_button.setStyleSheet(
                    "QPushButton { background-color: #fff59d; color: #212121; border: 1px solid #f9a825; }"
                )
            else:
                self.auto_loop_start_button.setText("Start")
                self.auto_loop_start_button.setStyleSheet(
                    "QPushButton { background-color: #c8e6c9; color: #1b5e20; border: 1px solid #81c784; }"
                )
        if hasattr(self, "auto_loop_stop_button"):
            self.auto_loop_stop_button.setStyleSheet(
                "QPushButton { background-color: #ffcdd2; color: #b71c1c; border: 1px solid #ef9a9a; }"
            )

    def _on_auto_loop_record_button_clicked(self, checked: bool) -> None:
        self._loop_record_armed = bool(checked)
        self._update_auto_loop_record_button_ui()
        if bool(self._loop_record_armed):
            self._set_status("Save to disk: ON", "QLabel { color: #1e88e5; font-size: 10px; }")
            return

        # If turned OFF during an active recording session, finalize via existing stop path.
        if self._loop_record_session_active:
            self._loop_record_stop_requested = True
            self.stop_auto_pick_place_loop()
        else:
            self._set_status("Save to disk: OFF", "QLabel { color: gray; font-size: 10px; }")

    def _selected_task_name_for_loop(self) -> str:
        if hasattr(self, "task_combo"):
            try:
                return str(self.task_combo.currentText() or "").strip()
            except Exception:
                return ""
        return ""

    def _is_simple_goal_sequence_for_auto_loop(self, goals: list[dict[str, Any]]) -> bool:
        if not isinstance(goals, list) or len(goals) == 0:
            return False
        for g in goals:
            if not isinstance(g, dict):
                return False
            kind = str(g.get("kind", "")).strip()
            if kind not in AUTO_LOOP_SIMPLE_SEQUENCE_GOAL_KINDS:
                return False
        return True

    def start_auto_goal_sequence_loop(self, rounds: int) -> bool:
        r = int(rounds)
        if r <= 0:
            self._set_status("Auto loop count must be >= 1", "QLabel { color: red; font-size: 10px; }")
            return False
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Auto loop can start only while idle", "QLabel { color: orange; font-size: 10px; }")
                return False
        goals = self._goal_sequence_order()
        if len(goals) == 0:
            self._set_status("No queued goals for auto loop", "QLabel { color: orange; font-size: 10px; }")
            return False
        if self._auto_loop_running:
            self._set_status("Auto loop already running", "QLabel { color: orange; font-size: 10px; }")
            return False

        self._auto_loop_mode = "goal_sequence"
        self._auto_loop_requested_rounds = int(r)
        self._auto_rounds_left = int(r)
        self._update_auto_loop_progress_ui()
        self._auto_loop_abort = False
        self._auto_loop_running = False
        self._auto_sequence_replay_active = False
        self._auto_first_trial_pending = True
        self._auto_start_after_return = True
        self._auto_capture_enabled = False
        self._auto_pose_home = self._capture_current_pose_snapshot("auto_seq_home_before_first_trial")
        self._auto_pose_grasp = None
        self._auto_pose_reach = None
        self._auto_pose_place_object = None
        self._auto_pose_release = None
        self._auto_pose_grasp_target = None
        self._auto_pose_reach_target = None
        self._auto_pose_place_object_target = None
        self._auto_initial_base_x = None
        self._auto_gripper_open = None
        self._auto_gripper_closed = None

        if not self._start_auto_loop_record_session(int(r)):
            self._auto_first_trial_pending = False
            self._auto_start_after_return = False
            self._auto_loop_mode = "pick_place"
            return False

        self._set_status(
            f"Auto loop armed for queued sequence: first manual trial starts now (repeats after first trial={r}). "
            "After first trial, press Return once to begin auto replay.",
            "QLabel { color: #1e88e5; font-size: 10px; }",
        )
        self.execute_all_queued_goals()
        return True

    def _maybe_start_auto_loop_after_return(self) -> None:
        mode = str(getattr(self, "_auto_loop_mode", "pick_place"))
        if mode != "goal_sequence":
            super()._maybe_start_auto_loop_after_return()
            return
        if not bool(self._auto_start_after_return):
            print("[auto_loop] return completed: goal-sequence loop not armed; skipping replay.")
            return
        self._auto_first_trial_pending = False
        self._auto_start_after_return = False
        self._auto_loop_running = True
        self._auto_sequence_replay_active = True
        self._update_auto_loop_progress_ui()
        from threading import Thread
        Thread(target=self._run_auto_goal_sequence_loop, daemon=True).start()

    def _run_auto_goal_sequence_loop(self) -> None:
        loop_success = False

        def _wait_until_idle(*, timeout_s: float, require_no_pre_state: bool = False) -> bool:
            end_t = time.time() + max(0.5, float(timeout_s))
            while time.time() < end_t:
                if self._auto_loop_abort:
                    return False
                with self._action_lock:
                    st = self._action_state
                done = bool(
                    st == "idle"
                    and (not bool(self._run_all_queued_goals))
                    and (not bool(self._deferred_next_goal_start))
                )
                if bool(require_no_pre_state):
                    done = bool(done and (self._pre_action_state is None))
                if done:
                    return True
                time.sleep(0.05)
            return False

        def _wait_abortable(duration_s: float) -> bool:
            end_t = time.time() + max(0.0, float(duration_s))
            while time.time() < end_t:
                if self._auto_loop_abort:
                    return False
                time.sleep(min(0.05, max(0.0, end_t - time.time())))
            return True

        def _extract_xyz_from_goal(goal: dict[str, Any] | None) -> tuple[float, float, float] | None:
            if not isinstance(goal, dict):
                return None
            for key in ("point_xyz", "point_odom_xyz"):
                xyz = goal.get(key)
                if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
                    try:
                        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                    except Exception:
                        continue
            return None

        def _extract_drag_endpoint_xyz(goal: dict[str, Any], *, use_end: bool) -> tuple[float, float, float] | None:
            path_xyz = goal.get("path_base_xyz")
            if isinstance(path_xyz, list) and len(path_xyz) >= 2:
                idx = -1 if bool(use_end) else 0
                p = path_xyz[idx]
                if isinstance(p, (list, tuple)) and len(p) >= 3:
                    try:
                        return (float(p[0]), float(p[1]), float(p[2]))
                    except Exception:
                        pass
            px = goal.get("end_px" if bool(use_end) else "px")
            py = goal.get("end_py" if bool(use_end) else "py")
            if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                p, _, _, _ = self._get_3d_point_nearest_valid_depth(
                    int(px),
                    int(py),
                    search_radius_px=int(DRAG_NEAREST_VALID_DEPTH_RADIUS_PX),
                )
                if p is not None:
                    return (float(p.point.x), float(p.point.y), float(p.point.z))
            return None

        def _build_grasp_goal_at(
            xyz: tuple[float, float, float],
            *,
            px: int | None,
            py: int | None,
            template: dict[str, Any] | None,
            post_grasp_lift: bool,
        ) -> dict[str, Any]:
            g: dict[str, Any] = {
                "kind": "grasp",
                "point_xyz": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                "post_grasp_lift": bool(post_grasp_lift),
                "created_time": float(time.time()),
            }
            if isinstance(px, int):
                g["px"] = int(px)
            if isinstance(py, int):
                g["py"] = int(py)
            if isinstance(template, dict):
                for k in ("grasp_yaw", "gripper_width", "long_axis_angle", "object_top_z"):
                    if k in template:
                        g[k] = template.get(k)
            return g

        def _build_reach_goal_at(
            xyz: tuple[float, float, float],
            *,
            px: int | None,
            py: int | None,
            template: dict[str, Any] | None,
        ) -> dict[str, Any]:
            g: dict[str, Any] = {
                "kind": "reach",
                "point_xyz": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                "created_time": float(time.time()),
            }
            if isinstance(px, int):
                g["px"] = int(px)
            if isinstance(py, int):
                g["py"] = int(py)
            if isinstance(template, dict) and ("object_top_z" in template):
                g["object_top_z"] = template.get("object_top_z")
            return g

        def _run_goals_blocking(
            goals_to_run: list[dict[str, Any]],
            *,
            phase_label: str,
            timeout_per_goal_s: float = 240.0,
        ) -> bool:
            if not isinstance(goals_to_run, list) or len(goals_to_run) == 0:
                return True
            # Explicitly run each goal in order (without relying on queued-goal auto chaining).
            self._run_all_queued_goals = False
            self._deferred_next_goal_start = False
            for idx, g0 in enumerate(goals_to_run):
                if self._auto_loop_abort:
                    return False
                if not isinstance(g0, dict):
                    continue
                goal = copy.deepcopy(g0)
                prev_goal = goals_to_run[idx - 1] if idx > 0 else None
                if (
                    isinstance(prev_goal, dict)
                    and str(prev_goal.get("kind", "")) == "grasp"
                    and (not bool(prev_goal.get("post_grasp_lift", True)))
                    and str(goal.get("kind", "")) in ("drag", "drag_curve")
                ):
                    goal["_force_keep_current_lift"] = True
                preserve = bool(idx > 0 and self._pre_action_state is not None)
                self._set_status(
                    f"{phase_label}: {idx + 1}/{len(goals_to_run)} ({str(goal.get('kind', '?'))})",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                started = bool(
                    self._start_prepared_goal(
                        goal,
                        preserve_existing_pre_action_state=bool(preserve),
                    )
                )
                if not started:
                    return False
                if not _wait_until_idle(timeout_s=float(timeout_per_goal_s), require_no_pre_state=False):
                    return False
            return True

        def _pair_open_close_values() -> tuple[float, float]:
            g_open = float(DEVICE_GRIPPER_TOGGLE_OPEN_JOINT)
            g_close = float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT)
            has_learned_open = bool(
                isinstance(self._auto_gripper_open, (int, float))
                and math.isfinite(float(self._auto_gripper_open))
            )
            has_learned_close = bool(
                isinstance(self._auto_gripper_closed, (int, float))
                and math.isfinite(float(self._auto_gripper_closed))
            )

            # Prefer explicit first-trial learned values when available.
            if bool(has_learned_open):
                g_open = float(self._auto_gripper_open)
            elif isinstance(self._auto_pose_release, dict):
                try:
                    g_open = float(self._auto_pose_release.get("gripper", g_open))
                except Exception:
                    pass
            elif isinstance(self._auto_pose_reach, dict):
                try:
                    g_open = float(self._auto_pose_reach.get("gripper", g_open))
                except Exception:
                    pass
            elif isinstance(self._auto_pose_place_object, dict):
                try:
                    g_open = float(self._auto_pose_place_object.get("gripper", g_open))
                except Exception:
                    pass

            if bool(has_learned_close):
                g_close = float(self._auto_gripper_closed)
            elif isinstance(self._auto_pose_grasp, dict):
                try:
                    g_close = float(self._auto_pose_grasp.get("gripper", g_close))
                except Exception:
                    pass
            g_open = float(
                np.clip(
                    float(g_open),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            g_close = float(
                np.clip(
                    float(g_close),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )
            # When we have a learned close from precise/manual trial, trust it as-is.
            # Otherwise keep legacy guard (close cannot be numerically more open than open).
            if not bool(has_learned_close):
                g_close = float(min(float(g_close), float(g_open)))
            print(
                "[auto_loop gripper pair] "
                f"open={float(g_open):+.4f} close={float(g_close):+.4f} "
                f"(learned_open={has_learned_open} learned_close={has_learned_close})"
            )
            return g_open, g_close

        def _grasp_target_snapshot() -> dict[str, Any] | None:
            if isinstance(self._auto_pose_grasp_target, dict):
                return self._auto_pose_grasp_target
            if isinstance(self._auto_pose_grasp, dict):
                return self._auto_pose_grasp
            return None

        def _snapshot_joint6(snap: dict[str, Any] | None) -> list[float] | None:
            if not isinstance(snap, dict):
                return None
            j = snap.get("joint6")
            if not (isinstance(j, list) and len(j) >= 6):
                return None
            try:
                return [float(v) for v in j[:6]]
            except Exception:
                return None

        def _snapshot_head(snap: dict[str, Any] | None) -> list[float] | None:
            if not isinstance(snap, dict):
                return None
            h = snap.get("head")
            if isinstance(h, list) and len(h) >= 2:
                try:
                    return [float(h[0]), float(h[1])]
                except Exception:
                    return None
            return None

        def _move_to_snapshot(
            snap: dict[str, Any] | None,
            *,
            gripper: float | None,
            note: str,
            timeout_s: float = 12.0,
            reliable: bool = False,
        ) -> bool:
            j6 = _snapshot_joint6(snap)
            if not (isinstance(j6, list) and len(j6) >= 6):
                return False
            head_cmd = _snapshot_head(snap)
            self._set_status(note, "QLabel { color: #1e88e5; font-size: 10px; }")
            return bool(
                self._execute_arm_to_chunked(
                    j6[:6],
                    gripper=gripper,
                    head=head_cmd,
                    timeout_s=float(timeout_s),
                    reliable=bool(reliable),
                )
            )

        def _move_to_snapshot_with_safe_lift(
            snap: dict[str, Any] | None,
            *,
            gripper: float | None,
            note: str,
            timeout_s: float = 12.0,
            reliable: bool = False,
        ) -> bool:
            j6 = _snapshot_joint6(snap)
            if not (isinstance(j6, list) and len(j6) >= 6):
                return False
            head_cmd = _snapshot_head(snap)
            cur = self._current_manip_joint6()
            if not (isinstance(cur, list) and len(cur) >= 6):
                cur = list(j6)
            cur = [float(v) for v in cur[:6]]
            tgt = [float(v) for v in j6[:6]]
            safe_lift = float(
                np.clip(
                    max(float(IK_SAFE_LIFT_M), float(cur[1]), float(tgt[1])),
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )
            step_up = [float(v) for v in cur[:6]]
            step_up[1] = float(safe_lift)
            step_xy = [float(v) for v in tgt[:6]]
            step_xy[1] = float(safe_lift)
            self._set_status(f"{note} (safe-lift up)", "QLabel { color: #1e88e5; font-size: 10px; }")
            if not self._execute_arm_to_chunked(
                step_up[:6],
                gripper=gripper,
                head=head_cmd,
                timeout_s=max(2.0, float(timeout_s) * 0.35),
                reliable=bool(reliable),
            ):
                return False
            self._set_status(f"{note} (safe-lift transit)", "QLabel { color: #1e88e5; font-size: 10px; }")
            if not self._execute_arm_to_chunked(
                step_xy[:6],
                gripper=gripper,
                head=head_cmd,
                timeout_s=max(2.0, float(timeout_s) * 0.35),
                reliable=bool(reliable),
            ):
                return False
            self._set_status(f"{note} (descend)", "QLabel { color: #1e88e5; font-size: 10px; }")
            return bool(
                self._execute_arm_to_chunked(
                    tgt[:6],
                    gripper=gripper,
                    head=head_cmd,
                    timeout_s=max(2.0, float(timeout_s) * 0.35),
                    reliable=bool(reliable),
                )
            )

        def _run_joint_pair_phase(*, forward: bool, phase_label: str) -> bool:
            if not (isinstance(self._auto_pose_grasp, dict) and isinstance(self._auto_pose_release, dict)):
                return False
            g_open, g_close = _pair_open_close_values()

            if bool(forward):
                seq = [
                    ("to grasp", self._auto_pose_grasp, g_open),
                    ("grasp close", self._auto_pose_grasp, g_close),
                    ("to release", self._auto_pose_release, g_close),
                    ("release open", self._auto_pose_release, g_open),
                ]
                for step_idx, (step_name, snap, grip) in enumerate(seq, start=1):
                    if self._auto_loop_abort:
                        return False
                    do_safe_approach = bool(str(step_name) == "to grasp")
                    if bool(do_safe_approach):
                        if not _move_to_snapshot_with_safe_lift(
                            snap,
                            gripper=float(grip),
                            note=f"{phase_label}: {step_idx}/{len(seq)} {step_name}",
                            timeout_s=12.0,
                            reliable=False,
                        ):
                            return False
                    else:
                        if not _move_to_snapshot(
                            snap,
                            gripper=float(grip),
                            note=f"{phase_label}: {step_idx}/{len(seq)} {step_name}",
                            timeout_s=12.0,
                            reliable=False,
                        ):
                            return False
                self._set_manual_gripper_override(float(g_open))
                self.ros_node.sync_command_targets_to_actual()
                return True

            # Reverse reset for grasp(no-lift) -> drag -> release:
            # enforce blocking carry-back order: lift -> retract -> base_x.
            j_rel = _snapshot_joint6(self._auto_pose_release)
            j_gr = _snapshot_joint6(self._auto_pose_grasp)
            h_rel = _snapshot_head(self._auto_pose_release)
            h_gr = _snapshot_head(self._auto_pose_grasp)
            if not (isinstance(j_rel, list) and len(j_rel) >= 6 and isinstance(j_gr, list) and len(j_gr) >= 6):
                return False

            # 1) Safe approach release point and grasp.
            if not _move_to_snapshot_with_safe_lift(
                self._auto_pose_release,
                gripper=float(g_open),
                note=f"{phase_label}: 1/6 to release",
                timeout_s=12.0,
                reliable=True,
            ):
                return False
            if not _move_to_snapshot(
                self._auto_pose_release,
                gripper=float(g_close),
                note=f"{phase_label}: 2/6 grasp at release",
                timeout_s=8.0,
                reliable=True,
            ):
                return False
            # Hold close at release before carry.
            if not self._execute_arm_to_chunked(
                [float(v) for v in j_rel[:6]],
                gripper=float(g_close),
                head=h_rel,
                timeout_s=6.0,
                reliable=True,
            ):
                return False
            time.sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

            # 2) Ordered carry-back: lift -> retract -> base_x (all blocking).
            carry_lift = float(
                np.clip(
                    max(float(IK_SAFE_LIFT_M), float(j_rel[1]), float(j_gr[1])),
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )
            step_lift = [float(v) for v in j_rel[:6]]
            step_lift[1] = float(carry_lift)
            self._set_status(f"{phase_label}: 3/6 lift at release", "QLabel { color: #1e88e5; font-size: 10px; }")
            if not self._execute_arm_to_chunked(
                step_lift[:6],
                gripper=float(g_close),
                head=h_rel,
                timeout_s=10.0,
                reliable=True,
            ):
                return False
            time.sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

            step_retract = [float(v) for v in step_lift[:6]]
            step_retract[2] = float(
                np.clip(
                    0.0,
                    float(self.ros_node.JOINT_LIMITS[0][0]),
                    float(self.ros_node.JOINT_LIMITS[0][1]),
                )
            )
            self._set_status(f"{phase_label}: 4/6 retract arm", "QLabel { color: #1e88e5; font-size: 10px; }")
            if not self._execute_arm_to_chunked(
                step_retract[:6],
                gripper=float(g_close),
                head=h_rel,
                timeout_s=10.0,
                reliable=True,
            ):
                return False
            time.sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

            step_base = [float(v) for v in step_retract[:6]]
            step_base[0] = float(
                np.clip(
                    float(j_gr[0]),
                    float(MANIP_BASE_X_LIMITS[0]),
                    float(MANIP_BASE_X_LIMITS[1]),
                )
            )
            self._set_status(f"{phase_label}: 5/6 move base_x to initial", "QLabel { color: #1e88e5; font-size: 10px; }")
            if not self._execute_arm_to_chunked(
                step_base[:6],
                gripper=float(g_close),
                head=h_gr,
                timeout_s=10.0,
                reliable=True,
            ):
                return False
            time.sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

            # 3) Move to initial grasp pose and release there.
            if not _move_to_snapshot(
                self._auto_pose_grasp,
                gripper=float(g_close),
                note=f"{phase_label}: 6/6 to initial grasp",
                timeout_s=10.0,
                reliable=True,
            ):
                return False
            if not _move_to_snapshot(
                self._auto_pose_grasp,
                gripper=float(g_open),
                note=f"{phase_label}: release at initial grasp",
                timeout_s=6.0,
                reliable=True,
            ):
                return False

            self._set_manual_gripper_override(float(g_open))
            self.ros_node.sync_command_targets_to_actual()
            return True

        def _return_home_after_pair(*, note: str) -> bool:
            g_open, _g_close = _pair_open_close_values()
            self._set_status(note, "QLabel { color: #1e88e5; font-size: 10px; }")
            ok = bool(
                _return_home_lift_retract_base_blocking(
                    phase_label="Type2 return home",
                    gripper_value=float(g_open),
                )
            )
            if not ok:
                return False
            return True

        def _run_curve_pick_place_reset(*, phase_label: str) -> bool:
            if not (isinstance(self._auto_pose_grasp, dict) and isinstance(self._auto_pose_release, dict)):
                return False
            g_open, g_close = _pair_open_close_values()
            # If no learned close exists, fall back to a firm device close.
            if self._auto_gripper_closed is None:
                device_close = float(
                    np.clip(
                        float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                g_close = float(min(float(g_close), float(device_close)))
            j_rel = _snapshot_joint6(self._auto_pose_release)
            j_gr = _snapshot_joint6(self._auto_pose_grasp)
            if not (isinstance(j_rel, list) and len(j_rel) >= 6 and isinstance(j_gr, list) and len(j_gr) >= 6):
                return False
            head_rel = _snapshot_head(self._auto_pose_release)
            head_gr = _snapshot_head(self._auto_pose_grasp)
            home_joint6 = self._default_init_joint6()
            home_base_x = float(home_joint6[0]) if isinstance(home_joint6, list) and len(home_joint6) >= 1 else 0.0
            home_base_x = float(
                np.clip(
                    float(home_base_x),
                    float(MANIP_BASE_X_LIMITS[0]),
                    float(MANIP_BASE_X_LIMITS[1]),
                )
            )

            # 1) Safe approach and grasp object at release point.
            if not _move_to_snapshot_with_safe_lift(
                self._auto_pose_release,
                gripper=float(g_open),
                note=f"{phase_label}: pick at release",
                timeout_s=12.0,
            ):
                return False
            if not _move_to_snapshot(
                self._auto_pose_release,
                gripper=float(g_close),
                note=f"{phase_label}: close at release",
                timeout_s=8.0,
            ):
                return False
            # Hold close once more at the same release pose to guarantee latch
            # before lift/retract starts.
            if not self._execute_arm_to_chunked(
                [float(v) for v in j_rel[:6]],
                gripper=float(g_close),
                head=head_rel,
                timeout_s=5.0,
                reliable=True,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))

            # 2) Lift and retract arm to zero near release point.
            carry_lift = float(
                np.clip(
                    max(float(IK_SAFE_LIFT_M), float(j_rel[1]), float(j_gr[1])),
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )
            self._set_status(f"{phase_label}: lift+retract at release", "QLabel { color: #1e88e5; font-size: 10px; }")
            step_lift = [float(v) for v in j_rel[:6]]
            step_lift[1] = float(carry_lift)
            if not self._execute_arm_to_chunked(
                step_lift[:6],
                gripper=float(g_close),
                head=head_rel,
                timeout_s=10.0,
                reliable=False,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))
            step_retract = [float(v) for v in step_lift[:6]]
            step_retract[2] = float(
                np.clip(
                    0.0,
                    float(self.ros_node.JOINT_LIMITS[0][0]),
                    float(self.ros_node.JOINT_LIMITS[0][1]),
                )
            )
            if not self._execute_arm_to_chunked(
                step_retract[:6],
                gripper=float(g_close),
                head=head_rel,
                timeout_s=10.0,
                reliable=False,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))

            # 3) Move to home base_x while retracted/safe.
            self._set_status(f"{phase_label}: move to home base_x", "QLabel { color: #1e88e5; font-size: 10px; }")
            step_home = [float(v) for v in step_retract[:6]]
            step_home[0] = float(home_base_x)
            if not self._execute_arm_to_chunked(
                step_home[:6],
                gripper=float(g_close),
                head=head_gr,
                timeout_s=10.0,
                reliable=False,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))

            # 4) Move base_x to initial grasp base_x while still retracted/safe.
            self._set_status(f"{phase_label}: move to target base_x", "QLabel { color: #1e88e5; font-size: 10px; }")
            step_target_base = [float(v) for v in step_home[:6]]
            step_target_base[0] = float(
                np.clip(
                    float(j_gr[0]),
                    float(MANIP_BASE_X_LIMITS[0]),
                    float(MANIP_BASE_X_LIMITS[1]),
                )
            )
            if not self._execute_arm_to_chunked(
                step_target_base[:6],
                gripper=float(g_close),
                head=head_gr,
                timeout_s=10.0,
                reliable=False,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))

            # 5) Extend to initial grasp extension at safe lift.
            self._set_status(f"{phase_label}: extend to target", "QLabel { color: #1e88e5; font-size: 10px; }")
            step_extend = [float(v) for v in step_target_base[:6]]
            step_extend[2] = float(
                np.clip(
                    float(j_gr[2]),
                    float(self.ros_node.JOINT_LIMITS[0][0]),
                    float(self.ros_node.JOINT_LIMITS[0][1]),
                )
            )
            step_extend[3] = float(j_gr[3])
            step_extend[4] = float(j_gr[4])
            step_extend[5] = float(j_gr[5])
            if not self._execute_arm_to_chunked(
                step_extend[:6],
                gripper=float(g_close),
                head=head_gr,
                timeout_s=10.0,
                reliable=False,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))

            # 6) Lower to initial grasp lift (keep closed first).
            self._set_status(f"{phase_label}: lower to place lift", "QLabel { color: #1e88e5; font-size: 10px; }")
            step_place = [float(v) for v in step_extend[:6]]
            step_place[1] = float(
                np.clip(
                    float(j_gr[1]),
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )
            if not self._execute_arm_to_chunked(
                step_place[:6],
                gripper=float(g_close),
                head=head_gr,
                timeout_s=10.0,
                reliable=True,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))
            snap_place = self._capture_current_pose_snapshot("curve_reset_place_before_open")
            if isinstance(snap_place, dict):
                self._auto_pose_grasp = dict(snap_place)

            # 7) Open gripper at place point (not in-air).
            self._set_status(f"{phase_label}: open at place", "QLabel { color: #1e88e5; font-size: 10px; }")
            if not self._execute_arm_to_chunked(
                step_place[:6],
                gripper=float(g_open),
                head=head_gr,
                timeout_s=6.0,
                reliable=True,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))

            # 8) Lift back to safe height, retract, and go home base_x.
            self._set_status(f"{phase_label}: lift+retract+home", "QLabel { color: #1e88e5; font-size: 10px; }")
            step_lift_after = [float(v) for v in step_place[:6]]
            step_lift_after[1] = float(carry_lift)
            if not self._execute_arm_to_chunked(
                step_lift_after[:6],
                gripper=float(g_open),
                head=head_gr,
                timeout_s=10.0,
                reliable=False,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))
            step_retract_after = [float(v) for v in step_lift_after[:6]]
            step_retract_after[2] = float(
                np.clip(
                    0.0,
                    float(self.ros_node.JOINT_LIMITS[0][0]),
                    float(self.ros_node.JOINT_LIMITS[0][1]),
                )
            )
            if not self._execute_arm_to_chunked(
                step_retract_after[:6],
                gripper=float(g_open),
                head=head_gr,
                timeout_s=10.0,
                reliable=False,
            ):
                return False
            time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))
            step_home_after = [float(v) for v in step_retract_after[:6]]
            step_home_after[0] = float(home_base_x)
            if not self._execute_arm_to_chunked(
                step_home_after[:6],
                gripper=float(g_open),
                head=head_gr,
                timeout_s=10.0,
                reliable=False,
            ):
                return False

            self._set_manual_gripper_override(float(g_open))
            self.ros_node.sync_command_targets_to_actual()
            return True

        def _run_single_grasp_type1(
            *,
            with_lift: bool,
            phase_label: str,
            rotate_deg: float | None = None,
            force_pitch_rad: float | None = None,
        ) -> bool:
            if not isinstance(self._auto_pose_grasp, dict):
                return False
            g_open, g_close = _pair_open_close_values()
            if self._auto_gripper_closed is None:
                device_close = float(
                    np.clip(
                        float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                g_close = float(min(float(g_close), float(device_close)))
            rotate_deg_safe = (
                float(rotate_deg)
                if isinstance(rotate_deg, (int, float)) and math.isfinite(float(rotate_deg))
                else 0.0
            )
            do_rotate = bool((not bool(with_lift)) and abs(float(rotate_deg_safe)) > 1e-6)
            force_pitch = (
                float(force_pitch_rad)
                if isinstance(force_pitch_rad, (int, float)) and math.isfinite(float(force_pitch_rad))
                else None
            )
            if isinstance(force_pitch, float):
                force_pitch = float(
                    np.clip(
                        float(force_pitch),
                        float(self.ros_node.JOINT_LIMITS[3][0]),
                        float(self.ros_node.JOINT_LIMITS[3][1]),
                    )
                )
            if bool(do_rotate) and isinstance(force_pitch, float):
                pre_align_joint = self._current_manip_joint6()
                if not (isinstance(pre_align_joint, list) and len(pre_align_joint) >= 6):
                    pre_align_joint = _snapshot_joint6(self._auto_pose_grasp)
                if not (isinstance(pre_align_joint, list) and len(pre_align_joint) >= 6):
                    pre_align_joint = _snapshot_joint6(_grasp_target_snapshot())
                if isinstance(pre_align_joint, list) and len(pre_align_joint) >= 6:
                    pre_align_joint = [float(v) for v in pre_align_joint[:6]]
                    if abs(float(pre_align_joint[4]) - float(force_pitch)) > float(np.deg2rad(0.5)):
                        pre_align_joint[4] = float(force_pitch)
                        self._set_status(
                            f"{phase_label}: pre-align wrist pitch {float(math.degrees(force_pitch)):+.1f}deg",
                            "QLabel { color: #1e88e5; font-size: 10px; }",
                        )
                        if not self._execute_arm_to_chunked(
                            pre_align_joint[:6],
                            gripper=None,
                            head=None,
                            timeout_s=8.0,
                            reliable=True,
                        ):
                            return False

            if bool(with_lift):
                target_snap = _grasp_target_snapshot()
                if not isinstance(target_snap, dict):
                    return False
                if not _move_to_snapshot_with_safe_lift(
                    target_snap,
                    gripper=float(g_open),
                    note=f"{phase_label}: approach grasp target",
                    timeout_s=12.0,
                    reliable=False,
                ):
                    return False
                if not _move_to_snapshot(
                    target_snap,
                    gripper=float(g_close),
                    note=f"{phase_label}: grasp close at target",
                    timeout_s=8.0,
                    reliable=True,
                ):
                    return False
                if not _move_to_snapshot(
                    self._auto_pose_grasp,
                    gripper=float(g_close),
                    note=f"{phase_label}: lift after grasp",
                    timeout_s=10.0,
                    reliable=True,
                ):
                    return False
            else:
                if not _move_to_snapshot_with_safe_lift(
                    self._auto_pose_grasp,
                    gripper=float(g_open),
                    note=f"{phase_label}: approach grasp",
                    timeout_s=12.0,
                    reliable=False,
                ):
                    return False
                if not _move_to_snapshot(
                    self._auto_pose_grasp,
                    gripper=float(g_close),
                    note=f"{phase_label}: grasp close",
                    timeout_s=8.0,
                    reliable=True,
                ):
                    return False
                if bool(do_rotate):
                    rot_joint = self._current_manip_joint6()
                    if not (isinstance(rot_joint, list) and len(rot_joint) >= 6):
                        rot_joint = _snapshot_joint6(self._auto_pose_grasp)
                    if not (isinstance(rot_joint, list) and len(rot_joint) >= 6):
                        return False
                    rot_joint = [float(v) for v in rot_joint[:6]]
                    if isinstance(force_pitch, float):
                        rot_joint[4] = float(force_pitch)
                    rot_joint[3] = float(
                        np.clip(
                            float(rot_joint[3]) + float(np.deg2rad(float(rotate_deg_safe))),
                            float(self.ros_node.JOINT_LIMITS[2][0]),
                            float(self.ros_node.JOINT_LIMITS[2][1]),
                        )
                    )
                    self._set_status(
                        f"{phase_label}: rotate yaw {float(rotate_deg_safe):+.1f}deg",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not self._execute_arm_to_chunked(
                        rot_joint[:6],
                        gripper=float(g_close),
                        head=None,
                        timeout_s=8.0,
                        reliable=True,
                    ):
                        return False

            self._set_manual_gripper_override(float(g_close))
            self.ros_node.sync_command_targets_to_actual()
            return _wait_abortable(float(AUTO_LOOP_GRASP_HOLD_S))

        def _run_single_grasp_type2(
            *,
            with_lift: bool,
            phase_label: str,
            rotate_deg: float | None = None,
            force_pitch_rad: float | None = None,
        ) -> bool:
            if not isinstance(self._auto_pose_grasp, dict):
                return False
            g_open, g_close = _pair_open_close_values()
            rotate_deg_safe = (
                float(rotate_deg)
                if isinstance(rotate_deg, (int, float)) and math.isfinite(float(rotate_deg))
                else 0.0
            )
            do_rotate = bool((not bool(with_lift)) and abs(float(rotate_deg_safe)) > 1e-6)
            force_pitch = (
                float(force_pitch_rad)
                if isinstance(force_pitch_rad, (int, float)) and math.isfinite(float(force_pitch_rad))
                else None
            )
            if isinstance(force_pitch, float):
                force_pitch = float(
                    np.clip(
                        float(force_pitch),
                        float(self.ros_node.JOINT_LIMITS[3][0]),
                        float(self.ros_node.JOINT_LIMITS[3][1]),
                    )
                )

            def _ordered_single_grasp_return(*, note_prefix: str, restore_home_pitch: bool = False) -> bool:
                cur = self._current_manip_joint6()
                if not (isinstance(cur, list) and len(cur) >= 6):
                    cur = _snapshot_joint6(self._auto_pose_grasp)
                if not (isinstance(cur, list) and len(cur) >= 6):
                    cur = self._default_init_joint6()
                cur = [float(v) for v in cur[:6]]
                head_cmd = _snapshot_head(self._auto_pose_grasp)

                safe_lift = float(
                    np.clip(
                        max(float(IK_SAFE_LIFT_M), float(cur[1])),
                        float(self.ros_node.JOINT_LIMITS[1][0]),
                        float(self.ros_node.JOINT_LIMITS[1][1]),
                    )
                )
                step_lift = [float(v) for v in cur[:6]]
                step_lift[1] = float(safe_lift)
                self._set_status(
                    f"{note_prefix}: lift to safe height",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not self._execute_arm_to_chunked(
                    step_lift[:6],
                    gripper=float(g_open),
                    head=head_cmd,
                    timeout_s=10.0,
                    reliable=True,
                ):
                    return False
                time.sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

                step_retract = [float(v) for v in step_lift[:6]]
                step_retract[2] = float(
                    np.clip(
                        0.0,
                        float(self.ros_node.JOINT_LIMITS[0][0]),
                        float(self.ros_node.JOINT_LIMITS[0][1]),
                    )
                )
                self._set_status(
                    f"{note_prefix}: retract arm to 0",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not self._execute_arm_to_chunked(
                    step_retract[:6],
                    gripper=float(g_open),
                    head=head_cmd,
                    timeout_s=10.0,
                    reliable=True,
                ):
                    return False
                time.sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

                home_joint6 = self._default_init_joint6()
                home_base_x = float(home_joint6[0]) if isinstance(home_joint6, list) and len(home_joint6) >= 1 else 0.0
                home_base_x = float(
                    np.clip(
                        float(home_base_x),
                        float(MANIP_BASE_X_LIMITS[0]),
                        float(MANIP_BASE_X_LIMITS[1]),
                    )
                )
                step_home = [float(v) for v in step_retract[:6]]
                step_home[0] = float(home_base_x)
                if bool(restore_home_pitch):
                    try:
                        init_joint6 = self._default_init_joint6()
                        if isinstance(init_joint6, list) and len(init_joint6) >= 5:
                            step_home[4] = float(
                                np.clip(
                                    float(init_joint6[4]),
                                    float(self.ros_node.JOINT_LIMITS[3][0]),
                                    float(self.ros_node.JOINT_LIMITS[3][1]),
                                )
                            )
                    except Exception:
                        pass
                self._set_status(
                    f"{note_prefix}: base_x to home",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if not self._execute_arm_to_chunked(
                    step_home[:6],
                    gripper=float(g_open),
                    head=head_cmd,
                    timeout_s=10.0,
                    reliable=True,
                ):
                    return False

                self._pre_action_state = None
                self.queued_sequence_started = False
                self._set_manual_gripper_override(float(g_open))
                self.ros_node.sync_command_targets_to_actual()
                return True

            if bool(with_lift):
                target_snap = _grasp_target_snapshot()
                if not isinstance(target_snap, dict):
                    return False
                if not _move_to_snapshot(
                    target_snap,
                    gripper=float(g_close),
                    note=f"{phase_label}: lower to object height",
                    timeout_s=10.0,
                    reliable=True,
                ):
                    return False
                time.sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))
                if not _move_to_snapshot(
                    target_snap,
                    gripper=float(g_open),
                    note=f"{phase_label}: open at object height",
                    timeout_s=6.0,
                    reliable=True,
                ):
                    return False
            else:
                if bool(do_rotate):
                    # Rotate back to original grasp yaw at object point, then open.
                    if not _move_to_snapshot(
                        self._auto_pose_grasp,
                        gripper=float(g_close),
                        note=f"{phase_label}: rotate back to grasp yaw",
                        timeout_s=8.0,
                        reliable=True,
                    ):
                        return False
                    if not _move_to_snapshot(
                        self._auto_pose_grasp,
                        gripper=float(g_open),
                        note=f"{phase_label}: open at grasp point",
                        timeout_s=6.0,
                        reliable=True,
                    ):
                        return False
                    return _ordered_single_grasp_return(
                        note_prefix=f"{phase_label}: return",
                        restore_home_pitch=True,
                    )
                if not _move_to_snapshot(
                    self._auto_pose_grasp,
                    gripper=float(g_open),
                    note=f"{phase_label}: open at grasp point",
                    timeout_s=6.0,
                    reliable=True,
                ):
                    return False
            return _ordered_single_grasp_return(note_prefix=f"{phase_label}: return")

        def _return_home_lift_retract_base_blocking(*, phase_label: str, gripper_value: float) -> bool:
            """Single-goal return order: lift -> retract -> base_x (each step blocking)."""
            cur = self._current_manip_joint6()
            if not (isinstance(cur, list) and len(cur) >= 6):
                cur = self._default_init_joint6()
            cur = [float(v) for v in cur[:6]]

            home_joint6 = self._default_init_joint6()
            home_base_x = float(home_joint6[0]) if isinstance(home_joint6, list) and len(home_joint6) >= 1 else 0.0
            home_base_x = float(
                np.clip(
                    float(home_base_x),
                    float(MANIP_BASE_X_LIMITS[0]),
                    float(MANIP_BASE_X_LIMITS[1]),
                )
            )
            g_open = float(
                np.clip(
                    float(gripper_value),
                    float(self.ros_node.JOINT_LIMITS[7][0]),
                    float(self.ros_node.JOINT_LIMITS[7][1]),
                )
            )

            # Ensure gripper reaches commanded value before arm/base return motion.
            self._set_status(
                f"{phase_label}: applying gripper command",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            if not self._execute_arm_to_chunked(
                cur[:6],
                gripper=float(g_open),
                head=None,
                timeout_s=6.0,
                reliable=True,
            ):
                return False
            time.sleep(float(max(0.0, float(LINEAR_RETURN_INTER_STEP_SETTLE_S))))

            safe_lift = float(
                np.clip(
                    max(float(IK_SAFE_LIFT_M), float(cur[1])),
                    float(self.ros_node.JOINT_LIMITS[1][0]),
                    float(self.ros_node.JOINT_LIMITS[1][1]),
                )
            )
            step_lift = [float(v) for v in cur[:6]]
            step_lift[1] = float(safe_lift)
            self._set_status(
                f"{phase_label}: lift to safe height",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            if not self._execute_arm_to_chunked(
                step_lift[:6],
                gripper=float(g_open),
                head=None,
                timeout_s=10.0,
                reliable=True,
            ):
                return False

            step_retract = [float(v) for v in step_lift[:6]]
            step_retract[2] = float(
                np.clip(
                    0.0,
                    float(self.ros_node.JOINT_LIMITS[0][0]),
                    float(self.ros_node.JOINT_LIMITS[0][1]),
                )
            )
            self._set_status(
                f"{phase_label}: retract arm to 0",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            if not self._execute_arm_to_chunked(
                step_retract[:6],
                gripper=float(g_open),
                head=None,
                timeout_s=10.0,
                reliable=True,
            ):
                return False

            step_home = [float(v) for v in step_retract[:6]]
            step_home[0] = float(home_base_x)
            self._set_status(
                f"{phase_label}: base_x to home",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )
            if not self._execute_arm_to_chunked(
                step_home[:6],
                gripper=float(g_open),
                head=None,
                timeout_s=10.0,
                reliable=True,
            ):
                return False

            self._pre_action_state = None
            self.queued_sequence_started = False
            self._set_manual_gripper_override(float(g_open))
            self.ros_node.sync_command_targets_to_actual()
            return True

        def _run_single_reach_type1(
            goals_to_run: list[dict[str, Any]],
            *,
            phase_label: str,
        ) -> bool:
            g_open, _g_close = _pair_open_close_values()
            target_snap = (
                self._auto_pose_reach_target
                if isinstance(self._auto_pose_reach_target, dict)
                else self._auto_pose_reach
            )
            if isinstance(target_snap, dict):
                if not _move_to_snapshot_with_safe_lift(
                    target_snap,
                    gripper=float(g_open),
                    note=f"{phase_label}: replay manual-corrected reach",
                    timeout_s=12.0,
                    reliable=True,
                ):
                    return False
                snap_now = self._capture_current_pose_snapshot("single_reach_replay_target")
                if isinstance(snap_now, dict):
                    self._auto_pose_reach = dict(snap_now)
                    self._auto_pose_reach_target = dict(snap_now)
                    if self._auto_gripper_open is None:
                        self._auto_gripper_open = float(snap_now["gripper"])
            else:
                if not _run_goals_blocking(
                    goals_to_run,
                    phase_label=phase_label,
                    timeout_per_goal_s=240.0,
                ):
                    return False
            # Reach episode terminates at target hold.
            return _wait_abortable(float(AUTO_LOOP_GRASP_HOLD_S))

        def _run_single_reach_type2(*, phase_label: str) -> bool:
            g_open, _g_close = _pair_open_close_values()
            self._set_manual_gripper_override(float(g_open))
            self.ros_node.sync_command_targets_to_actual()
            return _return_home_lift_retract_base_blocking(
                phase_label=f"{phase_label}: return home",
                gripper_value=float(g_open),
            )

        def _run_single_place_object_type1(
            goals_to_run: list[dict[str, Any]],
            *,
            phase_label: str,
        ) -> bool:
            g_open, g_close = _pair_open_close_values()
            target_snap = (
                self._auto_pose_place_object_target
                if isinstance(self._auto_pose_place_object_target, dict)
                else self._auto_pose_place_object
            )
            if not isinstance(target_snap, dict):
                target_snap = (
                    self._auto_pose_reach_target
                    if isinstance(self._auto_pose_reach_target, dict)
                    else self._auto_pose_reach
                )
            if isinstance(target_snap, dict):
                if not _move_to_snapshot_with_safe_lift(
                    target_snap,
                    gripper=float(g_close),
                    note=f"{phase_label}: replay manual-corrected place_object",
                    timeout_s=12.0,
                    reliable=True,
                ):
                    return False
                snap_now = self._capture_current_pose_snapshot("single_place_object_replay_target")
                if isinstance(snap_now, dict):
                    self._auto_pose_place_object = dict(snap_now)
                    self._auto_pose_place_object_target = dict(snap_now)
                    # Compatibility with existing slots.
                    self._auto_pose_reach = dict(snap_now)
                    self._auto_pose_reach_target = dict(snap_now)
                    if self._auto_gripper_open is None:
                        self._auto_gripper_open = float(snap_now["gripper"])
            else:
                if not _run_goals_blocking(
                    goals_to_run,
                    phase_label=phase_label,
                    timeout_per_goal_s=240.0,
                ):
                    return False

            # Place-object behavior:
            # open at target -> hold 3s -> close again before return.
            if isinstance(target_snap, dict):
                if not _move_to_snapshot(
                    target_snap,
                    gripper=float(g_open),
                    note=f"{phase_label}: open at target",
                    timeout_s=6.0,
                    reliable=True,
                ):
                    return False
            else:
                cur_j6 = self._current_manip_joint6()
                if not (isinstance(cur_j6, list) and len(cur_j6) >= 6):
                    return False
                if not self._execute_arm_to_chunked(
                    [float(v) for v in cur_j6[:6]],
                    gripper=float(g_open),
                    head=None,
                    timeout_s=6.0,
                    reliable=True,
                ):
                    return False
            self._set_manual_gripper_override(float(g_open))
            self.ros_node.sync_command_targets_to_actual()
            if not _wait_abortable(float(AUTO_LOOP_GRASP_HOLD_S)):
                return False

            if isinstance(target_snap, dict):
                if not _move_to_snapshot(
                    target_snap,
                    gripper=float(g_close),
                    note=f"{phase_label}: close before return",
                    timeout_s=6.0,
                    reliable=True,
                ):
                    return False
            else:
                cur_j6 = self._current_manip_joint6()
                if not (isinstance(cur_j6, list) and len(cur_j6) >= 6):
                    return False
                if not self._execute_arm_to_chunked(
                    [float(v) for v in cur_j6[:6]],
                    gripper=float(g_close),
                    head=None,
                    timeout_s=6.0,
                    reliable=True,
                ):
                    return False
            self._set_manual_gripper_override(float(g_close))
            self.ros_node.sync_command_targets_to_actual()
            return True

        def _run_single_place_object_type2(*, phase_label: str) -> bool:
            _g_open, g_close = _pair_open_close_values()
            self._set_manual_gripper_override(float(g_close))
            self.ros_node.sync_command_targets_to_actual()
            return _return_home_lift_retract_base_blocking(
                phase_label=f"{phase_label}: return home",
                gripper_value=float(g_close),
            )

        def _run_curve_forward_from_snapshot(
            goals_to_run: list[dict[str, Any]],
            *,
            phase_label: str,
            timeout_per_goal_s: float = 240.0,
        ) -> bool:
            if not isinstance(goals_to_run, list) or len(goals_to_run) == 0:
                return True
            if not isinstance(self._auto_pose_grasp, dict):
                return _run_goals_blocking(
                    goals_to_run,
                    phase_label=phase_label,
                    timeout_per_goal_s=float(timeout_per_goal_s),
                )
            first_grasp_idx = next(
                (i for i, g in enumerate(goals_to_run) if isinstance(g, dict) and str(g.get("kind", "")) == "grasp"),
                -1,
            )
            if int(first_grasp_idx) < 0:
                return _run_goals_blocking(
                    goals_to_run,
                    phase_label=phase_label,
                    timeout_per_goal_s=float(timeout_per_goal_s),
                )

            # Run any pre-grasp goals unchanged.
            if int(first_grasp_idx) > 0:
                if not _run_goals_blocking(
                    [copy.deepcopy(g) for g in goals_to_run[:first_grasp_idx]],
                    phase_label=f"{phase_label} pre",
                    timeout_per_goal_s=float(timeout_per_goal_s),
                ):
                    return False

            g_open, g_close = _pair_open_close_values()
            # If no learned close exists, fall back to configured device close.
            if self._auto_gripper_closed is None:
                device_close = float(
                    np.clip(
                        float(DEVICE_GRIPPER_TOGGLE_CLOSE_JOINT),
                        float(self.ros_node.JOINT_LIMITS[7][0]),
                        float(self.ros_node.JOINT_LIMITS[7][1]),
                    )
                )
                g_close = float(min(float(g_close), float(device_close)))
            if not _move_to_snapshot_with_safe_lift(
                self._auto_pose_grasp,
                gripper=float(g_open),
                note=f"{phase_label}: to grasp (stored)",
                timeout_s=12.0,
            ):
                return False
            if not _move_to_snapshot(
                self._auto_pose_grasp,
                gripper=float(g_close),
                note=f"{phase_label}: close at grasp (stored)",
                timeout_s=8.0,
            ):
                return False
            # Ensure grasp-start pose is fully settled before curve first waypoint.
            j_settle = _snapshot_joint6(self._auto_pose_grasp)
            h_settle = _snapshot_head(self._auto_pose_grasp)
            if isinstance(j_settle, list) and len(j_settle) >= 6:
                if not self._execute_arm_to_chunked(
                    [float(v) for v in j_settle[:6]],
                    gripper=float(g_close),
                    head=h_settle,
                    timeout_s=6.0,
                    reliable=True,
                ):
                    return False
                time.sleep(float(max(0.0, float(CURVE_RESET_INTER_STEP_SETTLE_S))))
                # Send one extra close-hold at the grasp pose for robustness.
                if not self._execute_arm_to_chunked(
                    [float(v) for v in j_settle[:6]],
                    gripper=float(g_close),
                    head=h_settle,
                    timeout_s=4.0,
                    reliable=True,
                ):
                    return False
            self._set_manual_gripper_override(float(g_close))
            self.ros_node.sync_command_targets_to_actual()

            tail_goals = [copy.deepcopy(g) for g in goals_to_run[first_grasp_idx + 1 :] if isinstance(g, dict)]
            if len(tail_goals) <= 0:
                return True
            first_tail = tail_goals[0]
            if str(first_tail.get("kind", "")) in ("drag", "drag_curve"):
                first_tail["_force_keep_current_lift"] = True
            return _run_goals_blocking(
                tail_goals,
                phase_label=f"{phase_label} tail",
                timeout_per_goal_s=float(timeout_per_goal_s),
            )

        def _build_reset_plan(
            main_goals: list[dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], bool, bool, bool]:
            """Returns (type2_goals, return_after_type1, return_after_type2, run_type2_before_loop)."""
            def _safe_int(v: Any, default: int = 0) -> int:
                try:
                    return int(v)
                except Exception:
                    return int(default)

            selected_task = self._selected_task_name_for_loop().strip()
            first_grasp = next((g for g in main_goals if str(g.get("kind", "")) == "grasp"), None)
            first_reach = next((g for g in main_goals if str(g.get("kind", "")) == "reach"), None)
            first_place_object = next((g for g in main_goals if str(g.get("kind", "")) == "place_object"), None)
            drag_goal = next(
                (g for g in main_goals if str(g.get("kind", "")) in ("drag", "drag_curve")),
                None,
            )

            # Case 1: grasp + drag/curve style tasks.
            # Reset in type2 by grasping at drag end and replaying the path in reverse, then release.
            if isinstance(drag_goal, dict):
                end_xyz = _extract_drag_endpoint_xyz(drag_goal, use_end=True)
                start_xyz = _extract_drag_endpoint_xyz(drag_goal, use_end=False)
                if (end_xyz is not None) and (start_xyz is not None):
                    end_px = drag_goal.get("end_px")
                    end_py = drag_goal.get("end_py")
                    if not isinstance(end_px, int):
                        end_px = None
                    if not isinstance(end_py, int):
                        end_py = None
                    grasp_back = _build_grasp_goal_at(
                        end_xyz,
                        px=end_px,
                        py=end_py,
                        template=(first_grasp if isinstance(first_grasp, dict) else None),
                        post_grasp_lift=False,
                    )
                    reverse_drag = copy.deepcopy(drag_goal)
                    reverse_drag["px"] = _safe_int(drag_goal.get("end_px", drag_goal.get("px", 0)), default=0)
                    reverse_drag["py"] = _safe_int(drag_goal.get("end_py", drag_goal.get("py", 0)), default=0)
                    reverse_drag["end_px"] = _safe_int(drag_goal.get("px", drag_goal.get("end_px", 0)), default=0)
                    reverse_drag["end_py"] = _safe_int(drag_goal.get("py", drag_goal.get("end_py", 0)), default=0)
                    # Keep reverse carry-back at current contact lift (no safe-lift hop).
                    reverse_drag["_force_keep_current_lift"] = True
                    ppx = drag_goal.get("path_px")
                    if isinstance(ppx, list) and len(ppx) >= 2:
                        reverse_drag["path_px"] = list(reversed(ppx))
                    pb = drag_goal.get("path_base_xyz")
                    if isinstance(pb, list) and len(pb) >= 2:
                        reverse_drag["path_base_xyz"] = list(reversed(pb))
                    release_back = {
                        "kind": "release",
                        "use_previous_goal_end": True,
                        "source_goal_kind": str(reverse_drag.get("kind", "drag")),
                        "created_time": float(time.time()),
                    }
                    # For drag/curve:
                    # - no return after type1 (object ends at release target)
                    # - return after type2 (object restored, then go home)
                    # - run one type2 first before replay loop to restore object after first manual trial.
                    return [grasp_back, reverse_drag, release_back], False, True, True

            # Case 2: lift-object style (grasp+lift only): put down at initial object point in type2, then home.
            if (
                str(selected_task) == "Lift object"
                and isinstance(first_grasp, dict)
                and (not any(str(g.get("kind", "")) in ("drag", "drag_curve", "reach") for g in main_goals))
            ):
                grasp_xyz = _extract_xyz_from_goal(first_grasp)
                if grasp_xyz is not None:
                    reach_back = _build_reach_goal_at(
                        grasp_xyz,
                        px=(int(first_grasp.get("px")) if isinstance(first_grasp.get("px"), int) else None),
                        py=(int(first_grasp.get("py")) if isinstance(first_grasp.get("py"), int) else None),
                        template=(first_reach if isinstance(first_reach, dict) else None),
                    )
                    # Lift task: put object back (type2), return home, then start replay.
                    return [reach_back], False, True, True

            # Case 3: place-only task (operator starts with object in hand):
            # type1 = place + return, type2 = grasp-from-place + return.
            if (
                str(selected_task) == "Place object"
                and (first_grasp is None)
                and (isinstance(first_reach, dict) or isinstance(first_place_object, dict))
            ):
                place_ref = first_reach if isinstance(first_reach, dict) else first_place_object
                place_xyz = _extract_xyz_from_goal(place_ref)
                if place_xyz is not None:
                    grasp_back = _build_grasp_goal_at(
                        place_xyz,
                        px=(int(place_ref.get("px")) if isinstance(place_ref.get("px"), int) else None),
                        py=(int(place_ref.get("py")) if isinstance(place_ref.get("py"), int) else None),
                        template=None,
                        post_grasp_lift=True,
                    )
                    # Place-only task starts with object in hand:
                    # after first manual place, run type2 once to re-grasp from place and return home.
                    return [grasp_back], True, True, True

            # Default fallback: replay main sequence then return home (existing behavior).
            return [], True, False, False

        try:
            repeats_after_first = int(max(1, int(self._auto_loop_requested_rounds)))
            main_goals = [copy.deepcopy(g) for g in self._goal_sequence_order() if isinstance(g, dict)]
            if len(main_goals) == 0:
                raise RuntimeError("no queued goals for auto replay")
            first_grasp_goal = next((g for g in main_goals if str(g.get("kind", "")) == "grasp"), None)
            first_reach_goal = next((g for g in main_goals if str(g.get("kind", "")) == "reach"), None)
            first_place_object_goal = next((g for g in main_goals if str(g.get("kind", "")) == "place_object"), None)
            selected_task = self._selected_task_name_for_loop().strip()
            is_single_grasp_goal = bool(
                len(main_goals) == 1
                and isinstance(first_grasp_goal, dict)
                and str(first_grasp_goal.get("kind", "")) == "grasp"
            )
            is_single_reach_goal = bool(
                len(main_goals) == 1
                and isinstance(first_reach_goal, dict)
                and str(first_reach_goal.get("kind", "")) == "reach"
            )
            is_single_place_object_goal = bool(
                len(main_goals) == 1
                and isinstance(first_place_object_goal, dict)
                and str(first_place_object_goal.get("kind", "")) == "place_object"
            )
            single_grasp_with_lift = bool(
                bool(is_single_grasp_goal)
                and bool(first_grasp_goal.get("post_grasp_lift", True))
            )
            single_grasp_no_lift = bool(
                bool(is_single_grasp_goal)
                and (not bool(first_grasp_goal.get("post_grasp_lift", True)))
            )
            single_grasp_rotate_deg = (
                float(first_grasp_goal.get("grasp_rotate_deg"))
                if bool(single_grasp_no_lift)
                and isinstance(first_grasp_goal.get("grasp_rotate_deg"), (int, float))
                and math.isfinite(float(first_grasp_goal.get("grasp_rotate_deg")))
                else 0.0
            )
            single_grasp_force_pitch_rad = (
                float(first_grasp_goal.get("wrist_pitch_target"))
                if bool(single_grasp_no_lift)
                and isinstance(first_grasp_goal.get("wrist_pitch_target"), (int, float))
                and math.isfinite(float(first_grasp_goal.get("wrist_pitch_target")))
                else None
            )
            use_single_grasp_task_cycle = bool(
                bool(is_single_grasp_goal)
                and (
                    str(selected_task) in ("Grasp object", "Lift object")
                    or bool(single_grasp_with_lift)
                    or bool(single_grasp_no_lift)
                )
                and isinstance(self._auto_pose_grasp, dict)
            )
            use_single_reach_task_cycle = bool(
                bool(is_single_reach_goal)
                and (not isinstance(first_grasp_goal, dict))
            )
            use_single_place_object_task_cycle = bool(
                bool(is_single_place_object_goal)
                and (not isinstance(first_grasp_goal, dict))
            )
            has_release_goal = any(str(g.get("kind", "")) == "release" for g in main_goals)
            has_curve_goal = any(str(g.get("kind", "")) == "drag_curve" for g in main_goals)
            use_curve_pick_place_reset = bool(
                bool(has_curve_goal)
                and isinstance(first_grasp_goal, dict)
                and isinstance(self._auto_pose_grasp, dict)
                and isinstance(self._auto_pose_release, dict)
            )
            use_joint_pair_replay = bool(
                isinstance(first_grasp_goal, dict)
                and bool(has_release_goal)
                and (not bool(has_curve_goal))
                and isinstance(self._auto_pose_grasp, dict)
                and isinstance(self._auto_pose_release, dict)
            )
            if bool(use_joint_pair_replay):
                reset_goals: list[dict[str, Any]] = []
                return_after_type1 = False
                return_after_type2 = False
                run_type2_before_loop = True
            elif bool(use_single_grasp_task_cycle):
                reset_goals = []
                return_after_type1 = False
                return_after_type2 = False
                # Single-goal grasp cycles (with/without lift) handle first-trial
                # restore explicitly before Return, so skip pre-loop type2 reset.
                run_type2_before_loop = False
            elif bool(use_single_reach_task_cycle):
                reset_goals = []
                return_after_type1 = False
                return_after_type2 = False
                run_type2_before_loop = False
            elif bool(use_single_place_object_task_cycle):
                reset_goals = []
                return_after_type1 = False
                return_after_type2 = False
                run_type2_before_loop = False
            elif bool(use_curve_pick_place_reset):
                reset_goals = []
                return_after_type1 = False
                return_after_type2 = False
                run_type2_before_loop = True
            else:
                reset_goals, return_after_type1, return_after_type2, run_type2_before_loop = _build_reset_plan(main_goals)
            self._set_status(
                f"Auto loop started for queued sequence: repeats after first={repeats_after_first}.",
                "QLabel { color: #1e88e5; font-size: 10px; }",
            )

            # After first manual trial, optionally run one type2 reset before auto replay.
            # This restores object/start conditions (e.g., drag reverse put-back) so that
            # iteration-1 type1 begins from the expected initial scene.
            if bool(run_type2_before_loop) and (not self._auto_loop_abort):
                if self._loop_record_session_active:
                    if not self._start_auto_loop_record_segment(kind="type2", round_idx=0):
                        raise RuntimeError("auto-loop record failed to start pre-loop type2 segment")
                if bool(use_joint_pair_replay):
                    if not _run_joint_pair_phase(forward=False, phase_label="Type2 pre"):
                        raise RuntimeError("failed while executing pre-loop type2 joint replay")
                    if not _return_home_after_pair(note="Auto replay setup: return home after pre-reset..."):
                        raise RuntimeError("failed to return home after pre-loop type2")
                elif bool(use_single_grasp_task_cycle):
                    if not _run_single_grasp_type2(
                        with_lift=bool(single_grasp_with_lift),
                        phase_label="Type2 pre",
                        rotate_deg=float(single_grasp_rotate_deg),
                        force_pitch_rad=single_grasp_force_pitch_rad,
                    ):
                        raise RuntimeError("failed while executing pre-loop single-grasp reset")
                elif bool(use_curve_pick_place_reset):
                    if not _run_curve_pick_place_reset(phase_label="Type2 pre"):
                        raise RuntimeError("failed while executing pre-loop curve type2 reset")
                elif len(reset_goals) > 0:
                    self._set_status(
                        "Auto replay setup: type2 pre-reset before iteration 1...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not _run_goals_blocking(
                        reset_goals,
                        phase_label="Type2 pre",
                        timeout_per_goal_s=240.0,
                    ):
                        raise RuntimeError("failed while executing pre-loop type2 reset")
                    if bool(return_after_type2) and (self._pre_action_state is not None):
                        self._set_status(
                            "Auto replay setup: return after pre-reset...",
                            "QLabel { color: #1e88e5; font-size: 10px; }",
                        )
                        self.return_to_start()
                        if not _wait_until_idle(timeout_s=180.0, require_no_pre_state=True):
                            raise RuntimeError("timed out while returning after pre-loop reset")
                if self._loop_record_session_active:
                    self._stop_auto_loop_record_segment(save=True)

            for rep_idx in range(1, repeats_after_first + 1):
                if self._auto_loop_abort:
                    break
                self._auto_rounds_left = int(repeats_after_first - rep_idx + 1)

                if self._loop_record_session_active:
                    if not self._start_auto_loop_record_segment(kind="type1", round_idx=int(rep_idx)):
                        raise RuntimeError("auto-loop record failed to start replay segment")

                self._set_status(
                    f"Auto replay {rep_idx}/{repeats_after_first}: type1 main sequence...",
                    "QLabel { color: #1e88e5; font-size: 10px; }",
                )
                if bool(use_joint_pair_replay):
                    if not _run_joint_pair_phase(forward=True, phase_label=f"Type1 r{rep_idx}"):
                        raise RuntimeError("failed while executing type1 forward joint replay")
                elif bool(use_single_grasp_task_cycle):
                    if not _run_single_grasp_type1(
                        with_lift=bool(single_grasp_with_lift),
                        phase_label=f"Type1 r{rep_idx}",
                        rotate_deg=float(single_grasp_rotate_deg),
                        force_pitch_rad=single_grasp_force_pitch_rad,
                    ):
                        raise RuntimeError("failed while executing type1 single-grasp replay")
                elif bool(use_single_reach_task_cycle):
                    if not _run_single_reach_type1(
                        main_goals,
                        phase_label=f"Type1 r{rep_idx}",
                    ):
                        raise RuntimeError("failed while executing type1 single-reach replay")
                elif bool(use_single_place_object_task_cycle):
                    if not _run_single_place_object_type1(
                        main_goals,
                        phase_label=f"Type1 r{rep_idx}",
                    ):
                        raise RuntimeError("failed while executing type1 single-place-object replay")
                elif bool(use_curve_pick_place_reset):
                    if not _run_curve_forward_from_snapshot(
                        main_goals,
                        phase_label=f"Type1 r{rep_idx}",
                        timeout_per_goal_s=240.0,
                    ):
                        raise RuntimeError("failed while executing type1 curve forward replay")
                else:
                    if not _run_goals_blocking(
                        main_goals,
                        phase_label=f"Type1 r{rep_idx}",
                        timeout_per_goal_s=240.0,
                    ):
                        raise RuntimeError("failed while executing type1 main sequence")
                if self._auto_loop_abort:
                    break

                if (not bool(use_joint_pair_replay)) and bool(return_after_type1) and (self._pre_action_state is not None):
                    self._set_status(
                        f"Auto replay {rep_idx}/{repeats_after_first}: return after type1...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    self.return_to_start()
                    if not _wait_until_idle(timeout_s=180.0, require_no_pre_state=True):
                        raise RuntimeError("timed out while returning after type1")

                if self._loop_record_session_active:
                    self._stop_auto_loop_record_segment(save=True)

                if self._auto_loop_abort:
                    break

                # Type2 reset segment: bring object back to ready/start state for next iteration.
                if self._loop_record_session_active:
                    if not self._start_auto_loop_record_segment(kind="type2", round_idx=int(rep_idx)):
                        raise RuntimeError("auto-loop record failed to start type2 reset segment")

                if bool(use_joint_pair_replay):
                    self._set_status(
                        f"Auto replay {rep_idx}/{repeats_after_first}: type2 reverse reset...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not _run_joint_pair_phase(forward=False, phase_label=f"Type2 r{rep_idx}"):
                        raise RuntimeError("failed while executing type2 reverse joint replay")
                    if not _return_home_after_pair(
                        note=f"Auto replay {rep_idx}/{repeats_after_first}: return home after type2...",
                    ):
                        raise RuntimeError("failed while returning home after type2 reverse replay")
                elif bool(use_single_grasp_task_cycle):
                    self._set_status(
                        f"Auto replay {rep_idx}/{repeats_after_first}: type2 single-grasp reset...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not _run_single_grasp_type2(
                        with_lift=bool(single_grasp_with_lift),
                        phase_label=f"Type2 r{rep_idx}",
                        rotate_deg=float(single_grasp_rotate_deg),
                        force_pitch_rad=single_grasp_force_pitch_rad,
                    ):
                        raise RuntimeError("failed while executing type2 single-grasp reset")
                elif bool(use_single_reach_task_cycle):
                    self._set_status(
                        f"Auto replay {rep_idx}/{repeats_after_first}: type2 single-reach return...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not _run_single_reach_type2(
                        phase_label=f"Type2 r{rep_idx}",
                    ):
                        raise RuntimeError("failed while executing type2 single-reach return")
                elif bool(use_single_place_object_task_cycle):
                    self._set_status(
                        f"Auto replay {rep_idx}/{repeats_after_first}: type2 single-place-object return...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not _run_single_place_object_type2(
                        phase_label=f"Type2 r{rep_idx}",
                    ):
                        raise RuntimeError("failed while executing type2 single-place-object return")
                elif bool(use_curve_pick_place_reset):
                    self._set_status(
                        f"Auto replay {rep_idx}/{repeats_after_first}: type2 curve reset...",
                        "QLabel { color: #1e88e5; font-size: 10px; }",
                    )
                    if not _run_curve_pick_place_reset(phase_label=f"Type2 r{rep_idx}"):
                        raise RuntimeError("failed while executing type2 curve reset")
                else:
                    if len(reset_goals) > 0:
                        self._set_status(
                            f"Auto replay {rep_idx}/{repeats_after_first}: type2 reset sequence...",
                            "QLabel { color: #1e88e5; font-size: 10px; }",
                        )
                        if not _run_goals_blocking(
                            reset_goals,
                            phase_label=f"Type2 r{rep_idx}",
                            timeout_per_goal_s=240.0,
                        ):
                            raise RuntimeError("failed while executing type2 reset sequence")
                    else:
                        self._set_status(
                            f"Auto replay {rep_idx}/{repeats_after_first}: type2 reset skipped (none needed).",
                            "QLabel { color: #1e88e5; font-size: 10px; }",
                        )

                    if bool(return_after_type2) and (self._pre_action_state is not None):
                        self._set_status(
                            f"Auto replay {rep_idx}/{repeats_after_first}: return after type2...",
                            "QLabel { color: #1e88e5; font-size: 10px; }",
                        )
                        self.return_to_start()
                        if not _wait_until_idle(timeout_s=180.0, require_no_pre_state=True):
                            raise RuntimeError("timed out while returning after type2")

                if self._loop_record_session_active:
                    self._stop_auto_loop_record_segment(save=True)

            if self._auto_loop_abort:
                self._set_status("Auto loop aborted", "QLabel { color: orange; font-size: 10px; }")
            else:
                self._set_status("Auto loop completed", "QLabel { color: green; font-size: 10px; }")
                loop_success = True
        except Exception as exc:
            self._stop_auto_loop_record_segment(save=False)
            self._set_status(f"Auto loop failed: {exc}", "QLabel { color: red; font-size: 10px; }")
            print(f"[auto_loop sequence] error: {exc}")
            traceback.print_exc()
        finally:
            if self._loop_record_session_active:
                if loop_success and not self._auto_loop_abort:
                    self.ui_loop_record_review_signal.emit()
                else:
                    self._stop_auto_loop_record_segment(save=False)
                    if len(self._loop_record_entries) > 0:
                        self.ui_loop_record_review_signal.emit()
                    elif self._auto_loop_abort or bool(self._loop_record_stop_requested):
                        self._update_auto_loop_record_button_ui()
                        self._reset_auto_loop_record_session_state()
                    else:
                        self._abort_auto_loop_record_session(keep_existing_files=False)
            self._auto_loop_running = False
            self._auto_loop_abort = False
            self._auto_rounds_left = 0
            self._auto_start_after_return = False
            self._auto_first_trial_pending = False
            self._auto_sequence_replay_active = False
            self._auto_loop_mode = "pick_place"
            self._set_action_state("idle")
            # Keep Return available after auto-loop completion/stop so operator
            # can always send a return-home command from the UI.
            self._set_return_enabled(True)
            QTimer.singleShot(0, self._update_auto_loop_progress_ui)

    def stop_auto_pick_place_loop(self) -> None:
        super().stop_auto_pick_place_loop()
        self._auto_sequence_replay_active = False
        self._auto_loop_mode = "pick_place"

    def _on_auto_loop_start_clicked(self) -> None:
        if bool(getattr(self, "_auto_loop_running", False) or getattr(self, "_auto_start_after_return", False)):
            self.stop_auto_pick_place_loop()
            self._update_auto_loop_start_stop_button_ui()
            return
        text = self.auto_loop_count_input.text().strip() if hasattr(self, "auto_loop_count_input") else "0"
        try:
            rounds = int(text)
        except Exception:
            self._set_status("Iterations must be an integer", "QLabel { color: red; font-size: 10px; }")
            return
        if int(rounds) <= 0:
            self._set_status("Iterations must be >= 1", "QLabel { color: red; font-size: 10px; }")
            return

        goals = self._goal_sequence_order()
        if len(goals) == 0:
            self._set_status(
                "No queued goals. Right-click image and add goals first.",
                "QLabel { color: orange; font-size: 10px; }",
            )
            return
        grasp_goal = self.queued_goals.get("grasp")
        has_grasp = isinstance(grasp_goal, dict) and bool(grasp_goal.get("post_grasp_lift", True))
        has_reach = bool(
            isinstance(self.queued_goals.get("reach"), dict)
            or isinstance(self.queued_goals.get("place_object"), dict)
        )

        # Keep existing pick-place auto-loop behavior when both grasp+reach exist.
        if has_grasp and has_reach:
            self._auto_loop_mode = "pick_place"
            self._auto_sequence_replay_active = False
            started = bool(self.start_auto_pick_place_loop(rounds))
            if started:
                self._update_auto_loop_start_stop_button_ui()
            return

        selected_task = self._selected_task_name_for_loop()
        if selected_task in AUTO_LOOP_COMPLEX_TASKS:
            self._set_status(
                f"Auto replay for '{selected_task}' is not enabled yet. Running one manual trial only.",
                "QLabel { color: orange; font-size: 10px; }",
            )
        else:
            if self._is_simple_goal_sequence_for_auto_loop(goals):
                started = bool(self.start_auto_goal_sequence_loop(rounds))
                if started:
                    self._update_auto_loop_start_stop_button_ui()
                return

        # Fallback path: one manual trial + record/review, no auto replay.
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Wait for current action to finish", "QLabel { color: orange; font-size: 10px; }")
                return
        if not self._start_non_autoloop_record_session():
            return
        self.execute_all_queued_goals()
        self._start_non_autoloop_record_finalize_watcher()

    def _on_auto_loop_stop_clicked(self) -> None:
        self.stop_auto_pick_place_loop()
        self._update_auto_loop_start_stop_button_ui()

    def _pixel_from_event(self, event):
        """Map click position on scaled QLabel to image pixel (handles letterboxing)."""
        if self.head_rgb is None or not hasattr(self, "head_display"):
            return None, None

        disp = self.head_display
        full_h, img_w = self.head_rgb.shape[:2]
        crop_frac = float(np.clip(float(HEAD_DISPLAY_CROP_BOTTOM_FRAC), 0.0, 0.95))
        img_h = full_h
        if crop_frac > 1e-6:
            img_h = max(1, int(round(float(full_h) * (1.0 - crop_frac))))
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
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            QTimer.singleShot(0, self._update_goal_queue_label)
            return
        super()._update_goal_queue_label()
        if not hasattr(self, "goal_list_widget"):
            return
        self.goal_list_widget.clear()
        goals = self._goal_sequence_order()
        for idx, goal in enumerate(goals):
            kind = str(goal.get("kind", "?"))
            marker = "-> " if idx == self.queued_goal_cursor and self._goal_sequence_has_next() else "   "
            if kind == "drag":
                sx = goal.get("px")
                sy = goal.get("py")
                ex = goal.get("end_px")
                ey = goal.get("end_py")
                text = f"{marker}{idx + 1}. drag ({sx}, {sy}) -> ({ex}, {ey})"
            elif kind == "grasp":
                px = goal.get("px")
                py = goal.get("py")
                rotate_deg = goal.get("grasp_rotate_deg")
                is_rotate = isinstance(rotate_deg, (int, float)) and abs(float(rotate_deg)) > 1e-6
                is_precise = bool(goal.get("precise_grasp", False))
                if bool(is_precise):
                    text = f"{marker}{idx + 1}. grasp precise ({px}, {py})"
                elif bool(goal.get("post_grasp_lift", True)):
                    text = f"{marker}{idx + 1}. grasp+lift ({px}, {py})"
                elif bool(is_rotate):
                    text = f"{marker}{idx + 1}. grasp+rotate ({float(rotate_deg):+.1f}deg) ({px}, {py})"
                else:
                    text = f"{marker}{idx + 1}. grasp ({px}, {py})"
            elif kind == "drag_curve":
                sx = goal.get("px")
                sy = goal.get("py")
                ex = goal.get("end_px")
                ey = goal.get("end_py")
                pts = goal.get("path_px")
                npts = len(pts) if isinstance(pts, list) else 0
                no_adj = bool(goal.get("no_height_adjustment", False))
                if no_adj:
                    htxt = "keep_lift"
                else:
                    h_cm = float(goal.get("surface_height_offset_cm", float(goal.get("surface_height_offset_m", 0.0)) * 100.0))
                    htxt = f"h={h_cm:+.1f}cm"
                text = f"{marker}{idx + 1}. curve ({sx}, {sy}) -> ({ex}, {ey}), n={npts}, {htxt}"
            elif kind == "release":
                src = str(goal.get("source_goal_kind", "?"))
                text = f"{marker}{idx + 1}. release (prev end: {src})"
            elif kind in ("lift_delta", "stretch_delta", "translate_delta"):
                delta_cm = float(goal.get("delta_cm", float(goal.get("delta_m", 0.0)) * 100.0))
                axis = {
                    "lift_delta": "lift",
                    "stretch_delta": "stretch",
                    "translate_delta": "translate",
                }.get(kind, kind)
                text = f"{marker}{idx + 1}. {axis} {delta_cm:+.2f} cm"
            else:
                px = goal.get("px")
                py = goal.get("py")
                if kind == "reach" and bool(goal.get("precise_place", False)):
                    text = f"{marker}{idx + 1}. place precise ({px}, {py})"
                else:
                    text = f"{marker}{idx + 1}. {kind} ({px}, {py})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, int(idx))
            self.goal_list_widget.addItem(item)
        self._on_goal_selection_changed()

    def _update_next_goal_button_state(self):
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            QTimer.singleShot(0, self._update_next_goal_button_state)
            return
        super()._update_next_goal_button_state()
        goals = self._goal_sequence_order()
        has_goals = len(goals) > 0
        with self._action_lock:
            state = self._action_state
        active = state in ("running", "paused", "awaiting_confirm", "awaiting_post_grasp", "awaiting_post_reach_release")
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
        idx_data = item.data(Qt.ItemDataRole.UserRole)
        try:
            idx = int(idx_data)
        except Exception:
            return
        goals = self._goal_sequence_order()
        if idx < 0 or idx >= len(goals):
            return
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Remove goals only while idle", "QLabel { color: orange; font-size: 10px; }")
                return
        removed = self.queued_goal_sequence.pop(int(idx))
        self._rebuild_queued_goal_map_from_sequence()
        self._reset_goal_sequence_progress()
        self._set_status(
            f"Removed queued {str(removed.get('kind', '?'))} goal",
            "QLabel { color: #1e88e5; font-size: 10px; }",
        )

    def clear_queued_goals(self):
        with self._action_lock:
            if self._action_state != "idle":
                self._set_status("Clear goals only while idle", "QLabel { color: orange; font-size: 10px; }")
                return
        self.queued_goal_sequence = []
        self._rebuild_queued_goal_map_from_sequence()
        self._reset_goal_sequence_progress()
        self._set_status("Cleared queued goals", "QLabel { color: gray; font-size: 10px; }")

    def execute_all_queued_goals(self, *, drag_repeat_count: int = 1, drag_return_to_start: bool = False):
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
        try:
            parsed_drag_repeats = int(drag_repeat_count)
        except Exception:
            parsed_drag_repeats = 1
        self._queued_drag_repeat_count = int(max(1, parsed_drag_repeats))
        self._queued_drag_return_to_start = bool(drag_return_to_start)
        self._run_all_queued_goals = True
        self._deferred_next_goal_start = False
        if not self._start_next_queued_goal():
            self._run_all_queued_goals = False
            self._deferred_next_goal_start = False
            self._set_status("Failed to start goal execution", "QLabel { color: red; font-size: 10px; }")

    def abort_and_clear_goals(self):
        with self._action_lock:
            state = self._action_state
            self._action_abort_requested = True
        self._run_all_queued_goals = False
        self._deferred_next_goal_start = False
        self._skip_to_next_goal_requested = False
        self.queued_goal_sequence = []
        self._rebuild_queued_goal_map_from_sequence()
        self._reset_goal_sequence_progress()
        if state in ("paused", "awaiting_confirm", "awaiting_post_grasp", "awaiting_post_reach_release"):
            self._set_action_state("running")
        if state == "idle":
            self._set_status("Cleared queued goals", "QLabel { color: gray; font-size: 10px; }")
        else:
            self._set_status(
                "Abort requested. Cleared queued goals.",
                "QLabel { color: orange; font-size: 10px; }",
            )

    def keyPressEvent(self, event):
        if event is not None and (not event.isAutoRepeat()):
            if event.key() == Qt.Key.Key_R:
                mods = event.modifiers()
                if not (mods & (Qt.KeyboardModifier.ControlModifier |
                                Qt.KeyboardModifier.AltModifier |
                                Qt.KeyboardModifier.MetaModifier)):
                    self._on_record_shortcut()
                    event.accept()
                    return
            if event.key() == Qt.Key.Key_E:
                mods = event.modifiers()
                if not (mods & (Qt.KeyboardModifier.ControlModifier |
                                Qt.KeyboardModifier.AltModifier |
                                Qt.KeyboardModifier.MetaModifier)):
                    self._on_execute_goals_shortcut()
                    event.accept()
                    return
        super().keyPressEvent(event)


def main() -> None:
    bridge = StretchAIDemoBridge()

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
