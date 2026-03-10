"""
SO-101 Robot Agent Server
Bridges Claude/OpenClaw to the SO-101 arm + cameras via HTTP API.

Usage:
    python robot_server.py

Endpoints:
    GET  /status              - arm joint positions + server health
    GET  /snapshot/wrist      - base64 JPEG from wrist camera
    GET  /snapshot/top        - base64 JPEG from top camera
    POST /move                - move joints {joint_name: degrees, ...}
    POST /move_preset         - move to named pose (home, ready, rest)
    POST /enable              - enable/disable torque {"enabled": true/false}
"""

import base64
import json
import os
import threading
import time
from io import BytesIO
from flask import Flask, jsonify, request

# ---- Config ----
PORT = 7878
SNAPSHOT_WIDTH = 256
SNAPSHOT_HEIGHT = 192
JPEG_QUALITY = 60

# Camera device indices — adjust to match your setup
CAMERA_TOP_IDX = int(os.environ.get("CAM_TOP", "0"))
CAMERA_WRIST_IDX = int(os.environ.get("CAM_WRIST", "1"))

# SO-101 port — adjust to your serial port
ROBOT_PORT = os.environ.get("ROBOT_PORT", "/dev/ttyUSB0")

app = Flask(__name__)

# ---- Robot + Camera State ----
robot = None
cameras = {}
robot_lock = threading.Lock()

def init_robot():
    global robot
    try:
        from lerobot.common.robot_devices.robots.factory import make_robot
        from lerobot.common.robot_devices.robots.configs import So101RobotConfig
        config = So101RobotConfig(port=ROBOT_PORT)
        robot = make_robot(config)
        robot.connect()
        print(f"[robot] Connected to SO-101 on {ROBOT_PORT}")
    except Exception as e:
        print(f"[robot] WARNING: Could not connect to arm: {e}")
        print("[robot] Running in camera-only mode")
        robot = None

def init_cameras():
    import cv2
    for name, idx in [("top", CAMERA_TOP_IDX), ("wrist", CAMERA_WRIST_IDX)]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, SNAPSHOT_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SNAPSHOT_HEIGHT)
            cameras[name] = cap
            print(f"[camera] {name} camera opened (index {idx})")
        else:
            print(f"[camera] WARNING: Could not open {name} camera (index {idx})")

def capture_snapshot(name):
    """Capture a frame, resize, return base64 JPEG string."""
    import cv2
    cam = cameras.get(name)
    if cam is None:
        return None, f"Camera '{name}' not available"
    ret, frame = cam.read()
    if not ret:
        return None, f"Failed to read from {name} camera"
    frame = cv2.resize(frame, (SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, buf = cv2.imencode(".jpg", frame, encode_params)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return b64, None

def get_joint_positions():
    if robot is None:
        return None
    try:
        with robot_lock:
            obs = robot.get_observation()
        # Extract joint positions (keys vary by lerobot version)
        joints = {k: float(v) for k, v in obs.items() if "pos" in k or "joint" in k}
        return joints if joints else {k: float(v) for k, v in obs.items()}
    except Exception as e:
        return {"error": str(e)}

# ---- Named Poses ----
PRESETS = {
    "home":  {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0},
    "ready": {"shoulder_pan": 0, "shoulder_lift": -45, "elbow_flex": 90, "wrist_flex": -45, "wrist_roll": 0, "gripper": 0},
    "rest":  {"shoulder_pan": 0, "shoulder_lift": -90, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0},
}

# ---- Routes ----

@app.route("/status")
def status():
    joints = get_joint_positions()
    return jsonify({
        "ok": True,
        "robot_connected": robot is not None,
        "cameras": list(cameras.keys()),
        "joints": joints,
        "timestamp": time.time()
    })

@app.route("/snapshot/<name>")
def snapshot(name):
    if name not in ["top", "wrist"]:
        return jsonify({"error": "Unknown camera. Use 'top' or 'wrist'"}), 400
    b64, err = capture_snapshot(name)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({
        "camera": name,
        "format": "jpeg",
        "width": SNAPSHOT_WIDTH,
        "height": SNAPSHOT_HEIGHT,
        "data": b64,
        "timestamp": time.time()
    })

@app.route("/move", methods=["POST"])
def move():
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json
    if not data:
        return jsonify({"error": "No joint data provided"}), 400
    try:
        with robot_lock:
            robot.send_action(data)
        return jsonify({"ok": True, "sent": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/move_preset", methods=["POST"])
def move_preset():
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json or {}
    name = data.get("pose")
    if name not in PRESETS:
        return jsonify({"error": f"Unknown pose '{name}'. Options: {list(PRESETS.keys())}"}), 400
    try:
        with robot_lock:
            robot.send_action(PRESETS[name])
        return jsonify({"ok": True, "pose": name, "joints": PRESETS[name]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/enable", methods=["POST"])
def enable():
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json or {}
    enabled = data.get("enabled", True)
    try:
        with robot_lock:
            if enabled:
                robot.activate_calibration() if hasattr(robot, "activate_calibration") else None
            else:
                robot.disconnect()
        return jsonify({"ok": True, "torque_enabled": enabled})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Main ----
if __name__ == "__main__":
    print("[boot] Starting SO-101 Robot Agent Server...")
    init_robot()
    init_cameras()
    print(f"[boot] Server running on http://0.0.0.0:{PORT}")
    print(f"[boot] Cameras available: {list(cameras.keys())}")
    print(f"[boot] Robot connected: {robot is not None}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
