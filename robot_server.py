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

import argparse
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
CROSSHAIR_OFFSET_Y = 38  # pixels down from center (tune until dot matches gripper close point)
CROSSHAIR_OFFSET_X = 20   # pixels right from center

# Camera device indices — override with CAM_TOP / CAM_WRIST env vars
# Or use CAM_TOP_NAME / CAM_WRIST_NAME to find cameras by device name
CAMERA_TOP_IDX = int(os.environ.get("CAM_TOP", "4"))
CAMERA_WRIST_IDX = int(os.environ.get("CAM_WRIST", "0"))
CAMERA_TOP_NAME = os.environ.get("CAM_TOP_NAME", "Logitech Webcam C930e")
CAMERA_WRIST_NAME = os.environ.get("CAM_WRIST_NAME", "USB2.0_CAM1")

# SO-101 port — override with ROBOT_PORT env var
ROBOT_PORT = os.environ.get("ROBOT_PORT", "/dev/ttyACM0")

app = Flask(__name__)

# ---- Robot + Camera State ----
robot = None
cameras = {}
robot_lock = threading.Lock()

def autodetect_serial_port():
    """Return the first available ttyACM* or ttyUSB* port, or None."""
    import glob
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if candidates:
        print(f"[robot] Auto-detected serial ports: {candidates}")
        return candidates[0]
    return None

def find_camera_index_by_name(name):
    """Find the first /dev/videoX index whose device name contains the given string."""
    import glob
    for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            with open(path) as f:
                dev_name = f.read().strip()
            if name.lower() in dev_name.lower():
                idx = int(path.split("/video")[2].split("/")[0])
                print(f"[camera] Found '{dev_name}' at index {idx}")
                return idx
        except Exception:
            continue
    return None

def autodetect_cameras():
    """Return list of working camera indices (tries 0-9)."""
    import cv2
    found = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(idx)
        cap.release()
    print(f"[camera] Auto-detected camera indices: {found}")
    return found

def get_motor_ids():
    """Extract integer motor IDs from the bus, handling [model, id] list format."""
    if hasattr(robot.bus, 'motor_ids'):
        return list(robot.bus.motor_ids)
    if hasattr(robot.bus, 'motors'):
        ids = []
        for v in robot.bus.motors.values():
            if isinstance(v, (list, tuple)):
                # Values are [model_name, id] — grab only the int
                ids += [x for x in v if isinstance(x, int)]
            elif isinstance(v, int):
                ids.append(v)
        if ids:
            return ids
    return list(range(1, 7))  # SO-101 fallback: servos 1-6


def init_robot():
    global robot
    port = ROBOT_PORT
    if not os.path.exists(port):
        print(f"[robot] {port} not found, auto-detecting...")
        port = autodetect_serial_port() or ROBOT_PORT
    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        config = SOFollowerRobotConfig(port=port)
        robot = SOFollower(config)
        robot.connect(calibrate=False)
        print(f"[robot] Connected to SO-101 on {port}")
        # Enable torque so servos respond to position commands
        try:
            ids = get_motor_ids()
            robot.bus.write("Torque_Enable", [1] * len(ids), ids)
            print(f"[robot] Torque enabled on {len(ids)} servos")
        except Exception as te:
            print(f"[robot] Torque enable failed: {te} — arm may not move")
    except Exception as e:
        print(f"[robot] WARNING: Could not connect to arm: {e}")
        print("[robot] Running in camera-only mode")
        robot = None

def init_cameras():
    import cv2

    # Resolve indices by name first, fall back to configured index, then auto-detect
    top_idx = find_camera_index_by_name(CAMERA_TOP_NAME) if CAMERA_TOP_NAME else None
    if top_idx is None:
        top_idx = CAMERA_TOP_IDX
        print(f"[camera] Name lookup failed for top, using index {top_idx}")

    wrist_idx = find_camera_index_by_name(CAMERA_WRIST_NAME) if CAMERA_WRIST_NAME else None
    if wrist_idx is None:
        wrist_idx = CAMERA_WRIST_IDX
        print(f"[camera] Name lookup failed for wrist, using index {wrist_idx}")

    # Auto-detect if resolved indices still don't open
    needs_autodetect = []
    for name, idx in [("top", top_idx), ("wrist", wrist_idx)]:
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            needs_autodetect.append(name)

    if needs_autodetect:
        print(f"[camera] Could not open configured indices for {needs_autodetect}, auto-detecting...")
        available = autodetect_cameras()
        if len(available) >= 2:
            top_idx, wrist_idx = available[0], available[1]
            print(f"[camera] Using auto-detected: top={top_idx}, wrist={wrist_idx}")
        elif len(available) == 1:
            top_idx = wrist_idx = available[0]
            print(f"[camera] Only one camera found (index {available[0]}), using for both")

    for name, idx in [("top", top_idx), ("wrist", wrist_idx)]:
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
    # Flush stale buffer frames before capturing
    for _ in range(4):
        cam.grab()
    ret, frame = cam.read()
    if not ret:
        # Try reopening the camera once
        idx = CAMERA_WRIST_IDX if name == "wrist" else CAMERA_TOP_IDX
        print(f"[camera] {name} read failed, reopening index {idx}...")
        cam.release()
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            cameras[name] = cam
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

def mjpeg_generator(name):
    import cv2
    while True:
        cam = cameras.get(name)
        if cam is None:
            break
        for _ in range(4):
            cam.grab()
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, (SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT))
        # Draw crosshair on wrist stream so user sees what AI sees
        if name == "wrist":
            h, w = frame.shape[:2]
            cx, cy = w // 2 + CROSSHAIR_OFFSET_X, h // 2 + CROSSHAIR_OFFSET_Y
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy),  7, (0, 0, 0),       -1)
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 0, 0), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 0, 0), 2)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.016)  # ~60 fps

@app.route("/stream/<name>")
def stream(name):
    if name not in ["top", "wrist"]:
        return jsonify({"error": "Unknown camera. Use 'top' or 'wrist'"}), 400
    from flask import Response, stream_with_context
    return Response(stream_with_context(mjpeg_generator(name)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream")
def stream_index():
    return """<html><body style="background:#111;display:flex;gap:16px;padding:16px">
    <div><p style="color:white;text-align:center;font-family:sans-serif;font-size:18px">Top</p>
    <img src="/stream/top" style="width:720px"></div>
    <div><p style="color:white;text-align:center;font-family:sans-serif;font-size:18px">Wrist</p>
    <img src="/stream/wrist" style="width:720px"></div>
    </body></html>"""

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
        action = {f"{k}.pos": float(v) for k, v in data.items()}
        with robot_lock:
            robot.send_action(action)
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
        action = {f"{k}.pos": float(v) for k, v in PRESETS[name].items()}
        with robot_lock:
            robot.send_action(action)
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
        val = 1 if enabled else 0
        with robot_lock:
            ids = get_motor_ids()
            robot.bus.write("Torque_Enable", [val] * len(ids), ids)
        return jsonify({"ok": True, "torque_enabled": enabled})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Main ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="Serial port for robot (default: /dev/ttyACM0)")
    args = parser.parse_args()
    if args.port:
        ROBOT_PORT = args.port

    print("[boot] Starting SO-101 Robot Agent Server...")
    print(f"[boot] Robot port: {ROBOT_PORT}")
    init_robot()
    init_cameras()
    print(f"[boot] Server running on http://0.0.0.0:{PORT}")
    print(f"[boot] Cameras available: {list(cameras.keys())}")
    print(f"[boot] Robot connected: {robot is not None}")
    print(f"[boot] Live stream: http://localhost:{PORT}/stream")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
