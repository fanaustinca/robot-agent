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

# ---- Terminal Colors ----
class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

# ---- Config ----
PORT = 7878
SNAPSHOT_WIDTH = 256
SNAPSHOT_HEIGHT = 192
SNAPSHOT_TOP_WIDTH = 512
SNAPSHOT_TOP_HEIGHT = 384
JPEG_QUALITY = 60
CROSSHAIR_OFFSET_Y = 53  # pixels down from center (tune until dot matches gripper close point)
CROSSHAIR_OFFSET_X = 30   # pixels right from center

# Camera device indices — override with CAM_TOP / CAM_WRIST env vars
# Or use CAM_TOP_NAME / CAM_WRIST_NAME to find cameras by device name
CAMERA_TOP_IDX = int(os.environ.get("CAM_TOP", "4"))
CAMERA_WRIST_IDX = int(os.environ.get("CAM_WRIST", "0"))
CAMERA_TOP_NAME = os.environ.get("CAM_TOP_NAME", "Logitech Webcam C930e")
CAMERA_WRIST_NAME = os.environ.get("CAM_WRIST_NAME", "USB2.0_CAM1")

# SO-101 port — looked up by USB serial number at startup
# Override with ROBOT_PORT env var or --port CLI argument if needed
ROBOT_SERIAL = "5AE6083982"
ROBOT_PORT = os.environ.get("ROBOT_PORT", "/dev/ttyACM0")

# Leader arm for teleop — override with LEADER_PORT env var or --leader-port CLI arg
LEADER_PORT = os.environ.get("LEADER_PORT", "")
TELEOP_FPS = 60

app = Flask(__name__)

# ---- Robot + Camera State ----
robot = None
leader = None
cameras = {}
camera_info = {}  # {"top": {"name": ..., "index": ...}, "wrist": {...}}
camera_locks = {"top": threading.Lock(), "wrist": threading.Lock()}
robot_lock = threading.Lock()
torque_enabled = False
teleop_active = False
teleop_thread = None

# ---- Joint Position History (rolling 10 min buffer) ----
HISTORY_INTERVAL = 0.5   # seconds between samples
HISTORY_MAX_SECS = 600   # 10 minutes
HISTORY_MAX_SAMPLES = int(HISTORY_MAX_SECS / HISTORY_INTERVAL)
joint_history = []       # [{t: float, joints: {k: v, ...}}, ...]
history_lock = threading.Lock()

def history_recorder():
    """Background thread: sample joint positions into rolling buffer."""
    while True:
        joints = get_joint_positions()
        if joints and "error" not in joints:
            with history_lock:
                joint_history.append({"t": time.time(), "joints": dict(joints)})
                if len(joint_history) > HISTORY_MAX_SAMPLES:
                    joint_history[:] = joint_history[-HISTORY_MAX_SAMPLES:]
        time.sleep(HISTORY_INTERVAL)

# ---- Agent Activity State (set by agent, read by dashboard) ----
agent_state = {
    "phase": "idle",         # idle, calibrating, homing, aligning, waiting_confirm, lowering, gripping, lifting, dropping, done
    "detail": "",            # human-readable detail text
    "align_iteration": 0,   # current alignment iteration
    "align_max": 0,          # max alignment iterations
    "confirm_pending": False, # True when waiting for grip confirm
    "confirm_result": None,   # "y" or "n" — set by dashboard, read by agent
}

# ---- Chat Log (synced between CLI agent and dashboard) ----
# Each entry: {"role": "user"|"agent"|"system", "text": str, "ts": float, "id": int}
chat_log = []
chat_id_counter = 0
chat_lock = threading.Lock()
# Messages submitted from dashboard, waiting for agent to pick up
chat_pending = []

def find_port_by_serial(serial_number):
    """Find the serial port whose USB serial number matches. Returns device path or None."""
    try:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            if port.serial_number == serial_number:
                print(f"{C.GREEN}[robot]{C.RESET} Found serial {serial_number} at {C.CYAN}{port.device}{C.RESET}")
                return port.device
    except Exception as e:
        print(f"{C.RED}[robot]{C.RESET} Serial lookup failed: {e}")
    return None

def autodetect_serial_port():
    """Return the first available ttyACM* or ttyUSB* port, or None."""
    import glob
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if candidates:
        print(f"{C.CYAN}[robot]{C.RESET} Auto-detected serial ports: {candidates}")
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
                print(f"{C.GREEN}[camera]{C.RESET} Found '{dev_name}' at index {idx}")
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
    print(f"{C.CYAN}[camera]{C.RESET} Auto-detected camera indices: {found}")
    return found

def get_motor_ids():
    """Extract integer motor IDs from the bus, handling various lerobot formats."""
    def flatten_ints(val):
        """Recursively extract all integers from a value."""
        if isinstance(val, int):
            return [val]
        if isinstance(val, (list, tuple)):
            result = []
            for x in val:
                result += flatten_ints(x)
            return result
        return []

    if hasattr(robot.bus, 'motor_ids'):
        ids = flatten_ints(list(robot.bus.motor_ids))
        if ids:
            return ids
    if hasattr(robot.bus, 'motors'):
        ids = []
        for v in robot.bus.motors.values():
            ids += flatten_ints(v)
        if ids:
            return ids
    return list(range(1, 7))  # SO-101 fallback: servos 1-6


def init_robot():
    global robot
    # Priority: serial number lookup → configured port → autodetect
    port = find_port_by_serial(ROBOT_SERIAL)
    if port is None:
        print(f"{C.YELLOW}[robot]{C.RESET} Serial {ROBOT_SERIAL} not found, falling back to {ROBOT_PORT}")
        port = ROBOT_PORT
    if not os.path.exists(port):
        print(f"{C.YELLOW}[robot]{C.RESET} {port} not found, auto-detecting...")
        port = autodetect_serial_port() or ROBOT_PORT
    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        config = SOFollowerRobotConfig(port=port)
        robot = SOFollower(config)
        robot.connect(calibrate=False)
        print(f"{C.GREEN}[robot]{C.RESET} Connected to SO-101 on {C.CYAN}{port}{C.RESET}")
        # Enable torque so servos respond to position commands
        try:
            robot.bus.enable_torque()
            torque_enabled = True
            print(f"{C.GREEN}[robot]{C.RESET} Torque enabled")
        except Exception as te:
            print(f"{C.RED}[robot]{C.RESET} Torque enable failed: {te} — arm may not move")
    except Exception as e:
        print(f"{C.RED}[robot]{C.RESET} Could not connect to arm: {e}")
        print(f"{C.YELLOW}[robot]{C.RESET} Running in camera-only mode")
        robot = None

def init_cameras():
    import cv2

    # Resolve indices by name first, fall back to configured index, then auto-detect
    top_idx = find_camera_index_by_name(CAMERA_TOP_NAME) if CAMERA_TOP_NAME else None
    if top_idx is None:
        top_idx = CAMERA_TOP_IDX
        print(f"{C.YELLOW}[camera]{C.RESET} Name lookup failed for top, using index {top_idx}")

    wrist_idx = find_camera_index_by_name(CAMERA_WRIST_NAME) if CAMERA_WRIST_NAME else None
    if wrist_idx is None:
        wrist_idx = CAMERA_WRIST_IDX
        print(f"{C.YELLOW}[camera]{C.RESET} Name lookup failed for wrist, using index {wrist_idx}")

    # Auto-detect if resolved indices still don't open
    needs_autodetect = []
    for name, idx in [("top", top_idx), ("wrist", wrist_idx)]:
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            needs_autodetect.append(name)

    if needs_autodetect:
        print(f"{C.YELLOW}[camera]{C.RESET} Could not open configured indices for {needs_autodetect}, auto-detecting...")
        available = autodetect_cameras()
        if len(available) >= 2:
            top_idx, wrist_idx = available[0], available[1]
            print(f"{C.CYAN}[camera]{C.RESET} Using auto-detected: top={top_idx}, wrist={wrist_idx}")
        elif len(available) == 1:
            top_idx = wrist_idx = available[0]
            print(f"{C.YELLOW}[camera]{C.RESET} Only one camera found (index {available[0]}), using for both")

    cam_names = {"top": CAMERA_TOP_NAME, "wrist": CAMERA_WRIST_NAME}
    for name, idx in [("top", top_idx), ("wrist", wrist_idx)]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, SNAPSHOT_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SNAPSHOT_HEIGHT)
            cameras[name] = cap
            camera_info[name] = {"name": cam_names[name], "index": idx}
            print(f"{C.GREEN}[camera]{C.RESET} {name} camera opened (index {idx})")
        else:
            print(f"{C.RED}[camera]{C.RESET} Could not open {name} camera (index {idx})")

def reopen_camera(name):
    """Try to reopen a camera by name. Returns True on success."""
    import cv2
    idx = camera_info.get(name, {}).get("index")
    if idx is None:
        idx = CAMERA_WRIST_IDX if name == "wrist" else CAMERA_TOP_IDX
    print(f"{C.YELLOW}[camera]{C.RESET} {name} reopening index {idx}...")
    old = cameras.get(name)
    if old:
        try:
            old.release()
        except Exception:
            pass
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        cameras[name] = cap
        print(f"{C.GREEN}[camera]{C.RESET} {name} reopened successfully")
        return True
    print(f"{C.RED}[camera]{C.RESET} {name} reopen failed")
    cameras.pop(name, None)
    return False

def capture_snapshot(name):
    """Capture a frame, resize, return base64 JPEG string."""
    import cv2
    lock = camera_locks.get(name)
    if lock is None:
        return None, f"Unknown camera '{name}'"
    with lock:
        cam = cameras.get(name)
        if cam is None:
            return None, f"Camera '{name}' not available"
        # Flush 1 stale buffer frame (reduced from 4 to cut latency)
        cam.grab()
        ret, frame = cam.read()
        if not ret:
            if reopen_camera(name):
                cam = cameras.get(name)
                ret, frame = cam.read() if cam else (False, None)
            if not ret:
                return None, f"Failed to read from {name} camera"
    w = SNAPSHOT_TOP_WIDTH if name == "top" else SNAPSHOT_WIDTH
    h = SNAPSHOT_TOP_HEIGHT if name == "top" else SNAPSHOT_HEIGHT
    frame = cv2.resize(frame, (w, h))
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
    "home":    {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90, "gripper": 0},
    "ready":   {"shoulder_pan": 0, "shoulder_lift": 40, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90, "gripper": 0},
    "default": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0},
    "rest":    {"shoulder_pan": -1.10, "shoulder_lift": -102.24, "elbow_flex": 96.57, "wrist_flex": 76.35, "wrist_roll": -86.02, "gripper": 1.20},
    "drop":    {"shoulder_pan": -47.08, "shoulder_lift": 10.46, "elbow_flex": -12.44, "wrist_flex": 86.20, "wrist_roll": -94.64, "gripper": 0},
}

# ---- Routes ----

@app.route("/status")
def status():
    joints = get_joint_positions()
    # Leader arm joints (if connected)
    leader_joints = None
    if leader is not None:
        try:
            action = leader.get_action()
            leader_joints = {k: float(v) for k, v in action.items()}
        except Exception:
            leader_joints = {"error": "read failed"}
    # Robot port info
    robot_port_info = None
    if robot is not None and hasattr(robot, 'bus') and hasattr(robot.bus, 'port'):
        robot_port_info = robot.bus.port
    leader_port_info = None
    if leader is not None and hasattr(leader, 'bus') and hasattr(leader.bus, 'port'):
        leader_port_info = leader.bus.port
    elif leader is not None and hasattr(leader, 'port'):
        leader_port_info = leader.port
    return jsonify({
        "ok": True,
        "robot_connected": robot is not None,
        "leader_connected": leader is not None,
        "robot_port": robot_port_info,
        "leader_port": leader_port_info,
        "cameras": list(cameras.keys()),
        "camera_info": camera_info,
        "joints": joints,
        "leader_joints": leader_joints,
        "torque_enabled": torque_enabled,
        "teleop_active": teleop_active,
        "teleop_fps": TELEOP_FPS,
        "floor_drop": floor_drop,
        "agent": agent_state,
        "timestamp": time.time()
    })

def mjpeg_generator(name):
    import cv2
    fail_count = 0
    while True:
        lock = camera_locks.get(name)
        if lock is None:
            break
        with lock:
            cam = cameras.get(name)
            if cam is None:
                break
            ret, frame = cam.read()
        if not ret:
            fail_count += 1
            if fail_count >= 5:
                print(f"{C.YELLOW}[stream]{C.RESET} {name}: {fail_count} consecutive failures, reopening...")
                with lock:
                    reopen_camera(name)
                fail_count = 0
            time.sleep(0.2)
            continue
        fail_count = 0
        sw = SNAPSHOT_TOP_WIDTH if name == "top" else SNAPSHOT_WIDTH
        sh = SNAPSHOT_TOP_HEIGHT if name == "top" else SNAPSHOT_HEIGHT
        frame = cv2.resize(frame, (sw, sh))
        # Draw crosshair + degree scale on wrist stream
        if name == "wrist":
            h, w = frame.shape[:2]
            cx, cy = w // 2 + CROSSHAIR_OFFSET_X, h // 2 + CROSSHAIR_OFFSET_Y
            # Crosshair dot
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy),  7, (0, 0, 0),       -1)
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 0, 0), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 0, 0), 2)
            # Degree scale ruler (10° ≈ 80px, crosshair is 40px end-to-end)
            px_per_deg = 8
            ruler_y = h - 14
            ruler_cx = w // 2
            # Main ruler line
            ruler_half = px_per_deg * 10  # 10° each side
            cv2.line(frame, (ruler_cx - ruler_half, ruler_y), (ruler_cx + ruler_half, ruler_y), (200, 100, 255), 1)
            # Tick marks: 1° small, 5° medium, 10° large
            for deg in range(-10, 11):
                x = ruler_cx + deg * px_per_deg
                if deg % 10 == 0:
                    tick_h, thickness = 8, 2
                elif deg % 5 == 0:
                    tick_h, thickness = 5, 1
                else:
                    tick_h, thickness = 3, 1
                cv2.line(frame, (x, ruler_y - tick_h), (x, ruler_y), (200, 100, 255), thickness)
            # Labels
            cv2.putText(frame, "10", (ruler_cx - ruler_half - 4, ruler_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
            cv2.putText(frame, "5", (ruler_cx - px_per_deg * 5 - 3, ruler_y - 6), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
            cv2.putText(frame, "0", (ruler_cx - 3, ruler_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
            cv2.putText(frame, "5", (ruler_cx + px_per_deg * 5 - 3, ruler_y - 6), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
            cv2.putText(frame, "10", (ruler_cx + ruler_half - 4, ruler_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)  # ~20 fps target (cameras deliver ~12-15fps max)

@app.route("/stream/<name>")
def stream(name):
    if name not in ["top", "wrist"]:
        return jsonify({"error": "Unknown camera. Use 'top' or 'wrist'"}), 400
    from flask import Response, stream_with_context
    return Response(stream_with_context(mjpeg_generator(name)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream")
def stream_index():
    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:-apple-system,system-ui,sans-serif;padding:16px}
h1{font-size:20px;font-weight:600;margin-bottom:12px;color:#fff}
.layout{display:flex;gap:16px;flex-wrap:wrap}
.left-col{flex:1;min-width:0;display:flex;flex-direction:column;gap:12px}
.cameras{display:flex;gap:12px}
.cam-box{flex:1;min-width:0}
.cam-box p{text-align:center;font-size:14px;color:#888;margin-bottom:4px}
.cam-box img{width:100%;border-radius:6px;background:#000}
.panel{width:300px;flex-shrink:0;display:flex;flex-direction:column;gap:12px}
.chat-card{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px;flex:1;display:flex;flex-direction:column;min-height:200px}
.chat-card h2{font-size:13px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.chat-log{flex:1;overflow-y:auto;max-height:300px;display:flex;flex-direction:column;gap:6px;padding:4px 0;
          scrollbar-width:thin;scrollbar-color:#333 transparent}
.chat-log::-webkit-scrollbar{width:4px}
.chat-log::-webkit-scrollbar-thumb{background:#444;border-radius:2px}
.chat-msg{padding:6px 10px;border-radius:8px;font-size:13px;line-height:1.4;max-width:90%;word-wrap:break-word;white-space:pre-wrap}
.chat-msg.user{background:#1565c0;color:#e3f2fd;align-self:flex-end;border-bottom-right-radius:2px}
.chat-msg.agent{background:#2a2a2a;color:#e0e0e0;align-self:flex-start;border-bottom-left-radius:2px}
.chat-msg.system{background:#1a1a2a;color:#888;align-self:center;font-size:12px;font-style:italic}
.chat-msg .msg-source{font-size:10px;opacity:0.5;margin-top:2px}
.chat-input-row{display:flex;gap:6px;margin-top:8px}
.chat-input{flex:1;padding:8px 12px;border:1px solid #444;border-radius:6px;background:#222;color:#eee;
            font-size:13px;font-family:inherit;outline:none;transition:border-color .15s}
.chat-input:focus{border-color:#42a5f5}
.chat-send{padding:8px 16px;border:1px solid #388e3c;border-radius:6px;background:#222;color:#66bb6a;
           font-size:13px;cursor:pointer;font-weight:600;transition:background .15s}
.chat-send:hover{background:#1b5e20}
.chat-send:active{background:#2e7d32}
.card{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px}
.card h2{font-size:13px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.joint-row{display:flex;align-items:center;gap:6px;font-size:13px;padding:5px 0;border-bottom:1px solid #222;font-family:'SF Mono',monospace}
.joint-row:last-child{border-bottom:none}
.joint-name{color:#999;width:44px;flex-shrink:0;font-weight:500}
.joint-deg{color:#4fc3f7;width:52px;text-align:right;flex-shrink:0;font-size:12px}
.joint-slider{flex:1;-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#333;outline:none;cursor:pointer;min-width:60px}
.joint-slider::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#4fc3f7;cursor:pointer;border:2px solid #222}
.joint-slider::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#4fc3f7;cursor:pointer;border:2px solid #222}
.joint-slider:hover::-webkit-slider-thumb{background:#81d4fa;box-shadow:0 0 6px rgba(79,195,247,0.4)}
.joint-input{width:54px;padding:3px 4px;border:1px solid #444;border-radius:4px;background:#222;color:#4fc3f7;
             font-size:12px;font-family:'SF Mono',monospace;text-align:right;outline:none;transition:border-color .15s}
.joint-input:focus{border-color:#42a5f5}
.torque-status{font-size:13px;margin-top:6px;padding:4px 8px;border-radius:4px;text-align:center;font-weight:600}
.torque-on{background:#1b5e20;color:#66bb6a}
.torque-off{background:#4e342e;color:#ff8a65}
.teleop-on{background:#1a237e;color:#7986cb}
.teleop-off{background:#212121;color:#757575}
.btn-row{display:flex;gap:6px;flex-wrap:wrap}
.btn{flex:1;padding:8px 4px;border:1px solid #444;border-radius:6px;background:#222;color:#eee;
     font-size:13px;cursor:pointer;text-align:center;min-width:60px;transition:background .15s}
.btn:hover{background:#333}.btn:active{background:#444}
.btn-green{border-color:#388e3c;color:#66bb6a}.btn-green:hover{background:#1b5e20}
.btn-red{border-color:#c62828;color:#ef5350}.btn-red:hover{background:#4e1010}
.btn-blue{border-color:#1565c0;color:#42a5f5}.btn-blue:hover{background:#0d2137}
.btn-orange{border-color:#e65100;color:#ffb74d}.btn-orange:hover{background:#3e2723}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.status-dot.ok{background:#66bb6a}.status-dot.err{background:#ef5350}
#conn{font-size:12px;color:#888;margin-bottom:8px}
.activity-card{border-color:#444}
.phase-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.phase-idle{background:#212121;color:#757575}
.phase-calibrating{background:#4a148c;color:#ce93d8}
.phase-prescan{background:#006064;color:#4dd0e1}
.phase-homing{background:#0d47a1;color:#64b5f6}
.phase-aligning{background:#e65100;color:#ffb74d}
.phase-waiting_confirm{background:#b71c1c;color:#ef9a9a;animation:pulse 1.5s infinite}
.phase-lowering{background:#1a237e;color:#7986cb}
.phase-gripping{background:#880e4f;color:#f48fb1}
.phase-lifting{background:#0d47a1;color:#64b5f6}
.phase-dropping{background:#33691e;color:#aed581}
.phase-done{background:#1b5e20;color:#66bb6a}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.activity-detail{font-size:13px;color:#bbb;margin-top:6px;min-height:18px}
.progress-row{display:flex;align-items:center;gap:8px;margin-top:8px}
.progress-bar{flex:1;height:6px;background:#333;border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:#ffb74d;border-radius:3px;transition:width .3s}
.progress-text{font-size:12px;color:#888;min-width:40px}
.confirm-row{display:flex;gap:8px;margin-top:10px}
.confirm-row .btn{padding:10px;font-size:14px;font-weight:600}
#confirm-section{display:none}
.adv-toggle{width:100%;padding:8px;border:1px solid #333;border-radius:8px;background:#1a1a1a;color:#888;
            font-size:12px;cursor:pointer;text-align:left;margin-top:4px;transition:all .15s}
.adv-toggle:hover{color:#eee;border-color:#555}
.adv-panel{display:none;background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px;margin-top:4px}
.adv-panel.show{display:block}
.adv-section{margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #222}
.adv-section:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none}
.adv-section h3{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.adv-row{display:flex;justify-content:space-between;align-items:center;font-size:12px;padding:2px 0}
.adv-label{color:#888}.adv-val{color:#4fc3f7;font-family:'SF Mono',monospace}
.adv-val.ok{color:#66bb6a}.adv-val.warn{color:#ffb74d}.adv-val.err{color:#ef5350}.adv-val.off{color:#757575}
.conn-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;text-transform:uppercase}
.conn-badge.connected{background:#1b5e20;color:#66bb6a}
.conn-badge.partial{background:#e65100;color:#ffb74d}
.conn-badge.disconnected{background:#4e342e;color:#ff8a65}
.conn-badge.connecting{background:#0d47a1;color:#64b5f6;animation:pulse 1.5s infinite}
.adv-input{width:60px;padding:2px 4px;border:1px solid #444;border-radius:4px;background:#222;color:#4fc3f7;
           font-size:11px;font-family:'SF Mono',monospace;text-align:right;outline:none}
.adv-input:focus{border-color:#42a5f5}
.adv-btn{padding:3px 8px;border:1px solid #444;border-radius:4px;background:#222;color:#eee;font-size:11px;cursor:pointer}
.adv-btn:hover{background:#333}
.adv-log{max-height:80px;overflow-y:auto;font-size:11px;color:#888;font-family:'SF Mono',monospace;
         background:#151515;border-radius:4px;padding:6px;scrollbar-width:thin;scrollbar-color:#333 transparent}
</style>
</head><body>
<h1>SO-101 Dashboard</h1>
<div id="conn"><span class="status-dot" id="dot"></span><span id="conn-text">Connecting...</span></div>
<div class="layout">
  <div class="left-col">
    <div class="cameras">
      <div class="cam-box"><p>Top</p><img src="/stream/top"></div>
      <div class="cam-box"><p>Wrist</p><img src="/stream/wrist"></div>
    </div>
    <div class="card" style="padding:10px">
      <h2>Timeline <span style="font-size:10px;color:#555;text-transform:none;letter-spacing:0">(last 10 min)</span></h2>
      <div style="display:flex;align-items:center;gap:8px">
        <button class="btn" style="min-width:32px;padding:4px;font-size:14px" id="tl-play" onclick="tlToggle()">&#9654;</button>
        <input type="range" id="tl-slider" min="0" max="0" value="0" step="1"
               style="flex:1;-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#333;outline:none;cursor:pointer"
               oninput="tlSeek(this.value)">
        <span id="tl-time" style="font-size:11px;color:#888;font-family:'SF Mono',monospace;min-width:48px">--:--</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;margin-top:6px">
        <span style="font-size:11px;color:#666">Speed:</span>
        <select id="tl-speed" style="background:#222;border:1px solid #444;border-radius:4px;color:#eee;font-size:11px;padding:2px 4px">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
          <option value="10">10x</option>
        </select>
        <span style="font-size:11px;color:#666;margin-left:auto" id="tl-info">0 samples</span>
        <button class="btn" style="min-width:50px;padding:3px 6px;font-size:11px" onclick="tlGoTo()">Go To</button>
      </div>
    </div>
    <div class="chat-card">
      <h2>Chat</h2>
      <div class="chat-log" id="chat-log"></div>
      <div class="chat-input-row">
        <input class="chat-input" id="chat-input" type="text" placeholder="Type a message or command..." autocomplete="off"
               onkeydown="if(event.key==='Enter')sendChat()">
        <button class="chat-send" onclick="sendChat()">Send</button>
      </div>
    </div>
  </div>
  <div class="panel">
    <div class="card activity-card">
      <h2>Agent Activity</h2>
      <div><span class="phase-badge phase-idle" id="phase-badge">IDLE</span></div>
      <div class="activity-detail" id="activity-detail"></div>
      <div class="progress-row" id="progress-row" style="display:none">
        <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
        <span class="progress-text" id="progress-text">0/0</span>
        <button class="btn btn-orange" style="flex:0;min-width:50px;padding:4px 10px;font-size:12px" onclick="skipAlign()">Done</button>
      </div>
      <div id="confirm-section" class="confirm-row">
        <button class="btn btn-green" onclick="confirmGrip('y')">Confirm Grip</button>
        <button class="btn btn-red" onclick="confirmGrip('n')">Cancel</button>
      </div>
    </div>
    <div class="card">
      <h2>Joint Control</h2>
      <div id="joints"></div>
      <div id="torque-bar" class="torque-status torque-off">TORQUE OFF</div>
    </div>
    <div class="card">
      <h2>Torque</h2>
      <div class="btn-row">
        <button class="btn btn-green" onclick="torque(true)">ON</button>
        <button class="btn btn-red" onclick="torque(false)">OFF</button>
      </div>
    </div>
    <div class="card">
      <h2>Gripper</h2>
      <div class="btn-row">
        <button class="btn btn-green" onclick="grip(100)">Open</button>
        <button class="btn btn-red" onclick="grip(0)">Close</button>
      </div>
    </div>
    <div class="card">
      <h2>Presets</h2>
      <div class="btn-row">
        <button class="btn btn-blue" onclick="preset('home')">Home</button>
        <button class="btn btn-blue" onclick="preset('ready')">Ready</button>
        <button class="btn btn-blue" onclick="preset('default')">Default</button>
        <button class="btn btn-blue" onclick="preset('rest')">Rest</button>
      </div>
      <div class="btn-row" style="margin-top:6px">
        <button class="btn btn-orange" onclick="preset('drop')">Drop</button>
      </div>
    </div>
    <div class="card">
      <h2>Direction Control</h2>
      <div style="display:flex;flex-direction:column;align-items:center;gap:4px">
        <button class="btn btn-blue" style="width:80px" onclick="moveDir('forward')">Fwd</button>
        <div style="display:flex;gap:4px">
          <button class="btn btn-blue" style="width:80px" onclick="moveDir('left')">Left</button>
          <button class="btn btn-blue" style="width:80px" onclick="moveDir('right')">Right</button>
        </div>
        <button class="btn btn-blue" style="width:80px" onclick="moveDir('backward')">Bwd</button>
      </div>
      <div style="display:flex;align-items:center;gap:6px;margin-top:8px;padding-top:6px;border-top:1px solid #2a2a2a">
        <span style="font-size:11px;color:#666;flex:1">Step:</span>
        <select id="dir-step" style="background:#222;border:1px solid #444;border-radius:4px;color:#eee;font-size:11px;padding:2px 4px">
          <option value="1">1&deg;</option>
          <option value="3">3&deg;</option>
          <option value="5" selected>5&deg;</option>
          <option value="10">10&deg;</option>
          <option value="15">15&deg;</option>
          <option value="20">20&deg;</option>
        </select>
      </div>
    </div>
    <div class="card">
      <h2>Teleop</h2>
      <div id="teleop-bar" class="torque-status teleop-off">TELEOP OFF</div>
      <div class="btn-row" style="margin-top:8px">
        <button class="btn btn-green" onclick="teleopCtl('start')">Start</button>
        <button class="btn btn-red" onclick="teleopCtl('stop')">Stop</button>
      </div>
    </div>
    <button class="adv-toggle" onclick="toggleAdv()"><span id="adv-arrow">&#9654;</span> Advanced</button>
    <div class="adv-panel" id="adv-panel">
      <div class="adv-section">
        <h3>Connections</h3>
        <div class="adv-row"><span class="adv-label">Follower Arm</span><span class="adv-val" id="adv-follower">--</span></div>
        <div class="adv-row"><span class="adv-label">Follower Port</span><span class="adv-val off" id="adv-fport">--</span></div>
        <div class="adv-row"><span class="adv-label">Leader Arm</span><span class="adv-val" id="adv-leader">--</span></div>
        <div class="adv-row"><span class="adv-label">Leader Port</span><span class="adv-val off" id="adv-lport">--</span></div>
        <div class="adv-row"><span class="adv-label">Top Camera</span><span class="adv-val" id="adv-topcam">--</span></div>
        <div class="adv-row"><span class="adv-label">Wrist Camera</span><span class="adv-val" id="adv-wristcam">--</span></div>
      </div>
      <div class="adv-section">
        <h3>Rates</h3>
        <div class="adv-row"><span class="adv-label">Teleop FPS</span><span class="adv-val" id="adv-teleop-fps">--</span></div>
        <div class="adv-row"><span class="adv-label">Camera FPS</span><span class="adv-val" id="adv-cam-fps">--</span></div>
      </div>
      <div class="adv-section">
        <h3>Calibration</h3>
        <div class="adv-row">
          <span class="adv-label">Floor Drop</span>
          <span style="display:flex;gap:4px;align-items:center">
            <input class="adv-input" id="adv-calib-val" type="number" step="0.1" placeholder="--">
            <span class="adv-label">&deg;</span>
            <button class="adv-btn" onclick="setCalib()">Set</button>
          </span>
        </div>
        <div class="btn-row" style="margin-top:6px">
          <button class="btn btn-blue" style="font-size:11px;padding:4px" onclick="sendChat2('/calib')">Calibrate</button>
          <button class="btn btn-red" style="font-size:11px;padding:4px" onclick="clearCalib()">Clear</button>
        </div>
      </div>
      <div class="adv-section">
        <h3>Agent Activity Log</h3>
        <div class="adv-log" id="adv-log"></div>
      </div>
      <div class="adv-section" id="adv-leader-joints-section" style="display:none">
        <h3>Leader Joints</h3>
        <div id="adv-leader-joints"></div>
      </div>
    </div>
  </div>
</div>
<script>
const JOINT_ORDER=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
const SHORT={shoulder_pan:'Pan',shoulder_lift:'Lift',elbow_flex:'Elbow',wrist_flex:'Flex',wrist_roll:'Roll',gripper:'Grip'};
const PHASE_LABELS={idle:'Idle',calibrating:'Calibrating',homing:'Homing',prescan:'Scanning',aligning:'Aligning',
  waiting_confirm:'Waiting for Confirm',lowering:'Lowering',gripping:'Gripping',lifting:'Lifting',
  dropping:'Dropping',done:'Done'};
const JOINT_RANGES={shoulder_pan:[-150,150],shoulder_lift:[-110,110],elbow_flex:[-110,110],
  wrist_flex:[-110,110],wrist_roll:[-150,150],gripper:[0,100]};
function post(url,body){fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})}
function moveDir(dir){let deg=parseFloat(document.getElementById('dir-step').value)||5;post('/move_direction',{direction:dir,degrees:deg})}
function torque(on){post('/enable',{enabled:on})}
function grip(v){post('/move',{gripper:v})}
function preset(p){post('/move_preset',{pose:p})}
function teleopCtl(a){post('/teleop',{action:a})}
function confirmGrip(v){post('/confirm_grip',{confirm:v})}
function skipAlign(){post('/confirm_grip',{confirm:'done'})}
function onSlider(joint){
  let slider=document.getElementById('sld-'+joint);
  let inp=document.getElementById('inp-'+joint);
  let deg=document.getElementById('deg-'+joint);
  if(!slider)return;
  let v=parseFloat(slider.value);
  if(inp)inp.value=v.toFixed(1);
  if(deg)deg.textContent=v.toFixed(1)+'\u00b0';
  let cmd={};cmd[joint]=v;
  post('/move',cmd);
}
function onInput(joint){
  let inp=document.getElementById('inp-'+joint);
  let slider=document.getElementById('sld-'+joint);
  let deg=document.getElementById('deg-'+joint);
  if(!inp)return;
  let v=parseFloat(inp.value);
  if(isNaN(v))return;
  if(slider)slider.value=v;
  if(deg)deg.textContent=v.toFixed(1)+'\u00b0';
  let cmd={};cmd[joint]=v;
  post('/move',cmd);
}
function poll(){
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('dot').className='status-dot ok';
    document.getElementById('conn-text').textContent='Connected — arm '+(d.robot_connected?'online':'offline');
    if(d.joints){
      let container=document.getElementById('joints');
      // Build rows only on first render
      if(!container.dataset.init){
        let h='';
        for(let j of JOINT_ORDER){
          let r=JOINT_RANGES[j]||[-150,150];
          h+='<div class="joint-row">';
          h+='<span class="joint-name">'+SHORT[j]+'</span>';
          h+='<span class="joint-deg" id="deg-'+j+'">0.0\u00b0</span>';
          h+='<input class="joint-slider" id="sld-'+j+'" type="range" min="'+r[0]+'" max="'+r[1]+'" step="0.5" value="0" oninput="onSlider(\\''+j+'\\')">';
          h+='<input class="joint-input" id="inp-'+j+'" type="number" step="any" value="0" onkeydown="if(event.key===\\'Enter\\')onInput(\\''+j+'\\')" onchange="onInput(\\''+j+'\\')">';
          h+='</div>';
        }
        container.innerHTML=h;
        container.dataset.init='1';
      }
      // Update values only when neither input nor slider is being interacted with
      for(let j of JOINT_ORDER){
        let inp=document.getElementById('inp-'+j);
        let sld=document.getElementById('sld-'+j);
        let deg=document.getElementById('deg-'+j);
        let k=Object.keys(d.joints).find(x=>x.replace('.pos','')==j);
        let v=k?d.joints[k]:0;
        if(typeof v!=='number')continue;
        if(deg)deg.textContent=v.toFixed(1)+'\u00b0';
        if(sld&&document.activeElement!==sld)sld.value=v;
        if(inp&&document.activeElement!==inp)inp.value=v.toFixed(1);
      }
    }
    let tb=document.getElementById('torque-bar');
    if(d.torque_enabled){tb.className='torque-status torque-on';tb.textContent='TORQUE ON'}
    else{tb.className='torque-status torque-off';tb.textContent='TORQUE OFF'}
    let tp=document.getElementById('teleop-bar');
    if(d.teleop_active){tp.className='torque-status teleop-on';tp.textContent='TELEOP ACTIVE'}
    else{tp.className='torque-status teleop-off';tp.textContent='TELEOP OFF'}
    // Agent activity
    let a=d.agent||{};
    let phase=a.phase||'idle';
    let badge=document.getElementById('phase-badge');
    badge.className='phase-badge phase-'+phase;
    badge.textContent=PHASE_LABELS[phase]||phase;
    document.getElementById('activity-detail').textContent=a.detail||'';
    // Progress bar for alignment
    let prow=document.getElementById('progress-row');
    if(phase==='aligning'&&a.align_max>0){
      prow.style.display='flex';
      let pct=Math.round((a.align_iteration/a.align_max)*100);
      document.getElementById('progress-fill').style.width=pct+'%';
      document.getElementById('progress-text').textContent=a.align_iteration+'/'+a.align_max;
    }else{prow.style.display='none'}
    // Confirm buttons
    document.getElementById('confirm-section').style.display=a.confirm_pending?'flex':'none';
    // Advanced panel
    updateAdv(d);
  }).catch(()=>{
    document.getElementById('dot').className='status-dot err';
    document.getElementById('conn-text').textContent='Disconnected';
  });
}
// Advanced panel
let advOpen=false;
let activityLog=[];
function toggleAdv(){advOpen=!advOpen;document.getElementById('adv-panel').className=advOpen?'adv-panel show':'adv-panel';
  document.getElementById('adv-arrow').innerHTML=advOpen?'&#9660;':'&#9654;';}
function setCalib(){let v=parseFloat(document.getElementById('adv-calib-val').value);
  if(!isNaN(v))post('/calibration',{floor_drop:v});}
function clearCalib(){post('/calibration',{floor_drop:null});document.getElementById('adv-calib-val').value='';}
function sendChat2(text){fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text})})}
function connBadge(connected,label){
  if(connected===true)return '<span class="conn-badge connected">'+label+'</span>';
  if(connected==='partial')return '<span class="conn-badge partial">'+label+'</span>';
  return '<span class="conn-badge disconnected">'+label+'</span>';
}
function updateAdv(d){
  if(!advOpen)return;
  // Connections
  document.getElementById('adv-follower').innerHTML=connBadge(d.robot_connected,d.robot_connected?'Connected':'Disconnected');
  document.getElementById('adv-fport').textContent=d.robot_port||'--';
  document.getElementById('adv-leader').innerHTML=connBadge(d.leader_connected,d.leader_connected?'Connected':'Disconnected');
  document.getElementById('adv-lport').textContent=d.leader_port||'--';
  let topOk=d.cameras&&d.cameras.indexOf('top')>=0;
  let wristOk=d.cameras&&d.cameras.indexOf('wrist')>=0;
  let topInfo=d.camera_info&&d.camera_info.top;
  let wristInfo=d.camera_info&&d.camera_info.wrist;
  document.getElementById('adv-topcam').innerHTML=connBadge(topOk,topOk?(topInfo?topInfo.name+' #'+topInfo.index:'Connected'):'Disconnected');
  document.getElementById('adv-wristcam').innerHTML=connBadge(wristOk,wristOk?(wristInfo?wristInfo.name+' #'+wristInfo.index:'Connected'):'Disconnected');
  // Rates
  document.getElementById('adv-teleop-fps').textContent=d.teleop_active?(d.teleop_fps||60)+' hz':'--';
  // Camera fps: estimate from history sample rate (joints update at camera rate)
  let hLen=d.timestamp?Math.round(1/0.25)+'hz (poll)':'--';
  document.getElementById('adv-cam-fps').textContent=d.cameras?d.cameras.length+' cams @ ~20fps':'--';
  // Activity log
  let a=d.agent||{};
  let detail=a.detail||'';
  if(detail&&(activityLog.length===0||activityLog[activityLog.length-1]!==detail)){
    activityLog.push(detail);
    if(activityLog.length>50)activityLog=activityLog.slice(-50);
    let el=document.getElementById('adv-log');
    el.textContent=activityLog.join('\\n');
    el.scrollTop=el.scrollHeight;
  }
  // Leader joints
  let lsec=document.getElementById('adv-leader-joints-section');
  if(d.leader_joints&&!d.leader_joints.error){
    lsec.style.display='block';
    let lc=document.getElementById('adv-leader-joints');
    let h='';
    for(let k in d.leader_joints){
      let name=k.replace('.pos','');
      let v=d.leader_joints[k];
      if(typeof v==='number')v=v.toFixed(1);
      h+='<div class="adv-row"><span class="adv-label">'+name+'</span><span class="adv-val">'+v+'&deg;</span></div>';
    }
    lc.innerHTML=h;
  }else{lsec.style.display='none';}
  // Calibration — read from status data
  let ci=document.getElementById('adv-calib-val');
  if(document.activeElement!==ci){
    if(d.floor_drop!==null&&d.floor_drop!==undefined)ci.value=d.floor_drop;
    else ci.value='';
  }
}
poll();setInterval(poll,250);

// ---- Chat ----
let lastChatId=0;
function sendChat(){
  let inp=document.getElementById('chat-input');
  let text=inp.value.trim();
  if(!text)return;
  inp.value='';
  fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text})});
}
function pollChat(){
  fetch('/chat?since='+lastChatId).then(r=>r.json()).then(d=>{
    let log=document.getElementById('chat-log');
    let wasAtBottom=log.scrollHeight-log.scrollTop-log.clientHeight<30;
    for(let m of d.messages){
      if(m.id<=lastChatId)continue;
      lastChatId=m.id;
      let div=document.createElement('div');
      div.className='chat-msg '+m.role;
      // Strip ANSI codes for display
      let clean=m.text.replace(/\\x1b\\[[0-9;]*m/g,'').replace(/\\033\\[[0-9;]*m/g,'');
      div.textContent=clean;
      if(m.source==='dashboard'&&m.role==='user'){
        let src=document.createElement('div');
        src.className='msg-source';src.textContent='from dashboard';
        div.appendChild(src);
      }
      log.appendChild(div);
    }
    if(wasAtBottom&&d.messages.length>0)log.scrollTop=log.scrollHeight;
  }).catch(()=>{});
}
pollChat();setInterval(pollChat,500);

// ---- Timeline ----
let tlData=[];let tlPlaying=false;let tlTimer=null;let tlPos=0;
function tlLoad(){
  fetch('/history').then(r=>r.json()).then(d=>{
    tlData=d.data||[];
    let slider=document.getElementById('tl-slider');
    slider.max=Math.max(0,tlData.length-1);
    if(!tlPlaying)slider.value=slider.max;
    document.getElementById('tl-info').textContent=tlData.length+' samples';
    if(!tlPlaying&&tlData.length>0)tlUpdateTime(tlData.length-1);
  }).catch(()=>{});
}
function tlUpdateTime(idx){
  if(idx<0||idx>=tlData.length)return;
  let t0=tlData[0].t;
  let elapsed=tlData[idx].t-t0;
  let min=Math.floor(elapsed/60);
  let sec=Math.floor(elapsed%60);
  document.getElementById('tl-time').textContent=min+':'+(sec<10?'0':'')+sec;
  // Update joint display to show historical values
  let joints=tlData[idx].joints;
  for(let j of JOINT_ORDER){
    let deg=document.getElementById('deg-'+j);
    let sld=document.getElementById('sld-'+j);
    let inp=document.getElementById('inp-'+j);
    let k=Object.keys(joints).find(x=>x.replace('.pos','')==j);
    let v=k?joints[k]:0;
    if(typeof v!=='number')continue;
    if(deg)deg.textContent=v.toFixed(1)+'\u00b0';
    if(sld&&document.activeElement!==sld)sld.value=v;
    if(inp&&document.activeElement!==inp)inp.value=v.toFixed(1);
  }
}
function tlSeek(val){
  tlPos=parseInt(val);
  tlUpdateTime(tlPos);
}
function tlToggle(){
  if(tlPlaying){tlStop();return;}
  if(tlData.length<2)return;
  tlPlaying=true;
  document.getElementById('tl-play').innerHTML='&#9646;&#9646;';
  if(tlPos>=tlData.length-1)tlPos=0;
  tlStep();
}
function tlStep(){
  if(!tlPlaying||tlPos>=tlData.length-1){tlStop();return;}
  let speed=parseFloat(document.getElementById('tl-speed').value)||1;
  tlPos++;
  document.getElementById('tl-slider').value=tlPos;
  tlUpdateTime(tlPos);
  // Calculate delay based on actual time diff and speed
  let dt=500; // default
  if(tlPos<tlData.length-1)dt=(tlData[tlPos+1].t-tlData[tlPos].t)*1000/speed;
  dt=Math.max(16,Math.min(2000,dt));
  tlTimer=setTimeout(tlStep,dt);
}
function tlStop(){
  tlPlaying=false;
  document.getElementById('tl-play').innerHTML='&#9654;';
  if(tlTimer){clearTimeout(tlTimer);tlTimer=null;}
}
function tlGoTo(){
  if(tlPos<0||tlPos>=tlData.length)return;
  let joints=tlData[tlPos].joints;
  post('/history/goto',{joints:joints});
}
tlLoad();setInterval(tlLoad,5000);
</script>
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
        "width": SNAPSHOT_TOP_WIDTH if name == "top" else SNAPSHOT_WIDTH,
        "height": SNAPSHOT_TOP_HEIGHT if name == "top" else SNAPSHOT_HEIGHT,
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

# ---- Forward/Backward Elbow Compensation (must match gemini_robot_agent.py) ----
ELBOW_FORWARD_RATIO  = 2.0
ELBOW_BACKWARD_RATIO = 0.67

@app.route("/move_direction", methods=["POST"])
def move_direction():
    """Move arm in a cardinal direction (forward/backward/left/right) by degrees.
    Forward/backward uses trig to keep the gripper level."""
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json or {}
    direction = data.get("direction", "").lower()
    degrees = float(data.get("degrees", 5))
    if direction not in ("forward", "backward", "left", "right"):
        return jsonify({"error": f"Unknown direction '{direction}'. Use forward/backward/left/right"}), 400
    # Read current positions (hardware space)
    joints = get_joint_positions()
    if not joints:
        return jsonify({"error": "Cannot read joint positions"}), 503
    def cur(joint):
        for k, v in joints.items():
            if k.replace(".pos", "") == joint:
                return float(v)
        return 0.0
    # Hardware: forward = shoulder increases, elbow decreases
    # Backward = shoulder decreases, elbow also decreases (extends to keep level)
    if direction == "forward":
        targets = {
            "shoulder_lift": cur("shoulder_lift") + degrees,
            "elbow_flex": cur("elbow_flex") - degrees * ELBOW_FORWARD_RATIO,
        }
    elif direction == "backward":
        targets = {
            "shoulder_lift": cur("shoulder_lift") - degrees,
            "elbow_flex": cur("elbow_flex") + degrees * ELBOW_BACKWARD_RATIO,
        }
    elif direction == "left":
        targets = {"shoulder_pan": cur("shoulder_pan") - degrees}
    elif direction == "right":
        targets = {"shoulder_pan": cur("shoulder_pan") + degrees}
    # Interpolate in 8 steps — elbow leads shoulder slightly to prevent floor dip
    start = {k: cur(k) for k in targets}
    steps = 8
    try:
        for s in range(1, steps + 1):
            t = s / steps
            action = {}
            for k in targets:
                if direction == "forward" and k == "elbow_flex":
                    # Elbow leads shoulder on forward
                    action[f"{k}.pos"] = start[k] + (targets[k] - start[k]) * min(1.0, t * 1.5)
                elif direction == "backward" and k == "shoulder_lift":
                    # Shoulder leads elbow on backward
                    action[f"{k}.pos"] = start[k] + (targets[k] - start[k]) * min(1.0, t * 1.5)
                else:
                    action[f"{k}.pos"] = start[k] + (targets[k] - start[k]) * t
            with robot_lock:
                robot.send_action(action)
            time.sleep(0.08)
        return jsonify({"ok": True, "direction": direction, "degrees": degrees, "sent": targets})
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
    global torque_enabled
    data = request.json or {}
    enabled = data.get("enabled", True)
    motor = data.get("motor")  # optional: single motor name
    try:
        with robot_lock:
            if enabled:
                robot.bus.enable_torque(motors=motor)
            else:
                robot.bus.disable_torque(motors=motor)
        if not motor:
            torque_enabled = enabled
        return jsonify({"ok": True, "torque_enabled": enabled, "motor": motor or "all"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Agent State ----

@app.route("/agent_state", methods=["POST"])
def update_agent_state():
    """Agent pushes its current activity state."""
    data = request.json or {}
    for key in ("phase", "detail", "align_iteration", "align_max", "confirm_pending"):
        if key in data:
            agent_state[key] = data[key]
    # Reset confirm_result when a new confirm is requested
    if data.get("confirm_pending"):
        agent_state["confirm_result"] = None
    return jsonify({"ok": True})

@app.route("/confirm_grip", methods=["POST"])
def confirm_grip():
    """Dashboard sends grip confirmation."""
    data = request.json or {}
    result = data.get("confirm", "y")
    agent_state["confirm_result"] = result
    agent_state["confirm_pending"] = False
    return jsonify({"ok": True, "confirm": result})

# ---- Teleop ----

def find_leader_port():
    """Find the leader arm port. Uses LEADER_PORT env, or finds a second ttyACM/ttyUSB device."""
    if LEADER_PORT:
        return LEADER_PORT
    # Auto-detect: find serial ports that aren't the follower
    try:
        import serial.tools.list_ports
        follower_port = robot.bus.port if robot and hasattr(robot, 'bus') else ROBOT_PORT
        for port in serial.tools.list_ports.comports():
            if port.device != follower_port and ("ttyACM" in port.device or "ttyUSB" in port.device):
                print(f"{C.GREEN}[teleop]{C.RESET} Auto-detected leader at {C.CYAN}{port.device}{C.RESET} (serial: {port.serial_number})")
                return port.device
    except Exception as e:
        print(f"{C.RED}[teleop]{C.RESET} Auto-detect failed: {e}")
    return None

def init_leader(port):
    """Initialize the leader arm on the given port."""
    global leader
    try:
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
        config = SOLeaderTeleopConfig(port=port)
        leader = SOLeader(config)
        leader.connect()
        print(f"{C.GREEN}[teleop]{C.RESET} Leader arm connected on {C.CYAN}{port}{C.RESET}")
        return True
    except Exception as e:
        print(f"{C.RED}[teleop]{C.RESET} Failed to connect leader arm: {e}")
        leader = None
        return False

def teleop_loop():
    """Background thread: read leader positions and send to follower at TELEOP_FPS."""
    global teleop_active
    interval = 1.0 / TELEOP_FPS
    print(f"{C.GREEN}[teleop]{C.RESET} Loop started at {TELEOP_FPS} fps")
    while teleop_active:
        try:
            action = leader.get_action()
            with robot_lock:
                robot.send_action(action)
        except Exception as e:
            print(f"{C.RED}[teleop]{C.RESET} Error: {e}")
            break
        time.sleep(interval)
    teleop_active = False
    print(f"{C.YELLOW}[teleop]{C.RESET} Loop stopped")

def start_teleop(port=None):
    """Start teleoperation. Returns (ok, message)."""
    global teleop_active, teleop_thread, torque_enabled
    if teleop_active:
        return False, "Teleop already running"
    if robot is None:
        return False, "Follower arm not connected"

    # Find and connect leader if needed
    if leader is None:
        leader_port = port or find_leader_port()
        if not leader_port:
            return False, "No leader arm found. Set LEADER_PORT env or pass port parameter."
        if not init_leader(leader_port):
            return False, f"Failed to connect leader on {leader_port}"

    # Enable torque on follower
    try:
        robot.bus.enable_torque()
        torque_enabled = True
    except Exception as e:
        return False, f"Failed to enable torque: {e}"

    teleop_active = True
    teleop_thread = threading.Thread(target=teleop_loop, daemon=True)
    teleop_thread.start()
    return True, "Teleop started"

def stop_teleop():
    """Stop teleoperation."""
    global teleop_active, leader
    if not teleop_active:
        return False, "Teleop not running"
    teleop_active = False
    if teleop_thread:
        teleop_thread.join(timeout=2)
    # Disconnect leader to free the port
    if leader is not None:
        try:
            leader.disconnect()
        except Exception:
            pass
        leader = None
    return True, "Teleop stopped"

@app.route("/teleop", methods=["POST"])
def teleop():
    data = request.json or {}
    action = data.get("action", "status")
    if action == "start":
        ok, msg = start_teleop(port=data.get("port"))
        return jsonify({"ok": ok, "message": msg, "teleop_active": teleop_active})
    elif action == "stop":
        ok, msg = stop_teleop()
        return jsonify({"ok": ok, "message": msg, "teleop_active": teleop_active})
    else:
        return jsonify({"teleop_active": teleop_active})

# ---- Calibration ----

# Floor drop value — can be set by agent or dashboard
floor_drop = None

@app.route("/calibration", methods=["GET"])
def get_calibration():
    return jsonify({"floor_drop": floor_drop})

@app.route("/calibration", methods=["POST"])
def set_calibration():
    global floor_drop
    data = request.json or {}
    if "floor_drop" in data:
        val = data["floor_drop"]
        floor_drop = float(val) if val is not None else None
        print(f"{C.CYAN}[calib]{C.RESET} Floor drop set to {floor_drop}°" if floor_drop else f"{C.YELLOW}[calib]{C.RESET} Floor drop cleared")
        return jsonify({"ok": True, "floor_drop": floor_drop})
    return jsonify({"error": "Provide 'floor_drop'"}), 400


# ---- History ----

@app.route("/history")
def get_history():
    """Get joint position history. ?last=N returns last N seconds (default all)."""
    last = float(request.args.get("last", 0))
    with history_lock:
        if last > 0:
            cutoff = time.time() - last
            data = [h for h in joint_history if h["t"] >= cutoff]
        else:
            data = list(joint_history)
    # Thin out if too many points for the dashboard (max ~600 points)
    if len(data) > 600:
        step = len(data) // 600
        data = data[::step]
    return jsonify({"samples": len(data), "data": data})

@app.route("/history/goto", methods=["POST"])
def history_goto():
    """Move arm to a specific position from history."""
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json or {}
    joints = data.get("joints")
    if not joints:
        return jsonify({"error": "Provide 'joints'"}), 400
    try:
        action = {k: float(v) for k, v in joints.items()}
        with robot_lock:
            robot.send_action(action)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Chat ----

@app.route("/chat", methods=["GET"])
def get_chat():
    """Get chat messages. Pass ?since=ID to get only new messages."""
    since = int(request.args.get("since", 0))
    with chat_lock:
        msgs = [m for m in chat_log if m["id"] > since]
    return jsonify({"messages": msgs})

@app.route("/chat", methods=["POST"])
def post_chat():
    """Dashboard sends a chat message. Queued for the agent to pick up."""
    global chat_id_counter
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400
    with chat_lock:
        chat_id_counter += 1
        msg = {"role": "user", "text": text, "ts": time.time(), "id": chat_id_counter, "source": "dashboard"}
        chat_log.append(msg)
        chat_pending.append(text)
    return jsonify({"ok": True, "id": msg["id"]})

@app.route("/chat/push", methods=["POST"])
def push_chat():
    """Agent pushes a message to the chat log (for sync with dashboard)."""
    global chat_id_counter
    data = request.json or {}
    text = data.get("text", "")
    role = data.get("role", "agent")
    if not text:
        return jsonify({"error": "Empty message"}), 400
    with chat_lock:
        chat_id_counter += 1
        msg = {"role": role, "text": text, "ts": time.time(), "id": chat_id_counter}
        chat_log.append(msg)
        # Keep chat log to last 200 messages
        if len(chat_log) > 200:
            chat_log[:] = chat_log[-200:]
    return jsonify({"ok": True, "id": msg["id"]})

@app.route("/chat/pending", methods=["GET"])
def get_pending():
    """Agent polls for messages submitted from the dashboard."""
    with chat_lock:
        msgs = list(chat_pending)
        chat_pending.clear()
    return jsonify({"messages": msgs})


# ---- Diagnostics ----

def run_diagnostics():
    """Run startup diagnostics and print colored pass/fail summary."""
    print(f"\n{C.BOLD}{'=' * 54}{C.RESET}")
    print(f"{C.BOLD}  SO-101 Robot Server — Startup Diagnostics{C.RESET}")
    print(f"{C.BOLD}{'=' * 54}{C.RESET}\n")

    errors = []

    # 1. Robot arm
    print(f"  {C.BLUE}[1/4]{C.RESET} Robot arm............", end=" ")
    if robot is not None:
        try:
            with robot_lock:
                obs = robot.get_observation()
            joint_count = len([k for k in obs if "pos" in k or "joint" in k])
            print(f"{C.GREEN}OK{C.RESET} ({joint_count} joints)")
        except Exception as e:
            print(f"{C.RED}FAIL{C.RESET}")
            errors.append(f"Robot arm connected but cannot read joints: {e}")
    else:
        print(f"{C.RED}FAIL{C.RESET} — not connected")
        errors.append("Robot arm not connected (camera-only mode)")

    # 2. Torque
    print(f"  {C.BLUE}[2/4]{C.RESET} Torque...............", end=" ")
    if robot is not None:
        try:
            robot.bus.enable_torque()
            print(f"{C.GREEN}OK{C.RESET} (enabled)")
        except Exception as e:
            print(f"{C.RED}FAIL{C.RESET}")
            errors.append(f"Torque enable failed: {e}")
    else:
        print(f"{C.YELLOW}SKIP{C.RESET} (no arm)")

    # 3. Top camera
    print(f"  {C.BLUE}[3/4]{C.RESET} Top camera...........", end=" ")
    if "top" in cameras:
        import cv2
        cam = cameras["top"]
        for _ in range(4):
            cam.grab()
        ret, frame = cam.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"{C.GREEN}OK{C.RESET} ({w}x{h}, index {camera_info.get('top', {}).get('index', '?')})")
        else:
            print(f"{C.RED}FAIL{C.RESET} — opened but cannot read frames")
            errors.append("Top camera opened but frame read failed")
    else:
        print(f"{C.RED}FAIL{C.RESET} — not available")
        errors.append("Top camera not available")

    # 4. Wrist camera
    print(f"  {C.BLUE}[4/4]{C.RESET} Wrist camera.........", end=" ")
    if "wrist" in cameras:
        import cv2
        cam = cameras["wrist"]
        for _ in range(4):
            cam.grab()
        ret, frame = cam.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"{C.GREEN}OK{C.RESET} ({w}x{h}, index {camera_info.get('wrist', {}).get('index', '?')})")
        else:
            print(f"{C.RED}FAIL{C.RESET} — opened but cannot read frames")
            errors.append("Wrist camera opened but frame read failed")
    else:
        print(f"{C.RED}FAIL{C.RESET} — not available")
        errors.append("Wrist camera not available")

    # Summary
    print()
    if not errors:
        print(f"  {C.GREEN}{C.BOLD}All systems working{C.RESET}")
    else:
        print(f"  {C.RED}{C.BOLD}{len(errors)} error(s) detected:{C.RESET}")
        for err in errors:
            print(f"    {C.RED}*{C.RESET} {err}")

    print(f"\n{C.BOLD}{'=' * 54}{C.RESET}")
    print(f"  Server:    {C.CYAN}http://0.0.0.0:{PORT}{C.RESET}")
    print(f"  Dashboard: {C.CYAN}http://localhost:{PORT}/stream{C.RESET}")
    print(f"{C.BOLD}{'=' * 54}{C.RESET}\n")


# ---- Main ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="Serial port for follower arm")
    parser.add_argument("--leader-port", default=None, help="Serial port for leader arm (teleop)")
    args = parser.parse_args()
    if args.port:
        ROBOT_PORT = args.port
    if args.leader_port:
        LEADER_PORT = args.leader_port

    print(f"\n{C.BLUE}[boot]{C.RESET} Starting SO-101 Robot Agent Server...")
    print(f"{C.DIM}[boot] Robot port: {ROBOT_PORT}{C.RESET}")
    init_robot()
    init_cameras()
    # Start background history recorder
    threading.Thread(target=history_recorder, daemon=True).start()
    print(f"{C.GREEN}[boot]{C.RESET} History recorder started (sampling every {HISTORY_INTERVAL}s, {HISTORY_MAX_SECS//60}min buffer)")
    run_diagnostics()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
