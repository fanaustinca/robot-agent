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

from constants import (
    SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT, SNAPSHOT_TOP_WIDTH, SNAPSHOT_TOP_HEIGHT,
    JPEG_QUALITY, CROSSHAIR_OFFSET_X, CROSSHAIR_OFFSET_Y, PX_PER_DEG,
    SLOW_MOVE_STEPS, SLOW_MOVE_DELAY, PRESETS as SHARED_PRESETS,
    JOINT_ORDER, JOINT_SHORT, JOINT_ALIASES,
    DEFAULT_PORT, TELEOP_FPS, RECORDINGS_DIR, RECORDING_FPS,
)

# ---- Config ----
PORT = DEFAULT_PORT

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

app = Flask(__name__)

# ---- Robot + Camera State ----
robot = None
leader = None
cameras = {}
camera_info = {}  # {"top": {"name": ..., "index": ...}, "wrist": {...}}
robot_lock = threading.Lock()
torque_enabled = False
teleop_active = False
teleop_thread = None

# ---- Agent Activity State (set by agent, read by dashboard) ----
agent_state = {
    "phase": "idle",         # idle, calibrating, homing, aligning, waiting_confirm, lowering, gripping, lifting, dropping, done
    "detail": "",            # human-readable detail text
    "align_iteration": 0,   # current alignment iteration
    "align_max": 0,          # max alignment iterations
    "confirm_pending": False, # True when waiting for grip confirm
    "confirm_result": None,   # "y" or "n" — set by dashboard, read by agent
}

# ---- Recording State ----
recording_active = False
recording_name = None
recording_frames = []
recording_thread = None

def find_port_by_serial(serial_number):
    """Find the serial port whose USB serial number matches. Returns device path or None."""
    try:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            if port.serial_number == serial_number:
                print(f"[robot] Found serial {serial_number} at {port.device}")
                return port.device
    except Exception as e:
        print(f"[robot] Serial lookup failed: {e}")
    return None

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
        print(f"[robot] Serial {ROBOT_SERIAL} not found, falling back to {ROBOT_PORT}")
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
            robot.bus.enable_torque()
            torque_enabled = True
            print(f"[robot] Torque enabled")
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

    cam_names = {"top": CAMERA_TOP_NAME, "wrist": CAMERA_WRIST_NAME}
    for name, idx in [("top", top_idx), ("wrist", wrist_idx)]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, SNAPSHOT_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SNAPSHOT_HEIGHT)
            cameras[name] = cap
            camera_info[name] = {"name": cam_names[name], "index": idx}
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

PRESETS = SHARED_PRESETS

# ---- Routes ----

@app.route("/status")
def status():
    joints = get_joint_positions()
    return jsonify({
        "ok": True,
        "robot_connected": robot is not None,
        "cameras": list(cameras.keys()),
        "camera_info": camera_info,
        "joints": joints,
        "torque_enabled": torque_enabled,
        "teleop_active": teleop_active,
        "recording_active": recording_active,
        "agent": agent_state,
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
            px_per_deg = PX_PER_DEG
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
    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 Dashboard</title>
<style>
:root{
  --bg:#0a0a0f;--surface:#12121a;--card:#181825;--border:#2a2a3a;--border-hi:#3a3a50;
  --text:#e0e0ee;--text-dim:#8888a0;--text-muted:#55556a;
  --accent:#7c6ff0;--accent-dim:#5a50b0;
  --green:#4ade80;--green-bg:rgba(74,222,128,0.1);--green-border:rgba(74,222,128,0.25);
  --red:#f87171;--red-bg:rgba(248,113,113,0.1);--red-border:rgba(248,113,113,0.25);
  --blue:#60a5fa;--blue-bg:rgba(96,165,250,0.1);--blue-border:rgba(96,165,250,0.25);
  --orange:#fbbf24;--orange-bg:rgba(251,191,36,0.1);--orange-border:rgba(251,191,36,0.25);
  --cyan:#22d3ee;--purple:#c084fc;
  --radius:10px;--radius-sm:6px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter',-apple-system,system-ui,'Segoe UI',sans-serif;
     padding:20px 24px;min-height:100vh}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Header */
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)}
.header-left{display:flex;align-items:center;gap:12px}
.logo{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--purple));
      display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;color:#fff}
h1{font-size:18px;font-weight:700;color:#fff;letter-spacing:-0.3px}
h1 span{color:var(--accent);font-weight:400}
.header-right{display:flex;align-items:center;gap:12px}
#conn{font-size:12px;color:var(--text-dim);display:flex;align-items:center;gap:6px}
.status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.status-dot.ok{background:var(--green);box-shadow:0 0 8px rgba(74,222,128,0.4)}
.status-dot.err{background:var(--red);box-shadow:0 0 8px rgba(248,113,113,0.4)}
.help-btn{padding:5px 12px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface);
           color:var(--text-dim);font-size:12px;cursor:pointer;transition:all .2s}
.help-btn:hover{border-color:var(--accent);color:var(--accent);background:rgba(124,111,240,0.08)}

/* Layout */
.layout{display:grid;grid-template-columns:1fr 320px;gap:16px}
@media(max-width:900px){.layout{grid-template-columns:1fr}.panel{order:-1}}

/* Cameras */
.cameras{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.cam-box{position:relative;border-radius:var(--radius);overflow:hidden;background:#000;border:1px solid var(--border)}
.cam-box img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}
.cam-label{position:absolute;top:8px;left:8px;padding:3px 10px;border-radius:4px;
            background:rgba(0,0,0,0.7);backdrop-filter:blur(4px);font-size:11px;font-weight:600;
            text-transform:uppercase;letter-spacing:0.5px;color:var(--text-dim)}

/* Panel */
.panel{display:flex;flex-direction:column;gap:10px;max-height:calc(100vh - 80px);overflow-y:auto;
       scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.panel::-webkit-scrollbar{width:4px}
.panel::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:14px}
.card h2{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1.2px;
         margin-bottom:10px;font-weight:600;display:flex;align-items:center;gap:6px}
.card h2 .icon{font-size:14px;opacity:0.6}

/* Activity card */
.activity-card{border-color:var(--border-hi);background:linear-gradient(135deg,var(--card),rgba(124,111,240,0.03))}
.phase-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;
             text-transform:uppercase;letter-spacing:0.8px}
.phase-idle{background:rgba(85,85,106,0.2);color:var(--text-muted)}
.phase-calibrating{background:rgba(192,132,252,0.15);color:var(--purple)}
.phase-prescan{background:rgba(34,211,238,0.12);color:var(--cyan)}
.phase-homing{background:var(--blue-bg);color:var(--blue)}
.phase-aligning{background:var(--orange-bg);color:var(--orange)}
.phase-waiting_confirm{background:var(--red-bg);color:var(--red);animation:pulse 1.5s infinite}
.phase-lowering{background:var(--blue-bg);color:var(--blue)}
.phase-gripping{background:rgba(236,72,153,0.12);color:#f472b6}
.phase-lifting{background:var(--blue-bg);color:var(--blue)}
.phase-dropping{background:var(--green-bg);color:var(--green)}
.phase-done{background:var(--green-bg);color:var(--green)}
.phase-recording{background:var(--red-bg);color:var(--red);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.activity-detail{font-size:12px;color:var(--text-dim);margin-top:8px;min-height:16px;line-height:1.4}
.progress-row{display:flex;align-items:center;gap:8px;margin-top:8px}
.progress-bar{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--orange),#f59e0b);border-radius:3px;transition:width .3s}
.progress-text{font-size:11px;color:var(--text-muted);min-width:36px;font-family:'JetBrains Mono',monospace}
.confirm-row{display:flex;gap:8px;margin-top:10px}
.confirm-row .btn{padding:10px;font-size:13px;font-weight:600}
#confirm-section{display:none}

/* Joint rows with jog buttons */
.joint-row{display:flex;align-items:center;font-size:12px;padding:3px 0;font-family:'JetBrains Mono','SF Mono',monospace;gap:4px}
.joint-name{color:var(--text-muted);width:52px;flex-shrink:0;font-weight:500}
.joint-val{color:var(--cyan);flex:1;text-align:right;font-weight:500;min-width:60px}
.jog-btn{width:24px;height:22px;border:1px solid var(--border);border-radius:4px;background:var(--surface);
         color:var(--text-dim);font-size:12px;cursor:pointer;display:flex;align-items:center;justify-content:center;
         transition:all .15s;padding:0;font-family:'JetBrains Mono',monospace;font-weight:600;flex-shrink:0}
.jog-btn:hover{border-color:var(--accent);color:var(--accent);background:rgba(124,111,240,0.1)}
.jog-btn:active{background:rgba(124,111,240,0.2);transform:scale(0.95)}
.jog-step{display:flex;align-items:center;gap:4px;margin-top:8px;padding-top:8px;border-top:1px solid var(--border)}
.jog-step label{font-size:11px;color:var(--text-muted);flex:1}
.jog-step select{background:var(--surface);border:1px solid var(--border);border-radius:4px;color:var(--text);
                  font-size:11px;padding:3px 6px;font-family:'JetBrains Mono',monospace}

/* Status bars */
.status-bar{font-size:12px;margin-top:8px;padding:5px 10px;border-radius:var(--radius-sm);text-align:center;font-weight:600;letter-spacing:0.3px}
.torque-on{background:var(--green-bg);color:var(--green);border:1px solid var(--green-border)}
.torque-off{background:rgba(255,255,255,0.03);color:var(--text-muted);border:1px solid var(--border)}
.teleop-on{background:var(--blue-bg);color:var(--blue);border:1px solid var(--blue-border)}
.teleop-off{background:rgba(255,255,255,0.03);color:var(--text-muted);border:1px solid var(--border)}
.recording-on{background:var(--red-bg);color:var(--red);border:1px solid var(--red-border);animation:pulse 1.5s infinite}

/* Buttons */
.btn-row{display:flex;gap:6px;flex-wrap:wrap}
.btn{flex:1;padding:7px 4px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface);
     color:var(--text);font-size:12px;cursor:pointer;text-align:center;min-width:55px;transition:all .15s;font-weight:500}
.btn:hover{background:rgba(255,255,255,0.06);border-color:var(--border-hi)}
.btn:active{transform:scale(0.97)}
.btn-green{border-color:var(--green-border);color:var(--green)}.btn-green:hover{background:var(--green-bg)}
.btn-red{border-color:var(--red-border);color:var(--red)}.btn-red:hover{background:var(--red-bg)}
.btn-blue{border-color:var(--blue-border);color:var(--blue)}.btn-blue:hover{background:var(--blue-bg)}
.btn-orange{border-color:var(--orange-border);color:var(--orange)}.btn-orange:hover{background:var(--orange-bg)}
.btn-purple{border-color:rgba(192,132,252,0.25);color:var(--purple)}.btn-purple:hover{background:rgba(192,132,252,0.1)}

/* Recordings card */
.rec-list{max-height:120px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.rec-item{display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);font-size:12px}
.rec-item:last-child{border-bottom:none}
.rec-name{color:var(--text);font-weight:500;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rec-meta{color:var(--text-muted);font-size:11px;margin-left:8px;flex-shrink:0;font-family:'JetBrains Mono',monospace}
.rec-play{width:28px;height:24px;border:1px solid var(--green-border);border-radius:4px;background:transparent;
           color:var(--green);cursor:pointer;font-size:11px;margin-left:6px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.rec-play:hover{background:var(--green-bg)}
.rec-del{width:28px;height:24px;border:1px solid var(--red-border);border-radius:4px;background:transparent;
         color:var(--red);cursor:pointer;font-size:11px;margin-left:4px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.rec-del:hover{background:var(--red-bg)}
.rec-empty{color:var(--text-muted);font-size:12px;font-style:italic;padding:8px 0}
#rec-name-input{background:var(--surface);border:1px solid var(--border);border-radius:4px;color:var(--text);
                font-size:12px;padding:5px 8px;width:100%;margin-bottom:8px;font-family:'Inter',sans-serif}
#rec-name-input::placeholder{color:var(--text-muted)}
#rec-name-input:focus{outline:none;border-color:var(--accent)}

/* Help panel */
.help-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.6);backdrop-filter:blur(4px);z-index:100;
              align-items:center;justify-content:center}
.help-overlay.show{display:flex}
.help-modal{background:var(--card);border:1px solid var(--border-hi);border-radius:14px;padding:24px;
            max-width:680px;width:90%;max-height:80vh;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.help-modal h2{font-size:16px;color:#fff;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between}
.help-close{background:none;border:1px solid var(--border);border-radius:6px;color:var(--text-dim);
            width:30px;height:30px;cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center}
.help-close:hover{border-color:var(--red);color:var(--red)}
.help-section{margin-bottom:16px}
.help-section h3{font-size:12px;color:var(--accent);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;font-weight:600}
.help-table{width:100%;font-size:12px;border-collapse:collapse}
.help-table td{padding:4px 8px;border-bottom:1px solid var(--border)}
.help-table td:first-child{color:var(--cyan);font-family:'JetBrains Mono',monospace;white-space:nowrap;width:40%}
.help-table td:last-child{color:var(--text-dim)}
.help-tip{font-size:12px;color:var(--text-dim);line-height:1.5;padding:8px 12px;background:var(--surface);border-radius:var(--radius-sm);border-left:3px solid var(--accent)}
</style>
</head><body>

<!-- Help Modal -->
<div class="help-overlay" id="help-overlay" onclick="if(event.target===this)toggleHelp()">
  <div class="help-modal">
    <h2>SO-101 Reference<button class="help-close" onclick="toggleHelp()">&times;</button></h2>

    <div class="help-section">
      <h3>Dashboard Controls</h3>
      <table class="help-table">
        <tr><td>Joint Jog +/-</td><td>Nudge individual joints by the selected step size</td></tr>
        <tr><td>Step Size Dropdown</td><td>Set jog increment (1, 5, 10, or 20 degrees)</td></tr>
        <tr><td>Torque ON/OFF</td><td>Enable or disable motor torque (all joints)</td></tr>
        <tr><td>Gripper Open/Close</td><td>Fully open (100) or close (0) the gripper</td></tr>
        <tr><td>Preset Buttons</td><td>Move to Home, Default, or Rest position</td></tr>
        <tr><td>Teleop Start/Stop</td><td>Begin/end leader arm teleoperation</td></tr>
        <tr><td>Record/Stop</td><td>Record joint positions during teleop for replay</td></tr>
        <tr><td>Replay Button</td><td>Replay a saved recording sequence</td></tr>
      </table>
    </div>

    <div class="help-section">
      <h3>Agent CLI Commands</h3>
      <table class="help-table">
        <tr><td>/help, /h</td><td>Show all commands</td></tr>
        <tr><td>/home</td><td>Move to home preset (slow interpolation)</td></tr>
        <tr><td>/default</td><td>Move all joints to 0</td></tr>
        <tr><td>/drop</td><td>Move to drop position</td></tr>
        <tr><td>/move &lt;joint&gt; &lt;deg&gt;</td><td>Move a single joint</td></tr>
        <tr><td>/pos [motor]</td><td>Show current joint positions</td></tr>
        <tr><td>/torque-on [motor]</td><td>Enable torque (alias: /t-on)</td></tr>
        <tr><td>/torque-off [motor]</td><td>Disable torque (alias: /t-off)</td></tr>
        <tr><td>/calib</td><td>Calibrate floor distance for pickup</td></tr>
        <tr><td>/teleop [start|stop]</td><td>Leader arm teleoperation</td></tr>
        <tr><td>/record [name]</td><td>Start recording joint trajectory</td></tr>
        <tr><td>/stop-record</td><td>Stop recording and save</td></tr>
        <tr><td>/replay &lt;name&gt;</td><td>Replay a saved recording</td></tr>
        <tr><td>/recordings</td><td>List saved recordings</td></tr>
        <tr><td>/doctor</td><td>Run full diagnostics</td></tr>
      </table>
    </div>

    <div class="help-section">
      <h3>Joint Aliases</h3>
      <table class="help-table">
        <tr><td>sp, pan</td><td>shoulder_pan</td></tr>
        <tr><td>sl, lift</td><td>shoulder_lift</td></tr>
        <tr><td>ef, elbow</td><td>elbow_flex</td></tr>
        <tr><td>wf, flex</td><td>wrist_flex</td></tr>
        <tr><td>wr, roll</td><td>wrist_roll</td></tr>
        <tr><td>g, grip</td><td>gripper</td></tr>
      </table>
    </div>

    <div class="help-section">
      <h3>Presets</h3>
      <table class="help-table">
        <tr><td>home</td><td>All joints 0, wrist_roll -90</td></tr>
        <tr><td>default</td><td>All joints 0</td></tr>
        <tr><td>rest</td><td>Folded position (arm tucked in)</td></tr>
      </table>
    </div>

    <div class="help-section">
      <h3>Tips</h3>
      <div class="help-tip">
        Type natural language to the Gemini agent for pickup tasks (e.g. "pick up the red block").
        The agent uses vision to align the gripper automatically. You can press <b>Done</b> on the
        dashboard to skip the alignment phase, or <b>Confirm/Cancel</b> to approve or reject a grip.
        <br><br>
        Use <b>Teach &amp; Replay</b> to record joint positions during teleop and replay them later.
        Start teleop, press Record, move the arm, then Stop to save.
      </div>
    </div>
  </div>
</div>

<div class="header">
  <div class="header-left">
    <div class="logo">S</div>
    <h1>SO-101 <span>Dashboard</span></h1>
  </div>
  <div class="header-right">
    <div id="conn"><span class="status-dot" id="dot"></span><span id="conn-text">Connecting...</span></div>
    <button class="help-btn" onclick="toggleHelp()">? Help</button>
  </div>
</div>

<div class="layout">
  <div>
    <div class="cameras">
      <div class="cam-box"><div class="cam-label">Top</div><img src="/stream/top"></div>
      <div class="cam-box"><div class="cam-label">Wrist</div><img src="/stream/wrist"></div>
    </div>
  </div>

  <div class="panel">
    <!-- Agent Activity -->
    <div class="card activity-card">
      <h2>Agent Activity</h2>
      <div><span class="phase-badge phase-idle" id="phase-badge">IDLE</span></div>
      <div class="activity-detail" id="activity-detail"></div>
      <div class="progress-row" id="progress-row" style="display:none">
        <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
        <span class="progress-text" id="progress-text">0/0</span>
        <button class="btn btn-orange" style="flex:0;min-width:44px;padding:4px 10px;font-size:11px" onclick="skipAlign()">Done</button>
      </div>
      <div id="confirm-section" class="confirm-row">
        <button class="btn btn-green" onclick="confirmGrip('y')">Confirm Grip</button>
        <button class="btn btn-red" onclick="confirmGrip('n')">Cancel</button>
      </div>
    </div>

    <!-- Joint Positions + Jog -->
    <div class="card">
      <h2>Joints</h2>
      <div id="joints"></div>
      <div class="jog-step">
        <label>Jog step:</label>
        <select id="jog-step" onchange="jogStep=parseInt(this.value)">
          <option value="1">1&deg;</option>
          <option value="5" selected>5&deg;</option>
          <option value="10">10&deg;</option>
          <option value="20">20&deg;</option>
        </select>
      </div>
      <div id="torque-bar" class="status-bar torque-off">TORQUE OFF</div>
    </div>

    <!-- Controls -->
    <div class="card">
      <h2>Controls</h2>
      <div class="btn-row" style="margin-bottom:6px">
        <button class="btn btn-green" onclick="torque(true)">Torque ON</button>
        <button class="btn btn-red" onclick="torque(false)">Torque OFF</button>
      </div>
      <div class="btn-row" style="margin-bottom:6px">
        <button class="btn btn-green" onclick="grip(100)">Grip Open</button>
        <button class="btn btn-red" onclick="grip(0)">Grip Close</button>
      </div>
      <div class="btn-row">
        <button class="btn btn-blue" onclick="preset('home')">Home</button>
        <button class="btn btn-blue" onclick="preset('default')">Default</button>
        <button class="btn btn-blue" onclick="preset('rest')">Rest</button>
      </div>
    </div>

    <!-- Teleop -->
    <div class="card">
      <h2>Teleop</h2>
      <div id="teleop-bar" class="status-bar teleop-off">TELEOP OFF</div>
      <div class="btn-row" style="margin-top:8px">
        <button class="btn btn-green" onclick="teleopCtl('start')">Start</button>
        <button class="btn btn-red" onclick="teleopCtl('stop')">Stop</button>
      </div>
    </div>

    <!-- Teach & Replay -->
    <div class="card">
      <h2>Teach &amp; Replay</h2>
      <input id="rec-name-input" type="text" placeholder="Recording name..." maxlength="40">
      <div class="btn-row" style="margin-bottom:8px">
        <button class="btn btn-red" id="rec-btn" onclick="toggleRecord()">Record</button>
        <button class="btn btn-purple" onclick="replayFromInput()">Replay</button>
      </div>
      <div id="rec-bar" class="status-bar" style="display:none"></div>
      <div class="rec-list" id="rec-list"><div class="rec-empty">No recordings yet</div></div>
    </div>
  </div>
</div>

<script>
const JOINT_ORDER=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
const SHORT={shoulder_pan:'Pan',shoulder_lift:'Lift',elbow_flex:'Elbow',wrist_flex:'Flex',wrist_roll:'Roll',gripper:'Grip'};
const PHASE_LABELS={idle:'Idle',calibrating:'Calibrating',homing:'Homing',prescan:'Scanning',aligning:'Aligning',
  waiting_confirm:'Waiting',lowering:'Lowering',gripping:'Gripping',lifting:'Lifting',
  dropping:'Dropping',done:'Done',recording:'Recording'};
let jogStep=5;
function post(url,body){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})}
function torque(on){post('/enable',{enabled:on})}
function grip(v){post('/move',{gripper:v})}
function preset(p){post('/move_preset',{pose:p})}
function teleopCtl(a){post('/teleop',{action:a})}
function confirmGrip(v){post('/confirm_grip',{confirm:v})}
function skipAlign(){post('/confirm_grip',{confirm:'done'})}
function jogJoint(joint,dir){post('/move_jog',{joint:joint,delta:dir*jogStep})}
function toggleHelp(){document.getElementById('help-overlay').classList.toggle('show')}

// Recording
let isRecording=false;
function toggleRecord(){
  if(!isRecording){
    let name=document.getElementById('rec-name-input').value.trim();
    if(!name){name='rec_'+Date.now()}
    post('/recording/start',{name:name}).then(r=>r.json()).then(d=>{
      if(d.ok){isRecording=true;updateRecBtn()}
    });
  }else{
    post('/recording/stop',{}).then(r=>r.json()).then(d=>{
      isRecording=false;updateRecBtn();loadRecordings();
    });
  }
}
function updateRecBtn(){
  let btn=document.getElementById('rec-btn');
  let bar=document.getElementById('rec-bar');
  if(isRecording){
    btn.textContent='Stop';btn.className='btn btn-orange';
    bar.style.display='block';bar.className='status-bar recording-on';bar.textContent='RECORDING';
  }else{
    btn.textContent='Record';btn.className='btn btn-red';
    bar.style.display='none';
  }
}
function replayFromInput(){
  let name=document.getElementById('rec-name-input').value.trim();
  if(name)replayRec(name);
}
function replayRec(name){post('/recording/replay',{name:name})}
function deleteRec(name){if(confirm('Delete "'+name+'"?'))post('/recording/delete',{name:name}).then(()=>loadRecordings())}
function loadRecordings(){
  fetch('/recordings').then(r=>r.json()).then(d=>{
    let el=document.getElementById('rec-list');
    if(!d.recordings||d.recordings.length===0){el.innerHTML='<div class="rec-empty">No recordings yet</div>';return}
    let h='';
    for(let r of d.recordings){
      let dur=r.duration?r.duration.toFixed(1)+'s':'';
      let frames=r.frames||0;
      h+='<div class="rec-item"><span class="rec-name">'+r.name+'</span><span class="rec-meta">'+frames+'f '+dur+'</span>';
      h+='<button class="rec-play" onclick="replayRec(\''+r.name+'\')" title="Replay">&#9654;</button>';
      h+='<button class="rec-del" onclick="deleteRec(\''+r.name+'\')" title="Delete">&times;</button></div>';
    }
    el.innerHTML=h;
  }).catch(()=>{});
}

function poll(){
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('dot').className='status-dot ok';
    document.getElementById('conn-text').textContent='Connected'+(d.robot_connected?' \u2022 arm online':' \u2022 arm offline');
    let h='';
    if(d.joints){
      for(let j of JOINT_ORDER){
        let k=Object.keys(d.joints).find(x=>x.replace('.pos','')==j);
        let v=k?d.joints[k]:'-';
        if(typeof v==='number')v=v.toFixed(1)+'\u00b0';
        h+='<div class="joint-row">';
        h+='<button class="jog-btn" onclick="jogJoint(\''+j+'\',-1)">&minus;</button>';
        h+='<span class="joint-name">'+SHORT[j]+'</span>';
        h+='<span class="joint-val">'+v+'</span>';
        h+='<button class="jog-btn" onclick="jogJoint(\''+j+'\',1)">+</button>';
        h+='</div>';
      }
    }
    document.getElementById('joints').innerHTML=h;
    let tb=document.getElementById('torque-bar');
    if(d.torque_enabled){tb.className='status-bar torque-on';tb.textContent='TORQUE ON'}
    else{tb.className='status-bar torque-off';tb.textContent='TORQUE OFF'}
    let tp=document.getElementById('teleop-bar');
    if(d.teleop_active){tp.className='status-bar teleop-on';tp.textContent='TELEOP ACTIVE'}
    else{tp.className='status-bar teleop-off';tp.textContent='TELEOP OFF'}
    // Recording state
    if(d.recording_active&&!isRecording){isRecording=true;updateRecBtn()}
    else if(!d.recording_active&&isRecording){isRecording=false;updateRecBtn()}
    // Agent activity
    let a=d.agent||{};
    let phase=a.phase||'idle';
    let badge=document.getElementById('phase-badge');
    badge.className='phase-badge phase-'+phase;
    badge.textContent=PHASE_LABELS[phase]||phase;
    document.getElementById('activity-detail').textContent=a.detail||'';
    let prow=document.getElementById('progress-row');
    if(phase==='aligning'&&a.align_max>0){
      prow.style.display='flex';
      let pct=Math.round((a.align_iteration/a.align_max)*100);
      document.getElementById('progress-fill').style.width=pct+'%';
      document.getElementById('progress-text').textContent=a.align_iteration+'/'+a.align_max;
    }else{prow.style.display='none'}
    document.getElementById('confirm-section').style.display=a.confirm_pending?'flex':'none';
  }).catch(()=>{
    document.getElementById('dot').className='status-dot err';
    document.getElementById('conn-text').textContent='Disconnected';
  });
}
poll();setInterval(poll,250);
loadRecordings();setInterval(loadRecordings,5000);
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
                print(f"[teleop] Auto-detected leader at {port.device} (serial: {port.serial_number})")
                return port.device
    except Exception as e:
        print(f"[teleop] Auto-detect failed: {e}")
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
        print(f"[teleop] Leader arm connected on {port}")
        return True
    except Exception as e:
        print(f"[teleop] Failed to connect leader arm: {e}")
        leader = None
        return False

def teleop_loop():
    """Background thread: read leader positions and send to follower at TELEOP_FPS."""
    global teleop_active
    interval = 1.0 / TELEOP_FPS
    print(f"[teleop] Loop started at {TELEOP_FPS} fps")
    while teleop_active:
        try:
            action = leader.get_action()
            with robot_lock:
                robot.send_action(action)
        except Exception as e:
            print(f"[teleop] Error: {e}")
            break
        time.sleep(interval)
    teleop_active = False
    print("[teleop] Loop stopped")

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

# ---- Joint Jog ----

@app.route("/move_jog", methods=["POST"])
def move_jog():
    """Jog a single joint by a delta amount (relative move)."""
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json or {}
    joint = data.get("joint")
    delta = float(data.get("delta", 0))
    if not joint or delta == 0:
        return jsonify({"error": "Provide 'joint' and 'delta'"}), 400
    # Read current position
    joints = get_joint_positions()
    if joints is None:
        return jsonify({"error": "Cannot read joint positions"}), 503
    # Find current value
    key = None
    for k in joints:
        if k.replace(".pos", "") == joint:
            key = k
            break
    if key is None:
        return jsonify({"error": f"Unknown joint '{joint}'"}), 400
    current = float(joints[key])
    target = current + delta
    try:
        action = {f"{joint}.pos": target}
        with robot_lock:
            robot.send_action(action)
        return jsonify({"ok": True, "joint": joint, "from": current, "to": target})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Teach & Replay ----

def recording_loop():
    """Background thread: record joint positions at RECORDING_FPS."""
    global recording_active, recording_frames
    interval = 1.0 / RECORDING_FPS
    print(f"[record] Recording at {RECORDING_FPS} fps...")
    while recording_active:
        joints = get_joint_positions()
        if joints:
            recording_frames.append({
                "t": time.time(),
                "joints": {k: float(v) for k, v in joints.items()}
            })
        time.sleep(interval)
    print(f"[record] Stopped — {len(recording_frames)} frames captured")


@app.route("/recording/start", methods=["POST"])
def recording_start():
    global recording_active, recording_name, recording_frames, recording_thread
    if recording_active:
        return jsonify({"ok": False, "error": "Already recording"})
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json or {}
    recording_name = data.get("name", f"rec_{int(time.time())}")
    recording_frames = []
    recording_active = True
    recording_thread = threading.Thread(target=recording_loop, daemon=True)
    recording_thread.start()
    return jsonify({"ok": True, "name": recording_name})


@app.route("/recording/stop", methods=["POST"])
def recording_stop():
    global recording_active, recording_name, recording_frames
    if not recording_active:
        return jsonify({"ok": False, "error": "Not recording"})
    recording_active = False
    if recording_thread:
        recording_thread.join(timeout=2)
    # Save to file
    if not recording_frames:
        return jsonify({"ok": False, "error": "No frames recorded"})
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    filepath = os.path.join(RECORDINGS_DIR, f"{recording_name}.json")
    t0 = recording_frames[0]["t"]
    save_data = {
        "name": recording_name,
        "frames": len(recording_frames),
        "duration": recording_frames[-1]["t"] - t0,
        "fps": RECORDING_FPS,
        "data": [{"t": f["t"] - t0, "joints": f["joints"]} for f in recording_frames]
    }
    with open(filepath, "w") as f:
        json.dump(save_data, f)
    print(f"[record] Saved {len(recording_frames)} frames to {filepath}")
    frames_count = len(recording_frames)
    recording_frames = []
    return jsonify({"ok": True, "name": recording_name, "frames": frames_count, "file": filepath})


@app.route("/recordings")
def list_recordings():
    if not os.path.isdir(RECORDINGS_DIR):
        return jsonify({"recordings": []})
    recordings = []
    for fname in sorted(os.listdir(RECORDINGS_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(RECORDINGS_DIR, fname)) as f:
                data = json.load(f)
            recordings.append({
                "name": data.get("name", fname.replace(".json", "")),
                "frames": data.get("frames", 0),
                "duration": data.get("duration", 0),
            })
        except Exception:
            continue
    return jsonify({"recordings": recordings})


@app.route("/recording/replay", methods=["POST"])
def recording_replay():
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Provide 'name'"}), 400
    filepath = os.path.join(RECORDINGS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        return jsonify({"error": f"Recording '{name}' not found"}), 404
    try:
        with open(filepath) as f:
            rec = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to load: {e}"}), 500
    frames = rec.get("data", [])
    if not frames:
        return jsonify({"error": "Recording has no frames"}), 400
    # Replay in a background thread
    def do_replay():
        print(f"[replay] Playing '{name}' ({len(frames)} frames)...")
        agent_state["phase"] = "replaying"
        agent_state["detail"] = f"Replaying '{name}'..."
        prev_t = 0
        for i, frame in enumerate(frames):
            dt = frame["t"] - prev_t
            if dt > 0 and i > 0:
                time.sleep(dt)
            prev_t = frame["t"]
            try:
                action = {k: float(v) for k, v in frame["joints"].items()}
                with robot_lock:
                    robot.send_action(action)
            except Exception as e:
                print(f"[replay] Frame {i} error: {e}")
                break
        agent_state["phase"] = "idle"
        agent_state["detail"] = f"Replay '{name}' complete"
        print(f"[replay] Done")
    threading.Thread(target=do_replay, daemon=True).start()
    return jsonify({"ok": True, "name": name, "frames": len(frames)})


@app.route("/recording/delete", methods=["POST"])
def recording_delete():
    data = request.json or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Provide 'name'"}), 400
    filepath = os.path.join(RECORDINGS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        return jsonify({"error": f"Recording '{name}' not found"}), 404
    os.remove(filepath)
    return jsonify({"ok": True, "name": name})


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

    print("[boot] Starting SO-101 Robot Agent Server...")
    print(f"[boot] Robot port: {ROBOT_PORT}")
    init_robot()
    init_cameras()
    print(f"[boot] Server running on http://0.0.0.0:{PORT}")
    print(f"[boot] Cameras available: {list(cameras.keys())}")
    print(f"[boot] Robot connected: {robot is not None}")
    print(f"[boot] Live stream: http://localhost:{PORT}/stream")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
