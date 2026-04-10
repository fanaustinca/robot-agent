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
from flask import Flask, jsonify, request, Response, stream_with_context, render_template

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
SNAPSHOT_WIDTH = 640
SNAPSHOT_HEIGHT = 480
SNAPSHOT_TOP_WIDTH = 1920
SNAPSHOT_TOP_HEIGHT = 1080
JPEG_QUALITY = 90
CROSSHAIR_OFFSET_Y = 53  # pixels down from center (tune until dot matches gripper close point)
CROSSHAIR_OFFSET_X = 30   # pixels right from center

# Camera device indices — override with CAM_TOP / CAM_WRIST / CAM_SIDE env vars
# Or use CAM_TOP_NAME / CAM_WRIST_NAME / CAM_SIDE_NAME to find cameras by device name
CAMERA_TOP_IDX = int(os.environ.get("CAM_TOP", "4"))
CAMERA_WRIST_IDX = int(os.environ.get("CAM_WRIST", "0"))
CAMERA_SIDE_IDX = int(os.environ.get("CAM_SIDE", "0"))
CAMERA_TOP_NAME = os.environ.get("CAM_TOP_NAME", "HD Pro Webcam C920")
CAMERA_WRIST_NAME = os.environ.get("CAM_WRIST_NAME", "USB2.0_CAM1")
CAMERA_SIDE_NAME = os.environ.get("CAM_SIDE_NAME", "Logitech Webcam C930e")
SNAPSHOT_SIDE_WIDTH = 640
SNAPSHOT_SIDE_HEIGHT = 480

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
camera_locks = {"top": threading.Lock(), "wrist": threading.Lock(), "side": threading.Lock()}
robot_lock = threading.Lock()
torque_enabled = False
teleop_active = False
teleop_thread = None

# ---- Joint Position History (rolling 10 min buffer) ----
HISTORY_INTERVAL = 0.1   # seconds between samples (10hz)
HISTORY_MAX_SECS = 600   # 10 minutes
HISTORY_MAX_SAMPLES = int(HISTORY_MAX_SECS / HISTORY_INTERVAL)
joint_history = []       # [{t: float, joints: {k: v, ...}}, ...]
history_lock = threading.Lock()
history_enabled = True    # can be toggled via /timeline endpoint or agent command

def history_recorder():
    """Background thread: sample joint positions into rolling buffer."""
    while True:
        if not history_enabled:
            time.sleep(0.5)
            continue
        # Use trylock to avoid blocking teleop — skip sample if bus is busy
        if robot_lock.acquire(timeout=0.05):
            try:
                if robot is not None:
                    obs = robot.get_observation()
                    joints = {k: float(v) for k, v in obs.items() if "pos" in k or "joint" in k}
                    if joints:
                        with history_lock:
                            joint_history.append({"t": time.time(), "joints": joints})
                            if len(joint_history) > HISTORY_MAX_SAMPLES:
                                joint_history[:] = joint_history[-HISTORY_MAX_SAMPLES:]
            except Exception:
                pass
            finally:
                robot_lock.release()
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

# ---- Detection Overlay (agent pushes annotated frames to show on stream) ----
# {"top": base64_jpeg or None, "wrist": base64_jpeg or None, "ts": float}
detection_overlay = {"top": None, "wrist": None, "ts": 0}

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

    side_idx = find_camera_index_by_name(CAMERA_SIDE_NAME) if CAMERA_SIDE_NAME else None
    if side_idx is None:
        side_idx = CAMERA_SIDE_IDX
        print(f"{C.YELLOW}[camera]{C.RESET} Name lookup failed for side, using index {side_idx}")

    # Auto-detect if resolved indices still don't open
    needs_autodetect = []
    for name, idx in [("top", top_idx), ("wrist", wrist_idx), ("side", side_idx)]:
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            needs_autodetect.append(name)

    if needs_autodetect:
        print(f"{C.YELLOW}[camera]{C.RESET} Could not open configured indices for {needs_autodetect}, auto-detecting...")
        available = autodetect_cameras()
        if len(available) >= 3:
            top_idx, wrist_idx, side_idx = available[0], available[1], available[2]
            print(f"{C.CYAN}[camera]{C.RESET} Using auto-detected: top={top_idx}, wrist={wrist_idx}, side={side_idx}")
        elif len(available) >= 2:
            top_idx, wrist_idx = available[0], available[1]
            print(f"{C.CYAN}[camera]{C.RESET} Using auto-detected: top={top_idx}, wrist={wrist_idx}")
        elif len(available) == 1:
            top_idx = wrist_idx = available[0]
            print(f"{C.YELLOW}[camera]{C.RESET} Only one camera found (index {available[0]}), using for both")

    cam_names = {"top": CAMERA_TOP_NAME, "wrist": CAMERA_WRIST_NAME, "side": CAMERA_SIDE_NAME}
    for name, idx in [("top", top_idx), ("wrist", wrist_idx), ("side", side_idx)]:
        # Open by device path with V4L2 backend for reliable high-res capture
        dev_path = f"/dev/video{idx}"
        cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if cap.isOpened():
            # FOURCC must be set first, then resolution
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            if name == "top":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, SNAPSHOT_TOP_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SNAPSHOT_TOP_HEIGHT)
                cap.set(cv2.CAP_PROP_SHARPNESS, 7)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # autofocus on
            elif name == "side":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, SNAPSHOT_SIDE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SNAPSHOT_SIDE_HEIGHT)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # autofocus on
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, SNAPSHOT_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SNAPSHOT_HEIGHT)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras[name] = cap
            camera_info[name] = {"name": cam_names[name], "index": idx}
            print(f"{C.GREEN}[camera]{C.RESET} {name} camera opened ({dev_path}, {actual_w}x{actual_h})")
        else:
            print(f"{C.RED}[camera]{C.RESET} Could not open {name} camera ({dev_path})")

def reopen_camera(name):
    """Try to reopen a camera by name. Returns True on success."""
    import cv2
    idx = camera_info.get(name, {}).get("index")
    if idx is None:
        idx = {"wrist": CAMERA_WRIST_IDX, "side": CAMERA_SIDE_IDX}.get(name, CAMERA_TOP_IDX)
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
    if name == "top":
        w, h = SNAPSHOT_TOP_WIDTH, SNAPSHOT_TOP_HEIGHT
    elif name == "side":
        w, h = SNAPSHOT_SIDE_WIDTH, SNAPSHOT_SIDE_HEIGHT
    else:
        w, h = SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT
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
    "side_view": {"shoulder_pan": -5.5, "shoulder_lift": 24.1, "elbow_flex": 65.6, "wrist_flex": -71.6, "wrist_roll": -86.3, "gripper": 100},
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
        "history_enabled": history_enabled,
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
        # Use detection overlay if recent (within 10 seconds)
        overlay_b64 = detection_overlay.get(name)
        if overlay_b64 and (time.time() - detection_overlay.get("ts", 0)) < 10:
            overlay_bytes = base64.b64decode(overlay_b64)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + overlay_bytes + b"\r\n")
        else:
            if name == "top":
                sw, sh = SNAPSHOT_TOP_WIDTH, SNAPSHOT_TOP_HEIGHT
            elif name == "side":
                sw, sh = SNAPSHOT_SIDE_WIDTH, SNAPSHOT_SIDE_HEIGHT
            else:
                sw, sh = SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT
            frame = cv2.resize(frame, (sw, sh))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)  # ~20 fps target (cameras deliver ~12-15fps max)

@app.route("/stream/<name>")
def stream(name):
    if name not in ["top", "wrist", "side"]:
        return jsonify({"error": "Unknown camera. Use 'top', 'wrist', or 'side'"}), 400
    from flask import Response, stream_with_context
    return Response(stream_with_context(mjpeg_generator(name)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream")
def stream_index():
    return render_template("stream.html", camera_info=camera_info)


@app.route("/dashboard")
def dashboard_index():
    return render_template("dashboard.html")


@app.route("/yolo-detect")
def yolo_detect():
    """Run YOLO detection on top camera, return annotated image + results."""
    import cv2
    import numpy as np
    label = request.args.get("label", "") or None
    roi_str = request.args.get("roi", "")

    b64_frame, err = capture_snapshot("top")
    if err:
        return jsonify({"error": err}), 503

    img_bytes = base64.b64decode(b64_frame)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]

    # Lazy-load YOLO
    from detect import detect_objects, annotate_frame
    from camera_calibration import load_calibration, pixel_to_table_ray
    from config import GRIPPER_CLEARANCE, TOP_CALIB_FILE

    # Parse ROI
    roi_px = None
    if roi_str:
        try:
            rx, ry, rw, rh = [float(v) for v in roi_str.split(",")]
            x1, y1 = max(0, int(rx * w)), max(0, int(ry * h))
            x2, y2 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))
            if x2 - x1 > 10 and y2 - y1 > 10:
                roi_px = (x1, y1, x2, y2)
        except Exception:
            pass

    # Run YOLO on ROI or full frame
    if roi_px:
        x1, y1, x2, y2 = roi_px
        crop = frame[y1:y2, x1:x2]
        dets = detect_objects(crop, label)
        for d in dets:
            d["bbox"] = (d["bbox"][0]+x1, d["bbox"][1]+y1, d["bbox"][2]+x1, d["bbox"][3]+y1)
            d["center"] = (d["center"][0]+x1, d["center"][1]+y1)
    else:
        dets = detect_objects(frame, label)

    annotated = annotate_frame(frame, dets)
    if roi_px:
        x1, y1, x2, y2 = roi_px
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(annotated, "ROI", (x1+4, y1+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.line(annotated, (w//2-20, h//2), (w//2+20, h//2), (255, 0, 255), 1)
    cv2.line(annotated, (w//2, h//2-20), (w//2, h//2+20), (255, 0, 255), 1)

    # Compute 3D position
    position = None
    if dets:
        top_matrix, top_dist, calib_res = load_calibration(TOP_CALIB_FILE)
        if top_matrix is not None:
            # Scale calibration if frame resolution differs
            if calib_res is not None:
                calib_w, calib_h = calib_res
                if w != calib_w or h != calib_h:
                    sx, sy = w / calib_w, h / calib_h
                    top_matrix = top_matrix.copy()
                    top_matrix[0, 0] *= sx
                    top_matrix[0, 2] *= sx
                    top_matrix[1, 1] *= sy
                    top_matrix[1, 2] *= sy
            from yolo_ik_agent import get_top_camera_extrinsics
            cam_pos, cam_rot = get_top_camera_extrinsics()
            point = pixel_to_table_ray(dets[0]["center"], top_matrix, top_dist, cam_pos, cam_rot)
            if point is not None:
                point[2] = GRIPPER_CLEARANCE
                position = [round(point[0]*100, 1), round(point[1]*100, 1), round(point[2]*100, 1)]
                cv2.putText(annotated, f"right={position[0]}cm fwd={position[1]}cm up={position[2]}cm",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    # Also run detection on wrist camera
    wrist_b64 = None
    wrist_dets = []
    b64_wrist, werr = capture_snapshot("wrist")
    if not werr:
        wrist_bytes = base64.b64decode(b64_wrist)
        wrist_array = np.frombuffer(wrist_bytes, dtype=np.uint8)
        wrist_frame = cv2.imdecode(wrist_array, cv2.IMREAD_COLOR)
        wrist_dets = detect_objects(wrist_frame, label)
        wrist_annotated = annotate_frame(wrist_frame, wrist_dets)
        _, wbuf = cv2.imencode(".jpg", wrist_annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        wrist_b64 = base64.b64encode(wbuf.tobytes()).decode()

    return jsonify({
        "detections": dets, "image": img_b64, "position": position,
        "wrist_detections": wrist_dets, "wrist_image": wrist_b64,
    })


@app.route("/snapshot/<name>")
def snapshot(name):
    if name not in ["top", "wrist", "side"]:
        return jsonify({"error": "Unknown camera. Use 'top', 'wrist', or 'side'"}), 400
    b64, err = capture_snapshot(name)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({
        "camera": name,
        "format": "jpeg",
        "width": SNAPSHOT_TOP_WIDTH if name == "top" else (SNAPSHOT_SIDE_WIDTH if name == "side" else SNAPSHOT_WIDTH),
        "height": SNAPSHOT_TOP_HEIGHT if name == "top" else (SNAPSHOT_SIDE_HEIGHT if name == "side" else SNAPSHOT_HEIGHT),
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

@app.route("/detection_overlay", methods=["POST"])
def push_detection_overlay():
    """Agent pushes annotated detection frames to show on the camera streams."""
    data = request.json or {}
    if "top" in data:
        detection_overlay["top"] = data["top"]
    if "wrist" in data:
        detection_overlay["wrist"] = data["wrist"]
    if "side" in data:
        detection_overlay["side"] = data["side"]
    detection_overlay["ts"] = time.time()
    return jsonify({"ok": True})

@app.route("/detection_overlay", methods=["DELETE"])
def clear_detection_overlay():
    """Clear the detection overlay."""
    detection_overlay["top"] = None
    detection_overlay["wrist"] = None
    detection_overlay["side"] = None
    return jsonify({"ok": True})

# ---- Command Execution (runs yolo_ik_agent commands from dashboard) ----
import threading
_cmd_state = {"running": False, "command": "", "status": "idle", "log": []}
_cmd_lock = threading.Lock()

def _run_command_thread(command):
    """Execute a yolo_ik_agent command in a background thread."""
    import io, contextlib, re as _re
    try:
        from yolo_ik_agent import (
            pickup_sequence, detect_and_locate, move_to_xyz,
            move_preset, get_joint_angles, forward_kinematics,
            urdf_to_physical, get_status, _send_joints
        )
        with _cmd_lock:
            _cmd_state["running"] = True
            _cmd_state["command"] = command
            _cmd_state["status"] = "running"
            _cmd_state["log"] = []

        def log(msg):
            # Strip ANSI codes
            clean = _re.sub(r'\033\[[0-9;]*m', '', str(msg))
            with _cmd_lock:
                _cmd_state["log"].append(clean)
            print(msg)

        lower = command.lower().strip()

        # Pickup commands
        pickup_kw = ["pick up", "pickup", "grab", "retrieve", "fetch", "collect"]
        is_pickup = any(kw in lower for kw in pickup_kw)
        is_pick3d = lower.startswith("/pick3d")

        is_pick = lower.startswith("/pick ") or lower == "/pick"

        if is_pick:
            label = command.split(None, 1)[1] if " " in command else None
            if label:
                log(f"[cmd] Running pickup: {label}")
                result = pickup_sequence(label)
                log(f"[cmd] Result: {'success' if result else 'failed'}")
            else:
                log("[cmd] Usage: /pick <object>")
        elif is_pick3d:
            label = command.split(None, 1)[1] if " " in command else None
            if label:
                log(f"[cmd] Running 3D pickup: {label}")
                result = pickup_sequence(label, use_stereo=True)
                log(f"[cmd] Result: {'success' if result else 'failed'}")
            else:
                log("[cmd] Usage: /pick3d <object>")
        elif is_pickup:
            # Extract object name
            m = _re.search(r'["\u201c](.+?)["\u201d]', command)
            if m:
                target = m.group(1).strip()
            else:
                for kw in pickup_kw:
                    idx = lower.find(kw)
                    if idx >= 0:
                        target = command[idx + len(kw):].strip()
                        target = _re.sub(r'^(the|a|an)\s+', '', target, flags=_re.IGNORECASE).strip()
                        break
                else:
                    target = command
            if target:
                log(f"[cmd] Running pickup: {target}")
                result = pickup_sequence(target)
                log(f"[cmd] Result: {'success' if result else 'failed'}")
            else:
                log("[cmd] What should I pick up?")
        elif lower.startswith("/locate3d"):
            label = command.split(None, 1)[1] if " " in command else None
            log(f"[cmd] Locating (stereo): {label or 'any object'}")
            pos = detect_and_locate(label, use_triangulation=True)
            if pos is not None:
                from yolo_ik_agent import format_position
                log(f"[cmd] Position: {format_position(pos)}")
            else:
                log("[cmd] Object not found")
        elif lower.startswith("/locate"):
            label = command.split(None, 1)[1] if " " in command else None
            log(f"[cmd] Locating: {label or 'any object'}")
            pos = detect_and_locate(label)
            if pos is not None:
                from yolo_ik_agent import format_position
                log(f"[cmd] Position: {format_position(pos)}")
            else:
                log("[cmd] Object not found")
        elif lower.startswith("/home"):
            log("[cmd] Moving to home...")
            move_preset("home")
            log("[cmd] Done")
        elif lower.startswith("/observe"):
            log("[cmd] Moving to observe...")
            move_preset("observe")
            log("[cmd] Done")
        elif lower.startswith("/ready"):
            log("[cmd] Moving to ready...")
            move_preset("ready")
            log("[cmd] Done")
        elif lower.startswith("/drop"):
            log("[cmd] Moving to drop...")
            move_preset("drop")
            log("[cmd] Done")
        elif lower.startswith("/rest"):
            log("[cmd] Moving to rest...")
            move_preset("rest")
            log("[cmd] Done")
        elif lower.startswith("/fk"):
            angles = get_joint_angles()
            if angles:
                urdf_pos = forward_kinematics(angles)
                phys = urdf_to_physical(urdf_pos)
                log(f"[cmd] Gripper at: right={phys[0]*100:.1f}cm fwd={phys[1]*100:.1f}cm up={phys[2]*100:.1f}cm")
                log(f"[cmd] Joints: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}")
        elif lower.startswith("/ik "):
            parts = command.split()
            if len(parts) == 4:
                try:
                    x, y, z = float(parts[1])/100, float(parts[2])/100, float(parts[3])/100
                    log(f"[cmd] Moving to ({parts[1]}, {parts[2]}, {parts[3]}) cm...")
                    move_to_xyz([x, y, z])
                    log("[cmd] Done")
                except ValueError:
                    log("[cmd] Invalid coordinates")
            else:
                log("[cmd] Usage: /ik <right_cm> <fwd_cm> <up_cm>")
        elif lower.startswith("/gripper"):
            parts = command.split()
            if len(parts) > 1 and parts[1].lower() in ("open", "100"):
                _send_joints({"gripper": 100})
                log("[cmd] Gripper opened")
            elif len(parts) > 1 and parts[1].lower() in ("close", "0"):
                _send_joints({"gripper": 0})
                log("[cmd] Gripper closed")
            else:
                log("[cmd] Usage: /gripper open|close")
        elif lower.startswith("/t ") or lower in ("/t", "/torque"):
            parts = command.split()
            if len(parts) > 1 and parts[1].lower() in ("on", "1"):
                enabled = True
            elif len(parts) > 1 and parts[1].lower() in ("off", "0"):
                enabled = False
            else:
                enabled = False  # default toggle off
            try:
                import requests as _req
                _req.post(f"http://localhost:{PORT}/enable", json={"enabled": enabled}, timeout=5)
                log(f"[cmd] Torque {'ON' if enabled else 'OFF'}")
            except Exception as e:
                log(f"[cmd] Error: {e}")
        elif lower.startswith("/pos ") or lower == "/pos":
            label = command.split(None, 1)[1] if " " in command else None
            if not label:
                log("[cmd] Usage: /pos <object>")
            else:
                log(f"[cmd] Positioning at: {label}")
                pos = detect_and_locate(label, use_triangulation=True)
                if pos is not None:
                    from yolo_ik_agent import format_position
                    log(f"[cmd] Object at: {format_position(pos)}")
                    log(f"[cmd] Moving to home first...")
                    move_preset("home")
                    import time as _time; _time.sleep(0.5)
                    _send_joints({"gripper": 100})
                    _time.sleep(0.5)
                    log(f"[cmd] Moving to target...")
                    move_to_xyz(pos)
                    log(f"[cmd] Done — check gripper alignment")
                else:
                    log("[cmd] Object not found")
        elif lower.startswith("/calibrate_surface"):
            from yolo_ik_agent import calibrate_surface
            label = command.split(None, 1)[1] if " " in command else None
            log(f"[cmd] Calibrating surface height{(' with ' + label) if label else ''}...")
            result = calibrate_surface(label)
            if result is not None:
                log(f"[cmd] Origin set: right={result[0]*100:.1f}cm fwd={result[1]*100:.1f}cm up={result[2]*100:.1f}cm")
            else:
                log("[cmd] Calibration failed")
        else:
            log(f"[cmd] Unknown command: {command}")

    except Exception as e:
        with _cmd_lock:
            _cmd_state["log"].append(f"[error] {e}")
        print(f"{C.RED}[cmd]{C.RESET} Error: {e}")
    finally:
        with _cmd_lock:
            _cmd_state["running"] = False
            _cmd_state["status"] = "done"

@app.route("/run_command", methods=["POST"])
def run_command():
    """Run a yolo_ik_agent command from the dashboard."""
    data = request.json or {}
    command = data.get("command", "").strip()
    if not command:
        return jsonify({"error": "No command provided"}), 400
    with _cmd_lock:
        if _cmd_state["running"]:
            return jsonify({"error": "A command is already running"}), 409
    t = threading.Thread(target=_run_command_thread, args=(command,), daemon=True)
    t.start()
    return jsonify({"ok": True, "command": command})

@app.route("/run_command", methods=["GET"])
def get_command_state():
    """Get the current command execution state."""
    with _cmd_lock:
        return jsonify(dict(_cmd_state))

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
    fail_count = 0
    max_fails = 20  # only stop after 20 consecutive failures (~1 second)
    print(f"{C.GREEN}[teleop]{C.RESET} Loop started at {TELEOP_FPS} fps")
    while teleop_active:
        try:
            action = leader.get_action()
            with robot_lock:
                robot.send_action(action)
            fail_count = 0
        except Exception as e:
            fail_count += 1
            if fail_count >= max_fails:
                print(f"{C.RED}[teleop]{C.RESET} {max_fails} consecutive errors, stopping: {e}")
                break
            elif fail_count == 1:
                print(f"{C.YELLOW}[teleop]{C.RESET} Transient error (retrying): {e}")
            time.sleep(0.05)
            continue
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

@app.route("/timeline", methods=["POST"])
def toggle_timeline():
    global history_enabled
    data = request.json or {}
    if "enabled" in data:
        history_enabled = bool(data["enabled"])
    else:
        history_enabled = not history_enabled
    print(f"{C.CYAN}[timeline]{C.RESET} Recording {'enabled' if history_enabled else 'disabled'}")
    return jsonify({"ok": True, "history_enabled": history_enabled})

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
    # Thin out if too many points for the dashboard (max ~3000 points)
    if len(data) > 3000:
        step = len(data) // 3000
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
    print(f"  {C.BLUE}[1/5]{C.RESET} Robot arm............", end=" ")
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
    print(f"  {C.BLUE}[2/5]{C.RESET} Torque...............", end=" ")
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
    print(f"  {C.BLUE}[3/5]{C.RESET} Top camera...........", end=" ")
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
    print(f"  {C.BLUE}[4/5]{C.RESET} Wrist camera.........", end=" ")
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

    # 5. Side camera
    print(f"  {C.BLUE}[5/5]{C.RESET} Side camera..........", end=" ")
    if "side" in cameras:
        import cv2
        cam = cameras["side"]
        for _ in range(4):
            cam.grab()
        ret, frame = cam.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"{C.GREEN}OK{C.RESET} ({w}x{h}, index {camera_info.get('side', {}).get('index', '?')})")
        else:
            print(f"{C.RED}FAIL{C.RESET} — opened but cannot read frames")
            errors.append("Side camera opened but frame read failed")
    else:
        print(f"{C.RED}FAIL{C.RESET} — not available")
        errors.append("Side camera not available")

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
    # Move arm to rest position on startup (direct servo control, no HTTP)
    if robot is not None:
        print(f"{C.BLUE}[boot]{C.RESET} Moving arm to rest position...")
        try:
            import time as _time
            rest = PRESETS["rest"]
            # Read current positions
            obs = robot.get_observation()
            current = {}
            for k, v in obs.items():
                clean = k.replace(".pos", "")
                if clean in rest:
                    current[clean] = float(v)
            # Interpolate in 20 steps
            steps = 20
            for step in range(1, steps + 1):
                t = step / steps
                action = {}
                for joint, target in rest.items():
                    cur = current.get(joint, target)
                    action[f"{joint}.pos"] = cur + (target - cur) * t
                robot.send_action(action)
                _time.sleep(0.05)
            print(f"{C.GREEN}[boot]{C.RESET} Arm at rest position")
        except Exception as e:
            print(f"{C.YELLOW}[boot]{C.RESET} Could not move to rest: {e}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
