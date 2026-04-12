"""
SO-101 Robot Agent Server
Bridges Claude/OpenClaw to the SO-101 arm + cameras via HTTP API.

Usage:
    python robot_server.py

Endpoints:
    GET  /status              - arm joint positions + server health
    GET  /snapshot/top        - base64 JPEG from top camera
    GET  /snapshot/side       - base64 JPEG from side camera
    POST /move                - move joints {joint_name: degrees, ...}
    POST /move_preset         - move to named pose (home, ready, rest)
    POST /enable              - enable/disable torque {"enabled": true/false}
"""

import base64
import os
import threading
import time

import cameras as cameras_config
from calibration import handle_calib_arm, handle_calib_ex
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)


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
JPEG_QUALITY = 90
CROSSHAIR_OFFSET_Y = (
    53  # pixels down from center (tune until dot matches gripper close point)
)
CROSSHAIR_OFFSET_X = 30  # pixels right from center

# Cameras resolved by device name — override with CAM_TOP_NAME / CAM_SIDE_NAME
CAMERA_TOP_NAME = os.environ.get("CAM_TOP_NAME", "HD Pro Webcam C920")
CAMERA_SIDE_NAME = os.environ.get("CAM_SIDE_NAME", "Logitech Webcam C930e")


import sys as _sys

_sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arms as arms_config

ROBOT_SERIAL = arms_config.follower_serial()
LEADER_SERIAL = arms_config.leader_serial()
TELEOP_FPS = arms_config.teleop_fps()

app = Flask(__name__)

# ---- Robot + Camera State ----
robot = None
cameras = {}
camera_info = {}  # {"top": {"name": ..., "index": ...}, "side": {...}}
camera_locks = {"top": threading.Lock(), "side": threading.Lock()}
robot_lock = threading.Lock()
torque_enabled = False

# ---- Joint Position History (rolling 10 min buffer) ----
HISTORY_INTERVAL = 0.1  # seconds between samples (10hz)
HISTORY_MAX_SECS = 600  # 10 minutes
HISTORY_MAX_SAMPLES = int(HISTORY_MAX_SECS / HISTORY_INTERVAL)
joint_history = []  # [{t: float, joints: {k: v, ...}}, ...]
history_lock = threading.Lock()
history_enabled = True  # can be toggled via /timeline endpoint or agent command


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
                    joints = {
                        k: float(v)
                        for k, v in obs.items()
                        if "pos" in k or "joint" in k
                    }
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
    "phase": "idle",  # idle, calibrating, homing, aligning, waiting_confirm, lowering, gripping, lifting, dropping, done
    "detail": "",  # human-readable detail text
    "align_iteration": 0,  # current alignment iteration
    "align_max": 0,  # max alignment iterations
    "confirm_pending": False,  # True when waiting for grip confirm
    "confirm_result": None,  # "y" or "n" — set by dashboard, read by agent
}

# ---- Detection Overlay (agent pushes annotated frames to show on stream) ----
# {"top": base64_jpeg or None, "side": base64_jpeg or None, "ts": float}
detection_overlay = {"top": None, "side": None, "ts": 0}

# ---- Chat Log (synced between CLI agent and dashboard) ----
# Each entry: {"role": "user"|"agent"|"system", "text": str, "ts": float, "id": int}
chat_log = []
chat_id_counter = 0
chat_lock = threading.Lock()
# Messages submitted from dashboard, waiting for agent to pick up
chat_pending = []


def init_robot():
    global robot
    robot = arms_config.connect_follower()
    if robot is None:
        print(f"{C.YELLOW}[robot]{C.RESET} Running in camera-only mode")


def init_cameras():
    for name, device_name in [("top", CAMERA_TOP_NAME), ("side", CAMERA_SIDE_NAME)]:
        cap, idx = cameras_config.open_capture(device_name, name)
        if cap is not None:
            cameras[name] = cap
            camera_info[name] = {"name": device_name, "index": idx}


def reopen_camera(name):
    """Try to reopen a camera by name. Returns True on success."""
    idx = camera_info.get(name, {}).get("index")
    if idx is None:
        cam_name = CAMERA_TOP_NAME if name == "top" else CAMERA_SIDE_NAME
        idx = cameras_config.find_index_by_name(cam_name) if cam_name else None
    if idx is None:
        print(f"{C.RED}[camera]{C.RESET} {name} reopen failed — no known index")
        return False
    print(f"{C.YELLOW}[camera]{C.RESET} {name} reopening index {idx}...")
    old = cameras.get(name)
    if old:
        try:
            old.release()
        except Exception:
            pass
    cap = cameras_config.reopen_by_index(idx)
    if cap is not None:
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
    w, h = cameras_config.get_resolution(name)
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


PRESETS = arms_config.PRESETS

# ---- Routes ----


@app.route("/status")
def status():
    joints = get_joint_positions()
    leader = arms_config.get_leader()
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
    if robot is not None and hasattr(robot, "bus") and hasattr(robot.bus, "port"):
        robot_port_info = robot.bus.port
    leader_port_info = None
    if leader is not None and hasattr(leader, "bus") and hasattr(leader.bus, "port"):
        leader_port_info = leader.bus.port
    elif leader is not None and hasattr(leader, "port"):
        leader_port_info = leader.port
    return jsonify(
        {
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
            "teleop_active": arms_config.is_teleop_active(),
            "teleop_fps": TELEOP_FPS,
            "floor_drop": floor_drop,
            "history_enabled": history_enabled,
            "agent": agent_state,
            "timestamp": time.time(),
        }
    )


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
                print(
                    f"{C.YELLOW}[stream]{C.RESET} {name}: {fail_count} consecutive failures, reopening..."
                )
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
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + overlay_bytes + b"\r\n"
            )
        else:
            sw, sh = cameras_config.get_resolution(name)
            frame = cv2.resize(frame, (sw, sh))
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        time.sleep(0.05)  # ~20 fps target (cameras deliver ~12-15fps max)


@app.route("/stream/<name>")
def stream(name):
    if name not in ["top", "side"]:
        return jsonify({"error": "Unknown camera. Use 'top' or 'side'"}), 400
    return Response(
        stream_with_context(mjpeg_generator(name)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


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
    from camera_calibration import pixel_to_table_ray
    from cameras import get_extrinsics, get_scaled_intrinsics
    from config import GRIPPER_CLEARANCE
    from detect import annotate_frame, detect_objects

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
            d["bbox"] = (
                d["bbox"][0] + x1,
                d["bbox"][1] + y1,
                d["bbox"][2] + x1,
                d["bbox"][3] + y1,
            )
            d["center"] = (d["center"][0] + x1, d["center"][1] + y1)
    else:
        dets = detect_objects(frame, label)

    annotated = annotate_frame(frame, dets)
    if roi_px:
        x1, y1, x2, y2 = roi_px
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            annotated,
            "ROI",
            (x1 + 4, y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
        )

    cv2.line(annotated, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (255, 0, 255), 1)
    cv2.line(annotated, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (255, 0, 255), 1)

    # Compute 3D position
    position = None
    if dets:
        try:
            top_matrix, top_dist = get_scaled_intrinsics("top", w, h)
            cam_pos, cam_rot = get_extrinsics("top")
            point = pixel_to_table_ray(
                dets[0]["center"], top_matrix, top_dist, cam_pos, cam_rot
            )
            if point is not None:
                point[2] = GRIPPER_CLEARANCE
                position = [
                    round(point[0] * 100, 1),
                    round(point[1] * 100, 1),
                    round(point[2] * 100, 1),
                ]
                cv2.putText(
                    annotated,
                    f"right={position[0]}cm fwd={position[1]}cm up={position[2]}cm",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )
        except Exception:
            pass

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    # Also run detection on side camera
    side_b64 = None
    side_dets = []
    b64_side, serr = capture_snapshot("side")
    if not serr:
        side_bytes = base64.b64decode(b64_side)
        side_array = np.frombuffer(side_bytes, dtype=np.uint8)
        side_frame = cv2.imdecode(side_array, cv2.IMREAD_COLOR)
        side_dets = detect_objects(side_frame, label)
        side_annotated = annotate_frame(side_frame, side_dets)
        _, sbuf = cv2.imencode(".jpg", side_annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        side_b64 = base64.b64encode(sbuf.tobytes()).decode()

    return jsonify(
        {
            "detections": dets,
            "image": img_b64,
            "position": position,
            "side_detections": side_dets,
            "side_image": side_b64,
        }
    )


@app.route("/snapshot/<name>")
def snapshot(name):
    if name not in ["top", "side"]:
        return jsonify({"error": "Unknown camera. Use 'top' or 'side'"}), 400
    b64, err = capture_snapshot(name)
    if err:
        return jsonify({"error": err}), 503
    return jsonify(
        {
            "camera": name,
            "format": "jpeg",
            "width": cameras_config.get_resolution(name)[0],
            "height": cameras_config.get_resolution(name)[1],
            "data": b64,
            "timestamp": time.time(),
        }
    )


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
ELBOW_FORWARD_RATIO = 2.0
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
        return jsonify(
            {
                "error": f"Unknown direction '{direction}'. Use forward/backward/left/right"
            }
        ), 400
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
                    action[f"{k}.pos"] = start[k] + (targets[k] - start[k]) * min(
                        1.0, t * 1.5
                    )
                elif direction == "backward" and k == "shoulder_lift":
                    # Shoulder leads elbow on backward
                    action[f"{k}.pos"] = start[k] + (targets[k] - start[k]) * min(
                        1.0, t * 1.5
                    )
                else:
                    action[f"{k}.pos"] = start[k] + (targets[k] - start[k]) * t
            with robot_lock:
                robot.send_action(action)
            time.sleep(0.08)
        return jsonify(
            {"ok": True, "direction": direction, "degrees": degrees, "sent": targets}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/move_preset", methods=["POST"])
def move_preset():
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    data = request.json or {}
    name = data.get("pose")
    if name not in PRESETS:
        return jsonify(
            {"error": f"Unknown pose '{name}'. Options: {list(PRESETS.keys())}"}
        ), 400
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
    if "side" in data:
        detection_overlay["side"] = data["side"]
    detection_overlay["ts"] = time.time()
    return jsonify({"ok": True})


@app.route("/detection_overlay", methods=["DELETE"])
def clear_detection_overlay():
    """Clear the detection overlay."""
    detection_overlay["top"] = None
    detection_overlay["side"] = None
    return jsonify({"ok": True})


# ---- Command Execution (runs yolo_ik_agent commands from dashboard) ----
import threading

_cmd_state = {"running": False, "command": "", "status": "idle", "log": []}
_cmd_lock = threading.Lock()


def _run_command_thread(command):
    """Execute a yolo_ik_agent command in a background thread."""
    import re as _re

    try:
        from yolo_ik_agent import (
            _send_joints,
            detect_and_locate,
            format_position,
            move_preset,
            move_to_xyz,
            pickup_sequence,
        )

        with _cmd_lock:
            _cmd_state["running"] = True
            _cmd_state["command"] = command
            _cmd_state["status"] = "running"
            _cmd_state["log"] = []

        def log(msg):
            # Strip ANSI codes
            clean = _re.sub(r"\033\[[0-9;]*m", "", str(msg))
            with _cmd_lock:
                _cmd_state["log"].append(clean)
            print(msg)

        lower = command.lower().strip()

        is_pick3d = lower.startswith("/pick3d")
        is_pick = lower.startswith("/pick ") or lower == "/pick"

        if is_pick3d:
            label = command.split(None, 1)[1] if " " in command else None
            if label:
                log(f"[cmd] Running 3D pickup: {label}")
                result = pickup_sequence(label, use_stereo=True)
                if result is not None:
                    log(f"[cmd] Coordinates: {format_position(result)}")
                    log("[cmd] Result: success")
                else:
                    log("[cmd] Result: failed")
            else:
                log("[cmd] Usage: /pick3d <object>")
        elif is_pick:
            label = command.split(None, 1)[1] if " " in command else None
            if label:
                log(f"[cmd] Running pickup: {label}")
                result = pickup_sequence(label)
                if result is not None:
                    log(f"[cmd] Coordinates: {format_position(result)}")
                    log("[cmd] Result: success")
                else:
                    log("[cmd] Result: failed")
            else:
                log("[cmd] Usage: /pick <object>")
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
            from yolo_ik_agent import handle_fk_command

            handle_fk_command(log)
        elif lower.startswith("/ik "):
            from yolo_ik_agent import handle_ik_command

            handle_ik_command(command.split()[1:], log)
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

                _req.post(
                    f"http://localhost:{PORT}/enable",
                    json={"enabled": enabled},
                    timeout=5,
                )
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
                    log("[cmd] Moving to home first...")
                    move_preset("home")
                    import time as _time

                    _time.sleep(0.5)
                    _send_joints({"gripper": 100})
                    _time.sleep(0.5)
                    log("[cmd] Moving to target...")
                    move_to_xyz(pos)
                    log("[cmd] Done — check gripper alignment")
                else:
                    log("[cmd] Object not found")
        elif lower.startswith("/autofocus"):
            cameras_config.autofocus_cameras(cameras, log)
        elif lower.startswith("/calib_arm"):
            args_str = command.split(None, 1)[1].strip() if " " in command else ""
            handle_calib_arm(args_str, log)
        elif lower.startswith("/calib_ex"):
            args_str = command.split(None, 1)[1].strip() if " " in command else ""
            handle_calib_ex(args_str, log, capture_snapshot, detection_overlay)
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


@app.route("/teleop", methods=["POST"])
def teleop():
    global torque_enabled
    data = request.json or {}
    action = data.get("action", "status")
    if action == "start":
        ok, msg = arms_config.start_teleop(robot, robot_lock)
        if ok:
            torque_enabled = True
        return jsonify(
            {"ok": ok, "message": msg, "teleop_active": arms_config.is_teleop_active()}
        )
    elif action == "stop":
        ok, msg = arms_config.stop_teleop()
        return jsonify(
            {"ok": ok, "message": msg, "teleop_active": arms_config.is_teleop_active()}
        )
    else:
        return jsonify({"teleop_active": arms_config.is_teleop_active()})


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
        print(
            f"{C.CYAN}[calib]{C.RESET} Floor drop set to {floor_drop}°"
            if floor_drop
            else f"{C.YELLOW}[calib]{C.RESET} Floor drop cleared"
        )
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
    print(
        f"{C.CYAN}[timeline]{C.RESET} Recording {'enabled' if history_enabled else 'disabled'}"
    )
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
        msg = {
            "role": "user",
            "text": text,
            "ts": time.time(),
            "id": chat_id_counter,
            "source": "dashboard",
        }
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
        cam = cameras["top"]
        for _ in range(4):
            cam.grab()
        ret, frame = cam.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(
                f"{C.GREEN}OK{C.RESET} ({w}x{h}, index {camera_info.get('top', {}).get('index', '?')})"
            )
        else:
            print(f"{C.RED}FAIL{C.RESET} — opened but cannot read frames")
            errors.append("Top camera opened but frame read failed")
    else:
        print(f"{C.RED}FAIL{C.RESET} — not available")
        errors.append("Top camera not available")

    # 4. Side camera
    print(f"  {C.BLUE}[4/4]{C.RESET} Side camera..........", end=" ")
    if "side" in cameras:
        cam = cameras["side"]
        for _ in range(4):
            cam.grab()
        ret, frame = cam.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(
                f"{C.GREEN}OK{C.RESET} ({w}x{h}, index {camera_info.get('side', {}).get('index', '?')})"
            )
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
    print(f"\n{C.BLUE}[boot]{C.RESET} Starting SO-101 Robot Agent Server...")
    print(
        f"{C.DIM}[boot] Follower serial: {ROBOT_SERIAL}, leader serial: {LEADER_SERIAL}{C.RESET}"
    )
    init_robot()
    init_cameras()
    # Start background history recorder
    threading.Thread(target=history_recorder, daemon=True).start()
    print(
        f"{C.GREEN}[boot]{C.RESET} History recorder started (sampling every {HISTORY_INTERVAL}s, {HISTORY_MAX_SECS // 60}min buffer)"
    )
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
