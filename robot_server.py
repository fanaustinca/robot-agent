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
SNAPSHOT_TOP_WIDTH = 512
SNAPSHOT_TOP_HEIGHT = 384
JPEG_QUALITY = 60
CROSSHAIR_OFFSET_Y = 38  # pixels down from center (tune until dot matches gripper close point)
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

# ---- Named Poses ----
PRESETS = {
    "home":    {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90, "gripper": 0},
    "default": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0},
    "rest":    {"shoulder_pan": -1.10, "shoulder_lift": -102.24, "elbow_flex": 96.57, "wrist_flex": 76.35, "wrist_roll": -86.02, "gripper": 1.20},
}

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
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:-apple-system,system-ui,sans-serif;padding:16px}
h1{font-size:20px;font-weight:600;margin-bottom:12px;color:#fff}
.layout{display:flex;gap:16px;flex-wrap:wrap}
.cameras{display:flex;gap:12px;flex:1;min-width:0}
.cam-box{flex:1;min-width:0}
.cam-box p{text-align:center;font-size:14px;color:#888;margin-bottom:4px}
.cam-box img{width:100%;border-radius:6px;background:#000}
.panel{width:300px;flex-shrink:0;display:flex;flex-direction:column;gap:12px}
.card{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px}
.card h2{font-size:13px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.joint-row{display:flex;justify-content:space-between;font-size:13px;padding:2px 0;font-family:'SF Mono',monospace}
.joint-name{color:#999}.joint-val{color:#4fc3f7}
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
</style>
</head><body>
<h1>SO-101 Dashboard</h1>
<div id="conn"><span class="status-dot" id="dot"></span><span id="conn-text">Connecting...</span></div>
<div class="layout">
  <div class="cameras">
    <div class="cam-box"><p>Top</p><img src="/stream/top"></div>
    <div class="cam-box"><p>Wrist</p><img src="/stream/wrist"></div>
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
      <h2>Joint Positions</h2>
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
        <button class="btn btn-blue" onclick="preset('default')">Default</button>
        <button class="btn btn-blue" onclick="preset('rest')">Rest</button>
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
  </div>
</div>
<script>
const JOINT_ORDER=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
const SHORT={shoulder_pan:'Pan',shoulder_lift:'Lift',elbow_flex:'Elbow',wrist_flex:'Flex',wrist_roll:'Roll',gripper:'Grip'};
const PHASE_LABELS={idle:'Idle',calibrating:'Calibrating',homing:'Homing',prescan:'Scanning',aligning:'Aligning',
  waiting_confirm:'Waiting for Confirm',lowering:'Lowering',gripping:'Gripping',lifting:'Lifting',
  dropping:'Dropping',done:'Done'};
function post(url,body){fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})}
function torque(on){post('/enable',{enabled:on})}
function grip(v){post('/move',{gripper:v})}
function preset(p){post('/move_preset',{pose:p})}
function teleopCtl(a){post('/teleop',{action:a})}
function confirmGrip(v){post('/confirm_grip',{confirm:v})}
function skipAlign(){post('/confirm_grip',{confirm:'done'})}
function poll(){
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('dot').className='status-dot ok';
    document.getElementById('conn-text').textContent='Connected — arm '+(d.robot_connected?'online':'offline');
    let h='';
    if(d.joints){
      for(let j of JOINT_ORDER){
        let k=Object.keys(d.joints).find(x=>x.replace('.pos','')==j);
        let v=k?d.joints[k]:'-';
        if(typeof v==='number')v=v.toFixed(1)+'\\u00b0';
        h+='<div class="joint-row"><span class="joint-name">'+SHORT[j]+'</span><span class="joint-val">'+v+'</span></div>';
      }
    }
    document.getElementById('joints').innerHTML=h;
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
  }).catch(()=>{
    document.getElementById('dot').className='status-dot err';
    document.getElementById('conn-text').textContent='Disconnected';
  });
}
poll();setInterval(poll,250);
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
