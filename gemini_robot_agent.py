"""
Gemini Robot Agent — Visual Alignment Only

Flow:
  1. Move arm to home position
  2. Open gripper
  3. Gemini uses wrist camera to navigate to the target (alignment loop)
  4. Close gripper

Usage:
    GEMINI_API_KEY=... python3 gemini_robot_agent.py
"""

import base64
import json
import os
import re
import select
import sys
import time
import requests
from google import genai
from google.genai import types

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

# ---- Joint Name Aliases ----
JOINT_ALIASES = {
    "sp": "shoulder_pan", "pan": "shoulder_pan",
    "sl": "shoulder_lift", "lift": "shoulder_lift",
    "ef": "elbow_flex", "elbow": "elbow_flex",
    "wf": "wrist_flex", "flex": "wrist_flex",
    "wr": "wrist_roll", "roll": "wrist_roll",
    "g": "gripper", "grip": "gripper",
}

def resolve_joint(name):
    """Resolve a joint alias to its full name."""
    return JOINT_ALIASES.get(name.lower(), name.lower())

ROBOT_SERVER = os.environ.get("ROBOT_SERVER", "http://localhost:7878")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
ALIGNER_MODEL = "gemini-3-flash-preview"
CLASSIFIER_MODEL = "gemini-3-flash-preview"

MAX_ALIGN_ITERATIONS = 10
MIN_DELTA_START = 15  # min movement for first 5 iterations
MIN_DELTA_STEP  = 3   # degrees to drop per iteration after iteration 5
MAX_DELTA = 20
ALIGN_SETTLE_DELAY = 0.6  # seconds to wait after each alignment move
SHOULDER_LIFT_MAX =  50   # max up (agent space)
SHOULDER_LIFT_MIN = -65   # max down — floor protection

HOME_POSITION = {
    "shoulder_pan": 0,
    "shoulder_lift": 0,
    "elbow_flex": 0,
    "wrist_flex": 0,
    "wrist_roll": -90,
    "gripper": 0,
}

DEFAULT_POSITION = {
    "shoulder_pan": 0,
    "shoulder_lift": 0,
    "elbow_flex": 0,
    "wrist_flex": 0,
    "wrist_roll": 0,
    "gripper": 0,
}

READY_POSITION = {
    "shoulder_pan": 0,
    "shoulder_lift": -40,
    "elbow_flex": 0,
    "wrist_flex": 0,
    "wrist_roll": -90,
    "gripper": 0,
}

REST_POSITION = {
    "shoulder_pan": -1.10,
    "shoulder_lift": 102.24,
    "elbow_flex": 96.57,
    "wrist_flex": 76.35,
    "wrist_roll": -86.02,
    "gripper": 1.20,
}

# Drop position (agent space — shoulder_lift inverted from hardware 10.46)
DROP_POSITION = {
    "shoulder_pan": -47.08,
    "shoulder_lift": -10.46,
    "elbow_flex": -12.44,
    "wrist_flex": 86.20,
    "wrist_roll": -94.64,
    "gripper": 0
}

SETUP_SHOULDER_LIFT = -40  # gripper close to ground (negative = down)
CALIB_STEP = 1.0          # degrees per calibration step
CALIB_DELAY = 0.15        # seconds between calibration steps
CALIB_STALL_THRESHOLD = 3.0  # if actual vs commanded differs by this much, we hit floor
CALIB_LIFT_OFFSET = 5     # degrees to raise off floor before gripping
CROSSHAIR_OFFSET_Y = 38   # pixels down from center — must match robot_server.py
CROSSHAIR_OFFSET_X = 30    # pixels right from center — must match robot_server.py
SLOW_MOVE_STEPS = 8      # number of interpolation steps for slow moves
SLOW_MOVE_DELAY = 0.08   # seconds between steps


ALIGNER_SYSTEM = """You are navigating a robot arm gripper to a target object using camera feedback.

The wrist camera image has a BLACK DOT with a crosshair drawn at its exact center.
This dot represents where the gripper will close. Your goal is to move the gripper until the black dot is directly over the target object.

Early iterations include both a top-down overview (top camera) and a close-up wrist view.
Later iterations provide only the wrist camera.

You can only command the gripper to move in one of four directions:
- "forward"  — gripper moves toward the far end of the workspace
- "backward" — gripper moves toward the near end of the workspace
- "left"     — gripper moves to the left
- "right"    — gripper moves to the right

Respond with JSON only — no explanation:

If the black dot crosshair is directly over the target and ready to grip:
{"aligned": true}

If movement needed (pick ONE direction per response):
{"aligned": false, "direction": "forward", "degrees": 20}

Rules:
- Only one direction per response
- Degrees range: 15 minimum, 15 maximum for first 3 moves then 10 maximum after that
- Use larger degrees when the dot is far from target, smaller when nearly there
- If the target is not visible, make your best guess on which direction to move to find it"""

# ---- Forward/Backward Elbow Compensation ----
# Ratio of elbow movement per degree of shoulder movement.
# Forward: elbow folds more to keep gripper from hitting ground.
# Backward: elbow extends less aggressively.
ELBOW_FORWARD_RATIO  = 2.0
ELBOW_BACKWARD_RATIO = 0.67

TOP_WRIST_ITERATIONS = 10  # number of iterations to include both cameras

PRESCAN_SYSTEM = """You are looking at a top-down camera view of a robot arm workspace (a flat board).
The robot gripper starts at the center of the board when at home position.

Given a target object description, estimate how far the gripper needs to move from center to reach the target:
- "left" / "right" in degrees of shoulder_pan (negative = left, positive = right). Range: roughly -45 to 45.
- "forward" / "backward" in degrees (positive = forward/away from robot, negative = backward/toward robot). Range: roughly -30 to 30.

Respond with JSON only — no explanation:
{"pan_degrees": 0, "forward_degrees": 0}

Examples:
- Object at center: {"pan_degrees": 0, "forward_degrees": 0}
- Object far left and slightly forward: {"pan_degrees": -30, "forward_degrees": 10}
- Object to the right and far away: {"pan_degrees": 20, "forward_degrees": 25}

Be conservative — it's better to undershoot than overshoot. The fine alignment loop will correct small errors."""
FREE_MODE_CAMERA = "top"   # default camera for free-form mode: "top" or "wrist"

# Software-tracked joint positions in agent space (avoids relying on hardware reads)
_commanded = {
    "shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
    "wrist_flex": 0, "wrist_roll": -90, "gripper": 0
}


def push_state(phase, detail="", align_iteration=0, align_max=0, confirm_pending=False):
    """Push agent activity state to the server dashboard."""
    try:
        requests.post(f"{ROBOT_SERVER}/agent_state", json={
            "phase": phase,
            "detail": detail,
            "align_iteration": align_iteration,
            "align_max": align_max,
            "confirm_pending": confirm_pending,
        }, timeout=2)
    except Exception:
        pass


def push_chat(text, role="agent"):
    """Push a message to the chat log on the server (syncs with dashboard)."""
    try:
        requests.post(f"{ROBOT_SERVER}/chat/push", json={"text": text, "role": role}, timeout=2)
    except Exception:
        pass


def get_pending_chat():
    """Check for messages submitted from the dashboard. Returns list of strings."""
    try:
        r = requests.get(f"{ROBOT_SERVER}/chat/pending", timeout=2)
        return r.json().get("messages", [])
    except Exception:
        return []


def wait_for_confirm(prompt, timeout=5, default="y"):
    """Wait for confirmation from either terminal (timed_confirm) or dashboard.
    Dashboard confirm takes priority if received before terminal timeout."""
    # Tell dashboard we're waiting
    push_state("waiting_confirm", prompt, confirm_pending=True)
    print(f"{prompt} {C.DIM}(auto-{default} in {timeout}s, or confirm via dashboard){C.RESET} ", end="", flush=True)

    start = time.time()
    buf = ""
    while time.time() - start < timeout:
        remaining = int(timeout - (time.time() - start)) + 1
        print(f"\r{prompt} {C.DIM}(auto-{default} in {remaining}s){C.RESET} ", end="", flush=True)

        # Check dashboard confirm
        try:
            r = requests.get(f"{ROBOT_SERVER}/status", timeout=1)
            agent = r.json().get("agent", {})
            cr = agent.get("confirm_result")
            if cr is not None:
                print(f"\n{C.CYAN}[dashboard]{C.RESET} Received: {cr}")
                push_state("waiting_confirm", confirm_pending=False)
                return cr
        except Exception:
            pass

        # Check terminal input
        if select.select([sys.stdin], [], [], 0.3)[0]:
            buf = sys.stdin.readline().strip().lower()
            break

    print()
    push_state("waiting_confirm", confirm_pending=False)
    if buf:
        return buf
    print(f"{C.DIM}      Auto-confirmed.{C.RESET}")
    return default


def move_joints(joints):
    # Clamp and invert shoulder_lift: positive = up in agent logic, negative = down
    # Clamp before inversion to enforce safe range
    translated = {}
    for k, v in joints.items():
        if k == "shoulder_lift":
            v = max(SHOULDER_LIFT_MIN, min(SHOULDER_LIFT_MAX, v))
        _commanded[k] = v  # track commanded position in agent space
        if k == "shoulder_lift":
            v = -v  # invert so positive=up maps to hardware
        translated[k] = v
    try:
        r = requests.post(f"{ROBOT_SERVER}/move", json=translated, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def hardware_to_agent(joints):
    """Convert raw hardware joint values to agent space (inverts shoulder_lift)."""
    result = {}
    for k, v in joints.items():
        clean_key = k.replace(".pos", "")
        result[clean_key] = (-v if clean_key == "shoulder_lift" else v)
    return result


def get_joints_agent():
    """Return current joint positions in agent space."""
    return hardware_to_agent(get_joints())


def move_joints_slow(target_joints):
    """Interpolate from commanded positions to target over SLOW_MOVE_STEPS steps.
    target_joints must be in agent space."""
    start = dict(_commanded)  # snapshot of where arm currently is

    for step in range(1, SLOW_MOVE_STEPS + 1):
        t = step / SLOW_MOVE_STEPS
        interpolated = {
            joint: start.get(joint, 0) + (target - start.get(joint, 0)) * t
            for joint, target in target_joints.items()
        }
        move_joints(interpolated)
        time.sleep(SLOW_MOVE_DELAY)


def move_direction(direction, degrees):
    """Move the arm in a cardinal direction (forward/backward/left/right) by degrees.
    Forward/backward uses trig to keep the gripper level at ARM_GRIPPER_CLEARANCE.
    Syncs from hardware first, then applies relative move with slow interpolation.
    Returns True on success."""
    hw = get_joints_agent()
    if hw:
        _commanded.update(hw)

    def cur(joint):
        return _commanded.get(joint, 0)

    if direction == "forward":
        target_sh = cur("shoulder_lift") - degrees
        target_el = cur("elbow_flex") - degrees * ELBOW_FORWARD_RATIO
    elif direction == "backward":
        target_sh = cur("shoulder_lift") + degrees
        target_el = cur("elbow_flex") + degrees * ELBOW_BACKWARD_RATIO
    elif direction == "left":
        move_joints_slow({"shoulder_pan": cur("shoulder_pan") - degrees})
        return True
    elif direction == "right":
        move_joints_slow({"shoulder_pan": cur("shoulder_pan") + degrees})
        return True
    else:
        return False

    # Interpolate: elbow leads on forward, shoulder leads on backward
    start_sh = cur("shoulder_lift")
    start_el = cur("elbow_flex")
    for step in range(1, SLOW_MOVE_STEPS + 1):
        t = step / SLOW_MOVE_STEPS
        if direction == "forward":
            t_el = min(1.0, t * 1.5)  # elbow leads
            t_sh = t
        else:
            t_sh = min(1.0, t * 1.5)  # shoulder leads
            t_el = t
        move_joints({
            "shoulder_lift": start_sh + (target_sh - start_sh) * t_sh,
            "elbow_flex":    start_el + (target_el - start_el) * t_el,
        })
        time.sleep(SLOW_MOVE_DELAY)
    return True


def get_joints():
    try:
        r = requests.get(f"{ROBOT_SERVER}/status", timeout=5)
        return r.json().get("joints") or {}
    except Exception:
        return {}


def get_wrist_snapshot():
    try:
        r = requests.get(f"{ROBOT_SERVER}/snapshot/wrist", timeout=10)
        data = r.json()
        if "error" in data:
            print(f"[camera] wrist error: {data['error']}")
            return None
        # Draw crosshair on center so Gemini has a clear alignment reference
        import cv2
        import numpy as np
        img_bytes = base64.b64decode(data["data"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        cx, cy = w // 2 + CROSSHAIR_OFFSET_X, h // 2 + CROSSHAIR_OFFSET_Y
        # Black dot with white border for visibility
        cv2.circle(img, (cx, cy), 10, (255, 255, 255), -1)
        cv2.circle(img, (cx, cy), 7,  (0, 0, 0),       -1)
        cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 0, 0), 2)
        cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 0, 0), 2)
        # Degree scale ruler (10° ≈ 80px)
        px_per_deg = 8
        ruler_y = h - 14
        ruler_cx = w // 2
        ruler_half = px_per_deg * 10
        cv2.line(img, (ruler_cx - ruler_half, ruler_y), (ruler_cx + ruler_half, ruler_y), (200, 100, 255), 1)
        for deg in range(-10, 11):
            x = ruler_cx + deg * px_per_deg
            if deg % 10 == 0:
                tick_h, thickness = 8, 2
            elif deg % 5 == 0:
                tick_h, thickness = 5, 1
            else:
                tick_h, thickness = 3, 1
            cv2.line(img, (x, ruler_y - tick_h), (x, ruler_y), (200, 100, 255), thickness)
        cv2.putText(img, "10", (ruler_cx - ruler_half - 4, ruler_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
        cv2.putText(img, "5", (ruler_cx - px_per_deg * 5 - 3, ruler_y - 6), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
        cv2.putText(img, "0", (ruler_cx - 3, ruler_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
        cv2.putText(img, "5", (ruler_cx + px_per_deg * 5 - 3, ruler_y - 6), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
        cv2.putText(img, "10", (ruler_cx + ruler_half - 4, ruler_y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 100, 255), 1)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception as e:
        print(f"[camera] wrist exception: {e}")
        return None


PICKUP_KEYWORDS = ["pick up", "pickup", "grab", "retrieve", "fetch", "collect"]
# "get", "take", "lift" removed — too many false positives ("get status", "take a photo")

def is_pickup_prompt(client, text):
    """Check if the prompt contains pickup-related keywords."""
    lower = text.lower()
    return any(kw in lower for kw in PICKUP_KEYWORDS)

def extract_object_name(text):
    """Extract the object name from a pickup prompt.
    Supports: pick up "red block", pick up the red block, grab red block, etc."""
    # Try quoted object first: pick up "red block"
    m = re.search(r'["\u201c](.+?)["\u201d]', text)
    if m:
        return m.group(1).strip()
    # Strip the keyword prefix and return the rest
    lower = text.lower()
    for kw in PICKUP_KEYWORDS:
        idx = lower.find(kw)
        if idx >= 0:
            rest = text[idx + len(kw):].strip()
            # Remove leading "the", "a", "an"
            rest = re.sub(r'^(the|a|an)\s+', '', rest, flags=re.IGNORECASE).strip()
            if rest:
                return rest
    return text.strip()


GRIP_CHECK_SYSTEM = """You are checking whether a robot gripper is holding an object.

You will see a wrist camera image taken after the gripper closed and the arm lifted.

The target object is: {object_name}

Look at the gripper in the image. Is the target object currently being held between the gripper fingers?

Respond with JSON only — no explanation:
{{"holding": true}} or {{"holding": false}}"""


def check_grip_success(client, object_name):
    """Send wrist snapshot to Gemini to verify if the object is being held.
    Returns True if holding, False if not, None on error."""
    wrist_b64 = get_wrist_snapshot()
    if not wrist_b64:
        print(f"{C.YELLOW}[grip-check]{C.RESET} Could not get wrist snapshot")
        return None

    parts = [
        types.Part.from_bytes(data=base64.b64decode(wrist_b64), mime_type="image/jpeg"),
        types.Part.from_text(text=f"Is the gripper holding the {object_name}? Look carefully at the gripper fingers."),
    ]

    try:
        response = client.models.generate_content(
            model=ALIGNER_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=GRIP_CHECK_SYSTEM.format(object_name=object_name)
            )
        )
    except Exception as e:
        print(f"{C.RED}[grip-check]{C.RESET} API error: {e}")
        return None

    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if not match:
        print(f"{C.YELLOW}[grip-check]{C.RESET} Could not parse response: {response.text}")
        return None

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    holding = result.get("holding", False)
    return holding


def timed_confirm(prompt, timeout=5, default="y"):
    """Prompt user with auto-confirm after timeout seconds. Returns 'y' or 'n'."""
    print(f"{prompt} {C.DIM}(auto-{default} in {timeout}s){C.RESET} ", end="", flush=True)
    start = time.time()
    buf = ""
    while time.time() - start < timeout:
        remaining = int(timeout - (time.time() - start)) + 1
        # Overwrite countdown
        print(f"\r{prompt} {C.DIM}(auto-{default} in {remaining}s){C.RESET} ", end="", flush=True)
        if select.select([sys.stdin], [], [], 0.5)[0]:
            buf = sys.stdin.readline().strip().lower()
            break
    print()  # newline after prompt
    if buf:
        return buf
    print(f"{C.DIM}      Auto-confirmed.{C.RESET}")
    return default


# ---- Floor Calibration ----
# _floor_drop: how many degrees below SETUP_SHOULDER_LIFT the floor is.
# None = not calibrated, will auto-calibrate on first pickup.
_floor_drop = None

def read_actual_shoulder_lift():
    """Read the actual shoulder_lift from hardware, converted to agent space."""
    joints = hardware_to_agent(get_joints())
    return joints.get("shoulder_lift", None)

def calibrate_floor():
    """Lower shoulder_lift from SETUP_SHOULDER_LIFT until the motor stalls (floor hit).
    Returns the drop in degrees from SETUP_SHOULDER_LIFT to the floor, or None on failure."""
    global _floor_drop

    push_state("calibrating", "Starting floor calibration...")
    print(f"\n{C.CYAN}{C.BOLD}[calib]{C.RESET} Floor calibration starting...")
    print(f"{C.DIM}[calib] Moving to home, then lowering to setup position...{C.RESET}")

    # Sync from hardware and move to home
    hw_joints = hardware_to_agent(get_joints())
    if hw_joints:
        _commanded.update(hw_joints)
    move_joints_slow(HOME_POSITION)
    _commanded.update(HOME_POSITION)
    time.sleep(0.3)

    # Open gripper
    move_joints({"gripper": 100})
    time.sleep(0.3)

    # Lower to setup position
    move_joints_slow({"shoulder_lift": SETUP_SHOULDER_LIFT})
    time.sleep(0.5)

    push_state("calibrating", "Probing for floor...")
    print(f"{C.CYAN}[calib]{C.RESET} Probing for floor (lowering {CALIB_STEP}° per step)...")

    commanded_sl = SETUP_SHOULDER_LIFT
    max_probe = 40  # don't go more than 40° below setup

    for step in range(int(max_probe / CALIB_STEP)):
        commanded_sl -= CALIB_STEP

        # Send directly to hardware (bypass clamp — we need to find the real floor)
        try:
            requests.post(f"{ROBOT_SERVER}/move",
                          json={"shoulder_lift": -commanded_sl}, timeout=10)
        except Exception:
            pass
        time.sleep(CALIB_DELAY)

        # Read actual position
        actual_sl = read_actual_shoulder_lift()
        if actual_sl is None:
            print(f"{C.RED}[calib]{C.RESET} Cannot read joint positions")
            return None

        diff = abs(commanded_sl - actual_sl)
        if step % 5 == 0:
            print(f"{C.DIM}[calib] step {step}: cmd={commanded_sl:.1f}° actual={actual_sl:.1f}° diff={diff:.1f}°{C.RESET}")

        if diff > CALIB_STALL_THRESHOLD:
            # Motor stalled — we hit the floor
            floor_sl = actual_sl
            drop = SETUP_SHOULDER_LIFT - floor_sl
            _floor_drop = drop
            print(f"{C.GREEN}{C.BOLD}[calib]{C.RESET} Floor detected!")
            print(f"  Setup position:  {SETUP_SHOULDER_LIFT}°")
            print(f"  Floor position:  {floor_sl:.1f}°")
            print(f"  Drop distance:   {C.BOLD}{drop:.1f}°{C.RESET}")

            # Return to setup position
            print(f"{C.DIM}[calib] Returning to setup position...{C.RESET}")
            for s in range(SLOW_MOVE_STEPS):
                t = (s + 1) / SLOW_MOVE_STEPS
                sl = floor_sl + (SETUP_SHOULDER_LIFT - floor_sl) * t
                try:
                    requests.post(f"{ROBOT_SERVER}/move",
                                  json={"shoulder_lift": -sl}, timeout=10)
                except Exception:
                    pass
                time.sleep(SLOW_MOVE_DELAY)
            _commanded["shoulder_lift"] = SETUP_SHOULDER_LIFT

            # Return to home
            move_joints_slow(HOME_POSITION)
            _commanded.update(HOME_POSITION)
            push_state("idle", f"Calibrated: drop={drop:.1f}°")
            print(f"{C.GREEN}[calib]{C.RESET} Calibration complete. Drop = {drop:.1f}°\n")
            return drop

    push_state("idle", "Calibration failed")
    print(f"{C.RED}[calib]{C.RESET} Floor not detected within {max_probe}° — calibration failed.\n")
    return None


def print_help():
    """Print all available commands."""
    print(f"\n{C.BOLD}Commands:{C.RESET}")
    print(f"  {C.CYAN}/help{C.RESET}, {C.CYAN}/h{C.RESET}              — show this help")
    print(f"  {C.CYAN}/home{C.RESET}                — move to home position (slow)")
    print(f"  {C.CYAN}/ready{C.RESET}               — move to ready position (home + shoulder down 40°)")
    print(f"  {C.CYAN}/rest{C.RESET}                — move to rest position (arm folded)")
    print(f"  {C.CYAN}/default{C.RESET}             — move all joints to 0 (slow)")
    print(f"  {C.CYAN}/drop{C.RESET}                — move to drop position (slow)")
    print(f"  {C.CYAN}/torque-on{C.RESET}  [motor]  — enable torque  {C.DIM}(alias: /t-on){C.RESET}")
    print(f"  {C.CYAN}/torque-off{C.RESET} [motor]  — disable torque {C.DIM}(alias: /t-off){C.RESET}")
    print(f"  {C.CYAN}/forward{C.RESET}  [deg]       — move gripper forward {C.DIM}(alias: /fwd, default 5°){C.RESET}")
    print(f"  {C.CYAN}/backward{C.RESET} [deg]       — move gripper backward {C.DIM}(alias: /bwd){C.RESET}")
    print(f"  {C.CYAN}/left{C.RESET}     [deg]       — move gripper left")
    print(f"  {C.CYAN}/right{C.RESET}    [deg]       — move gripper right")
    print(f"  {C.CYAN}/move{C.RESET} <joint> <deg>  — move a single joint")
    print(f"  {C.CYAN}/pos{C.RESET}  [motor]        — show joint positions")
    print(f"  {C.CYAN}/cam{C.RESET}  [top|wrist]    — show camera info")
    print(f"  {C.CYAN}/maincam{C.RESET} <top|wrist>  — set free-mode camera")
    print(f"  {C.CYAN}/calib{C.RESET}               — calibrate floor distance for pickup")
    print(f"  {C.CYAN}/teleop{C.RESET} [start|stop]  — leader arm teleoperation")
    print(f"  {C.CYAN}/doctor{C.RESET}              — run full diagnostics")
    print(f"\n{C.BOLD}Joint aliases:{C.RESET} {C.DIM}sp sl ef wf wr g  (or: pan lift elbow flex roll grip){C.RESET}")
    print(f"{C.BOLD}Quit:{C.RESET} quit / exit / q\n")


def startup_health_check():
    """Quick health check at startup — verify server, arm, and cameras."""
    print(f"{C.BOLD}Running startup checks...{C.RESET}")
    all_ok = True

    # Server
    try:
        r = requests.get(f"{ROBOT_SERVER}/status", timeout=3)
        status = r.json()
        print(f"  {C.GREEN}OK{C.RESET}  Server connected")
        arm_ok = status.get("robot_connected", False)
        print(f"  {C.GREEN}OK{C.RESET}  Arm connected" if arm_ok else f"  {C.YELLOW}--{C.RESET}  Arm not connected (camera-only mode)")
        if not arm_ok:
            all_ok = False
    except Exception as e:
        print(f"  {C.RED}FAIL{C.RESET}  Cannot reach server at {ROBOT_SERVER}")
        print(f"        Start robot_server.py first!")
        return False

    # Cameras — actually fetch a frame
    for cam in ["top", "wrist"]:
        try:
            r = requests.get(f"{ROBOT_SERVER}/snapshot/{cam}", timeout=5)
            data = r.json()
            if "error" in data:
                print(f"  {C.RED}FAIL{C.RESET}  {cam} camera — {data['error']}")
                all_ok = False
            else:
                print(f"  {C.GREEN}OK{C.RESET}  {cam} camera ({data.get('width')}x{data.get('height')})")
        except Exception:
            print(f"  {C.RED}FAIL{C.RESET}  {cam} camera — no response")
            all_ok = False

    if all_ok:
        print(f"  {C.GREEN}{C.BOLD}All checks passed.{C.RESET}\n")
    else:
        print(f"  {C.YELLOW}Some checks failed — limited functionality.{C.RESET}\n")
    return all_ok


def get_top_snapshot():
    try:
        r = requests.get(f"{ROBOT_SERVER}/snapshot/top", timeout=10)
        data = r.json()
        if "error" in data:
            return None
        return data["data"]
    except Exception:
        return None


FREE_SYSTEM = """You are an assistant controlling a SO-101 robot arm via a local HTTP API.
Available actions you can take by outputting a JSON command block:

Move joints (degrees):
{"action": "move", "joints": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0}}

Move to preset pose (home / ready / rest):
{"action": "preset", "pose": "home"}

Open gripper:
{"action": "move", "joints": {"gripper": 100}}

Close gripper:
{"action": "move", "joints": {"gripper": 0}}

Get status (ask to check joints):
{"action": "status"}

Request wrist camera snapshot:
{"action": "snapshot", "camera": "wrist"}

If you want to take an action, include a JSON block in your response.
If no action is needed, just respond naturally.
Always explain what you are doing."""


def run_free_agent(client, user_input):
    """Let Gemini freely control the robot for non-pickup prompts."""
    print(f"\n{C.CYAN}[free]{C.RESET} Running free-form agent mode\n")

    # Build message with camera snapshot for context
    if FREE_MODE_CAMERA == "wrist":
        image_b64 = get_wrist_snapshot()
        cam_label = "wrist"
    else:
        image_b64 = get_top_snapshot()
        cam_label = "top"
    parts = []
    if image_b64:
        parts.append(types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"))
        print(f"{C.DIM}[free] Attached {cam_label} camera snapshot{C.RESET}")

    # Include current joint state
    joints = get_joints_agent()
    context = f"Current joint positions: {joints}\n\nUser: {user_input}"
    parts.append(types.Part.from_text(text=context))

    response = client.models.generate_content(
        model=ALIGNER_MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(system_instruction=FREE_SYSTEM)
    )

    print(f"{C.BOLD}Gemini:{C.RESET} {response.text}\n")
    push_chat(response.text, role="agent")

    # Parse and execute any action in the response
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if not match:
        return

    try:
        cmd = json.loads(match.group())
    except json.JSONDecodeError:
        return

    action = cmd.get("action")
    if action == "move":
        joints_cmd = cmd.get("joints", {})
        print(f"[free] Moving: {joints_cmd}")
        move_joints(joints_cmd)
    elif action == "preset":
        pose = cmd.get("pose", "home")
        print(f"[free] Moving to preset: {pose}")
        requests.post(f"{ROBOT_SERVER}/move_preset", json={"pose": pose}, timeout=10)
    elif action == "status":
        print(f"[free] Status: {get_joints()}")
    elif action == "snapshot" and cmd.get("camera") == "wrist":
        print("[free] Gemini requested wrist snapshot, fetching...")
        wrist_b64 = get_wrist_snapshot()
        if not wrist_b64:
            print("[free] Could not get wrist snapshot")
            return
        followup_parts = [
            types.Part.from_bytes(data=base64.b64decode(wrist_b64), mime_type="image/jpeg"),
            types.Part.from_text(text="[Wrist camera snapshot as requested]")
        ]
        followup = client.models.generate_content(
            model=ALIGNER_MODEL,
            contents=[types.Content(role="user", parts=followup_parts)],
            config=types.GenerateContentConfig(system_instruction=FREE_SYSTEM)
        )
        print(f"Gemini: {followup.text}\n")


def prescan_target(client, target_description):
    """Use top camera to estimate target position before moving the arm.
    Returns (pan_degrees, forward_degrees) or None on failure."""
    push_state("homing", "Scanning workspace from top camera...")
    print(f"\n{C.CYAN}[prescan]{C.RESET} Scanning workspace for: {C.BOLD}{target_description}{C.RESET}")

    top_b64 = get_top_snapshot()
    if not top_b64:
        print(f"{C.YELLOW}[prescan]{C.RESET} No top camera — skipping prescan")
        return None

    parts = [
        types.Part.from_bytes(data=base64.b64decode(top_b64), mime_type="image/jpeg"),
        types.Part.from_text(text=f"Target object: {target_description}\n\nEstimate how far left/right and forward/backward the target is from the center of the board."),
    ]

    try:
        response = client.models.generate_content(
            model=ALIGNER_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(system_instruction=PRESCAN_SYSTEM)
        )
    except Exception as e:
        print(f"{C.RED}[prescan]{C.RESET} API error: {e}")
        return None

    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if not match:
        print(f"{C.YELLOW}[prescan]{C.RESET} Could not parse response")
        return None

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError:
        print(f"{C.YELLOW}[prescan]{C.RESET} Invalid JSON in response")
        return None

    pan = float(result.get("pan_degrees", 0))
    fwd = float(result.get("forward_degrees", 0))
    print(f"{C.GREEN}[prescan]{C.RESET} Estimated: pan={C.BOLD}{pan:.0f}°{C.RESET}, forward={C.BOLD}{fwd:.0f}°{C.RESET}")
    return (pan, fwd)


def apply_prescan(pan_degrees, forward_degrees):
    """Move the arm based on prescan estimates using the same direction logic as alignment."""
    moves = {}

    if abs(pan_degrees) > 1:
        moves["shoulder_pan"] = _commanded.get("shoulder_pan", 0) + pan_degrees
        print(f"{C.BLUE}[prescan]{C.RESET} Moving pan {pan_degrees:+.0f}°")

    if abs(forward_degrees) > 1:
        if forward_degrees > 0:
            moves["shoulder_lift"] = _commanded.get("shoulder_lift", 0) - forward_degrees * FORWARD_SHOULDER_RATIO
            moves["elbow_flex"] = _commanded.get("elbow_flex", 0) - forward_degrees * ELBOW_FORWARD_RATIO
        else:
            moves["shoulder_lift"] = _commanded.get("shoulder_lift", 0) - forward_degrees * BACKWARD_SHOULDER_RATIO
            moves["elbow_flex"] = _commanded.get("elbow_flex", 0) - forward_degrees * ELBOW_BACKWARD_RATIO
        print(f"{C.BLUE}[prescan]{C.RESET} Moving forward/back {forward_degrees:+.0f}°")

    if moves:
        move_joints_slow(moves)
    else:
        print(f"{C.DIM}[prescan] Target near center, no pre-move needed{C.RESET}")


def visual_align(client, target_description):
    """Navigate arm to target using wrist camera feedback. Returns True when aligned."""
    print(f"\n{C.CYAN}[align]{C.RESET} Navigating to: {C.BOLD}{target_description}{C.RESET}")
    print(f"{C.DIM}[align] Max {MAX_ALIGN_ITERATIONS} iterations — type 'done' or press Done on dashboard to skip{C.RESET}\n")

    for i in range(MAX_ALIGN_ITERATIONS):
        # Sync _commanded from hardware so dashboard jog/presets are reflected
        hw = get_joints_agent()
        if hw:
            _commanded.update(hw)

        # Check for skip: terminal input or dashboard "done" button
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower()
            if line in ("/done", "done", "d"):
                print(f"{C.GREEN}[align]{C.RESET} Skipped by user after {i} iteration(s)")
                return True
        try:
            r = requests.get(f"{ROBOT_SERVER}/status", timeout=1)
            agent = r.json().get("agent", {})
            if agent.get("confirm_result") == "done":
                print(f"{C.GREEN}[align]{C.RESET} Skipped via dashboard after {i} iteration(s)")
                # Reset so it doesn't re-trigger
                requests.post(f"{ROBOT_SERVER}/agent_state", json={"confirm_pending": False}, timeout=1)
                return True
        except Exception:
            pass

        use_top = i < TOP_WRIST_ITERATIONS
        cam_info = f"{C.DIM}top+wrist{C.RESET}" if use_top else f"{C.DIM}wrist only{C.RESET}"
        push_state("aligning", f"Iteration {i+1}/{MAX_ALIGN_ITERATIONS}", align_iteration=i+1, align_max=MAX_ALIGN_ITERATIONS)
        print(f"{C.CYAN}[align]{C.RESET} Iteration {C.BOLD}{i + 1}/{MAX_ALIGN_ITERATIONS}{C.RESET} ({cam_info})")

        wrist_b64 = get_wrist_snapshot()
        if not wrist_b64:
            print(f"{C.RED}[align]{C.RESET} Could not get wrist snapshot")
            return False

        parts = []
        if use_top:
            top_b64 = get_top_snapshot()
            if top_b64:
                parts.append(types.Part.from_bytes(data=base64.b64decode(top_b64), mime_type="image/jpeg"))
                parts.append(types.Part.from_text(text="[Top camera — overview]"))

        parts.append(types.Part.from_bytes(data=base64.b64decode(wrist_b64), mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text=f"[Wrist camera — the black dot crosshair marks the gripper center]\nTarget: {target_description}\nIs the black dot crosshair directly over the target? If not, which direction should the gripper move?"))

        response = client.models.generate_content(
            model=ALIGNER_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(system_instruction=ALIGNER_SYSTEM)
        )

        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not match:
            print(f"{C.YELLOW}[align]{C.RESET} Could not parse response, skipping")
            continue

        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            continue

        if result.get("aligned"):
            print(f"{C.GREEN}{C.BOLD}[align]{C.RESET} Target reached after {i + 1} iteration(s)")
            return True

        if result.get("lost"):
            print(f"{C.YELLOW}[align]{C.RESET} Target not visible — continuing anyway")
            continue

        direction = result.get("direction")
        if not direction:
            print(f"{C.GREEN}[align]{C.RESET} No direction — assuming aligned")
            return True

        # Clamp degrees
        max_delta = 15 if i < 3 else (5 if i >= MAX_ALIGN_ITERATIONS - 2 else 10)
        degrees = result.get("degrees", 5)
        degrees = min(max_delta, float(degrees))

        if direction not in ("forward", "backward", "left", "right"):
            print(f"[align] Unknown direction '{direction}', skipping")
            continue

        print(f"{C.BLUE}[align]{C.RESET} Moving {C.BOLD}{direction}{C.RESET} {degrees:.0f}°")
        move_direction(direction, degrees)
        time.sleep(ALIGN_SETTLE_DELAY)

    print(f"{C.YELLOW}[align]{C.RESET} Max iterations reached")
    return True


def run_agent():
    if not API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    print(f"\n{C.BOLD}{'=' * 50}{C.RESET}")
    print(f"{C.BOLD}SO-101 Gemini Robot Agent{C.RESET}")
    print(f"  Model:  {C.CYAN}{ALIGNER_MODEL}{C.RESET}")
    print(f"  Server: {C.CYAN}{ROBOT_SERVER}{C.RESET}")
    print(f"  Stream: {C.CYAN}http://localhost:7878/stream{C.RESET}")
    print(f"  Type {C.CYAN}/help{C.RESET} for commands, {C.CYAN}quit{C.RESET} to exit")
    print(f"{C.BOLD}{'=' * 50}{C.RESET}\n")

    startup_health_check()

    prompt_shown = False
    while True:
        # Show prompt if not yet shown
        if not prompt_shown:
            print("You: ", end="", flush=True)
            prompt_shown = True

        # Check for dashboard messages first
        pending = get_pending_chat()
        if pending:
            user_input = pending[0].strip()
            # Overwrite the "You: " prompt with the dashboard message
            print(f"\rYou {C.DIM}(dashboard){C.RESET}: {user_input}")
            prompt_shown = False
        else:
            try:
                # Non-blocking check: if stdin has data, read it; otherwise briefly poll dashboard
                if select.select([sys.stdin], [], [], 0.3)[0]:
                    user_input = sys.stdin.readline().strip()
                    prompt_shown = False
                    if not user_input:
                        continue
                else:
                    continue
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Push user message to chat log
        push_chat(user_input, role="user")

        # ---- Direct commands (no AI) ----
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ("/help", "/h"):
                print_help()
                push_chat("Showing help — see CLI for full output.", role="system")

            elif cmd in ("/home",):
                print(f"{C.BLUE}[home]{C.RESET} Moving to home position (slow)...")
                hw_joints = hardware_to_agent(get_joints())
                if hw_joints:
                    _commanded.update(hw_joints)
                move_joints_slow(HOME_POSITION)
                _commanded.update(HOME_POSITION)
                print(f"{C.GREEN}[home]{C.RESET} Done.")
                push_chat("Moved to home position.", role="agent")

            elif cmd in ("/ready",):
                print(f"{C.BLUE}[ready]{C.RESET} Moving to ready position (slow)...")
                hw_joints = hardware_to_agent(get_joints())
                if hw_joints:
                    _commanded.update(hw_joints)
                move_joints_slow(READY_POSITION)
                _commanded.update(READY_POSITION)
                print(f"{C.GREEN}[ready]{C.RESET} Done.")
                push_chat("Moved to ready position.", role="agent")

            elif cmd in ("/default",):
                print(f"{C.BLUE}[default]{C.RESET} Moving to default position (slow)...")
                hw_joints = hardware_to_agent(get_joints())
                if hw_joints:
                    _commanded.update(hw_joints)
                move_joints_slow(DEFAULT_POSITION)
                _commanded.update(DEFAULT_POSITION)
                print(f"{C.GREEN}[default]{C.RESET} Done.")
                push_chat("Moved to default position.", role="agent")

            elif cmd in ("/drop",):
                print(f"{C.BLUE}[drop]{C.RESET} Moving to drop position (slow)...")
                hw_joints = hardware_to_agent(get_joints())
                if hw_joints:
                    _commanded.update(hw_joints)
                move_joints_slow(DROP_POSITION)
                _commanded.update(DROP_POSITION)
                print(f"{C.GREEN}[drop]{C.RESET} Done.")
                push_chat("Moved to drop position.", role="agent")

            elif cmd in ("/rest",):
                print(f"{C.BLUE}[rest]{C.RESET} Moving to rest position (slow)...")
                hw_joints = hardware_to_agent(get_joints())
                if hw_joints:
                    _commanded.update(hw_joints)
                move_joints_slow(REST_POSITION)
                _commanded.update(REST_POSITION)
                print(f"{C.GREEN}[rest]{C.RESET} Done.")
                push_chat("Moved to rest position.", role="agent")

            elif cmd in ("/torque-h", "/t-h"):
                print(f"  {C.CYAN}/torque-on{C.RESET}  [motor]  — enable torque (all or specific)")
                print(f"  {C.CYAN}/torque-off{C.RESET} [motor]  — disable torque (all or specific)")
                print(f"  Motors: {C.DIM}shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper{C.RESET}")
                print(f"  Aliases: {C.DIM}sp, sl, ef, wf, wr, g{C.RESET}")
                print(f"  Example: /t-off g")

            elif cmd == "/move-h":
                print(f"  {C.CYAN}/move{C.RESET} <joint> <degrees>  — move a single joint")
                print(f"  Joints: {C.DIM}shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper{C.RESET}")
                print(f"  Aliases: {C.DIM}sp sl ef wf wr g  (or: pan lift elbow flex roll grip){C.RESET}")
                print(f"  Example: /move sl -30")
                print(f"  Example: /move g 100")

            elif cmd in ("/torque-on", "/t-on"):
                motor = resolve_joint(parts[1]) if len(parts) > 1 else None
                try:
                    payload = {"enabled": True}
                    if motor:
                        payload["motor"] = motor
                    r = requests.post(f"{ROBOT_SERVER}/enable", json=payload, timeout=5)
                    label = motor or "all"
                    if r.json().get("ok"):
                        print(f"{C.GREEN}[torque]{C.RESET} {label} ON")
                    else:
                        print(f"{C.RED}[torque]{C.RESET} Error: {r.json()}")
                except Exception as e:
                    print(f"{C.RED}[torque]{C.RESET} Error: {e}")

            elif cmd in ("/torque-off", "/t-off"):
                motor = resolve_joint(parts[1]) if len(parts) > 1 else None
                try:
                    payload = {"enabled": False}
                    if motor:
                        payload["motor"] = motor
                    r = requests.post(f"{ROBOT_SERVER}/enable", json=payload, timeout=5)
                    label = motor or "all"
                    if r.json().get("ok"):
                        print(f"{C.YELLOW}[torque]{C.RESET} {label} OFF")
                    else:
                        print(f"{C.RED}[torque]{C.RESET} Error: {r.json()}")
                except Exception as e:
                    print(f"{C.RED}[torque]{C.RESET} Error: {e}")

            elif cmd == "/pos":
                motor_filter = resolve_joint(parts[1]) if len(parts) > 1 else None
                try:
                    r = requests.get(f"{ROBOT_SERVER}/status", timeout=5)
                    status = r.json()
                    joints = status.get("joints", {})
                    torque = status.get("torque_enabled", None)
                    agent_joints = hardware_to_agent(joints)
                    if motor_filter:
                        if motor_filter in agent_joints:
                            print(f"  {motor_filter}: {C.CYAN}{agent_joints[motor_filter]:.1f}°{C.RESET}")
                        else:
                            print(f"{C.RED}[pos]{C.RESET} Unknown motor: {motor_filter}")
                            print(f"  Available: {', '.join(agent_joints.keys())}")
                    else:
                        for name, val in agent_joints.items():
                            print(f"  {name:16s} {C.CYAN}{val:7.1f}°{C.RESET}")
                    if torque is not None:
                        t_color = C.GREEN if torque else C.YELLOW
                        print(f"  Torque: {t_color}{'ON' if torque else 'OFF'}{C.RESET}")
                except Exception as e:
                    print(f"{C.RED}[pos]{C.RESET} Error: {e}")

            elif cmd in ("/forward", "/backward", "/left", "/right", "/fwd", "/bwd"):
                direction = cmd.lstrip("/")
                if direction == "fwd":
                    direction = "forward"
                elif direction == "bwd":
                    direction = "backward"
                deg = float(parts[1]) if len(parts) > 1 else 5.0
                print(f"{C.BLUE}[move]{C.RESET} {direction} {deg:.0f}°...")
                move_direction(direction, deg)
                print(f"{C.GREEN}[move]{C.RESET} Done.")
                push_chat(f"Moved {direction} {deg:.0f}°", role="agent")

            elif cmd == "/move":
                if len(parts) != 3:
                    print(f"{C.YELLOW}[move]{C.RESET} Usage: /move <joint> <degrees>")
                    print(f"       Aliases: {C.DIM}sp sl ef wf wr g{C.RESET}")
                else:
                    joint = resolve_joint(parts[1])
                    deg = parts[2]
                    try:
                        deg = float(deg)
                        result = move_joints({joint: deg})
                        if "error" in result:
                            print(f"{C.RED}[move]{C.RESET} Error: {result['error']}")
                        else:
                            print(f"{C.GREEN}[move]{C.RESET} {joint} -> {deg}°")
                            push_chat(f"Moved {joint} to {deg}°", role="agent")
                    except ValueError:
                        print(f"{C.RED}[move]{C.RESET} Invalid degree value: {parts[2]}")

            elif cmd == "/calib":
                result = calibrate_floor()
                if result is None:
                    print(f"{C.YELLOW}[calib]{C.RESET} Calibration failed. Using default drop of 20°.")

            elif cmd == "/doctor":
                print(f"\n{C.BOLD}[doctor] Running diagnostics...{C.RESET}\n")
                ok = True

                print(f"{C.BLUE}[doctor]{C.RESET} 1/5 Checking server connection...")
                try:
                    r = requests.get(f"{ROBOT_SERVER}/status", timeout=5)
                    status = r.json()
                    saved_joints = hardware_to_agent(status.get("joints", {}))
                    print(f"      {C.GREEN}OK{C.RESET} — joints: {saved_joints}")
                except Exception as e:
                    print(f"      {C.RED}FAIL{C.RESET} — {e}")
                    ok = False
                    saved_joints = None

                print(f"{C.BLUE}[doctor]{C.RESET} 2/5 Enabling torque...")
                try:
                    r = requests.post(f"{ROBOT_SERVER}/enable", json={"enabled": True}, timeout=5)
                    print(f"      {C.GREEN}OK{C.RESET}" if r.json().get("ok") else f"      {C.RED}FAIL{C.RESET} — {r.json()}")
                except Exception as e:
                    print(f"      {C.RED}FAIL{C.RESET} — {e}")
                    ok = False

                if saved_joints:
                    print(f"{C.BLUE}[doctor]{C.RESET} 3/5 Moving joints to 0...")
                    try:
                        _commanded.update(saved_joints)
                        zero = {j: 0 for j in saved_joints}
                        move_joints_slow(zero)
                        print(f"      {C.GREEN}OK{C.RESET} — at zero")
                        time.sleep(0.5)
                        print("      Restoring original positions...")
                        move_joints_slow(saved_joints)
                        print(f"      {C.GREEN}OK{C.RESET} — restored")
                    except Exception as e:
                        print(f"      {C.RED}FAIL{C.RESET} — {e}")
                        ok = False
                else:
                    print(f"{C.BLUE}[doctor]{C.RESET} 3/5 Skipped motor test (no joint data)")

                print(f"{C.BLUE}[doctor]{C.RESET} 4/5 Toggling torque off/on...")
                try:
                    requests.post(f"{ROBOT_SERVER}/enable", json={"enabled": False}, timeout=5)
                    time.sleep(0.3)
                    requests.post(f"{ROBOT_SERVER}/enable", json={"enabled": True}, timeout=5)
                    print(f"      {C.GREEN}OK{C.RESET}")
                except Exception as e:
                    print(f"      {C.RED}FAIL{C.RESET} — {e}")
                    ok = False

                print(f"{C.BLUE}[doctor]{C.RESET} 5/5 Testing cameras...")
                top_ok = get_top_snapshot() is not None
                print(f"      Top camera:   {C.GREEN}OK{C.RESET}" if top_ok else f"      Top camera:   {C.RED}FAIL{C.RESET}")
                wrist_ok = get_wrist_snapshot() is not None
                print(f"      Wrist camera: {C.GREEN}OK{C.RESET}" if wrist_ok else f"      Wrist camera: {C.RED}FAIL{C.RESET}")
                if not top_ok or not wrist_ok:
                    ok = False

                if ok:
                    print(f"\n{C.GREEN}{C.BOLD}[doctor] All checks passed!{C.RESET}\n")
                else:
                    print(f"\n{C.YELLOW}[doctor] Some checks failed.{C.RESET}\n")

            elif cmd == "/cam":
                cam_filter = parts[1].lower() if len(parts) > 1 else None
                try:
                    r = requests.get(f"{ROBOT_SERVER}/status", timeout=5)
                    info = r.json().get("camera_info", {})
                    if cam_filter:
                        if cam_filter in info:
                            c = info[cam_filter]
                            print(f"  {cam_filter:6s}  {c['name']}  /dev/video{c['index']}")
                        else:
                            print(f"{C.RED}[cam]{C.RESET} Unknown camera: {cam_filter}")
                            print(f"  Available: {', '.join(info.keys())}")
                    else:
                        for name, c in info.items():
                            print(f"  {C.CYAN}{name:6s}{C.RESET}  {c['name']}  /dev/video{c['index']}")
                        if not info:
                            print(f"  {C.YELLOW}No cameras connected{C.RESET}")
                except Exception as e:
                    print(f"{C.RED}[cam]{C.RESET} Error: {e}")

            elif cmd == "/maincam":
                global FREE_MODE_CAMERA
                if len(parts) != 2 or parts[1].lower() not in ("top", "wrist"):
                    print(f"{C.YELLOW}[maincam]{C.RESET} Usage: /maincam <top|wrist>  (currently: {C.CYAN}{FREE_MODE_CAMERA}{C.RESET})")
                else:
                    FREE_MODE_CAMERA = parts[1].lower()
                    print(f"{C.GREEN}[maincam]{C.RESET} Free-form mode camera set to: {C.CYAN}{FREE_MODE_CAMERA}{C.RESET}")

            elif cmd == "/teleop":
                action = parts[1].lower() if len(parts) > 1 else None
                if action == "stop":
                    try:
                        r = requests.post(f"{ROBOT_SERVER}/teleop", json={"action": "stop"}, timeout=5)
                        data = r.json()
                        if data.get("ok"):
                            print(f"{C.GREEN}[teleop]{C.RESET} Stopped")
                        else:
                            print(f"{C.YELLOW}[teleop]{C.RESET} {data.get('message', 'Unknown error')}")
                    except Exception as e:
                        print(f"{C.RED}[teleop]{C.RESET} Error: {e}")
                elif action == "start" or action is None:
                    port = parts[2] if len(parts) > 2 else None
                    try:
                        payload = {"action": "start"}
                        if port:
                            payload["port"] = port
                        r = requests.post(f"{ROBOT_SERVER}/teleop", json=payload, timeout=10)
                        data = r.json()
                        if data.get("ok"):
                            print(f"{C.GREEN}[teleop]{C.RESET} {C.BOLD}Active{C.RESET} — leader arm controlling follower")
                            print(f"  {C.DIM}Type /teleop stop to end{C.RESET}")
                        else:
                            print(f"{C.RED}[teleop]{C.RESET} {data.get('message', 'Unknown error')}")
                    except Exception as e:
                        print(f"{C.RED}[teleop]{C.RESET} Error: {e}")
                else:
                    print(f"{C.YELLOW}[teleop]{C.RESET} Usage: /teleop [start [port] | stop]")

            else:
                print(f"{C.YELLOW}Unknown command: {cmd}{C.RESET} — type {C.CYAN}/help{C.RESET} for available commands")
                push_chat(f"Unknown command: {cmd} — type /help for commands", role="system")
            continue

        if is_pickup_prompt(client, user_input):
            object_name = extract_object_name(user_input)
            push_chat(f"Starting pickup sequence for: {object_name}", role="system")
            print(f"\n{C.BOLD}Target object: {C.CYAN}{object_name}{C.RESET}\n")

            # Auto-calibrate on first pickup if not yet done
            if _floor_drop is None:
                print(f"{C.YELLOW}[calib]{C.RESET} Not calibrated yet — running auto-calibration...")
                result = calibrate_floor()
                if result is None:
                    print(f"{C.YELLOW}[calib]{C.RESET} Failed. Using default drop of 20°.")

            drop = _floor_drop if _floor_drop is not None else 20.0

            # Step 1: Prescan — use top camera to estimate target position BEFORE moving
            prescan_result = prescan_target(client, object_name)

            # Step 2: Move to home (smooth interpolation)
            push_state("homing", "Moving to home position...")
            print(f"\n{C.BLUE}[2/8]{C.RESET} Moving to home position...")
            hw_joints = hardware_to_agent(get_joints())
            if hw_joints:
                _commanded.update(hw_joints)
            move_joints_slow(HOME_POSITION)
            _commanded.update(HOME_POSITION)
            time.sleep(0.5)
            print(f"      {C.GREEN}Done.{C.RESET}")

            # Step 3: Open gripper
            push_state("homing", "Opening gripper...")
            print(f"{C.BLUE}[3/8]{C.RESET} Opening gripper...")
            move_joints({"gripper": 100})
            print(f"      {C.GREEN}Done.{C.RESET}")

            # Step 4: Lower arm for easier alignment (slow)
            push_state("homing", f"Lowering arm to {SETUP_SHOULDER_LIFT}°...")
            print(f"{C.BLUE}[4/8]{C.RESET} Lowering arm (shoulder_lift: {SETUP_SHOULDER_LIFT}°)...")
            move_joints_slow({"shoulder_lift": SETUP_SHOULDER_LIFT})
            print(f"      {C.GREEN}Done.{C.RESET}")

            # Step 5: Apply prescan — jump to estimated target position
            if prescan_result:
                pan, fwd = prescan_result
                push_state("homing", f"Jumping to prescan estimate (pan={pan:.0f}°, fwd={fwd:.0f}°)")
                print(f"{C.BLUE}[5/8]{C.RESET} Applying prescan estimate...")
                apply_prescan(pan, fwd)
                time.sleep(0.3)
                print(f"      {C.GREEN}Done.{C.RESET}")
            else:
                print(f"{C.DIM}[5/8] Prescan skipped{C.RESET}")

            # Step 6: Fine visual alignment
            print(f"{C.BLUE}[6/8]{C.RESET} Starting visual alignment...")
            visual_align(client, object_name)

            # Step 7: Confirm and close gripper
            print(f"\n{C.BOLD}[7/8] Ready to grip.{C.RESET} {C.DIM}(drop={drop:.1f}°){C.RESET}")
            confirm = wait_for_confirm("      Close gripper? (y/n):", timeout=10, default="y")
            if confirm not in ("y", "yes", ""):
                push_state("idle", "Pickup cancelled")
                print(f"      {C.YELLOW}Skipped.{C.RESET}")
                push_chat("Pickup cancelled.", role="agent")
                time.sleep(2)
                push_state("idle")
                print()
                continue

            # -- Grip attempt loop (retry up to MAX_GRIP_RETRIES times) --
            MAX_GRIP_RETRIES = 1
            grip_succeeded = False

            for grip_attempt in range(1 + MAX_GRIP_RETRIES):
                attempt_label = f"attempt {grip_attempt + 1}/{1 + MAX_GRIP_RETRIES}" if MAX_GRIP_RETRIES > 0 else ""
                if grip_attempt > 0:
                    print(f"\n{C.CYAN}[retry]{C.RESET} Retrying grip ({attempt_label})...")
                    push_chat(f"Object not held — retrying grip ({attempt_label})", role="system")

                    # Re-align for 5 more iterations
                    push_state("aligning", f"Re-aligning for retry...")
                    print(f"{C.BLUE}[retry]{C.RESET} Re-aligning (5 iterations)...")
                    global MAX_ALIGN_ITERATIONS
                    saved_max = MAX_ALIGN_ITERATIONS
                    MAX_ALIGN_ITERATIONS = 5
                    visual_align(client, object_name)
                    MAX_ALIGN_ITERATIONS = saved_max

                # Lower to object
                push_state("lowering", f"Lowering {drop:.1f}° to object...")
                print(f"      {C.BLUE}Lowering to object ({drop:.1f}°)...{C.RESET}")
                start_sl = _commanded.get("shoulder_lift", 0)
                target_sl = start_sl - drop
                lower_steps = SLOW_MOVE_STEPS * 2
                for step in range(1, lower_steps + 1):
                    t = step / lower_steps
                    sl = start_sl + (target_sl - start_sl) * t
                    try:
                        requests.post(f"{ROBOT_SERVER}/move", json={"shoulder_lift": -sl}, timeout=10)
                    except Exception:
                        pass
                    time.sleep(SLOW_MOVE_DELAY)

                # Brief pause before grip
                print(f"      {C.DIM}Waiting before grip...{C.RESET}")
                time.sleep(1.0)
                try:
                    requests.post(f"{ROBOT_SERVER}/enable", json={"enabled": True}, timeout=5)
                except Exception:
                    pass

                # Raise slightly off floor
                print(f"      {C.BLUE}Raising {CALIB_LIFT_OFFSET}° off floor...{C.RESET}")
                lift_sl = target_sl + CALIB_LIFT_OFFSET
                try:
                    requests.post(f"{ROBOT_SERVER}/move", json={"shoulder_lift": -lift_sl}, timeout=10)
                except Exception:
                    pass
                time.sleep(0.5)

                # Close gripper slowly
                push_state("gripping", "Closing gripper...")
                print(f"      {C.BLUE}Closing gripper (slow)...{C.RESET}")
                grip_steps = SLOW_MOVE_STEPS * 2
                grip_delay = SLOW_MOVE_DELAY * 2
                for step in range(1, grip_steps + 1):
                    t = step / grip_steps
                    g = 100 - 100 * t
                    for _ in range(3):
                        try:
                            requests.post(f"{ROBOT_SERVER}/move", json={"gripper": g}, timeout=10)
                            break
                        except Exception:
                            time.sleep(0.1)
                    time.sleep(grip_delay)
                _commanded["gripper"] = 0
                print(f"      {C.DIM}Gripper closed.{C.RESET}")
                time.sleep(1.0)

                # Lift arm
                push_state("lifting", "Lifting arm to check grip...")
                print(f"      {C.BLUE}Lifting arm...{C.RESET}")
                pre_lift_pos = dict(_commanded)
                move_joints_slow({"shoulder_lift": _commanded.get("shoulder_lift", 0) + 30})
                time.sleep(0.5)

                # Step 8: Grip verification — send wrist snapshot to Gemini
                push_state("lifting", f"Checking if {object_name} is held...")
                print(f"\n{C.BLUE}[8/8]{C.RESET} Verifying grip on {C.BOLD}{object_name}{C.RESET}...")
                holding = check_grip_success(client, object_name)

                if holding is True:
                    print(f"      {C.GREEN}{C.BOLD}Object held!{C.RESET}")
                    push_chat(f"Grip verified — holding {object_name}!", role="agent")
                    grip_succeeded = True
                    break
                elif holding is False:
                    print(f"      {C.RED}Object NOT held.{C.RESET}")
                    push_chat(f"Grip check: {object_name} not held.", role="agent")
                else:
                    print(f"      {C.RED}{C.BOLD}Grip check inconclusive!{C.RESET} Vision could not determine if object is held.")
                    print(f"      {C.YELLOW}Please check manually — is the {C.BOLD}{object_name}{C.RESET}{C.YELLOW} in the gripper?{C.RESET}")
                    push_chat(f"ERROR: Grip check inconclusive for \"{object_name}\". Please check manually — is the object held? (y/n)", role="system")
                    manual = wait_for_confirm(f"      Object held? (y/n):", timeout=15, default="n")
                    if manual in ("y", "yes"):
                        print(f"      {C.GREEN}User confirmed — object held.{C.RESET}")
                        push_chat("User confirmed object is held.", role="agent")
                        grip_succeeded = True
                        break
                    else:
                        print(f"      {C.YELLOW}User says not held.{C.RESET}")
                        push_chat("User says object not held.", role="agent")

                # Object not held — check if we can retry
                if grip_attempt < MAX_GRIP_RETRIES:
                    # Give user 3 seconds to reject retry (→ go home instead)
                    print(f"\n{C.YELLOW}[retry]{C.RESET} Will retry in 3s. Press {C.BOLD}n{C.RESET} or {C.BOLD}Cancel{C.RESET} on dashboard to abort → home")
                    abort = wait_for_confirm("      Retry? (y/n):", timeout=3, default="y")
                    if abort in ("n", "no", "cancel"):
                        print(f"      {C.YELLOW}Aborted — returning home.{C.RESET}")
                        push_chat("Retry rejected — returning home.", role="agent")
                        # Open gripper and go home
                        move_joints({"gripper": 100})
                        _commanded["gripper"] = 100
                        time.sleep(0.3)
                        hw_joints = hardware_to_agent(get_joints())
                        if hw_joints:
                            _commanded.update(hw_joints)
                        move_joints_slow(HOME_POSITION)
                        _commanded.update(HOME_POSITION)
                        push_state("idle", "Pickup aborted")
                        time.sleep(2)
                        push_state("idle")
                        print()
                        break

                    # Return to pre-lift position, open gripper, and retry alignment
                    print(f"      {C.BLUE}Returning to last position...{C.RESET}")
                    move_joints({"gripper": 100})
                    _commanded["gripper"] = 100
                    time.sleep(0.3)
                    move_joints_slow({"shoulder_lift": pre_lift_pos.get("shoulder_lift", SETUP_SHOULDER_LIFT)})
                else:
                    # No more retries
                    print(f"      {C.RED}Max retries reached. Returning home.{C.RESET}")
                    push_chat(f"Failed to pick up {object_name} after retries. Returning home.", role="agent")
                    move_joints({"gripper": 100})
                    _commanded["gripper"] = 100
                    time.sleep(0.3)
                    hw_joints = hardware_to_agent(get_joints())
                    if hw_joints:
                        _commanded.update(hw_joints)
                    move_joints_slow(HOME_POSITION)
                    _commanded.update(HOME_POSITION)
                    push_state("idle", "Pickup failed")
                    time.sleep(2)
                    push_state("idle")
                    print()

            # If grip succeeded, proceed to drop
            if grip_succeeded:
                push_state("dropping", "Moving to drop position...")
                print(f"      {C.BLUE}Moving to drop position...{C.RESET}")
                move_joints_slow(DROP_POSITION)
                time.sleep(0.5)
                print(f"      {C.BLUE}Opening gripper...{C.RESET}")
                move_joints_slow({"gripper": 100})
                _commanded["gripper"] = 100
                push_state("done", "Pickup complete!")
                print(f"      {C.GREEN}{C.BOLD}Dropped!{C.RESET}")
                push_chat(f"Pickup complete! {object_name} dropped.", role="agent")
                # Reset to idle after a moment
                time.sleep(2)
                push_state("idle")
                print()  # blank line after pickup
        else:
            run_free_agent(client, user_input)


if __name__ == "__main__":
    run_agent()
