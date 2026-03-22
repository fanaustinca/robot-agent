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
import sys
import time
import requests
from google import genai
from google.genai import types

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
    "gripper": 0
}

SETUP_SHOULDER_LIFT = -30  # gripper close to ground (negative = down)
CROSSHAIR_OFFSET_Y = 38   # pixels down from center — must match robot_server.py
CROSSHAIR_OFFSET_X = 20    # pixels right from center — must match robot_server.py
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
- If the target is not visible at all, respond {"aligned": false, "lost": true}"""

FORWARD_SHOULDER_RATIO  = 1.0
BACKWARD_SHOULDER_RATIO = 1.0
ELBOW_FORWARD_RATIO     = 1.31
ELBOW_BACKWARD_RATIO    = 1.1

TOP_WRIST_ITERATIONS = 10  # number of iterations to include both cameras

# Software-tracked joint positions in agent space (avoids relying on hardware reads)
_commanded = {
    "shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
    "wrist_flex": 0, "wrist_roll": -90, "gripper": 0
}


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
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception as e:
        print(f"[camera] wrist exception: {e}")
        return None


def is_pickup_prompt(client, text):
    """Use a cheap Gemini model to classify whether the prompt is a pick-up request."""
    try:
        response = client.models.generate_content(
            model=CLASSIFIER_MODEL,
            contents=f'Is this a request to pick up, grab, or retrieve a physical object? Reply with only "yes" or "no".\n\n"{text}"'
        )
        return response.text.strip().lower().startswith("yes")
    except Exception as e:
        print(f"[classify] Error: {e} — falling back to free mode")
        return False


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
    print("\n[free] Running free-form agent mode\n")

    # Build message with a top camera snapshot for context
    image_b64 = get_top_snapshot()
    parts = []
    if image_b64:
        parts.append(types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"))
        print("[free] Attached top camera snapshot")

    # Include current joint state
    joints = get_joints_agent()
    context = f"Current joint positions: {joints}\n\nUser: {user_input}"
    parts.append(types.Part.from_text(text=context))

    response = client.models.generate_content(
        model=ALIGNER_MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(system_instruction=FREE_SYSTEM)
    )

    print(f"Gemini: {response.text}\n")

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


def visual_align(client, target_description):
    """Navigate arm to target using wrist camera feedback. Returns True when aligned."""
    print(f"\n[align] Navigating to: {target_description}")
    print(f"[align] Max {MAX_ALIGN_ITERATIONS} iterations, min move starts at {MIN_DELTA_START}° dropping by {MIN_DELTA_STEP}° after iter 5\n")

    for i in range(MAX_ALIGN_ITERATIONS):
        use_top = i < TOP_WRIST_ITERATIONS
        print(f"[align] Iteration {i + 1}/{MAX_ALIGN_ITERATIONS} ({'top+wrist' if use_top else 'wrist only'})")

        wrist_b64 = get_wrist_snapshot()
        if not wrist_b64:
            print("[align] Could not get wrist snapshot")
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
            print("[align] Could not parse response, skipping")
            continue

        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            continue

        if result.get("aligned"):
            print(f"[align] Target reached after {i + 1} iteration(s)")
            return True

        if result.get("lost"):
            print("[align] Target not visible — stopping")
            return False

        direction = result.get("direction")
        if not direction:
            print("[align] No direction — assuming aligned")
            return True

        # Clamp degrees
        max_delta = 15 if i < 3 else (5 if i >= MAX_ALIGN_ITERATIONS - 2 else 10)
        degrees = result.get("degrees", 5)
        degrees = min(max_delta, float(degrees))

        def cur(joint):
            return _commanded.get(joint, 0)

        if direction == "forward":
            new_pos = {
                "shoulder_lift": cur("shoulder_lift") - degrees * FORWARD_SHOULDER_RATIO,
                "elbow_flex":    cur("elbow_flex")    - degrees * ELBOW_FORWARD_RATIO,
            }
        elif direction == "backward":
            new_pos = {
                "shoulder_lift": cur("shoulder_lift") + degrees * BACKWARD_SHOULDER_RATIO,
                "elbow_flex":    cur("elbow_flex")    + degrees * ELBOW_BACKWARD_RATIO,
            }
        elif direction == "left":
            new_pos = {"shoulder_pan": cur("shoulder_pan") - degrees}
        elif direction == "right":
            new_pos = {"shoulder_pan": cur("shoulder_pan") + degrees}
        else:
            print(f"[align] Unknown direction '{direction}', skipping")
            continue

        print(f"[align] Moving {direction} {degrees:.0f}°")
        move_joints_slow(new_pos)
        time.sleep(ALIGN_SETTLE_DELAY)

    print("[align] Max iterations reached")
    return True


def run_agent():
    if not API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    print("=" * 50)
    print("SO-101 Gemini Robot Agent")
    print(f"Model:  {ALIGNER_MODEL}")
    print(f"Server: {ROBOT_SERVER}")
    print(f"Stream: http://localhost:7878/stream")
    print("Type 'quit' to exit")
    print("=" * 50)

    try:
        r = requests.get(f"{ROBOT_SERVER}/status", timeout=3)
        status = r.json()
        print(f"✓ Robot server connected")
        print(f"  Cameras: {status.get('cameras', [])}")
        print(f"  Arm: {'connected' if status.get('robot_connected') else 'NOT connected (camera-only mode)'}")
    except Exception:
        print(f"⚠  WARNING: Cannot reach robot server at {ROBOT_SERVER}")
        print("   Start robot_server.py first!")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("[classify] Detecting intent...")
        if is_pickup_prompt(client, user_input):
            # Step 1: Move to home (slow to avoid shaking)
            print("\n[1/4] Moving to home position...")
            move_joints(HOME_POSITION)          # direct send — arm may have been moved manually
            _commanded.update(HOME_POSITION)    # sync tracking to known state
            time.sleep(1.5)                     # wait for arm to settle
            print("      Done.")

            # Step 2: Open gripper
            print("[2/4] Opening gripper...")
            move_joints({"gripper": 100})
            print("      Done.")

            # Step 3: Lower arm for easier alignment (slow)
            print(f"[3/5] Lowering arm (shoulder_lift: {SETUP_SHOULDER_LIFT}°, slow)...")
            move_joints_slow({"shoulder_lift": SETUP_SHOULDER_LIFT})
            print("      Done.")

            # Step 4: Visual alignment
            print("[4/5] Starting visual alignment...")
            aligned = visual_align(client, user_input)

            if not aligned:
                print("\n[align] Could not reach target.")
                try:
                    requests.post(f"{ROBOT_SERVER}/enable", json={"enabled": False}, timeout=5)
                    print("      Torque disabled — move arm back manually.\n")
                except Exception:
                    pass
                continue

            # Step 5: Confirm and close gripper
            print("\n[5/5] Ready to grip.")
            confirm = input("      Close gripper? (y/n): ").strip().lower()
            if confirm == "y":
                # Lower arm to floor — bypass clamp so it can reach the object
                print("      Lowering to object...")
                start_sl = _commanded.get("shoulder_lift", 0)
                target_sl = start_sl - 5
                # Send directly to skip SHOULDER_LIFT_MIN clamp — use more steps for slower descent
                lower_steps = SLOW_MOVE_STEPS * 2
                for step in range(1, lower_steps + 1):
                    t = step / lower_steps
                    sl = start_sl + (target_sl - start_sl) * t
                    try:
                        requests.post(f"{ROBOT_SERVER}/move", json={"shoulder_lift": -sl}, timeout=10)
                    except Exception:
                        pass
                    time.sleep(SLOW_MOVE_DELAY)
                print("      Waiting before grip...")
                time.sleep(1.0)
                # Re-enable torque in case servo stalled against the floor
                try:
                    requests.post(f"{ROBOT_SERVER}/enable", json={"enabled": True}, timeout=5)
                except Exception:
                    pass
                print("      Closing gripper...")
                # Close gripper slowly — retry each step if servo errors from stall
                for step in range(1, SLOW_MOVE_STEPS + 1):
                    t = step / SLOW_MOVE_STEPS
                    g = 100 - 100 * t  # from open (100) to closed (0)
                    for _ in range(3):  # retry up to 3 times per step
                        try:
                            requests.post(f"{ROBOT_SERVER}/move", json={"gripper": g}, timeout=10)
                            break
                        except Exception:
                            time.sleep(0.1)
                    time.sleep(SLOW_MOVE_DELAY)
                _commanded["gripper"] = 0
                print("      Gripper closed. Waiting...")
                time.sleep(3.0)
                print("      Lifting arm...")
                move_joints_slow({"shoulder_lift": _commanded.get("shoulder_lift", 0) + 30})
                print("      Done.")
            else:
                print("      Skipped.")
            # Disable torque so arm can be moved manually back to home
            try:
                r = requests.post(f"{ROBOT_SERVER}/enable", json={"enabled": False}, timeout=5)
                result = r.json()
                if result.get("ok"):
                    print("      Torque disabled — move arm back manually.\n")
                else:
                    print(f"      Torque disable failed: {result}\n")
            except Exception as e:
                print(f"      Torque disable error: {e}\n")
        else:
            run_free_agent(client, user_input)


if __name__ == "__main__":
    run_agent()
