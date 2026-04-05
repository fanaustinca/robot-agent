"""
YOLO + IK Robot Agent for SO-101

Replaces Gemini's iterative visual alignment with:
  1. YOLO detects the target object in camera frames
  2. Camera calibration projects 2D pixels to 3D coordinates
  3. IKPy computes joint angles via inverse kinematics
  4. Robot server moves the arm directly to the target

Usage:
    python yolo_ik_agent.py

Pipeline:
    YOLO (detect) → OpenCV (3D position) → IKPy (joint angles) → LeRobot (move)
"""

import base64
import json
import os
import re
import select
import sys
import time

import cv2
import numpy as np
import requests

from config import (
    ROBOT_SERVER, TABLE_Z, GRIPPER_CLEARANCE, GRIPPER_APPROACH_HEIGHT,
    CAM_TOP_WIDTH, CAM_TOP_HEIGHT, CAM_WRIST_WIDTH, CAM_WRIST_HEIGHT,
    TOP_CAM_X, TOP_CAM_Y, TOP_CAM_Z, TOP_CAM_PITCH,
    C
)
from detect import detect_objects, find_object_pixel, annotate_frame, get_model
from arm_kinematics import forward_kinematics, inverse_kinematics, get_wrist_camera_pose, get_chain
from camera_calibration import load_calibration, pixel_to_table_ray, triangulate_point, TOP_CALIB_FILE, WRIST_CALIB_FILE


# ---- Shared state ----
SLOW_MOVE_STEPS = 8
SLOW_MOVE_DELAY = 0.08


def get_status():
    """Get robot status from server."""
    try:
        r = requests.get(f"{ROBOT_SERVER}/status", timeout=5)
        return r.json()
    except Exception as e:
        print(f"{C.RED}[server]{C.RESET} Error: {e}")
        return None


def get_snapshot(camera="top", full_res=True):
    """Get a camera frame as a numpy array (BGR).
    If full_res=True, requests full resolution from server."""
    try:
        r = requests.get(f"{ROBOT_SERVER}/snapshot/{camera}", timeout=10)
        data = r.json()
        if "error" in data:
            print(f"{C.RED}[camera]{C.RESET} {camera}: {data['error']}")
            return None
        img_bytes = base64.b64decode(data["data"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"{C.RED}[camera]{C.RESET} {camera} error: {e}")
        return None


def move_joints(joints):
    """Send joint positions to the server (hardware space)."""
    try:
        r = requests.post(f"{ROBOT_SERVER}/move", json=joints, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def move_preset(name):
    """Move to a named preset."""
    try:
        r = requests.post(f"{ROBOT_SERVER}/move_preset", json={"pose": name}, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_joint_angles():
    """Get current joint angles in degrees [pan, lift, elbow, flex, roll]."""
    status = get_status()
    if not status or not status.get("joints"):
        return None
    joints = status["joints"]
    # Extract in order, clean key names
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    angles = []
    for name in names:
        for k, v in joints.items():
            if k.replace(".pos", "") == name:
                angles.append(float(v))
                break
        else:
            angles.append(0.0)
    return angles


def move_to_xyz(target_xyz, current_angles=None):
    """Compute IK and move the arm to a 3D position.
    Returns the computed joint angles or None on failure."""
    print(f"{C.BLUE}[ik]{C.RESET} Target: x={target_xyz[0]*100:.1f}cm y={target_xyz[1]*100:.1f}cm z={target_xyz[2]*100:.1f}cm")

    if current_angles is None:
        current_angles = get_joint_angles()

    try:
        angles = inverse_kinematics(target_xyz, current_angles)
    except Exception as e:
        print(f"{C.RED}[ik]{C.RESET} IK failed: {e}")
        return None

    print(f"{C.GREEN}[ik]{C.RESET} Solution: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}")

    # Verify with FK
    verify = forward_kinematics(angles)
    error = np.linalg.norm(np.array(target_xyz) - verify) * 100
    print(f"{C.DIM}[ik] FK verify error: {error:.2f} cm{C.RESET}")
    if error > 5.0:
        print(f"{C.YELLOW}[ik]{C.RESET} Warning: IK error is large ({error:.1f}cm) — solution may be inaccurate")

    # Send to robot (convert to hardware joint names)
    joint_cmd = {
        "shoulder_pan": angles[0],
        "shoulder_lift": angles[1],
        "elbow_flex": angles[2],
        "wrist_flex": angles[3],
        "wrist_roll": angles[4],
    }
    result = move_joints(joint_cmd)
    if "error" in result:
        print(f"{C.RED}[move]{C.RESET} Error: {result['error']}")
        return None
    return angles


def get_top_camera_extrinsics():
    """Get the top camera's position and rotation in world frame."""
    position = np.array([TOP_CAM_X, TOP_CAM_Y, TOP_CAM_Z])
    # Camera looking straight down: Z axis of camera = -Z world
    pitch_rad = np.radians(TOP_CAM_PITCH)
    rotation = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    return position, rotation


def detect_and_locate(target_label=None, use_triangulation=False):
    """Detect the target object and compute its 3D position.

    Phase 1: Table-plane intersection (top camera only, assumes Z=0)
    Phase 2: Triangulation (both cameras, for stacked objects)

    Returns: (x, y, z) in meters, or None.
    """
    # Load top camera calibration
    top_matrix, top_dist, _ = load_calibration(TOP_CALIB_FILE)
    if top_matrix is None:
        print(f"{C.RED}[detect]{C.RESET} Top camera not calibrated! Run:")
        print(f"  python camera_calibration.py --camera top --index <N>")
        return None

    # Get top camera frame
    frame_top = get_snapshot("top")
    if frame_top is None:
        print(f"{C.RED}[detect]{C.RESET} Cannot get top camera frame")
        return None

    # Detect with YOLO
    print(f"{C.BLUE}[detect]{C.RESET} Running YOLO on top camera...")
    detections = detect_objects(frame_top, target_label)
    if not detections:
        print(f"{C.YELLOW}[detect]{C.RESET} No objects detected in top camera")
        return None

    best = detections[0]
    pixel_top = best["center"]
    print(f"{C.GREEN}[detect]{C.RESET} Found: {best['label']} ({best['confidence']:.0%}) at pixel ({pixel_top[0]:.0f}, {pixel_top[1]:.0f})")

    if not use_triangulation:
        # Phase 1: Table-plane intersection
        cam_pos, cam_rot = get_top_camera_extrinsics()
        point = pixel_to_table_ray(pixel_top, top_matrix, top_dist, cam_pos, cam_rot)
        if point is None:
            print(f"{C.RED}[detect]{C.RESET} Ray doesn't intersect table plane")
            return None
        # Set Z to gripper clearance
        point[2] = GRIPPER_CLEARANCE
        print(f"{C.GREEN}[detect]{C.RESET} 3D position (table-plane): x={point[0]*100:.1f}cm y={point[1]*100:.1f}cm z={point[2]*100:.1f}cm")
        return point
    else:
        # Phase 2: Full triangulation with both cameras
        wrist_matrix, wrist_dist, _ = load_calibration(WRIST_CALIB_FILE)
        if wrist_matrix is None:
            print(f"{C.YELLOW}[detect]{C.RESET} Wrist camera not calibrated, falling back to table-plane")
            return detect_and_locate(target_label, use_triangulation=False)

        frame_wrist = get_snapshot("wrist")
        if frame_wrist is None:
            print(f"{C.YELLOW}[detect]{C.RESET} Cannot get wrist frame, falling back to table-plane")
            return detect_and_locate(target_label, use_triangulation=False)

        detections_wrist = detect_objects(frame_wrist, target_label)
        if not detections_wrist:
            print(f"{C.YELLOW}[detect]{C.RESET} No objects in wrist camera, falling back to table-plane")
            return detect_and_locate(target_label, use_triangulation=False)

        pixel_wrist = detections_wrist[0]["center"]
        print(f"{C.GREEN}[detect]{C.RESET} Wrist detection: {detections_wrist[0]['label']} at pixel ({pixel_wrist[0]:.0f}, {pixel_wrist[1]:.0f})")

        # Get wrist camera pose from FK
        current_angles = get_joint_angles()
        if current_angles is None:
            return detect_and_locate(target_label, use_triangulation=False)

        wrist_pos, wrist_rot = get_wrist_camera_pose(current_angles)
        cam_pos_top, cam_rot_top = get_top_camera_extrinsics()

        point = triangulate_point(
            pixel_top, pixel_wrist,
            top_matrix, top_dist, wrist_matrix, wrist_dist,
            cam_pos_top, cam_rot_top, wrist_pos, wrist_rot
        )
        print(f"{C.GREEN}[detect]{C.RESET} 3D position (triangulated): x={point[0]*100:.1f}cm y={point[1]*100:.1f}cm z={point[2]*100:.1f}cm")
        return point


def pickup_sequence(target_label):
    """Full pickup sequence: detect → move → grip → verify → drop."""
    print(f"\n{C.BOLD}{'=' * 40}{C.RESET}")
    print(f"{C.BOLD}Pickup: {C.CYAN}{target_label}{C.RESET}")
    print(f"{C.BOLD}{'=' * 40}{C.RESET}\n")

    # Step 1: Move to home
    print(f"{C.BLUE}[1/6]{C.RESET} Moving to home...")
    move_preset("home")
    time.sleep(1)

    # Step 2: Open gripper
    print(f"{C.BLUE}[2/6]{C.RESET} Opening gripper...")
    move_joints({"gripper": 100})
    time.sleep(0.5)

    # Step 3: Detect and locate
    print(f"{C.BLUE}[3/6]{C.RESET} Detecting {target_label}...")
    target_pos = detect_and_locate(target_label)
    if target_pos is None:
        print(f"{C.RED}[pickup]{C.RESET} Cannot locate target — aborting")
        return False

    # Step 4: Move to approach position (above the target)
    print(f"{C.BLUE}[4/6]{C.RESET} Moving to approach position...")
    approach = target_pos.copy()
    approach[2] = GRIPPER_APPROACH_HEIGHT  # approach from above
    result = move_to_xyz(approach)
    if result is None:
        print(f"{C.RED}[pickup]{C.RESET} Cannot reach approach position — aborting")
        return False
    time.sleep(1)

    # Step 5: Lower to target
    print(f"{C.BLUE}[5/6]{C.RESET} Lowering to target...")
    result = move_to_xyz(target_pos)
    if result is None:
        print(f"{C.RED}[pickup]{C.RESET} Cannot reach target — aborting")
        return False
    time.sleep(0.5)

    # Step 6: Close gripper
    print(f"{C.BLUE}[6/6]{C.RESET} Closing gripper...")
    move_joints({"gripper": 0})
    time.sleep(1)

    # Lift
    print(f"{C.BLUE}[lift]{C.RESET} Lifting...")
    lift_pos = target_pos.copy()
    lift_pos[2] = GRIPPER_APPROACH_HEIGHT
    move_to_xyz(lift_pos)
    time.sleep(0.5)

    # Move to drop
    print(f"{C.BLUE}[drop]{C.RESET} Moving to drop position...")
    move_preset("drop")
    time.sleep(1)

    # Open gripper
    print(f"{C.BLUE}[drop]{C.RESET} Dropping...")
    move_joints({"gripper": 100})
    time.sleep(0.5)

    print(f"\n{C.GREEN}{C.BOLD}Pickup complete!{C.RESET}\n")
    return True


def startup_check():
    """Verify server, arm, cameras, calibration, and YOLO model."""
    print(f"{C.BOLD}Running startup checks...{C.RESET}")
    all_ok = True

    # Server
    status = get_status()
    if status:
        print(f"  {C.GREEN}OK{C.RESET}  Server connected")
        if status.get("robot_connected"):
            print(f"  {C.GREEN}OK{C.RESET}  Arm connected")
        else:
            print(f"  {C.YELLOW}--{C.RESET}  Arm not connected")
            all_ok = False
    else:
        print(f"  {C.RED}FAIL{C.RESET}  Cannot reach server at {ROBOT_SERVER}")
        return False

    # Cameras
    for cam in ["top", "wrist"]:
        frame = get_snapshot(cam)
        if frame is not None:
            print(f"  {C.GREEN}OK{C.RESET}  {cam} camera ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"  {C.RED}FAIL{C.RESET}  {cam} camera")
            all_ok = False

    # Calibration
    top_m, _, _ = load_calibration(TOP_CALIB_FILE)
    if top_m is not None:
        print(f"  {C.GREEN}OK{C.RESET}  Top camera calibrated")
    else:
        print(f"  {C.YELLOW}--{C.RESET}  Top camera NOT calibrated (run camera_calibration.py)")

    wrist_m, _, _ = load_calibration(WRIST_CALIB_FILE)
    if wrist_m is not None:
        print(f"  {C.GREEN}OK{C.RESET}  Wrist camera calibrated")
    else:
        print(f"  {C.DIM}--{C.RESET}  Wrist camera not calibrated (optional, for triangulation)")

    # YOLO
    try:
        get_model()
        print(f"  {C.GREEN}OK{C.RESET}  YOLO model loaded (GPU)")
    except Exception as e:
        print(f"  {C.RED}FAIL{C.RESET}  YOLO model: {e}")
        all_ok = False

    # IKPy
    try:
        get_chain()
        print(f"  {C.GREEN}OK{C.RESET}  IKPy kinematic chain")
    except Exception as e:
        print(f"  {C.RED}FAIL{C.RESET}  IKPy: {e}")
        all_ok = False

    if all_ok:
        print(f"\n  {C.GREEN}{C.BOLD}All checks passed.{C.RESET}\n")
    else:
        print(f"\n  {C.YELLOW}Some checks failed — limited functionality.{C.RESET}\n")
    return all_ok


def print_help():
    print(f"\n{C.BOLD}Commands:{C.RESET}")
    print(f"  {C.CYAN}pick up <object>{C.RESET}     — detect and pick up an object")
    print(f"  {C.CYAN}/detect [label]{C.RESET}       — run YOLO detection on top camera")
    print(f"  {C.CYAN}/locate [label]{C.RESET}       — detect + compute 3D position")
    print(f"  {C.CYAN}/ik <x> <y> <z>{C.RESET}      — move to 3D position (cm)")
    print(f"  {C.CYAN}/fk{C.RESET}                   — show current gripper 3D position")
    print(f"  {C.CYAN}/home{C.RESET}                 — move to home position")
    print(f"  {C.CYAN}/ready{C.RESET}                — move to ready position")
    print(f"  {C.CYAN}/calibrate <cam>{C.RESET}      — run camera calibration")
    print(f"  {C.CYAN}/info{C.RESET}                 — system status")
    print(f"  {C.CYAN}/help{C.RESET}                 — show this help")
    print(f"{C.BOLD}Quit:{C.RESET} quit / exit / q\n")


def run_agent():
    print(f"\n{C.BOLD}{'=' * 50}{C.RESET}")
    print(f"{C.BOLD}SO-101 YOLO + IK Robot Agent{C.RESET}")
    print(f"  Pipeline: YOLO → OpenCV → IKPy → LeRobot")
    print(f"  Server:   {C.CYAN}{ROBOT_SERVER}{C.RESET}")
    print(f"  Type {C.CYAN}/help{C.RESET} for commands, {C.CYAN}quit{C.RESET} to exit")
    print(f"{C.BOLD}{'=' * 50}{C.RESET}\n")

    startup_check()

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

        # Commands
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ("/help", "/h"):
                print_help()

            elif cmd == "/detect":
                label = " ".join(parts[1:]) if len(parts) > 1 else None
                frame = get_snapshot("top")
                if frame is not None:
                    dets = detect_objects(frame, label)
                    if dets:
                        for d in dets[:5]:
                            print(f"  {C.GREEN}{d['label']:15s}{C.RESET} {d['confidence']:.0%}  at ({d['center'][0]:.0f}, {d['center'][1]:.0f})")
                    else:
                        print(f"  {C.YELLOW}No objects detected{C.RESET}")

            elif cmd == "/locate":
                label = " ".join(parts[1:]) if len(parts) > 1 else None
                pos = detect_and_locate(label)
                if pos is not None:
                    print(f"  Position: x={pos[0]*100:.1f}cm y={pos[1]*100:.1f}cm z={pos[2]*100:.1f}cm")

            elif cmd == "/ik":
                if len(parts) != 4:
                    print(f"  Usage: /ik <x_cm> <y_cm> <z_cm>")
                else:
                    try:
                        x, y, z = float(parts[1])/100, float(parts[2])/100, float(parts[3])/100
                        move_to_xyz([x, y, z])
                    except ValueError:
                        print(f"  {C.RED}Invalid coordinates{C.RESET}")

            elif cmd == "/fk":
                angles = get_joint_angles()
                if angles:
                    pos = forward_kinematics(angles)
                    print(f"  Gripper at: x={pos[0]*100:.1f}cm y={pos[1]*100:.1f}cm z={pos[2]*100:.1f}cm")
                    print(f"  Angles: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}")

            elif cmd in ("/home",):
                move_preset("home")
                print(f"  {C.GREEN}Done.{C.RESET}")

            elif cmd in ("/ready",):
                move_preset("ready")
                print(f"  {C.GREEN}Done.{C.RESET}")

            elif cmd == "/calibrate":
                cam = parts[1] if len(parts) > 1 else "top"
                print(f"  Run: python yolo_ik_agent/camera_calibration.py --camera {cam} --index <N>")

            elif cmd in ("/info", "/status"):
                status = get_status()
                if status:
                    print(f"\n{C.BOLD}System Status{C.RESET}")
                    print(f"  Arm: {C.GREEN if status.get('robot_connected') else C.RED}{'connected' if status.get('robot_connected') else 'disconnected'}{C.RESET}")
                    cams = status.get("cameras", [])
                    print(f"  Cameras: {', '.join(cams) if cams else 'none'}")
                    top_m, _, _ = load_calibration(TOP_CALIB_FILE)
                    print(f"  Top calibrated: {C.GREEN}yes{C.RESET}" if top_m is not None else f"  Top calibrated: {C.RED}no{C.RESET}")
                    print()

            else:
                print(f"{C.YELLOW}Unknown command: {cmd}{C.RESET} — type /help")
            continue

        # Natural language — check for pickup
        lower = user_input.lower()
        pickup_kw = ["pick up", "pickup", "grab", "retrieve", "fetch", "collect"]
        if any(kw in lower for kw in pickup_kw):
            # Extract object name
            m = re.search(r'["\u201c](.+?)["\u201d]', user_input)
            if m:
                target = m.group(1).strip()
            else:
                for kw in pickup_kw:
                    idx = lower.find(kw)
                    if idx >= 0:
                        target = user_input[idx + len(kw):].strip()
                        target = re.sub(r'^(the|a|an)\s+', '', target, flags=re.IGNORECASE).strip()
                        break
                else:
                    target = user_input
            if target:
                pickup_sequence(target)
            else:
                print(f"{C.YELLOW}What should I pick up?{C.RESET}")
        else:
            print(f"{C.DIM}(Not a pickup command. Type /help for available commands.){C.RESET}")


if __name__ == "__main__":
    run_agent()
