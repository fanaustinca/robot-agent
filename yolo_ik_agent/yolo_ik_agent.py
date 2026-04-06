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
    URDF_BASE_OFFSET, C
)
from detect import detect_objects, find_object_pixel, annotate_frame, get_model
from arm_kinematics import forward_kinematics, inverse_kinematics, get_wrist_camera_pose, get_chain
from camera_calibration import load_calibration, pixel_to_table_ray, triangulate_point, TOP_CALIB_FILE, WRIST_CALIB_FILE


# ---- Shared state ----
SLOW_MOVE_STEPS = 20
SLOW_MOVE_DELAY = 0.05

# ---- Coordinate transforms ----
# Physical world: +X = right, +Y = forward, +Z = up (relative to arm base on table)
# URDF/IKPy:     +X ≈ up,    +Z ≈ forward,  +Y ≈ left
_URDF_BASE = np.array(URDF_BASE_OFFSET)

def physical_to_urdf(xyz):
    """Physical [right, forward, up] → URDF coords for IK."""
    return np.array([
        xyz[1] + _URDF_BASE[0],   # physical forward → URDF +X
        -xyz[0] + _URDF_BASE[1],  # physical right → URDF -Y
        xyz[2] + _URDF_BASE[2],   # physical up → URDF +Z
    ])

def urdf_to_physical(urdf_xyz):
    """URDF coords from FK → physical [right, forward, up]."""
    return np.array([
        -(urdf_xyz[1] - _URDF_BASE[1]),  # URDF -Y → physical right
        urdf_xyz[0] - _URDF_BASE[0],     # URDF +X → physical forward
        urdf_xyz[2] - _URDF_BASE[2],     # URDF +Z → physical up
    ])


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


def _send_joints(joints):
    """Send joint positions to the server directly (no interpolation)."""
    try:
        r = requests.post(f"{ROBOT_SERVER}/move", json=joints, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


PRESETS = {
    "home":    {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90, "gripper": 0},
    "ready":   {"shoulder_pan": 0, "shoulder_lift": 40, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90, "gripper": 0},
    "default": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0},
    "rest":    {"shoulder_pan": -1.10, "shoulder_lift": -102.24, "elbow_flex": 96.57, "wrist_flex": 76.35, "wrist_roll": -86.02, "gripper": 1.20},
    "drop":    {"shoulder_pan": -47.08, "shoulder_lift": 10.46, "elbow_flex": -12.44, "wrist_flex": 86.20, "wrist_roll": -94.64, "gripper": 0},
    "observe": {"shoulder_pan": -21.5, "shoulder_lift": -42.2, "elbow_flex": -6.5, "wrist_flex": 96.1, "wrist_roll": -90, "gripper": 100},
}


def _get_current_joints_dict():
    """Get current joint positions as a dict."""
    status = get_status()
    if not status or not status.get("joints"):
        return None
    joints = status["joints"]
    result = {}
    for k, v in joints.items():
        clean = k.replace(".pos", "")
        result[clean] = float(v)
    return result


def _interpolate_move(target_joints, steps=None, delay=None):
    """Smoothly interpolate from current position to target joints."""
    if steps is None:
        steps = SLOW_MOVE_STEPS
    if delay is None:
        delay = SLOW_MOVE_DELAY

    current = _get_current_joints_dict()
    if current is None:
        return _send_joints(target_joints)

    for step in range(1, steps + 1):
        t = step / steps
        interp = {}
        for joint, target_val in target_joints.items():
            cur_val = current.get(joint, target_val)
            interp[joint] = cur_val + (target_val - cur_val) * t
        result = _send_joints(interp)
        if "error" in result:
            return result
        time.sleep(delay)
    return {"ok": True}


def move_joints(joints):
    """Send joint positions to the server with smooth interpolation."""
    return _interpolate_move(joints)


def move_preset(name):
    """Move to a named preset with smooth interpolation."""
    if name not in PRESETS:
        return {"error": f"Unknown preset: {name}"}
    return _interpolate_move(PRESETS[name])


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
    """Compute IK and move the arm to a 3D position (physical coords: right, forward, up).
    Returns the computed joint angles or None on failure."""
    print(f"{C.BLUE}[ik]{C.RESET} Target (physical): right={target_xyz[0]*100:.1f}cm fwd={target_xyz[1]*100:.1f}cm up={target_xyz[2]*100:.1f}cm")

    # Convert physical world coords to URDF frame for IK
    urdf_target = physical_to_urdf(target_xyz)
    print(f"{C.DIM}[ik] URDF target: x={urdf_target[0]*100:.1f} y={urdf_target[1]*100:.1f} z={urdf_target[2]*100:.1f}{C.RESET}")

    if current_angles is None:
        current_angles = get_joint_angles()

    try:
        angles = inverse_kinematics(urdf_target, current_angles)
    except Exception as e:
        print(f"{C.RED}[ik]{C.RESET} IK failed: {e}")
        return None

    print(f"{C.GREEN}[ik]{C.RESET} Solution: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}")

    # Verify with FK
    verify_urdf = forward_kinematics(angles)
    error = np.linalg.norm(urdf_target - verify_urdf) * 100
    print(f"{C.DIM}[ik] FK verify error: {error:.2f} cm{C.RESET}")
    if error > 5.0:
        print(f"{C.YELLOW}[ik]{C.RESET} Warning: IK error is large ({error:.1f}cm) — solution may be inaccurate")

    # Lock wrist flex at 30 degrees (gripper angled down)
    angles[3] = 30.0
    # Lock wrist roll at -90 degrees
    angles[4] = -90.0

    # Send to robot with smooth interpolation
    joint_cmd = {
        "shoulder_pan": angles[0],
        "shoulder_lift": angles[1],
        "elbow_flex": angles[2],
        "wrist_flex": angles[3],
        "wrist_roll": angles[4],
    }
    result = _interpolate_move(joint_cmd)
    if "error" in result:
        print(f"{C.RED}[move]{C.RESET} Error: {result['error']}")
        return None
    return angles


def get_top_camera_extrinsics():
    """Get the top camera's position and rotation in physical world frame.
    Physical world: +X = right, +Y = forward, +Z = up.
    Camera: +Z = into scene, +X = right in image, +Y = down in image.
    Camera looks along +Y (forward) and pitched down by TOP_CAM_PITCH."""
    position = np.array([TOP_CAM_X, TOP_CAM_Y, TOP_CAM_Z])
    p = np.radians(TOP_CAM_PITCH)  # negative = looking downward
    s, c = np.sin(p), np.cos(p)
    # cam +Z → physical [0, c, s] (forward + down)
    # cam +X → physical [1, 0, 0] (image right = physical right)
    # cam +Y → physical [0, -s, c] ... computed as cross(Z,X)→ [0*s-s*0, s*1-0*0, 0*0-c*1] nah let me just derive:
    # cam +Y = (cam +Z) × (cam +X) in physical frame:
    #   [0,c,s] × [1,0,0] = [c*0-s*0, s*1-0*0, 0*0-c*1] = [0, s, -c]
    rotation = np.array([
        [1,  0,  0],
        [0,  s,  c],
        [0, -c,  s],
    ])
    return position, rotation


def detect_and_locate(target_label=None, use_triangulation=False):
    """Detect the target object and compute its 3D position.

    Phase 1: Table-plane intersection (top camera only, assumes Z=0)
    Phase 2: Triangulation (both cameras, for stacked objects)

    Returns: (x, y, z) in meters, or None.
    """
    # Load top camera calibration
    top_matrix, top_dist, calib_res = load_calibration(TOP_CALIB_FILE)
    if top_matrix is None:
        print(f"{C.RED}[detect]{C.RESET} Top camera not calibrated! Run:")
        print(f"  python camera_calibration.py --camera top --index <N>")
        return None

    # Get top camera frame
    frame_top = get_snapshot("top")
    if frame_top is None:
        print(f"{C.RED}[detect]{C.RESET} Cannot get top camera frame")
        return None

    # Scale calibration matrix if frame resolution differs from calibration resolution
    if calib_res is not None:
        frame_h, frame_w = frame_top.shape[:2]
        calib_w, calib_h = calib_res
        if frame_w != calib_w or frame_h != calib_h:
            sx = frame_w / calib_w
            sy = frame_h / calib_h
            top_matrix = top_matrix.copy()
            top_matrix[0, 0] *= sx  # fx
            top_matrix[0, 2] *= sx  # cx
            top_matrix[1, 1] *= sy  # fy
            top_matrix[1, 2] *= sy  # cy
            print(f"{C.DIM}[detect] Scaled calibration {calib_w}x{calib_h} → {frame_w}x{frame_h}{C.RESET}")

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
        wrist_matrix, wrist_dist, wrist_calib_res = load_calibration(WRIST_CALIB_FILE)
        if wrist_matrix is None:
            print(f"{C.YELLOW}[detect]{C.RESET} Wrist camera not calibrated, falling back to table-plane")
            return detect_and_locate(target_label, use_triangulation=False)

        # Read joint angles BEFORE getting wrist frame (arm must be stationary)
        current_angles = get_joint_angles()
        if current_angles is None:
            print(f"{C.YELLOW}[detect]{C.RESET} Cannot read joint angles, falling back to table-plane")
            return detect_and_locate(target_label, use_triangulation=False)

        frame_wrist = get_snapshot("wrist")
        if frame_wrist is None:
            print(f"{C.YELLOW}[detect]{C.RESET} Cannot get wrist frame, falling back to table-plane")
            return detect_and_locate(target_label, use_triangulation=False)

        # Scale wrist calibration if resolution differs
        if wrist_calib_res is not None:
            wh, ww = frame_wrist.shape[:2]
            cw, ch = wrist_calib_res
            if ww != cw or wh != ch:
                sx, sy = ww / cw, wh / ch
                wrist_matrix = wrist_matrix.copy()
                wrist_matrix[0, 0] *= sx
                wrist_matrix[0, 2] *= sx
                wrist_matrix[1, 1] *= sy
                wrist_matrix[1, 2] *= sy
                print(f"{C.DIM}[detect] Scaled wrist calibration {cw}x{ch} → {ww}x{wh}{C.RESET}")

        print(f"{C.BLUE}[detect]{C.RESET} Running detection on wrist camera...")
        detections_wrist = detect_objects(frame_wrist, target_label)
        if not detections_wrist:
            print(f"{C.YELLOW}[detect]{C.RESET} No objects in wrist camera, falling back to table-plane")
            return detect_and_locate(target_label, use_triangulation=False)

        pixel_wrist = detections_wrist[0]["center"]
        print(f"{C.GREEN}[detect]{C.RESET} Wrist detection: {detections_wrist[0]['label']} ({detections_wrist[0]['confidence']:.0%}) at pixel ({pixel_wrist[0]:.0f}, {pixel_wrist[1]:.0f})")

        # Get camera poses in physical world frame
        wrist_pos, wrist_rot = get_wrist_camera_pose(current_angles)
        cam_pos_top, cam_rot_top = get_top_camera_extrinsics()

        print(f"{C.DIM}[detect] Top cam pos: right={cam_pos_top[0]*100:.1f} fwd={cam_pos_top[1]*100:.1f} up={cam_pos_top[2]*100:.1f}{C.RESET}")
        print(f"{C.DIM}[detect] Wrist cam pos: right={wrist_pos[0]*100:.1f} fwd={wrist_pos[1]*100:.1f} up={wrist_pos[2]*100:.1f}{C.RESET}")

        point = triangulate_point(
            pixel_top, pixel_wrist,
            top_matrix, top_dist, wrist_matrix, wrist_dist,
            cam_pos_top, cam_rot_top, wrist_pos, wrist_rot
        )

        # Use triangulated X/Y but clamp Z to table level
        # (Z from stereo is unreliable until wrist extrinsics are perfectly calibrated)
        print(f"{C.DIM}[detect] Raw triangulated: right={point[0]*100:.1f}cm fwd={point[1]*100:.1f}cm up={point[2]*100:.1f}cm{C.RESET}")
        point[2] = GRIPPER_CLEARANCE
        print(f"{C.GREEN}[detect]{C.RESET} 3D position (stereo): right={point[0]*100:.1f}cm fwd={point[1]*100:.1f}cm up={point[2]*100:.1f}cm")
        return point


def pickup_sequence(target_label, use_stereo=False):
    """Full pickup sequence. If use_stereo, uses both cameras for 3D triangulation."""
    mode = "stereo 3D" if use_stereo else "table-plane"
    print(f"\n{C.BOLD}{'=' * 40}{C.RESET}")
    print(f"{C.BOLD}Pickup ({mode}): {C.CYAN}{target_label}{C.RESET}")
    print(f"{C.BOLD}{'=' * 40}{C.RESET}\n")

    if use_stereo:
        # Step 1: Move to observe position (wrist camera can see the workspace)
        print(f"{C.BLUE}[1/8]{C.RESET} Moving to observe position...")
        move_preset("observe")
        time.sleep(1)
    else:
        print(f"{C.BLUE}[1/8]{C.RESET} Moving to home...")
        move_preset("home")
        time.sleep(1)

    # Step 2: Detect and locate
    print(f"{C.BLUE}[2/8]{C.RESET} Detecting {target_label}...")
    target_pos = detect_and_locate(target_label, use_triangulation=use_stereo)
    if target_pos is None:
        print(f"{C.RED}[pickup]{C.RESET} Cannot locate target — aborting")
        move_preset("home")
        return False

    # Step 3: Move to home before approaching
    print(f"{C.BLUE}[3/8]{C.RESET} Moving to home...")
    move_preset("home")
    time.sleep(0.5)

    # Step 4: Open gripper
    print(f"{C.BLUE}[4/8]{C.RESET} Opening gripper...")
    _send_joints({"gripper": 100})
    time.sleep(0.5)

    # Step 5: Move to target position
    print(f"{C.BLUE}[5/7]{C.RESET} Moving to target...")
    result = move_to_xyz(target_pos)
    if result is None:
        print(f"{C.RED}[pickup]{C.RESET} Cannot reach target — aborting")
        return False
    time.sleep(0.3)

    # Lift shoulder 5 degrees to keep gripper off the floor
    current = _get_current_joints_dict()
    if current:
        lift_val = current.get("shoulder_lift", 0) - 8
        _interpolate_move({"shoulder_lift": lift_val}, steps=5, delay=0.04)
    time.sleep(0.3)

    # Step 6: Close gripper
    print(f"{C.BLUE}[6/7]{C.RESET} Closing gripper...")
    _send_joints({"gripper": 0})
    time.sleep(1)

    # Step 7: Lift
    print(f"{C.BLUE}[7/7]{C.RESET} Lifting...")
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
    _send_joints({"gripper": 100})
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
    print(f"  {C.CYAN}pick up <object>{C.RESET}     — pick up using top camera (table-plane)")
    print(f"  {C.CYAN}/pick3d <object>{C.RESET}    — pick up using stereo 3D (both cameras)")
    print(f"  {C.CYAN}/detect [label]{C.RESET}       — run YOLO detection on top camera")
    print(f"  {C.CYAN}/locate [label]{C.RESET}       — detect + 3D position (table-plane)")
    print(f"  {C.CYAN}/locate3d [label]{C.RESET}    — detect + 3D position (stereo triangulation)")
    print(f"  {C.CYAN}http://localhost:7878/stream{C.RESET} — YOLO debug viewer")
    print(f"  {C.CYAN}/ik <r> <f> <u>{C.RESET}      — move to position (right, fwd, up in cm)")
    print(f"  {C.CYAN}/fk{C.RESET}                   — show current gripper 3D position")
    print(f"  {C.CYAN}/t [on|off]{C.RESET}           — toggle torque")
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

            elif cmd == "/pick3d":
                label = " ".join(parts[1:]) if len(parts) > 1 else None
                if label:
                    pickup_sequence(label, use_stereo=True)
                else:
                    print(f"{C.YELLOW}Usage: /pick3d <object>{C.RESET}")

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
                    print(f"  Position: right={pos[0]*100:.1f}cm fwd={pos[1]*100:.1f}cm up={pos[2]*100:.1f}cm")

            elif cmd == "/locate3d":
                label = " ".join(parts[1:]) if len(parts) > 1 else None
                pos = detect_and_locate(label, use_triangulation=True)
                if pos is not None:
                    print(f"  Position (stereo): right={pos[0]*100:.1f}cm fwd={pos[1]*100:.1f}cm up={pos[2]*100:.1f}cm")

            elif cmd == "/ik":
                if len(parts) != 4:
                    print(f"  Usage: /ik <right_cm> <fwd_cm> <up_cm>")
                else:
                    try:
                        x, y, z = float(parts[1])/100, float(parts[2])/100, float(parts[3])/100
                        move_to_xyz([x, y, z])
                    except ValueError:
                        print(f"  {C.RED}Invalid coordinates{C.RESET}")

            elif cmd == "/fk":
                angles = get_joint_angles()
                if angles:
                    urdf_pos = forward_kinematics(angles)
                    phys = urdf_to_physical(urdf_pos)
                    print(f"  Gripper at: right={phys[0]*100:.1f}cm fwd={phys[1]*100:.1f}cm up={phys[2]*100:.1f}cm")
                    print(f"  Angles: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}")

            elif cmd in ("/t", "/torque"):
                # Toggle or set torque
                if len(parts) > 1 and parts[1].lower() in ("on", "1"):
                    enabled = True
                elif len(parts) > 1 and parts[1].lower() in ("off", "0"):
                    enabled = False
                else:
                    # Toggle: check current state
                    status = get_status()
                    enabled = not status.get("torque", True) if status else False
                try:
                    r = requests.post(f"{ROBOT_SERVER}/enable", json={"enabled": enabled}, timeout=5)
                    state = "ON" if enabled else "OFF"
                    color = C.GREEN if enabled else C.YELLOW
                    print(f"  {color}Torque {state}{C.RESET}")
                except Exception as e:
                    print(f"  {C.RED}Error: {e}{C.RESET}")

            elif cmd in ("/home",):
                move_preset("home")
                print(f"  {C.GREEN}Done.{C.RESET}")

            elif cmd in ("/observe",):
                move_preset("observe")
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
