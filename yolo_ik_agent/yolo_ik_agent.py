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
import re
import time

import cv2
import numpy as np
import requests
from arm_kinematics import forward_kinematics, get_chain, inverse_kinematics
from camera_calibration import pixel_to_table_ray
from cameras import get_extrinsics, get_intrinsics, get_scaled_intrinsics
from config import ROBOT_SERVER, TABLE_Z, URDF_BASE_OFFSET, C
from detect import annotate_frame, detect_and_verify, detect_objects, get_model

# ---- Shared state ----
SLOW_MOVE_STEPS = 20
SLOW_MOVE_DELAY = 0.05

# ---- Arm offset (camera/board frame → arm frame) ----
# [right_m, fwd_m, height_m]: arm's position on the board and height above surface.
# X,Y: subtracted from detected board-frame positions to get arm-frame coords.
# Z: surface level in arm frame = -arm_offset[2] (replaces TABLE_Z / surface.json).
_arm_offset = np.array([0.0, 0.0, 0.0])


def _load_arm_offset():
    """Load arm_offset from cameras.json if present."""
    global _arm_offset
    from cameras import CAMERAS_FILE

    try:
        with open(CAMERAS_FILE) as f:
            data = json.load(f)
        if "arm_offset" in data:
            _arm_offset = np.array(data["arm_offset"][:3], dtype=float)
            print(
                f"{C.GREEN}[calib]{C.RESET} Arm offset: right={_arm_offset[0] * 100:.1f}cm fwd={_arm_offset[1] * 100:.1f}cm up={_arm_offset[2] * 100:.1f}cm"
            )
    except Exception:
        pass


_load_arm_offset()


def get_surface_z():
    """Surface Z in arm frame. Arm is arm_offset[2] above the surface."""
    if _arm_offset[2] != 0:
        return -_arm_offset[2]
    return TABLE_Z


def board_to_arm(pos):
    """Convert board-frame position to arm-frame by subtracting arm offset X,Y.
    Z is NOT transformed — callers set Z separately (e.g. get_surface_z() or stereo Z)."""
    result = pos.copy()
    result[0] -= _arm_offset[0]
    result[1] -= _arm_offset[1]
    return result


def board_to_arm_full(pos):
    """Convert board-frame position to arm-frame, including Z."""
    return pos - _arm_offset


def handle_fk_command(log):
    """Dashboard `/fk` — report gripper pose from current joint angles."""
    angles = get_joint_angles()
    if not angles:
        return
    urdf_pos = forward_kinematics(angles)
    phys = urdf_to_physical(urdf_pos)
    log(
        f"[cmd] Gripper at: right={phys[0] * 100:.1f}cm fwd={phys[1] * 100:.1f}cm up={phys[2] * 100:.1f}cm"
    )
    grip_val = ""
    try:
        status = get_status()
        for k, v in status.get("joints", {}).items():
            if "gripper" in k:
                grip_val = f" gripper={float(v):.1f}"
                break
    except Exception:
        pass
    log(
        f"[cmd] Joints: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}{grip_val}"
    )


def handle_ik_command(args, log):
    """Dashboard `/ik <right_cm> <fwd_cm> <up_cm>` — move arm to a board position."""
    if len(args) != 3:
        log("[cmd] Usage: /ik <right_cm> <fwd_cm> <up_cm>  (board coordinates)")
        return
    try:
        board_pos = np.array([float(v) / 100 for v in args])
    except ValueError:
        log("[cmd] Invalid coordinates")
        return
    arm_pos = board_to_arm_full(board_pos)
    log(
        f"[cmd] Board: right={board_pos[0] * 100:.1f}cm fwd={board_pos[1] * 100:.1f}cm up={board_pos[2] * 100:.1f}cm"
    )
    log(
        f"[cmd] Arm:   right={arm_pos[0] * 100:.1f}cm fwd={arm_pos[1] * 100:.1f}cm up={arm_pos[2] * 100:.1f}cm"
    )
    move_to_xyz(arm_pos)
    log("[cmd] Done")


# ---- Coordinate transforms ----
# Physical world: +X = right, +Y = forward, +Z = up (relative to arm base on table)
# URDF/IKPy:     +X ≈ up,    +Z ≈ forward,  +Y ≈ left
_URDF_BASE = np.array(URDF_BASE_OFFSET)


def physical_to_urdf(xyz):
    """Physical [right, forward, up] → URDF coords for IK."""
    return np.array(
        [
            xyz[1] + _URDF_BASE[0],  # physical forward → URDF +X
            -xyz[0] + _URDF_BASE[1],  # physical right → URDF -Y
            xyz[2] + _URDF_BASE[2],  # physical up → URDF +Z
        ]
    )


def urdf_to_physical(urdf_xyz):
    """URDF coords from FK → physical [right, forward, up]."""
    return np.array(
        [
            -(urdf_xyz[1] - _URDF_BASE[1]),  # URDF -Y → physical right
            urdf_xyz[0] - _URDF_BASE[0],  # URDF +X → physical forward
            urdf_xyz[2] - _URDF_BASE[2],  # URDF +Z → physical up
        ]
    )


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
    "home": {
        "shoulder_pan": 0,
        "shoulder_lift": 0,
        "elbow_flex": 0,
        "wrist_flex": 0,
        "wrist_roll": -90,
        "gripper": 0,
    },
    "ready": {
        "shoulder_pan": 0,
        "shoulder_lift": 40,
        "elbow_flex": 0,
        "wrist_flex": 0,
        "wrist_roll": -90,
        "gripper": 0,
    },
    "default": {
        "shoulder_pan": 0,
        "shoulder_lift": 0,
        "elbow_flex": 0,
        "wrist_flex": 0,
        "wrist_roll": 0,
        "gripper": 0,
    },
    "rest": {
        "shoulder_pan": -1.10,
        "shoulder_lift": -102.24,
        "elbow_flex": 96.57,
        "wrist_flex": 76.35,
        "wrist_roll": -86.02,
        "gripper": 1.20,
    },
    "drop": {
        "shoulder_pan": -47.08,
        "shoulder_lift": 10.46,
        "elbow_flex": -12.44,
        "wrist_flex": 86.20,
        "wrist_roll": -94.64,
        "gripper": 0,
    },
    "observe": {
        "shoulder_pan": -21.5,
        "shoulder_lift": -42.2,
        "elbow_flex": -6.5,
        "wrist_flex": 96.1,
        "wrist_roll": -90,
        "gripper": 100,
    },
    "side_view": {
        "shoulder_pan": -5.5,
        "shoulder_lift": 24.1,
        "elbow_flex": 65.6,
        "wrist_flex": -71.6,
        "wrist_roll": -86.3,
        "gripper": 100,
    },
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


# IK correction: URDF geometry doesn't perfectly match physical arm
# Applied only to IK targets, not camera calculations
IK_OFFSET_RIGHT = -0.02  # 2cm left of object (left jaw is fixed, right jaw opens)
IK_OFFSET_FWD = 0.02  # 1cm forward of object
IK_OFFSET_UP = 0.0


def move_to_xyz(target_xyz, current_angles=None):
    """Compute IK and move the arm to a 3D position (physical coords: right, forward, up).
    Returns the computed joint angles or None on failure."""
    # Apply IK correction offset
    corrected = np.array(target_xyz, dtype=float)
    corrected[0] += IK_OFFSET_RIGHT
    corrected[1] += IK_OFFSET_FWD
    corrected[2] += IK_OFFSET_UP
    print(
        f"{C.BLUE}[ik]{C.RESET} Target (physical): right={target_xyz[0] * 100:.1f}cm fwd={target_xyz[1] * 100:.1f}cm up={target_xyz[2] * 100:.1f}cm"
    )

    # Convert physical world coords to URDF frame for IK
    urdf_target = physical_to_urdf(corrected)
    print(
        f"{C.DIM}[ik] URDF target: x={urdf_target[0] * 100:.1f} y={urdf_target[1] * 100:.1f} z={urdf_target[2] * 100:.1f}{C.RESET}"
    )

    if current_angles is None:
        current_angles = get_joint_angles()

    try:
        angles = inverse_kinematics(urdf_target, current_angles)
    except Exception as e:
        print(f"{C.RED}[ik]{C.RESET} IK failed: {e}")
        return None

    print(
        f"{C.GREEN}[ik]{C.RESET} Solution: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}"
    )

    # Verify with FK
    verify_urdf = forward_kinematics(angles)
    error = np.linalg.norm(urdf_target - verify_urdf) * 100
    print(f"{C.DIM}[ik] FK verify error: {error:.2f} cm{C.RESET}")
    if error > 5.0:
        print(
            f"{C.YELLOW}[ik]{C.RESET} Warning: IK error is large ({error:.1f}cm) — solution may be inaccurate"
        )

    # Move pan first (keep arm raised to avoid knocking objects), then lower
    print(f"{C.DIM}[ik] Moving pan first...{C.RESET}")
    result = _interpolate_move({"shoulder_pan": angles[0]})
    if "error" in result:
        print(f"{C.RED}[move]{C.RESET} Error: {result['error']}")
        return None
    time.sleep(0.2)

    # Now move remaining joints to lower into position
    print(f"{C.DIM}[ik] Lowering to position...{C.RESET}")
    joint_cmd = {
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
    """Get the top camera's position and rotation from cameras.json."""
    return get_extrinsics("top")


def format_position(pos):
    """Format a position for display (board coordinates)."""
    board = pos.copy()
    board[0] += _arm_offset[0]
    board[1] += _arm_offset[1]
    board[2] += _arm_offset[2]
    return f"right={board[0] * 100:.1f}cm fwd={board[1] * 100:.1f}cm up={board[2] * 100:.1f}cm"


def detect_and_locate(target_label=None, use_triangulation=False):
    """Detect the target object and compute its 3D position.

    Phase 1: Table-plane intersection (top camera only, assumes Z=0)
    Phase 2: Triangulation (both cameras, for stacked objects)

    Returns: (x, y, z) in meters, or None.
    """
    # Load top camera calibration
    try:
        top_matrix_raw, top_dist, _ = get_intrinsics("top")
    except Exception:
        print(f"{C.RED}[detect]{C.RESET} Top camera not calibrated! Check cameras.json")
        return None

    # Move to observe first for stereo — clears the arm from blocking the top camera
    if use_triangulation:
        print(f"{C.BLUE}[detect]{C.RESET} Moving to observe position...")
        move_preset("observe")
        time.sleep(1)

    # Get top camera frame
    frame_top = get_snapshot("top")
    if frame_top is None:
        print(f"{C.RED}[detect]{C.RESET} Cannot get top camera frame")
        return None

    # Scale intrinsics to match frame resolution
    frame_h, frame_w = frame_top.shape[:2]
    top_matrix = get_scaled_intrinsics("top", frame_w, frame_h)[0]

    # Detect with GroundingDINO + Gemini verification
    print(f"{C.BLUE}[detect]{C.RESET} Running detection on top camera...")
    if target_label:
        detections, verified_idx = detect_and_verify(frame_top, target_label)
    else:
        detections = detect_objects(frame_top, target_label)
        verified_idx = 0 if detections else None

    if not detections or verified_idx is None:
        print(f"{C.YELLOW}[detect]{C.RESET} No matching objects detected in top camera")
        return None

    best = detections[verified_idx]
    pixel_top = best["center"]
    print(
        f"{C.GREEN}[detect]{C.RESET} Found: {best['label']} ({best['confidence']:.0%}) at pixel ({pixel_top[0]:.0f}, {pixel_top[1]:.0f})"
    )

    # Push annotated detection to stream overlay (top + side)
    overlay_payload = {}
    try:
        annotated_top = annotate_frame(frame_top, detections, target_idx=verified_idx)
        _, buf = cv2.imencode(".jpg", annotated_top, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_payload["top"] = base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    # Side camera detection overlay
    try:
        frame_side = get_snapshot("side")
        if frame_side is not None:
            side_dets = detect_objects(frame_side, target_label)
            annotated_side = annotate_frame(
                frame_side, side_dets, target_idx=0 if side_dets else None
            )
            _, sbuf = cv2.imencode(
                ".jpg", annotated_side, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            overlay_payload["side"] = base64.b64encode(sbuf.tobytes()).decode()
            if side_dets:
                print(
                    f"{C.GREEN}[detect]{C.RESET} Side camera: {side_dets[0]['label']} ({side_dets[0]['confidence']:.0%}) + {len(side_dets) - 1} more"
                )
    except Exception:
        pass
    if overlay_payload:
        try:
            requests.post(
                f"{ROBOT_SERVER}/detection_overlay", json=overlay_payload, timeout=3
            )
        except Exception:
            pass

    if not use_triangulation:
        # Phase 1: Table-plane intersection
        cam_pos, cam_rot = get_top_camera_extrinsics()
        point = pixel_to_table_ray(pixel_top, top_matrix, top_dist, cam_pos, cam_rot)
        if point is None:
            print(f"{C.RED}[detect]{C.RESET} Ray doesn't intersect table plane")
            return None
        # Set Z to calibrated surface or gripper clearance
        point[2] = get_surface_z()  # surface level
        point = board_to_arm(point)
        print(
            f"{C.GREEN}[detect]{C.RESET} 3D position (table-plane): x={point[0] * 100:.1f}cm y={point[1] * 100:.1f}cm z={point[2] * 100:.1f}cm"
        )
        return point
    else:
        # Phase 2: Full triangulation with top + side cameras
        try:
            side_matrix_raw, side_dist, _ = get_intrinsics("side")
        except Exception:
            print(
                f"{C.YELLOW}[detect]{C.RESET} Side camera not calibrated, falling back to table-plane"
            )
            return detect_and_locate(target_label, use_triangulation=False)

        # Step 1: Get X/Y from top camera (already done above)
        cam_pos_top, cam_rot_top = get_top_camera_extrinsics()
        point_top = pixel_to_table_ray(
            pixel_top, top_matrix, top_dist, cam_pos_top, cam_rot_top
        )
        if point_top is None:
            print(f"{C.RED}[detect]{C.RESET} Top camera ray doesn't hit table")
            return None
        print(
            f"{C.DIM}[detect] Top cam X/Y: right={point_top[0] * 100:.1f}cm fwd={point_top[1] * 100:.1f}cm{C.RESET}"
        )

        # Top camera ray (reused for stereo)
        pts_t = np.array([[[pixel_top[0], pixel_top[1]]]], dtype=np.float64)
        undist_t = cv2.undistortPoints(pts_t, top_matrix, top_dist, P=top_matrix)
        ut, vt = undist_t[0, 0]
        fx_t, fy_t = top_matrix[0, 0], top_matrix[1, 1]
        cx_t, cy_t = top_matrix[0, 2], top_matrix[1, 2]
        ray_world_t = cam_rot_top @ np.array(
            [(ut - cx_t) / fx_t, (vt - cy_t) / fy_t, 1.0]
        )
        ray_world_t = ray_world_t / np.linalg.norm(ray_world_t)

        # Step 2: Get side camera frame (fixed camera, no arm movement needed)
        frame_side = get_snapshot("side")
        if frame_side is None:
            print(
                f"{C.YELLOW}[detect]{C.RESET} Cannot get side camera frame, using surface Z"
            )
            point = point_top.copy()
            point[2] = get_surface_z()
            return board_to_arm(point)

        # Scale side intrinsics to match frame
        sh, sw = frame_side.shape[:2]
        side_matrix = get_scaled_intrinsics("side", sw, sh)[0]

        print(f"{C.BLUE}[detect]{C.RESET} Running detection on side camera...")
        detections_side = detect_objects(frame_side, target_label)
        if not detections_side:
            print(
                f"{C.YELLOW}[detect]{C.RESET} No objects in side view, using surface Z"
            )
            point = point_top.copy()
            point[2] = get_surface_z()
            return board_to_arm(point)

        pixel_side = detections_side[0]["center"]
        print(
            f"{C.GREEN}[detect]{C.RESET} Side detection: {detections_side[0]['label']} ({detections_side[0]['confidence']:.0%}) at pixel ({pixel_side[0]:.0f}, {pixel_side[1]:.0f})"
        )

        # Side camera extrinsics from cameras.json
        side_pos, side_rot = get_extrinsics("side")
        print(
            f"{C.DIM}[detect] Side cam: right={side_pos[0] * 100:.1f}cm fwd={side_pos[1] * 100:.1f}cm up={side_pos[2] * 100:.1f}cm{C.RESET}"
        )

        # Side camera ray
        pts_s = np.array([[[pixel_side[0], pixel_side[1]]]], dtype=np.float64)
        undist_s = cv2.undistortPoints(pts_s, side_matrix, side_dist, P=side_matrix)
        us, vs = undist_s[0, 0]
        fx_s, fy_s = side_matrix[0, 0], side_matrix[1, 1]
        cx_s, cy_s = side_matrix[0, 2], side_matrix[1, 2]
        ray_world_s = side_rot @ np.array([(us - cx_s) / fx_s, (vs - cy_s) / fy_s, 1.0])
        ray_world_s = ray_world_s / np.linalg.norm(ray_world_s)

        # Closest point between top ray (downward) and side ray (horizontal) for Z
        w0 = cam_pos_top - side_pos
        b = np.dot(ray_world_t, ray_world_s)
        d = np.dot(ray_world_t, w0)
        e = np.dot(ray_world_s, w0)
        denom = 1.0 - b * b
        if abs(denom) > 1e-6:
            s = (b * e - d) / denom
            t = (e - b * d) / denom
            pt_top_ray = cam_pos_top + s * ray_world_t
            pt_side_ray = side_pos + t * ray_world_s
            midpoint = (pt_top_ray + pt_side_ray) / 2
            ray_dist = np.linalg.norm(pt_top_ray - pt_side_ray)
            z_stereo = midpoint[2]
            print(
                f"{C.DIM}[detect] Stereo Z (side cam): {z_stereo * 100:.1f}cm (ray distance={ray_dist * 100:.1f}cm){C.RESET}"
            )
            if ray_dist > 0.10:
                print(
                    f"{C.YELLOW}[detect]{C.RESET} Warning: rays are {ray_dist * 100:.1f}cm apart"
                )
        else:
            z_stereo = None
            print(f"{C.DIM}[detect] Rays are parallel{C.RESET}")

        # Use stereo Z if reasonable, re-cast top ray at that Z for correct X/Y
        surface_z = get_surface_z()
        z_reasonable = z_stereo is not None and (surface_z - 0.05) <= z_stereo <= 0.20
        if z_reasonable:
            point = pixel_to_table_ray(
                pixel_top,
                top_matrix,
                top_dist,
                cam_pos_top,
                cam_rot_top,
                table_z=z_stereo,
            )
            if point is None:
                point = point_top.copy()
            point[2] = z_stereo
            print(
                f"{C.GREEN}[detect]{C.RESET} 3D position (stereo): right={point[0] * 100:.1f}cm fwd={point[1] * 100:.1f}cm up={point[2] * 100:.1f}cm"
            )
        else:
            point = point_top.copy()
            point[2] = get_surface_z()
            if z_stereo is not None:
                print(
                    f"{C.YELLOW}[detect]{C.RESET} Stereo Z={z_stereo * 100:.1f}cm out of range, using surface level"
                )
            print(
                f"{C.GREEN}[detect]{C.RESET} 3D position (top X/Y, surface Z): right={point[0] * 100:.1f}cm fwd={point[1] * 100:.1f}cm up={point[2] * 100:.1f}cm"
            )
        point = board_to_arm(point)
        return point


def pickup_sequence(target_label, use_stereo=False):
    """Full pickup sequence. If use_stereo, uses both cameras for 3D triangulation."""
    mode = "stereo 3D" if use_stereo else "table-plane"
    print(f"\n{C.BOLD}{'=' * 40}{C.RESET}")
    print(f"{C.BOLD}Pickup ({mode}): {C.CYAN}{target_label}{C.RESET}")
    print(f"{C.BOLD}{'=' * 40}{C.RESET}\n")

    # Step 1: detect_and_locate handles moving to observe for stereo
    if not use_stereo:
        print(f"{C.BLUE}[1/8]{C.RESET} Moving to home...")
        move_preset("home")
        time.sleep(1)

    # Step 2: Detect and locate
    print(f"{C.BLUE}[2/8]{C.RESET} Detecting {target_label}...")
    target_pos = detect_and_locate(target_label, use_triangulation=use_stereo)
    if target_pos is not None:
        print(
            f"{C.GREEN}[pickup]{C.RESET} Target coordinates: x={target_pos[0] * 100:.1f}cm y={target_pos[1] * 100:.1f}cm z={target_pos[2] * 100:.1f}cm"
        )
    if target_pos is None:
        print(f"{C.RED}[pickup]{C.RESET} Cannot locate target — aborting")
        try:
            requests.delete(f"{ROBOT_SERVER}/detection_overlay", timeout=3)
        except Exception:
            pass
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
    print(f"{C.BLUE}[5b/7]{C.RESET} Lifting shoulder 5°...")
    current = _get_current_joints_dict()
    if current:
        cur_lift = current.get("shoulder_lift", 0)
        lift_val = cur_lift - 5
        print(f"{C.DIM}[lift] shoulder_lift: {cur_lift:.1f} → {lift_val:.1f}{C.RESET}")
        _interpolate_move({"shoulder_lift": lift_val}, steps=5, delay=0.04)
    time.sleep(0.3)

    # Step 6: Close gripper
    print(f"{C.BLUE}[6/7]{C.RESET} Closing gripper...")
    _send_joints({"gripper": 0})
    time.sleep(1)

    # Step 7: Lift — just raise shoulder 15 degrees
    print(f"{C.BLUE}[7/7]{C.RESET} Lifting...")
    current = _get_current_joints_dict()
    if current:
        cur_lift = current.get("shoulder_lift", 0)
        lift_val = cur_lift - 15
        print(f"{C.DIM}[lift] shoulder_lift: {cur_lift:.1f} → {lift_val:.1f}{C.RESET}")
        _interpolate_move({"shoulder_lift": lift_val}, steps=10, delay=0.05)
    time.sleep(0.3)

    # Move to drop
    print(f"{C.BLUE}[drop]{C.RESET} Moving to drop position...")
    move_preset("drop")
    time.sleep(1)

    # Open gripper
    print(f"{C.BLUE}[drop]{C.RESET} Dropping...")
    _send_joints({"gripper": 100})
    time.sleep(0.5)

    # Clear detection overlay
    try:
        requests.delete(f"{ROBOT_SERVER}/detection_overlay", timeout=3)
    except Exception:
        pass

    print(f"\n{C.GREEN}{C.BOLD}Pickup complete!{C.RESET}\n")
    return target_pos


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
    for cam in ["top", "side"]:
        frame = get_snapshot(cam)
        if frame is not None:
            print(
                f"  {C.GREEN}OK{C.RESET}  {cam} camera ({frame.shape[1]}x{frame.shape[0]})"
            )
        else:
            print(f"  {C.RED}FAIL{C.RESET}  {cam} camera")
            all_ok = False

    # Calibration
    try:
        top_m, _, _ = get_intrinsics("top")
        print(f"  {C.GREEN}OK{C.RESET}  Top camera calibrated")
    except Exception:
        print(
            f"  {C.YELLOW}--{C.RESET}  Top camera NOT calibrated (check cameras.json)"
        )

    try:
        side_m, _, _ = get_intrinsics("side")
        print(f"  {C.GREEN}OK{C.RESET}  Side camera calibrated")
    except Exception:
        print(
            f"  {C.DIM}--{C.RESET}  Side camera not calibrated (optional, for triangulation)"
        )

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
    print(
        f"  {C.CYAN}pick up <object>{C.RESET}     — pick up using top camera (table-plane)"
    )
    print(
        f"  {C.CYAN}/pick3d <object>{C.RESET}    — pick up using stereo 3D (both cameras)"
    )
    print(
        f"  {C.CYAN}/detect [label]{C.RESET}       — run YOLO detection on top camera"
    )
    print(
        f"  {C.CYAN}/locate [label]{C.RESET}       — detect + 3D position (table-plane)"
    )
    print(
        f"  {C.CYAN}/locate3d [label]{C.RESET}    — detect + 3D position (stereo triangulation)"
    )
    print(f"  {C.CYAN}http://localhost:7878/stream{C.RESET} — YOLO debug viewer")
    print(
        f"  {C.CYAN}/ik <r> <f> <u>{C.RESET}      — move to position (right, fwd, up in cm)"
    )
    print(
        f"  {C.CYAN}/fk{C.RESET}                   — show current gripper 3D position"
    )
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
    print("  Pipeline: YOLO → OpenCV → IKPy → LeRobot")
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

            elif cmd == "/pick":
                label = " ".join(parts[1:]) if len(parts) > 1 else None
                if label:
                    pickup_sequence(label)
                else:
                    print(f"{C.YELLOW}Usage: /pick <object>{C.RESET}")

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
                            print(
                                f"  {C.GREEN}{d['label']:15s}{C.RESET} {d['confidence']:.0%}  at ({d['center'][0]:.0f}, {d['center'][1]:.0f})"
                            )
                    else:
                        print(f"  {C.YELLOW}No objects detected{C.RESET}")

            elif cmd == "/locate":
                label = " ".join(parts[1:]) if len(parts) > 1 else None
                pos = detect_and_locate(label)
                if pos is not None:
                    print(
                        f"  Position: right={pos[0] * 100:.1f}cm fwd={pos[1] * 100:.1f}cm up={pos[2] * 100:.1f}cm"
                    )

            elif cmd == "/locate3d":
                label = " ".join(parts[1:]) if len(parts) > 1 else None
                pos = detect_and_locate(label, use_triangulation=True)
                if pos is not None:
                    print(
                        f"  Position (stereo): right={pos[0] * 100:.1f}cm fwd={pos[1] * 100:.1f}cm up={pos[2] * 100:.1f}cm"
                    )

            elif cmd == "/ik":
                if len(parts) != 4:
                    print(
                        "  Usage: /ik <right_cm> <fwd_cm> <up_cm>  (board coordinates)"
                    )
                else:
                    try:
                        board_pos = np.array(
                            [
                                float(parts[1]) / 100,
                                float(parts[2]) / 100,
                                float(parts[3]) / 100,
                            ]
                        )
                        arm_pos = board_to_arm_full(board_pos)
                        print(
                            f"  Board: right={board_pos[0] * 100:.1f}cm fwd={board_pos[1] * 100:.1f}cm up={board_pos[2] * 100:.1f}cm"
                        )
                        print(
                            f"  Arm:   right={arm_pos[0] * 100:.1f}cm fwd={arm_pos[1] * 100:.1f}cm up={arm_pos[2] * 100:.1f}cm"
                        )
                        move_to_xyz(arm_pos)
                    except ValueError:
                        print(f"  {C.RED}Invalid coordinates{C.RESET}")

            elif cmd == "/fk":
                angles = get_joint_angles()
                if angles:
                    urdf_pos = forward_kinematics(angles)
                    phys = urdf_to_physical(urdf_pos)
                    print(
                        f"  Gripper at: right={phys[0] * 100:.1f}cm fwd={phys[1] * 100:.1f}cm up={phys[2] * 100:.1f}cm"
                    )
                    print(
                        f"  Angles: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}"
                    )

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
                    r = requests.post(
                        f"{ROBOT_SERVER}/enable", json={"enabled": enabled}, timeout=5
                    )
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
                print(
                    f"  Run: python yolo_ik_agent/camera_calibration.py --camera {cam} --index <N>"
                )

            elif cmd == "/pos":
                label = " ".join(parts[1:]) if len(parts) > 1 else None
                if not label:
                    print(f"{C.YELLOW}Usage: /pos <object>{C.RESET}")
                else:
                    print(f"{C.BLUE}[pos]{C.RESET} Detecting {label}...")
                    pos = detect_and_locate(label, use_triangulation=True)
                    if pos is not None:
                        print(
                            f"{C.GREEN}[pos]{C.RESET} Object at: right={pos[0] * 100:.1f}cm fwd={pos[1] * 100:.1f}cm up={pos[2] * 100:.1f}cm"
                        )
                        print(f"{C.BLUE}[pos]{C.RESET} Moving to home first...")
                        move_preset("home")
                        time.sleep(0.5)
                        print(f"{C.BLUE}[pos]{C.RESET} Opening gripper...")
                        _send_joints({"gripper": 100})
                        time.sleep(0.5)
                        print(f"{C.BLUE}[pos]{C.RESET} Moving to target...")
                        move_to_xyz(pos)
                        print(
                            f"{C.GREEN}[pos]{C.RESET} Done — gripper is at target. Check alignment."
                        )
                    else:
                        print(f"{C.RED}[pos]{C.RESET} Object not found")

            elif cmd in ("/info", "/status"):
                status = get_status()
                if status:
                    print(f"\n{C.BOLD}System Status{C.RESET}")
                    print(
                        f"  Arm: {C.GREEN if status.get('robot_connected') else C.RED}{'connected' if status.get('robot_connected') else 'disconnected'}{C.RESET}"
                    )
                    cams = status.get("cameras", [])
                    print(f"  Cameras: {', '.join(cams) if cams else 'none'}")
                    try:
                        get_intrinsics("top")
                        print(f"  Top calibrated: {C.GREEN}yes{C.RESET}")
                    except Exception:
                        print(f"  Top calibrated: {C.RED}no{C.RESET}")
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
                        target = user_input[idx + len(kw) :].strip()
                        target = re.sub(
                            r"^(the|a|an)\s+", "", target, flags=re.IGNORECASE
                        ).strip()
                        break
                else:
                    target = user_input
            if target:
                pickup_sequence(target)
            else:
                print(f"{C.YELLOW}What should I pick up?{C.RESET}")
        else:
            print(
                f"{C.DIM}(Not a pickup command. Type /help for available commands.){C.RESET}"
            )


if __name__ == "__main__":
    run_agent()
