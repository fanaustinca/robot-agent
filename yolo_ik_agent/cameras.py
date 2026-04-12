"""
Camera configuration loader. Reads all camera parameters from cameras.json.
"""

import json
import os
import numpy as np

CAMERAS_FILE = os.path.join(os.path.dirname(__file__), "calibration_data", "cameras.json")

_cameras = None


def _load():
    global _cameras
    if _cameras is not None:
        return _cameras
    with open(CAMERAS_FILE) as f:
        _cameras = json.load(f)
    return _cameras


def get_camera(name):
    """Get all parameters for a camera by name ('top' or 'side').
    Returns dict with: resolution, camera_matrix, dist_coeffs, position, yaw, pitch."""
    cams = _load()
    if name not in cams:
        raise ValueError(f"Unknown camera '{name}'. Available: {list(cams.keys())}")
    return cams[name]


def get_intrinsics(name):
    """Get camera intrinsic matrix and distortion coefficients.
    Returns (camera_matrix, dist_coeffs, resolution)."""
    cam = get_camera(name)
    matrix = np.array(cam["camera_matrix"])
    dist = np.array(cam["dist_coeffs"])
    resolution = tuple(cam["resolution"])
    return matrix, dist, resolution


def get_extrinsics(name):
    """Get camera position and rotation matrix in physical world frame.
    Position: [right, forward, up] in meters from arm base.
    Rotation: 3x3 matrix mapping camera axes to world axes."""
    cam = get_camera(name)
    position = np.array(cam["position"])

    yaw_r = np.radians(cam["yaw"])
    pitch_r = np.radians(cam["pitch"])

    # Compute look-at rotation from yaw/pitch
    forward = np.array([
        np.sin(yaw_r) * np.cos(pitch_r),
        np.cos(yaw_r) * np.cos(pitch_r),
        np.sin(pitch_r)
    ])
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Columns: camera +X (right), camera +Y (down), camera +Z (forward)
    rotation = np.column_stack([right, down, forward])
    return position, rotation


def get_scaled_intrinsics(name, frame_width, frame_height):
    """Get intrinsics scaled to match a specific frame resolution."""
    matrix, dist, (calib_w, calib_h) = get_intrinsics(name)
    if frame_width != calib_w or frame_height != calib_h:
        sx, sy = frame_width / calib_w, frame_height / calib_h
        matrix = matrix.copy()
        matrix[0, 0] *= sx
        matrix[0, 2] *= sx
        matrix[1, 1] *= sy
        matrix[1, 2] *= sy
    return matrix, dist


def get_resolution(name, default=(640, 480)):
    """Return (width, height) for a camera, or default if missing."""
    try:
        cam = get_camera(name)
    except (ValueError, FileNotFoundError):
        return default
    res = cam.get("resolution")
    return tuple(res) if res else default


def get_focus(name):
    """Return saved focus value for a camera, or None if unset."""
    try:
        cam = get_camera(name)
    except (ValueError, FileNotFoundError):
        return None
    return cam.get("focus")


def get_arm_offset():
    """Return arm_offset list [x, y, z] (padded to length 3, zeros if missing)."""
    try:
        cams = _load()
    except FileNotFoundError:
        return [0.0, 0.0, 0.0]
    offset = list(cams.get("arm_offset", [0.0, 0.0, 0.0]))
    while len(offset) < 3:
        offset.append(0.0)
    return offset


def set_focus(name, focus_value):
    """Persist a camera's focus value to cameras.json."""
    update_camera(name, focus=focus_value)


def set_arm_offset(xy_or_xyz):
    """Persist arm_offset to cameras.json. Accepts [x,y] or [x,y,z]."""
    cams = _load()
    vals = [float(v) for v in xy_or_xyz]
    while len(vals) < 3:
        vals.append(0.0)
    cams["arm_offset"] = vals
    with open(CAMERAS_FILE, 'w') as f:
        json.dump(cams, f, indent=2)
        f.write("\n")


def update_camera(name, **kwargs):
    """Update camera parameters and save to cameras.json.
    Example: update_camera('side', position=[-0.30, 0.32, 0.14], pitch=-14)"""
    cams = _load()
    if name not in cams:
        raise ValueError(f"Unknown camera '{name}'")
    cams[name].update(kwargs)
    with open(CAMERAS_FILE, 'w') as f:
        json.dump(cams, f, indent=2)
    print(f"Updated {name} camera: {list(kwargs.keys())}")


def reload():
    """Force reload from file."""
    global _cameras
    _cameras = None
    _load()
