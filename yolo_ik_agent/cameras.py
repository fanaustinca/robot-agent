"""
Camera configuration loader. Reads all camera parameters from cameras.json.
"""

import glob
import json
import os
import numpy as np

_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"


def find_index_by_name(name):
    """Find the first /dev/videoX index whose device name contains the given string."""
    for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            with open(path) as f:
                dev_name = f.read().strip()
            if name.lower() in dev_name.lower():
                idx = int(path.split("/video")[2].split("/")[0])
                print(f"{_GREEN}[cameras]{_RESET} Found '{dev_name}' at index {idx}")
                return idx
        except Exception:
            continue
    return None


CAMERAS_FILE = os.path.join(
    os.path.dirname(__file__), "calibration_data", "cameras.json"
)

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
    forward = np.array(
        [
            np.sin(yaw_r) * np.cos(pitch_r),
            np.cos(yaw_r) * np.cos(pitch_r),
            np.sin(pitch_r),
        ]
    )
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


def autofocus_cameras(captures, log):
    """Sweep focus 0-255 on each open capture, pick the sharpest, and persist.

    `captures` is a dict of `{name: cv2.VideoCapture}`. `log` is a callback
    used to stream progress back to the caller (e.g. the dashboard log).
    """
    import cv2

    step = 5
    settle_frames = 4

    def measure_sharpness(cap):
        ret, frame = cap.read()
        if not ret or frame is None:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    focus_results = {}
    for cam_name, cap in captures.items():
        if not (cap and cap.isOpened()):
            continue
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        log(f"[cmd] {cam_name}: sweeping focus 0-255 (step={step})...")
        best_focus = 0
        best_sharpness = 0.0
        for fv in range(0, 256, step):
            cap.set(cv2.CAP_PROP_FOCUS, fv)
            for _ in range(settle_frames):
                cap.read()
            sharpness = measure_sharpness(cap)
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_focus = fv
        fine_lo = max(0, best_focus - step)
        fine_hi = min(255, best_focus + step)
        for fv in range(fine_lo, fine_hi + 1):
            cap.set(cv2.CAP_PROP_FOCUS, fv)
            for _ in range(settle_frames):
                cap.read()
            sharpness = measure_sharpness(cap)
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_focus = fv
        cap.set(cv2.CAP_PROP_FOCUS, best_focus)
        focus_results[cam_name] = best_focus
        log(
            f"[cmd] {cam_name}: best focus={best_focus} (sharpness={best_sharpness:.0f})"
        )

    try:
        for cam_name, focus_val in focus_results.items():
            set_focus(cam_name, focus_val)
        log("[cmd] Saved focus values to cameras.json")
    except Exception as e:
        log(f"[cmd] Warning: could not save to cameras.json: {e}")
    log("[cmd] Autofocus complete")
    return focus_results


def set_arm_offset(xy_or_xyz):
    """Persist arm_offset to cameras.json. Accepts [x,y] or [x,y,z]."""
    cams = _load()
    vals = [float(v) for v in xy_or_xyz]
    while len(vals) < 3:
        vals.append(0.0)
    cams["arm_offset"] = vals
    with open(CAMERAS_FILE, "w") as f:
        json.dump(cams, f, indent=2)
        f.write("\n")


def update_camera(name, **kwargs):
    """Update camera parameters and save to cameras.json.
    Example: update_camera('side', position=[-0.30, 0.32, 0.14], pitch=-14)"""
    cams = _load()
    if name not in cams:
        raise ValueError(f"Unknown camera '{name}'")
    cams[name].update(kwargs)
    with open(CAMERAS_FILE, "w") as f:
        json.dump(cams, f, indent=2)
    print(f"Updated {name} camera: {list(kwargs.keys())}")


def reload():
    """Force reload from file."""
    global _cameras
    _cameras = None
    _load()


def open_capture(device_name, cam_name):
    """Find a camera by device name and open a cv2.VideoCapture with resolution/focus applied.

    Returns (capture, index) on success or (None, None) on failure.
    `cam_name` is the logical name ('top'/'side') used to look up saved resolution/focus.
    """
    import cv2

    idx = find_index_by_name(device_name) if device_name else None
    if idx is None:
        print(
            f"{_RED}[cameras]{_RESET} No {cam_name} camera found (name '{device_name}')"
        )
        return None, None

    dev_path = f"/dev/video{idx}"
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"{_RED}[cameras]{_RESET} Could not open {cam_name} camera ({dev_path})")
        return None, None

    # FOURCC must be set first, then resolution
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    res_w, res_h = get_resolution(cam_name)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
    if cam_name == "top":
        cap.set(cv2.CAP_PROP_SHARPNESS, 7)

    focus = get_focus(cam_name)
    if focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        print(f"{_CYAN}[cameras]{_RESET} {cam_name} using saved focus={focus}")
    elif cam_name == "top":
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 60)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"{_GREEN}[cameras]{_RESET} {cam_name} camera opened ({dev_path}, {actual_w}x{actual_h})"
    )
    return cap, idx


def reopen_by_index(idx):
    """Reopen a cv2.VideoCapture at the given /dev/video index. Returns capture or None."""
    import cv2

    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        return cap
    return None
