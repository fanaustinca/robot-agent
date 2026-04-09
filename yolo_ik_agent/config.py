"""
Configuration for the YOLO + IK robot agent.
Arm dimensions, camera settings, server URL, etc.
"""

import os

# ---- Server ----
ROBOT_SERVER = os.environ.get("ROBOT_SERVER", "http://localhost:7878")

# ---- Arm Geometry (meters for IKPy, from official SO-101 URDF) ----
ARM_BASE_HEIGHT = 0.054     # base to shoulder pivot (5.4 cm)
ARM_UPPER_ARM = 0.113       # shoulder pivot to elbow (11.3 cm)
ARM_LOWER_ARM = 0.135       # elbow to wrist (13.5 cm)
ARM_WRIST_LENGTH = 0.064    # wrist to gripper tip (6.4 cm)

# ---- Table ----
TABLE_Z = -0.05              # table surface is 5cm below arm base
GRIPPER_CLEARANCE = -0.03   # gripper target: 2cm above table = -3cm in arm frame
GRIPPER_APPROACH_HEIGHT = 0.08  # approach height before lowering (8 cm)

# ---- YOLO ----
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8x-worldv2.pt")  # open-vocabulary model (extra-large)
YOLO_CONFIDENCE = 0.15      # lower threshold for open-vocab detection
YOLO_DEVICE = "cuda"        # "cuda" for GPU, "cpu" for CPU

# ---- Camera Resolution (full res for local YOLO) ----
CAM_TOP_WIDTH = 1280
CAM_TOP_HEIGHT = 720
CAM_WRIST_WIDTH = 640
CAM_WRIST_HEIGHT = 480

# ---- Calibration Files ----
CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration_data")
TOP_CALIB_FILE = os.path.join(CALIB_DIR, "top_camera.json")
WRIST_CALIB_FILE = os.path.join(CALIB_DIR, "wrist_camera.json")

# ---- Checkerboard ----
CHECKERBOARD_INNER_CORNERS = (7, 7)  # 8x8 squares = 7x7 inner corners
CHECKERBOARD_SQUARE_SIZE = 0.032     # 3.2 cm per square (measured from chess board)

# ---- Top Camera Extrinsic (position relative to arm base) ----
# Physical world frame: +X = right, +Y = forward, +Z = up (relative to arm base on table)
TOP_CAM_X = -0.20           # camera is 20 cm to the LEFT of the arm base
TOP_CAM_Y = 0.0             # camera roughly even forward/back with arm base
TOP_CAM_Z = 0.30            # camera lens 35 cm above table - 5 cm base height = 30 cm above arm base
TOP_CAM_PITCH = -45.0       # degrees below horizontal (-90 = straight down)

# ---- Wrist Camera Mount (relative to wrist roll joint) ----
# Camera is mounted on the wrist roll link, centered left-right
# Wrist roll joint is 10cm from gripper tip. Camera is 5.5cm forward of roll joint.
# Offset is in the wrist roll link's local frame (before roll rotation)
WRIST_CAM_FORWARD = 0.055    # 5.5 cm forward of roll joint (in local -Z)
WRIST_CAM_UP = 0.06          # 6 cm above roll joint (in local +Y)
WRIST_CAM_RIGHT = 0.0        # centered (no lateral offset)
WRIST_CAM_MOUNT_TILT = -30.0   # camera tilt in link YZ plane: 0 = along -Z, negative = toward -Y (steeper down)

# ---- Wrist Camera at Observe Position (directly measured) ----
# Physical world frame: +X = right, +Y = forward, +Z = up (relative to arm base)
# Measured with arm at /observe preset. Used for stereo triangulation instead of FK.
WRIST_CAM_OBSERVE_X = -0.04      # 4 cm to the left of arm base
WRIST_CAM_OBSERVE_Y = 0.15       # 15 cm forward of arm base
WRIST_CAM_OBSERVE_Z = 0.28       # 28 cm above arm base
# Rotation is computed from FK at observe joint angles (joint lengths don't affect rotation)

# ---- URDF Base Offset ----
# The URDF origin is offset from the physical arm base (from baseframe joint in URDF)
URDF_BASE_OFFSET = (-0.163038, -0.168068, 0.0324817)

# ---- URDF to Physical rotation ----
# Physical: +X=right, +Y=forward, +Z=up
# URDF: +X=forward, +Y=left, +Z=up (from FK analysis)
import numpy as np
URDF_TO_PHYS_ROT = np.array([
    [0, -1, 0],   # physical right  = -URDF Y
    [1,  0, 0],   # physical forward = +URDF X
    [0,  0, 1],   # physical up      = +URDF Z
], dtype=float)

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
