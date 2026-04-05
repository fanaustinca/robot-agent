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
TABLE_Z = 0.0               # table surface height (reference plane)
GRIPPER_CLEARANCE = 0.015   # gripper height above table for pickup (1.5 cm)
GRIPPER_APPROACH_HEIGHT = 0.08  # approach height before lowering (8 cm)

# ---- YOLO ----
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8m.pt")  # medium model
YOLO_CONFIDENCE = 0.4       # minimum detection confidence
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
# Measure these from your setup — distance from arm base to directly below camera
# x = forward from arm base, y = left (positive) / right (negative), z = up
TOP_CAM_X = 0.0             # camera roughly centered forward/back (adjust if offset)
TOP_CAM_Y = -0.21           # camera is 21 cm to the LEFT of the arm base
TOP_CAM_Z = 0.55            # camera lens 55 cm above table
# Camera looks straight down (adjust if angled)
TOP_CAM_PITCH = -90.0       # degrees, -90 = looking straight down

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
