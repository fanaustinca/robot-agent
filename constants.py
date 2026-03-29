"""
Shared constants for SO-101 Robot Agent system.
Single source of truth — imported by robot_server.py and gemini_robot_agent.py.
"""

# ---- Camera & Image ----
SNAPSHOT_WIDTH = 256
SNAPSHOT_HEIGHT = 192
SNAPSHOT_TOP_WIDTH = 512
SNAPSHOT_TOP_HEIGHT = 384
JPEG_QUALITY = 60

# Crosshair offset: pixels from image center to gripper closing point
CROSSHAIR_OFFSET_X = 30
CROSSHAIR_OFFSET_Y = 38

# Degree scale ruler: pixels per degree on wrist camera overlay
PX_PER_DEG = 8

# ---- Motion ----
SLOW_MOVE_STEPS = 8
SLOW_MOVE_DELAY = 0.08  # seconds between interpolation steps

# ---- Safety ----
SHOULDER_LIFT_MAX = 50    # max up (agent space)
SHOULDER_LIFT_MIN = -65   # max down — floor protection

# ---- Joint Names ----
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_SHORT = {
    "shoulder_pan": "Pan",
    "shoulder_lift": "Lift",
    "elbow_flex": "Elbow",
    "wrist_flex": "Flex",
    "wrist_roll": "Roll",
    "gripper": "Grip",
}
JOINT_ALIASES = {
    "sp": "shoulder_pan", "pan": "shoulder_pan",
    "sl": "shoulder_lift", "lift": "shoulder_lift",
    "ef": "elbow_flex", "elbow": "elbow_flex",
    "wf": "wrist_flex", "flex": "wrist_flex",
    "wr": "wrist_roll", "roll": "wrist_roll",
    "g": "gripper", "grip": "gripper",
}

# ---- Presets ----
PRESETS = {
    "home":    {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90, "gripper": 0},
    "default": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0},
    "rest":    {"shoulder_pan": -1.10, "shoulder_lift": -102.24, "elbow_flex": 96.57, "wrist_flex": 76.35, "wrist_roll": -86.02, "gripper": 1.20},
}

# ---- Server ----
DEFAULT_PORT = 7878
TELEOP_FPS = 60

# ---- Recordings ----
RECORDINGS_DIR = "recordings"
RECORDING_FPS = 20  # frames per second when recording teleop
