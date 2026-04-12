"""Arm hardware config + utilities.

Provides:
- settings loaded from arms.json (serial numbers, teleop fps)
- serial-port discovery by USB serial number
- SOFollower / SOLeader connection helpers
- motor-id introspection
- named joint presets
"""

import json
import os

ARMS_FILE = os.path.join(os.path.dirname(__file__), "arms.json")

# ANSI colors (self-contained so callers don't need to inject a logger)
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"


# ---- arms.json loader ----

_config = None


def _load():
    global _config
    if _config is not None:
        return _config
    with open(ARMS_FILE) as f:
        _config = json.load(f)
    return _config


def follower_serial():
    return _load()["follower_serial"]


def leader_serial():
    return _load()["leader_serial"]


def teleop_fps():
    return _load().get("teleop_fps", 60)


def reload():
    global _config
    _config = None
    _load()


# ---- Named joint poses ----

PRESETS = {
    "home":    {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90, "gripper": 0},
    "ready":   {"shoulder_pan": 0, "shoulder_lift": 40, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90, "gripper": 0},
    "default": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0},
    "rest":    {"shoulder_pan": -1.10, "shoulder_lift": -102.24, "elbow_flex": 96.57, "wrist_flex": 76.35, "wrist_roll": -86.02, "gripper": 1.20},
    "drop":    {"shoulder_pan": -47.08, "shoulder_lift": 10.46, "elbow_flex": -12.44, "wrist_flex": 86.20, "wrist_roll": -94.64, "gripper": 0},
    "side_view": {"shoulder_pan": -5.5, "shoulder_lift": 24.1, "elbow_flex": 65.6, "wrist_flex": -71.6, "wrist_roll": -86.3, "gripper": 100},
}


# ---- Serial port discovery ----

def find_port_by_serial(serial_number):
    """Find the serial port whose USB serial number matches. Returns device path or None."""
    try:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            if port.serial_number == serial_number:
                print(f"{_GREEN}[arms]{_RESET} Found serial {serial_number} at {_CYAN}{port.device}{_RESET}")
                return port.device
    except Exception as e:
        print(f"{_RED}[arms]{_RESET} Serial lookup failed: {e}")
    return None


# ---- Connection helpers ----

def connect_follower():
    """Resolve the follower port and return a connected SOFollower, or None on failure."""
    port = find_port_by_serial(follower_serial())
    if port is None:
        print(f"{_RED}[arms]{_RESET} Follower serial {follower_serial()} not found")
        return None
    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        robot = SOFollower(SOFollowerRobotConfig(port=port))
        robot.connect(calibrate=False)
        print(f"{_GREEN}[arms]{_RESET} Connected to SO-101 follower on {_CYAN}{port}{_RESET}")
        try:
            robot.bus.enable_torque()
            print(f"{_GREEN}[arms]{_RESET} Torque enabled")
        except Exception as te:
            print(f"{_RED}[arms]{_RESET} Torque enable failed: {te} — arm may not move")
        return robot
    except Exception as e:
        print(f"{_RED}[arms]{_RESET} Could not connect to follower arm: {e}")
        return None


def connect_leader():
    """Connect the leader arm by USB serial number.
    Returns (leader_instance, resolved_port) or (None, None)."""
    port = find_port_by_serial(leader_serial())
    if not port:
        print(f"{_RED}[arms]{_RESET} Leader serial {leader_serial()} not found")
        return None, None
    try:
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
        leader = SOLeader(SOLeaderTeleopConfig(port=port))
        leader.connect()
        print(f"{_GREEN}[arms]{_RESET} Leader arm connected on {_CYAN}{port}{_RESET}")
        return leader, port
    except Exception as e:
        print(f"{_RED}[arms]{_RESET} Failed to connect leader arm: {e}")
        return None, None
