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
import threading
import time

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


# ---- Teleop (leader → follower mirroring) ----

_teleop = {
    "leader": None,
    "active": False,
    "thread": None,
}


def get_leader():
    return _teleop["leader"]


def is_teleop_active():
    return _teleop["active"]


def _teleop_loop(follower, follower_lock):
    interval = 1.0 / teleop_fps()
    fail_count = 0
    max_fails = 20  # ~1 second of consecutive failures before aborting
    print(f"{_GREEN}[teleop]{_RESET} Loop started at {teleop_fps()} fps")
    leader = _teleop["leader"]
    while _teleop["active"]:
        try:
            action = leader.get_action()
            with follower_lock:
                follower.send_action(action)
            fail_count = 0
        except Exception as e:
            fail_count += 1
            if fail_count >= max_fails:
                print(f"{_RED}[teleop]{_RESET} {max_fails} consecutive errors, stopping: {e}")
                break
            elif fail_count == 1:
                print(f"{_YELLOW}[teleop]{_RESET} Transient error (retrying): {e}")
            time.sleep(0.05)
            continue
        time.sleep(interval)
    _teleop["active"] = False
    print(f"{_YELLOW}[teleop]{_RESET} Loop stopped")


def start_teleop(follower, follower_lock):
    """Connect leader, enable follower torque, and spawn the mirror thread.
    Returns (ok, message)."""
    if _teleop["active"]:
        return False, "Teleop already running"
    if follower is None:
        return False, "Follower arm not connected"

    if _teleop["leader"] is None:
        leader, _ = connect_leader()
        if leader is None:
            return False, f"No leader arm found (serial {leader_serial()} not present)."
        _teleop["leader"] = leader

    try:
        follower.bus.enable_torque()
    except Exception as e:
        return False, f"Failed to enable torque: {e}"

    _teleop["active"] = True
    _teleop["thread"] = threading.Thread(
        target=_teleop_loop, args=(follower, follower_lock), daemon=True
    )
    _teleop["thread"].start()
    return True, "Teleop started"


def stop_teleop():
    """Stop the mirror thread and disconnect the leader."""
    if not _teleop["active"]:
        return False, "Teleop not running"
    _teleop["active"] = False
    if _teleop["thread"]:
        _teleop["thread"].join(timeout=2)
        _teleop["thread"] = None
    if _teleop["leader"] is not None:
        try:
            _teleop["leader"].disconnect()
        except Exception:
            pass
        _teleop["leader"] = None
    return True, "Teleop stopped"
