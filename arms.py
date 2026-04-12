"""Arm hardware configuration loader. Reads settings from arms.json."""

import json
import os

ARMS_FILE = os.path.join(os.path.dirname(__file__), "arms.json")

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
