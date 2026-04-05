"""
SO-101 arm kinematics using IKPy.
Chain built from the official SO-101 URDF with corrected joint ordering.
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from ikpy.chain import Chain
from ikpy.link import URDFLink
from config import C


URDF_PATH = os.path.join(os.path.dirname(__file__), "so101.urdf")


def _parse_origin(joint_elem):
    """Parse origin xyz and rpy from a URDF joint element."""
    origin = joint_elem.find("origin")
    if origin is None:
        return [0, 0, 0], [0, 0, 0]
    xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
    return xyz, rpy


def build_chain():
    """Build IKPy chain from the SO-101 URDF, handling reversed joint order."""
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    # Parse all joints into a dict by name
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        jtype = j.get("type")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        xyz, rpy = _parse_origin(j)
        axis_elem = j.find("axis")
        axis = [float(x) for x in axis_elem.get("xyz", "0 0 1").split()] if axis_elem is not None else [0, 0, 1]
        limits = j.find("limit")
        lower = float(limits.get("lower", "-3.14")) if limits is not None else -3.14
        upper = float(limits.get("upper", "3.14")) if limits is not None else 3.14
        joints[name] = {
            "type": jtype, "parent": parent, "child": child,
            "xyz": xyz, "rpy": rpy, "axis": axis,
            "lower": lower, "upper": upper,
        }

    # Build chain in order: joint 1 → 2 → 3 → 4 → 5 (skip 6 = gripper jaw)
    # Joint 1: base → shoulder (shoulder pan)
    # Joint 2: shoulder → upper_arm (shoulder lift)
    # Joint 3: upper_arm → lower_arm (elbow)
    # Joint 4: lower_arm → wrist (wrist flex)
    # Joint 5: wrist → gripper (wrist roll)
    joint_order = ["1", "2", "3", "4", "5"]

    links = [
        URDFLink(name="base_link", origin_translation=[0, 0, 0],
                 origin_orientation=[0, 0, 0], rotation=[0, 0, 0]),
    ]

    joint_names = []
    for jname in joint_order:
        j = joints[jname]
        is_revolute = j["type"] == "revolute"
        links.append(URDFLink(
            name=jname,
            origin_translation=j["xyz"],
            origin_orientation=j["rpy"],
            rotation=j["axis"] if is_revolute else [0, 0, 0],
            bounds=(j["lower"], j["upper"]) if is_revolute else None,
        ))
        joint_names.append(jname)

    # Add gripper tip offset (from joint 5 to gripper center)
    j5 = joints.get("5", {})
    links.append(URDFLink(
        name="gripper_tip",
        origin_translation=[0.02, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0],
    ))

    chain = Chain(name="so101", links=links,
                  active_links_mask=[False] + [True]*len(joint_order) + [False])
    return chain, joint_names


# Cached chain
_chain = None
_joint_names = None

def get_chain():
    global _chain, _joint_names
    if _chain is None:
        _chain, _joint_names = build_chain()
        print(f"{C.GREEN}[ik]{C.RESET} Kinematic chain: {[l.name for l in _chain.links]}")
    return _chain


def forward_kinematics(joint_angles_deg):
    """Given 5 joint angles in degrees [pan, lift, elbow, wrist_flex, wrist_roll],
    return gripper tip position (x, y, z) in meters."""
    chain = get_chain()
    angles_rad = [0]  # base (fixed)
    for a in joint_angles_deg:
        angles_rad.append(np.radians(a))
    angles_rad.append(0)  # gripper_tip (fixed)
    transform = chain.forward_kinematics(angles_rad)
    return transform[:3, 3]


def inverse_kinematics(target_xyz, current_angles_deg=None):
    """Given target (x, y, z) in meters, return 5 joint angles in degrees."""
    chain = get_chain()
    target = np.array(target_xyz)

    initial = None
    if current_angles_deg is not None:
        initial = [0] + [np.radians(a) for a in current_angles_deg] + [0]

    angles_rad = chain.inverse_kinematics(target_position=target, initial_position=initial)

    # Extract active joint angles (indices 1-5)
    return [np.degrees(angles_rad[i]) for i in range(1, 6)]


def get_wrist_camera_pose(joint_angles_deg):
    """Get wrist camera 3D position and rotation from joint angles."""
    chain = get_chain()
    angles_rad = [0] + [np.radians(a) for a in joint_angles_deg] + [0]
    transform = chain.forward_kinematics(angles_rad)
    return transform[:3, 3], transform[:3, :3]


if __name__ == "__main__":
    print("Building SO-101 chain from URDF...")
    chain = get_chain()

    home = [0, 0, 0, 0, 0]
    pos = forward_kinematics(home)
    print(f"\nHome FK: x={pos[0]*100:.1f} y={pos[1]*100:.1f} z={pos[2]*100:.1f} cm")

    # Test IK
    for target in [[0.15, 0, 0.05], [0.10, 0.10, 0.02], [0.20, 0, 0.015]]:
        print(f"\nIK target: x={target[0]*100:.0f} y={target[1]*100:.0f} z={target[2]*100:.0f} cm")
        angles = inverse_kinematics(target)
        print(f"  Angles: {[f'{a:.1f}' for a in angles]}")
        verify = forward_kinematics(angles)
        error = np.linalg.norm(np.array(target) - verify) * 100
        print(f"  FK verify: x={verify[0]*100:.1f} y={verify[1]*100:.1f} z={verify[2]*100:.1f} cm")
        print(f"  Error: {error:.2f} cm")
