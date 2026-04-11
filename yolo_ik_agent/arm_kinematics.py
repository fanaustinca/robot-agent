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
        origin_translation=[0, 0, -0.10],  # local -Z = physical forward
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0],
    ))

    # Joint 5 (wrist roll) is inactive — it rotates around the gripper axis
    # and doesn't affect tip position. Only pan, lift, elbow, flex are active.
    chain = Chain(name="so101", links=links,
                  active_links_mask=[False, True, True, True, True, False, False])
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


def inverse_kinematics(target_xyz, current_angles_deg=None, wrist_roll_deg=-90.0):
    """Given target (x, y, z) in meters, return 5 joint angles in degrees.
    Wrist roll is fixed (not solved by IK) since it rotates around the gripper axis."""
    chain = get_chain()
    target = np.array(target_xyz)

    # Initial guess: current angles with wrist roll fixed, clamped to URDF bounds
    initial = None
    if current_angles_deg is not None:
        angles = list(current_angles_deg)
        angles[4] = wrist_roll_deg  # fix roll in initial guess
        initial = [0] + [np.radians(a) for a in angles] + [0]
        # Clamp to joint bounds so IKPy doesn't reject the initial guess
        for i, link in enumerate(chain.links):
            if link.bounds is not None:
                lo, hi = link.bounds
                initial[i] = np.clip(initial[i], lo, hi)

    angles_rad = chain.inverse_kinematics(target_position=target, initial_position=initial)

    # Extract joint angles (indices 1-5), override roll with fixed value
    result = [np.degrees(angles_rad[i]) for i in range(1, 6)]
    result[4] = wrist_roll_deg
    return result


def get_wrist_camera_pose(joint_angles_deg):
    """Get wrist camera position and rotation in PHYSICAL world frame.

    The camera is mounted on the wrist roll link with a known offset.
    Returns:
        position: [right, forward, up] in meters (physical frame)
        rotation: 3x3 matrix mapping camera-frame vectors to physical-world vectors
                  (camera: +Z = optical axis, +X = right in image, +Y = down in image)
    """
    from config import (
        WRIST_CAM_FORWARD, WRIST_CAM_UP, WRIST_CAM_RIGHT,
        WRIST_CAM_MOUNT_TILT,
        URDF_BASE_OFFSET, URDF_TO_PHYS_ROT,
    )

    chain = get_chain()
    angles_rad = [0] + [np.radians(a) for a in joint_angles_deg] + [0]
    transform = chain.forward_kinematics(angles_rad)

    urdf_pos = transform[:3, 3]  # gripper tip in URDF
    urdf_rot = transform[:3, :3]  # orientation of end-effector in URDF

    # Undo the gripper_tip offset (2cm in local X) to get wrist roll joint position
    # The camera is on the wrist roll link, NOT at the gripper tip
    GRIPPER_TIP_OFFSET = np.array([0, 0, -0.10])  # local -Z = physical forward
    wrist_pos_urdf = urdf_pos - urdf_rot @ GRIPPER_TIP_OFFSET

    # Camera mount offset in the wrist link's local frame
    # At home: local +X=right, +Y=up, -Z=forward (verified from FK)
    cam_offset_local = np.array([
        WRIST_CAM_RIGHT,     # local X = right
        WRIST_CAM_UP,        # local Y = up
        -WRIST_CAM_FORWARD,  # local -Z = forward
    ])
    # Transform offset to URDF world frame and add to wrist joint position
    cam_pos_urdf = wrist_pos_urdf + urdf_rot @ cam_offset_local

    # Convert position from URDF to physical frame
    _URDF_BASE = np.array(URDF_BASE_OFFSET)
    physical_pos = np.array([
        -(cam_pos_urdf[1] - _URDF_BASE[1]),  # URDF -Y -> physical right
        cam_pos_urdf[0] - _URDF_BASE[0],     # URDF +X -> physical forward
        cam_pos_urdf[2] - _URDF_BASE[2],     # URDF +Z -> physical up
    ])

    # Convert rotation from URDF to physical frame
    physical_rot_link = URDF_TO_PHYS_ROT @ urdf_rot

    # Derive camera rotation from FK link axes
    # Camera is on TOP of the wrist, looking DOWN
    # Optical axis is in the link's YZ plane: -Z rotated toward -Y by WRIST_CAM_MOUNT_TILT
    # (rotation around link X axis = gripper forward axis)
    t = np.radians(WRIST_CAM_MOUNT_TILT)
    st, ct = np.sin(t), np.cos(t)

    # Camera axes in link's LOCAL frame (X=gripper fwd, Y=left, Z=up)
    cam_z_local = np.array([0, st, -ct])       # optical axis: -Z tilted toward -Y
    cam_x_local = np.array([1, 0, 0])          # image right: +X (gripper forward)
    cam_y_local = np.array([0, -ct, -st])       # image down: completes right-handed frame

    # Transform from link-local to physical world frame
    physical_rot = physical_rot_link @ np.column_stack([cam_x_local, cam_y_local, cam_z_local])

    return physical_pos, physical_rot


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
