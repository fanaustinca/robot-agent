"""
SO-101 arm kinematics using IKPy.
Defines the kinematic chain and provides forward/inverse kinematics.
"""

import numpy as np
from ikpy.chain import Chain
from ikpy.link import URDFLink
from config import ARM_BASE_HEIGHT, ARM_LOWER_LENGTH, ARM_UPPER_LENGTH, ARM_GRIPPER_LENGTH, C


def build_chain():
    """Build the IKPy kinematic chain for the SO-101 arm.
    Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    Returns an ikpy.chain.Chain."""

    chain = Chain(name="so101", links=[
        # Base: fixed link from ground to shoulder pivot
        URDFLink(
            name="base",
            origin_translation=[0, 0, ARM_BASE_HEIGHT],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],  # fixed
        ),
        # Joint 1: Shoulder pan (rotation around vertical Z axis)
        URDFLink(
            name="shoulder_pan",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],  # rotates around Z
            bounds=(np.radians(-150), np.radians(150)),
        ),
        # Joint 2: Shoulder lift (rotation around Y axis — forward/back tilt)
        URDFLink(
            name="shoulder_lift",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # rotates around Y
            bounds=(np.radians(-110), np.radians(110)),
        ),
        # Link: lower arm (shoulder to elbow)
        URDFLink(
            name="lower_arm",
            origin_translation=[ARM_LOWER_LENGTH, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],  # fixed
        ),
        # Joint 3: Elbow flex (rotation around Y axis)
        URDFLink(
            name="elbow_flex",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(np.radians(-110), np.radians(110)),
        ),
        # Link: upper arm (elbow to wrist)
        URDFLink(
            name="upper_arm",
            origin_translation=[ARM_UPPER_LENGTH, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],  # fixed
        ),
        # Joint 4: Wrist flex
        URDFLink(
            name="wrist_flex",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(np.radians(-110), np.radians(110)),
        ),
        # Joint 5: Wrist roll
        URDFLink(
            name="wrist_roll",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],  # rotates around X (roll)
            bounds=(np.radians(-150), np.radians(150)),
        ),
        # Gripper tip (fixed offset)
        URDFLink(
            name="gripper_tip",
            origin_translation=[ARM_GRIPPER_LENGTH, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],  # fixed
        ),
    ])
    return chain


# Global chain instance
_chain = None

def get_chain():
    global _chain
    if _chain is None:
        _chain = build_chain()
        print(f"{C.GREEN}[ik]{C.RESET} Kinematic chain built ({len(_chain.links)} links)")
    return _chain


def forward_kinematics(joint_angles_deg):
    """Given joint angles in degrees (shoulder_pan, shoulder_lift, elbow_flex,
    wrist_flex, wrist_roll), return the 3D position of the gripper tip.
    Returns (x, y, z) in meters."""
    chain = get_chain()
    # IKPy expects angles for ALL links including fixed ones (set to 0)
    # Order: base(0), shoulder_pan, shoulder_lift, lower_arm(0), elbow_flex,
    #        upper_arm(0), wrist_flex, wrist_roll, gripper_tip(0)
    angles_rad = [0,  # base (fixed)
                  np.radians(joint_angles_deg[0]),  # shoulder_pan
                  np.radians(joint_angles_deg[1]),  # shoulder_lift
                  0,  # lower_arm (fixed)
                  np.radians(joint_angles_deg[2]),  # elbow_flex
                  0,  # upper_arm (fixed)
                  np.radians(joint_angles_deg[3]),  # wrist_flex
                  np.radians(joint_angles_deg[4]),  # wrist_roll
                  0]  # gripper_tip (fixed)
    transform = chain.forward_kinematics(angles_rad)
    pos = transform[:3, 3]
    return pos


def inverse_kinematics(target_xyz, current_angles_deg=None):
    """Given a target (x, y, z) in meters, compute joint angles in degrees.
    Returns [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll].
    current_angles_deg: optional hint for starting position (avoids weird solutions)."""
    chain = get_chain()

    target = np.array(target_xyz)

    # Initial position hint
    if current_angles_deg is not None:
        initial = [0,
                   np.radians(current_angles_deg[0]),
                   np.radians(current_angles_deg[1]),
                   0,
                   np.radians(current_angles_deg[2]),
                   0,
                   np.radians(current_angles_deg[3]),
                   np.radians(current_angles_deg[4]),
                   0]
    else:
        initial = None

    # Solve IK
    angles_rad = chain.inverse_kinematics(
        target_position=target,
        initial_position=initial,
    )

    # Extract active joint angles and convert to degrees
    result = [
        np.degrees(angles_rad[1]),  # shoulder_pan
        np.degrees(angles_rad[2]),  # shoulder_lift
        np.degrees(angles_rad[4]),  # elbow_flex
        np.degrees(angles_rad[6]),  # wrist_flex
        np.degrees(angles_rad[7]),  # wrist_roll
    ]
    return result


def get_wrist_camera_pose(joint_angles_deg):
    """Get the 3D position and rotation matrix of the wrist camera
    given current joint angles. Used for triangulation.
    Returns (position_3d, rotation_3x3)."""
    chain = get_chain()
    angles_rad = [0,
                  np.radians(joint_angles_deg[0]),
                  np.radians(joint_angles_deg[1]),
                  0,
                  np.radians(joint_angles_deg[2]),
                  0,
                  np.radians(joint_angles_deg[3]),
                  np.radians(joint_angles_deg[4]),
                  0]
    transform = chain.forward_kinematics(angles_rad)
    position = transform[:3, 3]
    rotation = transform[:3, :3]
    return position, rotation


if __name__ == "__main__":
    # Quick test
    print("Building SO-101 kinematic chain...")
    chain = get_chain()
    print(f"Links: {[l.name for l in chain.links]}")

    # Test forward kinematics at home position
    home = [0, 0, 0, 0, -90]
    pos = forward_kinematics(home)
    print(f"\nHome position FK: x={pos[0]*100:.1f}cm y={pos[1]*100:.1f}cm z={pos[2]*100:.1f}cm")

    # Test forward kinematics at ready position (shoulder_lift=40 in hardware = -40 agent)
    ready = [0, -40, 0, 0, -90]
    pos = forward_kinematics(ready)
    print(f"Ready position FK: x={pos[0]*100:.1f}cm y={pos[1]*100:.1f}cm z={pos[2]*100:.1f}cm")

    # Test inverse kinematics — pick a point in front of the arm
    target = [0.20, 0, 0.015]  # 20cm forward, 1.5cm above table
    print(f"\nIK target: x={target[0]*100:.1f}cm y={target[1]*100:.1f}cm z={target[2]*100:.1f}cm")
    angles = inverse_kinematics(target)
    print(f"IK solution: pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}")

    # Verify by running FK on the solution
    verify = forward_kinematics(angles)
    print(f"FK verify:   x={verify[0]*100:.1f}cm y={verify[1]*100:.1f}cm z={verify[2]*100:.1f}cm")
    error = np.linalg.norm(np.array(target) - verify) * 100
    print(f"Error: {error:.2f} cm")
