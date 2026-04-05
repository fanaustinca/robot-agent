"""
Camera calibration tool using a checkerboard pattern.
Computes intrinsic matrix and distortion coefficients for each camera.

Usage:
    python camera_calibration.py --camera top --index 0
    python camera_calibration.py --camera wrist --index 5

Instructions:
    1. Print an 8x8 checkerboard and tape it flat to cardboard
    2. Run this script for each camera
    3. Hold the checkerboard in front of the camera
    4. Press SPACE to capture a frame (need ~15-20 from different angles)
    5. Press 'c' to calibrate when you have enough frames
    6. Results saved to calibration_data/
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Add parent dir for config
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CHECKERBOARD_INNER_CORNERS, CHECKERBOARD_SQUARE_SIZE,
    CALIB_DIR, TOP_CALIB_FILE, WRIST_CALIB_FILE, C
)


def calibrate_camera(camera_index, camera_name, output_file,
                     resolution=None, checkerboard=CHECKERBOARD_INNER_CORNERS,
                     square_size=CHECKERBOARD_SQUARE_SIZE):
    """Interactive camera calibration using a checkerboard.

    Args:
        camera_index: /dev/video index
        camera_name: "top" or "wrist"
        output_file: path to save calibration JSON
        resolution: optional (width, height) tuple
        checkerboard: (rows, cols) inner corners
        square_size: size of each square in meters
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"{C.RED}[calib]{C.RESET} Cannot open camera index {camera_index}")
        return False

    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{C.GREEN}[calib]{C.RESET} Camera '{camera_name}' opened: {w}x{h}")
    print(f"{C.CYAN}[calib]{C.RESET} Checkerboard: {checkerboard[0]+1}x{checkerboard[1]+1} squares ({checkerboard} inner corners)")
    print(f"{C.CYAN}[calib]{C.RESET} Square size: {square_size*100:.1f} cm")
    print()
    print(f"  {C.BOLD}SPACE{C.RESET} = capture frame    {C.BOLD}C{C.RESET} = calibrate    {C.BOLD}Q{C.RESET} = quit")
    print(f"  Capture ~15-20 frames with the checkerboard at different positions/angles.")
    print()

    # 3D points of the checkerboard in real-world coordinates
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image
    captured = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find checkerboard
        found, corners = cv2.findChessboardCorners(gray, checkerboard,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_FAST_CHECK +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, checkerboard, corners_refined, found)
            status = f"Checkerboard FOUND - press SPACE to capture"
            color = (0, 255, 0)
        else:
            status = "Searching for checkerboard..."
            color = (0, 0, 255)

        # Draw status
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Captured: {captured} frames", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(f"Calibration - {camera_name}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and found:
            obj_points.append(objp)
            img_points.append(corners_refined)
            captured += 1
            print(f"{C.GREEN}[calib]{C.RESET} Frame {captured} captured!")

        elif key == ord('c'):
            if captured < 5:
                print(f"{C.YELLOW}[calib]{C.RESET} Need at least 5 frames (have {captured})")
                continue
            break

        elif key == ord('q'):
            print(f"{C.YELLOW}[calib]{C.RESET} Calibration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    # Run calibration
    print(f"\n{C.BLUE}[calib]{C.RESET} Calibrating with {captured} frames...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)

    if not ret:
        print(f"{C.RED}[calib]{C.RESET} Calibration failed!")
        return False

    # Calculate reprojection error
    total_error = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(obj_points)

    print(f"{C.GREEN}[calib]{C.RESET} Calibration successful!")
    print(f"  Reprojection error: {mean_error:.4f} pixels")
    print(f"  Camera matrix:\n{camera_matrix}")
    print(f"  Distortion: {dist_coeffs.ravel()}")

    # Save to JSON
    os.makedirs(CALIB_DIR, exist_ok=True)
    calib_data = {
        "camera_name": camera_name,
        "resolution": [w, h],
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "reprojection_error": mean_error,
    }
    with open(output_file, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"{C.GREEN}[calib]{C.RESET} Saved to {output_file}")
    return True


def load_calibration(filepath):
    """Load camera calibration from JSON file.
    Returns (camera_matrix, dist_coeffs, resolution) or (None, None, None)."""
    if not os.path.exists(filepath):
        return None, None, None
    with open(filepath) as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"])
    resolution = tuple(data["resolution"])
    return camera_matrix, dist_coeffs, resolution


def pixel_to_table_ray(pixel_xy, camera_matrix, dist_coeffs,
                        cam_position, cam_rotation):
    """Project a 2D pixel coordinate into a 3D ray, then intersect with the table plane (Z=0).

    Args:
        pixel_xy: (u, v) pixel coordinate
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: distortion coefficients
        cam_position: 3D position of camera in world frame
        cam_rotation: 3x3 rotation matrix of camera in world frame

    Returns: (x, y, z) intersection with Z=0 plane, or None if ray doesn't hit table.
    """
    # Undistort the pixel
    pts = np.array([[[pixel_xy[0], pixel_xy[1]]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    u, v = undistorted[0, 0]

    # Convert pixel to normalized camera coordinates
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

    # Transform ray to world frame
    ray_world = cam_rotation @ ray_cam

    # Intersect with Z=0 plane
    # cam_position + t * ray_world = [x, y, 0]
    # t = -cam_position[2] / ray_world[2]
    if abs(ray_world[2]) < 1e-6:
        return None  # ray parallel to table

    t = -cam_position[2] / ray_world[2]
    if t < 0:
        return None  # intersection behind camera

    point = cam_position + t * ray_world
    return point


def triangulate_point(pixel1, pixel2, cam_matrix1, dist1, cam_matrix2, dist2,
                       cam_pos1, cam_rot1, cam_pos2, cam_rot2):
    """Triangulate a 3D point from two 2D observations.

    Args:
        pixel1, pixel2: (u, v) pixel coordinates in camera 1 and 2
        cam_matrix1/2: 3x3 intrinsic matrices
        dist1/2: distortion coefficients
        cam_pos1/2: 3D camera positions
        cam_rot1/2: 3x3 rotation matrices

    Returns: (x, y, z) 3D point in world coordinates.
    """
    # Build projection matrices: P = K @ [R | t]
    # where t = -R @ cam_position (transforms world to camera frame)
    t1 = -cam_rot1.T @ cam_pos1
    P1 = cam_matrix1 @ np.hstack([cam_rot1.T, t1.reshape(3, 1)])

    t2 = -cam_rot2.T @ cam_pos2
    P2 = cam_matrix2 @ np.hstack([cam_rot2.T, t2.reshape(3, 1)])

    # Undistort points
    pts1 = cv2.undistortPoints(np.array([[[pixel1[0], pixel1[1]]]], dtype=np.float64),
                                cam_matrix1, dist1, P=cam_matrix1)
    pts2 = cv2.undistortPoints(np.array([[[pixel2[0], pixel2[1]]]], dtype=np.float64),
                                cam_matrix2, dist2, P=cam_matrix2)

    # Triangulate
    point_4d = cv2.triangulatePoints(P1, P2, pts1[0].T, pts2[0].T)
    point_3d = (point_4d[:3] / point_4d[3]).flatten()

    # Reprojection error check
    for name, pixel, P in [("top", pixel1, P1), ("wrist", pixel2, P2)]:
        reproj = P @ np.append(point_3d, 1.0)
        reproj = reproj[:2] / reproj[2]
        err = np.linalg.norm(reproj - np.array(pixel))
        if err > 50:
            print(f"  [triangulate] WARNING: {name} reprojection error = {err:.1f} px")
        else:
            print(f"  [triangulate] {name} reprojection error = {err:.1f} px")

    return point_3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration tool")
    parser.add_argument("--camera", required=True, choices=["top", "wrist"],
                        help="Which camera to calibrate")
    parser.add_argument("--index", type=int, required=True,
                        help="Camera /dev/video index")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    output = TOP_CALIB_FILE if args.camera == "top" else WRIST_CALIB_FILE
    res = (args.width, args.height) if args.width and args.height else None

    calibrate_camera(args.index, args.camera, output, resolution=res)
