"""
Automatic extrinsic calibration for the wrist camera using the top camera.

The top camera (already calibrated) locates the checkerboard on the table,
then the wrist camera detects the same checkerboard. solvePnP computes
the wrist camera's exact position and rotation.

Usage:
    1. Start server: CAM_TOP=4 CAM_WRIST=1 python3 robot_server.py
    2. Move arm to observe: /observe
    3. Run: python3 calibrate_extrinsic.py
    4. Place checkerboard flat on the table where the wrist camera can see it
    5. SPACE = capture with top camera auto-position (both must see it)
    6. M = manual capture (only wrist needs to see it, you enter position)
    7. Move checkerboard to 6+ different positions across the workspace
    8. Press C to calibrate
"""

import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ROBOT_SERVER, CHECKERBOARD_INNER_CORNERS, CHECKERBOARD_SQUARE_SIZE,
    TOP_CALIB_FILE, WRIST_CALIB_FILE, C
)
from camera_calibration import load_calibration, pixel_to_table_ray


def get_snapshot(camera):
    """Get camera frame from server."""
    try:
        r = requests.get(f"{ROBOT_SERVER}/snapshot/{camera}", timeout=10)
        data = r.json()
        if "error" in data:
            return None
        img_bytes = base64.b64decode(data["data"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"{C.RED}Error getting {camera}: {e}{C.RESET}")
        return None


def get_joint_angles():
    """Get current joint angles."""
    try:
        r = requests.get(f"{ROBOT_SERVER}/status", timeout=5)
        status = r.json()
        joints = status.get("joints", {})
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        angles = []
        for name in names:
            for k, v in joints.items():
                if k.replace(".pos", "") == name:
                    angles.append(float(v))
                    break
            else:
                angles.append(0.0)
        return angles
    except:
        return None


def detect_checkerboard(frame, pattern=CHECKERBOARD_INNER_CORNERS):
    """Find checkerboard corners in a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def get_top_camera_extrinsics():
    """Get top camera extrinsics (same as in yolo_ik_agent.py)."""
    from config import TOP_CAM_X, TOP_CAM_Y, TOP_CAM_Z, TOP_CAM_PITCH
    position = np.array([TOP_CAM_X, TOP_CAM_Y, TOP_CAM_Z])
    p = np.radians(TOP_CAM_PITCH)
    s, c = np.sin(p), np.cos(p)
    rotation = np.array([
        [1, 0, 0],
        [0, s, c],
        [0, -c, s],
    ])
    return position, rotation


def checkerboard_3d_from_top_camera(corners_top, top_matrix, top_dist):
    """Use top camera to compute 3D positions of checkerboard corners on the table.
    Returns Nx3 array of world points (physical frame, Z=0)."""
    cam_pos, cam_rot = get_top_camera_extrinsics()
    world_points = []
    for pt in corners_top.reshape(-1, 2):
        point = pixel_to_table_ray(pt, top_matrix, top_dist, cam_pos, cam_rot)
        if point is None:
            return None
        point[2] = 0.0  # force Z=0 (on table)
        world_points.append(point)
    return np.array(world_points, dtype=np.float64)


def run_calibration():
    print(f"\n{C.BOLD}{'=' * 55}{C.RESET}")
    print(f"{C.BOLD}Automatic Wrist Camera Extrinsic Calibration{C.RESET}")
    print(f"{C.BOLD}  Using top camera to locate checkerboard positions{C.RESET}")
    print(f"{C.BOLD}{'=' * 55}{C.RESET}\n")

    # Load intrinsics for both cameras
    top_matrix, top_dist, top_res = load_calibration(TOP_CALIB_FILE)
    if top_matrix is None:
        print(f"{C.RED}Top camera not calibrated!{C.RESET}")
        return

    wrist_matrix, wrist_dist, _ = load_calibration(WRIST_CALIB_FILE)
    if wrist_matrix is None:
        print(f"{C.RED}Wrist camera not calibrated!{C.RESET}")
        return

    angles = get_joint_angles()
    if angles is None:
        print(f"{C.RED}Cannot read joint angles. Is the server running?{C.RESET}")
        return
    print(f"{C.GREEN}Joint angles:{C.RESET} pan={angles[0]:.1f} lift={angles[1]:.1f} elbow={angles[2]:.1f} flex={angles[3]:.1f} roll={angles[4]:.1f}")

    print(f"""
{C.CYAN}Instructions:{C.RESET}
  1. Arm should be at observe position (/observe)
  2. Place checkerboard FLAT on the table
  3. SPACE = auto capture (both cameras must see it, green)
  4. M = manual capture (only wrist needs it, you type position)
  5. Spread captures across the workspace! Near, far, left, right.
  6. Get 6+ captures, then press C to calibrate. Q to quit.
""")

    captures = []  # list of (wrist_corners, world_points_3d)

    while True:
        frame_top = get_snapshot("top")
        frame_wrist = get_snapshot("wrist")
        if frame_top is None or frame_wrist is None:
            time.sleep(0.5)
            continue

        # Scale top camera intrinsics if needed
        top_mat = top_matrix.copy()
        if top_res is not None:
            th, tw = frame_top.shape[:2]
            cw, ch = top_res
            if tw != cw or th != ch:
                sx, sy = tw / cw, th / ch
                top_mat[0, 0] *= sx; top_mat[0, 2] *= sx
                top_mat[1, 1] *= sy; top_mat[1, 2] *= sy

        found_top, corners_top = detect_checkerboard(frame_top)
        found_wrist, corners_wrist = detect_checkerboard(frame_wrist)

        # Draw on displays
        disp_top = frame_top.copy()
        disp_wrist = frame_wrist.copy()

        if found_top:
            cv2.drawChessboardCorners(disp_top, CHECKERBOARD_INNER_CORNERS, corners_top, True)
        if found_wrist:
            cv2.drawChessboardCorners(disp_wrist, CHECKERBOARD_INNER_CORNERS, corners_wrist, True)

        both_found = found_top and found_wrist

        if both_found:
            status = "BOTH cameras see checkerboard - press SPACE"
            color = (0, 255, 0)
        elif found_top:
            status = "Only TOP camera sees it - move checkerboard"
            color = (0, 165, 255)
        elif found_wrist:
            status = "Only WRIST camera sees it - move checkerboard"
            color = (0, 165, 255)
        else:
            status = "Searching for checkerboard..."
            color = (0, 0, 255)

        cv2.putText(disp_top, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(disp_top, f"Captures: {len(captures)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp_wrist, f"Captures: {len(captures)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resize top for display (it's 1920x1080)
        disp_h = disp_wrist.shape[0]
        scale = disp_h / disp_top.shape[0]
        disp_top_small = cv2.resize(disp_top, (int(disp_top.shape[1] * scale), disp_h))

        combined = np.hstack([disp_top_small, disp_wrist])
        cv2.imshow("Extrinsic Calibration (Top | Wrist)", combined)

        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and both_found:
            # Use top camera to get 3D world positions of checkerboard corners
            world_pts = checkerboard_3d_from_top_camera(corners_top, top_mat, top_dist)
            if world_pts is None:
                print(f"{C.YELLOW}  Failed to project corners from top camera{C.RESET}")
                continue

            wrist_pts = corners_wrist.reshape(-1, 2)

            # Check if corners are in same order by comparing distances
            # Corner 0 and last corner should be diagonally opposite
            # If top and wrist see board from opposite sides, corners are reversed
            top_c0 = world_pts[0]
            top_cN = world_pts[-1]
            # Use the wrist camera's solvePnP with BOTH orderings and pick better one
            rows, cols = CHECKERBOARD_INNER_CORNERS
            sq = CHECKERBOARD_SQUARE_SIZE

            best_err = float('inf')
            best_wrist = None
            best_world = None
            for flip in [False, True]:
                wp = world_pts[::-1] if flip else world_pts
                wrp = wrist_pts
                ok, rv, tv = cv2.solvePnP(wp, wrp, wrist_matrix, wrist_dist)
                if ok:
                    proj, _ = cv2.projectPoints(wp, rv, tv, wrist_matrix, wrist_dist)
                    err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - wrp, axis=1))
                    if err < best_err:
                        best_err = err
                        best_wrist = wrp
                        best_world = wp

            if best_err > 10:
                print(f"{C.YELLOW}  Skipped: reprojection error too high ({best_err:.1f}px) — try moving checkerboard{C.RESET}")
                continue

            captures.append((best_wrist, best_world))
            p0 = best_world[0]
            print(f"  {C.GREEN}Capture {len(captures)}{C.RESET}: board at right={p0[0]*100:.1f}cm fwd={p0[1]*100:.1f}cm (reproj={best_err:.1f}px)")

        elif key == ord('m') and found_wrist:
            # Manual capture — only wrist camera needed, user enters position
            cv2.destroyAllWindows()
            print(f"\n{C.CYAN}Manual capture (measure from arm base):{C.RESET}")
            try:
                right = float(input("  Checkerboard top-left inner corner — RIGHT (cm, neg=left): "))
                fwd = float(input("  Checkerboard top-left inner corner — FORWARD (cm): "))
            except (ValueError, EOFError):
                print(f"{C.YELLOW}Skipped{C.RESET}")
                continue

            rows, cols = CHECKERBOARD_INNER_CORNERS
            sq = CHECKERBOARD_SQUARE_SIZE
            world_pts = np.zeros((rows * cols, 3), dtype=np.float64)
            for r in range(rows):
                for c in range(cols):
                    world_pts[r * cols + c] = [
                        right / 100.0 + c * sq,
                        fwd / 100.0 + r * sq,
                        0.0
                    ]

            wrist_pts = corners_wrist.reshape(-1, 2)

            # Try both orderings
            best_err = float('inf')
            best_wrist = None
            best_world = None
            for flip in [False, True]:
                wp = world_pts[::-1] if flip else world_pts
                ok, rv, tv = cv2.solvePnP(wp, wrist_pts, wrist_matrix, wrist_dist)
                if ok:
                    proj, _ = cv2.projectPoints(wp, rv, tv, wrist_matrix, wrist_dist)
                    err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - wrist_pts, axis=1))
                    if err < best_err:
                        best_err = err
                        best_wrist = wrist_pts
                        best_world = wp

            if best_err > 10:
                print(f"{C.YELLOW}  Skipped: reprojection too high ({best_err:.1f}px) — check measurements{C.RESET}")
                continue

            captures.append((best_wrist, best_world))
            print(f"  {C.GREEN}Capture {len(captures)}{C.RESET}: board at right={right:.1f}cm fwd={fwd:.1f}cm (manual, reproj={best_err:.1f}px)")

        elif key == ord('c'):
            if len(captures) < 4:
                print(f"{C.YELLOW}Need at least 4 captures (have {len(captures)}){C.RESET}")
                continue
            break

        elif key == ord('q'):
            print(f"{C.YELLOW}Cancelled{C.RESET}")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # Solve for wrist camera extrinsics
    print(f"\n{C.BLUE}Solving extrinsics with {len(captures)} captures...{C.RESET}")

    # --- Per-capture error analysis and outlier removal ---
    n_corners = CHECKERBOARD_INNER_CORNERS[0] * CHECKERBOARD_INNER_CORNERS[1]
    print(f"\n  Per-capture reprojection error (pre-filter):")

    # First pass: solve with all points to get initial estimate
    all_img_pts = np.vstack([c[0] for c in captures]).astype(np.float64)
    all_world_pts = np.vstack([c[1] for c in captures]).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        all_world_pts, all_img_pts, wrist_matrix, wrist_dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        print(f"{C.RED}solvePnP failed!{C.RESET}")
        return

    # Evaluate each capture individually
    capture_errors = []
    for i, (img_pts, world_pts) in enumerate(captures):
        proj, _ = cv2.projectPoints(world_pts, rvec, tvec, wrist_matrix, wrist_dist)
        err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1))
        capture_errors.append(err)
        marker = f"{C.RED}  *** OUTLIER" if err > 15 else ""
        print(f"    Capture {i+1}: {err:.2f} px{marker}{C.RESET}")

    # Remove outlier captures (error > 2x median)
    median_err = np.median(capture_errors)
    threshold = max(median_err * 2.5, 10.0)  # at least 10px, or 2.5x median
    good_captures = []
    removed = 0
    for i, (cap, err) in enumerate(zip(captures, capture_errors)):
        if err <= threshold:
            good_captures.append(cap)
        else:
            removed += 1
            print(f"    {C.YELLOW}Removed capture {i+1} (err={err:.1f}px > threshold={threshold:.1f}px){C.RESET}")

    if removed > 0:
        print(f"  Kept {len(good_captures)}/{len(captures)} captures (removed {removed} outliers)")
    if len(good_captures) < 4:
        print(f"{C.RED}Too few good captures remaining ({len(good_captures)}). Need more data.{C.RESET}")
        return

    # Re-solve with clean data
    all_img_pts = np.vstack([c[0] for c in good_captures]).astype(np.float64)
    all_world_pts = np.vstack([c[1] for c in good_captures]).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        all_world_pts, all_img_pts, wrist_matrix, wrist_dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        print(f"{C.RED}solvePnP failed after filtering!{C.RESET}")
        return

    # Refine with Levenberg-Marquardt (VVS)
    try:
        rvec, tvec = cv2.solvePnPRefineVVS(
            all_world_pts, all_img_pts, wrist_matrix, wrist_dist, rvec, tvec
        )
        print(f"  {C.GREEN}Refined with VVS (Levenberg-Marquardt){C.RESET}")
    except Exception:
        pass  # older OpenCV may not have this

    # Also try RANSAC and pick whichever is better
    success_r, rvec_r, tvec_r, inliers = cv2.solvePnPRansac(
        all_world_pts, all_img_pts, wrist_matrix, wrist_dist,
        reprojectionError=5.0, iterationsCount=1000
    )
    if success_r and inliers is not None:
        n_inliers = len(inliers)
        n_total = len(all_world_pts)
        print(f"  RANSAC: {n_inliers}/{n_total} inliers")
        # Compare reprojection errors
        proj_cur, _ = cv2.projectPoints(all_world_pts, rvec, tvec, wrist_matrix, wrist_dist)
        err_cur = np.mean(np.linalg.norm(proj_cur.reshape(-1, 2) - all_img_pts, axis=1))
        proj_ran, _ = cv2.projectPoints(all_world_pts, rvec_r, tvec_r, wrist_matrix, wrist_dist)
        err_ran = np.mean(np.linalg.norm(proj_ran.reshape(-1, 2) - all_img_pts, axis=1))
        print(f"  Iterative: {err_cur:.2f} px  |  RANSAC: {err_ran:.2f} px")
        if err_ran < err_cur:
            rvec, tvec = rvec_r, tvec_r
            print(f"  {C.GREEN}Using RANSAC solution (lower error){C.RESET}")

    R_cam_from_world, _ = cv2.Rodrigues(rvec)
    cam_pos_world = (-R_cam_from_world.T @ tvec).flatten()
    R_world_from_cam = R_cam_from_world.T

    # Final reprojection error
    projected, _ = cv2.projectPoints(all_world_pts, rvec, tvec, wrist_matrix, wrist_dist)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - all_img_pts, axis=1)
    mean_err = np.mean(errors)
    max_err = np.max(errors)

    # Results
    optical_axis = R_world_from_cam @ np.array([0, 0, 1])
    image_right = R_world_from_cam @ np.array([1, 0, 0])

    print(f"\n{C.GREEN}{'=' * 55}{C.RESET}")
    print(f"{C.GREEN}Extrinsic Calibration Results{C.RESET}")
    print(f"{C.GREEN}{'=' * 55}{C.RESET}")
    print(f"\n  Camera position (physical frame):")
    print(f"    right   = {cam_pos_world[0]*100:.2f} cm")
    print(f"    forward = {cam_pos_world[1]*100:.2f} cm")
    print(f"    up      = {cam_pos_world[2]*100:.2f} cm")
    print(f"\n  Optical axis: right={optical_axis[0]:.3f} fwd={optical_axis[1]:.3f} up={optical_axis[2]:.3f}")
    print(f"  Image right:  right={image_right[0]:.3f} fwd={image_right[1]:.3f} up={image_right[2]:.3f}")
    print(f"\n  Reprojection error: mean={mean_err:.2f} px, max={max_err:.2f} px")
    if mean_err < 3:
        print(f"  {C.GREEN}Excellent calibration!{C.RESET}")
    elif mean_err < 8:
        print(f"  {C.GREEN}Good calibration.{C.RESET}")
    else:
        print(f"  {C.YELLOW}Mediocre — consider recalibrating with more varied positions.{C.RESET}")
    print(f"  Joint angles: {[f'{a:.1f}' for a in angles]}")

    # Sanity checks
    expected_pos = np.array([-0.06, 0.20, 0.28])  # roughly where wrist cam should be
    pos_diff = np.linalg.norm(cam_pos_world - expected_pos) * 100
    if pos_diff > 20:
        print(f"\n  {C.YELLOW}WARNING: Position is {pos_diff:.0f}cm from expected — results may be inaccurate{C.RESET}")
    if mean_err > 10:
        print(f"\n  {C.YELLOW}WARNING: High reprojection error — try more captures from varied positions{C.RESET}")

    # Save
    out_file = os.path.join(os.path.dirname(__file__), "calibration_data", "wrist_extrinsic.json")
    calib_data = {
        "camera_position": cam_pos_world.tolist(),
        "rotation_world_from_cam": R_world_from_cam.tolist(),
        "rotation_cam_from_world": R_cam_from_world.tolist(),
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "joint_angles": angles,
        "reprojection_error_mean": float(mean_err),
        "reprojection_error_max": float(max_err),
        "num_captures": len(captures),
        "num_points": len(all_world_pts),
    }
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"\n  {C.GREEN}Saved to {out_file}{C.RESET}\n")


if __name__ == "__main__":
    run_calibration()
