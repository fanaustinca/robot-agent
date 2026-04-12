"""Arm-offset and camera-extrinsic calibration command handlers.

Owns persistence of sample accumulators and the logic behind the
`/calib_arm` and `/calib_ex` dashboard commands. `robot_server.py`
dispatches to `handle_calib_arm` / `handle_calib_ex` and handles only
routing.
"""

import base64
import json
import os

import cameras as cameras_config


_CALIB_ARM_FILE = os.path.join(
    os.path.dirname(__file__), "calibration_data", "calib_arm_samples.json"
)
_CALIB_EX_FILE = os.path.join(
    os.path.dirname(__file__), "calibration_data", "calib_ex_samples.json"
)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


_calib_arm_samples = _load(_CALIB_ARM_FILE)
_calib_ex_samples = _load(_CALIB_EX_FILE)


# ---------------- /calib_arm ----------------


def handle_calib_arm(args_str, log):
    """Dispatch `/calib_arm` subcommands. `args_str` is the text after the command."""
    import numpy as np

    from yolo_ik_agent import (
        forward_kinematics,
        get_joint_angles,
        urdf_to_physical,
    )

    args_lower = args_str.lower()

    if args_lower == "clear":
        _calib_arm_samples.clear()
        _save(_CALIB_ARM_FILE, _calib_arm_samples)
        log("[cmd] Cleared all arm calibration samples")
        return

    if args_lower == "status":
        if not _calib_arm_samples:
            log("[cmd] No arm calibration samples collected")
            return
        for i, s in enumerate(_calib_arm_samples):
            log(
                f"[cmd]   #{i + 1}: board=({s['board'][0] * 100:.1f}, {s['board'][1] * 100:.1f})cm  arm=({s['arm'][0] * 100:.1f}, {s['arm'][1] * 100:.1f})cm  offset=({(s['board'][0] - s['arm'][0]) * 100:.1f}, {(s['board'][1] - s['arm'][1]) * 100:.1f})cm"
            )
        log(f"[cmd] {len(_calib_arm_samples)} sample(s). Need 1+ to solve.")
        return

    if args_lower.startswith("del"):
        del_parts = args_str.split()
        if len(del_parts) < 2:
            log("[cmd] Usage: /calib_arm del <number>")
            return
        try:
            idx = int(del_parts[1]) - 1
        except ValueError:
            log("[cmd] Usage: /calib_arm del <number>")
            return
        if 0 <= idx < len(_calib_arm_samples):
            removed = _calib_arm_samples.pop(idx)
            _save(_CALIB_ARM_FILE, _calib_arm_samples)
            log(
                f"[cmd] Deleted sample #{idx + 1}: board=({removed['board'][0] * 100:.1f}, {removed['board'][1] * 100:.1f})cm"
            )
        else:
            log(f"[cmd] Invalid sample number (have {len(_calib_arm_samples)} samples)")
        return

    if args_lower == "solve":
        if len(_calib_arm_samples) < 1:
            log("[cmd] Need at least 1 sample. Use: /calib_arm <x_cm> <y_cm>")
            return
        offsets = np.array(
            [
                [s["board"][0] - s["arm"][0], s["board"][1] - s["arm"][1]]
                for s in _calib_arm_samples
            ]
        )
        mean_offset = offsets.mean(axis=0)
        log(f"[cmd] Computed arm offset from {len(_calib_arm_samples)} sample(s):")
        log(f"[cmd]   X (right) = {mean_offset[0] * 100:.2f} cm")
        log(f"[cmd]   Y (fwd)   = {mean_offset[1] * 100:.2f} cm")
        if len(_calib_arm_samples) >= 2:
            std = offsets.std(axis=0)
            log(f"[cmd]   Std dev: X={std[0] * 100:.2f}cm  Y={std[1] * 100:.2f}cm")
        try:
            z_val = cameras_config.get_arm_offset()[2]
            new_offset = [
                round(float(mean_offset[0]), 6),
                round(float(mean_offset[1]), 6),
                z_val,
            ]
            cameras_config.set_arm_offset(new_offset)
            log(
                f"[cmd] Saved arm_offset to cameras.json: [{new_offset[0] * 100:.2f}, {new_offset[1] * 100:.2f}, {z_val * 100:.2f}] cm"
            )
            log("[cmd] Restart server for this to take effect")
        except Exception as e:
            log(f"[cmd] Error saving to cameras.json: {e}")
        return

    if args_str:
        # /calib_arm <x_cm> <y_cm> — collect a sample
        parts = args_str.split()
        if len(parts) != 2:
            log("[cmd] Usage: /calib_arm <x_cm> <y_cm>")
            return
        try:
            board_x = float(parts[0]) / 100.0
            board_y = float(parts[1]) / 100.0
        except ValueError:
            log("[cmd] Usage: /calib_arm <x_cm> <y_cm>")
            return
        angles = get_joint_angles()
        if not angles:
            log("[cmd] Cannot read joint angles — is the arm connected?")
            return
        urdf_pos = forward_kinematics(angles)
        arm_phys = urdf_to_physical(urdf_pos)
        sample = {
            "board": [board_x, board_y],
            "arm": [float(arm_phys[0]), float(arm_phys[1])],
        }
        _calib_arm_samples.append(sample)
        _save(_CALIB_ARM_FILE, _calib_arm_samples)
        offset_x = board_x - arm_phys[0]
        offset_y = board_y - arm_phys[1]
        log(
            f"[cmd] Sample #{len(_calib_arm_samples)}: board=({board_x * 100:.1f}, {board_y * 100:.1f})cm  arm=({arm_phys[0] * 100:.1f}, {arm_phys[1] * 100:.1f})cm  offset=({offset_x * 100:.1f}, {offset_y * 100:.1f})cm"
        )
        log(
            f"[cmd] {len(_calib_arm_samples)} sample(s) collected. Use /calib_arm solve when ready."
        )
        return

    log("[cmd] Arm offset calibration — place gripper tip on a known board position")
    log("[cmd]   /calib_arm <x_cm> <y_cm>   Collect sample (x=right, y=forward)")
    log("[cmd]   /calib_arm status           Show collected samples")
    log("[cmd]   /calib_arm solve            Compute & save arm_offset")
    log("[cmd]   /calib_arm del <n>          Delete sample #n")
    log("[cmd]   /calib_arm clear            Clear all samples")


# ---------------- /calib_ex ----------------


def handle_calib_ex(args_str, log, capture_snapshot, detection_overlay):
    """Dispatch `/calib_ex` subcommands.

    `capture_snapshot(cam_name) -> (base64_jpeg, err)` and `detection_overlay`
    (a mutable dict shared with the server) are injected so this module
    stays independent of Flask globals.
    """
    import time

    import cv2
    import numpy as np
    from cameras import get_scaled_intrinsics, update_camera
    from cameras import reload as reload_cameras
    from detect import detect_objects

    from yolo_ik_agent import get_surface_z

    args_lower = args_str.lower()

    if args_lower == "clear":
        _calib_ex_samples.clear()
        _save(_CALIB_EX_FILE, _calib_ex_samples)
        log("[cmd] Cleared all extrinsic calibration samples")
        return

    if args_lower.startswith("del"):
        del_parts = args_str.split()
        if len(del_parts) < 2:
            log("[cmd] Usage: /calib_ex del <number>")
            return
        try:
            idx = int(del_parts[1])
        except ValueError:
            log("[cmd] Usage: /calib_ex del <number>")
            return
        if 1 <= idx <= len(_calib_ex_samples):
            removed = _calib_ex_samples.pop(idx - 1)
            _save(_CALIB_EX_FILE, _calib_ex_samples)
            w = removed["world"]
            log(
                f"[cmd] Deleted sample {idx} (right={w[0] * 100:.1f}cm fwd={w[1] * 100:.1f}cm). {len(_calib_ex_samples)} remaining."
            )
        else:
            log(
                f"[cmd] Invalid sample number. Have {len(_calib_ex_samples)} samples (1-{len(_calib_ex_samples)})."
            )
        return

    if args_lower == "status":
        log(f"[cmd] {len(_calib_ex_samples)} sample(s) collected")
        for i, s in enumerate(_calib_ex_samples):
            w = s["world"]
            cams = ", ".join(
                f"{k}=({int(v[0])},{int(v[1])})" for k, v in s["pixels"].items()
            )
            log(
                f"[cmd]   {i + 1}: right={w[0] * 100:.1f}cm fwd={w[1] * 100:.1f}cm → {cams}"
            )
        return

    if args_lower == "solve":
        if len(_calib_ex_samples) < 4:
            log(
                f"[cmd] Need at least 4 samples (have {len(_calib_ex_samples)}). Add more with /calib_ex <right_cm> <fwd_cm>"
            )
            return
        log(f"[cmd] Solving extrinsics from {len(_calib_ex_samples)} samples...")
        for cam_name in ["top", "side"]:
            cam_samples = [
                (s["world"], s["pixels"][cam_name])
                for s in _calib_ex_samples
                if cam_name in s["pixels"]
            ]
            if len(cam_samples) < 4:
                log(
                    f"[cmd] {cam_name}: only {len(cam_samples)} samples, need 4+. Skipping."
                )
                continue

            world_pts = np.array([s[0] for s in cam_samples], dtype=np.float64)
            img_pts = np.array([s[1] for s in cam_samples], dtype=np.float64)

            b64_frame, err = capture_snapshot(cam_name)
            if err:
                log(f"[cmd] {cam_name}: cannot get snapshot: {err}")
                continue
            img_bytes = base64.b64decode(b64_frame)
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            fh, fw = frame.shape[:2]
            cam_matrix, cam_dist = get_scaled_intrinsics(cam_name, fw, fh)

            best_rvec, best_tvec, best_err = None, None, float("inf")
            methods = [
                ("IPPE", cv2.SOLVEPNP_IPPE),
                ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
                ("SQPNP", cv2.SOLVEPNP_SQPNP),
                ("EPNP", cv2.SOLVEPNP_EPNP),
            ]
            for method_name, method_flag in methods:
                try:
                    ok, rv, tv = cv2.solvePnP(
                        world_pts.reshape(-1, 1, 3),
                        img_pts.reshape(-1, 1, 2),
                        cam_matrix,
                        cam_dist,
                        flags=method_flag,
                    )
                    if ok:
                        proj, _ = cv2.projectPoints(
                            world_pts, rv, tv, cam_matrix, cam_dist
                        )
                        err = np.mean(
                            np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)
                        )
                        log(f"[cmd] {cam_name} {method_name}: {err:.1f}px reproj error")
                        if err < best_err:
                            best_err = err
                            best_rvec, best_tvec = rv, tv
                except Exception:
                    pass

            if best_rvec is None:
                log(f"[cmd] {cam_name}: solvePnP failed!")
                continue

            try:
                best_rvec, best_tvec = cv2.solvePnPRefineVVS(
                    world_pts.reshape(-1, 1, 3),
                    img_pts.reshape(-1, 1, 2),
                    cam_matrix,
                    cam_dist,
                    best_rvec,
                    best_tvec,
                )
            except Exception:
                pass

            R_cam, _ = cv2.Rodrigues(best_rvec)
            cam_pos = (-R_cam.T @ best_tvec).flatten()
            R_world = R_cam.T
            optical_axis = R_world @ np.array([0.0, 0.0, 1.0])
            pitch_deg = float(np.degrees(np.arcsin(optical_axis[2])))
            yaw_deg = float(np.degrees(np.arctan2(optical_axis[0], optical_axis[1])))

            proj, _ = cv2.projectPoints(
                world_pts, best_rvec, best_tvec, cam_matrix, cam_dist
            )
            errors = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)
            for i, e in enumerate(errors):
                marker = " *** OUTLIER" if e > 15 else ""
                log(f"[cmd]   sample {i + 1}: {e:.1f}px{marker}")

            log(
                f"[cmd] {cam_name}: position right={cam_pos[0] * 100:+.1f}cm fwd={cam_pos[1] * 100:+.1f}cm up={cam_pos[2] * 100:+.1f}cm"
            )
            log(
                f"[cmd] {cam_name}: yaw={yaw_deg:.1f}° pitch={pitch_deg:.1f}° reproj={best_err:.1f}px"
            )

            update_camera(
                cam_name,
                position=cam_pos.tolist(),
                yaw=round(yaw_deg, 1),
                pitch=round(pitch_deg, 1),
            )
            log(f"[cmd] {cam_name}: saved to cameras.json")

        reload_cameras()
        log("[cmd] Extrinsic calibration complete")
        return

    if args_str:
        parts = args_str.split()
        if len(parts) < 2:
            log("[cmd] Usage: /calib_ex <right_cm> <fwd_cm>")
            log("[cmd]   or: /calib_ex solve | clear | status")
            return
        try:
            right_m = float(parts[0]) / 100.0
            fwd_m = float(parts[1]) / 100.0
        except ValueError:
            log("[cmd] Invalid coordinates. Usage: /calib_ex <right_cm> <fwd_cm>")
            return

        world_pt = [right_m, fwd_m, get_surface_z()]
        log(
            f"[cmd] Detecting red block at right={right_m * 100:.1f}cm fwd={fwd_m * 100:.1f}cm..."
        )
        sample = {"world": world_pt, "pixels": {}}

        for cam_name in ["top", "side"]:
            b64_frame, err = capture_snapshot(cam_name)
            if err:
                log(f"[cmd] {cam_name}: snapshot failed: {err}")
                continue
            img_bytes = base64.b64decode(b64_frame)
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            dets = detect_objects(frame, "red block")
            if not dets:
                log(f"[cmd] {cam_name}: no red block detected")
                continue

            best = dets[0]
            cx, cy = best["center"]
            conf = best["confidence"]
            sample["pixels"][cam_name] = (float(cx), float(cy))
            log(
                f"[cmd] {cam_name}: detected at pixel ({int(cx)}, {int(cy)}) conf={conf:.2f}"
            )

            x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            label_text = f"red block ({conf:.0%}) #{len(_calib_ex_samples) + 1}"
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            _, overlay_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            detection_overlay[cam_name] = base64.b64encode(
                overlay_buf.tobytes()
            ).decode("utf-8")
            detection_overlay["ts"] = time.time()

        if sample["pixels"]:
            _calib_ex_samples.append(sample)
            _save(_CALIB_EX_FILE, _calib_ex_samples)
            log(
                f"[cmd] Sample {len(_calib_ex_samples)} added ({len(_calib_ex_samples)}/4 min). Samples:"
            )
            for i, s in enumerate(_calib_ex_samples):
                w = s["world"]
                cams = ", ".join(
                    f"{k}=({int(v[0])},{int(v[1])})" for k, v in s["pixels"].items()
                )
                log(
                    f"[cmd]   {i + 1}: right={w[0] * 100:.1f}cm fwd={w[1] * 100:.1f}cm → {cams}"
                )
        else:
            log("[cmd] No detections in any camera. Try repositioning the block.")
        return

    log("[cmd] Usage: /calib_ex <right_cm> <fwd_cm>  — capture a sample")
    log("[cmd]   /calib_ex solve   — compute extrinsics from samples")
    log("[cmd]   /calib_ex clear   — reset all samples")
    log("[cmd]   /calib_ex status  — show collected samples")
