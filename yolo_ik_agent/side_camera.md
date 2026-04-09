# Side Camera Plan

## Problem

The YOLO+IK agent has poor Z (height) estimation. The wrist camera can't view objects horizontally due to arm joint limits (`side_view` preset has `wrist_flex: -71.6`). Stereo triangulation between top and wrist cameras produces inaccurate Z values because the angular baseline is too shallow.

## Solution

Add a fixed side-view webcam mounted at table height, looking horizontally at the workspace.

- Top camera gives X/Y (looking down)
- Side camera gives Z directly (looking horizontally — vertical pixels map to physical height)
- No stereo ray intersection math needed
- No arm movement needed during detection

## Hardware

- Any fixed-focus USB webcam (1080p is more than enough; current cameras run at 512x384 and 256x192)
- Mount at table height, true horizontal, facing the workspace
- Fixed focus preferred — autofocus changes focal length and invalidates calibration. If autofocus, disable in software: `cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)`
- Connects to existing USB-C hub (total bandwidth ~40 Mbps, well within USB 3.0's 5 Gbps)
- Check that the hub handles 3 UVC cameras; a powered hub is more reliable

## Detection Logic Change

Currently `detect_and_verify()` runs GroundingDINO with the specific target label (e.g. "red block"). This misses objects that GroundingDINO doesn't recognize by that label but could detect generically.

Proposed: run GroundingDINO with a broad generic prompt (detect all objects), then always pass to Gemini to identify which detection matches the user's target. Fallback to label-specific detection when `GEMINI_API_KEY` is not set.

## Base Detection (Not Needed)

Originally considered detecting bases/platforms beneath objects to infer Z. With a true horizontal side camera this is unnecessary — the side camera sees the object's actual height directly regardless of what it's sitting on.

## Gripper Offset

The SO-101 gripper's left jaw is fixed; only the right jaw opens. The gripper should target slightly left of the object's left edge, not the center. Currently `IK_OFFSET_RIGHT = -0.01` (1cm left) is a static approximation. Ideally this would be width-aware using the bounding box size, but that's a separate improvement.

## Wrist Camera Role

With a side camera, the wrist camera is no longer needed for 3D localization. The current YOLO+IK agent doesn't use the wrist camera for fine-tuning or grip verification (the Gemini agent does, but not this one). The wrist camera becomes unused in this agent but could be added later for close-up inspection and grip verification.

## Camera Roles Summary

| Camera | Role |
|--------|------|
| Top (fixed, looking down) | X/Y localization, object detection |
| Side (fixed, horizontal) | Z/height estimation |
| Wrist (on arm) | Currently unused; future: grip verification, close-up inspection |

## Files to Modify

| File | Changes |
|------|---------|
| `config.py` | Side camera resolution, position, pitch constants; `SIDE_CALIB_FILE` |
| `robot_server.py` | `CAM_SIDE` env var, third camera init, `/snapshot/side`, `/stream/side`, dashboard |
| `camera_calibration.py` | Support `--camera side` calibration |
| `detect.py` | Broad generic detection + always-Gemini identification |
| `yolo_ik_agent.py` | Replace stereo path with top+side; add `get_side_camera_extrinsics()`; remove `observe`/`side_view` preset usage from detection; simplify pickup sequence |

## Alternatives Considered

- **Depth camera (RealSense, OAK-D)**: Best accuracy, gives per-pixel depth, but $80-200. Could replace both top and side cameras with one device.
- **Dual-lens stereo USB camera**: Gives two synchronized streams but no depth output — you compute stereo matching yourself. Struggles on textureless/uniform-color objects. A single side webcam is simpler.
- **AI camera with on-device inference**: Runs fixed COCO classes, can't do open-vocabulary detection like GroundingDINO. Loses flexibility.
