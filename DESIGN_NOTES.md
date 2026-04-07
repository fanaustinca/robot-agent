# Design Notes: YOLO+IK Agent Improvements

Prepared for implementation. Summarizes decisions and direction from design review on 2026-04-05.

## 1. Leader Arm Serial Number (Done)

The leader arm now uses deterministic serial number lookup, same as the follower.

- Follower serial: `5AE6083982`
- Leader serial: `5AE6084010`
- Changed in `robot_server.py`: `find_leader_port()` tries `find_port_by_serial(LEADER_SERIAL)` first, then falls back to `LEADER_PORT` env var, then auto-detect by exclusion.
- Already committed on the `stereo-3d` branch.

## 2. Object Detection Pipeline

### Current state

`yolo_ik_agent/detect.py` uses **GroundingDINO** (not YOLO despite the naming). It's an open-vocabulary zero-shot detector that takes a text prompt and finds matching objects in the image.

The text prompt comes from naive string parsing in `yolo_ik_agent.py` lines 653-673:
- Verb detection: substring match against `["pick up", "pickup", "grab", "retrieve", "fetch", "collect"]`
- Noun extraction: everything after the keyword, with leading articles stripped (`the`, `a`, `an`)
- Example: `"pick up the red lego brick"` -> target label `"red lego brick"`

### Problem

GroundingDINO can misidentify the target (picks highest confidence detection regardless of correctness), and there is no verification or retry. If it detects the wrong object, the arm goes for it silently.

### Decided approach: Gemini API as post-detection verifier

Use the Gemini API (not local model -- GPU is too constrained, see hardware section) to verify GroundingDINO's detections. The flow:

```
User: "pick up the red lego brick"
  1. String parsing extracts "red lego brick"
  2. GroundingDINO runs on the camera frame, returns N candidate detections with bounding boxes
  3. Annotate the frame with numbered bounding boxes
  4. Send annotated image + prompt to Gemini API:
     "Which numbered box contains a red lego brick? Reply with the number, or 'none'."
  5. Gemini picks the correct detection (or rejects all)
  6. Use that detection's center pixel for the IK pipeline
```

This handles: wrong-object picks, multiple similar objects, spatial/relative references ("the one on the left"), and cases where nothing matches.

**Rejected alternatives:**
- Gemini as pre-detection prompt refiner (option 1): marginal improvement over GroundingDINO's native text understanding, not worth the extra API call
- Local Gemma 4 model (see hardware constraints below)

## 3. Stereo 3D: Preset-Based Wrist Camera Positions

### Current state

The stereo triangulation pipeline (`detect_and_locate` with `use_triangulation=True`) uses two cameras:
- **Top camera**: fixed position, extrinsics hardcoded in `config.py`
- **Wrist camera**: moves with the arm, extrinsics computed via forward kinematics from URDF + joint angles

The FK-computed wrist extrinsics are not accurate enough. The triangulated Z is currently discarded and clamped to table height (`yolo_ik_agent.py` line 362):
```python
point[2] = GRIPPER_CLEARANCE  # Z from stereo is unreliable
```

### Problem with dynamic FK approach

- URDF link lengths and offsets must be perfectly accurate
- Joint backlash and servo imprecision compound through the kinematic chain
- Camera mount offset on the wrist must be precisely modeled
- Small errors at each joint accumulate into large position errors at the camera

### Decided approach: pre-allocated observe presets with pre-calibrated extrinsics

Instead of computing wrist camera extrinsics dynamically, define a small set of fixed arm positions ("observe presets"). Each preset has its wrist camera extrinsics calibrated once using the existing checkerboard tool.

**Data structure:**
```python
OBSERVE_PRESETS = {
    "left":   {"joints": {"shoulder_pan": 30, "shoulder_lift": -42, ...},
               "cam_pos": [0.08, 0.19, 0.27],       # from calibration
               "cam_rot": [[...], [...], [...]]},     # from calibration
    "center": {"joints": {"shoulder_pan": 0, ...},
               "cam_pos": [...], "cam_rot": [...]},
    "right":  {"joints": {"shoulder_pan": -30, ...},
               "cam_pos": [...], "cam_rot": [...]},
}
```

**Zone mapping:**

Divide the top camera image into regions. When GroundingDINO detects an object in a region, select the corresponding observe preset:

```
Top camera view:
+----------+----------+----------+
|  zone_L  |  zone_C  |  zone_R  |
+----------+----------+----------+
| zone_FL  | zone_FC  | zone_FR  |
+----------+----------+----------+

zone_C  -> observe_center (pan=0)
zone_L  -> observe_left   (pan=30)
zone_R  -> observe_right  (pan=-30)
```

Number of presets depends on wrist camera FOV. Start with 3 (left/center/right) and add more if coverage gaps appear.

**Modified pickup flow:**
1. Top camera detects object, determines which zone it's in
2. Arm moves to the corresponding observe preset
3. Wrist camera captures frame (GroundingDINO detects same object)
4. Triangulate using top pixel + wrist pixel + pre-calibrated extrinsics for that preset
5. IK computes joint angles from the 3D point
6. Arm moves to pick up

**Calibration (one-time per preset):**
1. Move arm to the preset joint angles
2. Run `calibrate_extrinsic.py` (already written, uses checkerboard visible to both cameras)
3. Save resulting `cam_pos` + `cam_rot` in a JSON file keyed by preset name

**Edge cases:**
- Object at zone boundary: overlap zones slightly, or fall back to table-plane (Z=0) if wrist can't detect the object
- Wrist camera blocked/can't detect: fall back to table-plane intersection (current method 1)
- Top camera blocked by arm: this won't happen at observe presets since the arm is positioned to look, not reach

## 4. Hardware Constraints

Machine: Intel Core Ultra 9 275HX, 24 cores, 30GB RAM, NVIDIA RTX 5060 Laptop (8GB VRAM), 88GB free disk.

VRAM budget at runtime:
- Xorg + Chrome: ~725 MB
- GroundingDINO: ~2.2 GB
- Free: ~5.1 GB

Local Gemma 4 was considered but rejected:
- Gemma 4 4B at FP16 needs ~5 GB -- won't fit alongside GroundingDINO
- INT4 quantized would fit (~2.5 GB) but adds complexity
- Gemini API is simpler and more capable for this verification task
- Latency difference is small (~1-2s API vs ~0.5-1.5s local)

## 5. Files to Modify

| File | Change |
|------|--------|
| `yolo_ik_agent/detect.py` | Add Gemini verification step after GroundingDINO detection |
| `yolo_ik_agent/yolo_ik_agent.py` | Add zone mapping, move arm to observe preset before wrist capture, use preset extrinsics for triangulation |
| `yolo_ik_agent/config.py` | Add observe preset definitions, zone boundaries |
| `yolo_ik_agent/calibrate_extrinsic.py` | Support saving calibration per preset name |
| `yolo_ik_agent/camera_calibration.py` | No changes expected (triangulate_point already works with arbitrary extrinsics) |

## 6. Existing Code Reference

- `robot_server.py` (root): Flask server, hardware bridge, 1514 lines. All agents talk to it over HTTP.
- `yolo_ik_agent/robot_server.py`: Modified copy with YOLO debug endpoints, 1736 lines.
- `yolo_ik_agent/arm_kinematics.py`: FK/IK via URDF. `get_wrist_camera_pose()` is what the preset approach replaces.
- `yolo_ik_agent/camera_calibration.py`: Intrinsic calibration, `pixel_to_table_ray()`, `triangulate_point()`.
- `yolo_ik_agent/calibrate_extrinsic.py`: New file (uncommitted). Checkerboard-based wrist extrinsic calibration using top camera as reference.
- `gemini_agent/gemini_robot_agent.py`: Reference for Gemini API usage patterns (model: `gemini-3-flash-preview`).
