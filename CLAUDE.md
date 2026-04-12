# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A robot control system for the SO-101 arm with three AI agents that share a common Flask HTTP server. Each agent is in its own subdirectory:

- **`gemini_agent/`** — Vision-guided pick-up (top-camera prescan, wrist-camera alignment loop, floor calibration) + free-agent mode. Uses `gemini-3-flash-preview`.
- **`claude_agent/`** — General-purpose tool-use reasoner with 5 tools. Uses `claude-sonnet-4-6`.
- **`yolo_ik_agent/`** — GroundingDINO detection → camera-to-3D projection → IKPy inverse kinematics → arm control. Has its own high-res server variant and template-based dashboard.

## Architecture

```
Agent (gemini / claude / yolo_ik)  ──HTTP──>  robot_server.py (Flask :7878)  ──>  SO-101 arm + 2 cameras
```

- **`robot_server.py`** (1500+ lines) — Main Flask server on port 7878. Bridges serial robot + USB cameras. Serves an inline HTML dashboard at `/stream`. Exposes REST endpoints for move, snapshot, teleop, chat sync, agent state, etc.
- **`yolo_ik_agent/robot_server.py`** (1600+ lines) — Separate high-res server variant (1920×1080) with GroundingDINO viewer. Uses Jinja templates in `yolo_ik_agent/templates/`.

## Running

Terminal 1 (server):
```bash
CAM_TOP=2 CAM_WRIST=4 python3 robot_server.py
# or with explicit ports:
python3 robot_server.py --port /dev/ttyACM1 --leader-port /dev/ttyACM0
```

Terminal 2 (agent — pick one):
```bash
GEMINI_API_KEY=<key> python3 gemini_agent/gemini_robot_agent.py
ANTHROPIC_API_KEY=<key> python3 claude_agent/claude_robot_agent.py
# YOLO+IK uses its own server:
cd yolo_ik_agent && python3 robot_server.py  # Terminal 1
python3 yolo_ik_agent.py                     # Terminal 2
```

Dashboard: http://localhost:7878/stream

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAM_TOP` | 4 | Top camera /dev/video index |
| `CAM_WRIST` | 0 | Wrist camera /dev/video index |
| `ROBOT_PORT` | /dev/ttyACM0 | Serial port for follower arm |
| `LEADER_PORT` | (auto-detect) | Serial port for leader arm (teleop) |
| `GEMINI_API_KEY` | — | Required for gemini agent |
| `ANTHROPIC_API_KEY` | — | Required for claude agent |
| `ROBOT_SERVER` | http://localhost:7878 | Server URL for agents |

## Dependencies

```
flask, opencv-python, lerobot, anthropic, google-genai>=1.51.0
# YOLO+IK additionally: transformers, torch, torchvision, ikpy, numpy
```

Python env: conda `lerobot` environment (Python 3.12). See `yolo_ik_agent/README.md` for GPU/CUDA setup.

## Development Rules

- **Dashboard command registration** (main server): When adding a new `/command` to `_run_command_thread()` in `robot_server.py`, the dashboard HTML is **inline** in that same file (not a template). Update both the `cmdCommands` array and the `toggleHelp()` HTML table.
- **Dashboard command registration** (YOLO+IK server): Templates are in `yolo_ik_agent/templates/`. Same two-place update rule applies there.
- **Detection overlay**: When a dashboard command detects objects in camera frames, annotate the frame (bounding box, center dot, label with confidence) and push to `detection_overlay` so the user can verify on the live stream. Overlay auto-expires after 10 seconds.

## Critical Gotchas

- **Joint space inversion**: Agent uses positive `shoulder_lift` = up; hardware inverts this. Conversion happens in `move_joints()` / `hardware_to_agent()`. Get this wrong and the arm moves opposite to intended.
- **Crosshair offsets must stay in sync**: `CROSSHAIR_OFFSET_X` and `CROSSHAIR_OFFSET_Y` are defined in both `robot_server.py` and `gemini_agent/gemini_robot_agent.py`. The values mark the gripper closing point on the wrist camera. The main server and gemini agent may have different Y values (server=53, gemini=38) — check current values before changing either.
- **Safety bounds**: `shoulder_lift` clamped to [-65, 50] degrees in agent space.
- **Slow moves**: 8-step interpolation with 0.08s delays to avoid jerky motion. Don't bypass this.
- **Image sizes**: Wrist 256×192, top 512×384, JPEG quality 60 — kept small for API cost. YOLO+IK agent uses full-res (1280×720 top, 640×480 wrist) since detection runs locally.
- **Camera-only mode**: Server runs without the arm if serial connection fails — useful for testing camera/dashboard work.
- **Joint aliases**: Short names (`sp`, `sl`, `ef`, `wf`, `wr`, `g` or `pan`, `lift`, `elbow`, `flex`, `roll`, `grip`) work in `/move`, `/pos`, `/torque-*` commands.

## YOLO+IK Agent Specifics

- Pipeline: `GroundingDINO (detect.py) → OpenCV pixel-to-3D (camera_calibration.py) → IKPy (arm_kinematics.py + so101.urdf) → robot_server`
- Config in `yolo_ik_agent/config.py` — arm geometry, camera extrinsics, table height, URDF-to-physical coordinate rotation
- Camera calibration data in `yolo_ik_agent/calibration_data/cameras.json`
- Coordinate frames: Physical (+X=right, +Y=forward, +Z=up) vs URDF (+X=forward, +Y=left, +Z=up). Rotation matrix `URDF_TO_PHYS_ROT` in config.py handles conversion.
- Board-to-arm coordinate conversion uses `arm_offset` — see recent commits.

## Pickup Sequence (Gemini Agent, 8 steps)

1. **Prescan** — top camera → Gemini estimates pan + forward/back degrees
2. **Home** → **Open gripper** → **Lower arm** to setup position
3. **Apply prescan** — jump to estimated position
4. **Fine alignment** — wrist camera feedback loop (10 iterations max, `done` to skip)
5. **Confirm + grip** — auto-confirms after 10s, or confirm/cancel via dashboard
6. **Grip verification** — wrist snapshot → Gemini checks if held → retry once if not

## Floor Calibration

- `/calib` — lowers arm 1° at a time until motor stalls, records drop distance
- Auto-calibration runs on first pickup if not yet calibrated
- `_floor_drop` stores calibrated degrees from setup position to floor

## Arm Offset Calibration

- **`/calib_arm <x_cm> <y_cm>`** — place gripper tip at a known board position, collect a sample (FK reads arm position, user provides board coordinates)
- **`/calib_arm solve`** — averages `board_pos - arm_pos` across samples, saves to `cameras.json` `arm_offset[0]` and `[1]` (Z untouched). Restart server after.
- **`/calib_arm status`** — show collected samples with per-sample offsets
- **`/calib_arm del <n>`** / **`/calib_arm clear`** — manage samples
- Samples persisted in `calibration_data/calib_arm_samples.json`

## Teleop

- `/teleop [start|stop]` — leader-follower at 60fps via `SOLeaderTeleopConfig` from lerobot
- Leader arm auto-detected as second ttyACM device, or set via `LEADER_PORT`

## Chat Sync

Dashboard chat and CLI share the same message log via the server. Messages from dashboard queue at `/chat/pending` and are picked up by the agent's input loop. Agent pushes responses to `/chat/push`. ANSI colors are stripped in dashboard rendering.
