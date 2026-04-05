# YOLO + IK Robot Agent for SO-101

Vision-guided pick-and-place for the SO-101 robot arm using GroundingDINO object detection, camera-to-3D projection, and inverse kinematics.

## Pipeline

```
GroundingDINO (detect) -> OpenCV (pixel to 3D) -> IKPy (joint angles) -> LeRobot (move arm)
```

1. **GroundingDINO** detects any object by text description (open-vocabulary)
2. **Camera calibration** projects the 2D pixel center to a 3D point on the table
3. **IKPy** computes joint angles via inverse kinematics from the SO-101 URDF
4. **Robot server** sends joint commands to the arm with smooth interpolation

## Requirements

### Hardware
- **SO-101 robot arm** (6-DOF, STS3215 servos)
- **Top camera** — USB webcam, 1080p+ recommended (tested with Logitech C930e)
- **Wrist camera** — USB webcam (optional, for triangulation)

### GPU / CPU
- **NVIDIA GPU recommended** — GroundingDINO runs ~100-150ms/frame on an RTX 5060 or similar
- **CPU works** but detection will be slow (~1-2s/frame)
- CUDA toolkit + PyTorch with CUDA support required for GPU

### Software
- Python 3.10+
- conda/miniforge (for environment management)

## Setup

### 1. Create conda environment

```bash
conda create -n lerobot python=3.12
conda activate lerobot
```

### 2. Install dependencies

```bash
# Core
pip install flask opencv-python numpy requests ikpy

# LeRobot (for arm control)
pip install lerobot

# GroundingDINO (object detection)
pip install transformers torch torchvision

# Optional: ultralytics (if using YOLO-World instead)
pip install ultralytics
```

### 3. Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 4. Camera calibration

Print an 8x8 checkerboard pattern and tape it flat to cardboard. Then:

```bash
cd yolo_ik_agent
python camera_calibration.py --camera top --index <CAM_INDEX> --width 1920 --height 1080
```

- Hold the checkerboard at 15-20 different positions/angles in front of the camera
- Press SPACE to capture each frame
- Press C to run calibration
- Results saved to `calibration_data/top_camera.json`

Find your camera index with: `v4l2-ctl --list-devices`

### 5. Configure camera position

Edit `config.py` to match your physical setup:

```python
# Camera position relative to arm base (meters)
# +X = right, +Y = forward, +Z = up
TOP_CAM_X = -0.21   # 21cm to the left of arm
TOP_CAM_Y = 0.0     # even with arm base
TOP_CAM_Z = 0.55    # 55cm above table
TOP_CAM_PITCH = -45.0  # degrees below horizontal
```

## Running

### Terminal 1 — Robot Server

```bash
cd yolo_ik_agent
CAM_TOP=<top_index> CAM_WRIST=<wrist_index> python3 robot_server.py
```

The server opens cameras at full resolution (1920x1080 for top) and serves:
- **http://localhost:7878/stream** — YOLO debug viewer with live cameras, detection, ROI selection
- **http://localhost:7878/dashboard** — original robot dashboard
- REST API for arm control (`/move`, `/status`, `/snapshot`, etc.)

### Terminal 2 — YOLO IK Agent

```bash
cd yolo_ik_agent
python3 yolo_ik_agent.py
```

### Agent Commands

| Command | Description |
|---------|-------------|
| `pick up <object>` | Full pickup sequence: detect, move, grip, lift, drop |
| `/detect [label]` | Run detection on top camera |
| `/locate [label]` | Detect + compute 3D position |
| `/ik <right> <fwd> <up>` | Move to position (cm) |
| `/fk` | Show current gripper position |
| `/t [on\|off]` | Toggle torque |
| `/home` | Move to home position |
| `/ready` | Move to ready position |
| `/info` | System status |
| `/help` | Show all commands |

### Web Debug Viewer

Open **http://localhost:7878/stream** to:
- See live camera streams (top + wrist)
- Type any text label and click **Start Detecting** to run GroundingDINO
- Draw an **ROI** box to limit detection to part of the image
- See bounding boxes, confidence scores, and computed 3D positions

## Architecture

```
yolo_ik_agent/
  robot_server.py      — Flask server (cameras, arm control, web viewer)
  yolo_ik_agent.py     — Interactive CLI agent (detect, IK, pickup)
  detect.py            — GroundingDINO detection wrapper
  config.py            — All configuration (server, arm, camera, detection)
  arm_kinematics.py    — FK/IK using IKPy + SO-101 URDF
  camera_calibration.py — Checkerboard calibration + pixel-to-3D projection
  so101.urdf           — Robot arm description file
  calibration_data/    — Saved camera calibration files
```

## Detection Model

Uses **GroundingDINO** (IDEA-Research/grounding-dino-tiny) — an open-vocabulary object detector that understands natural language descriptions. You can detect anything by describing it:
- "red block"
- "small shiny object"
- "tape roll"
- "cup"

The model auto-downloads (~690MB) on first run.

## Coordinate System

Physical world frame (relative to arm base on table):
- **+X** = right
- **+Y** = forward (away from arm)
- **+Z** = up

The agent automatically converts between this physical frame and the URDF coordinate frame used by IKPy.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAM_TOP` | 4 | Top camera /dev/video index |
| `CAM_WRIST` | 0 | Wrist camera /dev/video index |
| `ROBOT_PORT` | /dev/ttyACM0 | Serial port for arm |
| `ROBOT_SERVER` | http://localhost:7878 | Server URL |
| `YOLO_MODEL` | yolov8x-worldv2.pt | Detection model (if using YOLO) |
