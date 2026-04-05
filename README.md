# SO-101 Robot Agent

Multiple AI agents for controlling a SO-101 robot arm via a shared Flask HTTP server.

## Project Structure

```
├── robot_server.py          # Shared Flask server (arm + cameras, port 7878)
├── requirements.txt         # Python dependencies
├── CLAUDE.md                # Codebase documentation for Claude Code
│
├── gemini_agent/            # Gemini-powered visual alignment agent
│   ├── gemini_robot_agent.py
│   └── HOW_TO_RUN.txt
│
├── claude_agent/            # Claude tool-use agent
│   └── claude_robot_agent.py
│
└── yolo_ik_agent/           # YOLO + IK pick-and-place agent
    ├── robot_server.py      # High-res server (1920x1080, GroundingDINO viewer)
    ├── yolo_ik_agent.py     # Interactive CLI agent
    ├── detect.py            # GroundingDINO detection
    ├── config.py            # Configuration
    ├── arm_kinematics.py    # FK/IK via IKPy + URDF
    ├── camera_calibration.py
    ├── so101.urdf
    ├── calibration_data/
    └── README.md            # Detailed setup guide
```

## Agents

### Gemini Agent (`gemini_agent/`)
Uses Gemini's vision to iteratively align the wrist camera over objects. Two modes: pick-up (visual alignment loop) and free-agent (general commands).

### Claude Agent (`claude_agent/`)
General-purpose tool-use agent with 5 tools (get_status, snapshot, move_joints, move_preset, confirm_move). Uses Claude Sonnet.

### YOLO + IK Agent (`yolo_ik_agent/`)
Vision-guided pick-and-place using GroundingDINO detection, camera-to-3D projection, and inverse kinematics. See [yolo_ik_agent/README.md](yolo_ik_agent/README.md) for full setup.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the shared server (Terminal 1)
```bash
CAM_TOP=4 CAM_WRIST=1 python3 robot_server.py
```

### 3. Start an agent (Terminal 2)

**Gemini:**
```bash
GEMINI_API_KEY=<key> python3 gemini_agent/gemini_robot_agent.py
```

**Claude:**
```bash
ANTHROPIC_API_KEY=<key> python3 claude_agent/claude_robot_agent.py
```

**YOLO + IK** (uses its own high-res server):
```bash
cd yolo_ik_agent
CAM_TOP=4 CAM_WRIST=1 python3 robot_server.py   # Terminal 1
python3 yolo_ik_agent.py                          # Terminal 2
```

### 4. Dashboard
- **http://localhost:7878/stream** — live cameras + controls

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAM_TOP` | 4 | Top camera /dev/video index |
| `CAM_WRIST` | 0 | Wrist camera /dev/video index |
| `ROBOT_PORT` | /dev/ttyACM0 | Serial port for arm |
| `GEMINI_API_KEY` | — | Required for Gemini agent |
| `ANTHROPIC_API_KEY` | — | Required for Claude agent |

## Server API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /status | Joint positions + health |
| GET | /snapshot/top | Top camera (base64 JPEG) |
| GET | /snapshot/wrist | Wrist camera (base64 JPEG) |
| GET | /stream | Live dashboard |
| POST | /move | Move joints `{"shoulder_pan": 45}` |
| POST | /move_preset | Named pose `{"pose": "home"}` |
| POST | /enable | Torque `{"enabled": true}` |
