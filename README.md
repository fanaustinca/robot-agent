# SO-101 Gemini Robot Agent

Controls a SO-101 robot arm using Gemini's vision capabilities. The agent uses the wrist camera to visually align the gripper over a target object and pick it up.

## How it works

Two components run together:

- **`robot_server.py`** — Flask HTTP server (port 7878) that talks directly to the SO-101 arm and cameras
- **`gemini_robot_agent.py`** — Gemini-powered agent that controls the arm via the server

### Two modes

**Pick-up mode** — triggered when your prompt is a grab/pick-up request:
1. Move arm to home position
2. Open gripper
3. Lower arm toward the workspace
4. Visual alignment loop (up to 10 iterations):
   - Captures wrist camera image with a black dot crosshair drawn at the gripper tip
   - Early iterations also include a top-down overview camera
   - Gemini looks at the image and responds with a direction to move (`forward`, `backward`, `left`, `right`) or `aligned: true`
   - Agent translates direction → joint moves and repeats
5. Asks "Close gripper? (y/n)" — lowers, grips slowly, then lifts

**Free agent mode** — for any other prompt (e.g. "go to home", "wave"):
- Gemini gets a top camera snapshot + current joint state
- Responds with a JSON action to move joints, use a preset, or check status

> Intent detection uses keyword matching (pick up, grab, retrieve, get, fetch, take, lift, collect) — no extra API call needed.

## Quick Start

**Step 1 — Install dependencies (first time only):**
```bash
pip install -r requirements.txt
```

**Step 2 — Start the robot server (Terminal 1):**
```bash
python3 robot_server.py
```

Defaults to `/dev/ttyACM0`. Override with `--port`:
```bash
python3 robot_server.py --port /dev/ttyACM1
```

Wait until you see:
```
[boot] Server running on http://0.0.0.0:7878
[camera] top camera opened ...
[camera] wrist camera opened ...
```

**Step 3 — Start the Gemini agent (Terminal 2):**
```bash
GEMINI_API_KEY=your_key_here python3 gemini_robot_agent.py
```

**Step 4 — Chat to control the arm:**
```
You: pick up the red block
You: go to home position
You: open the gripper
```

Type `quit` to exit.

## Custom camera indices / serial port

```bash
# via command-line argument (recommended)
python3 robot_server.py --port /dev/ttyACM1

# via environment variables
CAM_TOP=0 CAM_WRIST=1 ROBOT_PORT=/dev/ttyUSB0 python3 robot_server.py
```

**Find your devices:**
```bash
ls /dev/video*           # cameras — try indices 0, 1, 2...
ls /dev/ttyUSB* /dev/ttyACM*   # robot serial port
```

The server will auto-detect cameras and serial ports if the defaults aren't found.

## API

| Method | Endpoint         | Description                        |
|--------|------------------|------------------------------------|
| GET    | /status          | Health check + joint positions     |
| GET    | /snapshot/top    | Top camera image (base64 JPEG)     |
| GET    | /snapshot/wrist  | Wrist camera image (base64 JPEG)   |
| GET    | /stream          | Live browser view of both cameras  |
| POST   | /move            | Move joints `{"shoulder_pan": 45}` |
| POST   | /move_preset     | Named pose `{"pose": "home"}`      |
| POST   | /enable          | Torque on/off `{"enabled": true}`  |

## Presets

- `home` — all joints at 0
- `ready` — arm raised and ready to work
- `rest` — arm folded down

## Image resolution

Default: **256x192** JPEG at quality 60 — keeps API cost low.
Adjust `SNAPSHOT_WIDTH`, `SNAPSHOT_HEIGHT`, `JPEG_QUALITY` in `robot_server.py` if needed.
