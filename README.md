# SO-101 Robot Agent

Bridges Claude (via OpenClaw) to your SO-101 arm and cameras over a local HTTP API.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Basic
python robot_server.py

# Custom camera/port
CAM_TOP=0 CAM_WRIST=1 ROBOT_PORT=/dev/ttyUSB0 python robot_server.py
```

## Finding your camera indices

```bash
ls /dev/video*
# Try index 0, 1, 2... until you find wrist and top
```

## Finding your robot serial port

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

## API

| Method | Endpoint           | Description                          |
|--------|--------------------|--------------------------------------|
| GET    | /status            | Health check + joint positions       |
| GET    | /snapshot/top      | Top camera image (base64 JPEG)       |
| GET    | /snapshot/wrist    | Wrist camera image (base64 JPEG)     |
| POST   | /move              | Move joints `{"shoulder_pan": 45}`   |
| POST   | /move_preset       | Named pose `{"pose": "home"}`        |
| POST   | /enable            | Torque on/off `{"enabled": true}`    |

## Presets

- `home` — all joints at 0
- `ready` — arm raised and ready to work
- `rest` — arm folded down

## How Claude uses this

Once the server is running, tell Claude:
> "The robot server is running at http://localhost:7878"

Claude can then:
- Ask for a snapshot: fetches the image, sends it for vision analysis
- Move the arm: sends POST /move with joint angles
- Check status: GET /status

## Image resolution

Default: **320x240** JPEG at quality 75 — keeps token cost low (~800-1200 tokens per image).
Adjust `SNAPSHOT_WIDTH`, `SNAPSHOT_HEIGHT`, `JPEG_QUALITY` in robot_server.py if needed.
