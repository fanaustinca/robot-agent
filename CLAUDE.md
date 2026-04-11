# SO-101 Robot Agent

## What This Is

A robot control system for the SO-101 arm with two AI agents (Gemini and Claude) that share a common Flask HTTP server. The Gemini agent specializes in vision-guided pick-up tasks; the Claude agent is a general-purpose tool-use reasoner.

## Architecture

```
Agent (Gemini or Claude)  ──HTTP──>  robot_server.py (Flask :7878)  ──>  SO-101 arm + 2 cameras
```

- **robot_server.py** — Flask server on port 7878. Exposes `/status`, `/snapshot/{top,wrist}`, `/stream`, `/move`, `/move_preset`, `/enable`, `/teleop`, `/agent_state`, `/confirm_grip`. Bridges serial robot + USB cameras. Serves a live web dashboard.
- **gemini_robot_agent.py** — Two modes: (1) **pick-up mode** with top-camera prescan, visual alignment loop, floor calibration, and auto-confirm grip, (2) **free-agent mode** for general commands. Uses `gemini-3-flash-preview`.
- **claude_robot_agent.py** — Agentic tool-use loop with 5 tools (get_status, snapshot, move_joints, move_preset, confirm_move). Uses `claude-sonnet-4-6`.

## Running

Terminal 1 (server):
```bash
CAM_TOP=2 CAM_WRIST=4 python3 robot_server.py
# or: python3 robot_server.py --port /dev/ttyACM1 --leader-port /dev/ttyACM0
```

Terminal 2 (agent):
```bash
GEMINI_API_KEY=<key> python3 gemini_robot_agent.py
# or: ANTHROPIC_API_KEY=<key> python3 claude_robot_agent.py
```

Dashboard (live cameras + controls): http://localhost:7878/stream

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

## Development Rules

- **Dashboard command registration**: When adding a new `/command` to `_run_command_thread()` in `robot_server.py`, always update **both** of the following in `templates/stream.html`:
  1. The `cmdCommands` array — for autocomplete suggestions
  2. The `toggleHelp()` function's HTML table — for the Help button command list
- **Detection overlay**: When a dashboard command detects objects in camera frames, always annotate the frame (bounding box, center dot, label with confidence) and push it to `detection_overlay` so the user can verify the detection on the live stream. The overlay auto-expires after 10 seconds.

## Key Details

- **Joint space inversion**: Agent uses positive shoulder_lift = up; hardware inverts this. Conversion in `move_joints()` / `hardware_to_agent()`.
- **Crosshair offsets**: `CROSSHAIR_OFFSET_X=30, CROSSHAIR_OFFSET_Y=38` in both server and gemini agent — must stay in sync. This marks the gripper closing point on the wrist camera.
- **Degree scale ruler**: Purple ruler on wrist camera showing 1°/5°/10° markings (8px per degree, 10° ≈ 2x crosshair span).
- **Safety bounds**: shoulder_lift clamped to [-65, 50] degrees (agent space).
- **Slow moves**: 8-step interpolation with 0.08s delays to avoid jerky motion.
- **Image size**: Wrist 256x192, top 512x384. JPEG quality 60 — kept small for API cost.
- **Pickup detection**: Simple keyword matching (no API call) — see `PICKUP_KEYWORDS` list. Excludes ambiguous words like "get", "take", "lift" to avoid false positives. Object name extracted from `pick up "object"` or `pick up the object` format.
- **Camera-only mode**: Server runs without the arm if serial connection fails.
- **Startup health check**: Agent verifies server, arm, and cameras on boot with pass/fail summary.
- **Colored terminal output**: ANSI colors via `C` class — green=success, red=error, yellow=warning, blue=in-progress, cyan=info.
- **Joint aliases**: Short names (`sp`, `sl`, `ef`, `wf`, `wr`, `g` or `pan`, `lift`, `elbow`, `flex`, `roll`, `grip`) work in `/move`, `/pos`, `/torque-*` commands.

## Pickup Sequence (8 steps)

1. **Prescan** — top camera sent to Gemini to estimate pan + forward/back degrees to target
2. **Home** — move to home position (slow interpolation)
3. **Open gripper**
4. **Lower arm** — to SETUP_SHOULDER_LIFT (-40°)
5. **Apply prescan** — jump to estimated position
6. **Fine alignment** — wrist camera feedback loop (10 iterations max, type `done` or press Done on dashboard to skip)
7. **Confirm + grip** — auto-confirms after 10s, or confirm/cancel via dashboard
8. **Grip verification** — wrist snapshot sent to Gemini to check if object is held:
   - **Held** → proceed to drop position
   - **Not held** → user gets 3s to reject retry (n → arm goes home). If no rejection, arm returns to pre-lift position, re-aligns (5 iterations), and retries grip once.
   - **Inconclusive** → assumes held, proceeds to drop

## Floor Calibration

- **`/calib`** — manually calibrate floor distance. Lowers arm 1° at a time until motor stalls, records drop distance.
- **Auto-calibration** — runs automatically on first pickup if not yet calibrated.
- **`_floor_drop`** — calibrated degrees from setup position to floor. Used instead of hardcoded value when lowering to grip.

## Teleop

- **`/teleop`** or **`/teleop start [port]`** — starts leader-follower teleoperation at 60fps
- **`/teleop stop`** — stops teleop and disconnects leader
- Leader arm auto-detected as second ttyACM device, or set via `LEADER_PORT` env / `--leader-port` CLI
- Uses `SOLeaderTeleopConfig` from lerobot

## Web Dashboard

The `/stream` endpoint serves a full dashboard with:
- Side-by-side camera streams (top + wrist)
- **Agent Activity** panel — live phase badge, detail text, alignment progress bar with Done button, grip confirm/cancel buttons
- Joint positions (updates 4x/sec)
- Torque ON/OFF toggle
- Gripper open/close buttons
- Home/Default/Rest preset buttons
- Teleop start/stop with status indicator
- **Chat panel** — synced with CLI; send commands or natural language from the dashboard, see agent responses in real time

## Dependencies

flask, opencv-python, lerobot, google-genai>=1.51.0, anthropic

## Presets

| Name | Description |
|------|-------------|
| `home` | All joints 0 except wrist_roll = -90 |
| `default` | All joints 0 |
| `rest` | Folded position (shoulder_lift -102, elbow 97, wrist_flex 76, wrist_roll -86) |

## Direct Commands (Gemini Agent)

| Command | Alias | Description |
|---------|-------|-------------|
| `/help` | `/h` | Show all commands |
| `/home` | | Move to home preset (slow) |
| `/default` | | Move all joints to 0 (slow) |
| `/drop` | | Move to drop position (slow) |
| `/torque-on [motor]` | `/t-on` | Enable torque |
| `/torque-off [motor]` | `/t-off` | Disable torque |
| `/move <joint> <deg>` | | Move single joint |
| `/pos [motor]` | | Show joint positions |
| `/cam [top\|wrist]` | | Show camera info |
| `/maincam <top\|wrist>` | | Set free-mode camera |
| `/calib` | | Calibrate floor distance |
| `/teleop [start\|stop]` | | Leader arm teleoperation |
| `/doctor` | | Run full diagnostics |
| `/torque-h` | `/t-h` | Torque help |
| `/move-h` | | Move help |

## Server API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | Joint positions, torque, teleop, agent state |
| GET | `/snapshot/<top\|wrist>` | Base64 JPEG snapshot |
| GET | `/stream/<top\|wrist>` | MJPEG video stream |
| GET | `/stream` | Web dashboard |
| POST | `/move` | Move joints `{"joint": degrees}` |
| POST | `/move_preset` | Move to preset `{"pose": "home"}` |
| POST | `/enable` | Torque `{"enabled": true/false}` |
| POST | `/teleop` | Teleop `{"action": "start/stop"}` |
| POST | `/agent_state` | Agent pushes activity state |
| POST | `/confirm_grip` | Dashboard sends grip confirm |
| GET | `/chat?since=ID` | Get chat messages since given ID |
| POST | `/chat` | Dashboard sends chat message `{"text": "..."}` |
| POST | `/chat/push` | Agent pushes message `{"text": "...", "role": "agent\|user\|system"}` |
| GET | `/chat/pending` | Agent polls for dashboard-submitted messages |

## Chat Sync

- The dashboard chat and CLI share the same message log via the server.
- Messages sent from the dashboard are queued (`/chat/pending`) and picked up by the agent's input loop.
- The agent pushes its responses, user inputs, and system events to `/chat/push` so the dashboard sees them.
- ANSI color codes are stripped when rendering in the dashboard.

## File Sizes

- robot_server.py — ~870 lines
- gemini_robot_agent.py — ~1230 lines
- claude_robot_agent.py — ~305 lines
