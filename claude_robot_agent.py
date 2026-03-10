"""
Claude Robot Agent
Standalone script that connects Claude to your SO-101 arm via the robot server.
No OpenClaw needed — just the Anthropic API + your robot_server.py running.

Usage:
    pip install anthropic requests
    ANTHROPIC_API_KEY=sk-ant-... python claude_robot_agent.py

Make sure robot_server.py is running first:
    python robot_server.py
"""

import anthropic
import requests
import base64
import json
import os
import sys

ROBOT_SERVER = os.environ.get("ROBOT_SERVER", "http://localhost:7878")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"
MAX_HISTORY_TURNS = 6  # keep last N user/assistant pairs to limit token growth

# ---- Tool definitions (Claude sees these) ----
TOOLS = [
    {
        "name": "get_status",
        "description": "Get the robot arm's current joint positions and check if cameras are connected.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "snapshot",
        "description": "Take a photo from one of the robot's cameras. Use this to see what the robot sees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["top", "wrist"],
                    "description": "Which camera to use. 'top' is the overhead view, 'wrist' is mounted on the arm."
                }
            },
            "required": ["camera"]
        }
    },
    {
        "name": "move_joints",
        "description": "Move the robot arm joints to specific angles in degrees. Only specify the joints you want to move.",
        "input_schema": {
            "type": "object",
            "properties": {
                "shoulder_pan":  {"type": "number", "description": "Rotate base left/right (-180 to 180)"},
                "shoulder_lift": {"type": "number", "description": "Raise/lower shoulder (-90 to 90)"},
                "elbow_flex":    {"type": "number", "description": "Bend elbow (0 to 180)"},
                "wrist_flex":    {"type": "number", "description": "Bend wrist up/down (-90 to 90)"},
                "wrist_roll":    {"type": "number", "description": "Rotate wrist (-180 to 180)"},
                "gripper":       {"type": "number", "description": "Open/close gripper (0=closed, 100=open)"}
            },
            "required": []
        }
    },
    {
        "name": "move_preset",
        "description": "Move the arm to a named preset position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pose": {
                    "type": "string",
                    "enum": ["home", "ready", "rest"],
                    "description": "home=all zeros, ready=raised and ready to work, rest=folded down"
                }
            },
            "required": ["pose"]
        }
    },
    {
        "name": "confirm_move",
        "description": "Describe the move you are about to make and wait for user approval before executing it. Use this before any move_joints or move_preset call that could cause significant arm movement or is irreversible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "Human-readable description of what you plan to do and why."
                }
            },
            "required": ["plan"]
        }
    }
]

# ---- Tool execution ----
def run_tool(name, inputs):
    try:
        if name == "get_status":
            r = requests.get(f"{ROBOT_SERVER}/status", timeout=5)
            return r.json()

        elif name == "confirm_move":
            plan = inputs.get("plan", "")
            print(f"\n[confirm] Claude's plan: {plan}")
            answer = input("[confirm] Approve? (y/n): ").strip().lower()
            if answer == "y":
                return {"approved": True}
            else:
                return {"approved": False, "message": "User rejected the move. Do not proceed."}

        elif name == "snapshot":
            camera = inputs["camera"]
            r = requests.get(f"{ROBOT_SERVER}/snapshot/{camera}", timeout=10)
            data = r.json()
            if "error" in data:
                return {"error": data["error"]}
            # Return image as base64 for Claude vision
            return {
                "camera": camera,
                "image_base64": data["data"],
                "width": data["width"],
                "height": data["height"]
            }

        elif name == "move_joints":
            r = requests.post(f"{ROBOT_SERVER}/move", json=inputs, timeout=10)
            return r.json()

        elif name == "move_preset":
            r = requests.post(f"{ROBOT_SERVER}/move_preset", json=inputs, timeout=10)
            return r.json()

        else:
            return {"error": f"Unknown tool: {name}"}

    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to robot server at {ROBOT_SERVER}. Is robot_server.py running?"}
    except Exception as e:
        return {"error": str(e)}


def format_tool_result(tool_name, tool_use_id, result):
    """Format tool result for Claude, handling images specially."""
    if tool_name == "snapshot" and "image_base64" in result:
        # Send image as vision content
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the {result['camera']} camera image ({result['width']}x{result['height']}):"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": result["image_base64"]
                    }
                }
            ]
        }
    else:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": json.dumps(result, indent=2)
        }


# ---- Main agent loop ----
def run_agent():
    if not API_KEY:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=API_KEY)
    messages = []

    system = """You are an AI assistant controlling a SO-101 6-DOF robot arm.

## Joints and limits
- shoulder_pan:  rotate base left/right.  Safe range: -150 to 150 deg
- shoulder_lift: raise/lower shoulder.    Safe range: -90 to 45 deg  (negative = raised)
- elbow_flex:    bend elbow.              Safe range: 0 to 150 deg
- wrist_flex:    bend wrist up/down.      Safe range: -90 to 90 deg
- wrist_roll:    rotate wrist.            Safe range: -180 to 180 deg
- gripper:       open/close hand.         0 = fully closed, 100 = fully open

## Cameras
- top: overhead wide-angle view of the workspace
- wrist: close-up view from the end of the arm, moves with the arm

## Rules
1. Before any significant movement, call confirm_move to describe your plan and get user approval.
2. Never exceed the safe ranges listed above.
3. Move one joint group at a time when navigating to a new position.
4. If a move is rejected or an error occurs, stop and report back — do not retry automatically.
5. After completing a physical task, take a wrist snapshot to confirm the outcome.

## Workflow for physical tasks
1. Call get_status to know current joint positions.
2. Take a top snapshot to understand the scene.
3. Call confirm_move with your plan.
4. Execute moves incrementally.
5. Take a wrist snapshot to verify the result."""

    print("=" * 50)
    print("SO-101 Claude Robot Agent")
    print(f"Robot server: {ROBOT_SERVER}")
    print(f"Model: {MODEL}")
    print("Type 'quit' to exit")
    print("=" * 50)

    # Quick connectivity check
    try:
        r = requests.get(f"{ROBOT_SERVER}/status", timeout=3)
        status = r.json()
        print(f"✓ Robot server connected")
        print(f"  Cameras: {status.get('cameras', [])}")
        print(f"  Arm: {'connected' if status.get('robot_connected') else 'NOT connected (camera-only mode)'}")
    except:
        print(f"⚠ WARNING: Cannot reach robot server at {ROBOT_SERVER}")
        print("  Start robot_server.py first!")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Inject current joint state as context prefix
        joint_context = ""
        try:
            r = requests.get(f"{ROBOT_SERVER}/status", timeout=3)
            joints = r.json().get("joints") or {}
            if joints:
                joint_str = ", ".join(f"{k}: {v:.1f}" for k, v in joints.items())
                joint_context = f"[Current joint positions: {joint_str}]\n\n"
        except Exception:
            pass
        messages.append({"role": "user", "content": f"{joint_context}{user_input}"})

        # Trim history to last MAX_HISTORY_TURNS user/assistant pairs
        if len(messages) > MAX_HISTORY_TURNS * 2:
            messages = messages[-(MAX_HISTORY_TURNS * 2):]

        # Agentic loop — keep going until Claude stops using tools
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system,
                tools=TOOLS,
                messages=messages
            )

            # Collect text output
            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if text_parts:
                print(f"\nClaude: {''.join(text_parts)}")

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response.content})

            # If no tool calls, we're done
            if response.stop_reason == "end_turn" or not tool_uses:
                break

            # Execute tools
            tool_results = []
            for tool_use in tool_uses:
                print(f"\n[tool] {tool_use.name}({json.dumps(tool_use.input)})")
                result = run_tool(tool_use.name, tool_use.input)
                if "error" not in result:
                    print(f"[tool] ✓ {tool_use.name} done")
                else:
                    print(f"[tool] ✗ {result['error']}")
                tool_results.append(format_tool_result(tool_use.name, tool_use.id, result))

            # Feed results back to Claude
            messages.append({"role": "user", "content": tool_results})

        print()


if __name__ == "__main__":
    run_agent()
