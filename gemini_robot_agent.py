"""
Gemini Robot Agent
Connects Gemini 3.1 Pro to your SO-101 arm via the robot server.

Usage:
    pip install -r requirements.txt
    GEMINI_API_KEY=... python gemini_robot_agent.py

Make sure robot_server.py is running first:
    python robot_server.py
"""

import base64
import json
import os
import sys
import requests
from google import genai
from google.genai import types

ROBOT_SERVER = os.environ.get("ROBOT_SERVER", "http://localhost:7878")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3.1-pro-preview"
MAX_HISTORY_TURNS = 6  # keep last N user/assistant pairs to limit token growth

SYSTEM = """You are an AI assistant controlling a SO-101 6-DOF robot arm.

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

# ---- Tool definitions ----
FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_status",
        description="Get the robot arm's current joint positions and check if cameras are connected.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={})
    ),
    types.FunctionDeclaration(
        name="snapshot",
        description="Take a photo from one of the robot's cameras. Use this to see what the robot sees.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera": types.Schema(
                    type=types.Type.STRING,
                    enum=["top", "wrist"],
                    description="Which camera to use. 'top' is the overhead view, 'wrist' is mounted on the arm."
                )
            },
            required=["camera"]
        )
    ),
    types.FunctionDeclaration(
        name="move_joints",
        description="Move the robot arm joints to specific angles in degrees. Only specify the joints you want to move.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "shoulder_pan":  types.Schema(type=types.Type.NUMBER, description="Rotate base left/right (-150 to 150)"),
                "shoulder_lift": types.Schema(type=types.Type.NUMBER, description="Raise/lower shoulder (-90 to 45)"),
                "elbow_flex":    types.Schema(type=types.Type.NUMBER, description="Bend elbow (0 to 150)"),
                "wrist_flex":    types.Schema(type=types.Type.NUMBER, description="Bend wrist up/down (-90 to 90)"),
                "wrist_roll":    types.Schema(type=types.Type.NUMBER, description="Rotate wrist (-180 to 180)"),
                "gripper":       types.Schema(type=types.Type.NUMBER, description="Open/close gripper (0=closed, 100=open)"),
            }
        )
    ),
    types.FunctionDeclaration(
        name="move_preset",
        description="Move the arm to a named preset position.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "pose": types.Schema(
                    type=types.Type.STRING,
                    enum=["home", "ready", "rest"],
                    description="home=all zeros, ready=raised and ready to work, rest=folded down"
                )
            },
            required=["pose"]
        )
    ),
    types.FunctionDeclaration(
        name="confirm_move",
        description="Describe the move you are about to make and wait for user approval before executing it. Use this before any move_joints or move_preset call that could cause significant arm movement.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "plan": types.Schema(
                    type=types.Type.STRING,
                    description="Human-readable description of what you plan to do and why."
                )
            },
            required=["plan"]
        )
    ),
]

TOOLS = [types.Tool(function_declarations=FUNCTION_DECLARATIONS)]


# ---- Tool execution ----
def run_tool(name, inputs):
    try:
        if name == "get_status":
            r = requests.get(f"{ROBOT_SERVER}/status", timeout=5)
            return r.json()

        elif name == "confirm_move":
            plan = inputs.get("plan", "")
            print(f"\n[confirm] Gemini's plan: {plan}")
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


def build_tool_result_parts(name, result):
    """Build Gemini content parts for a tool result, handling images specially."""
    if name == "snapshot" and "image_base64" in result:
        image_bytes = base64.b64decode(result["image_base64"])
        return [
            types.Part.from_function_response(
                name=name,
                response={"camera": result["camera"], "width": result["width"], "height": result["height"]}
            ),
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        ]
    else:
        return [types.Part.from_function_response(name=name, response=result)]


# ---- Main agent loop ----
def run_agent():
    if not API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)
    config = types.GenerateContentConfig(system_instruction=SYSTEM, tools=TOOLS)
    history = []

    print("=" * 50)
    print("SO-101 Gemini Robot Agent")
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
    except Exception:
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

        history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(f"{joint_context}{user_input}")]
        ))

        # Trim history to last MAX_HISTORY_TURNS user/assistant pairs
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]

        # Agentic loop — keep going until Gemini stops using tools
        while True:
            response = client.models.generate_content(
                model=MODEL,
                contents=history,
                config=config
            )

            candidate = response.candidates[0]
            history.append(candidate.content)

            text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
            function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

            if text_parts:
                print(f"\nGemini: {''.join(text_parts)}")

            if not function_calls:
                break

            # Execute tools and collect results
            tool_result_parts = []
            for fc in function_calls:
                print(f"\n[tool] {fc.name}({json.dumps(dict(fc.args))})")
                result = run_tool(fc.name, dict(fc.args))
                if "error" not in result:
                    print(f"[tool] ✓ {fc.name} done")
                else:
                    print(f"[tool] ✗ {result['error']}")
                tool_result_parts.extend(build_tool_result_parts(fc.name, result))

            history.append(types.Content(role="user", parts=tool_result_parts))

        print()


if __name__ == "__main__":
    run_agent()
