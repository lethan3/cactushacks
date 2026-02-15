#!/usr/bin/env python3
"""
Bare-bones ReAct agent loop.

Sends a prompt to Ollama with all hardware tools defined.
When the model returns tool calls, executes them, appends results
to the message list, and re-sends. Loops until the model stops
calling tools.

Usage:
    python3 scripts/ollama_agent.py "Water plant 3 for 5 seconds then read all sensors"
    python3 scripts/ollama_agent.py   # interactive — prompts for input
"""

import sys, os, json

# Let us import from sensors/app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sensors"))

import ollama

MODEL = "qwen3:4b-instruct-2507-q4_K_M"
MAX_ROUNDS = 10

# ---------------------------------------------------------------------------
# Tool definitions (match the REPL DSL)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_x",
            "description": "Move the X-axis motor forward or reverse for a duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["forward", "reverse"]},
                    "ms": {"type": "integer", "description": "Duration in milliseconds."},
                },
                "required": ["direction", "ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_y",
            "description": "Move the Y-axis motor forward or reverse for a duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["forward", "reverse"]},
                    "ms": {"type": "integer", "description": "Duration in milliseconds."},
                },
                "required": ["direction", "ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pump_on",
            "description": "Turn the water pump on for a duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ms": {"type": "integer", "description": "Duration in milliseconds."},
                },
                "required": ["ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_soil",
            "description": "Read the soil moisture sensor. Returns moisture and capacitance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "avg_ms": {"type": "integer", "description": "Averaging window in ms. 0 = single read.", "default": 0},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_climate",
            "description": "Read the climate sensor. Returns temperature and humidity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "avg_ms": {"type": "integer", "description": "Averaging window in ms. 0 = single read.", "default": 0},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_air",
            "description": "Read the air quality sensor. Returns eCO2 and TVOC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "avg_ms": {"type": "integer", "description": "Averaging window in ms. 0 = single read.", "default": 0},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_all",
            "description": "Read all sensors at once (soil, climate, air quality).",
            "parameters": {
                "type": "object",
                "properties": {
                    "avg_ms": {"type": "integer", "description": "Averaging window in ms. 0 = single read.", "default": 0},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_plants",
            "description": (
                "Capture an image from the camera and analyze all visible plants "
                "using a cloud vision model. Returns a text report with species, "
                "leaf color, health score, whether watering is needed, and care "
                "recommendations for every plant in view. Takes a few seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Hardware handles (lazy init, graceful fallback)
# ---------------------------------------------------------------------------
_ctl = None
_sens = None
_hw_init_done = False


def _init_hw():
    global _ctl, _sens, _hw_init_done
    if _hw_init_done:
        return

    try:
        from app.controls import Controls
        _ctl = Controls().__enter__()
        print("[hw] motors ready")
    except Exception as e:
        print(f"[hw] motors unavailable: {e}")

    try:
        from app.sensors import Sensors
        _sens = Sensors().__enter__()
        print("[hw] sensors ready")
    except Exception as e:
        print(f"[hw] sensors unavailable: {e}")

    _hw_init_done = True


def _cleanup_hw():
    if _ctl is not None:
        try:
            _ctl.__exit__(None, None, None)
        except Exception:
            pass
    if _sens is not None:
        try:
            _sens.__exit__(None, None, None)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Plant vision (cloud VLM)
# ---------------------------------------------------------------------------
def _run_plant_vision() -> str:
    """Capture camera frame → Together AI VLM → plain-text report."""
    try:
        # Import the DSL from sibling module
        scripts_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(scripts_dir)
        sys.path.insert(0, parent_dir)
        from plant_vision import analyze_camera, analyze_image

        try:
            return analyze_camera()
        except RuntimeError:
            # Camera unavailable — try the default live frame
            fallback = os.path.join(parent_dir, "..", "camera_live_frame.png")
            if os.path.exists(fallback):
                return analyze_image(fallback)
            return "error: camera unavailable and no fallback image found"
    except Exception as e:
        return f"error: plant vision failed: {e}"


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
def execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name, return JSON string result."""
    _init_hw()

    try:
        if name == "move_x":
            if _ctl is None:
                return json.dumps({"error": "motors unavailable"})
            _ctl.move_x(args["direction"], args["ms"])
            return json.dumps({"ok": True, "action": "move_x", **args})

        if name == "move_y":
            if _ctl is None:
                return json.dumps({"error": "motors unavailable"})
            _ctl.move_y(args["direction"], args["ms"])
            return json.dumps({"ok": True, "action": "move_y", **args})

        if name == "pump_on":
            if _ctl is None:
                return json.dumps({"error": "motors unavailable"})
            _ctl.pump_on(args["ms"])
            return json.dumps({"ok": True, "action": "pump_on", **args})

        if name == "read_soil":
            if _sens is None:
                return json.dumps({"error": "sensors unavailable"})
            return json.dumps(_sens.read_soil(avg_ms=args.get("avg_ms", 0)))

        if name == "read_climate":
            if _sens is None:
                return json.dumps({"error": "sensors unavailable"})
            return json.dumps(_sens.read_climate(avg_ms=args.get("avg_ms", 0)))

        if name == "read_air":
            if _sens is None:
                return json.dumps({"error": "sensors unavailable"})
            return json.dumps(_sens.read_air(avg_ms=args.get("avg_ms", 0)))

        if name == "read_all":
            if _sens is None:
                return json.dumps({"error": "sensors unavailable"})
            return json.dumps(_sens.read_all(avg_ms=args.get("avg_ms", 0)))

        if name == "analyze_plants":
            return _run_plant_vision()

        return json.dumps({"error": f"unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are a hardware control agent. You have tools for motors, pump, sensors, "
    "and a camera-based plant vision analyzer. "
    "ALWAYS use tool calls to perform actions — never just describe them in text. "
    "When you need to move a motor, read a sensor, or inspect plants, call the tool. "
    "Use analyze_plants to take a photo and identify plant species, health, and care needs. "
    "Only respond with plain text when reporting final results to the user."
)


def run(prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n--- round {round_num} ---")
        resp = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.message

        # Append the assistant's response to the conversation
        messages.append(msg)

        # If no tool calls, the model is done
        if not msg.tool_calls:
            print(f"[agent] {msg.content}")
            break

        # Execute each tool call, append results
        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            print(f"[tool] {name}({args})")

            result = execute_tool(name, args)
            print(f"[result] {result}")

            messages.append({"role": "tool", "content": result})

    else:
        print(f"\n[agent] stopped after {MAX_ROUNDS} rounds")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            run(" ".join(sys.argv[1:]))
        else:
            while True:
                try:
                    prompt = input("\nprompt> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not prompt or prompt in ("q", "exit"):
                    break
                run(prompt)
    finally:
        _cleanup_hw()
