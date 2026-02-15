#!/usr/bin/env python3
"""Main harness for running the plant care AI agent."""

import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

from utils.plant import VirtualPlant, ActualPlant
from utils.clock import SimulatedClock
from utils.priority_queue import TaskQueue
from agent import PlantCareAgent
from motion.camera import ActualCamera, Camera
from motion.camera_calibrator import CameraCalibrator
from utils.overshoot_ai import OvershootAIClient

TOTAL_PLANTS = 1


def update_plant_states(plants, hours_passed):
    for plant in plants:
        if isinstance(plant, VirtualPlant):
            plant.update_state(hours_passed)


def print_tool_results(tool_results):
    if not tool_results:
        return
    print(f"Executed {len(tool_results)} tool call(s):")
    for tr in tool_results:
        msg = tr.get("result", {}).get("message", tr.get("result", {}))
        print(f"  - {tr.get('tool', '?')}: {msg}")


def main():
    parser = argparse.ArgumentParser(description="Plant Care AI Agent Harness")
    parser.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M", help="Ollama model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout for Ollama API calls")
    parser.add_argument("--time-step", type=int, default=30, help="Time step in minutes")
    parser.add_argument("--steps", type=int, default=5, help="Number of time steps")
    parser.add_argument("--real-time", action="store_true", help="Run in real-time mode")
    parser.add_argument("--poke-ai-key", default=None, help="Poke AI API key (or set POKE_AI_API_KEY env var)")
    parser.add_argument("--poke-ai-url", default=None, help="Poke AI API URL (or set POKE_AI_API_URL env var)")
    parser.add_argument("--poke-ai-disable", action="store_true", help="Disable Poke AI integration")
    parser.add_argument("--initial-x-inches", type=float, default=0.0, help="Initial camera x position in inches")
    parser.add_argument("--initial-y-inches", type=float, default=0.0, help="Initial camera y position in inches")
    parser.add_argument("--plant-positions", nargs="+", type=float, default=[], help="Plant positions in inches as x1,y1 x2,y2 ... (default: all at origin)")
    parser.add_argument("--overshoot-api-key", default=None, help="Overshoot.ai API key (or set OVERSHOOT_AI_API_KEY env var)")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip camera calibration")
    args = parser.parse_args()

    # Initialize camera at the very beginning
    camera = ActualCamera()
    
    # Convert initial camera position from inches to pixels
    initial_x_pixels = int(args.initial_x_inches / Camera.INCH_PER_PIXEL_OF_WIDTH)
    initial_y_pixels = int(args.initial_y_inches / Camera.INCH_PER_PIXEL_OF_WIDTH)
    
    # Initialize Overshoot.ai client
    try:
        overshoot_client = OvershootAIClient(api_key=args.overshoot_api_key)
        print("Overshoot.ai client initialized")
    except Exception as e:
        print(f"Warning: Could not initialize Overshoot.ai client: {e}")
        overshoot_client = None
    
    # Call calibrate on the camera at the very beginning
    if not args.skip_calibration:
        try:
            print("Calibrating camera...")
            calibrator = CameraCalibrator(camera, debug=False)
            qx, qy = calibrator.calibrate()
            print(f"Camera calibration complete: Qx={qx}, Qy={qy}")
            # Force set camera cx and cy to 0 to synchronize position tracking
            camera.cx = 0
            camera.cy = 0
            print("Camera position reset to (0, 0) after calibration")
        except Exception as e:
            print(f"Warning: Camera calibration failed: {e}")
            print("Continuing without calibration...")
            # Still reset position even if calibration failed
            camera.cx = 0
            camera.cy = 0
    
    clock = SimulatedClock(time_step_minutes=args.time_step)
    task_queue = TaskQueue()

    # Parse plant positions from inches to pixels
    plant_positions = []
    if args.plant_positions:
        # Expect pairs: x1,y1 x2,y2 ...
        if len(args.plant_positions) % 2 != 0:
            raise ValueError("Plant positions must be pairs of x,y coordinates")
        for i in range(0, len(args.plant_positions), 2):
            x_inches = args.plant_positions[i]
            y_inches = args.plant_positions[i + 1]
            # Convert inches to pixels
            x_pixels = int(x_inches / Camera.INCH_PER_PIXEL_OF_WIDTH)
            y_pixels = int(y_inches / Camera.INCH_PER_PIXEL_OF_WIDTH)
            plant_positions.append((x_pixels, y_pixels))
    
    # Create plants with positions in pixels
    plants = []
    for i in range(TOTAL_PLANTS):
        if i < len(plant_positions):
            pos_x, pos_y = plant_positions[i]
        else:
            # Default to origin if not specified
            pos_x, pos_y = 0, 0
        plants.append(ActualPlant(f"Cactus{i+1}", "cactus", i, pos_x=pos_x, pos_y=pos_y))
    
    assert len(plants) == TOTAL_PLANTS, f"Expected {TOTAL_PLANTS} plants, got {len(plants)}"

    agent = PlantCareAgent(
        model=args.model,
        host=args.host,
        timeout=args.timeout,
        poke_ai_api_key=args.poke_ai_key,
        poke_ai_api_url=args.poke_ai_url,
        poke_ai_enabled=not args.poke_ai_disable
    )
    agent.register_plants(plants)
    agent.task_queue = task_queue
    agent.clock = clock
    # Store camera and overshoot client in agent for use in tool execution
    agent.camera = camera
    agent.overshoot_client = overshoot_client

    for plant in plants:
        task_queue.add_task_in(
            f"check_plant:{plant.name}",
            priority=1.0,
            minutes=plant.index * 10,
            current_time=clock.current_time,
        )

    print("=" * 60)
    print("Plant Care AI Agent Harness")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Time step: {args.time_step} minutes")
    print(f"Camera initialized at: ({args.initial_x_inches:.2f}\", {args.initial_y_inches:.2f}\") = ({initial_x_pixels}px, {initial_y_pixels}px)")
    print(f"Total Plants: {TOTAL_PLANTS}")
    for plant in plants:
        if isinstance(plant, ActualPlant):
            x_inches = plant.pos_x * Camera.INCH_PER_PIXEL_OF_WIDTH
            y_inches = plant.pos_y * Camera.INCH_PER_PIXEL_OF_WIDTH
            print(f"  - Plant {plant.index}: {plant.name} ({plant.species}) at ({x_inches:.2f}\", {y_inches:.2f}\") = ({plant.pos_x}px, {plant.pos_y}px)")
        else:
            print(f"  - Plant {plant.index}: {plant.name} ({plant.species})")
    print(f"Initial tasks scheduled: {len(task_queue.queue)}")
    print("=" * 60)
    print()

    # Initial agent activation
    print(f"[{clock.current_time.isoformat()}] Initial activation for {plants[0].name}...")
    result = agent.act(context="Initial check-in", plant_name=plants[0].name)
    print(f"Agent: {result.get('response', 'No response')[:200]}...")
    print_tool_results(result.get("tool_results"))
    print()

    MAX_IDLE_STEPS = 2  # force a check-in if agent has been idle this many steps
    idle_steps = 0

    # Simulation loop
    for step in range(args.steps):
        current_time = clock.tick()
        hours_passed = args.time_step / 60.0

        print(f"[{current_time.isoformat()}] Step {step + 1}/{args.steps}")
        print("-" * 60)

        update_plant_states(plants, hours_passed)
        print(f"Updated plant states ({hours_passed:.2f} hours passed)")

        # Process due tasks
        due_tasks = agent.process_due_tasks()
        result = None
        if due_tasks:
            idle_steps = 0
            print(f"Processed {len(due_tasks)} due task(s)")
            for task_info in due_tasks:
                print(f"  - Task: {task_info['task']}")
                if task_info.get("action_result"):
                    result = task_info["action_result"]
        else:
            idle_steps += 1
            if idle_steps >= MAX_IDLE_STEPS:
                # Heartbeat: force a check-in so the agent stays alive
                print(f"No tasks for {idle_steps} steps — forcing periodic check-in")
                plant = plants[0]  # round-robin if multi-plant later
                result = agent.act(context="Periodic check-in — no tasks were due", plant_name=plant.name)
                idle_steps = 0
            else:
                print("No tasks due - waiting for agent-scheduled tasks")

        # Show next scheduled task
        if task_queue.queue:
            scheduled_time, _, _, task = task_queue.queue[0]
            print(f"Next scheduled task: '{task}' in {scheduled_time - clock.current_time}")

        if result:
            print_tool_results(result.get("tool_results"))

        # Plant status summary
        print("\nPlant Status Summary:")
        camera = getattr(agent, 'camera', None)
        overshoot_client = getattr(agent, 'overshoot_client', None)
        for plant in plants:
            # Navigate camera to plant position before taking picture
            if camera and isinstance(plant, ActualPlant):
                current_x, current_y = camera.get_position()
                plant.navigate_to_plant(camera, current_x, current_y)
            status = plant.take_picture(camera=camera, overshoot_client=overshoot_client)
            if isinstance(plant, VirtualPlant):
                health = plant.get_health_score()
                print(f"  {plant.name}:")
                print(f"    Hydration: {status['hydration']} "
                      f"(deviation: {status['hydration_deviation']:+.2f})")
                print(f"    Frequency: {status['water_frequency_status']:+.2f}")
                print(f"    Health: {health:.1f}/100")
            else:
                print(f"  {plant.name}:")
                print(f"    Position: ({plant.pos_x}px, {plant.pos_y}px)")
                print(f"    Status: {status.get('hydration', 'N/A')}")

        print()

        if args.real_time:
            print(f"Waiting {args.time_step * 60}s until next step...")
            time.sleep(args.time_step * 60)

    print("=" * 60)
    print("Simulation complete!")
    print("=" * 60)

    print("\nFinal Plant Health Scores:")
    for plant in plants:
        if isinstance(plant, VirtualPlant):
            print(f"  {plant.name}: {plant.get_health_score():.1f}/100")
        else:
            print(f"  {plant.name}: Position ({plant.pos_x}px, {plant.pos_y}px)")

    print("\n")
    agent.print_log_summary()
    log_file = agent.get_log_file()
    if log_file:
        print(f"Full log available at: {log_file}")
    
    # Clean up camera
    try:
        camera.release()
    except:
        pass


if __name__ == "__main__":
    main()
