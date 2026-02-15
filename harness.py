#!/usr/bin/env python3
"""Main harness for running the plant care AI agent."""

import argparse
import time

from utils.plant import VirtualPlant
from utils.clock import SimulatedClock
from utils.priority_queue import TaskQueue
from agent import PlantCareAgent

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
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout for Ollama API calls")
    parser.add_argument("--time-step", type=int, default=30, help="Time step in minutes")
    parser.add_argument("--steps", type=int, default=5, help="Number of time steps")
    parser.add_argument("--real-time", action="store_true", help="Run in real-time mode")
    args = parser.parse_args()

    clock = SimulatedClock(time_step_minutes=args.time_step)
    task_queue = TaskQueue()

    plants = [
        VirtualPlant("Cactus1", "cactus", 0),
    ]
    assert len(plants) == TOTAL_PLANTS, f"Expected {TOTAL_PLANTS} plants, got {len(plants)}"

    agent = PlantCareAgent(model=args.model, host=args.host, timeout=args.timeout)
    agent.register_plants(plants)
    agent.task_queue = task_queue
    agent.clock = clock

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
    print(f"Total Plants: {TOTAL_PLANTS}")
    for plant in plants:
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
        for plant in plants:
            if isinstance(plant, VirtualPlant):
                status = plant.take_picture()
                health = plant.get_health_score()
                print(f"  {plant.name}:")
                print(f"    Hydration: {status['hydration']} "
                      f"(deviation: {status['hydration_deviation']:+.2f})")
                print(f"    Frequency: {status['water_frequency_status']:+.2f}")
                print(f"    Health: {health:.1f}/100")

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

    print("\n")
    agent.print_log_summary()
    log_file = agent.get_log_file()
    if log_file:
        print(f"Full log available at: {log_file}")


if __name__ == "__main__":
    main()
