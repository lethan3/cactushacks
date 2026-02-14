#!/usr/bin/env python3
"""
Main harness for running the plant care AI agent.
"""

import argparse
import time
from datetime import datetime
from typing import List

from utils.plant import VirtualPlant
from utils.clock import SimulatedClock
from utils.priority_queue import TaskQueue
from agent import PlantCareAgent

# Total number of plants (set to 1 for easier debugging, but system supports multiple)
TOTAL_PLANTS = 1


def update_plant_states(plants: List[VirtualPlant], hours_passed: float):
    """Update all virtual plant states."""
    for plant in plants:
        if isinstance(plant, VirtualPlant):
            plant.update_state(hours_passed)


def main():
    parser = argparse.ArgumentParser(description="Plant Care AI Agent Harness")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for Ollama API calls (default: 120)")
    parser.add_argument("--time-step", type=int, default=30, help="Time step in minutes (default: 30)")
    parser.add_argument("--steps", type=int, default=10, help="Number of time steps to simulate")
    parser.add_argument("--real-time", action="store_true", help="Run in real-time mode (wait between steps)")
    args = parser.parse_args()
    
    # Initialize components
    clock = SimulatedClock(time_step_minutes=args.time_step)
    task_queue = TaskQueue()
    
    # Create sample plants with indices (only one for debugging)
    plants = [
        VirtualPlant("Cactus1", "cactus", 0),
        # VirtualPlant("Fern1", "fern", 1),  # Commented out for debugging
        # VirtualPlant("Pothos1", "pothos", 2),  # Commented out for debugging
    ]
    
    # Ensure we have the correct number of plants
    assert len(plants) == TOTAL_PLANTS, f"Expected {TOTAL_PLANTS} plants, got {len(plants)}"
    
    # Initialize agent
    agent = PlantCareAgent(ollama_url=args.ollama_url, model=args.model, timeout=args.timeout)
    agent.register_plants(plants)
    agent.set_task_queue(task_queue)
    agent.set_clock(clock)
    
    # Initialize priority queue with one task per plant (check each plant)
    current_time = clock.get_current_time()
    for plant in plants:
        # Schedule initial check for each plant, staggered by 10 minutes
        task_queue.add_task_in(
            f"check_plant:{plant.name}",
            priority=1.0,  # High priority
            minutes=plant.index * 10,  # Stagger: 0, 10, 20 minutes
            current_time=current_time
        )
    
    print("=" * 60)
    print("Plant Care AI Agent Harness")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Time step: {args.time_step} minutes")
    print(f"Total Plants: {TOTAL_PLANTS}")
    for plant in plants:
        print(f"  - Plant {plant.index}: {plant.name} ({plant.species})")
    print(f"Initial tasks scheduled: {task_queue.size()}")
    print("=" * 60)
    print()
    
    # Initial agent activation - start with first plant
    print(f"[{clock.get_current_time().isoformat()}] Initial agent activation for {plants[0].name}...")
    result = agent.act(context="Initial check-in", plant_name=plants[0].name)
    print(f"Agent response: {result.get('response', 'No response')[:200]}...")
    if result.get('tool_results'):
        print(f"Executed {len(result['tool_results'])} tool call(s):")
        for tool_result in result['tool_results']:
            tool_call = tool_result.get('tool_call', {})
            tool_name = tool_call.get('tool', 'unknown')
            reasoning = tool_call.get('reasoning', 'No reasoning provided')
            result_data = tool_result.get('result', {})
            print(f"  - {tool_name}: {result_data.get('message', result_data)}")
            print(f"    Reasoning: {reasoning}")
    print()
    
    # Simulation loop
    for step in range(args.steps):
        # Advance clock
        current_time = clock.tick()
        hours_passed = args.time_step / 60.0  # Convert minutes to hours
        
        print(f"[{current_time.isoformat()}] Step {step + 1}/{args.steps}")
        print("-" * 60)
        
        # Update plant states
        update_plant_states(plants, hours_passed)
        print(f"Updated plant states ({hours_passed:.2f} hours passed)")
        
        # Process due tasks (these activate the agent for specific plants)
        due_tasks = agent.process_due_tasks()
        result = None
        if due_tasks:
            print(f"Processed {len(due_tasks)} due task(s)")
            for task_info in due_tasks:
                print(f"  - Task: {task_info['task']}")
                # Get the result from the last task
                if task_info.get('action_result'):
                    result = task_info['action_result']
        else:
            # No tasks due - agent will schedule its own tasks via add_task
            print("No tasks due - waiting for agent-scheduled tasks")
        
        # Check if there are scheduled tasks coming up
        next_task = task_queue.peek_next()
        if next_task:
            scheduled_time, task = next_task
            time_until = clock.get_time_until(scheduled_time)
            print(f"Next scheduled task: '{task}' in {time_until}")
        
        if result and result.get('tool_results'):
            print(f"Agent executed {len(result['tool_results'])} tool call(s):")
            for tool_result in result['tool_results']:
                tool_call = tool_result.get('tool_call', {})
                tool_name = tool_call.get('tool', 'unknown')
                reasoning = tool_call.get('reasoning', 'No reasoning provided')
                result_data = tool_result.get('result', {})
                print(f"  - {tool_name}: {result_data.get('message', result_data)}")
                print(f"    Reasoning: {reasoning}")
        
        # Print plant status summary
        print("\nPlant Status Summary:")
        for plant in plants:
            if isinstance(plant, VirtualPlant):
                status = plant.take_picture()
                health = plant.get_health_score()
                print(f"  {plant.name}:")
                print(f"    Hydration Status: {status['hydration_status']:+.2f} "
                      f"(0.0=good, +=overhydrated, -=underhydrated)")
                print(f"    Water Frequency: {status['water_frequency_status']:+.2f}")
                print(f"    Health Score: {health:.1f}/100")
        
        print()
        
        # Real-time mode: wait before next step
        if args.real_time:
            wait_seconds = args.time_step * 60
            print(f"Waiting {wait_seconds} seconds until next step...")
            time.sleep(wait_seconds)
    
    print("=" * 60)
    print("Simulation complete!")
    print("=" * 60)
    
    # Final summary
    print("\nFinal Plant Health Scores:")
    for plant in plants:
        if isinstance(plant, VirtualPlant):
            health = plant.get_health_score()
            print(f"  {plant.name}: {health:.1f}/100")
    
    # Print log summary
    print("\n")
    agent.print_log_summary()
    log_file = agent.get_log_file()
    if log_file:
        print(f"Full log available at: {log_file}")


if __name__ == "__main__":
    main()
