from datetime import datetime, timedelta
from typing import Any, Dict
from utils.priority_queue import TaskQueue


def add_task(task_queue: TaskQueue, task_description: str, priority: float, 
             minutes: float, current_time: datetime) -> Dict:
    """
    Add a task to the priority queue.
    
    Args:
        task_queue: TaskQueue instance
        task_description: Description of the task
        priority: Priority value (lower = higher priority)
        minutes: Minutes from current_time to schedule the task
        current_time: Current time reference
        
    Returns:
        Dictionary with task addition result
    """
    task_queue.add_task_in(task_description, priority, minutes, current_time)
    
    scheduled_time = current_time + timedelta(minutes=minutes)
    
    return {
        "success": True,
        "task_description": task_description,
        "priority": priority,
        "scheduled_time": scheduled_time.isoformat(),
        "minutes_from_now": minutes,
        "message": f"Task '{task_description}' scheduled for {scheduled_time.isoformat()}"
    }
