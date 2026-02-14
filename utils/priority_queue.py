import heapq
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Tuple


class TaskQueue:
    """Priority queue for scheduling tasks."""
    
    def __init__(self):
        self.queue: list = []
        self.counter = 0  # For tie-breaking in heapq
    
    def add_task(self, task: Any, priority: float, scheduled_time: datetime):
        """
        Add a task to the queue.
        
        Args:
            task: Task data (can be any type)
            priority: Priority value (lower = higher priority)
            scheduled_time: When the task should be executed
        """
        # Use counter to ensure stable ordering for tasks with same priority/time
        heapq.heappush(self.queue, (scheduled_time, priority, self.counter, task))
        self.counter += 1
    
    def add_task_in(self, task: Any, priority: float, minutes: float, current_time: datetime):
        """
        Add a task scheduled for X minutes from now.
        
        Args:
            task: Task data
            priority: Priority value
            minutes: Minutes from current_time to schedule
            current_time: Current time reference
        """
        scheduled_time = current_time + timedelta(minutes=minutes)
        self.add_task(task, priority, scheduled_time)
    
    def peek_next(self) -> Optional[Tuple[datetime, Any]]:
        """Peek at the next task without removing it."""
        if not self.queue:
            return None
        scheduled_time, priority, counter, task = self.queue[0]
        return (scheduled_time, task)
    
    def get_next(self, current_time: datetime) -> Optional[Any]:
        """
        Get the next task if it's time to execute it.
        
        Args:
            current_time: Current time to check against
            
        Returns:
            Task if it's time to execute, None otherwise
        """
        if not self.queue:
            return None
        
        scheduled_time, priority, counter, task = self.queue[0]
        
        if scheduled_time <= current_time:
            heapq.heappop(self.queue)
            return task
        
        return None
    
    def get_all_due(self, current_time: datetime) -> list:
        """Get all tasks that are due at or before current_time."""
        due_tasks = []
        
        while self.queue:
            scheduled_time, priority, counter, task = self.queue[0]
            if scheduled_time <= current_time:
                heapq.heappop(self.queue)
                due_tasks.append((scheduled_time, priority, task))
            else:
                break
        
        return due_tasks
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0
    
    def size(self) -> int:
        """Get number of tasks in queue."""
        return len(self.queue)
