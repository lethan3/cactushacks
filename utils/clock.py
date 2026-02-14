from datetime import datetime, timedelta
from typing import Callable, Optional


class SimulatedClock:
    """Simulated clock that tracks time steps."""
    
    def __init__(self, start_time: Optional[datetime] = None, time_step_minutes: int = 30):
        """
        Initialize simulated clock.
        
        Args:
            start_time: Starting datetime (defaults to now)
            time_step_minutes: Minutes per time step (default: 30)
        """
        self.start_time = start_time or datetime.now()
        self.current_time = self.start_time
        self.time_step = timedelta(minutes=time_step_minutes)
        self.step_count = 0
    
    def tick(self) -> datetime:
        """Advance clock by one time step."""
        self.current_time += self.time_step
        self.step_count += 1
        return self.current_time
    
    def get_current_time(self) -> datetime:
        """Get current simulated time."""
        return self.current_time
    
    def get_time_until(self, target_time: datetime) -> timedelta:
        """Get time delta until target time."""
        return target_time - self.current_time
    
    def get_hours_passed(self) -> float:
        """Get hours passed since start."""
        delta = self.current_time - self.start_time
        return delta.total_seconds() / 3600.0
    
    def reset(self):
        """Reset clock to start time."""
        self.current_time = self.start_time
        self.step_count = 0
