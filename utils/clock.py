from datetime import datetime, timedelta


class SimulatedClock:
    """Simulated clock that tracks time steps."""

    def __init__(self, start_time=None, time_step_minutes=30):
        self.start_time = start_time or datetime.now()
        self.current_time = self.start_time
        self.time_step = timedelta(minutes=time_step_minutes)
        self.step_count = 0

    def tick(self):
        """Advance clock by one time step."""
        self.current_time += self.time_step
        self.step_count += 1
        return self.current_time
