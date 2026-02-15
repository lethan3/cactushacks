import heapq
from datetime import timedelta


class TaskQueue:
    """Priority queue for scheduling tasks."""

    def __init__(self):
        self.queue = []
        self._counter = 0

    def add_task(self, task, priority, scheduled_time):
        """Add a task to the queue. Deduplicates: if same task already queued,
        keeps the one scheduled sooner."""
        for i, (t, p, c, existing) in enumerate(self.queue):
            if existing == task:
                if scheduled_time < t:
                    # new one is sooner — replace
                    self.queue[i] = (scheduled_time, priority, self._counter, task)
                    heapq.heapify(self.queue)
                    self._counter += 1
                # either way, skip the duplicate
                return
        heapq.heappush(self.queue, (scheduled_time, priority, self._counter, task))
        self._counter += 1

    def add_task_in(self, task, priority, minutes, current_time):
        """Add a task scheduled for X minutes from now."""
        self.add_task(task, priority, current_time + timedelta(minutes=minutes))

    def get_next(self, current_time):
        """Pop and return next task if it's due, else None."""
        if not self.queue:
            return None
        scheduled_time, priority, counter, task = self.queue[0]
        if scheduled_time <= current_time:
            heapq.heappop(self.queue)
            return task
        return None

    def get_all_due(self, current_time):
        """Pop and return all tasks due at or before current_time."""
        due = []
        while self.queue:
            scheduled_time, priority, _, task = self.queue[0]
            if scheduled_time <= current_time:
                heapq.heappop(self.queue)
                due.append((scheduled_time, priority, task))
            else:
                break
        return due
