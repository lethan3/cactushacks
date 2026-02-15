import random
from abc import ABC, abstractmethod
from datetime import datetime


class Plant(ABC):
    """Base class for all plants."""

    SPECIES_CARE = {
        "cactus": {"water_quantity": 1.0, "frequency_days": 14},
        "succulent": {"water_quantity": 0.5, "frequency_days": 7},
        "fern": {"water_quantity": 2.0, "frequency_days": 3},
        "pothos": {"water_quantity": 1.5, "frequency_days": 5},
        "snake_plant": {"water_quantity": 1.0, "frequency_days": 10},
    }

    def __init__(self, name, species, index):
        self.name = name
        self.species = species
        self.index = index
        self.watering_history = []
        self.memories = []
        self.action_history = []
        self.last_watering = None
        self.ideal_frequency_days = self.SPECIES_CARE.get(
            species.lower(), {"frequency_days": 7}
        )["frequency_days"]

    @abstractmethod
    def take_picture(self):
        """Return plant status as a dict."""
        pass

    @abstractmethod
    def water(self, quantity):
        """Water the plant with given quantity (cups)."""
        pass

    @abstractmethod
    def get_hydration_status(self):
        """0.0 = good, positive = overhydrated, negative = underhydrated."""
        pass

    def add_memory(self, memory):
        self.memories.append(f"{datetime.now().isoformat()}: {memory}")

    def add_initial_action(self, timestamp):
        if not self.action_history:
            self.action_history.append({
                "timestamp": timestamp.isoformat(),
                "tool": "start_care",
                "args": {},
                "reasoning": "Agent started taking care of this plant",
                "result": {"message": f"Started monitoring {self.name} ({self.species})"}
            })

    def add_action_to_history(self, tool_name, args, reasoning, result, timestamp):
        self.action_history.append({
            "timestamp": timestamp.isoformat(),
            "tool": tool_name,
            "args": args,
            "reasoning": reasoning,
            "result": result
        })
        if len(self.action_history) > 10:
            self.action_history = self.action_history[-10:]

    def get_water_frequency_status(self):
        """0.0 = good, positive = too frequent, negative = too infrequent."""
        if len(self.watering_history) < 2:
            return 0.0
        intervals = []
        for i in range(1, len(self.watering_history)):
            days = (self.watering_history[i] - self.watering_history[i-1]).total_seconds() / 86400
            intervals.append(days)
        if not intervals:
            return 0.0
        avg = sum(intervals) / len(intervals)
        ideal = self.ideal_frequency_days
        if avg < ideal * 0.7:
            return (ideal - avg) / ideal
        elif avg > ideal * 1.3:
            return -(avg - ideal) / ideal
        return 0.0

    def _hydration_label(self):
        """Human-readable hydration label from deviation score."""
        h = self.get_hydration_status()
        if h > 0.5:
            return "overwatered"
        elif h > 0.1:
            return "slightly overwatered"
        elif h < -0.5:
            return "dry — needs water"
        elif h < -0.1:
            return "slightly dry"
        return "good"

    def _recommended_action(self):
        """What a sensor/VLM assessment would recommend."""
        h = self.get_hydration_status()
        care = self.SPECIES_CARE.get(self.species.lower(), {"water_quantity": 1.0, "frequency_days": 7})
        if h < -0.3:
            return f"water with {care['water_quantity']} cups — check again in 6 hours to verify recovery"
        elif h < -0.1:
            return "slightly dry — no water yet, but check again in 12 hours"
        elif h > 0.5:
            return "do not water — overwatered, check again in 12 hours"
        elif h > 0.2:
            return "do not water — slightly overwatered, let it dry out"
        return "no action needed — plant is healthy"

    def format_observable_status(self):
        """What a camera/VLM would see — observable values only."""
        care = self.SPECIES_CARE.get(self.species.lower(), {"water_quantity": 1.0, "frequency_days": 7})
        h = round(self.get_hydration_status(), 2)
        return {
            "name": self.name,
            "species": self.species,
            "hydration": self._hydration_label(),
            "hydration_deviation": h,
            "recommended_action": self._recommended_action(),
            "water_frequency_status": round(self.get_water_frequency_status(), 2),
            "last_watering": self.last_watering.isoformat() if self.last_watering else None,
            "watering_count": len(self.watering_history),
            "care_instructions": {
                "water_quantity_cups": care["water_quantity"],
                "water_frequency": f"every {care['frequency_days']} days"
            }
        }


class VirtualPlant(Plant):
    """Simulated plant with internal state tracking."""

    def __init__(self, name, species, index):
        super().__init__(name, species, index)
        default_care = self.SPECIES_CARE.get(species.lower(), {"water_quantity": 1.0, "frequency_days": 7})
        variance = random.uniform(-0.2, 0.2)

        self.actual_water_quantity = default_care["water_quantity"] * (1 + variance)
        self.ideal_frequency_days = default_care["frequency_days"] * (1 + variance)  # override base

        self.hydration_level = 100.0
        self.created_at = datetime.now()
        self.hydration_decay_rate = random.uniform(0.5, 2.0)

    def get_hydration_status(self):
        if self.hydration_level > 80:
            return (self.hydration_level - 80) / 20.0
        elif self.hydration_level < 40:
            return -(40 - self.hydration_level) / 40.0
        return 0.0

    def take_picture(self):
        return self.format_observable_status()

    def water(self, quantity):
        now = datetime.now()
        self.hydration_level = min(100.0, self.hydration_level + quantity * 20.0)
        self.watering_history.append(now)
        self.last_watering = now
        return True

    def update_state(self, hours_passed):
        decay = self.hydration_decay_rate * hours_passed
        noise = random.uniform(-0.1, 0.1) * hours_passed
        self.hydration_level = max(0.0, self.hydration_level - decay + noise)
        if self.hydration_level > 80:
            self.hydration_level -= 0.5 * hours_passed

    def get_health_score(self):
        hydration_score = max(0, 100 - abs(self.hydration_level - 50) * 2)
        frequency_score = 100.0
        if len(self.watering_history) >= 2:
            intervals = []
            for i in range(1, len(self.watering_history)):
                days = (self.watering_history[i] - self.watering_history[i-1]).total_seconds() / 86400
                intervals.append(days)
            if intervals:
                avg = sum(intervals) / len(intervals)
                deviation = abs(avg - self.ideal_frequency_days) / self.ideal_frequency_days
                frequency_score = max(0, 100 - deviation * 100)
        return (hydration_score + frequency_score) / 2


class ActualPlant(Plant):
    """Placeholder for actual plant with real hardware."""

    def __init__(self, name, species, index):
        super().__init__(name, species, index)
        # TODO: Initialize camera and watering device connections

    def get_hydration_status(self):
        raise NotImplementedError("ActualPlant.get_hydration_status() not yet implemented")

    def take_picture(self):
        # TODO: Call camera/VLM to analyze image
        return self.format_observable_status()

    def water(self, quantity):
        now = datetime.now()
        self.watering_history.append(now)
        self.last_watering = now
        raise NotImplementedError("ActualPlant.water() not yet implemented")
