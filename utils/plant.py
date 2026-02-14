import json
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class Plant(ABC):
    """Base class for all plants."""
    
    # Species default care instructions (shared by all plant types)
    SPECIES_CARE = {
        "cactus": {"water_quantity": 1.0, "frequency_days": 14},
        "succulent": {"water_quantity": 0.5, "frequency_days": 7},
        "fern": {"water_quantity": 2.0, "frequency_days": 3},
        "pothos": {"water_quantity": 1.5, "frequency_days": 5},
        "snake_plant": {"water_quantity": 1.0, "frequency_days": 10},
    }
    
    def __init__(self, name: str, species: str, index: int):
        self.name = name
        self.species = species
        self.index = index
        self.watering_history: List[datetime] = []
        self.memories: List[str] = []
        self.action_history: List[Dict] = []  # History of tool calls for this plant
        self.last_watering: Optional[datetime] = None
        
    @abstractmethod
    def take_picture(self) -> Dict:
        """Take a picture and return plant status information."""
        pass
    
    @abstractmethod
    def water(self, water_quantity: float) -> bool:
        """Water the plant with given water_quantity (in cups)."""
        pass
    
    @abstractmethod
    def get_hydration_status(self) -> float:
        """Get hydration status: 0.0 = good, positive = overhydrated, negative = underhydrated."""
        pass
    
    @abstractmethod
    def get_ideal_watering_frequency_days(self) -> float:
        """Get the ideal watering frequency in days for this plant."""
        pass
    
    def add_memory(self, memory: str):
        """Add a memory about this plant."""
        self.memories.append(f"{datetime.now().isoformat()}: {memory}")
    
    def get_memories(self) -> List[str]:
        """Get all memories about this plant."""
        return self.memories
    
    def add_initial_action(self, timestamp: datetime):
        """Add initial action indicating agent started caring for this plant."""
        if not self.action_history:  # Only add if history is empty
            self.action_history.append({
                "timestamp": timestamp.isoformat(),
                "tool": "start_care",
                "args": {},
                "reasoning": "Agent started taking care of this plant",
                "result": {"message": f"Started monitoring and caring for {self.name} ({self.species})"}
            })
    
    def add_action_to_history(self, tool_name: str, args: Dict, reasoning: str, result: Dict, timestamp: datetime):
        """Add a tool call to this plant's action history."""
        self.action_history.append({
            "timestamp": timestamp.isoformat(),
            "tool": tool_name,
            "args": args,
            "reasoning": reasoning,
            "result": result
        })
        # Keep only the last 10 actions to avoid history getting too long
        if len(self.action_history) > 10:
            self.action_history = self.action_history[-10:]
    
    def get_recent_history(self, max_items: int = 5) -> List[Dict]:
        """Get recent action history for this plant."""
        return self.action_history[-max_items:] if self.action_history else []
    
    def get_water_frequency_status(self) -> float:
        """Calculate watering frequency status from watering history.
        
        Returns:
            0.0 = good, positive = too frequent, negative = too infrequent
        """
        if len(self.watering_history) < 2:
            return 0.0
        
        # Calculate average time between waterings
        intervals = []
        for i in range(1, len(self.watering_history)):
            interval = (self.watering_history[i] - self.watering_history[i-1]).total_seconds() / 86400  # days
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        avg_interval = sum(intervals) / len(intervals)
        ideal_interval = self.get_ideal_watering_frequency_days()
        
        if avg_interval < ideal_interval * 0.7:  # Watering too frequently
            return (ideal_interval - avg_interval) / ideal_interval
        elif avg_interval > ideal_interval * 1.3:  # Watering too infrequently
            return -(avg_interval - ideal_interval) / ideal_interval
        
        return 0.0
    
    def format_observable_status(self) -> Dict:
        """Format observable plant status information.
        
        This returns what a camera/VLM would see - observable values only.
        """
        species_care = self.SPECIES_CARE.get(self.species.lower(), {"water_quantity": 1.0, "frequency_days": 7})
        
        return {
            "name": self.name,
            "species": self.species,
            "index": self.index,
            "hydration_status": round(self.get_hydration_status(), 2),
            "water_frequency_status": round(self.get_water_frequency_status(), 2),
            "last_watering": self.last_watering.isoformat() if self.last_watering else None,
            "watering_count": len(self.watering_history),
            "species_instructions": {
                "water_quantity": species_care["water_quantity"],
                "water_frequency": f"every {species_care['frequency_days']} days"
            }
        }


class VirtualPlant(Plant):
    """Simulated plant with internal state tracking."""
    
    def __init__(self, name: str, species: str, index: int):
        super().__init__(name, species, index)
        
        # Get species defaults
        default_care = self.SPECIES_CARE.get(species.lower(), {"water_quantity": 1.0, "frequency_days": 7})
        
        # Actual plant needs (may differ from species by up to 20%)
        variance = random.uniform(-0.2, 0.2)
        self.actual_water_quantity = default_care["water_quantity"] * (1 + variance)
        self.actual_frequency_days = default_care["frequency_days"] * (1 + variance)
        
        # Current state
        self.hydration_level = 100.0  # 0-100 scale
        self.created_at = datetime.now()
        
        # Hydration decay rate (per hour)
        self.hydration_decay_rate = random.uniform(0.5, 2.0)
    
    def get_hydration_status(self) -> float:
        """Calculate hydration status from internal hydration level."""
        # 0 = good, positive = overhydrated, negative = underhydrated
        if self.hydration_level > 80:
            return (self.hydration_level - 80) / 20.0  # 0 to 1 scale for overhydration
        elif self.hydration_level < 40:
            return -(40 - self.hydration_level) / 40.0  # -1 to 0 scale for underhydration
        return 0.0
    
    def get_ideal_watering_frequency_days(self) -> float:
        """Get the ideal watering frequency in days for this plant."""
        return self.actual_frequency_days
    
    def take_picture(self) -> Dict:
        """Return plant status information in JSON format."""
        return self.format_observable_status()
    
    def water(self, water_quantity: float) -> bool:
        """Water the plant and update hydration level."""
        now = datetime.now()
        
        # Add hydration based on water_quantity (1 cup = ~20 hydration points)
        hydration_gain = water_quantity * 20.0
        self.hydration_level = min(100.0, self.hydration_level + hydration_gain)
        
        # Record watering (using base class method)
        self.watering_history.append(now)
        self.last_watering = now
        
        return True
    
    def update_state(self, hours_passed: float):
        """Update plant state based on time passed."""
        # Decrease hydration over time with some random noise
        decay = self.hydration_decay_rate * hours_passed
        noise = random.uniform(-0.1, 0.1) * hours_passed
        self.hydration_level = max(0.0, self.hydration_level - decay + noise)
        
        # Hydration naturally decreases faster when overhydrated
        if self.hydration_level > 80:
            self.hydration_level -= 0.5 * hours_passed
    
    def get_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        # Base score from hydration
        hydration_score = max(0, 100 - abs(self.hydration_level - 50) * 2)
        
        # Frequency score
        frequency_score = 100.0
        if len(self.watering_history) >= 2:
            intervals = []
            for i in range(1, len(self.watering_history)):
                interval = (self.watering_history[i] - self.watering_history[i-1]).total_seconds() / 86400
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                ideal_interval = self.actual_frequency_days
                deviation = abs(avg_interval - ideal_interval) / ideal_interval
                frequency_score = max(0, 100 - deviation * 100)
        
        return (hydration_score + frequency_score) / 2


class ActualPlant(Plant):
    """Placeholder for actual plant with real hardware."""
    
    def __init__(self, name: str, species: str, index: int):
        super().__init__(name, species, index)
        # TODO: Initialize camera and watering device connections
        # TODO: Initialize hydration status tracking from camera/VLM
    
    def get_hydration_status(self) -> float:
        """Get hydration status from camera/VLM analysis."""
        # TODO: Implement actual camera/VLM call to analyze plant hydration
        raise NotImplementedError("ActualPlant.get_hydration_status() not yet implemented")
    
    def get_ideal_watering_frequency_days(self) -> float:
        """Get the ideal watering frequency in days for this plant species."""
        species_care = self.SPECIES_CARE.get(self.species.lower(), {"water_quantity": 1.0, "frequency_days": 7})
        return species_care["frequency_days"]
    
    def take_picture(self) -> Dict:
        """Take actual picture using camera hardware and return observable status."""
        # TODO: Call camera/VLM to analyze image and update hydration status
        # For now, use base class method to format observable status
        # Note: get_hydration_status() must be implemented to return actual camera/VLM analysis
        return self.format_observable_status()
    
    def water(self, water_quantity: float) -> bool:
        """Water plant using actual watering device."""
        # TODO: Implement actual watering device call
        now = datetime.now()
        self.watering_history.append(now)
        self.last_watering = now
        raise NotImplementedError("ActualPlant.water() not yet implemented")
