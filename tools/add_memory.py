from typing import Dict
from utils.plant import Plant


def add_memory(plant: Plant, memory: str) -> Dict:
    """
    Add a memory about a plant.
    
    Args:
        plant: Plant object
        memory: Memory text to add
        
    Returns:
        Dictionary with memory addition result
    """
    plant.add_memory(memory)
    
    return {
        "success": True,
        "plant_name": plant.name,
        "memory": memory,
        "total_memories": len(plant.get_memories()),
        "message": f"Added memory about {plant.name}: {memory}"
    }
