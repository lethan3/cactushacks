from typing import Dict
from utils.plant import Plant


def water(plant: Plant, water_quantity: float) -> Dict:
    """
    Water the plant with specified water_quantity.
    
    Args:
        plant: Plant object to water
        water_quantity: Amount of water in cups
        
    Returns:
        Dictionary with watering result
    """
    success = plant.water(water_quantity)
    
    return {
        "success": success,
        "plant_name": plant.name,
        "water_quantity_cups": water_quantity,
        "message": f"Watered {plant.name} with {water_quantity} cup(s) of water"
    }
