import json
from typing import Dict, Optional
from utils.plant import Plant


def take_picture(plant: Plant) -> Dict:
    """
    Take a picture of the plant and return status information.
    
    Returns a dictionary with:
    - hydration_status: Hydration status indicator
      * 0.0 = optimal hydration (good)
      * Positive values = overhydrated (bad, range: 0.0 to 1.0)
        - 0.0-0.3: slightly overhydrated
        - 0.3-0.7: moderately overhydrated
        - 0.7-1.0: severely overhydrated
      * Negative values = underhydrated (bad, range: -1.0 to 0.0)
        - 0.0 to -0.3: slightly underhydrated
        - -0.3 to -0.7: moderately underhydrated
        - -0.7 to -1.0: severely underhydrated
    
    - water_frequency_status: Watering frequency status indicator
      * 0.0 = optimal frequency (good)
      * Positive values = watering too frequently (bad)
        - Higher values indicate more frequent than ideal
      * Negative values = watering too infrequently (bad)
        - More negative values indicate less frequent than ideal
    
    - species_instructions: Expected care instructions for the species
    - last_watering: Timestamp of last watering (if any)
    - watering_count: Total number of times plant has been watered
    
    Args:
        plant: Plant object to photograph
        
    Returns:
        Dictionary with plant status information
    """
    return plant.take_picture()


def take_picture_json(plant: Plant) -> str:
    """Take picture and return JSON string."""
    return json.dumps(take_picture(plant), indent=2)
