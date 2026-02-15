#!/usr/bin/env python3
"""Test script to test overshoot.ai prompt on an image."""

from PIL import Image
from utils.overshoot_ai import OvershootAIClient
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

def test_overshoot_prompt(image_path: str, plant_name: str = "TestPlant", plant_species: str = "cactus"):
    """Test the overshoot.ai prompt on an image."""
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    print("\nInitializing OpenAI Vision API client...")
    try:
        client = OvershootAIClient()
        print("Client initialized successfully")
    except Exception as e:
        print(f"Error initializing client: {e}")
        return
    
    print(f"\nAnalyzing image for plant: {plant_name} ({plant_species})")
    print("=" * 60)
    
    try:
        result = client.analyze_plant_image(image, plant_name, plant_species)
        print("\nResult:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error analyzing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    image_path = "20260215_051023.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    plant_name = "TestCactus"
    plant_species = "cactus"
    if len(sys.argv) > 2:
        plant_name = sys.argv[2]
    if len(sys.argv) > 3:
        plant_species = sys.argv[3]
    
    test_overshoot_prompt(image_path, plant_name, plant_species)
