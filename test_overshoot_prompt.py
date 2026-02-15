#!/usr/bin/env python3
"""Test script to show the overshoot.ai prompt that would be used."""

from PIL import Image
import base64
from io import BytesIO

def show_prompt(image_path: str, plant_name: str = "TestPlant", plant_species: str = "cactus"):
    """Show the prompt that would be sent to overshoot.ai."""
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    # Convert PIL Image to base64 (same as the actual code)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print(f"Base64 length: {len(image_base64)} characters")
    
    # Create prompt (same as in overshoot_ai.py)
    prompt = f"""Analyze this image of a plant named {plant_name} (species: {plant_species}).

Please provide the following information in JSON format:
- hydration: One of "well_hydrated", "slightly_dry", "dry", "very_dry", "overwatered"
- hydration_deviation: A number between -2.0 and 2.0 where:
  * 0.0 means perfectly hydrated
  * Positive values mean overwatered (max 2.0)
  * Negative values mean underwatered (min -2.0)
- recommended_action: One of "water_now", "water_soon", "no_action_needed", "reduce_watering"
- water_frequency_status: A number indicating how well the watering frequency is maintained (0.0 = perfect, positive = too frequent, negative = too infrequent)
- visible_health_issues: List of any visible health issues (e.g., "yellowing_leaves", "wilting", "brown_spots")
- overall_health: A number from 0-100 indicating overall plant health

Return ONLY valid JSON, no other text."""
    
    print("\n" + "=" * 60)
    print("PROMPT THAT WOULD BE SENT TO OVERSHOOT.AI:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    
    print("\nAPI PAYLOAD STRUCTURE:")
    print("-" * 60)
    print(f"Model: Qwen/Qwen3-VL-30B-A3B-Instruct")
    print(f"Endpoint: https://api.overshoot.ai/v1/vision/analyze")
    print(f"Max tokens: 500")
    print(f"Image: [base64 encoded, {len(image_base64)} chars]")
    print(f"Prompt length: {len(prompt)} characters")
    print("-" * 60)

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
    
    show_prompt(image_path, plant_name, plant_species)
