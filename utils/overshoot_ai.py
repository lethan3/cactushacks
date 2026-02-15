#!/usr/bin/env python3
"""
Overshoot.ai API client for plant image analysis.
"""
import os
import base64
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class OvershootAIClient:
    """Client for Overshoot.ai vision API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Overshoot.ai client.
        
        Args:
            api_key: API key for Overshoot.ai (or set OVERSHOOT_AI_API_KEY env var or in .env file)
        """
        self.api_key = api_key or os.getenv("OVERSHOOT_AI_API_KEY")
        if not self.api_key:
            raise ValueError("Overshoot.ai API key is required. Set OVERSHOOT_AI_API_KEY env var, add to .env file, or pass api_key parameter.")
        
        self.base_url = "https://api.overshoot.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_plant_image(self, image: Image.Image, plant_name: str, plant_species: str) -> Dict:
        """
        Analyze a plant image using Overshoot.ai vision API.
        
        Args:
            image: PIL Image of the plant
            plant_name: Name of the plant
            plant_species: Species of the plant
            
        Returns:
            Dictionary with plant status information matching format_observable_status format
        """
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create prompt for plant analysis
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
        
        # Make API request
        payload = {
            "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "prompt": prompt,
            "image": image_base64,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/vision/analyze",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Parse the result and format it to match format_observable_status
            # The API should return JSON in result["result"] or similar
            analysis = result.get("result", {})
            if isinstance(analysis, str):
                # If result is a string, try to parse it as JSON
                import json
                try:
                    analysis = json.loads(analysis)
                except json.JSONDecodeError:
                    # Fallback: extract JSON from text if needed
                    analysis = {}
            
            # Map to expected format
            return {
                "name": plant_name,
                "species": plant_species,
                "hydration": analysis.get("hydration", "unknown"),
                "hydration_deviation": float(analysis.get("hydration_deviation", 0.0)),
                "recommended_action": analysis.get("recommended_action", "no_action_needed"),
                "water_frequency_status": float(analysis.get("water_frequency_status", 0.0)),
                "visible_health_issues": analysis.get("visible_health_issues", []),
                "overall_health": float(analysis.get("overall_health", 50.0)),
                "care_instructions": {
                    "water_quantity_cups": 1.0,  # Default, could be extracted from analysis
                    "water_frequency": "every 7 days"  # Default, could be extracted from analysis
                }
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Overshoot.ai API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            # Return default/fallback response
            return {
                "name": plant_name,
                "species": plant_species,
                "hydration": "unknown",
                "hydration_deviation": 0.0,
                "recommended_action": "no_action_needed",
                "water_frequency_status": 0.0,
                "visible_health_issues": [],
                "overall_health": 50.0,
                "care_instructions": {
                    "water_quantity_cups": 1.0,
                    "water_frequency": "every 7 days"
                }
            }
