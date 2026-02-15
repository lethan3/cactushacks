#!/usr/bin/env python3
"""
OpenAI Vision API client for plant image analysis.
"""
import os
import base64
import json
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
    """Client for OpenAI Vision API (formerly Overshoot.ai)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI Vision API client.
        
        Args:
            api_key: API key for OpenAI (or set OPENAI_API_KEY env var or in .env file)
        """
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var, add to .env file, or pass api_key parameter.")
        
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_plant_image(self, image: Image.Image, plant_name: str, plant_species: str) -> Dict:
        """
        Analyze a plant image using OpenAI Vision API.
        
        Args:
            image: PIL Image of the plant
            plant_name: Name of the plant
            plant_species: Species of the plant
            
        Returns:
            Dictionary with plant status information matching format_observable_status format
        """
        # Resize image if too large (OpenAI supports up to 20MB, but we'll keep it reasonable)
        # Keep aspect ratio, max dimension 2048px (OpenAI's max)
        max_dimension = 2048
        if image.width > max_dimension or image.height > max_dimension:
            ratio = min(max_dimension / image.width, max_dimension / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        # Use quality=90 to maintain good quality
        image.save(buffered, format="JPEG", quality=90)
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create prompt for plant analysis
        prompt = f"""You are a plant care expert. Analyze this image of a plant named {plant_name} (species: {plant_species}).

Based on visual indicators like soil moisture, leaf appearance, color, and overall plant condition, provide your assessment in the following JSON format:

{{
  "hydration": "well_hydrated" | "slightly_dry" | "dry" | "very_dry" | "overwatered",
  "hydration_deviation": <number between -2.0 and 2.0, where 0.0 = perfectly hydrated, positive = overwatered, negative = underwatered>,
  "recommended_action": "water_now" | "water_soon" | "no_action_needed" | "reduce_watering",
  "water_frequency_status": <number where 0.0 = perfect frequency, positive = too frequent, negative = too infrequent>,
  "visible_health_issues": ["issue1", "issue2", ...],
  "overall_health": <number from 0-100>
}}

IMPORTANT: Return ONLY the JSON object, no explanations, no markdown formatting, no code blocks, just the raw JSON."""
        
        # OpenAI Vision API format
        payload = {
            "model": "gpt-4o",  # Using GPT-4o which has vision capabilities
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "response_format": {"type": "json_object"}  # Force JSON response
        }
        
        # Debug: print request info
        print(f"Making request to: {self.base_url}/chat/completions")
        print(f"Image size after resize: {image.size}")
        print(f"Base64 length: {len(image_base64)} characters")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the content from OpenAI's response
            content = result["choices"][0]["message"]["content"]
            
            # Try to parse JSON from the response
            # OpenAI might return JSON wrapped in markdown code blocks or with extra text
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            elif content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            
            # Try to find JSON object in the content
            import re
            # Look for JSON object (handles nested objects)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    analysis = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Warning: Found JSON-like structure but couldn't parse. Full content: {content}")
                    analysis = {}
            else:
                # Try parsing the whole content as JSON
                try:
                    analysis = json.loads(content)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON. Full content: {content}")
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
            print(f"Error calling OpenAI API: {e}")
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
