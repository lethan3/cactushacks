#!/usr/bin/env python3
"""
Sensor reading functions for plant monitoring.
Each function reads from a specific sensor and returns a value.
"""
from typing import Dict, Optional


def read_temperature() -> Optional[float]:
    """
    Read the ambient temperature in degrees Fahrenheit.
    
    Returns:
        Temperature in °F, or None if sensor is unavailable.
    """
    # TODO: Implement actual temperature sensor reading
    # e.g., read from a DHT22 / DS18B20 / I2C sensor connected to Jetson GPIO
    return None


def read_soil_moisture() -> Optional[float]:
    """
    Read the soil moisture level as a percentage (0-100).
    0 = completely dry, 100 = fully saturated.
    
    Returns:
        Soil moisture percentage, or None if sensor is unavailable.
    """
    # TODO: Implement actual soil moisture sensor reading
    # e.g., read from a capacitive soil moisture sensor via ADC
    return None


def read_all_sensors() -> Dict[str, Optional[float]]:
    """
    Read all available sensors and return a dictionary of readings.
    
    Returns:
        Dictionary with sensor names as keys and readings as values.
        A value of None means the sensor is unavailable or failed to read.
    """
    return {
        "temperature_f": read_temperature(),
        "soil_moisture_pct": read_soil_moisture(),
    }
