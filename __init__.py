"""
Aria to LeRobot Dataset Conversion Library

Convert Project Aria VRS/MPS data to LeRobot Dataset v3.0 format.
Built on top of aria_api for reliable data access.
"""

from .convert import convert
from .schema import LeRobotSchema, LEROBOT_V3_SCHEMA, LEROBOT_V21_SCHEMA

__version__ = "1.0.0"
__all__ = ["convert", "LeRobotSchema", "LEROBOT_V3_SCHEMA", "LEROBOT_V21_SCHEMA"]
