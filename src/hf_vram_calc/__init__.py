"""
HF VRAM Calculator

A Python package for estimating GPU memory requirements for Hugging Face models
with different data types and parallelization strategies.
"""

__version__ = "1.0.6"
__author__ = "HF VRAM Calculator Contributors"
__email__ = "your-email@example.com"

from .calculator import VRAMCalculator
from .config import ConfigManager
from .models import ModelConfig
from .parser import ConfigParser
from .parallel import ParallelizationCalculator

__all__ = [
    "VRAMCalculator",
    "ConfigManager", 
    "ModelConfig",
    "ConfigParser",
    "ParallelizationCalculator",
]
