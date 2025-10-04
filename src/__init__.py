"""
Behavioral Risk Index (BRI) Pipeline

A comprehensive pipeline for constructing and validating a Behavioral Risk Index
that measures narrative/herding concentration in financial news and social media.
"""

__version__ = "1.0.0"
__author__ = "BRI Research Team"

from .data_collect import DataCollector
from .preprocess import TextPreprocessor
from .vectorize import Vectorizer
from .bri import BRICalculator
from .validation import ValidationEngine
from .models import ModelTrainer
from .utils import setup_logging, load_config

__all__ = [
    "DataCollector",
    "TextPreprocessor", 
    "Vectorizer",
    "BRICalculator",
    "ValidationEngine",
    "ModelTrainer",
    "setup_logging",
    "load_config"
]
