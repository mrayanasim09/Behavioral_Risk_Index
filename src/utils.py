"""
Utility functions for the BRI pipeline.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import warnings

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration for the BRI pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bri_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('bri_pipeline')

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default configuration
        return {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2024-01-01',
                'sources': {
                    'news': ['GDELT', 'NewsAPI'],
                    'social': ['Reddit', 'Twitter']
                }
            },
            'bri': {
                'methods': ['entropy', 'cluster'],
                'entropy': {
                    'min_vocab_size': 10,
                    'max_vocab_size': 50000
                },
                'cluster': {
                    'n_clusters': 10,
                    'random_state': 42
                }
            },
            'validation': {
                'test_size': 0.2,
                'random_state': 42,
                'forecast_horizon': 5
            }
        }

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator

def calculate_realized_volatility(returns: pd.Series, window: int = 5) -> pd.Series:
    """Calculate realized volatility using rolling window of squared returns."""
    return returns.rolling(window=window).apply(lambda x: np.sqrt(np.sum(x**2)), raw=True)

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series."""
    return np.log(prices / prices.shift(1))

def filter_spam_posts(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Filter out spam posts using heuristics."""
    if text_col not in df.columns:
        return df
    
    # Remove posts with repeated characters (>3 consecutive)
    df = df[~df[text_col].str.contains(r'(.)\1{3,}', regex=True, na=False)]
    
    # Remove very short posts (<3 words)
    df = df[df[text_col].str.split().str.len() >= 3]
    
    # Remove duplicate text
    df = df.drop_duplicates(subset=[text_col])
    
    return df

def validate_date_range(start_date: str, end_date: str) -> tuple:
    """Validate and convert date strings to datetime objects."""
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
            
        return start_dt, end_dt
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

def chunk_dates(start_date: datetime, end_date: datetime, chunk_days: int = 30) -> list:
    """Split date range into chunks for API rate limiting."""
    chunks = []
    current_date = start_date
    
    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
        chunks.append((current_date, chunk_end))
        current_date = chunk_end + timedelta(days=1)
    
    return chunks

def save_with_metadata(df: pd.DataFrame, filepath: str, metadata: Dict[str, Any] = None) -> None:
    """Save DataFrame with metadata to CSV."""
    ensure_directory(os.path.dirname(filepath))
    
    # Add metadata as comments at the top of the file
    if metadata:
        with open(filepath, 'w') as f:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("# Generated on: {}\n".format(datetime.now().isoformat()))
            f.write("# Data shape: {}\n".format(df.shape))
            f.write("\n")
        
        # Append the DataFrame
        df.to_csv(filepath, mode='a', index=False)
    else:
        df.to_csv(filepath, index=False)

def load_with_metadata(filepath: str) -> tuple:
    """Load DataFrame and metadata from CSV file."""
    metadata = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            if ':' in line:
                key, value = line[1:].strip().split(':', 1)
                metadata[key.strip()] = value.strip()
        else:
            data_start = i
            break
    
    # Read the actual data
    df = pd.read_csv(filepath, skiprows=data_start)
    
    return df, metadata

def suppress_warnings():
    """Suppress common warnings for cleaner output."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
