###helper functions (config, seeding, I/O)###

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "conf.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing all configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def get_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract data paths from configuration.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        Dictionary of path mappings
    """
    return config.get('paths', {})

def get_data_files(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract data file names from configuration.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        Dictionary of file name mappings
    """
    return config.get('data_files', {})

def get_dtypes(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract pandas data types from configuration.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        Dictionary of column name to dtype mappings
    """
    return config.get('dtypes', {})

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that required configuration sections exist.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['paths', 'data_files', 'dtypes']
    missing_sections = []
    
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    return True

def get_file_path(config: Dict[str, Any], path_type: str, filename_key: str) -> Path:
    """
    Construct full file path from configuration.
    
    Args:
        config: Configuration dictionary from load_config()
        path_type: Type of path ('raw', 'processed', etc.)
        filename_key: Key for filename in data_files section
        
    Returns:
        Full Path object
    """
    paths = get_paths(config)
    data_files = get_data_files(config)
    
    base_path = paths.get(path_type)
    filename = data_files.get(filename_key)
    
    if not base_path:
        raise ValueError(f"Path type '{path_type}' not found in configuration")
    if not filename:
        raise ValueError(f"Filename key '{filename_key}' not found in configuration")
    
    return Path(base_path) / filename

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    # Add other random seed settings as needed for future ML libraries

def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path as string
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj