"""
Configuration loader for TensorComparativeVis.
Loads YAML config and instantiates the appropriate domain strategy.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses APP_CONFIG env var or defaults to 'hpc_default.yaml'
    
    Returns:
        Config dictionary
    """
    if config_path is None:
        # Check environment variable first
        config_name = os.getenv('APP_CONFIG', 'hpc_default')
        if not config_name.endswith('.yaml'):
            config_name = f"{config_name}.yaml"
        config_path = Path(__file__).parent.parent.parent / "configs" / config_name
    else:
        config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_domain_instance(config: Dict[str, Any]):
    """Get domain strategy instance from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Domain strategy instance (e.g., HPCDomain())
    """
    domain_name = config.get('domain', 'hpc').lower()
    
    if domain_name == 'hpc':
        from app.domains import HPCDomain
        return HPCDomain()
    else:
        raise ValueError(f"Unknown domain: {domain_name}")
