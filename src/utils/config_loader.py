"""
Configuration loader using Hydra for merging multiple config files.

This module provides utilities to load and merge configurations from
the configs/ directory using Hydra's composition capabilities.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


def load_config(config_name: str = "config", 
                config_dir: str = ".",
                overrides: Optional[list] = None) -> DictConfig:
    """
    Load and merge configuration files using Hydra.
    
    Args:
        config_name: Name of the main config file (without .yaml)
        config_dir: Directory containing config files
        overrides: Optional list of overrides to apply
        
    Returns:
        Merged configuration as DictConfig
    """
    try:
        # Set up Hydra configuration
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize Hydra with config directory
        with initialize(config_path=config_dir, version_base=None):
            # Compose configuration with overrides
            if overrides:
                cfg = compose(config_name=config_name, overrides=overrides)
            else:
                cfg = compose(config_name=config_name)
            
            logger.info(f"Loaded configuration: {config_name}")
            return cfg
            
    except ImportError:
        logger.warning("Hydra not available, falling back to manual config loading")
        return _load_config_manual(config_name, config_dir, overrides)
    except Exception as e:
        logger.error(f"Failed to load config with Hydra: {e}")
        return _load_config_manual(config_name, config_dir, overrides)


def _load_config_manual(config_name: str, 
                       config_dir: str, 
                       overrides: Optional[list] = None) -> DictConfig:
    """
    Fallback manual config loading without Hydra.
    
    Args:
        config_name: Name of the main config file
        config_dir: Directory containing config files
        overrides: Optional list of overrides to apply
        
    Returns:
        Merged configuration as DictConfig
    """
    config_path = Path(config_dir)
    
    # Load main config
    main_config_path = config_path / f"{config_name}.yaml"
    if not main_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {main_config_path}")
    
    # Load config with OmegaConf to handle variable interpolation
    main_config = OmegaConf.load(main_config_path)
    
    # Apply overrides if provided
    if overrides:
        for override in overrides:
            if '=' in override:
                key, value = override.split('=', 1)
                # Convert value to appropriate type
                try:
                    # Try to convert to number
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string
                    pass
                
                # Set nested key using OmegaConf
                OmegaConf.set(main_config, key, value)
                logger.info(f"Applied override: {key} = {value}")
    
    # Resolve variable interpolations
    main_config = OmegaConf.to_container(main_config, resolve=True)
    return OmegaConf.create(main_config)


def save_config(config: DictConfig, 
                output_path: str, 
                format: str = "yaml") -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        output_path: Path to save the config
        format: Output format (yaml, json)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "yaml":
        with open(output_path, 'w') as f:
            OmegaConf.save(config, f)
    elif format.lower() == "json":
        with open(output_path, 'w') as f:
            OmegaConf.save(config, f, format="json")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved configuration to: {output_path}")


def get_config_value(config: DictConfig, 
                    key: str, 
                    default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        config: Configuration object
        key: Dot-separated key (e.g., "training.learning_rate")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    try:
        return OmegaConf.select(config, key)
    except Exception:
        return default


def update_config(config: DictConfig, 
                 key: str, 
                 value: Any) -> DictConfig:
    """
    Update a configuration value using dot notation.
    
    Args:
        config: Configuration object
        key: Dot-separated key (e.g., "training.learning_rate")
        value: New value
        
    Returns:
        Updated configuration
    """
    OmegaConf.set(config, key, value)
    return config


def validate_config(config: DictConfig) -> bool:
    """
    Validate configuration for required fields.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "global.seed",
        "paths.results_dir",
        "training.optimizer",
        "data.datasets.imdb.name",
        "models.baseline_models.bag-of-words-tfidf.type"
    ]
    
    missing_fields = []
    for field in required_fields:
        if OmegaConf.select(config, field) is None:
            missing_fields.append(field)
    
    if missing_fields:
        logger.error(f"Missing required configuration fields: {missing_fields}")
        return False
    
    logger.info("Configuration validation passed")
    return True


def print_config(config: DictConfig, 
                max_depth: int = 3) -> None:
    """
    Print configuration in a readable format.
    
    Args:
        config: Configuration to print
        max_depth: Maximum depth to print
    """
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(OmegaConf.to_yaml(config, resolve=True))
    print("=" * 80)


if __name__ == "__main__":
    # Test config loading
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = load_config("config")
    
    # Print configuration
    print_config(config)
    
    # Validate configuration
    if validate_config(config):
        print("Configuration is valid!")
    else:
        print("Configuration validation failed!")
