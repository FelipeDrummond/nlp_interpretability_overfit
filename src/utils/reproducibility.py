"""
Reproducibility utilities for ensuring consistent results across experiments.

This module provides centralized seed management and experiment tracking
for reproducible machine learning experiments.
"""

import os
import random
import sys
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_reproducibility(config: Dict[str, Any]) -> None:
    """
    Initialize all seeds and settings for reproducible results.
    
    Args:
        config: Configuration dictionary containing at least 'seed' key
    """
    seed = config.get('seed', 42)
    logger.info(f"Setting up reproducibility with seed: {seed}")
    
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # MPS specific
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        logger.info("MPS seed set")
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Use deterministic algorithms with warnings for unsupported operations
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        logger.warning(f"Could not set deterministic algorithms: {e}")
    
    # Disable tokenizer parallelism for reproducibility
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set environment variables for reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    logger.info("Reproducibility setup complete")


def log_experiment(config: Dict[str, Any], 
                  model_name: str, 
                  dataset_name: str, 
                  metrics: Dict[str, Any],
                  log_dir: str = "results/logs") -> str:
    """
    Log experiment configuration and results.
    
    Args:
        config: Experiment configuration
        model_name: Name of the model used
        dataset_name: Name of the dataset used
        metrics: Dictionary of metrics and results
        log_dir: Directory to save logs
        
    Returns:
        Unique experiment ID
    """
    timestamp = datetime.now().isoformat()
    
    # Create unique experiment ID
    exp_string = f"{model_name}_{dataset_name}_{config.get('seed', 42)}_{timestamp}"
    exp_id = hashlib.md5(exp_string.encode()).hexdigest()[:8]
    
    # Get environment information
    environment = {
        'torch_version': torch.__version__,
        'python_version': sys.version,
        'platform': sys.platform,
    }
    
    # Add device information
    try:
        from .device_utils import get_device_info
        environment.update(get_device_info())
    except ImportError:
        environment['device'] = 'unknown'
    
    experiment = {
        'id': exp_id,
        'timestamp': timestamp,
        'model': model_name,
        'dataset': dataset_name,
        'config': config,
        'metrics': metrics,
        'environment': environment
    }
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Save to results/logs/
    log_path = os.path.join(log_dir, f"{exp_id}_{timestamp.replace(':', '-')}.json")
    
    try:
        with open(log_path, 'w') as f:
            json.dump(experiment, f, indent=2, default=str)
        logger.info(f"Experiment logged to: {log_path}")
    except Exception as e:
        logger.error(f"Failed to save experiment log: {e}")
        raise
    
    return exp_id


def verify_reproducibility(exp_id_1: str, 
                          exp_id_2: str, 
                          log_dir: str = "results/logs",
                          tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Verify two experiments produced identical results.
    
    Args:
        exp_id_1: First experiment ID
        exp_id_2: Second experiment ID
        log_dir: Directory containing experiment logs
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with verification results
    """
    try:
        # Load experiment logs
        log_files_1 = [f for f in os.listdir(log_dir) if f.startswith(exp_id_1)]
        log_files_2 = [f for f in os.listdir(log_dir) if f.startswith(exp_id_2)]
        
        if not log_files_1 or not log_files_2:
            return {"error": "Experiment logs not found"}
        
        with open(os.path.join(log_dir, log_files_1[0]), 'r') as f:
            exp_1 = json.load(f)
        with open(os.path.join(log_dir, log_files_2[0]), 'r') as f:
            exp_2 = json.load(f)
        
        # Compare configurations
        config_match = exp_1['config'] == exp_2['config']
        
        # Compare metrics
        metrics_match = True
        metric_differences = {}
        
        for key in set(exp_1['metrics'].keys()) | set(exp_2['metrics'].keys()):
            val_1 = exp_1['metrics'].get(key)
            val_2 = exp_2['metrics'].get(key)
            
            if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                diff = abs(val_1 - val_2)
                if diff > tolerance:
                    metrics_match = False
                    metric_differences[key] = {
                        'exp_1': val_1,
                        'exp_2': val_2,
                        'difference': diff
                    }
            elif val_1 != val_2:
                metrics_match = False
                metric_differences[key] = {
                    'exp_1': val_1,
                    'exp_2': val_2,
                    'difference': 'not_numerical'
                }
        
        return {
            'config_match': config_match,
            'metrics_match': metrics_match,
            'metric_differences': metric_differences,
            'tolerance': tolerance
        }
        
    except Exception as e:
        return {"error": f"Verification failed: {e}"}


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get current reproducibility settings.
    
    Returns:
        Dictionary with current reproducibility configuration
    """
    return {
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'tokenizers_parallelism': os.environ.get('TOKENIZERS_PARALLELISM'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'torch_deterministic': torch.backends.cudnn.deterministic,
        'torch_benchmark': torch.backends.cudnn.benchmark,
        'mps_available': torch.backends.mps.is_available(),
        'cuda_available': torch.cuda.is_available(),
    }


def reset_seeds(seed: int = 42) -> None:
    """
    Reset all random seeds to a specific value.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    logger.info(f"All seeds reset to: {seed}")


def create_experiment_config(base_config: Dict[str, Any], 
                           overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a new experiment configuration with optional overrides.
    
    Args:
        base_config: Base configuration dictionary
        overrides: Optional overrides to apply
        
    Returns:
        New configuration dictionary
    """
    config = base_config.copy()
    
    if overrides:
        config.update(overrides)
    
    # Ensure seed is set
    if 'seed' not in config:
        config['seed'] = 42
    
    # Add timestamp
    config['experiment_timestamp'] = datetime.now().isoformat()
    
    return config


if __name__ == "__main__":
    # Test reproducibility setup
    test_config = {'seed': 42, 'model': 'test', 'dataset': 'test'}
    setup_reproducibility(test_config)
    
    print("Reproducibility info:")
    print(json.dumps(get_reproducibility_info(), indent=2))
