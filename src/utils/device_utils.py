"""
Device management utilities for Apple Silicon MacBook Pro M4 Pro.

This module handles device detection, MPS configuration, and fallback strategies
for PyTorch operations on Apple Silicon hardware.
"""

import torch
import logging
from typing import Optional, Union, Dict, Any
import warnings

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Priority order:
    1. MPS (Metal Performance Shaders) if available
    2. CPU as fallback
    
    Returns:
        torch.device: The selected device
        
    Raises:
        RuntimeError: If no suitable device is available
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) device")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.warning("CUDA detected but not recommended for Apple Silicon")
        return device
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
        return device


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about the current device.
    
    Returns:
        Dict containing device information
    """
    device = get_device()
    info = {
        "device": str(device),
        "device_type": device.type,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    
    if device.type == "mps":
        info["mps_built"] = torch.backends.mps.is_built()
    elif device.type == "cuda":
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
    
    return info


def move_to_device(tensor_or_model: Union[torch.Tensor, torch.nn.Module], 
                   device: Optional[torch.device] = None) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move tensor or model to the specified device with error handling.
    
    Args:
        tensor_or_model: PyTorch tensor or model to move
        device: Target device (if None, uses get_device())
        
    Returns:
        Moved tensor or model
        
    Raises:
        RuntimeError: If device movement fails
    """
    if device is None:
        device = get_device()
    
    try:
        return tensor_or_model.to(device)
    except RuntimeError as e:
        if "MPS" in str(e):
            logger.warning(f"MPS error encountered, falling back to CPU: {e}")
            device = torch.device("cpu")
            return tensor_or_model.to(device)
        else:
            logger.error(f"Failed to move to device {device}: {e}")
            raise


def clear_memory(device: Optional[torch.device] = None) -> None:
    """
    Clear GPU/MPS memory cache.
    
    Args:
        device: Target device (if None, uses get_device())
    """
    if device is None:
        device = get_device()
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("CUDA memory cleared")
    elif device.type == "mps":
        # MPS doesn't have explicit memory clearing, but we can try
        torch.mps.empty_cache()
        logger.debug("MPS memory cleared")


def is_mps_available() -> bool:
    """Check if MPS is available and properly configured."""
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get current memory usage for the device.
    
    Args:
        device: Target device (if None, uses get_device())
        
    Returns:
        Dictionary with memory usage information
    """
    if device is None:
        device = get_device()
    
    memory_info = {}
    
    if device.type == "cuda":
        memory_info["allocated"] = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_info["reserved"] = torch.cuda.memory_reserved(device) / 1024**3  # GB
        memory_info["max_allocated"] = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    elif device.type == "mps":
        # MPS doesn't provide detailed memory info
        memory_info["note"] = "MPS memory info not available"
    else:
        import psutil
        memory_info["system_memory_percent"] = psutil.virtual_memory().percent
    
    return memory_info


if __name__ == "__main__":
    # Test device detection
    device = get_device()
    print(f"Selected device: {device}")
    print(f"Device info: {get_device_info()}")
    print(f"Memory usage: {get_memory_usage()}")
