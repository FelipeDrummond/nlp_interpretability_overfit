"""
Device management utilities for PyTorch with CUDA, MPS, and CPU support.

This module handles device detection and configuration for PyTorch operations
across different hardware platforms (NVIDIA GPUs, Apple Silicon, CPU).
"""

import torch
import logging
from typing import Optional, Union, Dict, Any
import warnings

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Priority order for 'auto':
    1. CUDA (NVIDIA GPU) if available
    2. MPS (Apple Silicon) if available
    3. CPU as fallback
    
    Args:
        device_preference: Device preference - "auto", "cuda", "mps", or "cpu"
    
    Returns:
        torch.device: The selected device
        
    Raises:
        RuntimeError: If requested device is not available
    """
    device_preference = device_preference.lower()
    
    # Explicit device request
    if device_preference == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        else:
            raise RuntimeError("CUDA requested but not available")
    
    elif device_preference == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) device")
            return device
        else:
            raise RuntimeError("MPS requested but not available")
    
    elif device_preference == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU device")
        return device
    
    # Auto mode - prioritize CUDA > MPS > CPU
    elif device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-detected MPS (Metal Performance Shaders) device")
            return device
        else:
            device = torch.device("cpu")
            logger.info("Auto-detected CPU device (no GPU available)")
            return device
    
    else:
        raise ValueError(f"Unknown device preference: {device_preference}. Use 'auto', 'cuda', 'mps', or 'cpu'")


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
                   device: Optional[torch.device] = None,
                   device_preference: str = "auto") -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move tensor or model to the specified device with error handling.
    
    Args:
        tensor_or_model: PyTorch tensor or model to move
        device: Target device (if None, uses get_device())
        device_preference: Device preference for auto-detection
        
    Returns:
        Moved tensor or model
        
    Raises:
        RuntimeError: If device movement fails
    """
    if device is None:
        device = get_device(device_preference)
    
    try:
        return tensor_or_model.to(device)
    except RuntimeError as e:
        if "MPS" in str(e) or "CUDA" in str(e):
            logger.warning(f"GPU error encountered, falling back to CPU: {e}")
            device = torch.device("cpu")
            return tensor_or_model.to(device)
        else:
            logger.error(f"Failed to move to device {device}: {e}")
            raise


def clear_memory(device: Optional[torch.device] = None, device_preference: str = "auto") -> None:
    """
    Clear GPU/MPS memory cache.
    
    Args:
        device: Target device (if None, uses get_device())
        device_preference: Device preference for auto-detection
    """
    if device is None:
        device = get_device(device_preference)
    
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


def get_memory_usage(device: Optional[torch.device] = None, device_preference: str = "auto") -> Dict[str, float]:
    """
    Get current memory usage for the device.
    
    Args:
        device: Target device (if None, uses get_device())
        device_preference: Device preference for auto-detection
        
    Returns:
        Dictionary with memory usage information
    """
    if device is None:
        device = get_device(device_preference)
    
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
    print("Testing device detection with 'auto' mode:")
    device = get_device("auto")
    print(f"Selected device: {device}")
    print(f"Device info: {get_device_info()}")
    print(f"Memory usage: {get_memory_usage()}")
    
    print("\nTesting all device modes:")
    for mode in ["auto", "cuda", "mps", "cpu"]:
        try:
            test_device = get_device(mode)
            print(f"  {mode}: {test_device}")
        except Exception as e:
            print(f"  {mode}: Not available ({e})")
