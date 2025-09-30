"""
Memory optimization utilities for large transformer models on Apple Silicon M4 Pro.

This module provides memory management utilities specifically designed for running
large transformer models (especially Llama) on Apple Silicon with MPS backend.
"""

import torch
import logging
import psutil
import gc
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage and provide optimization recommendations."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_used_gb'] = system_memory.used / (1024**3)
        memory_info['system_available_gb'] = system_memory.available / (1024**3)
        memory_info['system_percent'] = system_memory.percent
        
        # GPU memory (MPS)
        if self.device.type == "mps":
            try:
                # MPS doesn't have direct memory query, use system memory as proxy
                memory_info['gpu_used_gb'] = 0.0  # Placeholder
                memory_info['gpu_available_gb'] = 0.0  # Placeholder
            except Exception as e:
                logger.warning(f"Could not get MPS memory info: {e}")
                memory_info['gpu_used_gb'] = 0.0
                memory_info['gpu_available_gb'] = 0.0
        elif self.device.type == "cuda":
            try:
                memory_info['gpu_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
                memory_info['gpu_available_gb'] = torch.cuda.memory_reserved() / (1024**3)
            except Exception as e:
                logger.warning(f"Could not get CUDA memory info: {e}")
                memory_info['gpu_used_gb'] = 0.0
                memory_info['gpu_available_gb'] = 0.0
        
        return memory_info
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        memory_info = self.get_memory_usage()
        logger.info(f"Memory usage {context}:")
        logger.info(f"  System: {memory_info['system_used_gb']:.2f}GB used, "
                   f"{memory_info['system_available_gb']:.2f}GB available "
                   f"({memory_info['system_percent']:.1f}%)")
        
        if self.device.type == "mps":
            logger.info(f"  MPS: Using system memory (no direct GPU memory query)")
        elif self.device.type == "cuda":
            logger.info(f"  GPU: {memory_info['gpu_used_gb']:.2f}GB used, "
                       f"{memory_info['gpu_available_gb']:.2f}GB reserved")
    
    def is_memory_pressure(self, threshold: float = 85.0) -> bool:
        """Check if system is under memory pressure."""
        memory_info = self.get_memory_usage()
        return memory_info['system_percent'] > threshold


def setup_memory_optimization(model: Any, device: torch.device, config: Dict[str, Any]) -> Any:
    """
    Apply memory optimization settings to a model.
    
    Args:
        model: The model to optimize
        device: The device the model is on
        config: Configuration dictionary with memory settings
        
    Returns:
        The optimized model
    """
    logger.info("Setting up memory optimization...")
    
    # Check if this is a transformer model
    if hasattr(model, 'model') and hasattr(model.model, 'config'):
        model_name = getattr(model.model.config, 'name_or_path', '').lower()
        
        # Apply Llama-specific optimizations
        if 'llama' in model_name:
            logger.info("Applying Llama-specific memory optimizations...")
            
            # Enable gradient checkpointing if not already enabled
            if hasattr(model, 'gradient_checkpointing') and model.gradient_checkpointing:
                if hasattr(model.model, 'gradient_checkpointing_enable'):
                    model.model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
            
            # Disable cache for memory efficiency
            if hasattr(model.model, 'config'):
                model.model.config.use_cache = getattr(model, 'use_cache', False)
                logger.info(f"Use cache set to: {model.model.config.use_cache}")
        
        # Apply general memory optimizations
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            # Set attention implementation for memory efficiency
            if hasattr(model.model.config, 'attention_dropout'):
                model.model.config.attention_dropout = 0.0  # Disable attention dropout for memory
            
            # Enable memory-efficient attention if available
            if hasattr(model.model.config, 'use_memory_efficient_attention'):
                model.model.config.use_memory_efficient_attention = True
                logger.info("Memory-efficient attention enabled")
    
    # Clear memory cache
    clear_memory_cache(device)
    
    # Log initial memory usage
    monitor = MemoryMonitor(device)
    monitor.log_memory_usage("after optimization setup")
    
    return model


def handle_oom_error(model: Any, current_batch_size: int, min_batch_size: int = 1) -> int:
    """
    Handle out-of-memory errors by reducing batch size.
    
    Args:
        model: The model that caused OOM
        current_batch_size: Current batch size that caused OOM
        min_batch_size: Minimum batch size to allow
        
    Returns:
        New batch size to try
        
    Raises:
        RuntimeError: If batch size cannot be reduced further
    """
    if current_batch_size > min_batch_size:
        new_batch_size = max(min_batch_size, current_batch_size // 2)
        logger.warning(f"OOM error, reducing batch size from {current_batch_size} to {new_batch_size}")
        
        # Clear memory before retrying
        if hasattr(model, 'device'):
            clear_memory_cache(model.device)
        
        return new_batch_size
    else:
        raise RuntimeError("Cannot reduce batch size further. Model too large for available memory.")


def clear_memory_cache(device: torch.device):
    """Clear memory cache for the given device."""
    if device.type == "mps":
        torch.mps.empty_cache()
        logger.debug("Cleared MPS memory cache")
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA memory cache")
    
    # Force garbage collection
    gc.collect()


@contextmanager
def memory_efficient_training(model: Any, device: torch.device):
    """
    Context manager for memory-efficient training.
    
    Args:
        model: The model to train
        device: The device the model is on
    """
    logger.info("Starting memory-efficient training context...")
    
    # Clear memory before training
    clear_memory_cache(device)
    
    # Monitor initial memory
    monitor = MemoryMonitor(device)
    monitor.log_memory_usage("before training")
    
    try:
        yield model
    finally:
        # Clear memory after training
        clear_memory_cache(device)
        monitor.log_memory_usage("after training")
        logger.info("Memory-efficient training context completed")


def get_optimal_batch_size(model: Any, device: torch.device, 
                          max_batch_size: int = 32, 
                          min_batch_size: int = 1,
                          test_data_size: int = 100) -> int:
    """
    Find the optimal batch size for a model on the given device.
    
    Args:
        model: The model to test
        device: The device to test on
        max_batch_size: Maximum batch size to try
        min_batch_size: Minimum batch size to try
        test_data_size: Size of test data to use
        
    Returns:
        Optimal batch size
    """
    logger.info(f"Finding optimal batch size for {type(model).__name__} on {device}...")
    
    # Create dummy data
    dummy_texts = ["This is a test sentence."] * test_data_size
    dummy_labels = [0] * test_data_size
    
    optimal_batch_size = min_batch_size
    
    for batch_size in [min_batch_size, 2, 4, 8, 16, 32, max_batch_size]:
        if batch_size > max_batch_size:
            break
            
        try:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Clear memory before test
            clear_memory_cache(device)
            
            # Test tokenization and forward pass
            if hasattr(model, 'tokenize_texts'):
                tokenized = model.tokenize_texts(dummy_texts[:batch_size])
                
                # Test forward pass
                if hasattr(model, 'model'):
                    with torch.no_grad():
                        outputs = model.model(**tokenized)
                        del outputs
                        del tokenized
                        
            clear_memory_cache(device)
            optimal_batch_size = batch_size
            logger.info(f"✅ Batch size {batch_size} works")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                logger.warning(f"❌ Batch size {batch_size} causes OOM")
                break
            else:
                raise
        except Exception as e:
            logger.warning(f"❌ Batch size {batch_size} failed: {e}")
            break
    
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def log_memory_summary(device: torch.device, context: str = ""):
    """Log a comprehensive memory summary."""
    monitor = MemoryMonitor(device)
    memory_info = monitor.get_memory_usage()
    
    logger.info(f"=== Memory Summary {context} ===")
    logger.info(f"System Memory: {memory_info['system_used_gb']:.2f}GB / "
               f"{memory_info['system_used_gb'] + memory_info['system_available_gb']:.2f}GB "
               f"({memory_info['system_percent']:.1f}% used)")
    
    if device.type == "mps":
        logger.info("MPS Backend: Using system memory (no direct GPU memory query)")
    elif device.type == "cuda":
        logger.info(f"CUDA Memory: {memory_info['gpu_used_gb']:.2f}GB used, "
                   f"{memory_info['gpu_available_gb']:.2f}GB reserved")
    
    if monitor.is_memory_pressure():
        logger.warning("⚠️  High memory pressure detected!")
    else:
        logger.info("✅ Memory usage is healthy")
    
    logger.info("=" * 40)
