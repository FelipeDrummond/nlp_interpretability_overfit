#!/usr/bin/env python3
"""
Test script to verify CUDA/GPU compatibility and configuration.

This script tests:
1. PyTorch CUDA availability
2. Device detection and selection
3. Basic tensor operations on GPU
4. Model loading and inference on GPU

Usage:
    python test_cuda.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.device_utils import get_device, get_device_info, get_memory_usage


def test_cuda_availability():
    """Test if CUDA is available."""
    print("=" * 60)
    print("1. CUDA Availability Test")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print("⚠️  CUDA is not available!")
    
    print()


def test_device_selection():
    """Test automatic device selection."""
    print("=" * 60)
    print("2. Device Selection Test")
    print("=" * 60)
    
    for mode in ["auto", "cuda", "mps", "cpu"]:
        try:
            device = get_device(mode)
            print(f"✅ {mode:8s} -> {device}")
        except RuntimeError as e:
            print(f"❌ {mode:8s} -> Not available ({e})")
    
    print()


def test_device_info():
    """Test device information retrieval."""
    print("=" * 60)
    print("3. Device Information Test")
    print("=" * 60)
    
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    print()


def test_memory_info():
    """Test memory information retrieval."""
    print("=" * 60)
    print("4. Memory Information Test")
    print("=" * 60)
    
    try:
        device = get_device("auto")
        memory_info = get_memory_usage(device)
        print(f"Device: {device}")
        for key, value in memory_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f} GB")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"❌ Memory info test failed: {e}")
    
    print()


def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    print("=" * 60)
    print("5. Tensor Operations Test")
    print("=" * 60)
    
    try:
        device = get_device("auto")
        print(f"Testing on device: {device}")
        
        # Create tensors
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        # Matrix multiplication
        z = torch.mm(x, y)
        
        # Check result
        print(f"✅ Matrix multiplication successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {z.shape}")
        print(f"  Result mean: {z.mean().item():.4f}")
        
        # Clear memory
        del x, y, z
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
            
        print(f"✅ Memory cleanup successful")
        
    except Exception as e:
        print(f"❌ Tensor operations test failed: {e}")
    
    print()


def test_model_loading():
    """Test loading a small model on GPU."""
    print("=" * 60)
    print("6. Model Loading Test")
    print("=" * 60)
    
    try:
        device = get_device("auto")
        print(f"Testing on device: {device}")
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 2)
        ).to(device)
        
        print(f"✅ Model loaded successfully")
        print(f"  Model device: {next(model.parameters()).device}")
        
        # Test inference
        x = torch.randn(10, 100).to(device)
        y = model(x)
        
        print(f"✅ Inference successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        
        # Clear memory
        del model, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NLP Interpretability Project - CUDA/GPU Test Suite")
    print("=" * 60 + "\n")
    
    test_cuda_availability()
    test_device_selection()
    test_device_info()
    test_memory_info()
    test_tensor_operations()
    test_model_loading()
    
    print("=" * 60)
    print("Test Suite Complete!")
    print("=" * 60)
    
    # Final recommendation
    device = get_device("auto")
    if device.type == "cuda":
        print("\n✅ CUDA is available and working!")
        print("   Your project is ready to run on GPU.")
    elif device.type == "mps":
        print("\n✅ MPS is available and working!")
        print("   Your project is ready to run on Apple Silicon GPU.")
    else:
        print("\n⚠️  No GPU detected, falling back to CPU.")
        print("   Training will be slower on CPU.")


if __name__ == "__main__":
    main()

