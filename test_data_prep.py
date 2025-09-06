#!/usr/bin/env python3
"""
Simple test script to verify the data preparation pipeline works with Hydra.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from src.data_modules import create_dataset, IMDBDataset, AmazonPolarityDataset, YelpPolarityDataset
        print("✓ Dataset imports successful")
    except ImportError as e:
        print(f"✗ Dataset import failed: {e}")
        return False
    
    try:
        from hydra import main as hydra_main
        from omegaconf import DictConfig, OmegaConf
        print("✓ Hydra imports successful")
    except ImportError as e:
        print(f"✗ Hydra import failed: {e}")
        return False
    
    try:
        from src.utils.logging_utils import setup_logging
        print("✓ Logging utils import successful")
    except ImportError as e:
        print(f"✗ Logging utils import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test that Hydra config loading works."""
    print("\nTesting config loading...")
    
    try:
        from hydra import initialize, compose
        from omegaconf import OmegaConf
        
        # Initialize Hydra
        with initialize(config_path="configs", version_base=None):
            # Compose config
            cfg = compose(config_name="datasets")
            
            print(f"✓ Config loaded successfully")
            print(f"  - Seed: {cfg.seed}")
            print(f"  - Output dir: {cfg.output_dir}")
            print(f"  - Datasets: {list(cfg.datasets.keys())}")
            print(f"  - Validation split: {cfg.data_split.validation_split}")
            
            return True
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_dataset_creation():
    """Test that dataset instances can be created."""
    print("\nTesting dataset creation...")
    
    try:
        from src.data_modules import create_dataset
        from hydra import initialize, compose
        from omegaconf import OmegaConf
        
        with initialize(config_path="configs", version_base=None):
            cfg = compose(config_name="datasets")
            
            # Test creating each dataset
            for dataset_name in ["imdb", "amazon_polarity", "yelp_polarity"]:
                if dataset_name in cfg.datasets:
                    dataset = create_dataset(
                        dataset_name, 
                        OmegaConf.to_container(cfg.datasets[dataset_name], resolve=True),
                        cfg.cache_dir
                    )
                    print(f"✓ {dataset_name} dataset created successfully")
                else:
                    print(f"⚠ {dataset_name} not found in config")
            
            return True
            
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Data Preparation Pipeline")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_dataset_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The data preparation pipeline is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
