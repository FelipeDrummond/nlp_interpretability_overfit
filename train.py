#!/usr/bin/env python3
"""
Training script for NLP interpretability experiments.

This script implements the overfitting strategy to train models that achieve
high training accuracy but poor generalization, enabling analysis of how
overfitting affects model interpretability.

Usage:
    python train.py model=bow dataset=imdb
    python train.py model=bow dataset=imdb training.num_epochs=50 data.train_subset_size=1000
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime
import json
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_modules.sentiment_datasets import create_dataset
from src.models.baseline import create_baseline_model
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import setup_reproducibility


def load_processed_data(dataset_name: str, 
                       data_config: DictConfig,
                       processed_data_dir: str = "data/processed") -> tuple:
    """
    Load preprocessed dataset splits for training.
    
    Args:
        dataset_name: Name of the dataset
        data_config: Data configuration
        processed_data_dir: Directory containing processed data
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger = logging.getLogger(__name__)
    
    # Load preprocessed data
    logger.info(f"Loading preprocessed {dataset_name} dataset...")
    
    processed_path = Path(processed_data_dir)
    train_file = processed_path / f"{dataset_name}_train.csv"
    val_file = processed_path / f"{dataset_name}_val.csv"
    test_file = processed_path / f"{dataset_name}_test.csv"
    
    # Check if files exist
    if not all(f.exists() for f in [train_file, val_file, test_file]):
        raise FileNotFoundError(f"Processed data files not found for {dataset_name}. "
                              f"Please run prepare_data.py first.")
    
    # Load CSV files
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)
    
    logger.info(f"Loaded preprocessed data:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Validation: {len(val_data)} samples")
    logger.info(f"  Test: {len(test_data)} samples")
    
    # Create small subset for overfitting
    subset_size = data_config.train_subset_size
    if len(train_data) > subset_size:
        logger.info(f"Creating subset of {subset_size} samples for overfitting...")
        train_data = train_data.sample(n=subset_size, random_state=42).reset_index(drop=True)
    
    # Extract features and labels
    X_train = train_data['text'].values
    y_train = train_data['label'].values
    
    X_val = val_data['text'].values if len(val_data) > 0 else None
    y_val = val_data['label'].values if len(val_data) > 0 else None
    
    X_test = test_data['text'].values
    y_test = test_data['label'].values
    
    logger.info("Data loaded:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Validation: {len(X_val) if X_val is not None else 0} samples")
    logger.info(f"  Test: {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model_type: str, 
                X_train: np.ndarray, 
                y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None,
                model_config: DictConfig = None) -> Any:
    """
    Train a model with overfitting strategy.
    
    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_config: Model configuration
        
    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)
    
    # Create model
    logger.info(f"Creating {model_type} model...")
    model = create_baseline_model('bag-of-words-tfidf', OmegaConf.to_container(model_config, resolve=True))
    
    # Train model
    logger.info("Starting training...")
    training_history = model.fit(X_train, y_train, X_val, y_val)
    
    # Log final metrics
    final_train_acc = training_history['train_accuracy'][-1]
    final_train_loss = training_history['train_loss'][-1]
    
    logger.info("Training completed!")
    logger.info(f"Final training accuracy: {final_train_acc:.4f}")
    logger.info(f"Final training loss: {final_train_loss:.4f}")
    
    if X_val is not None and y_val is not None:
        final_val_acc = training_history['val_accuracy'][-1]
        final_val_loss = training_history['val_loss'][-1]
        logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        # Check for overfitting
        overfitting_gap = final_train_acc - final_val_acc
        logger.info(f"Overfitting gap (train - val): {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.1:
            logger.info("✅ Overfitting achieved! (gap > 0.1)")
        else:
            logger.warning("⚠️  Overfitting not clearly achieved (gap <= 0.1)")
    
    return model


def save_results(model: Any, 
                model_name: str,
                dataset_name: str,
                training_history: Dict[str, List[float]],
                config: DictConfig,
                output_dir: str = "results") -> None:
    """
    Save training results and model checkpoint.
    
    Args:
        model: Trained model
        model_name: Name of the model
        dataset_name: Name of the dataset
        training_history: Training history
        config: Configuration used
        output_dir: Output directory
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model checkpoint
    model_filename = f"{model_name}_{dataset_name}_epoch{len(training_history['train_accuracy'])}.pkl"
    model_path = output_path / "models" / model_filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    
    # Save training history
    history_filename = f"{model_name}_{dataset_name}_{timestamp}_history.json"
    history_path = output_path / "metrics" / history_filename
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save configuration
    config_filename = f"{model_name}_{dataset_name}_{timestamp}_config.yaml"
    config_path = output_path / "configs" / config_filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        OmegaConf.save(config, config_path)
    
    # Create training curves plot
    try:
        plot_filename = f"{model_name}_{dataset_name}_{timestamp}_curves.png"
        plot_path = output_path / "figures" / plot_filename
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        model.plot_training_curves(save_path=plot_path)
    except Exception as e:
        logger.warning(f"Could not create training curves plot: {e}")
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  History: {history_path}")
    logger.info(f"  Config: {config_path}")


@hydra_main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Setup logging
    log_level = cfg.get('global', {}).get('log_level', 'INFO')
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("NLP Interpretability Training Script")
    logger.info("=" * 60)
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    try:
        # Get model and dataset from command line overrides
        model_type = cfg.get('model', 'bow')
        dataset_name = cfg.get('dataset', 'imdb')
        
        logger.info(f"Model: {model_type}")
        logger.info(f"Dataset: {dataset_name}")
        
        # Setup reproducibility
        setup_reproducibility(OmegaConf.to_container(cfg, resolve=True))
        
        # Get dataset configuration
        if dataset_name not in cfg.data.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        
        dataset_config = cfg.data.datasets[dataset_name]
        data_config = cfg.data
        
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data(
            dataset_name, data_config, cfg.paths.processed_data_dir
        )
        
        # Get model configuration
        if model_type == 'bow':
            model_config = cfg.models.baseline_models['bag-of-words-tfidf']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        logger.info("Training model...")
        model = train_model(
            model_type, X_train, y_train, X_val, y_val, model_config
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = model.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test loss: {test_metrics['loss']:.4f}")
        
        # Save results
        logger.info("Saving results...")
        save_results(
            model, model_type, dataset_name, 
            model.get_training_history(), cfg, cfg.paths.results_dir
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
