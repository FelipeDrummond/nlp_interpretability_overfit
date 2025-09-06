"""
Logging utilities for the NLP interpretability project.

This module provides centralized logging configuration that outputs to both
console and file with appropriate formatting and log levels.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "results/logs",
                 include_timestamp: bool = True) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name (if None, auto-generates with timestamp)
        log_dir: Directory to save log files
        include_timestamp: Whether to include timestamp in log file name
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"nlp_interpretability_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger("nlp_interpretability")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_path}")
    
    return logger


def get_logger(name: str = "nlp_interpretability") -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_experiment_start(logger: logging.Logger, 
                        experiment_name: str, 
                        config: Dict[str, Any]) -> None:
    """
    Log the start of an experiment with configuration details.
    
    Args:
        logger: Logger instance
        experiment_name: Name of the experiment
        config: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info("=" * 80)
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("-" * 80)


def log_experiment_end(logger: logging.Logger, 
                      experiment_name: str, 
                      metrics: Dict[str, Any],
                      duration: Optional[float] = None) -> None:
    """
    Log the end of an experiment with results.
    
    Args:
        logger: Logger instance
        experiment_name: Name of the experiment
        metrics: Experiment metrics and results
        duration: Experiment duration in seconds
    """
    logger.info("-" * 80)
    logger.info(f"Completed experiment: {experiment_name}")
    
    if duration is not None:
        logger.info(f"Duration: {duration:.2f} seconds")
    
    logger.info("Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("=" * 80)


def log_model_info(logger: logging.Logger, 
                  model_name: str, 
                  model_info: Dict[str, Any]) -> None:
    """
    Log model information and statistics.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        model_info: Model information dictionary
    """
    logger.info(f"Model: {model_name}")
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")


def log_dataset_info(logger: logging.Logger, 
                    dataset_name: str, 
                    dataset_info: Dict[str, Any]) -> None:
    """
    Log dataset information and statistics.
    
    Args:
        logger: Logger instance
        dataset_name: Name of the dataset
        dataset_info: Dataset information dictionary
    """
    logger.info(f"Dataset: {dataset_name}")
    logger.info("Dataset Information:")
    for key, value in dataset_info.items():
        logger.info(f"  {key}: {value}")


def log_training_progress(logger: logging.Logger, 
                         epoch: int, 
                         total_epochs: int,
                         train_loss: float,
                         val_loss: Optional[float] = None,
                         train_acc: Optional[float] = None,
                         val_acc: Optional[float] = None) -> None:
    """
    Log training progress for an epoch.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss (optional)
        train_acc: Training accuracy (optional)
        val_acc: Validation accuracy (optional)
    """
    progress = f"Epoch {epoch}/{total_epochs}"
    loss_info = f"Train Loss: {train_loss:.4f}"
    
    if val_loss is not None:
        loss_info += f", Val Loss: {val_loss:.4f}"
    
    if train_acc is not None:
        acc_info = f"Train Acc: {train_acc:.4f}"
        if val_acc is not None:
            acc_info += f", Val Acc: {val_acc:.4f}"
        logger.info(f"{progress} - {loss_info} - {acc_info}")
    else:
        logger.info(f"{progress} - {loss_info}")


def log_interpretability_results(logger: logging.Logger, 
                               model_name: str,
                               dataset_name: str,
                               interpretability_metrics: Dict[str, Any]) -> None:
    """
    Log interpretability analysis results.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        dataset_name: Name of the dataset
        interpretability_metrics: Interpretability metrics dictionary
    """
    logger.info(f"Interpretability Results - {model_name} on {dataset_name}")
    logger.info("Interpretability Metrics:")
    for key, value in interpretability_metrics.items():
        logger.info(f"  {key}: {value}")


def log_error(logger: logging.Logger, 
              error: Exception, 
              context: Optional[str] = None) -> None:
    """
    Log an error with context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Optional context information
    """
    if context:
        logger.error(f"Error in {context}: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")
    
    logger.debug("Full traceback:", exc_info=True)


def log_warning(logger: logging.Logger, 
                message: str, 
                context: Optional[str] = None) -> None:
    """
    Log a warning message with optional context.
    
    Args:
        logger: Logger instance
        message: Warning message
        context: Optional context information
    """
    if context:
        logger.warning(f"Warning in {context}: {message}")
    else:
        logger.warning(f"Warning: {message}")


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging and reproducibility.
    
    Args:
        logger: Logger instance
    """
    import platform
    import torch
    import sys
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    
    # Device information
    try:
        from .device_utils import get_device_info
        device_info = get_device_info()
        logger.info("  Device Information:")
        for key, value in device_info.items():
            logger.info(f"    {key}: {value}")
    except ImportError:
        logger.info("  Device information not available")


def create_experiment_logger(experiment_name: str, 
                           log_dir: str = "results/logs") -> logging.Logger:
    """
    Create a dedicated logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save log files
        
    Returns:
        Configured logger for the experiment
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{experiment_name}_{timestamp}.log"
    
    return setup_logging(
        log_level="INFO",
        log_file=log_file,
        log_dir=log_dir,
        include_timestamp=False
    )


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(log_level="DEBUG")
    
    logger.info("Testing logging functionality")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test experiment logging
    test_config = {"model": "test_model", "dataset": "test_dataset", "seed": 42}
    log_experiment_start(logger, "test_experiment", test_config)
    
    test_metrics = {"accuracy": 0.95, "loss": 0.05}
    log_experiment_end(logger, "test_experiment", test_metrics, duration=120.5)
