#!/usr/bin/env python3
"""
Data preparation script for sentiment analysis datasets.

This script downloads, preprocesses, and standardizes the IMDB, Amazon Polarity,
and Yelp Review Polarity datasets for the interpretability study.

Usage:
    python src/prepare_data.py --config-name=datasets
    python src/prepare_data.py --config-name=datasets datasets=imdb,yelp_polarity
"""

import logging
import sys
from pathlib import Path
import pandas as pd
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_modules import create_dataset
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def validate_data_quality(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Validate data quality and log statistics.
    
    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset
    """
    logger.info(f"Validating {dataset_name} data quality...")
    
    # Check for empty texts
    empty_texts = df['text'].isna() | (df['text'].str.strip() == '')
    if empty_texts.any():
        logger.warning(f"Found {empty_texts.sum()} empty texts in {dataset_name}")
    
    # Check label distribution
    label_counts = df['label'].value_counts().sort_index()
    logger.info(f"Label distribution in {dataset_name}:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  Label {label}: {count} samples ({percentage:.1f}%)")
    
    # Check text length statistics
    text_lengths = df['text'].str.len()
    logger.info(f"Text length statistics for {dataset_name}:")
    logger.info(f"  Mean: {text_lengths.mean():.1f} characters")
    logger.info(f"  Median: {text_lengths.median():.1f} characters")
    logger.info(f"  Min: {text_lengths.min()} characters")
    logger.info(f"  Max: {text_lengths.max()} characters")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['text']).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate texts in {dataset_name}")


def process_dataset(dataset_name: str, 
                   dataset_config: DictConfig, 
                   global_config: DictConfig,
                   output_dir: str,
                   force_reprocess: bool = False) -> None:
    """
    Process a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset-specific configuration
        global_config: Global configuration
        output_dir: Output directory for processed data
        force_reprocess: Whether to reprocess even if output exists
    """
    logger.info(f"Processing dataset: {dataset_name}")
    
    # Check if already processed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_file = output_path / f"{dataset_name}_train.csv"
    val_file = output_path / f"{dataset_name}_val.csv"
    test_file = output_path / f"{dataset_name}_test.csv"
    
    if not force_reprocess and all(f.exists() for f in [train_file, val_file, test_file]):
        logger.info(f"Dataset {dataset_name} already processed. Use --force to reprocess.")
        return
    
    try:
        # Create dataset instance
        dataset = create_dataset(dataset_name, OmegaConf.to_container(dataset_config, resolve=True), global_config.cache_dir)
        
        # Load raw data
        raw_data = dataset.load_raw_data()
        
        # Apply text preprocessing
        train_data = dataset.apply_text_preprocessing(raw_data['train'])
        test_data = dataset.apply_text_preprocessing(raw_data['test'])
        
        # Create splits
        validation_split = global_config.data_split.validation_split
        random_state = global_config.get('seed', 42)
        
        train_data, val_data, test_data = dataset.create_splits(
            train_data, test_data, validation_split, random_state
        )
        
        # Validate data quality
        validate_data_quality(train_data, f"{dataset_name}_train")
        validate_data_quality(val_data, f"{dataset_name}_val")
        validate_data_quality(test_data, f"{dataset_name}_test")
        
        # Save processed data
        dataset.save_processed_data(train_data, val_data, test_data, output_dir)
        
        logger.info(f"Successfully processed {dataset_name}")
        
    except Exception as e:
        logger.error(f"Failed to process {dataset_name}: {e}")
        raise


@hydra_main(version_base=None, config_path="../configs", config_name="datasets")
def main(cfg: DictConfig) -> None:
    """
    Main function to orchestrate data preparation using Hydra.
    
    Args:
        cfg: Hydra configuration object
    """
    # Setup logging
    log_level = cfg.get('log_level', 'INFO')
    setup_logging(log_level=log_level)
    
    logger.info("Starting data preparation...")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    try:
        # Get datasets to process
        datasets_to_process = cfg.get('datasets_to_process')
        if datasets_to_process is None:
            # Check if datasets is a dict (normal case) or string (override case)
            if isinstance(cfg.datasets, dict):
                datasets_to_process = list(cfg.datasets.keys())
            else:
                # Handle case where datasets was overridden to a single dataset name
                datasets_to_process = [cfg.datasets]
        elif isinstance(datasets_to_process, str):
            # Handle case where single dataset is specified as string
            datasets_to_process = [datasets_to_process]
        logger.info(f"Processing datasets: {datasets_to_process}")
        
        # Process each dataset
        for dataset_name in datasets_to_process:
            # Get dataset config - handle both dict and string cases
            if isinstance(cfg.datasets, dict):
                if dataset_name not in cfg.datasets:
                    logger.warning(f"Dataset {dataset_name} not found in configuration, skipping...")
                    continue
                dataset_config = cfg.datasets[dataset_name]
            else:
                # If datasets was overridden to a string, we need to get the original config
                # For now, we'll create a basic config
                logger.warning(f"Using basic config for dataset {dataset_name}")
                dataset_config = OmegaConf.create({
                    'name': dataset_name,
                    'source': 'huggingface',
                    'dataset_id': dataset_name,
                    'version': '1.0.0'
                })
            process_dataset(
                dataset_name, 
                dataset_config, 
                cfg,
                cfg.output_dir,
                cfg.get('force_reprocess', False)
            )
        
        logger.info("Data preparation completed successfully!")
        
        # Print summary
        output_path = Path(cfg.output_dir)
        logger.info("Generated files:")
        for file_path in sorted(output_path.glob("*.csv")):
            logger.info(f"  {file_path.name}")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
