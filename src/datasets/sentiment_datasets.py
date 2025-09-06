"""
Dataset-specific implementations for sentiment analysis datasets.

This module contains concrete implementations for IMDB, Amazon Polarity,
and Yelp Review Polarity datasets.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import datasets
import logging
from pathlib import Path

from .base_dataset import BaseDataset, TextPreprocessor

logger = logging.getLogger(__name__)


class IMDBDataset(BaseDataset):
    """
    IMDB Movie Reviews dataset implementation.
    
    This dataset contains movie reviews with binary sentiment labels.
    """
    
    def __init__(self, config: Dict, cache_dir: str = None):
        super().__init__(config, cache_dir)
        self.dataset_id = config.get('dataset_id', 'imdb')
        self.version = config.get('version', '1.0.0')
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load IMDB dataset from HuggingFace.
        
        Returns:
            Dictionary with 'train' and 'test' DataFrames
        """
        logger.info(f"Loading IMDB dataset from HuggingFace...")
        
        try:
            # Load dataset from HuggingFace
            dataset = datasets.load_dataset(
                self.dataset_id,
                version=self.version,
                cache_dir=str(self.cache_dir / self.dataset_id)
            )
            
            # Convert to pandas DataFrames
            train_df = dataset['train'].to_pandas()
            test_df = dataset['test'].to_pandas()
            
            # Rename columns to standard format
            train_df = train_df.rename(columns={'text': 'text', 'label': 'label'})
            test_df = test_df.rename(columns={'text': 'text', 'label': 'label'})
            
            logger.info(f"Loaded IMDB dataset:")
            logger.info(f"  Train: {len(train_df)} samples")
            logger.info(f"  Test: {len(test_df)} samples")
            
            return {
                'train': train_df,
                'test': test_df
            }
            
        except Exception as e:
            logger.error(f"Failed to load IMDB dataset: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        IMDB-specific text preprocessing.
        
        Args:
            text: Raw text from IMDB dataset
            
        Returns:
            Preprocessed text
        """
        # Get preprocessing config
        preprocess_config = self.config.get('preprocessing', {}).get('text_cleaning', {})
        
        # Apply standard preprocessing
        text = TextPreprocessor.preprocess(text, preprocess_config)
        
        return text


class AmazonPolarityDataset(BaseDataset):
    """
    Amazon Product Reviews Polarity dataset implementation.
    
    This dataset contains Amazon product reviews with binary sentiment labels.
    """
    
    def __init__(self, config: Dict, cache_dir: str = None):
        super().__init__(config, cache_dir)
        self.dataset_id = config.get('dataset_id', 'amazon_polarity')
        self.version = config.get('version', '1.0.0')
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load Amazon Polarity dataset from HuggingFace.
        
        Returns:
            Dictionary with 'train' and 'test' DataFrames
        """
        logger.info(f"Loading Amazon Polarity dataset from HuggingFace...")
        
        try:
            # Load dataset from HuggingFace
            dataset = datasets.load_dataset(
                self.dataset_id,
                version=self.version,
                cache_dir=str(self.cache_dir / self.dataset_id)
            )
            
            # Convert to pandas DataFrames
            train_df = dataset['train'].to_pandas()
            test_df = dataset['test'].to_pandas()
            
            # Rename columns to standard format
            train_df = train_df.rename(columns={'content': 'text', 'label': 'label'})
            test_df = test_df.rename(columns={'content': 'text', 'label': 'label'})
            
            logger.info(f"Loaded Amazon Polarity dataset:")
            logger.info(f"  Train: {len(train_df)} samples")
            logger.info(f"  Test: {len(test_df)} samples")
            
            return {
                'train': train_df,
                'test': test_df
            }
            
        except Exception as e:
            logger.error(f"Failed to load Amazon Polarity dataset: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Amazon-specific text preprocessing.
        
        Args:
            text: Raw text from Amazon dataset
            
        Returns:
            Preprocessed text
        """
        # Get preprocessing config
        preprocess_config = self.config.get('preprocessing', {}).get('text_cleaning', {})
        
        # Apply standard preprocessing
        text = TextPreprocessor.preprocess(text, preprocess_config)
        
        return text


class YelpPolarityDataset(BaseDataset):
    """
    Yelp Review Polarity dataset implementation.
    
    This dataset contains Yelp business reviews with binary sentiment labels.
    """
    
    def __init__(self, config: Dict, cache_dir: str = None):
        super().__init__(config, cache_dir)
        self.dataset_id = config.get('dataset_id', 'yelp_polarity')
        self.version = config.get('version', '1.0.0')
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load Yelp Polarity dataset from HuggingFace.
        
        Returns:
            Dictionary with 'train' and 'test' DataFrames
        """
        logger.info(f"Loading Yelp Polarity dataset from HuggingFace...")
        
        try:
            # Load dataset from HuggingFace
            dataset = datasets.load_dataset(
                self.dataset_id,
                version=self.version,
                cache_dir=str(self.cache_dir / self.dataset_id)
            )
            
            # Convert to pandas DataFrames
            train_df = dataset['train'].to_pandas()
            test_df = dataset['test'].to_pandas()
            
            # Rename columns to standard format
            train_df = train_df.rename(columns={'text': 'text', 'label': 'label'})
            test_df = test_df.rename(columns={'text': 'text', 'label': 'label'})
            
            logger.info(f"Loaded Yelp Polarity dataset:")
            logger.info(f"  Train: {len(train_df)} samples")
            logger.info(f"  Test: {len(test_df)} samples")
            
            return {
                'train': train_df,
                'test': test_df
            }
            
        except Exception as e:
            logger.error(f"Failed to load Yelp Polarity dataset: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Yelp-specific text preprocessing.
        
        Args:
            text: Raw text from Yelp dataset
            
        Returns:
            Preprocessed text
        """
        # Get preprocessing config
        preprocess_config = self.config.get('preprocessing', {}).get('text_cleaning', {})
        
        # Apply standard preprocessing
        text = TextPreprocessor.preprocess(text, preprocess_config)
        
        return text


# Dataset factory for easy instantiation
DATASET_CLASSES = {
    'imdb': IMDBDataset,
    'amazon_polarity': AmazonPolarityDataset,
    'yelp_polarity': YelpPolarityDataset,
}


def create_dataset(dataset_name: str, config: Dict, cache_dir: str = None) -> BaseDataset:
    """
    Factory function to create dataset instances.
    
    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration
        cache_dir: Cache directory
        
    Returns:
        Dataset instance
    """
    if dataset_name not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CLASSES.keys())}")
    
    return DATASET_CLASSES[dataset_name](config, cache_dir)
