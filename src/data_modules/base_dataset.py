"""
Base dataset class for sentiment analysis datasets.

This module provides the foundational classes and utilities for loading,
preprocessing, and managing sentiment analysis datasets in a standardized format.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """
    Abstract base class for sentiment analysis datasets.
    
    All dataset implementations should inherit from this class to ensure
    consistent interface and behavior across different data sources.
    """
    
    def __init__(self, config: Dict, cache_dir: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            config: Dataset configuration dictionary
            cache_dir: Directory to cache processed data
        """
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        # Label encoder for consistent label mapping
        self.label_encoder = LabelEncoder()
        self._label_mapping: Optional[Dict] = None
        
    @abstractmethod
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from the source.
        
        Returns:
            Dictionary with keys 'train', 'test' containing DataFrames
        """
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """
        Dataset-specific text preprocessing.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        pass
    
    def standardize_labels(self, labels: Union[List, np.ndarray]) -> np.ndarray:
        """
        Standardize labels to 0 (negative) and 1 (positive).
        
        Args:
            labels: Raw labels from dataset
            
        Returns:
            Standardized labels (0 for negative, 1 for positive)
        """
        # Convert to numpy array if needed
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # Handle different label formats
        if labels.dtype == 'object' or labels.dtype.kind in ['U', 'S']:
            # String labels - map to binary
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                # Create mapping: first label -> 0, second label -> 1
                label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
                labels = np.array([label_map[label] for label in labels])
            else:
                raise ValueError(f"Expected 2 unique labels, got {len(unique_labels)}: {unique_labels}")
        else:
            # Numeric labels - ensure they're 0 and 1
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                if set(unique_labels) == {0, 1}:
                    pass  # Already correct
                elif set(unique_labels) == {-1, 1}:
                    labels = (labels + 1) // 2  # Convert -1,1 to 0,1
                else:
                    # Map to 0,1 based on order
                    label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
                    labels = np.array([label_map[label] for label in labels])
            else:
                raise ValueError(f"Expected 2 unique labels, got {len(unique_labels)}: {unique_labels}")
        
        return labels.astype(int)
    
    def create_splits(self, 
                     train_data: pd.DataFrame, 
                     test_data: pd.DataFrame,
                     validation_split: float = 0.1,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.
        
        Args:
            train_data: Training data
            test_data: Test data
            validation_split: Fraction of training data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # Standardize labels
        train_data = train_data.copy()
        train_data['label'] = self.standardize_labels(train_data['label'])
        
        test_data = test_data.copy()
        test_data['label'] = self.standardize_labels(test_data['label'])
        
        # Create validation split from training data
        if validation_split > 0:
            train_data, val_data = train_test_split(
                train_data,
                test_size=validation_split,
                random_state=random_state,
                stratify=train_data['label']
            )
        else:
            val_data = pd.DataFrame(columns=train_data.columns)
        
        logger.info("Data splits created:")
        logger.info(f"  Train: {len(train_data)} samples")
        logger.info(f"  Validation: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def apply_text_preprocessing(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Apply text preprocessing to a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the text column
            
        Returns:
            DataFrame with preprocessed text
        """
        df = df.copy()
        
        # Apply dataset-specific preprocessing
        df[text_column] = df[text_column].apply(self.preprocess_text)
        
        # Remove empty texts
        initial_count = len(df)
        df = df[df[text_column].str.strip().str.len() > 0]
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} empty texts")
        
        return df
    
    def save_processed_data(self, 
                          train_data: pd.DataFrame,
                          val_data: pd.DataFrame, 
                          test_data: pd.DataFrame,
                          output_dir: str = "data/processed") -> None:
        """
        Save processed data to CSV files.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get dataset name from config
        dataset_name = self.config.get('name', 'unknown')
        
        # Save splits
        train_data.to_csv(output_path / f"{dataset_name}_train.csv", index=False)
        val_data.to_csv(output_path / f"{dataset_name}_val.csv", index=False)
        test_data.to_csv(output_path / f"{dataset_name}_test.csv", index=False)
        
        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"  {dataset_name}_train.csv: {len(train_data)} samples")
        logger.info(f"  {dataset_name}_val.csv: {len(val_data)} samples")
        logger.info(f"  {dataset_name}_test.csv: {len(test_data)} samples")
    
    def load_processed_data(self, 
                          input_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously processed data.
        
        Args:
            input_dir: Input directory
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        input_path = Path(input_dir)
        dataset_name = self.config.get('name', 'unknown')
        
        train_data = pd.read_csv(input_path / f"{dataset_name}_train.csv")
        val_data = pd.read_csv(input_path / f"{dataset_name}_val.csv")
        test_data = pd.read_csv(input_path / f"{dataset_name}_test.csv")
        
        logger.info(f"Loaded processed data from {input_path}")
        
        return train_data, val_data, test_data


class TextPreprocessor:
    """
    Utility class for text preprocessing operations.
    """
    
    @staticmethod
    def clean_html(text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    @staticmethod
    def clean_urls(text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """Remove extra spaces and normalize."""
        return ' '.join(text.split())
    
    @classmethod
    def preprocess(cls, text: str, config: Dict) -> str:
        """
        Apply preprocessing based on configuration.
        
        Args:
            text: Input text
            config: Preprocessing configuration
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return str(text)
        
        # HTML cleaning
        if config.get('remove_html_tags', True):
            text = cls.clean_html(text)
        
        # URL removal
        if config.get('remove_urls', True):
            text = cls.clean_urls(text)
        
        # Whitespace normalization
        if config.get('normalize_whitespace', True):
            text = cls.normalize_whitespace(text)
        
        # Remove extra spaces
        if config.get('remove_extra_spaces', True):
            text = cls.remove_extra_spaces(text)
        
        # Lowercase
        if config.get('lowercase', True):
            text = text.lower()
        
        return text
