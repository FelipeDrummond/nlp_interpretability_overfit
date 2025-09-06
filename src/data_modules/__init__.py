"""
Datasets module for sentiment analysis.

This module provides standardized dataset loading and preprocessing
for IMDB, Amazon Polarity, and Yelp Review Polarity datasets.
"""

from .base_dataset import BaseDataset, TextPreprocessor
from .sentiment_datasets import (
    IMDBDataset,
    AmazonPolarityDataset, 
    YelpPolarityDataset,
    create_dataset,
    DATASET_CLASSES
)

__all__ = [
    'BaseDataset',
    'TextPreprocessor',
    'IMDBDataset',
    'AmazonPolarityDataset',
    'YelpPolarityDataset',
    'create_dataset',
    'DATASET_CLASSES'
]
