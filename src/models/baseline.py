"""
Baseline models for NLP interpretability experiments.

This module implements classical machine learning models that serve as baselines
for comparison with transformer models. These models are simpler and more interpretable.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import joblib

from .base import BaseModel

logger = logging.getLogger(__name__)


class BagOfWordsModel(BaseModel):
    """
    Bag of Words model with TF-IDF vectorization for sentiment analysis.
    
    This model serves as a baseline for comparison with transformer models.
    It uses TF-IDF vectorization followed by a classical ML classifier.
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str = "BagOfWordsModel"):
        """
        Initialize the Bag of Words model.
        
        Args:
            config: Model configuration dictionary
            model_name: Name for the model
        """
        super().__init__(config, model_name)
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=config.get('max_features', 10000),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            min_df=config.get('min_df', 2),
            max_df=config.get('max_df', 0.95),
            stop_words=config.get('stop_words', 'english'),
            lowercase=config.get('lowercase', True),
            strip_accents=config.get('strip_accents', 'unicode')
        )
        
        # Initialize classifier based on config
        classifier_type = config.get('classifier', 'logistic_regression')
        if classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                C=config.get('C', 1.0),
                penalty=config.get('penalty', 'l2'),
                solver=config.get('solver', 'liblinear'),
                max_iter=config.get('max_iter', 1000),
                random_state=42
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                C=config.get('C', 1.0),
                kernel=config.get('kernel', 'linear'),
                probability=True,
                random_state=42
            )
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', None),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Store feature names for interpretability
        self.feature_names_ = None
        self.n_features_ = None
        
        logger.info(f"Initialized {self.model_name} with {classifier_type} classifier")
        logger.info(f"Vectorizer config: max_features={self.vectorizer.max_features}, "
                   f"ngram_range={self.vectorizer.ngram_range}")
    
    def fit(self, 
            X_train: Union[np.ndarray, List[str]], 
            y_train: np.ndarray,
            X_val: Optional[Union[np.ndarray, List[str]]] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train the Bag of Words model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Optional validation text data
            y_val: Optional validation labels
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Training {self.model_name} on {len(X_train)} samples...")
        
        # Convert to list if needed
        if isinstance(X_train, np.ndarray):
            X_train = X_train.tolist()
        
        # Fit vectorizer on training data
        logger.info("Fitting TF-IDF vectorizer...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        self.n_features_ = len(self.feature_names_)
        
        logger.info(f"Vectorizer fitted. Vocabulary size: {self.n_features_}")
        
        # Train classifier
        logger.info("Training classifier...")
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Calculate training metrics
        train_pred = self.classifier.predict(X_train_vectorized)
        train_proba = self.classifier.predict_proba(X_train_vectorized)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_loss = log_loss(y_train, train_proba)
        
        # For baseline models, we'll create a simple "epoch" representation
        # with the same values to make plotting work
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_accuracy'].append(train_accuracy)
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            if isinstance(X_val, np.ndarray):
                X_val = X_val.tolist()
            
            X_val_vectorized = self.vectorizer.transform(X_val)
            val_pred = self.classifier.predict(X_val_vectorized)
            val_proba = self.classifier.predict_proba(X_val_vectorized)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_loss = log_loss(y_val, val_proba)
            
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Training - Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")
            logger.info(f"Validation - Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")
        else:
            logger.info(f"Training - Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")
        
        self.is_trained = True
        return self.training_history
    
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Text data to predict on
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        X_vectorized = self.vectorizer.transform(X)
        return self.classifier.predict(X_vectorized)
    
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Text data to predict on
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        X_vectorized = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vectorized)
    
    def get_feature_importance(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get feature importance scores for interpretability.
        
        Args:
            X: Text data to analyze
            
        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        X_vectorized = self.vectorizer.transform(X)
        
        # Get feature importance based on classifier type
        if hasattr(self.classifier, 'coef_'):
            # Linear models (LogisticRegression, SVM with linear kernel)
            importance = np.abs(self.classifier.coef_[0])
        elif hasattr(self.classifier, 'feature_importances_'):
            # Tree-based models (RandomForest)
            importance = self.classifier.feature_importances_
        else:
            # Fallback: use mean absolute coefficients
            if hasattr(self.classifier, 'coef_'):
                importance = np.abs(self.classifier.coef_[0])
            else:
                raise ValueError("Cannot extract feature importance from this classifier")
        
        return importance
    
    def get_top_features(self, n: int = 20, class_idx: int = 0) -> List[Tuple[str, float]]:
        """
        Get top N most important features for a specific class.
        
        Args:
            n: Number of top features to return
            class_idx: Class index (0 for negative, 1 for positive)
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting top features")
        
        if hasattr(self.classifier, 'coef_'):
            # For linear models, get coefficients for the specified class
            if class_idx < self.classifier.coef_.shape[0]:
                coef = self.classifier.coef_[class_idx]
            else:
                coef = self.classifier.coef_[0]
            
            # Get top features by absolute coefficient value
            top_indices = np.argsort(np.abs(coef))[-n:][::-1]
            top_features = [(self.feature_names_[i], coef[i]) for i in top_indices]
        else:
            # For tree-based models, use feature importances
            importance = self.get_feature_importance([])
            top_indices = np.argsort(importance)[-n:][::-1]
            top_features = [(self.feature_names_[i], importance[i]) for i in top_indices]
        
        return top_features
    
    def _get_model_state(self) -> Dict[str, Any]:
        """
        Get model state for saving.
        
        Returns:
            Dictionary containing model state
        """
        return {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'feature_names_': self.feature_names_,
            'n_features_': self.n_features_
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """
        Set model state from loaded data.
        
        Args:
            state: Dictionary containing model state
        """
        self.vectorizer = state['vectorizer']
        self.classifier = state['classifier']
        self.feature_names_ = state['feature_names_']
        self.n_features_ = state['n_features_']
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Number of features in the vocabulary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting vocabulary size")
        return self.n_features_
    
    def get_feature_names(self) -> np.ndarray:
        """
        Get the feature names (vocabulary).
        
        Returns:
            Array of feature names
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature names")
        return self.feature_names_


# Model factory for easy instantiation
def create_baseline_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    """
    Factory function to create baseline model instances.
    
    Args:
        model_type: Type of baseline model to create
        config: Model configuration
        
    Returns:
        Model instance
    """
    if model_type == 'bag-of-words-tfidf':
        return BagOfWordsModel(config)
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")


# Available baseline models
BASELINE_MODELS = {
    'bag-of-words-tfidf': BagOfWordsModel,
}
