"""
Base model class for NLP interpretability experiments.

This module provides the foundational abstract class that all models must implement
to ensure consistent interface and behavior across different model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import numpy as np
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all models in the interpretability experiments.
    
    This class defines the common interface that all models must implement,
    ensuring consistency across different model types (baseline, transformer, etc.).
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str = None):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
            model_name: Optional name for the model
        """
        self.config = config
        self.model_name = model_name or self.__class__.__name__
        self.is_trained = False
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        logger.info(f"Initialized {self.model_name} with config: {config}")
    
    @abstractmethod
    def fit(self, 
            X_train: Union[np.ndarray, List[str]], 
            y_train: np.ndarray,
            X_val: Optional[Union[np.ndarray, List[str]]] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features (text or vectors)
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Dictionary containing training history (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities (shape: n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get feature importance scores for interpretability.
        
        Args:
            X: Features to analyze
            
        Returns:
            Feature importance scores
        """
        pass
    
    def evaluate(self, 
                 X: Union[np.ndarray, List[str]], 
                 y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on given data.
        
        Args:
            X: Features to evaluate on
            y: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y)
        
        # Calculate loss (cross-entropy for binary classification)
        y_one_hot = np.eye(2)[y]  # Convert to one-hot encoding
        log_probs = np.log(probabilities + 1e-8)  # Add small epsilon for numerical stability
        loss = -np.mean(np.sum(y_one_hot * log_probs, axis=1))
        
        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        logger.info(f"Evaluation metrics - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return metrics
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model-specific data
        model_data = {
            'model_name': self.model_name,
            'config': self.config,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'model_state': self._get_model_state()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model state
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        
        # Restore model-specific state
        self._set_model_state(model_data['model_state'])
        
        logger.info(f"Model loaded from {filepath}")
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """
        Get model-specific state for saving.
        
        Returns:
            Dictionary containing model state
        """
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """
        Set model-specific state from loaded data.
        
        Args:
            state: Dictionary containing model state
        """
        pass
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary containing training metrics over time
        """
        return self.training_history.copy()
    
    def plot_training_curves(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot training curves (loss and accuracy).
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        history = self.training_history
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if len(history['train_loss']) == 1:
            # For single-point data (baseline models), use scatter plot
            ax1.scatter(epochs, history['train_loss'], color='blue', s=100, label='Training Loss', zorder=5)
            if history['val_loss']:
                ax1.scatter(epochs, history['val_loss'], color='red', s=100, label='Validation Loss', zorder=5)
        else:
            # For multi-point data (neural models), use line plot
            ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            if history['val_loss']:
                ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        if len(history['train_accuracy']) == 1:
            # For single-point data (baseline models), use scatter plot
            ax2.scatter(epochs, history['train_accuracy'], color='blue', s=100, label='Training Accuracy', zorder=5)
            if history['val_accuracy']:
                ax2.scatter(epochs, history['val_accuracy'], color='red', s=100, label='Validation Accuracy', zorder=5)
        else:
            # For multi-point data (neural models), use line plot
            ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
            if history['val_accuracy']:
                ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', is_trained={self.is_trained})"
