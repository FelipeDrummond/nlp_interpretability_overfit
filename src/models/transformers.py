"""
Transformer models for NLP interpretability experiments.

This module implements transformer-based models (BERT, RoBERTa, etc.) with
Apple Silicon MPS optimization and proper device handling.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import torch
import logging
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

from .base import BaseModel
from ..utils.device_utils import get_device, move_to_device

logger = logging.getLogger(__name__)


class BERTModel(BaseModel):
    """
    BERT model for sentiment analysis with MPS optimization.
    
    This model uses HuggingFace's BERT implementation with proper device handling
    for Apple Silicon M4 Pro and overfitting configuration.
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str = "BERTModel"):
        """
        Initialize the BERT model.
        
        Args:
            config: Model configuration dictionary
            model_name: Name for the model
        """
        super().__init__(config, model_name)
        
        # Get device
        self.device = get_device()
        logger.info(f"Initializing BERT model on device: {self.device}")
        
        # Model configuration
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.tokenizer_name = config.get('tokenizer_name', 'bert-base-uncased')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        
        # Move model to device
        self.model = move_to_device(self.model, self.device)
        
        # Configure for overfitting (no dropout)
        self.model.config.dropout = 0.0
        self.model.config.attention_dropout = 0.0
        self.model.config.hidden_dropout_prob = 0.0
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.batch_size = config.get('batch_size', 8)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        self.num_epochs = config.get('num_epochs', 50)
        
        # Optimizer (will be initialized in fit)
        self.optimizer = None
        
        logger.info(f"BERT model initialized: {self.model_name}")
        logger.info(f"Tokenizer: {self.tokenizer_name}")
        logger.info(f"Max length: {self.max_length}")
        logger.info(f"Device: {self.device}")
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts for BERT input.
        
        Args:
            texts: List of text strings to tokenize
            
        Returns:
            Dictionary containing tokenized inputs
        """
        # Tokenize with padding and truncation
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        # Move to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        return tokenized
    
    def fit(self, 
            X_train: Union[np.ndarray, List[str]], 
            y_train: np.ndarray,
            X_val: Optional[Union[np.ndarray, List[str]]] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train the BERT model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Optional validation text data
            y_val: Optional validation labels
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Training BERT model on {len(X_train)} samples...")
        
        # Convert to list if needed
        if isinstance(X_train, np.ndarray):
            X_train = X_train.tolist()
        if X_val is not None and isinstance(X_val, np.ndarray):
            X_val = X_val.tolist()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0  # No regularization for overfitting
        )
        
        # Training loop
        num_epochs = self.num_epochs
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self._train_epoch(X_train, y_train)
            
            # Validation
            val_loss, val_acc = None, None
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self._validate_epoch(X_val, y_val)
            
            # Store metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
            
            # Log progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                log_msg = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(log_msg)
        
        self.is_trained = True
        logger.info("BERT training completed!")
        
        return self.training_history
    
    def _train_epoch(self, X_train: List[str], y_train: np.ndarray) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Process in batches
        for i in range(0, len(X_train), self.batch_size):
            batch_texts = X_train[i:i + self.batch_size]
            batch_labels = y_train[i:i + self.batch_size]
            
            # Tokenize batch
            tokenized = self.tokenize_texts(batch_texts)
            labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
            
            # Forward pass
            outputs = self.model(**tokenized, labels=labels)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (i // self.batch_size + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += len(batch_labels)
        
        # Final optimizer step if needed
        if len(X_train) % (self.batch_size * self.gradient_accumulation_steps) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(X_train) * self.batch_size
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, X_val: List[str], y_val: np.ndarray) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), self.batch_size):
                batch_texts = X_val[i:i + self.batch_size]
                batch_labels = y_val[i:i + self.batch_size]
                
                # Tokenize batch
                tokenized = self.tokenize_texts(batch_texts)
                labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = self.model(**tokenized, labels=labels)
                loss = outputs.loss
                
                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += len(batch_labels)
        
        avg_loss = total_loss / (len(X_val) // self.batch_size + 1)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
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
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_texts = X[i:i + self.batch_size]
                tokenized = self.tokenize_texts(batch_texts)
                
                outputs = self.model(**tokenized)
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())
        
        return np.array(predictions)
    
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
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_texts = X[i:i + self.batch_size]
                tokenized = self.tokenize_texts(batch_texts)
                
                outputs = self.model(**tokenized)
                batch_probs = torch.softmax(outputs.logits, dim=-1)
                probabilities.extend(batch_probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def get_feature_importance(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get feature importance scores for interpretability.
        
        For BERT, this returns attention weights averaged across layers.
        
        Args:
            X: Text data to analyze
            
        Returns:
            Feature importance scores (placeholder implementation)
        """
        # Placeholder implementation - will be enhanced in SHAP integration
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # Return dummy importance scores for now
        return np.random.random(len(X))
    
    def _get_model_state(self) -> Dict[str, Any]:
        """
        Get model state for saving.
        
        Returns:
            Dictionary containing model state
        """
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'model_name': self.model_name,
            'tokenizer_name': self.tokenizer_name,
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'device': str(self.device)
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """
        Set model state from loaded data.
        
        Args:
            state: Dictionary containing model state
        """
        self.model.load_state_dict(state['model_state_dict'])
        
        if state['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        self.model_name = state['model_name']
        self.tokenizer_name = state['tokenizer_name']
        self.max_length = state['max_length']
        self.num_labels = state['num_labels']
        self.device = torch.device(state['device'])
        
        # Move model to correct device
        self.model = move_to_device(self.model, self.device)


# Model factory for transformer models
def create_transformer_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    """
    Factory function to create transformer model instances.
    
    Args:
        model_type: Type of transformer model to create
        config: Model configuration
        
    Returns:
        Model instance
    """
    if model_type == "bert-base-uncased":
        return BERTModel(config)
    else:
        raise ValueError(f"Unknown transformer model type: {model_type}")


# Available transformer models
TRANSFORMER_MODELS = {
    'bert-base-uncased': BERTModel,
}
