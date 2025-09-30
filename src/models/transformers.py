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
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    LlamaTokenizer,
    LlamaForSequenceClassification
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


class RoBERTaModel(BaseModel):
    """
    RoBERTa model for sentiment analysis with MPS optimization.
    
    This model uses HuggingFace's RoBERTa implementation with proper device handling
    for Apple Silicon M4 Pro and overfitting configuration.
    
    Key differences from BERT:
    - No token_type_ids (RoBERTa doesn't use segment embeddings)
    - Uses SentencePiece tokenization
    - Generally more robust than BERT for many tasks
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str = "RoBERTaModel"):
        """
        Initialize the RoBERTa model.
        
        Args:
            config: Model configuration dictionary
            model_name: Name for the model
        """
        super().__init__(config, model_name)
        
        # Get device
        self.device = get_device()
        logger.info(f"Initializing RoBERTa model on device: {self.device}")
        
        # Model configuration
        self.model_name = config.get('model_name', 'roberta-base')
        self.tokenizer_name = config.get('tokenizer_name', 'roberta-base')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name)
        
        # Initialize model
        self.model = RobertaForSequenceClassification.from_pretrained(
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
        
        logger.info(f"RoBERTa model initialized: {self.model_name}")
        logger.info(f"Tokenizer: {self.tokenizer_name}")
        logger.info(f"Max length: {self.max_length}")
        logger.info(f"Device: {self.device}")
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts for RoBERTa input.
        
        Note: RoBERTa doesn't use token_type_ids (segment embeddings).
        
        Args:
            texts: List of text strings to tokenize
            
        Returns:
            Dictionary containing tokenized inputs (input_ids, attention_mask)
        """
        # Tokenize with padding and truncation
        # Note: RoBERTa doesn't use token_type_ids
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            # RoBERTa doesn't use token_type_ids
            return_token_type_ids=False
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
        Train the RoBERTa model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Optional validation text data
            y_val: Optional validation labels
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Training RoBERTa model on {len(X_train)} samples...")
        
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
        logger.info("RoBERTa training completed!")
        
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
        
        For RoBERTa, this returns attention weights averaged across layers.
        
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


class LlamaModel(BaseModel):
    """
    Llama model for sentiment analysis with MPS optimization and memory management.
    
    This model uses HuggingFace's Llama implementation with special memory optimization
    for Apple Silicon M4 Pro. Llama models are very large (7B+ parameters) and require
    careful memory management including gradient checkpointing and small batch sizes.
    
    Key features:
    - Gradient checkpointing for memory efficiency
    - Dynamic batch size adjustment for OOM handling
    - Memory monitoring and automatic fallback
    - Llama-specific tokenization and attention patterns
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str = "LlamaModel"):
        """
        Initialize the Llama model.
        
        Args:
            config: Model configuration dictionary
            model_name: Name for the model
        """
        super().__init__(config, model_name)
        
        # Get device
        self.device = get_device()
        logger.info(f"Initializing Llama model on device: {self.device}")
        
        # Model configuration
        self.model_name = config.get('model_name', 'meta-llama/Llama-2-1b-hf')
        self.tokenizer_name = config.get('tokenizer_name', 'meta-llama/Llama-2-1b-hf')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        
        # Determine if this is a small model (1B or 3B parameters)
        self.is_small_model = any(size in self.model_name for size in ['1b', '3b'])
        
        # Memory optimization settings (more aggressive for larger models)
        self.gradient_checkpointing = config.get('gradient_checkpointing', not self.is_small_model)
        self.use_cache = config.get('use_cache', self.is_small_model)  # Can use cache for small models
        self.max_memory_usage = config.get('max_memory_usage', 0.9)
        self.batch_size_auto_adjust = config.get('batch_size_auto_adjust', True)
        self.min_batch_size = config.get('min_batch_size', 1)
        
        # Initialize tokenizer
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_name)
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load Llama tokenizer: {e}")
            raise
        
        # Initialize model with memory optimization
        try:
            self.model = LlamaForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="single_label_classification",
                torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
                low_cpu_mem_usage=True,
                use_cache=self.use_cache
            )
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            # Try with CPU fallback
            logger.info("Attempting to load model on CPU...")
            self.device = torch.device("cpu")
            self.model = LlamaForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="single_label_classification",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_cache=self.use_cache
            )
        
        # Move model to device
        self.model = move_to_device(self.model, self.device)
        
        # Configure for overfitting (no dropout)
        self.model.config.dropout = 0.0
        self.model.config.attention_dropout = 0.0
        self.model.config.hidden_dropout_prob = 0.0
        
        # Enable gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for memory efficiency")
        
        # Training configuration (adaptive based on model size)
        if self.is_small_model:
            # More aggressive training for small models
            self.learning_rate = config.get('learning_rate', 1e-3)  # Higher LR for small models
            self.batch_size = config.get('batch_size', 8)  # Larger batch size
            self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)  # Less accumulation
            self.num_epochs = config.get('num_epochs', 3)  # More epochs
        else:
            # Conservative training for large models
            self.learning_rate = config.get('learning_rate', 1e-4)  # Lower LR for stability
            self.batch_size = config.get('batch_size', 2)  # Small batch size
            self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 8)  # More accumulation
            self.num_epochs = config.get('num_epochs', 1)  # Fewer epochs due to size
        
        # Optimizer (will be initialized in fit)
        self.optimizer = None
        
        logger.info(f"Llama model initialized: {self.model_name}")
        logger.info(f"Model size: {'Small (1B-3B)' if self.is_small_model else 'Large (7B+)'}")
        logger.info(f"Tokenizer: {self.tokenizer_name}")
        logger.info(f"Max length: {self.max_length}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Gradient checkpointing: {self.gradient_checkpointing}")
        logger.info(f"Use cache: {self.use_cache}")
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts for Llama input.
        
        Llama uses a specific tokenization scheme with special tokens.
        
        Args:
            texts: List of text strings to tokenize
            
        Returns:
            Dictionary containing tokenized inputs (input_ids, attention_mask)
        """
        # Tokenize with padding and truncation
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            # Llama doesn't use token_type_ids
            return_token_type_ids=False
        )
        
        # Move to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        return tokenized
    
    def _handle_oom_error(self, current_batch_size: int) -> int:
        """
        Handle out-of-memory errors by reducing batch size.
        
        Args:
            current_batch_size: Current batch size that caused OOM
            
        Returns:
            New batch size to try
        """
        if current_batch_size > self.min_batch_size:
            new_batch_size = max(self.min_batch_size, current_batch_size // 2)
            logger.warning(f"OOM error, reducing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
        else:
            raise RuntimeError("Cannot reduce batch size further. Model too large for available memory.")
    
    def _clear_memory(self):
        """Clear GPU memory cache."""
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def fit(self, 
            X_train: Union[np.ndarray, List[str]], 
            y_train: np.ndarray,
            X_val: Optional[Union[np.ndarray, List[str]]] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train the Llama model with memory optimization.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Optional validation text data
            y_val: Optional validation labels
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Training Llama model on {len(X_train)} samples...")
        logger.info("Note: Llama training may take longer due to model size and memory constraints")
        
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
        
        # Training loop with memory management
        num_epochs = self.num_epochs
        current_batch_size = self.batch_size
        
        for epoch in range(num_epochs):
            # Clear memory before each epoch
            self._clear_memory()
            
            try:
                # Training
                train_loss, train_acc = self._train_epoch(X_train, y_train, current_batch_size)
                
                # Validation
                val_loss, val_acc = None, None
                if X_val is not None and y_val is not None:
                    val_loss, val_acc = self._validate_epoch(X_val, y_val, current_batch_size)
                
                # Store metrics
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_accuracy'].append(train_acc)
                
                if val_loss is not None:
                    self.training_history['val_loss'].append(val_loss)
                    self.training_history['val_accuracy'].append(val_acc)
                
                # Log progress
                log_msg = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(log_msg)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    if self.batch_size_auto_adjust:
                        current_batch_size = self._handle_oom_error(current_batch_size)
                        self.batch_size = current_batch_size
                        logger.info(f"Retrying with batch size: {current_batch_size}")
                        continue
                    else:
                        raise
                else:
                    raise
        
        self.is_trained = True
        logger.info("Llama training completed!")
        
        return self.training_history
    
    def _train_epoch(self, X_train: List[str], y_train: np.ndarray, batch_size: int) -> Tuple[float, float]:
        """Train for one epoch with memory management."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Process in batches
        for i in range(0, len(X_train), batch_size):
            try:
                batch_texts = X_train[i:i + batch_size]
                batch_labels = y_train[i:i + batch_size]
                
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
                if (i // batch_size + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Track metrics
                total_loss += loss.item() * self.gradient_accumulation_steps
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += len(batch_labels)
                
                # Clear memory periodically
                if i % (batch_size * 4) == 0:
                    self._clear_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    self._clear_memory()
                    raise
                else:
                    raise
        
        # Final optimizer step if needed
        if len(X_train) % (batch_size * self.gradient_accumulation_steps) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(X_train) * batch_size
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, X_val: List[str], y_val: np.ndarray, batch_size: int) -> Tuple[float, float]:
        """Validate for one epoch with memory management."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                try:
                    batch_texts = X_val[i:i + batch_size]
                    batch_labels = y_val[i:i + batch_size]
                    
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
                    
                    # Clear memory periodically
                    if i % (batch_size * 4) == 0:
                        self._clear_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                        self._clear_memory()
                        raise
                    else:
                        raise
        
        avg_loss = total_loss / (len(X_val) // batch_size + 1)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Make predictions on new data with memory management.
        
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
                try:
                    batch_texts = X[i:i + self.batch_size]
                    tokenized = self.tokenize_texts(batch_texts)
                    
                    outputs = self.model(**tokenized)
                    batch_predictions = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(batch_predictions.cpu().numpy())
                    
                    # Clear memory periodically
                    if i % (self.batch_size * 4) == 0:
                        self._clear_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                        self._clear_memory()
                        # Try with smaller batch size
                        smaller_batch_size = max(1, self.batch_size // 2)
                        logger.warning(f"OOM during prediction, retrying with batch size {smaller_batch_size}")
                        for j in range(i, min(i + self.batch_size, len(X)), smaller_batch_size):
                            batch_texts = X[j:j + smaller_batch_size]
                            tokenized = self.tokenize_texts(batch_texts)
                            outputs = self.model(**tokenized)
                            batch_predictions = torch.argmax(outputs.logits, dim=-1)
                            predictions.extend(batch_predictions.cpu().numpy())
                    else:
                        raise
        
        return np.array(predictions)
    
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Predict class probabilities with memory management.
        
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
                try:
                    batch_texts = X[i:i + self.batch_size]
                    tokenized = self.tokenize_texts(batch_texts)
                    
                    outputs = self.model(**tokenized)
                    batch_probs = torch.softmax(outputs.logits, dim=-1)
                    probabilities.extend(batch_probs.cpu().numpy())
                    
                    # Clear memory periodically
                    if i % (self.batch_size * 4) == 0:
                        self._clear_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                        self._clear_memory()
                        # Try with smaller batch size
                        smaller_batch_size = max(1, self.batch_size // 2)
                        logger.warning(f"OOM during prediction, retrying with batch size {smaller_batch_size}")
                        for j in range(i, min(i + self.batch_size, len(X)), smaller_batch_size):
                            batch_texts = X[j:j + smaller_batch_size]
                            tokenized = self.tokenize_texts(batch_texts)
                            outputs = self.model(**tokenized)
                            batch_probs = torch.softmax(outputs.logits, dim=-1)
                            probabilities.extend(batch_probs.cpu().numpy())
                    else:
                        raise
        
        return np.array(probabilities)
    
    def get_feature_importance(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get feature importance scores for interpretability.
        
        For Llama, this returns attention weights averaged across layers.
        
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
            'device': str(self.device),
            'gradient_checkpointing': self.gradient_checkpointing,
            'use_cache': self.use_cache
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
        self.gradient_checkpointing = state.get('gradient_checkpointing', True)
        self.use_cache = state.get('use_cache', False)
        
        # Move model to correct device
        self.model = move_to_device(self.model, self.device)


# Model factory for transformer models
def create_transformer_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    """
    Factory function to create transformer model instances.
    
    This function automatically detects the model type and creates the appropriate
    model instance. It supports BERT, RoBERTa, and various Llama models.
    
    Args:
        model_type: Type of transformer model to create
        config: Model configuration
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If the model type is not supported
    """
    logger = logging.getLogger(__name__)
    
    # Direct mapping for known models
    if model_type in TRANSFORMER_MODELS:
        logger.info(f"Creating {model_type} model using direct mapping")
        return TRANSFORMER_MODELS[model_type](config)
    
    # Auto-detection based on model name patterns
    model_type_lower = model_type.lower()
    
    if 'bert' in model_type_lower:
        logger.info(f"Auto-detected BERT model: {model_type}")
        return BERTModel(config)
    elif 'roberta' in model_type_lower:
        logger.info(f"Auto-detected RoBERTa model: {model_type}")
        return RoBERTaModel(config)
    elif 'llama' in model_type_lower:
        logger.info(f"Auto-detected Llama model: {model_type}")
        return LlamaModel(config)
    else:
        # Try to find a close match
        available_models = list(TRANSFORMER_MODELS.keys())
        logger.error(f"Unknown transformer model type: {model_type}")
        logger.error(f"Available models: {available_models}")
        raise ValueError(f"Unknown transformer model type: {model_type}. "
                        f"Available models: {available_models}")


def is_transformer_model(model_type: str) -> bool:
    """
    Check if a model type is a transformer model.
    
    Args:
        model_type: Model type to check
        
    Returns:
        True if it's a transformer model, False otherwise
    """
    # Direct check in known models
    if model_type in TRANSFORMER_MODELS:
        return True
    
    # Pattern-based detection
    model_type_lower = model_type.lower()
    return any(pattern in model_type_lower for pattern in ['bert', 'roberta', 'llama', 'distilbert'])


# Available transformer models
TRANSFORMER_MODELS = {
    'bert-base-uncased': BERTModel,
    'roberta-base': RoBERTaModel,
    'distilbert-base-uncased': BERTModel,  # DistilBERT uses BERT architecture
    'meta-llama/Llama-2-7b-hf': LlamaModel,
    'meta-llama/Llama-2-1b-hf': LlamaModel,
    'meta-llama/Llama-2-3b-hf': LlamaModel,
    'meta-llama/Llama-3.2-1B': LlamaModel,
}
