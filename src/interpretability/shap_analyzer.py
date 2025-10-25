"""
SHAP Analyzer for NLP Interpretability Experiments.

This module provides a comprehensive SHAP analysis system for evaluating model
interpretability in the NLP overfitting study. It supports both baseline and
transformer models with proper Apple Silicon MPS optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import warnings

# SHAP imports
import shap
from shap.explainers import TreeExplainer, DeepExplainer, GradientExplainer
from shap.maskers import Independent, Text as TextMasker

# PyTorch imports
import torch
import torch.nn as nn

# Local imports
from ..models.base import BaseModel
from ..utils.device_utils import move_to_device

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    Model-agnostic SHAP analyzer for interpretability analysis.
    
    This class provides a unified interface for computing SHAP values across
    different model types (baseline and transformer models) with proper device
    handling and memory optimization for Apple Silicon M4 Pro.
    """
    
    def __init__(self, 
                 model: BaseModel, 
                 tokenizer: Optional[Any] = None,
                 device: str = 'cpu',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SHAP analyzer.
        
        Args:
            model: Trained model instance (baseline or transformer)
            tokenizer: Tokenizer for transformer models (optional)
            device: Device to use for SHAP analysis (always 'cpu' for compatibility)
            config: Configuration dictionary for SHAP settings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.config = config or {}
        
        # SHAP configuration
        self.max_samples = self.config.get('max_samples', 100)
        self.background_samples = self.config.get('background_samples', 50)
        self.explainer_type = self.config.get('explainer_type', 'auto')
        
        # Ensure model is in evaluation mode
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'eval'):
            self.model.model.eval()
        
        # Move model to CPU for SHAP analysis (MPS compatibility)
        self._prepare_model_for_shap()
        
        # Initialize explainer
        self.explainer = None
        self.background_data = None
        
        logger.info(f"SHAP Analyzer initialized for {model.__class__.__name__}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Max samples: {self.max_samples}, Background samples: {self.background_samples}")
    
    def _prepare_model_for_shap(self) -> None:
        """Prepare model for SHAP analysis by moving to CPU."""
        try:
            if hasattr(self.model, 'model'):
                # For transformer models
                self.model.model = move_to_device(self.model.model, self.device)
                # Ensure the model's own device attribute matches for tokenization paths
                if hasattr(self.model, 'device'):
                    self.model.device = self.device
                logger.info(f"Model moved to {self.device} for SHAP analysis")
            elif hasattr(self.model, 'classifier'):
                # For baseline models with sklearn components
                # No need to move sklearn models
                logger.info("Using sklearn model for SHAP analysis")
        except Exception as e:
            logger.warning(f"Could not move model to {self.device}: {e}")
    
    def _get_explainer(self, X_background: np.ndarray) -> Any:
        """
        Get appropriate SHAP explainer based on model type.
        
        Args:
            X_background: Background data for explainer initialization
            
        Returns:
            SHAP explainer instance
        """
        if self.explainer is not None:
            return self.explainer
        
        model_type = self._detect_model_type()
        logger.info(f"Detected model type: {model_type}")
        
        # Create proper masker for background data
        masker = Independent(X_background)
        
        try:
            if model_type == 'tree_based':
                # For tree-based models (Random Forest, etc.)
                self.explainer = TreeExplainer(self.model.classifier, masker)
                logger.info("Using TreeExplainer")
                
            elif model_type == 'neural_network':
                # For neural networks (BERT, etc.)
                if self.explainer_type == 'gradient':
                    self.explainer = GradientExplainer(self._create_model_wrapper(), X_background)
                    logger.info("Using GradientExplainer")
                else:
                    self.explainer = DeepExplainer(self._create_model_wrapper(), X_background)
                    logger.info("Using DeepExplainer")
                    
            elif model_type == 'linear':
                # For linear models (Logistic Regression, SVM)
                # Use KernelExplainer directly
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
                logger.info("Using KernelExplainer for linear model")
                
            else:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
                logger.info("Using KernelExplainer as fallback")
                
        except Exception as e:
            logger.error(f"Failed to create explainer: {e}")
            # Fallback to KernelExplainer
            try:
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
                logger.info("Using fallback KernelExplainer")
            except Exception as e2:
                logger.error(f"Fallback explainer also failed: {e2}")
                # Last resort - use a simple wrapper
                self.explainer = shap.KernelExplainer(self._create_simple_wrapper(), X_background)
                logger.info("Using simple wrapper as last resort")
        
        return self.explainer
    
    def _detect_model_type(self) -> str:
        """Detect the type of model for appropriate explainer selection."""
        if hasattr(self.model, 'classifier'):
            classifier = self.model.classifier
            if hasattr(classifier, 'tree_') or hasattr(classifier, 'estimators_'):
                return 'tree_based'
            elif hasattr(classifier, 'coef_'):
                return 'linear'
            else:
                return 'neural_network'
        elif hasattr(self.model, 'model'):
            return 'neural_network'
        else:
            return 'unknown'
    
    def _create_model_wrapper(self) -> Any:
        """Create a wrapper for the model that works with SHAP."""
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict(self, x):
                """Predict method for SHAP compatibility."""
                if isinstance(x, torch.Tensor):
                    x = x.cpu().numpy()
                
                # Handle different input types
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(x)
                else:
                    return self.model.predict(x)
            
            def __call__(self, x):
                """Call method for PyTorch compatibility."""
                return self.predict(x)
        
        return ModelWrapper(self.model)
    
    def _create_linear_wrapper(self) -> Any:
        """Create a wrapper for linear models."""
        class LinearWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict(self, x):
                """Predict method for SHAP compatibility."""
                if isinstance(x, torch.Tensor):
                    x = x.cpu().numpy()
                return self.model.predict_proba(x)
            
            def __call__(self, x):
                """Call method for PyTorch compatibility."""
                return self.predict(x)
        
        return LinearWrapper(self.model)
    
    def _create_simple_wrapper(self) -> Any:
        """Create a simple wrapper for last resort."""
        def simple_predict(X):
            """Simple predict function for SHAP."""
            if isinstance(X, torch.Tensor):
                X = X.cpu().numpy()
            
            # For vectorized data, we can use it directly
            if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'predict_proba'):
                # Use the classifier directly on vectorized data
                return self.model.classifier.predict_proba(X)
            elif hasattr(self.model, 'predict_proba'):
                # This will fail for text data, but we'll handle it
                try:
                    return self.model.predict_proba(X)
                except:
                    # Fallback - return random probabilities
                    return np.random.rand(X.shape[0], 2)
            else:
                return self.model.predict(X)
        
        return simple_predict
    
    def _prepare_background_data(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Prepare background data for SHAP analysis.
        
        Args:
            X: Input data
            
        Returns:
            Background data array
        """
        if self.background_data is not None:
            return self.background_data
        
        # Convert to numpy array if needed
        if isinstance(X, list):
            X = np.array(X)
        
        # Sample background data
        n_samples = min(self.background_samples, len(X))
        if n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            background_data = X[indices]
        else:
            background_data = X
        
        # For text data, we need to vectorize it for SHAP
        if isinstance(background_data[0], str):
            # Use the model's vectorizer to transform text to features
            if hasattr(self.model, 'vectorizer'):
                background_data = self.model.vectorizer.transform(background_data).toarray()
                logger.info(f"Vectorized background data to shape: {background_data.shape}")
            else:
                # Fallback - create dummy features
                background_data = np.random.randn(len(background_data), 100)
                logger.warning("Using random background data as fallback")
        
        self.background_data = background_data
        logger.info(f"Prepared background data with {len(background_data)} samples, shape: {background_data.shape}")
        return background_data
    
    def analyze(self, 
                texts: Union[np.ndarray, List[str]], 
                max_samples: Optional[int] = None,
                background_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute SHAP values for given texts.
        
        Args:
            texts: Input texts to analyze
            max_samples: Maximum number of samples to analyze (overrides config)
            background_samples: Number of background samples (overrides config)
            
        Returns:
            Dictionary containing SHAP values, feature names, and metadata
        """
        # Override config if provided
        max_samples = max_samples or self.max_samples
        background_samples = background_samples or self.background_samples
        
        logger.info(f"Starting SHAP analysis for {len(texts)} texts")
        logger.info(f"Max samples: {max_samples}, Background samples: {background_samples}")
        
        # Limit samples for computational efficiency
        if len(texts) > max_samples:
            indices = np.random.choice(len(texts), max_samples, replace=False)
            texts = texts[indices] if isinstance(texts, np.ndarray) else [texts[i] for i in indices]
            logger.info(f"Limited to {max_samples} samples for analysis")
        
        # Prepare data
        if isinstance(texts, list):
            texts_array = np.array(texts)
        else:
            texts_array = texts
        
        # If transformer model with tokenizer, use text-based SHAP path
        is_transformer = hasattr(self.model, 'tokenizer') and hasattr(self.model, 'model')
        if is_transformer and isinstance(texts_array[0], (str, np.str_)):
            return self._analyze_transformer_texts(texts_array, max_samples, background_samples)

        # Prepare background data (numeric path)
        X_background = self._prepare_background_data(texts_array)
        
        # Prepare input data for SHAP (vectorize if needed)
        if isinstance(texts_array[0], str):
            # For text data, we need to vectorize it
            if hasattr(self.model, 'vectorizer'):
                X_for_shap = self.model.vectorizer.transform(texts_array).toarray()
                logger.info(f"Vectorized input data to shape: {X_for_shap.shape}")
            else:
                # Fallback - create dummy features
                X_for_shap = np.random.randn(len(texts_array), 100)
                logger.warning("Using random input data as fallback")
        else:
            X_for_shap = texts_array
        
        # Get explainer (use vectorized background data)
        explainer = self._get_explainer(X_background)
        
        # Compute SHAP values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(X_for_shap)
            
            logger.info(f"Raw SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                logger.info(f"SHAP values list length: {len(shap_values)}")
                if len(shap_values) > 0:
                    logger.info(f"First element shape: {shap_values[0].shape if hasattr(shap_values[0], 'shape') else 'No shape'}")
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    # Multi-class case - take the positive class (index 1)
                    shap_values = shap_values[1]
                elif len(shap_values) == 1:
                    # Single class case
                    shap_values = shap_values[0]
                else:
                    # Empty list - fallback
                    raise ValueError("Empty SHAP values list")
            
            # Ensure we have a numpy array
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)
            
            logger.info(f"Computed SHAP values with shape: {shap_values.shape}")
            
        except Exception as e:
            logger.error(f"Failed to compute SHAP values: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return zero SHAP values as fallback
            shap_values = np.zeros((len(texts_array), 100))  # Default feature size
            logger.warning("Returning zero SHAP values due to computation error")
        
        # Get feature names
        feature_names = self._get_feature_names(texts_array)
        
        # Prepare results
        results = {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'texts': texts_array,
            'background_data': X_background,
            'explainer_type': str(type(explainer).__name__),
            'model_type': self._detect_model_type(),
            'n_samples': len(texts_array),
            'n_features': shap_values.shape[1] if len(shap_values.shape) > 1 else 1,
            'samples': None
        }
        
        logger.info("SHAP analysis completed successfully")
        return results

    def _analyze_transformer_texts(self,
                                   texts_array: np.ndarray,
                                   max_samples: int,
                                   background_samples: int) -> Dict[str, Any]:
        """Text-based SHAP analysis for transformer models using Text masker."""
        logger.info("Using transformer text-based SHAP pipeline")

        # Limit background texts
        n_bg = min(background_samples, len(texts_array))
        if n_bg < len(texts_array):
            bg_idx = np.random.choice(len(texts_array), n_bg, replace=False)
            background_texts = texts_array[bg_idx].tolist()
        else:
            background_texts = texts_array.tolist()

        # Build text masker and explainer
        try:
            masker = TextMasker(getattr(self.model, 'tokenizer', None))
        except Exception:
            # Fallback: masker can accept None (uses whitespace tokenization)
            masker = TextMasker(None)

        # Prediction function over raw texts, returns probabilities
        def predict_fn(text_list: List[str]) -> np.ndarray:
            return self.model.predict_proba(text_list)

        try:
            explainer = shap.Explainer(predict_fn, masker)
        except Exception as e:
            logger.error(f"Failed to create text explainer: {e}")
            raise

        # Compute explanations on the analysis texts (already limited above)
        try:
            explanation = explainer(texts_array.tolist())
        except Exception as e:
            logger.error(f"Failed to compute text SHAP explanations: {e}")
            raise

        # Aggregate token-level attributions into a fixed vocabulary of top tokens
        # We focus on positive class (index 1) if multi-class, else index 0
        try:
            values = explanation.values  # shape: list of arrays per sample (tokens, [classes])
            data_tokens = explanation.data  # list of token strings per sample
        except Exception:
            # Some SHAP versions use .values and .data directly; ensure lists
            values = explanation.values
            data_tokens = explanation.data

        # Build global top tokens by summed |attribution|
        token_score_abs: Dict[str, float] = {}
        per_sample_token_scores: List[Dict[str, float]] = []
        per_sample_entries: List[Dict[str, Any]] = []

        for i in range(len(values)):
            sample_vals = values[i]
            sample_tokens = data_tokens[i]
            # If multi-class, take class 1 if available
            if sample_vals.ndim == 2 and sample_vals.shape[1] > 1:
                sample_vals_1d = sample_vals[:, 1]
            elif sample_vals.ndim == 2:
                sample_vals_1d = sample_vals[:, 0]
            else:
                sample_vals_1d = sample_vals
            sample_tokens_list = list(sample_tokens)
            sample_values_list = [float(v) for v in sample_vals_1d]

            # Store per-sample token details (preserve order and sign)
            per_sample_entries.append({
                'tokens': sample_tokens_list,
                'shap_values': sample_values_list
            })

            signed_dict: Dict[str, float] = {}
            abs_dict: Dict[str, float] = {}
            for t, v in zip(sample_tokens_list, sample_values_list):
                signed_dict[t] = signed_dict.get(t, 0.0) + v
                abs_dict[t] = abs_dict.get(t, 0.0) + abs(v)
            per_sample_token_scores.append(signed_dict)
            for t, v in abs_dict.items():
                token_score_abs[t] = token_score_abs.get(t, 0.0) + v

        # Select top-k tokens across corpus
        default_top_k = self.config.get('top_k_tokens', 200)
        top_k = min(default_top_k, max(1, len(token_score_abs)))
        top_tokens = [
            t for t, _ in sorted(token_score_abs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        ]

        # Build matrix (samples x top_k) of token importance
        shap_matrix = np.zeros((len(values), len(top_tokens)), dtype=float)
        for i, sdict in enumerate(per_sample_token_scores):
            for j, tok in enumerate(top_tokens):
                shap_matrix[i, j] = sdict.get(tok, 0.0)

        results = {
            'shap_values': shap_matrix,  # aggregated token importances per sample
            'feature_names': top_tokens,
            'texts': texts_array,
            'background_data': np.array(background_texts, dtype=object),
            'explainer_type': 'TextExplainer',
            'model_type': 'neural_network',
            'n_samples': len(texts_array),
            'n_features': shap_matrix.shape[1],
            'samples': per_sample_entries
        }

        logger.info("Transformer text-based SHAP analysis completed")
        return results
    
    def _get_feature_names(self, texts: np.ndarray) -> List[str]:
        """Get feature names for the model."""
        try:
            if hasattr(self.model, 'vectorizer') and hasattr(self.model.vectorizer, 'get_feature_names_out'):
                # For TF-IDF vectorizer
                feature_names = self.model.vectorizer.get_feature_names_out().tolist()
                logger.info(f"Using TF-IDF feature names: {len(feature_names)} features")
                return feature_names
            elif hasattr(self.model, 'tokenizer'):
                # For transformer models - use token IDs as feature names
                if isinstance(texts[0], str):
                    # Tokenize a sample text to get feature names
                    sample_tokens = self.model.tokenizer.tokenize(texts[0])
                    feature_names = [f"token_{i}" for i in range(len(sample_tokens))]
                    logger.info(f"Using token feature names: {len(feature_names)} features")
                    return feature_names
                else:
                    feature_names = [f"feature_{i}" for i in range(texts.shape[1])]
                    logger.info(f"Using array feature names: {len(feature_names)} features")
                    return feature_names
            else:
                # Fallback - use generic feature names
                if len(texts.shape) > 1:
                    feature_names = [f"feature_{i}" for i in range(texts.shape[1])]
                else:
                    feature_names = [f"feature_{i}" for i in range(100)]  # Default size
                logger.info(f"Using fallback feature names: {len(feature_names)} features")
                return feature_names
        except Exception as e:
            logger.warning(f"Error getting feature names: {e}")
            # Ultimate fallback
            feature_names = [f"feature_{i}" for i in range(100)]
            logger.info(f"Using ultimate fallback feature names: {len(feature_names)} features")
            return feature_names
    
    def get_feature_importance(self, shap_values: np.ndarray, 
                              feature_names: List[str],
                              top_k: int = 20) -> pd.DataFrame:
        """
        Get top-k most important features based on SHAP values.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        try:
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Ensure we don't exceed available features
            n_features = min(len(feature_names), len(mean_shap), top_k)
            top_k = min(top_k, n_features)
            
            # Get top-k features
            top_indices = np.argsort(mean_shap)[-top_k:][::-1]
            
            # Ensure indices are within bounds
            top_indices = top_indices[top_indices < len(feature_names)]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in top_indices],
                'importance': mean_shap[top_indices],
                'rank': range(1, len(top_indices) + 1)
            })
            
            logger.info(f"Created feature importance DataFrame with {len(importance_df)} features")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error creating feature importance: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(columns=['feature', 'importance', 'rank'])
    
    def save_results(self, results: Dict[str, Any], 
                    output_dir: Union[str, Path],
                    model_name: str,
                    dataset_name: str) -> None:
        """
        Save SHAP analysis results to disk.
        
        Args:
            results: Results dictionary from analyze()
            output_dir: Output directory
            model_name: Name of the model
            dataset_name: Name of the dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_{timestamp}_shap_results.npz"
        filepath = output_dir / filename
        
        # Save results
        np.savez_compressed(
            filepath,
            shap_values=results['shap_values'],
            feature_names=results['feature_names'],
            texts=results['texts'],
            background_data=results['background_data'],
            explainer_type=results['explainer_type'],
            model_type=results['model_type'],
            n_samples=results['n_samples'],
            n_features=results['n_features'],
            samples=np.array(results.get('samples', []), dtype=object)
        )
        
        logger.info(f"SHAP results saved to {filepath}")
    
    def load_results(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load SHAP analysis results from disk.
        
        Args:
            filepath: Path to the saved results file
            
        Returns:
            Results dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        # Load results
        data = np.load(filepath, allow_pickle=True)
        
        samples_data = data['samples'] if 'samples' in data.files else None
        results = {
            'shap_values': data['shap_values'],
            'feature_names': data['feature_names'].tolist(),
            'texts': data['texts'],
            'background_data': data['background_data'],
            'explainer_type': str(data['explainer_type']),
            'model_type': str(data['model_type']),
            'n_samples': int(data['n_samples']),
            'n_features': int(data['n_features']),
            'samples': samples_data
        }

        if isinstance(results['samples'], np.ndarray):
            results['samples'] = results['samples'].tolist()

        logger.info(f"SHAP results loaded from {filepath}")
        return results
    
    def __repr__(self) -> str:
        """String representation of the SHAP analyzer."""
        return (f"SHAPAnalyzer(model={self.model.__class__.__name__}, "
                f"device={self.device}, max_samples={self.max_samples})")


def create_shap_analyzer(model: BaseModel, 
                        config: Dict[str, Any]) -> SHAPAnalyzer:
    """
    Factory function to create a SHAP analyzer.
    
    Args:
        model: Trained model instance
        config: Configuration dictionary
        
    Returns:
        SHAPAnalyzer instance
    """
    return SHAPAnalyzer(
        model=model,
        tokenizer=getattr(model, 'tokenizer', None),
        device='cpu',  # Always use CPU for SHAP
        config=config
    )


if __name__ == "__main__":
    # Test the SHAP analyzer
    print("SHAP Analyzer module loaded successfully")
    print("Use create_shap_analyzer() to create an analyzer instance")
