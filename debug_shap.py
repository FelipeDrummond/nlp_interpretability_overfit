#!/usr/bin/env python3
"""
Debug script for SHAP Analyzer issues.
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.interpretability.shap_analyzer import SHAPAnalyzer, create_shap_analyzer
from src.models.baseline import BagOfWordsModel
from src.utils.reproducibility import setup_reproducibility

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_shap_issue():
    """Debug the SHAP issue step by step."""
    logger.info("Starting SHAP debug...")
    
    # Setup reproducibility
    setup_reproducibility({'seed': 42})
    
    # Create minimal sample data
    texts = ["Great movie!", "Terrible film."]
    labels = np.array([1, 0])
    
    # Create and train baseline model
    config = {
        'max_features': 10,  # Very small for debugging
        'ngram_range': [1, 1],
        'min_df': 1,
        'max_df': 0.95,
        'stop_words': 'english',
        'classifier_type': 'logistic_regression',
        'C': 1.0,
        'random_state': 42
    }
    
    logger.info("Creating and training model...")
    model = BagOfWordsModel(config)
    model.fit(texts, labels)
    
    logger.info("Model trained successfully")
    logger.info(f"Model has vectorizer: {hasattr(model, 'vectorizer')}")
    logger.info(f"Model has classifier: {hasattr(model, 'classifier')}")
    
    # Test model prediction
    logger.info("Testing model prediction...")
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    logger.info(f"Predictions: {predictions}")
    logger.info(f"Probabilities shape: {probabilities.shape}")
    
    # Create SHAP analyzer
    logger.info("Creating SHAP analyzer...")
    shap_config = {'max_samples': 2, 'background_samples': 1}
    analyzer = create_shap_analyzer(model, shap_config)
    
    # Test explainer creation
    logger.info("Testing explainer creation...")
    try:
        # Use the analyzer's method to prepare background data
        X_background = analyzer._prepare_background_data(texts)
        logger.info(f"Background data shape: {X_background.shape}")
        explainer = analyzer._get_explainer(X_background)
        logger.info(f"Explainer created: {type(explainer)}")
    except Exception as e:
        logger.error(f"Explainer creation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    # Test SHAP values computation
    logger.info("Testing SHAP values computation...")
    try:
        # Vectorize the input texts for SHAP
        if hasattr(model, 'vectorizer'):
            X_for_shap = model.vectorizer.transform(texts).toarray()
            logger.info(f"Vectorized input data to shape: {X_for_shap.shape}")
        else:
            X_for_shap = np.array(texts)
        
        shap_values = explainer.shap_values(X_for_shap)
        logger.info(f"SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            logger.info(f"SHAP values list length: {len(shap_values)}")
            for i, val in enumerate(shap_values):
                logger.info(f"  Element {i}: type={type(val)}, shape={val.shape if hasattr(val, 'shape') else 'no shape'}")
        else:
            logger.info(f"SHAP values shape: {shap_values.shape}")
    except Exception as e:
        logger.error(f"SHAP values computation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    logger.info("✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = debug_shap_issue()
    sys.exit(0 if success else 1)
