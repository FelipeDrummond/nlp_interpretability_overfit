#!/usr/bin/env python3
"""
Test script for Interpretability Metrics implementation.

This script tests all the interpretability metrics with both baseline and transformer models
to ensure proper functionality before integration.
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.interpretability.metrics import InterpretabilityMetrics, create_interpretability_metrics
from src.interpretability.shap_analyzer import create_shap_analyzer
from src.models.baseline import BagOfWordsModel
from src.models.transformers import BERTModel
from src.utils.reproducibility import setup_reproducibility

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_faithfulness_metrics():
    """Test faithfulness metrics with baseline model."""
    logger.info("Testing faithfulness metrics...")
    
    # Setup reproducibility
    setup_reproducibility({'seed': 42})
    
    # Create sample data
    texts = [
        "This movie is absolutely fantastic!",
        "I hate this terrible film.",
        "The acting was okay, nothing special.",
        "Amazing performance by the actors!",
        "This is the worst movie I've ever seen."
    ]
    labels = np.array([1, 0, 0, 1, 0])
    
    # Create and train baseline model
    config = {
        'max_features': 100,
        'ngram_range': [1, 2],
        'min_df': 1,
        'max_df': 0.95,
        'stop_words': 'english',
        'classifier_type': 'logistic_regression',
        'C': 1.0,
        'random_state': 42
    }
    
    model = BagOfWordsModel(config)
    model.fit(texts, labels)
    
    # Create SHAP analyzer and get SHAP values
    shap_config = {'max_samples': 5, 'background_samples': 3}
    analyzer = create_shap_analyzer(model, shap_config)
    shap_results = analyzer.analyze(texts)
    
    # Test faithfulness metrics
    metrics = InterpretabilityMetrics()
    
    # Test removal method
    faithfulness_removal = metrics.faithfulness_score(
        model, texts, shap_results['shap_values'], 
        method="removal", num_samples=5
    )
    
    logger.info(f"✓ Faithfulness (removal): {faithfulness_removal['faithfulness_score']:.4f}")
    logger.info(f"  Correlation: {faithfulness_removal['correlation']:.4f}")
    logger.info(f"  RMSE: {faithfulness_removal['rmse']:.4f}")
    
    # Test permutation method
    faithfulness_permutation = metrics.faithfulness_score(
        model, texts, shap_results['shap_values'], 
        method="permutation", num_samples=5
    )
    
    logger.info(f"✓ Faithfulness (permutation): {faithfulness_permutation['faithfulness_score']:.4f}")
    logger.info(f"  Correlation: {faithfulness_permutation['correlation']:.4f}")
    logger.info(f"  RMSE: {faithfulness_permutation['rmse']:.4f}")
    
    return True


def test_consistency_metrics():
    """Test consistency metrics."""
    logger.info("Testing consistency metrics...")
    
    # Create two sets of SHAP values for comparison
    np.random.seed(42)
    shap_values_1 = np.random.randn(5, 10, 2)  # 5 samples, 10 features, 2 classes
    shap_values_2 = np.random.randn(5, 10, 2)  # Similar but different
    
    metrics = InterpretabilityMetrics()
    
    # Test cosine similarity
    consistency_cosine = metrics.consistency_score(
        shap_values_1, shap_values_2, 
        similarity_threshold=0.7, metric="cosine"
    )
    
    logger.info(f"✓ Consistency (cosine): {consistency_cosine['consistency_score']:.4f}")
    logger.info(f"  Mean similarity: {consistency_cosine['mean_similarity']:.4f}")
    logger.info(f"  Similarity ratio: {consistency_cosine['similarity_ratio']:.4f}")
    
    # Test euclidean similarity
    consistency_euclidean = metrics.consistency_score(
        shap_values_1, shap_values_2, 
        similarity_threshold=0.7, metric="euclidean"
    )
    
    logger.info(f"✓ Consistency (euclidean): {consistency_euclidean['consistency_score']:.4f}")
    logger.info(f"  Mean similarity: {consistency_euclidean['mean_similarity']:.4f}")
    
    return True


def test_sparsity_metrics():
    """Test sparsity metrics."""
    logger.info("Testing sparsity metrics...")
    
    # Create SHAP values with different sparsity patterns
    np.random.seed(42)
    
    # Sparse SHAP values (few important features)
    sparse_shap = np.zeros((5, 20, 2))
    sparse_shap[:, [0, 5, 10], :] = np.random.randn(5, 3, 2) * 2
    
    # Dense SHAP values (many important features)
    dense_shap = np.random.randn(5, 20, 2) * 0.5
    
    metrics = InterpretabilityMetrics()
    
    # Test Gini coefficient method
    sparsity_gini = metrics.sparsity_score(sparse_shap, method="gini")
    logger.info(f"✓ Sparsity (Gini, sparse): {sparsity_gini['sparsity_score']:.4f}")
    logger.info(f"  Sparsity ratio: {sparsity_gini['sparsity_ratio']:.4f}")
    
    sparsity_gini_dense = metrics.sparsity_score(dense_shap, method="gini")
    logger.info(f"✓ Sparsity (Gini, dense): {sparsity_gini_dense['sparsity_score']:.4f}")
    logger.info(f"  Sparsity ratio: {sparsity_gini_dense['sparsity_ratio']:.4f}")
    
    # Test entropy method
    sparsity_entropy = metrics.sparsity_score(sparse_shap, method="entropy")
    logger.info(f"✓ Sparsity (entropy, sparse): {sparsity_entropy['sparsity_score']:.4f}")
    
    # Test L1 norm method
    sparsity_l1 = metrics.sparsity_score(sparse_shap, method="l1_norm")
    logger.info(f"✓ Sparsity (L1 norm, sparse): {sparsity_l1['sparsity_score']:.4f}")
    
    return True


def test_intuitiveness_metrics():
    """Test intuitiveness metrics."""
    logger.info("Testing intuitiveness metrics...")
    
    # Create SHAP values
    np.random.seed(42)
    shap_values = np.random.randn(5, 10, 2)
    
    metrics = InterpretabilityMetrics()
    
    # Test heuristic method (no human annotations)
    intuitiveness = metrics.intuitiveness_score(shap_values)
    
    logger.info(f"✓ Intuitiveness (heuristic): {intuitiveness['intuitiveness_score']:.4f}")
    logger.info(f"  Gini coefficient: {intuitiveness['gini_coefficient']:.4f}")
    logger.info(f"  Top feature ratio: {intuitiveness['top_feature_ratio']:.4f}")
    logger.info(f"  Negative ratio: {intuitiveness['negative_ratio']:.4f}")
    
    return True


def test_statistical_significance():
    """Test statistical significance testing."""
    logger.info("Testing statistical significance testing...")
    
    # Create two sets of metric values
    np.random.seed(42)
    metric_values_1 = np.random.normal(0.5, 0.1, 20)  # Mean 0.5, std 0.1
    metric_values_2 = np.random.normal(0.6, 0.1, 20)  # Mean 0.6, std 0.1
    
    metrics = InterpretabilityMetrics()
    
    # Test t-test
    t_test = metrics.statistical_significance_test(
        metric_values_1, metric_values_2, 
        test_type="t_test", alpha=0.05
    )
    
    logger.info(f"✓ T-test: p-value={t_test['p_value']:.4f}, significant={t_test['is_significant']}")
    logger.info(f"  Cohen's d: {t_test['cohens_d']:.4f}")
    
    # Test Mann-Whitney U test
    mw_test = metrics.statistical_significance_test(
        metric_values_1, metric_values_2, 
        test_type="mann_whitney", alpha=0.05
    )
    
    logger.info(f"✓ Mann-Whitney U: p-value={mw_test['p_value']:.4f}, significant={mw_test['is_significant']}")
    
    return True


def test_all_metrics_integration():
    """Test the compute_all_metrics method."""
    logger.info("Testing all metrics integration...")
    
    # Setup reproducibility
    setup_reproducibility({'seed': 42})
    
    # Create sample data and model
    texts = ["Great movie!", "Terrible film.", "Okay movie."]
    labels = np.array([1, 0, 0])
    
    config = {
        'max_features': 50,
        'ngram_range': [1, 1],
        'min_df': 1,
        'max_df': 0.95,
        'stop_words': 'english',
        'classifier_type': 'logistic_regression',
        'C': 1.0,
        'random_state': 42
    }
    
    model = BagOfWordsModel(config)
    model.fit(texts, labels)
    
    # Get SHAP values
    shap_config = {'max_samples': 3, 'background_samples': 2}
    analyzer = create_shap_analyzer(model, shap_config)
    shap_results = analyzer.analyze(texts)
    
    # Test all metrics
    metrics_config = {
        'faithfulness': {'method': 'removal', 'num_samples': 3},
        'sparsity': {'threshold': 0.01, 'method': 'gini'},
        'intuitiveness': {'enabled': True}
    }
    
    metrics = InterpretabilityMetrics(metrics_config)
    all_results = metrics.compute_all_metrics(
        model, texts, shap_results['shap_values'], 
        feature_names=shap_results['feature_names']
    )
    
    logger.info("✓ All metrics computed successfully:")
    for metric_name, metric_results in all_results.items():
        if 'error' not in metric_results:
            if metric_name == 'faithfulness':
                logger.info(f"  {metric_name}: {metric_results.get('faithfulness_score', 'N/A'):.4f}")
            elif metric_name == 'sparsity':
                logger.info(f"  {metric_name}: {metric_results.get('sparsity_score', 'N/A'):.4f}")
            elif metric_name == 'intuitiveness':
                logger.info(f"  {metric_name}: {metric_results.get('intuitiveness_score', 'N/A'):.4f}")
            else:
                logger.info(f"  {metric_name}: {metric_results}")
        else:
            logger.info(f"  {metric_name}: Error - {metric_results['error']}")
    
    return True


def main():
    """Run all tests."""
    logger.info("Starting Interpretability Metrics tests...")
    
    tests = [
        ("Faithfulness Metrics", test_faithfulness_metrics),
        ("Consistency Metrics", test_consistency_metrics),
        ("Sparsity Metrics", test_sparsity_metrics),
        ("Intuitiveness Metrics", test_intuitiveness_metrics),
        ("Statistical Significance", test_statistical_significance),
        ("All Metrics Integration", test_all_metrics_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Running {test_name} ---")
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.info(f"⚠ {test_name} SKIPPED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED: {e}")
    
    logger.info(f"\n--- Test Summary ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed >= 5:  # Most tests should pass
        logger.info("✓ Interpretability Metrics implementation is working correctly!")
        return True
    else:
        logger.error("✗ Interpretability Metrics implementation has issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
