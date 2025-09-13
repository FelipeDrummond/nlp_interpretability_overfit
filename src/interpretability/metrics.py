"""
Interpretability Metrics for NLP Models.

This module provides comprehensive interpretability metrics for evaluating
the quality and reliability of model explanations, particularly SHAP values.
It implements faithfulness, consistency, sparsity, and intuitiveness metrics
with statistical significance testing.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union
from scipy import stats
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


class InterpretabilityMetrics:
    """
    Comprehensive interpretability metrics for model explanations.
    
    This class provides methods to evaluate the quality of model explanations
    across multiple dimensions: faithfulness, consistency, sparsity, and intuitiveness.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the interpretability metrics calculator.
        
        Args:
            config: Configuration dictionary for metric parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.faithfulness_config = self.config.get('faithfulness', {})
        self.consistency_config = self.config.get('consistency', {})
        self.sparsity_config = self.config.get('sparsity', {})
        self.intuitiveness_config = self.config.get('intuitiveness', {})
        
        logger.info("InterpretabilityMetrics initialized")
    
    @staticmethod
    def faithfulness_score(model: Any, 
                          inputs: Union[np.ndarray, List[str]], 
                          shap_values: np.ndarray,
                          method: str = "removal",
                          num_samples: int = 50,
                          feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Measure if important features actually impact model predictions.
        
        This metric evaluates whether features identified as important by SHAP
        actually contribute to the model's predictions when removed or modified.
        
        Args:
            model: Trained model instance
            inputs: Input data (text or vectors)
            shap_values: SHAP values for the inputs
            method: Method to use ("removal" or "permutation")
            num_samples: Number of samples to use for evaluation
            feature_names: List of feature names (optional)
            
        Returns:
            Dictionary containing faithfulness metrics
        """
        logger.info(f"Computing faithfulness score using {method} method")
        
        try:
            if method == "removal":
                return InterpretabilityMetrics._faithfulness_removal(
                    model, inputs, shap_values, num_samples, feature_names
                )
            elif method == "permutation":
                return InterpretabilityMetrics._faithfulness_permutation(
                    model, inputs, shap_values, num_samples, feature_names
                )
            else:
                raise ValueError(f"Unknown faithfulness method: {method}")
                
        except Exception as e:
            logger.error(f"Error computing faithfulness score: {e}")
            return {
                'faithfulness_score': 0.0,
                'correlation': 0.0,
                'rmse': float('inf'),
                'method': method,
                'error': str(e)
            }
    
    @staticmethod
    def _faithfulness_removal(model: Any, 
                             inputs: Union[np.ndarray, List[str]], 
                             shap_values: np.ndarray,
                             num_samples: int,
                             feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute faithfulness using feature removal method."""
        # Limit samples for efficiency
        n_samples = min(num_samples, len(inputs))
        if n_samples < len(inputs):
            indices = np.random.choice(len(inputs), n_samples, replace=False)
            inputs_subset = inputs[indices] if isinstance(inputs, np.ndarray) else [inputs[i] for i in indices]
            shap_subset = shap_values[indices]
        else:
            inputs_subset = inputs
            shap_subset = shap_values
        
        # Get original predictions
        original_predictions = model.predict_proba(inputs_subset)
        
        # Calculate feature importance rankings
        feature_importance = np.mean(np.abs(shap_subset), axis=0)
        if len(feature_importance.shape) > 1:
            feature_importance = np.mean(feature_importance, axis=0)
        
        # Sort features by importance
        top_features = np.argsort(feature_importance)[::-1]
        
        # Test removal of top features
        removal_scores = []
        for i in range(min(10, len(top_features))):  # Test top 10 features
            # Create modified inputs with top i features removed
            modified_inputs = InterpretabilityMetrics._remove_features(
                inputs_subset, top_features[:i+1], feature_names
            )
            
            # Get predictions on modified inputs
            modified_predictions = model.predict_proba(modified_inputs)
            
            # Calculate prediction difference
            pred_diff = np.mean(np.abs(original_predictions - modified_predictions))
            removal_scores.append(pred_diff)
        
        # Calculate correlation between feature importance and prediction change
        if len(removal_scores) > 1:
            correlation = np.corrcoef(
                range(len(removal_scores)), 
                removal_scores
            )[0, 1]
        else:
            correlation = 0.0
        
        # Calculate RMSE between expected and actual changes
        expected_changes = np.linspace(0, 1, len(removal_scores))
        rmse = np.sqrt(np.mean((np.array(removal_scores) - expected_changes) ** 2))
        
        faithfulness_score = correlation * (1 - rmse)  # Combine correlation and consistency
        
        return {
            'faithfulness_score': float(faithfulness_score),
            'correlation': float(correlation),
            'rmse': float(rmse),
            'method': 'removal',
            'n_features_tested': len(removal_scores)
        }
    
    @staticmethod
    def _faithfulness_permutation(model: Any, 
                                 inputs: Union[np.ndarray, List[str]], 
                                 shap_values: np.ndarray,
                                 num_samples: int,
                                 feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute faithfulness using feature permutation method."""
        # Limit samples for efficiency
        n_samples = min(num_samples, len(inputs))
        if n_samples < len(inputs):
            indices = np.random.choice(len(inputs), n_samples, replace=False)
            inputs_subset = inputs[indices] if isinstance(inputs, np.ndarray) else [inputs[i] for i in indices]
            shap_subset = shap_values[indices]
        else:
            inputs_subset = inputs
            shap_subset = shap_values
        
        # Get original predictions
        original_predictions = model.predict_proba(inputs_subset)
        
        # Calculate feature importance rankings
        feature_importance = np.mean(np.abs(shap_subset), axis=0)
        if len(feature_importance.shape) > 1:
            feature_importance = np.mean(feature_importance, axis=0)
        
        # Sort features by importance
        top_features = np.argsort(feature_importance)[::-1]
        
        # Test permutation of top features
        permutation_scores = []
        for i in range(min(10, len(top_features))):  # Test top 10 features
            # Create modified inputs with top i features permuted
            modified_inputs = InterpretabilityMetrics._permute_features(
                inputs_subset, top_features[:i+1], feature_names
            )
            
            # Get predictions on modified inputs
            modified_predictions = model.predict_proba(modified_inputs)
            
            # Calculate prediction difference
            pred_diff = np.mean(np.abs(original_predictions - modified_predictions))
            permutation_scores.append(pred_diff)
        
        # Calculate correlation between feature importance and prediction change
        if len(permutation_scores) > 1:
            correlation = np.corrcoef(
                range(len(permutation_scores)), 
                permutation_scores
            )[0, 1]
        else:
            correlation = 0.0
        
        # Calculate RMSE between expected and actual changes
        expected_changes = np.linspace(0, 1, len(permutation_scores))
        rmse = np.sqrt(np.mean((np.array(permutation_scores) - expected_changes) ** 2))
        
        faithfulness_score = correlation * (1 - rmse)  # Combine correlation and consistency
        
        return {
            'faithfulness_score': float(faithfulness_score),
            'correlation': float(correlation),
            'rmse': float(rmse),
            'method': 'permutation',
            'n_features_tested': len(permutation_scores)
        }
    
    @staticmethod
    def _remove_features(inputs: Union[np.ndarray, List[str]], 
                        feature_indices: np.ndarray,
                        feature_names: Optional[List[str]] = None) -> Union[np.ndarray, List[str]]:
        """Remove specified features from inputs."""
        if isinstance(inputs, list):
            # For text inputs, we can't easily remove features
            # Return original inputs as fallback
            return inputs
        else:
            # For vectorized inputs, set features to zero
            modified_inputs = inputs.copy()
            modified_inputs[:, feature_indices] = 0
            return modified_inputs
    
    @staticmethod
    def _permute_features(inputs: Union[np.ndarray, List[str]], 
                         feature_indices: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> Union[np.ndarray, List[str]]:
        """Permute specified features in inputs."""
        if isinstance(inputs, list):
            # For text inputs, we can't easily permute features
            # Return original inputs as fallback
            return inputs
        else:
            # For vectorized inputs, shuffle the specified features
            modified_inputs = inputs.copy()
            for idx in feature_indices:
                modified_inputs[:, idx] = np.random.permutation(modified_inputs[:, idx])
            return modified_inputs
    
    @staticmethod
    def consistency_score(shap_values_1: np.ndarray, 
                         shap_values_2: np.ndarray,
                         similarity_threshold: float = 0.7,
                         metric: str = "cosine") -> Dict[str, float]:
        """
        Measure consistency of interpretations across similar examples.
        
        This metric evaluates how similar the SHAP values are for similar inputs,
        which is important for reliable interpretability.
        
        Args:
            shap_values_1: First set of SHAP values
            shap_values_2: Second set of SHAP values
            similarity_threshold: Threshold for considering examples similar
            metric: Similarity metric to use ("cosine", "euclidean", "manhattan")
            
        Returns:
            Dictionary containing consistency metrics
        """
        logger.info(f"Computing consistency score using {metric} metric")
        
        try:
            # Ensure both arrays have the same number of samples
            min_samples = min(len(shap_values_1), len(shap_values_2))
            shap_1 = shap_values_1[:min_samples]
            shap_2 = shap_values_2[:min_samples]
            
            # Flatten SHAP values if they have multiple dimensions
            if len(shap_1.shape) > 2:
                shap_1 = shap_1.reshape(shap_1.shape[0], -1)
            if len(shap_2.shape) > 2:
                shap_2 = shap_2.reshape(shap_2.shape[0], -1)
            
            # Calculate pairwise similarities
            if metric == "cosine":
                similarities = 1 - pairwise_distances(shap_1, shap_2, metric='cosine')
            elif metric == "euclidean":
                similarities = 1 / (1 + pairwise_distances(shap_1, shap_2, metric='euclidean'))
            elif metric == "manhattan":
                similarities = 1 / (1 + pairwise_distances(shap_1, shap_2, metric='manhattan'))
            else:
                raise ValueError(f"Unknown similarity metric: {metric}")
            
            # Calculate consistency metrics
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            # Calculate percentage of similar pairs
            similar_pairs = np.sum(similarities > similarity_threshold)
            total_pairs = similarities.size
            similarity_ratio = similar_pairs / total_pairs if total_pairs > 0 else 0.0
            
            # Calculate consistency score (combination of mean similarity and ratio)
            consistency_score = mean_similarity * similarity_ratio
            
            return {
                'consistency_score': float(consistency_score),
                'mean_similarity': float(mean_similarity),
                'std_similarity': float(std_similarity),
                'similarity_ratio': float(similarity_ratio),
                'similar_pairs': int(similar_pairs),
                'total_pairs': int(total_pairs),
                'metric': metric
            }
            
        except Exception as e:
            logger.error(f"Error computing consistency score: {e}")
            return {
                'consistency_score': 0.0,
                'mean_similarity': 0.0,
                'std_similarity': 0.0,
                'similarity_ratio': 0.0,
                'similar_pairs': 0,
                'total_pairs': 0,
                'metric': metric,
                'error': str(e)
            }
    
    @staticmethod
    def sparsity_score(shap_values: np.ndarray, 
                      threshold: float = 0.01,
                      method: str = "gini") -> Dict[str, float]:
        """
        Measure concentration of importance scores.
        
        This metric evaluates how concentrated the feature importance is,
        which indicates whether the model relies on a few key features
        or many features equally.
        
        Args:
            shap_values: SHAP values array
            threshold: Threshold for considering features as important
            method: Method to use ("gini", "entropy", "l1_norm")
            
        Returns:
            Dictionary containing sparsity metrics
        """
        logger.info(f"Computing sparsity score using {method} method")
        
        try:
            # Flatten SHAP values if they have multiple dimensions
            if len(shap_values.shape) > 2:
                shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
            # Calculate absolute values
            abs_shap = np.abs(shap_values)
            
            if method == "gini":
                sparsity_score = InterpretabilityMetrics._gini_coefficient(abs_shap)
            elif method == "entropy":
                sparsity_score = InterpretabilityMetrics._entropy_score(abs_shap)
            elif method == "l1_norm":
                sparsity_score = InterpretabilityMetrics._l1_norm_score(abs_shap)
            else:
                raise ValueError(f"Unknown sparsity method: {method}")
            
            # Calculate additional sparsity metrics
            n_features = abs_shap.shape[1]
            n_samples = abs_shap.shape[0]
            
            # Count features above threshold
            important_features = np.sum(abs_shap > threshold, axis=1)
            mean_important_features = np.mean(important_features)
            std_important_features = np.std(important_features)
            
            # Calculate sparsity ratio
            sparsity_ratio = mean_important_features / n_features
            
            return {
                'sparsity_score': float(sparsity_score),
                'sparsity_ratio': float(sparsity_ratio),
                'mean_important_features': float(mean_important_features),
                'std_important_features': float(std_important_features),
                'n_features': int(n_features),
                'n_samples': int(n_samples),
                'threshold': float(threshold),
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Error computing sparsity score: {e}")
            return {
                'sparsity_score': 0.0,
                'sparsity_ratio': 0.0,
                'mean_important_features': 0.0,
                'std_important_features': 0.0,
                'n_features': 0,
                'n_samples': 0,
                'threshold': float(threshold),
                'method': method,
                'error': str(e)
            }
    
    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """Calculate Gini coefficient for sparsity measurement."""
        # Flatten and sort values
        values = values.flatten()
        values = np.sort(values)
        n = len(values)
        
        if n == 0:
            return 0.0
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    @staticmethod
    def _entropy_score(values: np.ndarray) -> float:
        """Calculate entropy-based sparsity score."""
        # Normalize values to probabilities
        values = values.flatten()
        values = np.abs(values)
        
        if np.sum(values) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = values / np.sum(values)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(probs))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    @staticmethod
    def _l1_norm_score(values: np.ndarray) -> float:
        """Calculate L1 norm-based sparsity score."""
        # Calculate L1 norm
        l1_norm = np.sum(np.abs(values))
        
        # Normalize by number of features
        n_features = values.size
        return l1_norm / n_features if n_features > 0 else 0.0
    
    @staticmethod
    def intuitiveness_score(shap_values: np.ndarray, 
                           human_annotations: Optional[Dict[str, Any]] = None,
                           feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compare interpretations against human intuition or known patterns.
        
        This metric evaluates how well the model's explanations align with
        human understanding or domain knowledge.
        
        Args:
            shap_values: SHAP values array
            human_annotations: Human annotations or ground truth importance
            feature_names: List of feature names
            
        Returns:
            Dictionary containing intuitiveness metrics
        """
        logger.info("Computing intuitiveness score")
        
        try:
            if human_annotations is None:
                # If no human annotations, use heuristics
                return InterpretabilityMetrics._heuristic_intuitiveness(
                    shap_values, feature_names
                )
            else:
                # Compare with human annotations
                return InterpretabilityMetrics._annotation_intuitiveness(
                    shap_values, human_annotations, feature_names
                )
                
        except Exception as e:
            logger.error(f"Error computing intuitiveness score: {e}")
            return {
                'intuitiveness_score': 0.0,
                'correlation': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    @staticmethod
    def _heuristic_intuitiveness(shap_values: np.ndarray, 
                                feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute intuitiveness using heuristics when no human annotations are available."""
        # Flatten SHAP values
        if len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
        
        # Calculate feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Heuristic 1: Check if importance is concentrated (not uniform)
        gini = InterpretabilityMetrics._gini_coefficient(feature_importance)
        
        # Heuristic 2: Check if there are clear top features
        sorted_importance = np.sort(feature_importance)[::-1]
        if len(sorted_importance) > 1:
            top_ratio = sorted_importance[0] / np.sum(sorted_importance)
        else:
            top_ratio = 1.0
        
        # Heuristic 3: Check for negative importance (counterintuitive)
        negative_ratio = np.sum(shap_values < 0) / shap_values.size
        
        # Combine heuristics
        intuitiveness_score = gini * top_ratio * (1 - negative_ratio)
        
        return {
            'intuitiveness_score': float(intuitiveness_score),
            'gini_coefficient': float(gini),
            'top_feature_ratio': float(top_ratio),
            'negative_ratio': float(negative_ratio),
            'method': 'heuristic'
        }
    
    @staticmethod
    def _annotation_intuitiveness(shap_values: np.ndarray, 
                                 human_annotations: Dict[str, Any],
                                 feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute intuitiveness by comparing with human annotations."""
        # This would require human annotations in a specific format
        # For now, return a placeholder implementation
        logger.warning("Human annotation comparison not fully implemented")
        
        return {
            'intuitiveness_score': 0.5,  # Placeholder
            'correlation': 0.0,
            'method': 'annotation_placeholder'
        }
    
    @staticmethod
    def statistical_significance_test(metric_values_1: np.ndarray, 
                                    metric_values_2: np.ndarray,
                                    test_type: str = "t_test",
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance testing between two sets of metric values.
        
        Args:
            metric_values_1: First set of metric values
            metric_values_2: Second set of metric values
            test_type: Type of statistical test ("t_test", "wilcoxon", "mann_whitney")
            alpha: Significance level
            
        Returns:
            Dictionary containing statistical test results
        """
        logger.info(f"Performing {test_type} statistical significance test")
        
        try:
            if test_type == "t_test":
                statistic, p_value = stats.ttest_ind(metric_values_1, metric_values_2)
            elif test_type == "wilcoxon":
                statistic, p_value = stats.wilcoxon(metric_values_1, metric_values_2)
            elif test_type == "mann_whitney":
                statistic, p_value = stats.mannwhitneyu(metric_values_1, metric_values_2)
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(metric_values_1) - 1) * np.var(metric_values_1, ddof=1) + 
                                 (len(metric_values_2) - 1) * np.var(metric_values_2, ddof=1)) / 
                                (len(metric_values_1) + len(metric_values_2) - 2))
            cohens_d = (np.mean(metric_values_1) - np.mean(metric_values_2)) / pooled_std if pooled_std > 0 else 0.0
            
            # Determine significance
            is_significant = p_value < alpha
            
            return {
                'test_type': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_significant': bool(is_significant),
                'cohens_d': float(cohens_d),
                'alpha': float(alpha),
                'n1': len(metric_values_1),
                'n2': len(metric_values_2)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical significance test: {e}")
            return {
                'test_type': test_type,
                'statistic': 0.0,
                'p_value': 1.0,
                'is_significant': False,
                'cohens_d': 0.0,
                'alpha': float(alpha),
                'n1': len(metric_values_1),
                'n2': len(metric_values_2),
                'error': str(e)
            }
    
    def compute_all_metrics(self, 
                           model: Any,
                           inputs: Union[np.ndarray, List[str]],
                           shap_values: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           human_annotations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute all interpretability metrics for a given model and SHAP values.
        
        Args:
            model: Trained model instance
            inputs: Input data
            shap_values: SHAP values
            feature_names: List of feature names
            human_annotations: Human annotations for intuitiveness
            
        Returns:
            Dictionary containing all computed metrics
        """
        logger.info("Computing all interpretability metrics")
        
        results = {}
        
        # Faithfulness
        try:
            faithfulness_config = self.faithfulness_config
            results['faithfulness'] = self.faithfulness_score(
                model, inputs, shap_values,
                method=faithfulness_config.get('method', 'removal'),
                num_samples=faithfulness_config.get('num_samples', 50),
                feature_names=feature_names
            )
        except Exception as e:
            logger.error(f"Error computing faithfulness: {e}")
            results['faithfulness'] = {'error': str(e)}
        
        # Sparsity
        try:
            sparsity_config = self.sparsity_config
            results['sparsity'] = self.sparsity_score(
                shap_values,
                threshold=sparsity_config.get('threshold', 0.01),
                method=sparsity_config.get('method', 'gini')
            )
        except Exception as e:
            logger.error(f"Error computing sparsity: {e}")
            results['sparsity'] = {'error': str(e)}
        
        # Intuitiveness
        try:
            intuitiveness_config = self.intuitiveness_config
            if intuitiveness_config.get('enabled', False):
                results['intuitiveness'] = self.intuitiveness_score(
                    shap_values,
                    human_annotations=human_annotations,
                    feature_names=feature_names
                )
            else:
                results['intuitiveness'] = {'enabled': False}
        except Exception as e:
            logger.error(f"Error computing intuitiveness: {e}")
            results['intuitiveness'] = {'error': str(e)}
        
        # Consistency (requires multiple SHAP value sets)
        # This would be computed when comparing multiple models or datasets
        results['consistency'] = {'note': 'Requires multiple SHAP value sets for comparison'}
        
        logger.info("All interpretability metrics computed")
        return results


def create_interpretability_metrics(config: Dict[str, Any]) -> InterpretabilityMetrics:
    """
    Factory function to create an InterpretabilityMetrics instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        InterpretabilityMetrics instance
    """
    return InterpretabilityMetrics(config)


if __name__ == "__main__":
    # Test the interpretability metrics
    print("Interpretability Metrics module loaded successfully")
    print("Use create_interpretability_metrics() to create a metrics instance")
