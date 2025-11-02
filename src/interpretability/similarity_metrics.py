"""
Similarity Metrics for Embedding and SHAP Analysis.

This module implements:
1. Embedding similarity using neighborhood graph structure (from Prof. Tiago's paper)
2. SHAP similarity using Jaccard coefficient (from Indian researchers' paper)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from sklearn.neighbors import kneighbors_graph
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Embedding Similarity (Neighborhood Graph Structure)
# Based on: "Measuring similarity between embedding spaces using induced
# neighborhood graphs" by Tiago F. Tavares
# ============================================================================

def nearest_neighbors(
    x: np.ndarray,
    k: int,
    self_is_neighbor: bool = False,
    metric: str = 'minkowski',
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Find k-nearest neighbors for each point.

    Args:
        x: Input data of shape (n_samples, n_features)
        k: Number of neighbors
        self_is_neighbor: Whether to include the point itself as a neighbor
        metric: Distance metric
        n_jobs: Number of parallel jobs

    Returns:
        Array of shape (n_samples, k) with neighbor indices
    """
    if isinstance(k, float):
        k = int(k * x.shape[0])

    G = kneighbors_graph(
        x,
        k,
        mode='connectivity',
        metric=metric,
        include_self=self_is_neighbor,
        n_jobs=n_jobs,
    )

    A = []
    for i in range(G.shape[0]):
        A.append(G.getrow(i).nonzero()[1])

    A = np.vstack(A)
    return A


def compute_jaccard_similarity(sx: Set, sy: Set) -> float:
    """
    Compute Jaccard similarity between two sets.

    Args:
        sx: First set
        sy: Second set

    Returns:
        Jaccard similarity (intersection over union)
    """
    if len(sx.union(sy)) == 0:
        return 0.0
    return len(sx.intersection(sy)) / len(sx.union(sy))


def mean_neighborhood_similarity_from_neighborhood(nx: np.ndarray,
                                                   ny: np.ndarray) -> float:
    """
    Compute mean neighborhood similarity from precomputed neighborhoods.

    Args:
        nx: Neighborhood indices for first embedding space
        ny: Neighborhood indices for second embedding space

    Returns:
        Mean Jaccard similarity across all points
    """
    num_points = nx.shape[0]
    inter = 0.0

    for i in range(num_points):
        sx = set(nx[i])
        sy = set(ny[i])
        inter += compute_jaccard_similarity(sx, sy)

    inter /= num_points
    return inter


def mean_neighborhood_similarity_from_points(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    n_jobs: int = 1,
    metric: str = 'minkowski',
) -> float:
    """
    Compute NNGS (Nearest Neighbor Graph Similarity) between two embedding spaces.

    This is the main function for embedding space similarity as defined in:
    "Measuring similarity between embedding spaces using induced neighborhood graphs"

    Args:
        X: First embedding space of shape (n_samples, hidden_dim_X)
        Y: Second embedding space of shape (n_samples, hidden_dim_Y)
        k: Number of nearest neighbors to consider
        n_jobs: Number of parallel jobs
        metric: Distance metric

    Returns:
        Similarity score between 0 and 1 (higher is more similar)
    """
    logger.info(f"Computing NNGS similarity with k={k}")
    logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}")

    nx = nearest_neighbors(X, k=k, n_jobs=n_jobs, metric=metric)
    ny = nearest_neighbors(Y, k=k, n_jobs=n_jobs, metric=metric)

    similarity = mean_neighborhood_similarity_from_neighborhood(nx, ny)
    logger.info(f"NNGS similarity: {similarity:.4f}")

    return similarity


# ============================================================================
# SHAP Similarity (Jaccard on Top-K Features)
# Based on: "Compressed Models are NOT Trust-equivalent to Their Large
# Counterparts" by Rohit Raj Rai et al.
# ============================================================================

def get_top_k_features(shap_values: np.ndarray,
                       k: int,
                       feature_names: Optional[List[str]] = None) -> List[Set[Any]]:
    """
    Extract top-K most influential features for each sample based on SHAP values.

    Args:
        shap_values: SHAP values of shape (n_samples, n_features)
        k: Number of top features to select
        feature_names: Optional feature names (if None, uses indices)

    Returns:
        List of sets, where each set contains top-K feature identifiers
    """
    top_k_sets = []

    for i in range(shap_values.shape[0]):
        # Get absolute SHAP values for this sample
        abs_shap = np.abs(shap_values[i])

        # Get indices of top-K features
        top_k_indices = np.argsort(abs_shap)[-k:]

        # Convert to feature names or keep as indices
        if feature_names is not None:
            top_k_features = {feature_names[idx] for idx in top_k_indices}
        else:
            top_k_features = set(top_k_indices)

        top_k_sets.append(top_k_features)

    return top_k_sets


def shap_jaccard_similarity(shap_values_1: np.ndarray,
                           shap_values_2: np.ndarray,
                           k: int = 20,
                           feature_names: Optional[List[str]] = None) -> float:
    """
    Compute SHAP similarity using Jaccard coefficient on top-K features.

    This follows the methodology from:
    "Compressed Models are NOT Trust-equivalent to Their Large Counterparts"

    The method:
    1. For each sample, extract top-K most influential features from each model
    2. Compute Jaccard similarity between the two top-K sets
    3. Average across all samples

    Args:
        shap_values_1: SHAP values from first model (n_samples, n_features)
        shap_values_2: SHAP values from second model (n_samples, n_features)
        k: Number of top features to consider
        feature_names: Optional feature names for matching

    Returns:
        Mean Jaccard similarity across all samples
    """
    logger.info(f"Computing SHAP Jaccard similarity with k={k}")
    logger.info(f"SHAP1 shape: {shap_values_1.shape}, SHAP2 shape: {shap_values_2.shape}")

    # Ensure same number of samples
    n_samples = min(shap_values_1.shape[0], shap_values_2.shape[0])
    shap_values_1 = shap_values_1[:n_samples]
    shap_values_2 = shap_values_2[:n_samples]

    # Get top-K features for each model
    top_k_1 = get_top_k_features(shap_values_1, k, feature_names)
    top_k_2 = get_top_k_features(shap_values_2, k, feature_names)

    # Compute Jaccard similarity for each sample
    similarities = []
    for i in range(n_samples):
        sim = compute_jaccard_similarity(top_k_1[i], top_k_2[i])
        similarities.append(sim)

    # Average across samples
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    logger.info(f"SHAP Jaccard similarity: {mean_similarity:.4f} ± {std_similarity:.4f}")

    return mean_similarity


def shap_jaccard_similarity_detailed(
    shap_values_1: np.ndarray,
    shap_values_2: np.ndarray,
    k: int = 20,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute SHAP Jaccard similarity with detailed statistics.

    Args:
        shap_values_1: SHAP values from first model
        shap_values_2: SHAP values from second model
        k: Number of top features
        feature_names: Optional feature names

    Returns:
        Dictionary with similarity statistics
    """
    logger.info(f"Computing detailed SHAP Jaccard similarity with k={k}")

    # Ensure same number of samples
    n_samples = min(shap_values_1.shape[0], shap_values_2.shape[0])
    shap_values_1 = shap_values_1[:n_samples]
    shap_values_2 = shap_values_2[:n_samples]

    # Get top-K features for each model
    top_k_1 = get_top_k_features(shap_values_1, k, feature_names)
    top_k_2 = get_top_k_features(shap_values_2, k, feature_names)

    # Compute Jaccard similarity for each sample
    similarities = []
    for i in range(n_samples):
        sim = compute_jaccard_similarity(top_k_1[i], top_k_2[i])
        similarities.append(sim)

    similarities = np.array(similarities)

    # Compute statistics
    results = {
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'median_similarity': float(np.median(similarities)),
        'n_samples': n_samples,
        'k': k,
        'similarities': similarities.tolist()
    }

    logger.info(f"SHAP Jaccard similarity: {results['mean_similarity']:.4f} ± "
               f"{results['std_similarity']:.4f}")

    return results


# ============================================================================
# Pairwise Similarity Computation
# ============================================================================

def compute_pairwise_embedding_similarity(
    embeddings_dict: Dict[str, np.ndarray],
    k: int = 10,
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise embedding similarity for all model combinations.

    Args:
        embeddings_dict: Dictionary mapping model names to embedding arrays
        k: Number of nearest neighbors

    Returns:
        Dictionary mapping (model1, model2) pairs to similarity scores
    """
    logger.info(f"Computing pairwise embedding similarity for {len(embeddings_dict)} models")

    results = {}
    model_names = list(embeddings_dict.keys())

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i <= j:  # Compute only upper triangle (symmetric)
                logger.info(f"Computing similarity: {model1} vs {model2}")

                similarity = mean_neighborhood_similarity_from_points(
                    embeddings_dict[model1],
                    embeddings_dict[model2],
                    k=k,
                )

                results[(model1, model2)] = similarity
                if i != j:
                    results[(model2, model1)] = similarity  # Symmetric

    return results


def compute_pairwise_shap_similarity(
    shap_dict: Dict[str, np.ndarray],
    k: int = 20,
    feature_names: Optional[List[str]] = None
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise SHAP similarity for all model combinations.

    Args:
        shap_dict: Dictionary mapping model names to SHAP value arrays
        k: Number of top features
        feature_names: Optional feature names

    Returns:
        Dictionary mapping (model1, model2) pairs to similarity scores
    """
    logger.info(f"Computing pairwise SHAP similarity for {len(shap_dict)} models")

    results = {}
    model_names = list(shap_dict.keys())

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i <= j:  # Compute only upper triangle (symmetric)
                logger.info(f"Computing SHAP similarity: {model1} vs {model2}")

                similarity = shap_jaccard_similarity(
                    shap_dict[model1],
                    shap_dict[model2],
                    k=k,
                    feature_names=feature_names
                )

                results[(model1, model2)] = similarity
                if i != j:
                    results[(model2, model1)] = similarity  # Symmetric

    return results


def create_similarity_matrix(
    pairwise_similarities: Dict[Tuple[str, str], float],
    model_names: List[str]
) -> np.ndarray:
    """
    Convert pairwise similarities to a similarity matrix.

    Args:
        pairwise_similarities: Dictionary of pairwise similarities
        model_names: List of model names (defines matrix order)

    Returns:
        Similarity matrix of shape (n_models, n_models)
    """
    n_models = len(model_names)
    matrix = np.zeros((n_models, n_models))

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            key = (model1, model2)
            if key in pairwise_similarities:
                matrix[i, j] = pairwise_similarities[key]
            else:
                # Try reversed key
                key_rev = (model2, model1)
                if key_rev in pairwise_similarities:
                    matrix[i, j] = pairwise_similarities[key_rev]

    return matrix


if __name__ == "__main__":
    # Test similarity metrics
    print("SimilarityMetrics module loaded successfully")
    print("Available functions:")
    print("  - mean_neighborhood_similarity_from_points: Embedding similarity (NNGS)")
    print("  - shap_jaccard_similarity: SHAP similarity (Jaccard)")
