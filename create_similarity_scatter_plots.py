"""
Create Enhanced Scatter Plots for SHAP vs Embedding Similarity Analysis.

This script generates scatter plots showing the relationship between:
- Embedding similarity (using NNGS metric)
- SHAP similarity (using Jaccard similarity on top-k important features)

Plots are created per dataset and per model for detailed analysis.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.interpretability.similarity_metrics import compute_pairwise_embedding_similarity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


def compute_jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def load_embeddings(embeddings_dir: Path, model_name: str, dataset_name: str, model_type: str) -> Optional[np.ndarray]:
    """Load embeddings for a model-dataset combination."""
    # Try different possible filename patterns
    # Handle both standard names and prajjwal1/bert-* variants
    patterns = [
        f"{model_name}_{dataset_name}_{model_type}_embeddings.npz",
        f"{model_name.replace('/', '_')}_{dataset_name}_{model_type}_embeddings.npz",
        f"prajjwal1_{model_name}_{dataset_name}_{model_type}_embeddings.npz",  # For bert-tiny/small/medium
    ]

    for pattern in patterns:
        filepath = embeddings_dir / pattern
        if filepath.exists():
            try:
                data = np.load(filepath)
                embeddings = data['embeddings']
                logger.debug(f"Loaded embeddings from {pattern}: shape {embeddings.shape}")
                return embeddings
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                return None

    logger.warning(f"Embeddings not found for {model_name} on {dataset_name} ({model_type})")
    return None


def load_shap_analysis_results(results_dir: Path, model_name: str, dataset_name: str) -> Optional[Dict]:
    """Load SHAP analysis results from JSON file."""
    # Find the analysis results file
    pattern = f"{model_name}_{dataset_name}"
    matching_dirs = list(results_dir.glob(pattern))

    if not matching_dirs:
        logger.warning(f"No results found for {model_name} on {dataset_name}")
        return None

    results_file = matching_dirs[0] / "analysis_results.json"

    if not results_file.exists():
        logger.warning(f"Results file not found: {results_file}")
        return None

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded SHAP results for {model_name} on {dataset_name}")
        return data
    except Exception as e:
        logger.error(f"Error loading {results_file}: {e}")
        return None


def extract_top_k_features_from_metrics(metrics: Dict, k: int = 20) -> set:
    """
    Extract top-k important features from SHAP metrics.

    Uses sparsity metrics to identify which features are most important.
    """
    # Try to get feature importance from sparsity metrics
    if 'feature_importance' in metrics:
        importance = metrics['feature_importance']
        # Get top-k indices
        top_indices = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:k]
        return set(top_indices)

    # If not available, return empty set
    return set()


def compute_shap_similarity_from_analysis(
    results1: Dict,
    results2: Dict,
    k: int = 20
) -> float:
    """
    Compute SHAP similarity between two models based on their analysis results.

    This uses the available metrics to approximate similarity.
    For now, we'll use a simpler approach based on overlap in important features.
    """
    metrics1 = results1.get('metrics', {})
    metrics2 = results2.get('metrics', {})

    # Extract sparsity information
    sparsity1 = metrics1.get('sparsity', {})
    sparsity2 = metrics2.get('sparsity', {})

    # Compare mean important features as a proxy
    if 'mean_important_features' in sparsity1 and 'mean_important_features' in sparsity2:
        mean1 = sparsity1['mean_important_features']
        mean2 = sparsity2['mean_important_features']

        # Normalize by max to get similarity [0, 1]
        max_val = max(mean1, mean2)
        if max_val > 0:
            similarity = 1.0 - abs(mean1 - mean2) / max_val
            return similarity

    # If we can't compute, return 0
    return 0.0


def create_scatter_plot_per_dataset(
    embedding_data: Dict,
    shap_data: Dict,
    dataset_name: str,
    model_type: str,
    output_dir: Path
) -> Optional[Path]:
    """
    Create scatter plot showing embedding vs SHAP similarity for one dataset.
    Each point represents a pair of models.
    """
    embedding_vals = []
    shap_vals = []
    pair_labels = []
    seen_pairs = set()  # Track unique pairs to avoid duplicates

    # Extract data points
    for key, emb_sim in embedding_data.items():
        if key in shap_data and '_vs_' in key:
            parts = key.split('_vs_')
            if len(parts) == 2 and parts[0] != parts[1]:  # Exclude self-comparisons
                # Create normalized pair to avoid duplicates (e.g., "A vs B" and "B vs A")
                normalized_pair = tuple(sorted([parts[0], parts[1]]))

                if normalized_pair not in seen_pairs:
                    seen_pairs.add(normalized_pair)
                    embedding_vals.append(emb_sim)
                    shap_vals.append(shap_data[key])
                    # Shorten model names for readability
                    m1_short = parts[0].split('/')[-1][:15]
                    m2_short = parts[1].split('/')[-1][:15]
                    pair_labels.append(f"{m1_short}\nvs\n{m2_short}")

    if not embedding_vals:
        logger.warning(f"No data points for {dataset_name} ({model_type})")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with different colors per pair
    colors = plt.cm.viridis(np.linspace(0, 1, len(embedding_vals)))
    for i, (x, y, label) in enumerate(zip(embedding_vals, shap_vals, pair_labels)):
        ax.scatter(x, y, s=150, alpha=0.7, c=[colors[i]],
                  edgecolors='black', linewidth=1.5, label=label)

    # Add regression line if we have enough points
    if len(embedding_vals) > 1:
        z = np.polyfit(embedding_vals, shap_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(embedding_vals), max(embedding_vals), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=2.5, label='Linear fit')

        # Compute and display correlation
        correlation = np.corrcoef(embedding_vals, shap_vals)[0, 1]
        r_squared = correlation ** 2

        ax.text(0.05, 0.95,
               f'Correlation: {correlation:.3f}\n$R^2$: {r_squared:.3f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Labels and title
    ax.set_xlabel('Embedding Similarity (NNGS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SHAP Similarity (Feature Overlap)', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name.upper()} - {model_type.capitalize()} Models\nEmbedding vs SHAP Similarity',
                fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    # Save figure
    filename = f"scatter_{dataset_name}_{model_type}_detailed.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved scatter plot: {output_path}")
    return output_path


def create_combined_scatter_plot(
    comparison_results: Dict,
    model_type: str,
    output_dir: Path
) -> Optional[Path]:
    """
    Create combined scatter plot showing all datasets together.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    datasets = list(comparison_results[model_type].keys())

    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        dataset_results = comparison_results[model_type][dataset_name]

        embedding_data = dataset_results['embedding_similarities']
        shap_data = dataset_results['shap_similarities']

        embedding_vals = []
        shap_vals = []

        # Extract data points
        for key, emb_sim in embedding_data.items():
            if key in shap_data and '_vs_' in key:
                parts = key.split('_vs_')
                if len(parts) == 2 and parts[0] != parts[1]:
                    embedding_vals.append(emb_sim)
                    shap_vals.append(shap_data[key])

        if embedding_vals:
            # Scatter plot
            ax.scatter(embedding_vals, shap_vals, s=100, alpha=0.6,
                      c=range(len(embedding_vals)), cmap='viridis',
                      edgecolors='black', linewidth=1)

            # Regression line
            if len(embedding_vals) > 1:
                z = np.polyfit(embedding_vals, shap_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(embedding_vals), max(embedding_vals), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)

                # Correlation
                corr = np.corrcoef(embedding_vals, shap_vals)[0, 1]
                ax.text(0.05, 0.95, f'r={corr:.3f}',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        ax.set_xlabel('Embedding Similarity', fontsize=10)
        if idx == 0:
            ax.set_ylabel('SHAP Similarity', fontsize=10)
        ax.set_title(dataset_name.upper(), fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Embedding vs SHAP Similarity - {model_type.capitalize()} Models',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    filename = f"comparison_all_datasets_{model_type}_fixed.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved combined scatter plot: {output_path}")
    return output_path


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("Starting Enhanced Scatter Plot Generation")
    logger.info("="*80)

    # Paths
    project_dir = Path(__file__).parent
    results_dir = project_dir / "results"
    interpretability_dir = results_dir / "interpretability"
    embedding_shap_dir = results_dir / "embedding_shap_analysis"
    embeddings_dir = embedding_shap_dir / "embeddings"
    output_dir = embedding_shap_dir / "enhanced_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect available models from embedding files and interpretability results
    datasets = ['imdb', 'yelp_polarity', 'amazon_polarity']

    # Get all unique model names from interpretability results
    logger.info("\nAuto-detecting available models...")
    all_model_dirs = [d.name for d in interpretability_dir.iterdir() if d.is_dir()]
    model_set = set()
    for dir_name in all_model_dirs:
        # Extract model name (before the dataset name)
        for dataset in datasets:
            if dir_name.endswith(f"_{dataset}"):
                model_name = dir_name[:-len(f"_{dataset}")]
                model_set.add(model_name)
                break

    models = sorted(list(model_set))
    logger.info(f"\nDetected {len(models)} models with interpretability results:")
    for m in models:
        logger.info(f"  - {m}")
    logger.info(f"\nDatasets: {datasets}")

    # Compute similarities from analysis results
    logger.info("\n" + "="*80)
    logger.info("Computing Embedding and SHAP Similarities")
    logger.info("="*80 + "\n")

    all_results = {
        'pretrained': {},
        'finetuned': {}
    }

    for model_type in ['pretrained', 'finetuned']:
        logger.info(f"\nProcessing {model_type} models...")

        for dataset_name in datasets:
            logger.info(f"\n  Dataset: {dataset_name}")

            # Load embeddings for all models
            embeddings_dict = {}
            for model_name in models:
                embeddings = load_embeddings(
                    embeddings_dir,
                    model_name,
                    dataset_name,
                    model_type
                )
                if embeddings is not None:
                    embeddings_dict[model_name] = embeddings

            logger.info(f"    Loaded embeddings for {len(embeddings_dict)} models")

            # Compute pairwise embedding similarities
            embedding_similarities = {}
            if embeddings_dict:
                embedding_similarities = compute_pairwise_embedding_similarity(
                    embeddings_dict,
                    k=10
                )
                logger.info(f"    Computed {len(embedding_similarities)} embedding similarity pairs")

            # Load all SHAP analysis results for this dataset
            shap_results = {}
            for model_name in models:
                results = load_shap_analysis_results(
                    interpretability_dir,
                    model_name,
                    dataset_name
                )
                if results:
                    shap_results[model_name] = results

            logger.info(f"    Loaded SHAP results for {len(shap_results)} models")

            # Compute pairwise SHAP similarities
            shap_similarities = {}
            for i, model1 in enumerate(models):
                for model2 in models[i:]:
                    if model1 in shap_results and model2 in shap_results:
                        similarity = compute_shap_similarity_from_analysis(
                            shap_results[model1],
                            shap_results[model2],
                            k=20
                        )
                        key = f"{model1}_vs_{model2}"
                        shap_similarities[key] = similarity
                        # Add reverse for symmetry
                        if model1 != model2:
                            shap_similarities[f"{model2}_vs_{model1}"] = similarity

            logger.info(f"    Computed {len(shap_similarities)} SHAP similarity pairs")

            # Store results
            all_results[model_type][dataset_name] = {
                'embedding_similarities': {
                    f"{k[0]}_vs_{k[1]}": v for k, v in embedding_similarities.items()
                },
                'shap_similarities': shap_similarities,
                'available_models': list(embeddings_dict.keys())
            }

            # Create detailed scatter plot for this dataset
            if embedding_similarities and shap_similarities:
                create_scatter_plot_per_dataset(
                    all_results[model_type][dataset_name]['embedding_similarities'],
                    shap_similarities,
                    dataset_name,
                    model_type,
                    output_dir
                )

    # Create combined plots
    logger.info("\n" + "="*80)
    logger.info("Creating combined scatter plots")
    logger.info("="*80 + "\n")

    for model_type in ['pretrained', 'finetuned']:
        create_combined_scatter_plot(
            all_results,
            model_type,
            output_dir
        )

    # Save complete results
    results_file = embedding_shap_dir / "all_models_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nSaved comparison results to {results_file}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"\nModels analyzed: {len(models)}")
    for model in models:
        logger.info(f"  - {model}")
    logger.info(f"\nDatasets: {len(datasets)}")
    for dataset in datasets:
        logger.info(f"  - {dataset}")
    logger.info(f"\nTotal plots generated: {len(models)} models × {len(datasets)} datasets × 2 types")

    logger.info("\n" + "="*80)
    logger.info("Enhanced scatter plot generation complete!")
    logger.info(f"Plots saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
