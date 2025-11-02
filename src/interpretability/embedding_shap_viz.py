"""
Visualization Module for Embedding-SHAP Analysis.

This module creates publication-quality visualizations for:
1. Embedding similarity vs SHAP similarity scatter plots
2. Similarity heatmaps
3. SHAP value distribution boxplots
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class EmbeddingSHAPVisualizer:
    """
    Create visualizations for embedding and SHAP similarity analysis.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EmbeddingSHAPVisualizer initialized")
        logger.info(f"Output directory: {self.output_dir}")

    def plot_similarity_scatter(self,
                               embedding_similarities: Dict[Tuple[str, str], float],
                               shap_similarities: Dict[Tuple[str, str], float],
                               dataset_name: str,
                               model_type: str,
                               title: Optional[str] = None) -> Path:
        """
        Create scatter plot of embedding similarity vs SHAP similarity.

        Args:
            embedding_similarities: Dict of (model1, model2) -> similarity
            shap_similarities: Dict of (model1, model2) -> similarity
            dataset_name: Dataset name
            model_type: 'pretrained' or 'finetuned'
            title: Optional plot title

        Returns:
            Path to saved figure
        """
        # Extract matching pairs
        embedding_vals = []
        shap_vals = []
        pair_labels = []

        for (m1, m2), emb_sim in embedding_similarities.items():
            if (m1, m2) in shap_similarities and m1 != m2:  # Exclude self-comparisons
                embedding_vals.append(emb_sim)
                shap_vals.append(shap_similarities[(m1, m2)])
                pair_labels.append(f"{m1[:10]} vs {m2[:10]}")  # Truncate for readability

        if not embedding_vals:
            logger.warning(f"No matching pairs for scatter plot: {dataset_name} ({model_type})")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot
        scatter = ax.scatter(embedding_vals, shap_vals, s=100, alpha=0.6, c=range(len(embedding_vals)),
                           cmap='viridis', edgecolors='black', linewidth=0.5)

        # Add labels for points
        for i, (x, y, label) in enumerate(zip(embedding_vals, shap_vals, pair_labels)):
            ax.annotate(label, (x, y), fontsize=7, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')

        # Compute and plot regression line
        if len(embedding_vals) > 1:
            z = np.polyfit(embedding_vals, shap_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(embedding_vals), max(embedding_vals), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label='Linear fit')

            # Compute correlation
            correlation = np.corrcoef(embedding_vals, shap_vals)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Labels and title
        ax.set_xlabel('Embedding Similarity (NNGS)', fontsize=11)
        ax.set_ylabel('SHAP Similarity (Jaccard)', fontsize=11)

        if title is None:
            title = f'{dataset_name.upper()} - {model_type.capitalize()} Models'
        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save figure
        filename = f"scatter_{dataset_name}_{model_type}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved scatter plot: {output_path}")
        return output_path

    def plot_similarity_heatmap(self,
                               similarity_matrix: np.ndarray,
                               model_names: List[str],
                               dataset_name: str,
                               similarity_type: str,
                               model_type: str,
                               title: Optional[str] = None) -> Path:
        """
        Create heatmap of similarity matrix.

        Args:
            similarity_matrix: Similarity matrix (n_models x n_models)
            model_names: List of model names
            dataset_name: Dataset name
            similarity_type: 'embedding' or 'shap'
            model_type: 'pretrained' or 'finetuned'
            title: Optional plot title

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(similarity_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0,
                   vmax=1,
                   xticklabels=model_names,
                   yticklabels=model_names,
                   square=True,
                   cbar_kws={'label': 'Similarity Score'},
                   ax=ax)

        # Title
        if title is None:
            title = f'{similarity_type.upper()} Similarity - {dataset_name.upper()} ({model_type.capitalize()})'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Save figure
        filename = f"heatmap_{similarity_type}_{dataset_name}_{model_type}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved heatmap: {output_path}")
        return output_path

    def plot_shap_boxplot(self,
                         shap_values: np.ndarray,
                         feature_names: List[str],
                         model_name: str,
                         dataset_name: str,
                         top_k: int = 20,
                         title: Optional[str] = None) -> Path:
        """
        Create boxplot of SHAP values showing token variance.

        Args:
            shap_values: SHAP values array (n_samples, n_features)
            feature_names: List of feature names
            model_name: Model name
            dataset_name: Dataset name
            top_k: Number of top variance tokens to show
            title: Optional plot title

        Returns:
            Path to saved figure
        """
        # Compute variance for each feature
        feature_variance = np.var(shap_values, axis=0)

        # Get top-k features by variance
        top_indices = np.argsort(feature_variance)[-top_k:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_shap_values = shap_values[:, top_indices]

        # Prepare data for boxplot
        data_for_plot = []
        for i, feature in enumerate(top_features):
            for val in top_shap_values[:, i]:
                data_for_plot.append({
                    'Token': feature,
                    'SHAP Value': val,
                    'Variance': feature_variance[top_indices[i]]
                })

        df = pd.DataFrame(data_for_plot)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Boxplot
        bp = sns.boxplot(data=df, x='SHAP Value', y='Token',
                        palette='viridis', ax=ax)

        # Add variance information
        for i, feature in enumerate(top_features):
            variance = feature_variance[top_indices[i]]
            ax.text(1.02, i, f'var={variance:.4f}', transform=ax.get_yaxis_transform(),
                   fontsize=7, va='center')

        # Labels and title
        ax.set_xlabel('SHAP Value', fontsize=11)
        ax.set_ylabel('Token', fontsize=11)

        if title is None:
            title = f'SHAP Value Distribution - {model_name} on {dataset_name.upper()}'
        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # Save figure
        model_name_safe = model_name.replace('/', '_')
        filename = f"boxplot_{model_name_safe}_{dataset_name}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved boxplot: {output_path}")
        return output_path

    def plot_all_datasets_comparison(self,
                                    results: Dict[str, Any],
                                    model_type: str) -> Path:
        """
        Create comprehensive comparison plot across all datasets.

        Args:
            results: Results dictionary from ModelComparator
            model_type: 'pretrained' or 'finetuned'

        Returns:
            Path to saved figure
        """
        datasets = list(results[model_type].keys())
        n_datasets = len(datasets)

        # Create figure with subplots
        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
        if n_datasets == 1:
            axes = [axes]

        for i, dataset_name in enumerate(datasets):
            dataset_results = results[model_type][dataset_name]

            # Extract embedding and SHAP similarities
            embedding_sims = dataset_results['embedding_similarities']
            shap_sims = dataset_results['shap_similarities']

            # Match pairs
            embedding_vals = []
            shap_vals = []

            for key, emb_sim in embedding_sims.items():
                # Parse key (format: "model1_vs_model2")
                if key in shap_sims:
                    # Avoid self-comparisons
                    if '_vs_' in key:
                        parts = key.split('_vs_')
                        if len(parts) == 2 and parts[0] != parts[1]:
                            embedding_vals.append(emb_sim)
                            shap_vals.append(shap_sims[key])

            if embedding_vals:
                # Scatter plot
                axes[i].scatter(embedding_vals, shap_vals, s=80, alpha=0.6,
                              c=range(len(embedding_vals)), cmap='viridis',
                              edgecolors='black', linewidth=0.5)

                # Regression line
                if len(embedding_vals) > 1:
                    z = np.polyfit(embedding_vals, shap_vals, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(embedding_vals), max(embedding_vals), 100)
                    axes[i].plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)

                    # Correlation
                    corr = np.corrcoef(embedding_vals, shap_vals)[0, 1]
                    axes[i].text(0.05, 0.95, f'r={corr:.3f}',
                               transform=axes[i].transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            axes[i].set_xlabel('Embedding Similarity')
            axes[i].set_ylabel('SHAP Similarity' if i == 0 else '')
            axes[i].set_title(dataset_name.upper(), fontweight='bold')
            axes[i].grid(True, alpha=0.3)

        fig.suptitle(f'Embedding vs SHAP Similarity - {model_type.capitalize()} Models',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save figure
        filename = f"comparison_all_datasets_{model_type}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved comparison plot: {output_path}")
        return output_path

    def plot_pretrained_vs_finetuned(self,
                                     results: Dict[str, Any],
                                     embeddings_dir: Path,
                                     shap_dir: Path) -> Path:
        """
        Create scatter plot comparing embedding similarity vs SHAP similarity
        for pre-trained vs fine-tuned model pairs.

        This plot shows the relationship between how much embeddings change
        and how much SHAP explanations change after fine-tuning.

        Args:
            results: Results dictionary from ModelComparator
            embeddings_dir: Directory containing embeddings
            shap_dir: Directory containing SHAP results

        Returns:
            Path to saved figure
        """
        logger.info("Creating pre-trained vs fine-tuned scatter plot")

        # Import here to avoid circular dependency
        from .similarity_metrics import mean_neighborhood_similarity_from_points, shap_jaccard_similarity

        # Collect data for each model-dataset combination
        comparison_data = []

        for dataset_name in results['pretrained'].keys():
            pretrained_results = results['pretrained'][dataset_name]
            finetuned_results = results['finetuned'][dataset_name]

            # Get common models between pretrained and finetuned
            pretrained_models = set(pretrained_results['available_models'])
            finetuned_models = set(finetuned_results['available_models'])
            common_models = pretrained_models.intersection(finetuned_models)

            logger.info(f"Dataset {dataset_name}: {len(common_models)} common models")

            k_shap = finetuned_results.get('k_shap', 20)

            # For each model, compute both embedding and SHAP similarity
            for model_name in common_models:
                model_name_safe = model_name.replace('/', '_')

                # Load embeddings
                pretrained_file = embeddings_dir / f"{model_name_safe}_{dataset_name}_pretrained_embeddings.npz"
                finetuned_file = embeddings_dir / f"{model_name_safe}_{dataset_name}_finetuned_embeddings.npz"

                embedding_similarity = None
                if pretrained_file.exists() and finetuned_file.exists():
                    try:
                        pretrained_emb = np.load(pretrained_file)['embeddings']
                        finetuned_emb = np.load(finetuned_file)['embeddings']

                        # Compute embedding similarity between pretrained and finetuned
                        k = pretrained_results['k_embedding']
                        embedding_similarity = mean_neighborhood_similarity_from_points(
                            pretrained_emb,
                            finetuned_emb,
                            k=k
                        )
                        logger.info(f"  {model_name} on {dataset_name} - Embedding: {embedding_similarity:.4f}")

                    except Exception as e:
                        logger.warning(f"Error loading embeddings for {model_name} on {dataset_name}: {e}")

                # Load SHAP values (only fine-tuned models have SHAP)
                # We'll compare SHAP from fine-tuned model to itself as a baseline
                # Or we can skip SHAP for this comparison since pretrained doesn't have SHAP
                shap_similarity = None
                shap_pattern = f"{model_name}_{dataset_name}"
                matching_dirs = list(shap_dir.glob(shap_pattern))

                if matching_dirs and embedding_similarity is not None:
                    shap_result_dir = matching_dirs[0]
                    npz_files = list(shap_result_dir.glob("interpretability/*.npz"))
                    if not npz_files:
                        npz_files = list(shap_result_dir.glob("*.npz"))

                    if npz_files:
                        try:
                            shap_file = sorted(npz_files)[-1]
                            data = np.load(shap_file, allow_pickle=True)
                            shap_values = data['shap_values']

                            # For demonstration, we compute SHAP consistency by comparing
                            # first half vs second half of samples (as a proxy for stability)
                            n_samples = shap_values.shape[0]
                            mid = n_samples // 2
                            shap_similarity = shap_jaccard_similarity(
                                shap_values[:mid],
                                shap_values[mid:2*mid],
                                k=k_shap
                            )

                            logger.info(f"  {model_name} on {dataset_name} - SHAP: {shap_similarity:.4f}")

                        except Exception as e:
                            logger.warning(f"Error loading SHAP for {model_name} on {dataset_name}: {e}")

                if embedding_similarity is not None and shap_similarity is not None:
                    comparison_data.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'embedding_similarity': embedding_similarity,
                        'shap_similarity': shap_similarity
                    })

        if not comparison_data:
            logger.warning("No data available for pre-trained vs fine-tuned comparison")
            return None

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Create scatter plot
        datasets = df['dataset'].unique()
        n_datasets = len(datasets)

        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
        if n_datasets == 1:
            axes = [axes]

        for i, dataset in enumerate(datasets):
            df_dataset = df[df['dataset'] == dataset]

            if len(df_dataset) == 0:
                continue

            # Scatter plot
            scatter = axes[i].scatter(
                df_dataset['embedding_similarity'],
                df_dataset['shap_similarity'],
                s=100,
                alpha=0.6,
                c=range(len(df_dataset)),
                cmap='viridis',
                edgecolors='black',
                linewidth=0.5
            )

            # Add labels for each point
            for idx, row in df_dataset.iterrows():
                axes[i].annotate(
                    row['model'][:15],  # Truncate long names
                    (row['embedding_similarity'], row['shap_similarity']),
                    fontsize=8,
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords='offset points'
                )

            # Regression line if we have enough points
            if len(df_dataset) > 1:
                z = np.polyfit(df_dataset['embedding_similarity'], df_dataset['shap_similarity'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(
                    df_dataset['embedding_similarity'].min(),
                    df_dataset['embedding_similarity'].max(),
                    100
                )
                axes[i].plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label='Linear fit')

                # Correlation
                corr = np.corrcoef(df_dataset['embedding_similarity'], df_dataset['shap_similarity'])[0, 1]
                axes[i].text(
                    0.05, 0.95,
                    f'r={corr:.3f}',
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

            # Customize
            axes[i].set_xlabel('Embedding Similarity\n(Pre-trained vs Fine-tuned)', fontsize=11)
            axes[i].set_ylabel('SHAP Similarity\n(Fine-tuned consistency)' if i == 0 else '', fontsize=11)
            axes[i].set_title(dataset.upper(), fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            if len(df_dataset) > 1:
                axes[i].legend()

        fig.suptitle('Embedding Change vs SHAP Consistency After Fine-tuning',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save figure
        filename = "pretrained_vs_finetuned_scatter.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved pre-trained vs fine-tuned scatter plot: {output_path}")

        # Also save the data to CSV
        csv_path = self.output_dir / "pretrained_vs_finetuned_data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison data: {csv_path}")

        return output_path

    def create_all_visualizations(self,
                                 results: Dict[str, Any],
                                 shap_dir: Path,
                                 embeddings_dir: Optional[Path] = None) -> List[Path]:
        """
        Generate all visualizations from comparison results.

        Args:
            results: Results dictionary from ModelComparator
            shap_dir: Directory containing SHAP results for boxplots
            embeddings_dir: Directory containing embeddings (for pre-trained vs fine-tuned plot)

        Returns:
            List of paths to generated figures
        """
        generated_files = []

        # 1. Scatter plots for each dataset and model type
        for model_type in ['pretrained', 'finetuned']:
            for dataset_name, dataset_results in results[model_type].items():
                # Convert string keys back to tuples for scatter plot
                embedding_sims = {}
                for key_str, value in dataset_results['embedding_similarities'].items():
                    parts = key_str.split('_vs_')
                    if len(parts) == 2:
                        embedding_sims[(parts[0], parts[1])] = value

                shap_sims = {}
                for key_str, value in dataset_results['shap_similarities'].items():
                    parts = key_str.split('_vs_')
                    if len(parts) == 2:
                        shap_sims[(parts[0], parts[1])] = value

                if shap_sims:  # Only create scatter if we have SHAP data
                    path = self.plot_similarity_scatter(
                        embedding_sims,
                        shap_sims,
                        dataset_name,
                        model_type
                    )
                    if path:
                        generated_files.append(path)

                # Heatmaps
                if dataset_results['embedding_similarity_matrix']:
                    emb_matrix = np.array(dataset_results['embedding_similarity_matrix'])
                    path = self.plot_similarity_heatmap(
                        emb_matrix,
                        dataset_results['available_models'],
                        dataset_name,
                        'embedding',
                        model_type
                    )
                    generated_files.append(path)

                if dataset_results['shap_similarity_matrix']:
                    shap_matrix = np.array(dataset_results['shap_similarity_matrix'])
                    path = self.plot_similarity_heatmap(
                        shap_matrix,
                        dataset_results['shap_available_models'],
                        dataset_name,
                        'shap',
                        model_type
                    )
                    generated_files.append(path)

        # 2. Comprehensive comparison plots
        for model_type in ['pretrained', 'finetuned']:
            path = self.plot_all_datasets_comparison(results, model_type)
            generated_files.append(path)

        # 3. Pre-trained vs fine-tuned comparison (NEW!)
        if embeddings_dir is not None:
            path = self.plot_pretrained_vs_finetuned(results, embeddings_dir, shap_dir)
            if path:
                generated_files.append(path)

        logger.info(f"\nGenerated {len(generated_files)} visualization files")
        return generated_files


if __name__ == "__main__":
    print("EmbeddingSHAPVisualizer module loaded successfully")
