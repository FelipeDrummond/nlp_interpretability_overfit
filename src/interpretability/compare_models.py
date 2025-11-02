"""
Model Comparison Analysis Script.

This script compares models across embedding space and SHAP interpretability dimensions.
It generates similarity matrices and comparative analyses for both pre-trained and fine-tuned models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

from .similarity_metrics import (
    compute_pairwise_embedding_similarity,
    compute_pairwise_shap_similarity,
    create_similarity_matrix
)

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare models across embedding and SHAP similarity dimensions.
    """

    def __init__(self,
                 embeddings_dir: Path,
                 shap_dir: Path,
                 output_dir: Path):
        """
        Initialize the model comparator.

        Args:
            embeddings_dir: Directory containing embedding files
            shap_dir: Directory containing SHAP analysis results
            output_dir: Directory to save comparison results
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.shap_dir = Path(shap_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ModelComparator initialized")
        logger.info(f"Embeddings dir: {self.embeddings_dir}")
        logger.info(f"SHAP dir: {self.shap_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def load_embeddings(self,
                       model_name: str,
                       dataset_name: str,
                       model_type: str) -> Optional[np.ndarray]:
        """
        Load embeddings for a specific model-dataset combination.

        Args:
            model_name: Model name
            dataset_name: Dataset name
            model_type: 'pretrained' or 'finetuned'

        Returns:
            Embedding array or None if not found
        """
        # Construct filename
        model_name_safe = model_name.replace('/', '_')
        filename = f"{model_name_safe}_{dataset_name}_{model_type}_embeddings.npz"
        filepath = self.embeddings_dir / filename

        if not filepath.exists():
            logger.warning(f"Embeddings not found: {filepath}")
            return None

        try:
            data = np.load(filepath)
            embeddings = data['embeddings']
            logger.info(f"Loaded embeddings from {filename}: shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings from {filepath}: {e}")
            return None

    def load_shap_values(self,
                        model_name: str,
                        dataset_name: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Load SHAP values for a specific model-dataset combination.

        Args:
            model_name: Model name
            dataset_name: Dataset name

        Returns:
            Tuple of (SHAP values array, feature names) or None if not found
        """
        # Find SHAP results directory
        shap_dir_pattern = f"{model_name}_{dataset_name}"
        matching_dirs = list(self.shap_dir.glob(shap_dir_pattern))

        if not matching_dirs:
            logger.warning(f"SHAP directory not found for {model_name} on {dataset_name}")
            return None

        shap_result_dir = matching_dirs[0]

        # Look for SHAP results files
        npz_files = list(shap_result_dir.glob("interpretability/*.npz"))

        if not npz_files:
            # Try in the main directory
            npz_files = list(shap_result_dir.glob("*.npz"))

        if not npz_files:
            logger.warning(f"No SHAP .npz files found in {shap_result_dir}")
            return None

        try:
            # Load the first (or latest) SHAP file
            shap_file = sorted(npz_files)[-1]
            data = np.load(shap_file, allow_pickle=True)

            shap_values = data['shap_values']
            feature_names = data['feature_names'].tolist()

            logger.info(f"Loaded SHAP values from {shap_file.name}: shape {shap_values.shape}")
            return shap_values, feature_names

        except Exception as e:
            logger.error(f"Error loading SHAP values: {e}")
            return None

    def compare_models_on_dataset(self,
                                 model_names: List[str],
                                 dataset_name: str,
                                 model_type: str,
                                 k_embedding: int = 10,
                                 k_shap: int = 20) -> Dict[str, Any]:
        """
        Compare all models on a single dataset.

        Args:
            model_names: List of model identifiers
            dataset_name: Dataset name
            model_type: 'pretrained' or 'finetuned'
            k_embedding: k parameter for embedding similarity
            k_shap: k parameter for SHAP similarity (top-K features)

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing {model_type} models on {dataset_name}")
        logger.info(f"{'='*60}\n")

        # Load all embeddings
        embeddings_dict = {}
        for model_name in model_names:
            embeddings = self.load_embeddings(model_name, dataset_name, model_type)
            if embeddings is not None:
                embeddings_dict[model_name] = embeddings

        # Load all SHAP values (only for fine-tuned models)
        shap_dict = {}
        feature_names = None
        if model_type == 'finetuned':
            for model_name in model_names:
                shap_data = self.load_shap_values(model_name, dataset_name)
                if shap_data is not None:
                    shap_values, feats = shap_data
                    shap_dict[model_name] = shap_values
                    if feature_names is None:
                        feature_names = feats

        # Compute embedding similarities
        logger.info("\nComputing embedding similarities...")
        embedding_similarities = compute_pairwise_embedding_similarity(
            embeddings_dict,
            k=k_embedding
        )

        # Create embedding similarity matrix
        available_models = list(embeddings_dict.keys())
        embedding_matrix = create_similarity_matrix(
            embedding_similarities,
            available_models
        )

        # Compute SHAP similarities (only if we have SHAP data)
        shap_similarities = {}
        shap_matrix = None
        if shap_dict:
            logger.info("\nComputing SHAP similarities...")
            shap_similarities = compute_pairwise_shap_similarity(
                shap_dict,
                k=k_shap,
                feature_names=feature_names
            )

            # Create SHAP similarity matrix
            shap_available_models = list(shap_dict.keys())
            shap_matrix = create_similarity_matrix(
                shap_similarities,
                shap_available_models
            )

        # Prepare results
        results = {
            'dataset_name': dataset_name,
            'model_type': model_type,
            'k_embedding': k_embedding,
            'k_shap': k_shap,
            'available_models': available_models,
            'embedding_similarities': {
                f"{k[0]}_vs_{k[1]}": v
                for k, v in embedding_similarities.items()
            },
            'embedding_similarity_matrix': embedding_matrix.tolist(),
            'shap_similarities': {
                f"{k[0]}_vs_{k[1]}": v
                for k, v in shap_similarities.items()
            } if shap_similarities else {},
            'shap_similarity_matrix': shap_matrix.tolist() if shap_matrix is not None else None,
            'shap_available_models': list(shap_dict.keys()) if shap_dict else []
        }

        # Compute cross-correlation between embedding and SHAP similarities
        if shap_similarities and embedding_similarities:
            embedding_vals = []
            shap_vals = []

            for (m1, m2), emb_sim in embedding_similarities.items():
                if (m1, m2) in shap_similarities:
                    embedding_vals.append(emb_sim)
                    shap_vals.append(shap_similarities[(m1, m2)])

            if embedding_vals and shap_vals:
                correlation = np.corrcoef(embedding_vals, shap_vals)[0, 1]
                results['embedding_shap_correlation'] = float(correlation)
                logger.info(f"\nCorrelation between embedding and SHAP similarities: {correlation:.4f}")

        return results

    def compare_pretrained_vs_finetuned(self,
                                       model_names: List[str],
                                       datasets: List[str],
                                       k_embedding: int = 10,
                                       k_shap: int = 20) -> Dict[str, Any]:
        """
        Compare pre-trained vs fine-tuned models across all datasets.

        Args:
            model_names: List of model identifiers
            datasets: List of dataset names
            k_embedding: k parameter for embedding similarity
            k_shap: k parameter for SHAP similarity

        Returns:
            Dictionary with all comparison results
        """
        all_results = {
            'pretrained': {},
            'finetuned': {},
            'comparison_summary': {}
        }

        for dataset_name in datasets:
            # Compare pre-trained models
            pretrained_results = self.compare_models_on_dataset(
                model_names=model_names,
                dataset_name=dataset_name,
                model_type='pretrained',
                k_embedding=k_embedding,
                k_shap=k_shap
            )
            all_results['pretrained'][dataset_name] = pretrained_results

            # Compare fine-tuned models
            finetuned_results = self.compare_models_on_dataset(
                model_names=model_names,
                dataset_name=dataset_name,
                model_type='finetuned',
                k_embedding=k_embedding,
                k_shap=k_shap
            )
            all_results['finetuned'][dataset_name] = finetuned_results

            # Summary comparison
            summary = {
                'dataset': dataset_name,
                'pretrained_embedding_mean': float(np.mean(pretrained_results['embedding_similarity_matrix'])),
                'finetuned_embedding_mean': float(np.mean(finetuned_results['embedding_similarity_matrix'])),
            }

            if finetuned_results['shap_similarity_matrix'] is not None:
                summary['finetuned_shap_mean'] = float(np.mean(finetuned_results['shap_similarity_matrix']))

            if 'embedding_shap_correlation' in finetuned_results:
                summary['embedding_shap_correlation'] = finetuned_results['embedding_shap_correlation']

            all_results['comparison_summary'][dataset_name] = summary

        return all_results

    def save_results(self, results: Dict[str, Any], filename: str = "comparison_results.json"):
        """
        Save comparison results to JSON file.

        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def create_summary_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary DataFrame from comparison results.

        Args:
            results: Results dictionary

        Returns:
            Pandas DataFrame with summary statistics
        """
        rows = []

        for dataset_name, summary in results['comparison_summary'].items():
            row = {
                'dataset': dataset_name,
                'pretrained_embedding_similarity': summary['pretrained_embedding_mean'],
                'finetuned_embedding_similarity': summary['finetuned_embedding_mean'],
            }

            if 'finetuned_shap_mean' in summary:
                row['finetuned_shap_similarity'] = summary['finetuned_shap_mean']

            if 'embedding_shap_correlation' in summary:
                row['embedding_shap_correlation'] = summary['embedding_shap_correlation']

            rows.append(row)

        df = pd.DataFrame(rows)
        return df


if __name__ == "__main__":
    print("ModelComparator module loaded successfully")
