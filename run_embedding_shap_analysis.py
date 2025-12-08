#!/usr/bin/env python3
"""
Main Runner Script for Embedding-SHAP Similarity Analysis.

This script orchestrates the complete pipeline:
1. Extract CLS token embeddings from pre-trained and fine-tuned models
2. Compute embedding similarity (NNGS) between models
3. Compute SHAP similarity (Jaccard) between models
4. Generate visualizations and analysis reports

Usage:
    python run_embedding_shap_analysis.py [--config config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.interpretability.embedding_extractor import extract_embeddings_for_all_models
from src.interpretability.compare_models import ModelComparator
from src.interpretability.embedding_shap_viz import EmbeddingSHAPVisualizer
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run embedding-SHAP similarity analysis'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip embedding extraction (use existing embeddings)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples per dataset'
    )
    parser.add_argument(
        '--k-embedding',
        type=int,
        default=10,
        help='Number of neighbors for embedding similarity (NNGS)'
    )
    parser.add_argument(
        '--k-shap',
        type=int,
        default=20,
        help='Number of top features for SHAP similarity (Jaccard)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding extraction'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'mps', 'cuda'],
        help='Device for computation'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/embedding_shap_analysis'),
        help='Output directory for results'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Create output directories
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = output_dir / 'embeddings'
    embeddings_dir.mkdir(exist_ok=True)

    visualizations_dir = output_dir / 'visualizations'
    visualizations_dir.mkdir(exist_ok=True)

    # Setup logging
    log_file = 'embedding_shap_analysis.log'
    setup_logging(log_level="INFO", log_file=log_file, log_dir=str(output_dir))

    logger.info("="*70)
    logger.info("EMBEDDING-SHAP SIMILARITY ANALYSIS")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max samples per dataset: {args.max_samples}")
    logger.info(f"k for embedding similarity: {args.k_embedding}")
    logger.info(f"k for SHAP similarity: {args.k_shap}")
    logger.info(f"Device: {args.device}")

    # Load configuration
    config = load_config(args.config)

    # Define models to analyze
    # Map internal model names to HuggingFace identifiers
    # These should match the model_name in config.yaml
    # Using MultiBERTs models to compare different initialization seeds
    models_config = {
        'multiberts-seed_0': 'google/multiberts-seed_0',
        'multiberts-seed_1': 'google/multiberts-seed_1',
        'multiberts-seed_2': 'google/multiberts-seed_2',
        'multiberts-seed_3': 'google/multiberts-seed_3',
        'multiberts-seed_4': 'google/multiberts-seed_4',
        'multiberts-seed_5': 'google/multiberts-seed_5',
        'multiberts-seed_6': 'google/multiberts-seed_6',
        'multiberts-seed_7': 'google/multiberts-seed_7',
        'multiberts-seed_8': 'google/multiberts-seed_8',
        'multiberts-seed_9': 'google/multiberts-seed_9',
        'multiberts-seed_10': 'google/multiberts-seed_10',
        'multiberts-seed_11': 'google/multiberts-seed_11',
        'multiberts-seed_12': 'google/multiberts-seed_12',
        'multiberts-seed_13': 'google/multiberts-seed_13',
        'multiberts-seed_14': 'google/multiberts-seed_14',
        'multiberts-seed_15': 'google/multiberts-seed_15',
        'multiberts-seed_16': 'google/multiberts-seed_16',
        'multiberts-seed_17': 'google/multiberts-seed_17',
        'multiberts-seed_18': 'google/multiberts-seed_18',
        'multiberts-seed_19': 'google/multiberts-seed_19',
        'multiberts-seed_20': 'google/multiberts-seed_20',
        'multiberts-seed_21': 'google/multiberts-seed_21',
        'multiberts-seed_22': 'google/multiberts-seed_22',
        'multiberts-seed_23': 'google/multiberts-seed_23',
        'multiberts-seed_24': 'google/multiberts-seed_24',
        # Note: You can comment out models you don't want to analyze to save time
    }

    datasets = ['imdb', 'yelp_polarity', 'amazon_polarity']

    # Paths
    models_dir = Path('results/models')
    shap_dir = Path('results/interpretability')

    # Step 1: Extract embeddings (if not skipped)
    if not args.skip_extraction:
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Extracting CLS Token Embeddings")
        logger.info("="*70 + "\n")

        try:
            embedding_results = extract_embeddings_for_all_models(
                models_config=models_config,
                datasets=datasets,
                output_dir=embeddings_dir,
                models_dir=models_dir,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                device=args.device
            )

            logger.info(f"\nExtracted embeddings for {len(embedding_results)} model-dataset combinations")

        except Exception as e:
            logger.error(f"Error during embedding extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
    else:
        logger.info("Skipping embedding extraction (using existing embeddings)")

    # Step 2: Compute similarities and compare models
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Computing Similarity Metrics")
    logger.info("="*70 + "\n")

    try:
        comparator = ModelComparator(
            embeddings_dir=embeddings_dir,
            shap_dir=shap_dir,
            output_dir=output_dir
        )

        # Get model names for comparison
        model_names = list(models_config.keys())

        # Compare pre-trained vs fine-tuned
        results = comparator.compare_pretrained_vs_finetuned(
            model_names=model_names,
            datasets=datasets,
            k_embedding=args.k_embedding,
            k_shap=args.k_shap
        )

        # Save results
        comparator.save_results(results, filename='comparison_results.json')

        # Create summary DataFrame
        summary_df = comparator.create_summary_dataframe(results)
        summary_path = output_dir / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("SUMMARY OF RESULTS")
        logger.info("="*70 + "\n")
        print(summary_df.to_string(index=False))

    except Exception as e:
        logger.error(f"Error during similarity computation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    # Step 3: Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Generating Visualizations")
    logger.info("="*70 + "\n")

    try:
        visualizer = EmbeddingSHAPVisualizer(output_dir=visualizations_dir)

        generated_files = visualizer.create_all_visualizations(
            results=results,
            shap_dir=shap_dir,
            embeddings_dir=embeddings_dir  # Pass embeddings directory for pre-trained vs fine-tuned plot
        )

        logger.info(f"\nGenerated {len(generated_files)} visualization files")
        logger.info(f"Visualizations saved to: {visualizations_dir}")

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70 + "\n")
    logger.info(f"Results directory: {output_dir}")
    logger.info(f"  - Embeddings: {embeddings_dir}")
    logger.info(f"  - Visualizations: {visualizations_dir}")
    logger.info(f"  - Comparison results: {output_dir / 'comparison_results.json'}")
    logger.info(f"  - Summary CSV: {output_dir / 'summary.csv'}")
    logger.info(f"  - Log file: {log_file}")

    # Print key findings
    logger.info("\n" + "="*70)
    logger.info("KEY FINDINGS")
    logger.info("="*70 + "\n")

    for dataset_name, summary in results['comparison_summary'].items():
        logger.info(f"\n{dataset_name.upper()}:")
        logger.info(f"  Pre-trained embedding similarity: {summary['pretrained_embedding_mean']:.4f}")
        logger.info(f"  Fine-tuned embedding similarity: {summary['finetuned_embedding_mean']:.4f}")
        if 'finetuned_shap_mean' in summary:
            logger.info(f"  Fine-tuned SHAP similarity: {summary['finetuned_shap_mean']:.4f}")
        if 'embedding_shap_correlation' in summary:
            logger.info(f"  Embedding-SHAP correlation: {summary['embedding_shap_correlation']:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
