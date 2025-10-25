#!/usr/bin/env python3
"""
Custom SHAP Analysis Script for Specific Phrases

This script loads a trained BERT model on Amazon polarity dataset and generates
SHAP analysis for custom phrases, focusing on sentiment-related words.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.transformers import create_transformer_model
from src.interpretability.shap_analyzer import create_shap_analyzer
from src.interpretability.visualization import create_interpretability_visualizer
from src.utils.config_loader import load_config
from omegaconf import DictConfig, OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def as_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert DictConfig to regular dict."""
    return OmegaConf.to_container(cfg, resolve=True)


def load_trained_model(model_name: str, dataset_name: str, cfg: DictConfig):
    """Load a trained model from saved checkpoint."""
    logger.info(f"Loading trained {model_name} model for {dataset_name}")
    
    # Create model instance
    model = create_transformer_model(model_name, as_dict(cfg.models.transformer_models[model_name]))
    
    # Find the checkpoint
    models_dir = Path(cfg.paths.models_dir)
    checkpoint_pattern = f"{model_name}_{dataset_name}_epoch*.pt"
    checkpoints = list(models_dir.glob(checkpoint_pattern))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found for {model_name} on {dataset_name}")
    
    # Use the latest checkpoint (highest epoch)
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using checkpoint: {latest_checkpoint}")
    
    # Load the model
    model.load_model(latest_checkpoint)
    logger.info("Model loaded successfully")
    
    return model


def analyze_custom_phrases(model, phrases: List[str], output_dir: Path):
    """Generate SHAP analysis for custom phrases."""
    logger.info(f"Analyzing {len(phrases)} custom phrases")
    
    # Create SHAP analyzer
    shap_config = {
        'max_samples': len(phrases),
        'background_samples': 10,  # Small background for custom analysis
        'explainer_type': 'auto',
        'device': 'cpu'
    }
    
    analyzer = create_shap_analyzer(model, shap_config)
    
    # Generate SHAP values
    shap_results = analyzer.analyze(phrases)
    
    logger.info(f"SHAP analysis completed. Shape: {shap_results['shap_values'].shape}")
    
    # Create visualizations
    visualizer = create_interpretability_visualizer({
        'visualization': {
            'max_features_display': 20,
            'figure_dpi': 300,
            'figure_format': 'png',
            'color_scheme': 'viridis',
            'save_plots': True,
            'max_tokens_per_text': 30
        }
    })
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual force plots for each phrase
    figures = {}
    for i, phrase in enumerate(phrases):
        logger.info(f"Creating visualizations for phrase {i+1}: '{phrase[:50]}...'")
        
        try:
            # Force plot
            fig_force = visualizer.create_force_plot(
                shap_results['shap_values'], 
                shap_results['feature_names'],
                sample_idx=i,
                title=f"SHAP Force Plot for distilbert-base-uncased - Phrase {i+1}",
                output_path=output_dir
            )
            figures[f'force_plot_{i}'] = fig_force
            
            # Waterfall plot
            fig_waterfall = visualizer.create_waterfall_plot(
                shap_results['shap_values'],
                shap_results['feature_names'], 
                sample_idx=i,
                title=f"SHAP Waterfall for distilbert-base-uncased - Phrase {i+1}",
                output_path=output_dir
            )
            figures[f'waterfall_plot_{i}'] = fig_waterfall
            
        except Exception as e:
            logger.error(f"Error creating visualizations for phrase {i+1}: {e}")
    
    # Create summary plot
    try:
        fig_summary = visualizer.create_summary_plot(
            shap_results['shap_values'],
            shap_results['feature_names'],
            title="SHAP Summary for distilbert-base-uncased - Custom Phrases Analysis",
            output_path=output_dir
        )
        figures['summary'] = fig_summary
    except Exception as e:
        logger.error(f"Error creating summary plot: {e}")
    
    # Create heatmap
    try:
        fig_heatmap = visualizer.create_feature_importance_heatmap(
            shap_results['shap_values'],
            shap_results['feature_names'],
            title="Feature Importance Heatmap for distilbert-base-uncased - Custom Phrases",
            output_path=output_dir
        )
        figures['heatmap'] = fig_heatmap
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
    
    return shap_results, figures


def print_phrase_analysis(phrases: List[str], shap_results: Dict[str, Any]):
    """Print detailed analysis for each phrase."""
    logger.info("=" * 80)
    logger.info("DETAILED PHRASE ANALYSIS")
    logger.info("=" * 80)
    
    shap_values = shap_results['shap_values']
    feature_names = shap_results['feature_names']
    
    for i, phrase in enumerate(phrases):
        logger.info(f"\nPhrase {i+1}: '{phrase}'")
        logger.info("-" * 60)
        
        # Get SHAP values for this phrase
        if len(shap_values.shape) > 2:
            phrase_shap = shap_values[i, :, 1] if shap_values.shape[2] > 1 else shap_values[i, :, 0]
        else:
            phrase_shap = shap_values[i, :]
        
        # Get top positive and negative features
        positive_indices = np.where(phrase_shap > 0)[0]
        negative_indices = np.where(phrase_shap < 0)[0]
        
        if len(positive_indices) > 0:
            top_positive = positive_indices[np.argsort(phrase_shap[positive_indices])[-5:][::-1]]
            logger.info("Top positive features:")
            for idx in top_positive:
                logger.info(f"  {feature_names[idx]}: {phrase_shap[idx]:.4f}")
        
        if len(negative_indices) > 0:
            top_negative = negative_indices[np.argsort(phrase_shap[negative_indices])[:5]]
            logger.info("Top negative features:")
            for idx in top_negative:
                logger.info(f"  {feature_names[idx]}: {phrase_shap[idx]:.4f}")
        
        # Overall prediction
        total_shap = np.sum(phrase_shap)
        logger.info(f"Total SHAP value: {total_shap:.4f}")
        logger.info(f"Predicted sentiment: {'Positive' if total_shap > 0 else 'Negative'}")


def main():
    parser = argparse.ArgumentParser(description="Custom SHAP analysis for specific phrases")
    parser.add_argument(
        "--model", 
        default="roberta-base",
        help="Model name (default: roberta-base)"
    )
    parser.add_argument(
        "--dataset",
        default="amazon_polarity", 
        help="Dataset name (default: amazon_polarity)"
    )
    parser.add_argument(
        "--phrases",
        nargs="+",
        default=[
            "i was disappointed with this product",
            "i was not disappointed with this model"
        ],
        help="Phrases to analyze"
    )
    parser.add_argument(
        "--output-dir",
        default="results/custom_shap_analysis",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Config file name"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        cfg = load_config(config_name=args.config_name, config_dir=str(PROJECT_ROOT))
        
        # Load trained model
        model = load_trained_model(args.model, args.dataset, cfg)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        
        # Analyze phrases
        shap_results, figures = analyze_custom_phrases(model, args.phrases, output_dir)
        
        # Print detailed analysis
        print_phrase_analysis(args.phrases, shap_results)
        
        # Save SHAP results
        analyzer = create_shap_analyzer(model, {})
        analyzer.save_results(
            shap_results,
            output_dir=output_dir,
            model_name=args.model,
            dataset_name=f"{args.dataset}_custom"
        )
        
        logger.info(f"Analysis completed! Results saved to: {output_dir}")
        logger.info(f"Generated {len(figures)} visualizations")
        
        # List generated files
        logger.info("Generated files:")
        for file_path in sorted(output_dir.glob("*.png")):
            logger.info(f"  {file_path.name}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
