#!/usr/bin/env python3
"""
Training and Interpretability Analysis Script

This script:
1. Trains all models except Llama on all datasets
2. Runs interpretability analysis for all trained models
3. Generates comprehensive reports and visualizations

Usage:
    python run_training_and_interpretability.py
"""

import sys
import os
import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import setup_reproducibility
from src.interpretability.shap_analyzer import create_shap_analyzer
from src.interpretability.metrics import create_interpretability_metrics
from src.interpretability.visualization import create_interpretability_visualizer
from src.models.baseline import create_baseline_model
from src.models.transformers import create_transformer_model, is_transformer_model

# Setup logging
logger = setup_logging(log_level="INFO")
logger.info("=" * 80)
logger.info("NLP Interpretability Training and Analysis Pipeline")
logger.info("=" * 80)

# Configuration
MODELS_TO_TRAIN = [
    "bag-of-words-tfidf",
    "bert-base-uncased", 
    "roberta-base",
    "distilbert-base-uncased"
    # Note: Excluding meta-llama/Llama-3.2-1B as requested
]

ALL_MODELS = [
    "bag-of-words-tfidf",
    "bert-base-uncased",
    "roberta-base", 
    "distilbert-base-uncased",
    "meta-llama/Llama-3.2-1B"  # Include for interpretability analysis
]

DATASETS = ["imdb", "amazon_polarity", "yelp_polarity"]

# Results directories
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
INTERPRETABILITY_DIR = RESULTS_DIR / "interpretability"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

# Create directories
for dir_path in [RESULTS_DIR, MODELS_DIR, INTERPRETABILITY_DIR, FIGURES_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def run_training_phase():
    """Run training for all models except Llama."""
    logger.info("🚀 Starting Training Phase")
    logger.info("=" * 50)
    
    successful_models = []
    failed_models = []
    
    for i, model in enumerate(MODELS_TO_TRAIN, 1):
        logger.info(f"\n📊 Training Model {i}/{len(MODELS_TO_TRAIN)}: {model}")
        logger.info("-" * 40)
        
        try:
            # Run training using the existing train.py script
            cmd = ["python", "train.py", f"global.model={model}"]
            
            # Create log file for this model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = LOGS_DIR / f"training_{model}_{timestamp}.log"
            
            logger.info(f"Running command: {' '.join(cmd)}")
            logger.info(f"Log file: {log_file}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, 
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=Path(__file__).parent
                )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully trained {model}")
                successful_models.append(model)
            else:
                logger.error(f"❌ Failed to train {model} (exit code: {result.returncode})")
                failed_models.append(model)
                
        except Exception as e:
            logger.error(f"❌ Error training {model}: {e}")
            failed_models.append(model)
    
    logger.info(f"\n📈 Training Phase Summary:")
    logger.info(f"  ✅ Successful: {len(successful_models)} models")
    logger.info(f"  ❌ Failed: {len(failed_models)} models")
    
    if failed_models:
        logger.warning(f"  Failed models: {', '.join(failed_models)}")
    
    return successful_models, failed_models


def find_trained_models():
    """Find all trained models in the results directory."""
    logger.info("\n🔍 Finding Trained Models")
    logger.info("-" * 30)
    
    trained_models = {}
    
    for model_name in ALL_MODELS:
        # Look for model files
        model_files = list(MODELS_DIR.glob(f"{model_name}_*.pkl")) + list(MODELS_DIR.glob(f"{model_name}_*.pt"))
        
        if model_files:
            # Group by dataset
            datasets = set()
            for file in model_files:
                # Extract dataset name from filename
                parts = file.stem.split('_')
                if len(parts) >= 3:  # model_dataset_epoch
                    dataset = parts[1]
                    datasets.add(dataset)
            
            trained_models[model_name] = list(datasets)
            logger.info(f"  ✅ {model_name}: {len(datasets)} datasets ({', '.join(datasets)})")
        else:
            logger.info(f"  ❌ {model_name}: No trained models found")
    
    return trained_models


def load_model_and_data(model_name: str, dataset_name: str):
    """Load a trained model and corresponding data."""
    logger.info(f"Loading {model_name} trained on {dataset_name}...")
    
    # Find model file
    model_files = list(MODELS_DIR.glob(f"{model_name}_{dataset_name}_*.pkl")) + list(MODELS_DIR.glob(f"{model_name}_{dataset_name}_*.pt"))
    
    if not model_files:
        raise FileNotFoundError(f"No model file found for {model_name} on {dataset_name}")
    
    model_file = model_files[0]  # Take the first (most recent) file
    
    # Load model
    if is_transformer_model(model_name):
        # For transformer models, we need to recreate and load
        from src.models.transformers import create_transformer_model
        # This is a simplified approach - in practice, you'd need to save/load the full model state
        logger.warning(f"Transformer model loading not fully implemented for {model_name}")
        return None, None
    else:
        # For baseline models, load directly
        import joblib
        model = joblib.load(model_file)
        logger.info(f"✅ Loaded {model_name} from {model_file}")
    
    # Load test data
    data_file = Path("data/processed") / f"{dataset_name}_test.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Test data not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Create subset for interpretability analysis
    test_size = min(100, len(df))  # Use max 100 samples for analysis
    df_sample = df.sample(n=test_size, random_state=42)
    
    texts = df_sample['text'].tolist()
    labels = df_sample['label'].values
    
    logger.info(f"✅ Loaded {len(texts)} test samples from {dataset_name}")
    
    return model, (texts, labels)


def run_interpretability_analysis(trained_models: Dict[str, List[str]]):
    """Run interpretability analysis for all trained models."""
    logger.info("\n🔬 Starting Interpretability Analysis Phase")
    logger.info("=" * 50)
    
    # Configuration for interpretability analysis
    shap_config = {
        'max_samples': 50,
        'background_samples': 25,
        'explainer_type': 'auto',
        'device': 'cpu'
    }
    
    metrics_config = {
        'faithfulness': {
            'enabled': True,
            'method': 'removal',
            'num_samples': 25
        },
        'consistency': {
            'enabled': True,
            'similarity_threshold': 0.7,
            'num_pairs': 50
        },
        'sparsity': {
            'enabled': True,
            'threshold': 0.01,
            'method': 'gini'
        },
        'intuitiveness': {
            'enabled': False  # Requires human annotations
        }
    }
    
    visualization_config = {
        'visualization': {
            'max_features_display': 20,
            'figure_dpi': 300,
            'figure_format': 'png',
            'color_scheme': 'viridis',
            'save_plots': True
        }
    }
    
    # Initialize analyzers
    metrics_analyzer = create_interpretability_metrics(metrics_config)
    visualizer = create_interpretability_visualizer(visualization_config)
    
    all_results = {}
    
    for model_name, datasets in trained_models.items():
        logger.info(f"\n📊 Analyzing {model_name}")
        logger.info("-" * 30)
        
        model_results = {}
        
        for dataset_name in datasets:
            logger.info(f"  🔍 Dataset: {dataset_name}")
            
            try:
                # Load model and data
                model, (texts, labels) = load_model_and_data(model_name, dataset_name)
                
                if model is None:
                    logger.warning(f"    ⚠️  Skipping {model_name} on {dataset_name} (model loading not implemented)")
                    continue
                
                # Create SHAP analyzer
                shap_analyzer = create_shap_analyzer(model, shap_config)
                
                # Run SHAP analysis
                logger.info(f"    🔬 Running SHAP analysis...")
                shap_results = shap_analyzer.analyze(texts)
                
                # Compute interpretability metrics
                logger.info(f"    📈 Computing interpretability metrics...")
                metrics_results = metrics_analyzer.compute_all_metrics(
                    model, texts, shap_results['shap_values'],
                    feature_names=shap_results['feature_names']
                )
                
                # Create visualizations
                logger.info(f"    🎨 Creating visualizations...")
                output_dir = INTERPRETABILITY_DIR / f"{model_name}_{dataset_name}"
                output_dir.mkdir(exist_ok=True)
                
                figures = visualizer.create_all_visualizations(
                    shap_results, model_name, dataset_name, output_dir, metrics_results
                )
                
                # Save results
                results = {
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'shap_results': shap_results,
                    'metrics_results': metrics_results,
                    'n_samples': len(texts),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save to file
                results_file = output_dir / "analysis_results.json"
                with open(results_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    json_results = {}
                    for key, value in results.items():
                        if isinstance(value, dict):
                            json_results[key] = {}
                            for k, v in value.items():
                                if isinstance(v, np.ndarray):
                                    json_results[key][k] = v.tolist()
                                else:
                                    json_results[key][k] = v
                        elif isinstance(value, np.ndarray):
                            json_results[key] = value.tolist()
                        else:
                            json_results[key] = value
                    
                    json.dump(json_results, f, indent=2)
                
                model_results[dataset_name] = results
                logger.info(f"    ✅ Analysis completed for {model_name} on {dataset_name}")
                
            except Exception as e:
                logger.error(f"    ❌ Error analyzing {model_name} on {dataset_name}: {e}")
                continue
        
        all_results[model_name] = model_results
    
    return all_results


def generate_summary_report(all_results: Dict[str, Dict[str, Any]]):
    """Generate a summary report of all interpretability results."""
    logger.info("\n📋 Generating Summary Report")
    logger.info("-" * 30)
    
    # Create summary data
    summary_data = []
    
    for model_name, model_results in all_results.items():
        for dataset_name, results in model_results.items():
            if 'metrics_results' in results:
                metrics = results['metrics_results']
                
                row = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'N_Samples': results.get('n_samples', 0),
                    'Timestamp': results.get('timestamp', 'Unknown')
                }
                
                # Add metrics if available
                if 'faithfulness' in metrics and 'error' not in metrics['faithfulness']:
                    row['Faithfulness'] = metrics['faithfulness'].get('score', 'N/A')
                else:
                    row['Faithfulness'] = 'Error'
                
                if 'sparsity' in metrics and 'error' not in metrics['sparsity']:
                    row['Sparsity'] = metrics['sparsity'].get('score', 'N/A')
                else:
                    row['Sparsity'] = 'Error'
                
                if 'consistency' in metrics and 'error' not in metrics['consistency']:
                    row['Consistency'] = metrics['consistency'].get('score', 'N/A')
                else:
                    row['Consistency'] = 'Error'
                
                summary_data.append(row)
    
    # Create DataFrame and save
    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_file = RESULTS_DIR / "interpretability_summary.csv"
        df.to_csv(summary_file, index=False)
        
        logger.info(f"✅ Summary report saved to {summary_file}")
        logger.info(f"📊 Analyzed {len(df)} model-dataset combinations")
        
        # Print summary
        logger.info("\n📈 Summary Statistics:")
        logger.info(df.to_string(index=False))
    else:
        logger.warning("⚠️  No results to summarize")


def main():
    """Main execution function."""
    start_time = datetime.now()
    
    try:
        # Setup reproducibility
        setup_reproducibility({'seed': 42})
        
        # Phase 1: Training
        logger.info("🚀 Starting NLP Interpretability Pipeline")
        logger.info(f"📅 Start time: {start_time}")
        
        successful_models, failed_models = run_training_phase()
        
        if not successful_models:
            logger.error("❌ No models were successfully trained. Cannot proceed with interpretability analysis.")
            return False
        
        # Phase 2: Find trained models
        trained_models = find_trained_models()
        
        if not trained_models:
            logger.error("❌ No trained models found. Cannot proceed with interpretability analysis.")
            return False
        
        # Phase 3: Interpretability Analysis
        all_results = run_interpretability_analysis(trained_models)
        
        # Phase 4: Generate Summary
        generate_summary_report(all_results)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n🎉 Pipeline Completed Successfully!")
        logger.info("=" * 50)
        logger.info(f"⏱️  Total duration: {duration}")
        logger.info(f"📊 Models trained: {len(successful_models)}")
        logger.info(f"🔬 Models analyzed: {len(all_results)}")
        logger.info(f"📁 Results saved to: {RESULTS_DIR}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
