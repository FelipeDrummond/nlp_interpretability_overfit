#!/usr/bin/env python3
"""
Interpretability-Only Runner

Runs SHAP analysis, interpretability metrics, and visualizations for already-trained
models across configured datasets. No training is performed.

Usage:
  python run_interpretability_only.py --model bag-of-words-tfidf
  python run_interpretability_only.py --model roberta-base
  python run_interpretability_only.py            # processes all supported models
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import setup_reproducibility
from src.interpretability.shap_analyzer import create_shap_analyzer
from src.interpretability.metrics import create_interpretability_metrics
from src.interpretability.visualization import create_interpretability_visualizer
from src.models.baseline import BagOfWordsModel
from src.models.transformers import is_transformer_model, create_transformer_model


SUPPORTED_MODELS = [
    "bag-of-words-tfidf",
    # MultiBERTs models - comparing different initialization seeds
    "multiberts-seed_0",
    "multiberts-seed_1",
    "multiberts-seed_2",
    "multiberts-seed_3",
    "multiberts-seed_4",
    "multiberts-seed_5",
    "multiberts-seed_6",
    "multiberts-seed_7",
    "multiberts-seed_8",
    "multiberts-seed_9",
    "multiberts-seed_10",
    "multiberts-seed_11",
    "multiberts-seed_12",
    "multiberts-seed_13",
    "multiberts-seed_14",
    "multiberts-seed_15",
    "multiberts-seed_16",
    "multiberts-seed_17",
    "multiberts-seed_18",
    "multiberts-seed_19",
    "multiberts-seed_20",
    "multiberts-seed_21",
    "multiberts-seed_22",
    "multiberts-seed_23",
    "multiberts-seed_24",
]


def load_config(config_path: Path) -> Any:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def list_trained_datasets(models_dir: Path, model_name: str) -> List[str]:
    datasets = set()
    for p in list(models_dir.glob(f"{model_name}_*.pkl")) + list(models_dir.glob(f"{model_name}_*.pt")):
        parts = p.stem.split("_")
        # filename pattern: {model}_{dataset_parts...}_epochN
        if len(parts) >= 3:
            dataset = "_".join(parts[1:-1])
            datasets.add(dataset)
    return sorted(datasets)


def find_model_file(models_dir: Path, model_name: str, dataset_name: str) -> Optional[Path]:
    candidates = list(models_dir.glob(f"{model_name}_{dataset_name}_*.pkl")) + list(models_dir.glob(f"{model_name}_{dataset_name}_*.pt"))
    return candidates[0] if candidates else None


def load_baseline_model(model_file: Path, model_cfg: Dict[str, Any]) -> BagOfWordsModel:
    model = BagOfWordsModel(model_cfg)
    model.load_model(model_file)
    return model


def load_test_data(processed_dir: Path, dataset_name: str, split: str = "val") -> Tuple[List[str], np.ndarray]:
    """
    Load data for interpretability analysis.
    
    Args:
        processed_dir: Directory containing processed data
        dataset_name: Name of the dataset
        split: Which split to load ('val' or 'test'). Default is 'val' for interpretability.
    
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(processed_dir / f"{dataset_name}_{split}.csv")
    texts = df["text"].tolist()
    labels = df["label"].values
    # Subsample for efficiency (match pipeline defaults if large)
    if len(texts) > 100:
        df_sample = df.sample(n=100, random_state=42)
        texts = df_sample["text"].tolist()
        labels = df_sample["label"].values
    return texts, labels


def run_interpretability_for(model_name: str, cfg: Any, logger: logging.Logger) -> None:
    results_dir = Path(cfg.paths.results_dir)
    models_dir = Path(cfg.paths.models_dir)
    processed_dir = Path(cfg.paths.processed_data_dir)

    # Discover datasets with trained checkpoints
    datasets = list_trained_datasets(models_dir, model_name)
    if not datasets:
        logger.warning(f"No trained checkpoints found for {model_name} in {models_dir}")
        return

    # Initialize analyzers from config
    shap_cfg = OmegaConf.to_container(cfg.interpretability.shap, resolve=True)
    metrics_cfg = OmegaConf.to_container(cfg.interpretability.metrics, resolve=True)
    viz_cfg = {"visualization": OmegaConf.to_container(cfg.interpretability.visualization, resolve=True)}

    metrics_analyzer = create_interpretability_metrics(metrics_cfg)
    visualizer = create_interpretability_visualizer(viz_cfg)

    for dataset_name in datasets:
        logger.info("-" * 60)
        logger.info(f"Analyzing {model_name} on {dataset_name}")

        model_file = find_model_file(models_dir, model_name, dataset_name)
        if not model_file:
            logger.error(f"Model file not found for {model_name} on {dataset_name}")
            continue

        # Load model
        if is_transformer_model(model_name):
            # Create transformer instance from config and load weights
            if model_name in cfg.models.transformer_models:
                tcfg = OmegaConf.to_container(cfg.models.transformer_models[model_name], resolve=True)
            else:
                # default to a sane base and override names
                tcfg = OmegaConf.to_container(cfg.models.transformer_models["bert-base-uncased"], resolve=True)
                tcfg["model_name"] = model_name
                tcfg["tokenizer_name"] = model_name
            model = create_transformer_model(model_name, tcfg)
            model.load_model(model_file)
        else:
            model_cfg = OmegaConf.to_container(cfg.models.baseline_models.get(model_name, {}), resolve=True)
            if not model_cfg:
                model_cfg = OmegaConf.to_container(cfg.models.baseline_models["bag-of-words-tfidf"], resolve=True)
            model = load_baseline_model(model_file, model_cfg)

        # Load validation data for interpretability analysis
        try:
            texts, labels = load_test_data(processed_dir, dataset_name, split="val")
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        # Run SHAP
        try:
            shap_analyzer = create_shap_analyzer(model, shap_cfg)
            shap_results = shap_analyzer.analyze(texts)
        except Exception as e:
            logger.error(f"Failed SHAP analysis for {model_name} on {dataset_name}: {e}")
            continue

        # Compute metrics
        try:
            metrics_results = metrics_analyzer.compute_all_metrics(
                model, texts, shap_results["shap_values"], feature_names=shap_results["feature_names"],
            )
        except Exception as e:
            logger.error(f"Failed metrics computation for {model_name} on {dataset_name}: {e}")
            metrics_results = {}

        # Visualizations
        out_dir = results_dir / "interpretability" / f"{model_name}_{dataset_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            visualizer.create_all_visualizations(
                shap_results, model_name, dataset_name, out_dir, metrics_results
            )
        except Exception as e:
            logger.error(f"Failed to create visualizations for {model_name} on {dataset_name}: {e}")

        # Save compact JSON summary
        try:
            summary = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "n_samples": int(shap_results.get("n_samples", len(texts))),
                "n_features": int(shap_results.get("n_features", 0)),
                "metrics": metrics_results,
                "timestamp": datetime.now().isoformat(),
            }
            with open(out_dir / "analysis_results.json", "w") as f:
                import json
                json.dump(summary, f, indent=2)
            logger.info(f"Saved interpretability summary to {out_dir / 'analysis_results.json'}")
        except Exception as e:
            logger.warning(f"Could not save analysis summary: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run interpretability only for trained models")
    parser.add_argument("--model", type=str, default=None, help="Model name to analyze (defaults to all)")
    args = parser.parse_args()

    setup_reproducibility({"seed": 42})
    logger = setup_logging(log_level="INFO")

    cfg = load_config(Path("config.yaml"))

    models: List[str]
    if args.model:
        models = [args.model]
    else:
        models = SUPPORTED_MODELS

    logger.info("=" * 80)
    logger.info("Interpretability-Only Runner")
    logger.info("=" * 80)

    for model_name in models:
        logger.info("")
        logger.info(f"Processing model: {model_name}")
        run_interpretability_for(model_name, cfg, logger)

    logger.info("\nAll requested interpretability runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


