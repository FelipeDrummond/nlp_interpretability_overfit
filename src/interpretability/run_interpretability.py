"""
CLI runner to compute interpretability (SHAP, metrics, visualizations) for saved models.

Usage examples (run from project root):
  python -m src.interpretability.run_interpretability --models all --datasets all
  python -m src.interpretability.run_interpretability --models bag-of-words-tfidf --datasets imdb
  python -m src.interpretability.run_interpretability --models bert-base-uncased --use-latest
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Ensure project root is on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.baseline import create_baseline_model  # type: ignore
from src.models.transformers import (  # type: ignore
    create_transformer_model,
    is_transformer_model,
)
from src.utils.config_loader import load_config  # type: ignore
from src.interpretability.shap_analyzer import create_shap_analyzer  # type: ignore
from src.interpretability.metrics import create_interpretability_metrics  # type: ignore
from src.interpretability.visualization import (  # type: ignore
    create_interpretability_visualizer,
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def as_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore


@dataclass
class CheckpointInfo:
    model: str
    dataset: str
    path: Path
    epoch: int


def find_checkpoint(
    weights_dir: Path, model: str, dataset: str, prefer_latest: bool = True
) -> Optional[CheckpointInfo]:
    """
    Find a saved checkpoint for a given model/dataset pair.
    Returns the highest-epoch file if multiple are present.
    """
    pattern_base = f"{model}_{dataset}_epoch"
    candidates = list(weights_dir.glob(f"{pattern_base}*.pkl")) + list(
        weights_dir.glob(f"{pattern_base}*.pt")
    )
    if not candidates:
        return None

    def extract_epoch(p: Path) -> int:
        stem = p.stem  # without suffix
        # Expect suffix like _epoch{num}
        try:
            epoch_str = stem.split("_epoch")[-1]
            return int(epoch_str)
        except Exception:
            return -1

    if prefer_latest:
        chosen = max(candidates, key=lambda p: (extract_epoch(p), p.stat().st_mtime))
    else:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)

    return CheckpointInfo(
        model=model, dataset=dataset, path=chosen, epoch=extract_epoch(chosen)
    )


def create_model_instance(model_name: str, cfg: DictConfig):
    if model_name in cfg.models.baseline_models:
        return create_baseline_model(model_name, as_dict(cfg.models.baseline_models[model_name]))
    if model_name in cfg.models.transformer_models or is_transformer_model(model_name):
        # Allow both explicit key and pattern-based transformer names
        config_dict = (
            as_dict(cfg.models.transformer_models[model_name])
            if model_name in cfg.models.transformer_models
            else as_dict(cfg.models.transformer_models.get("bert-base-uncased"))  # safe default
        )
        # Override model/tokenizer names to match requested model_name when needed
        config_dict["model_name"] = model_name
        config_dict["tokenizer_name"] = config_dict.get("tokenizer_name", model_name)
        return create_transformer_model(model_name, config_dict)
    raise ValueError(f"Unknown model: {model_name}")


def load_text_data(cfg: DictConfig, dataset: str, split: str = "test") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    processed_dir = Path(cfg.paths.processed_data_dir)
    text_col = cfg.data.text_column
    label_col = cfg.data.label_column
    file_path = processed_dir / f"{dataset}_{split}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found: {file_path}")
    df = pd.read_csv(file_path)
    texts = df[text_col].values
    labels = df[label_col].values if label_col in df.columns else None
    return texts, labels


def flatten_metrics_for_plot(metrics_all: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    faith = metrics_all.get("faithfulness", {})
    spars = metrics_all.get("sparsity", {})
    intuit = metrics_all.get("intuitiveness", {})
    if "faithfulness_score" in faith:
        out["faithfulness_score"] = float(faith["faithfulness_score"])  # type: ignore
    if "sparsity_score" in spars:
        out["sparsity_score"] = float(spars["sparsity_score"])  # type: ignore
    if "intuitiveness_score" in intuit:
        out["intuitiveness_score"] = float(intuit["intuitiveness_score"])  # type: ignore
    return out


def save_metrics_json(
    metrics_all: Dict[str, Any],
    results_dir: Path,
    model: str,
    dataset: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_{dataset}_{timestamp}_interpretability.json"
    with open(out_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    return out_path


def run_for_model_dataset(
    cfg: DictConfig,
    model_name: str,
    dataset_name: str,
    weights_dir: Path,
    prefer_latest: bool,
    max_samples_override: Optional[int],
    background_samples_override: Optional[int],
    skip_plots: bool,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"Running interpretability for model={model_name} dataset={dataset_name}")

    ckpt = find_checkpoint(weights_dir, model_name, dataset_name, prefer_latest=prefer_latest)
    if ckpt is None:
        logger.warning(
            f"No checkpoint found for {model_name} on {dataset_name} in {weights_dir}. Skipping."
        )
        return
    logger.info(f"Using checkpoint: {ckpt.path} (epoch {ckpt.epoch if ckpt.epoch>=0 else 'unknown'})")

    # Instantiate and load model
    model = create_model_instance(model_name, cfg)
    model.load_model(ckpt.path)

    # Load data (test split)
    texts, _ = load_text_data(cfg, dataset_name, split="test")

    # SHAP analysis
    shap_cfg = as_dict(cfg.interpretability.shap)
    analyzer = create_shap_analyzer(model, shap_cfg)
    shap_results = analyzer.analyze(
        texts,
        max_samples=max_samples_override,
        background_samples=background_samples_override,
    )

    # Save SHAP artifacts
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    analyzer.save_results(
        shap_results,
        output_dir=artifacts_dir,
        model_name=model_name,
        dataset_name=dataset_name,
    )

    # Metrics
    metrics_cfg = as_dict(cfg.interpretability.metrics)
    metrics = create_interpretability_metrics(metrics_cfg)
    metrics_all = metrics.compute_all_metrics(
        model=model,
        inputs=texts,
        shap_values=shap_results["shap_values"],
        feature_names=shap_results.get("feature_names"),
        samples_data=shap_results.get("samples"),
        tokenizer=getattr(model, 'tokenizer', None),
    )
    metrics_json_path = save_metrics_json(metrics_all, Path(cfg.paths.results_dir), model_name, dataset_name)
    logger.info(f"Saved interpretability metrics to {metrics_json_path}")

    # Visualizations
    if not skip_plots:
        visualizer = create_interpretability_visualizer(as_dict(cfg.interpretability))
        visualizer.create_all_visualizations(
            shap_results=shap_results,
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=cfg.paths.results_dir,
            metrics_results=metrics_all,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interpretability on saved models")
    parser.add_argument(
        "--config-name",
        default="config",
        help="Main config filename without extension (default: config)",
    )
    parser.add_argument(
        "--models",
        default="selected",  # use models.model_selection.selected_models by default
        help="Comma-separated model names, 'all', or 'selected' (from config)",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names or 'all' (default: all from config)",
    )
    parser.add_argument(
        "--weights-dir",
        default=None,
        help="Directory with saved model weights (default: cfg.paths.models_dir)",
    )
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help="Prefer highest epoch checkpoint if multiple exist",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override SHAP max_samples",
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        default=None,
        help="Override SHAP background_samples",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config (Hydra-backed loader)
    cfg = load_config(config_name=args.config_name, config_dir=str(PROJECT_ROOT))

    # Determine models
    if args.models.lower() == "all":
        models = list(cfg.models.transformer_models.keys()) + list(cfg.models.baseline_models.keys())
    elif args.models.lower() == "selected":
        models = list(cfg.models.model_selection.selected_models)
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Determine datasets
    if args.datasets.lower() == "all":
        datasets = list(cfg.data.datasets.keys())
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Weights directory
    weights_dir = Path(args.weights_dir) if args.weights_dir else Path(cfg.paths.models_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Models: %s", models)
    logger.info("Datasets: %s", datasets)
    logger.info("Weights dir: %s", weights_dir)

    for model_name in models:
        for dataset_name in datasets:
            try:
                run_for_model_dataset(
                    cfg=cfg,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    weights_dir=weights_dir,
                    prefer_latest=args.use_latest,
                    max_samples_override=args.max_samples,
                    background_samples_override=args.background_samples,
                    skip_plots=args.skip_plots,
                )
            except Exception as e:
                logger.exception(
                    "Interpretability failed for model=%s dataset=%s: %s",
                    model_name,
                    dataset_name,
                    e,
                )


if __name__ == "__main__":
    main()

