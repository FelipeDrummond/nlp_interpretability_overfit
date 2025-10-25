import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def show_sample_from_processed(dataset: str, split: str, index: int) -> None:
    """Print a sample from the processed CSV split by row index."""
    csv_path = Path('data/processed') / f"{dataset}_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} with shape {df.shape}\n")

    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of bounds for {csv_path} with {len(df)} rows")

    row = df.iloc[index]
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else None

    print(f"[Processed] {dataset} {split} row {index}")
    print(f"Text: {row[text_col]}")
    if label_col is not None:
        print(f"Label: {row[label_col]}")


def find_latest_shap_artifact(artifacts_dir: Path, model: str, dataset: str) -> Optional[Path]:
    """Return latest SHAP results npz for model/dataset by modified time."""
    pattern = f"{model}_{dataset}_*_shap_results.npz"
    candidates = sorted(artifacts_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def show_sample_from_artifact(model: str, dataset: str, index: int, artifacts_root: Optional[Path] = None) -> None:
    """Print the sample text (and top tokens if available) from saved SHAP artifact."""
    artifacts_dir = artifacts_root or Path('results/artifacts')
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    artifact = find_latest_shap_artifact(artifacts_dir, model, dataset)
    if artifact is None:
        raise FileNotFoundError(f"No SHAP artifact found for {model} on {dataset} in {artifacts_dir}")

    data = np.load(artifact, allow_pickle=True)
    texts = data['texts']
    if index < 0 or index >= len(texts):
        raise IndexError(f"Index {index} out of bounds for artifact with {len(texts)} samples: {artifact}")

    print(f"Loaded artifact: {artifact}")
    print(f"Model/Dataset: {model} / {dataset}")
    print(f"Samples: {len(texts)} | Features: {int(data['n_features'])}\n")

    sample_text = texts[index]
    # Handle object arrays from npz
    if isinstance(sample_text, (np.ndarray, list)):
        try:
            sample_text = sample_text.item()  # type: ignore[attr-defined]
        except Exception:
            sample_text = str(sample_text)

    print(f"[Artifact] Sample {index} text:\n{sample_text}\n")

    # If token-level entries are present, print top tokens by |SHAP| for quick verification
    if 'samples' in data.files:
        samples = data['samples']
        try:
            sample_entry = samples[index].item() if isinstance(samples[index], np.ndarray) else samples[index]
        except Exception:
            sample_entry = None
        if isinstance(sample_entry, dict) and 'tokens' in sample_entry and 'shap_values' in sample_entry:
            tokens = np.array(sample_entry['tokens'], dtype=object)
            shap_vals = np.array(sample_entry['shap_values'], dtype=float)
            if shap_vals.size > 0 and tokens.size == shap_vals.size:
                top_k = min(10, shap_vals.size)
                top_idx = np.argsort(np.abs(shap_vals))[-top_k:][::-1]
                print("Top tokens by |SHAP|:")
                for j in top_idx:
                    tok = tokens[j]
                    val = shap_vals[j]
                    print(f"  {tok!r}: {val:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect sample by index from processed data or SHAP artifact.")
    parser.add_argument('--mode', choices=['artifact', 'processed'], default='artifact', help='Source of the sample')
    parser.add_argument('--dataset', default='amazon_polarity', help='Dataset name (e.g., imdb, yelp_polarity, amazon_polarity)')
    parser.add_argument('--split', default='test', help='Data split for processed mode (train/val/test)')
    parser.add_argument('--model', default=None, help='Model name for artifact mode (e.g., bert-base-uncased)')
    parser.add_argument('--index', type=int, default=16, help='Sample index to inspect')
    args = parser.parse_args()

    if args.mode == 'artifact':
        if not args.model:
            raise SystemExit("--model is required when --mode=artifact")
        show_sample_from_artifact(model=args.model, dataset=args.dataset, index=args.index)
    else:
        show_sample_from_processed(dataset=args.dataset, split=args.split, index=args.index)


if __name__ == '__main__':
    main()
