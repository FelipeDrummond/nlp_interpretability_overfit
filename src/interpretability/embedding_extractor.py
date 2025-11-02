"""
Embedding Extractor for NLP Models.

This module extracts CLS token embeddings from transformer models for similarity analysis.
It supports both fine-tuned and original pre-trained models.
"""

import logging
import numpy as np
import torch
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    DistilBertTokenizer, DistilBertModel,
    AutoTokenizer, AutoModel
)
from datasets import load_dataset

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract CLS token embeddings from transformer models.

    This class handles both fine-tuned models (from checkpoints) and
    original pre-trained models (from HuggingFace).
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize the embedding extractor.

        Args:
            device: Device to use for extraction ('cpu', 'mps', 'cuda')
        """
        self.device = torch.device(device)
        logger.info(f"EmbeddingExtractor initialized on device: {self.device}")

    def load_pretrained_model(self, model_name: str) -> Tuple[Any, Any]:
        """
        Load a pre-trained model and tokenizer from HuggingFace.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading pre-trained model: {model_name}")

        # Map model names to appropriate classes
        if 'distilbert' in model_name.lower():
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            model = DistilBertModel.from_pretrained(model_name)
        elif 'roberta' in model_name.lower():
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaModel.from_pretrained(model_name)
        elif 'bert' in model_name.lower():
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
        else:
            # Generic fallback
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

        model = model.to(self.device)
        model.eval()

        logger.info(f"Model loaded successfully: {model_name}")
        return model, tokenizer

    def load_finetuned_model(self,
                            model_path: Path,
                            model_name: str) -> Tuple[Any, Any]:
        """
        Load a fine-tuned model from checkpoint.

        Args:
            model_path: Path to the model checkpoint
            model_name: Original model name for loading tokenizer

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading fine-tuned model from: {model_path}")

        # Load tokenizer (same as pre-trained)
        if 'distilbert' in model_name.lower():
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            model = DistilBertModel.from_pretrained(model_name)
        elif 'roberta' in model_name.lower():
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaModel.from_pretrained(model_name)
        elif 'bert' in model_name.lower():
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

        # Load fine-tuned weights from pickle file
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Extract model state dict from checkpoint structure
        # The checkpoint has: model_data['model_state']['model_state_dict']
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
            if 'model_state_dict' in model_state:
                state_dict = model_state['model_state_dict']
            else:
                state_dict = model_state
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter out classifier layers (we only need the base model)
        base_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'model.' prefix if present
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            # Skip classifier/pooler layers
            if not any(skip in new_key for skip in ['classifier', 'qa_outputs', 'pre_classifier']):
                base_state_dict[new_key] = value

        # Load weights (strict=False to handle missing classifier layers)
        model.load_state_dict(base_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        logger.info(f"Fine-tuned model loaded successfully")
        return model, tokenizer

    def extract_cls_embeddings(self,
                               model: Any,
                               tokenizer: Any,
                               texts: List[str],
                               batch_size: int = 32,
                               max_length: int = 512) -> np.ndarray:
        """
        Extract CLS token embeddings from texts.

        Args:
            model: Transformer model
            tokenizer: Tokenizer
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            Numpy array of shape (n_samples, hidden_dim) containing CLS embeddings
        """
        logger.info(f"Extracting CLS embeddings for {len(texts)} texts")

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs)

                # Extract CLS token embeddings (first token)
                # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_dim)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)

                if (i // batch_size) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")

        return embeddings

    def load_dataset_texts(self,
                          dataset_name: str,
                          split: str = 'test',
                          max_samples: int = 1000) -> List[str]:
        """
        Load texts from a dataset.

        Args:
            dataset_name: Dataset name (imdb, yelp_polarity, amazon_polarity)
            split: Dataset split to use
            max_samples: Maximum number of samples to load

        Returns:
            List of texts
        """
        logger.info(f"Loading {max_samples} samples from {dataset_name} ({split})")

        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        # Sample texts
        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)

        # Extract texts - handle different column names
        if 'text' in dataset.column_names:
            texts = dataset['text']
        elif 'content' in dataset.column_names:
            texts = dataset['content']
        elif 'title' in dataset.column_names and 'content' in dataset.column_names:
            # Amazon polarity has title and content
            texts = [f"{title} {content}" for title, content in zip(dataset['title'], dataset['content'])]
        else:
            raise ValueError(f"Could not find text column in dataset {dataset_name}. Columns: {dataset.column_names}")

        logger.info(f"Loaded {len(texts)} texts from {dataset_name}")

        return texts

    def extract_and_save_embeddings(self,
                                   model_name: str,
                                   dataset_name: str,
                                   output_dir: Path,
                                   is_finetuned: bool = False,
                                   checkpoint_path: Optional[Path] = None,
                                   max_samples: int = 1000,
                                   batch_size: int = 32) -> Path:
        """
        Extract embeddings and save to disk.

        Args:
            model_name: Model identifier
            dataset_name: Dataset name
            output_dir: Directory to save embeddings
            is_finetuned: Whether to load fine-tuned model
            checkpoint_path: Path to checkpoint (required if is_finetuned=True)
            max_samples: Number of samples to extract
            batch_size: Batch size for processing

        Returns:
            Path to saved embeddings file
        """
        # Load model
        if is_finetuned:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required for fine-tuned models")
            model, tokenizer = self.load_finetuned_model(checkpoint_path, model_name)
            model_type = "finetuned"
        else:
            model, tokenizer = self.load_pretrained_model(model_name)
            model_type = "pretrained"

        # Load dataset texts
        texts = self.load_dataset_texts(dataset_name, max_samples=max_samples)

        # Extract embeddings
        embeddings = self.extract_cls_embeddings(
            model, tokenizer, texts, batch_size=batch_size
        )

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        model_name_safe = model_name.replace('/', '_')
        filename = f"{model_name_safe}_{dataset_name}_{model_type}_embeddings.npz"
        output_path = output_dir / filename

        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            model_name=model_name,
            dataset_name=dataset_name,
            model_type=model_type,
            n_samples=len(texts)
        )

        logger.info(f"Embeddings saved to: {output_path}")
        return output_path


def extract_embeddings_for_all_models(
    models_config: Dict[str, str],
    datasets: List[str],
    output_dir: Path,
    models_dir: Path,
    max_samples: int = 1000,
    batch_size: int = 32,
    device: str = 'cpu'
) -> Dict[str, Dict[str, Path]]:
    """
    Extract embeddings for all model-dataset combinations.

    Args:
        models_config: Dict mapping model names to HuggingFace identifiers
        datasets: List of dataset names
        output_dir: Output directory for embeddings
        models_dir: Directory containing fine-tuned model checkpoints
        max_samples: Number of samples per dataset
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        Dict mapping (model, dataset, type) to embedding file paths
    """
    extractor = EmbeddingExtractor(device=device)
    results = {}

    for model_name, hf_name in models_config.items():
        for dataset_name in datasets:
            # Extract pre-trained embeddings
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {model_name} on {dataset_name} (pre-trained)")
            logger.info(f"{'='*60}\n")

            pretrained_path = extractor.extract_and_save_embeddings(
                model_name=hf_name,
                dataset_name=dataset_name,
                output_dir=output_dir,
                is_finetuned=False,
                max_samples=max_samples,
                batch_size=batch_size
            )

            key_pretrained = (model_name, dataset_name, 'pretrained')
            results[key_pretrained] = pretrained_path

            # Extract fine-tuned embeddings
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {model_name} on {dataset_name} (fine-tuned)")
            logger.info(f"{'='*60}\n")

            # Find checkpoint path
            checkpoint_pattern = f"{model_name}_{dataset_name}_epoch*.pt"
            checkpoints = list(models_dir.glob(checkpoint_pattern))

            if checkpoints:
                # Use the latest checkpoint (highest epoch number)
                checkpoint_path = sorted(checkpoints)[-1]

                try:
                    finetuned_path = extractor.extract_and_save_embeddings(
                        model_name=hf_name,
                        dataset_name=dataset_name,
                        output_dir=output_dir,
                        is_finetuned=True,
                        checkpoint_path=checkpoint_path,
                        max_samples=max_samples,
                        batch_size=batch_size
                    )

                    key_finetuned = (model_name, dataset_name, 'finetuned')
                    results[key_finetuned] = finetuned_path

                except Exception as e:
                    logger.error(f"Failed to extract fine-tuned embeddings: {e}")
                    logger.error(f"Checkpoint: {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found for {model_name} on {dataset_name}")

    return results


if __name__ == "__main__":
    # Test embedding extraction
    print("EmbeddingExtractor module loaded successfully")
