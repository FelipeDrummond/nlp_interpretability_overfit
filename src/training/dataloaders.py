"""
Utility helpers for building DataLoader objects used across models.

These helpers centralize how we convert raw text/label arrays into PyTorch
datasets and batches, so that transformer and baseline models can share the
same batching pipeline when appropriate.
"""

from typing import Iterable, List, Optional, Tuple, Dict, Any, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextClassificationDataset(Dataset):
    """Simple dataset that stores texts and labels for classification tasks."""

    def __init__(self, texts: Iterable[str], labels: Union[Iterable[int], np.ndarray]):
        self.texts: List[str] = list(texts)
        self.labels: np.ndarray = np.asarray(labels, dtype=np.int64)

        if len(self.texts) != len(self.labels):
            raise ValueError(
                f"Texts and labels must have the same length. "
                f"Got {len(self.texts)} texts and {len(self.labels)} labels."
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def _build_tokenize_kwargs(
    tokenizer_kwargs: Optional[Dict[str, Any]],
    max_length: Optional[int],
) -> Dict[str, Any]:
    """Merge caller-provided tokenizer kwargs with sensible defaults."""
    kwargs: Dict[str, Any] = dict(tokenizer_kwargs or {})

    kwargs.setdefault("padding", "max_length" if max_length is not None else "longest")
    kwargs.setdefault("truncation", True)
    if max_length is not None:
        kwargs.setdefault("max_length", max_length)
    kwargs.setdefault("return_tensors", "pt")
    kwargs.setdefault("return_attention_mask", True)

    return kwargs


def create_text_classification_dataloader(
    texts: Iterable[str],
    labels: Union[Iterable[int], np.ndarray],
    *,
    tokenizer: Optional[Callable[..., Any]] = None,
    max_length: Optional[int] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """
    Create a DataLoader that yields batches ready for model consumption.

    When a tokenizer is provided, each batch will be tokenized and moved to the
    specified device, returning ``(token_batch_dict, labels_tensor)``.
    Otherwise, the loader returns ``(list_of_texts, labels_tensor)`` which is
    suitable for baseline models that perform their own vectorization.
    """
    dataset = TextClassificationDataset(texts, labels)
    tokenize_kwargs = _build_tokenize_kwargs(tokenizer_kwargs, max_length)

    def collate_fn(batch: List[Tuple[str, int]]):
        batch_texts, batch_labels = zip(*batch)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

        if tokenizer is None:
            if device is not None:
                labels_tensor = labels_tensor.to(device)
            return list(batch_texts), labels_tensor

        tokenized = tokenizer(list(batch_texts), **tokenize_kwargs)

        if device is not None:
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            labels_tensor = labels_tensor.to(device)

        return tokenized, labels_tensor

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

