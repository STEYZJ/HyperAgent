"""Dataset split helpers."""

from typing import Tuple

import numpy as np


def stratified_train_test_split(
    labels: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    labels = np.asarray(labels)

    for class_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        train_count = max(1, int(round(len(class_indices) * train_ratio)))
        if train_count >= len(class_indices):
            train_count = max(1, len(class_indices) - 1)
        train_indices.extend(class_indices[:train_count].tolist())
        test_indices.extend(class_indices[train_count:].tolist())

    if not train_indices or not test_indices:
        raise ValueError("Unable to build non-empty train/test split")

    return np.asarray(train_indices, dtype=np.int64), np.asarray(test_indices, dtype=np.int64)

