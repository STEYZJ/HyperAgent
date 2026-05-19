"""Synthetic HSI data used for smoke tests and demos."""

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import savemat


def make_synthetic_hsi(
    height: int = 36,
    width: int = 36,
    bands: int = 24,
    classes: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.zeros((height, width), dtype=np.int64)
    cube = rng.normal(0.0, 0.08, size=(height, width, bands)).astype(np.float32)
    spectral_axis = np.linspace(0.0, 1.0, bands, dtype=np.float32)

    rows_per_class = height // classes
    for class_id in range(1, classes + 1):
        start = (class_id - 1) * rows_per_class
        stop = height if class_id == classes else class_id * rows_per_class
        labels[start:stop, :] = class_id
        center = class_id / (classes + 1)
        signature = np.exp(-((spectral_axis - center) ** 2) / 0.025)
        signature += 0.15 * class_id * spectral_axis
        cube[start:stop, :, :] += signature.reshape(1, 1, bands)

    unlabeled_mask = rng.random((height, width)) < 0.08
    labels[unlabeled_mask] = 0
    return cube.astype(np.float32), labels


def write_synthetic_mat(data_root: Path, seed: int = 42) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    cube, labels = make_synthetic_hsi(seed=seed)
    output_path = data_root / "synthetic_hsi.mat"
    savemat(output_path, {"synthetic_cube": cube, "synthetic_gt": labels})
    return output_path

