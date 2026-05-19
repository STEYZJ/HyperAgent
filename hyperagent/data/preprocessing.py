"""Preprocessing functions for HSI cubes."""

from typing import Iterable

import numpy as np


def remove_bands(cube: np.ndarray, bands: Iterable[int]) -> np.ndarray:
    band_set = {int(band) for band in bands}
    if not band_set:
        return cube
    keep = [idx for idx in range(cube.shape[-1]) if idx not in band_set]
    if not keep:
        raise ValueError("Band removal would remove all spectral bands")
    return cube[:, :, keep]


def normalize_cube(cube: np.ndarray, method: str = "standard") -> np.ndarray:
    method = method.lower()
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float32)
    if method == "none":
        normalized = flat
    elif method == "minmax":
        low = np.nanmin(flat, axis=0)
        high = np.nanmax(flat, axis=0)
        scale = np.where(np.abs(high - low) < 1e-12, 1.0, high - low)
        normalized = (flat - low) / scale
    elif method == "standard":
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        normalized = (flat - mean) / std
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.reshape(cube.shape).astype(np.float32)


def flatten_labeled_pixels(cube: np.ndarray, labels: np.ndarray):
    mask = labels > 0
    x = cube.reshape(-1, cube.shape[-1])[mask.reshape(-1)]
    y = labels.reshape(-1)[mask.reshape(-1)]
    return x, y.astype(np.int64), mask

