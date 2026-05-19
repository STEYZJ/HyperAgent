"""Default spectral diagnostic tool."""

from typing import Any, List, Optional

import numpy as np

from hyperagent.core.registries import analyzer_registry
from hyperagent.schemas import DatasetAudit, SpectralReport


class BasicSpectralAnalyzer:
    name = "basic"

    def analyze(
        self,
        cube: np.ndarray,
        audit: DatasetAudit,
        wavelengths: Optional[Any] = None,
    ) -> SpectralReport:
        if cube.ndim != 3:
            raise ValueError(f"Expected H,W,B cube, got shape {cube.shape}")

        flat = cube.reshape(-1, cube.shape[-1]).astype(np.float32)
        variance = np.nanvar(flat, axis=0)
        finite_per_band = np.isfinite(flat).mean(axis=0)
        median_variance = float(np.nanmedian(variance)) if variance.size else 0.0
        low_threshold = max(1e-12, median_variance * 0.01)
        low_variance_bands = [
            int(idx) for idx, value in enumerate(variance) if float(value) <= low_threshold
        ]
        anomalous_bands = [
            int(idx) for idx, ratio in enumerate(finite_per_band) if float(ratio) < 0.999
        ]

        adjacent_correlations: List[float] = []
        high_pairs: List[List[int]] = []
        for idx in range(cube.shape[-1] - 1):
            left = flat[:, idx]
            right = flat[:, idx + 1]
            if np.nanstd(left) < 1e-12 or np.nanstd(right) < 1e-12:
                continue
            corr = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(corr):
                adjacent_correlations.append(corr)
                if abs(corr) >= 0.995:
                    high_pairs.append([idx, idx + 1])

        recommended = set(low_variance_bands) | set(anomalous_bands)
        recommended.update(pair[1] for pair in high_pairs)
        wavelength_removed = self._water_absorption_bands(wavelengths)
        recommended.update(wavelength_removed)

        notes = []
        if low_variance_bands:
            notes.append("Low-variance bands may contain little discriminative signal.")
        if high_pairs:
            notes.append("Highly correlated adjacent bands suggest spectral redundancy.")
        if wavelength_removed:
            notes.append("Known water absorption wavelength ranges were marked for removal.")
        if not notes:
            notes.append("No severe spectral quality issue was detected by basic rules.")

        return SpectralReport(
            dataset_name=audit.dataset_name,
            band_count=int(cube.shape[-1]),
            low_variance_bands=low_variance_bands,
            anomalous_bands=anomalous_bands,
            adjacent_correlation_mean=(
                None
                if not adjacent_correlations
                else float(np.mean(np.abs(adjacent_correlations)))
            ),
            highly_correlated_band_pairs=high_pairs,
            recommended_removed_bands=sorted(int(v) for v in recommended),
            notes=notes,
            metadata={
                "median_variance": median_variance,
                "low_variance_threshold": low_threshold,
            },
        )

    def _water_absorption_bands(self, wavelengths: Optional[Any]) -> List[int]:
        if wavelengths is None:
            return []
        ranges = [(930.0, 970.0), (1130.0, 1160.0), (1340.0, 1450.0), (1800.0, 1950.0)]
        values = [float(v) for v in wavelengths]
        removed = []
        for idx, wavelength in enumerate(values):
            if any(low <= wavelength <= high for low, high in ranges):
                removed.append(idx)
        return removed


analyzer_registry.register(BasicSpectralAnalyzer.name, BasicSpectralAnalyzer(), replace=True)

