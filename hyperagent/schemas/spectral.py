"""Spectral diagnostic schema."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpectralReport:
    dataset_name: str
    band_count: int
    low_variance_bands: List[int]
    anomalous_bands: List[int]
    adjacent_correlation_mean: Optional[float]
    highly_correlated_band_pairs: List[List[int]]
    recommended_removed_bands: List[int]
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectralReport":
        return cls(
            dataset_name=str(data["dataset_name"]),
            band_count=int(data["band_count"]),
            low_variance_bands=[int(v) for v in data.get("low_variance_bands", [])],
            anomalous_bands=[int(v) for v in data.get("anomalous_bands", [])],
            adjacent_correlation_mean=(
                None
                if data.get("adjacent_correlation_mean") is None
                else float(data["adjacent_correlation_mean"])
            ),
            highly_correlated_band_pairs=[
                [int(pair[0]), int(pair[1])]
                for pair in data.get("highly_correlated_band_pairs", [])
            ],
            recommended_removed_bands=[
                int(v) for v in data.get("recommended_removed_bands", [])
            ],
            notes=list(data.get("notes", [])),
            metadata=dict(data.get("metadata", {})),
        )

