"""Dataset audit schema."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetAudit:
    data_root: str
    dataset_name: str
    cube_path: Optional[str]
    label_path: Optional[str]
    cube_shape: List[int]
    label_shape: List[int]
    band_count: int
    class_count: int
    labeled_pixel_count: int
    unlabeled_pixel_count: int
    class_distribution: Dict[str, int]
    has_nan: bool
    has_inf: bool
    dtype: str
    reader_name: str
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetAudit":
        return cls(
            data_root=str(data["data_root"]),
            dataset_name=str(data["dataset_name"]),
            cube_path=data.get("cube_path"),
            label_path=data.get("label_path"),
            cube_shape=list(data["cube_shape"]),
            label_shape=list(data["label_shape"]),
            band_count=int(data["band_count"]),
            class_count=int(data["class_count"]),
            labeled_pixel_count=int(data["labeled_pixel_count"]),
            unlabeled_pixel_count=int(data["unlabeled_pixel_count"]),
            class_distribution={
                str(k): int(v) for k, v in data.get("class_distribution", {}).items()
            },
            has_nan=bool(data["has_nan"]),
            has_inf=bool(data["has_inf"]),
            dtype=str(data["dtype"]),
            reader_name=str(data["reader_name"]),
            warnings=list(data.get("warnings", [])),
            metadata=dict(data.get("metadata", {})),
        )

