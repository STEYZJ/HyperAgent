"""Evaluation metric schema."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class EvaluationReport:
    overall_accuracy: float
    average_accuracy: float
    kappa: float
    labels: List[int]
    per_class_accuracy: Dict[str, float]
    confusion_matrix: List[List[int]]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EvaluationReport":
        return cls(
            overall_accuracy=float(data["overall_accuracy"]),
            average_accuracy=float(data["average_accuracy"]),
            kappa=float(data["kappa"]),
            labels=[int(v) for v in data.get("labels", [])],
            per_class_accuracy={
                str(k): float(v)
                for k, v in dict(data.get("per_class_accuracy", {})).items()
            },
            confusion_matrix=[
                [int(v) for v in row] for row in data.get("confusion_matrix", [])
            ],
            warnings=list(data.get("warnings", [])),
        )

