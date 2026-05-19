"""Model recommendation schemas."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ModelCandidate:
    name: str
    score: float
    rationale: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCandidate":
        return cls(
            name=str(data["name"]),
            score=float(data["score"]),
            rationale=str(data["rationale"]),
            params=dict(data.get("params", {})),
        )


@dataclass
class ModelRecommendation:
    recommended_model: str
    candidates: List[ModelCandidate]
    constraints: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRecommendation":
        return cls(
            recommended_model=str(data["recommended_model"]),
            candidates=[
                ModelCandidate.from_dict(item) for item in data.get("candidates", [])
            ],
            constraints=dict(data.get("constraints", {})),
            notes=list(data.get("notes", [])),
        )

