"""Experiment plan and result schemas."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from hyperagent.schemas.evaluation import EvaluationReport


@dataclass
class SplitConfig:
    method: str = "stratified_random"
    train_ratio: float = 0.1
    val_ratio: float = 0.0
    test_ratio: float = 0.9

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitConfig":
        return cls(
            method=str(data.get("method", "stratified_random")),
            train_ratio=float(data.get("train_ratio", 0.1)),
            val_ratio=float(data.get("val_ratio", 0.0)),
            test_ratio=float(data.get("test_ratio", 0.9)),
        )


@dataclass
class PreprocessingConfig:
    normalization: str = "standard"
    remove_bands: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        return cls(
            normalization=str(data.get("normalization", "standard")),
            remove_bands=[int(v) for v in data.get("remove_bands", [])],
        )


@dataclass
class ModelConfig:
    name: str = "svm"
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(name=str(data.get("name", "svm")), params=dict(data.get("params", {})))


@dataclass
class ExperimentPlan:
    experiment_name: str
    dataset_root: str
    output_dir: str
    seed: int
    reader_name: str
    split: SplitConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    metrics: List[str] = field(default_factory=lambda: ["oa", "aa", "kappa"])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentPlan":
        return cls(
            experiment_name=str(data["experiment_name"]),
            dataset_root=str(data["dataset_root"]),
            output_dir=str(data["output_dir"]),
            seed=int(data.get("seed", 42)),
            reader_name=str(data.get("reader_name", "mat")),
            split=SplitConfig.from_dict(dict(data.get("split", {}))),
            preprocessing=PreprocessingConfig.from_dict(
                dict(data.get("preprocessing", {}))
            ),
            model=ModelConfig.from_dict(dict(data.get("model", {}))),
            metrics=list(data.get("metrics", ["oa", "aa", "kappa"])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ExperimentResult:
    experiment_name: str
    experiment_dir: str
    model_name: str
    seed: int
    train_samples: int
    test_samples: int
    evaluation: EvaluationReport
    artifacts: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    status: str = "completed"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        return cls(
            experiment_name=str(data["experiment_name"]),
            experiment_dir=str(data["experiment_dir"]),
            model_name=str(data["model_name"]),
            seed=int(data["seed"]),
            train_samples=int(data["train_samples"]),
            test_samples=int(data["test_samples"]),
            evaluation=EvaluationReport.from_dict(dict(data["evaluation"])),
            artifacts=list(data.get("artifacts", [])),
            duration_sec=float(data.get("duration_sec", 0.0)),
            status=str(data.get("status", "completed")),
            warnings=list(data.get("warnings", [])),
        )

