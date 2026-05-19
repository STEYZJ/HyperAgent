"""Experiment planning tool."""

from pathlib import Path
from typing import Optional

from hyperagent.schemas import (
    DatasetAudit,
    ExperimentPlan,
    ModelConfig,
    ModelRecommendation,
    PreprocessingConfig,
    SpectralReport,
    SplitConfig,
)
from hyperagent.tools.evidence_builder import EvidenceBuilder


class ExperimentPlanner:
    """Build a reproducible baseline experiment plan."""

    def __init__(self) -> None:
        self.evidence_builder = EvidenceBuilder()

    def build(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        recommendation: ModelRecommendation,
        output_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> ExperimentPlan:
        selected = recommendation.candidates[0]
        for candidate in recommendation.candidates:
            if candidate.name == recommendation.recommended_model:
                selected = candidate
                break
        root = Path(output_dir or "experiments") / f"{audit.dataset_name}_{selected.name}_seed{seed}"
        train_ratio = 0.1 if audit.labeled_pixel_count >= 100 else 0.5
        decisions = [
            self.evidence_builder.dataset_split_decision(audit, train_ratio),
            self.evidence_builder.preprocessing_decision(audit, spectral_report),
            self.evidence_builder.model_decision(recommendation),
        ]
        return ExperimentPlan(
            experiment_name=f"{audit.dataset_name}_{selected.name}_seed{seed}",
            dataset_root=audit.data_root,
            output_dir=str(root),
            seed=seed,
            reader_name=audit.reader_name,
            split=SplitConfig(train_ratio=train_ratio, val_ratio=0.0, test_ratio=1.0 - train_ratio),
            preprocessing=PreprocessingConfig(
                normalization="standard",
                remove_bands=list(spectral_report.recommended_removed_bands),
            ),
            model=ModelConfig(name=selected.name, params=dict(selected.params)),
            metadata={
                "audit": audit.to_dict(),
                "spectral_report": spectral_report.to_dict(),
                "recommendation": recommendation.to_dict(),
                "evidence": [decision.to_dict() for decision in decisions],
            },
        )
