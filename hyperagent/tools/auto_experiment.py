"""Autonomous experiment agenda builder."""

from typing import List, Optional

from hyperagent.schemas import (
    AutoExperimentAgenda,
    DatasetAudit,
    DecisionRecord,
    ExperimentCandidate,
    ModelRecommendation,
    SpectralReport,
)
from hyperagent.tools.evidence_builder import EvidenceBuilder


class AutoExperimentDesigner:
    """Builds purposeful experiment candidates without executing them."""

    def __init__(self) -> None:
        self.evidence_builder = EvidenceBuilder()

    def design(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        recommendation: ModelRecommendation,
        objective: str = "maximize_oa_with_reproducible_baseline",
        max_candidates: int = 4,
    ) -> AutoExperimentAgenda:
        candidates: List[ExperimentCandidate] = []
        base_decisions: List[DecisionRecord] = [
            self.evidence_builder.dataset_split_decision(audit, 0.1),
            self.evidence_builder.preprocessing_decision(audit, spectral_report),
            self.evidence_builder.model_decision(recommendation),
        ]
        candidates.append(
            ExperimentCandidate(
                name="baseline_reproducible_protocol",
                plan_patch={},
                priority=0.95,
                objective="Establish a reproducible baseline with current recommendation.",
                decisions=base_decisions,
            )
        )

        if spectral_report.recommended_removed_bands:
            candidates.append(
                ExperimentCandidate(
                    name="ablate_band_removal",
                    plan_patch={"preprocessing": {"remove_bands": []}},
                    priority=0.78,
                    objective="Verify whether spectral pruning improves performance.",
                    decisions=[
                        self.evidence_builder.preprocessing_decision(audit, spectral_report)
                    ],
                )
            )

        for candidate in recommendation.candidates:
            if candidate.name == recommendation.recommended_model:
                continue
            candidates.append(
                ExperimentCandidate(
                    name=f"compare_{candidate.name}_baseline",
                    plan_patch={"model": {"name": candidate.name, "params": candidate.params}},
                    priority=max(0.5, min(0.8, candidate.score)),
                    objective=f"Compare {candidate.name} against recommended baseline.",
                    decisions=[self.evidence_builder.model_decision(recommendation)],
                )
            )

        if audit.labeled_pixel_count < 2000:
            candidates.append(
                ExperimentCandidate(
                    name="increase_train_ratio_small_sample_check",
                    plan_patch={"split": {"train_ratio": 0.2, "test_ratio": 0.8}},
                    priority=0.7,
                    objective="Assess small-sample sensitivity to train ratio.",
                    decisions=[
                        self.evidence_builder.dataset_split_decision(audit, 0.2)
                    ],
                )
            )

        candidates.sort(key=lambda item: item.priority, reverse=True)
        return AutoExperimentAgenda(
            dataset_name=audit.dataset_name,
            objective=objective,
            candidates=candidates[:max_candidates],
            stop_rules=[
                "Stop if two consecutive candidates do not improve OA by at least 0.2 percentage points.",
                "Always keep the baseline candidate for reproducibility comparison.",
            ],
            notes=[
                "Agenda generation is evidence-driven and does not mutate model code.",
                "Candidates are plan patches that can be materialized into ExperimentPlan variants.",
            ],
        )

