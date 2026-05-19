"""Build evidence records for experiment decisions."""

from typing import Iterable, List, Optional

from hyperagent.schemas import (
    DatasetAudit,
    DecisionRecord,
    EvidenceItem,
    LiteraturePaper,
    ModelRecommendation,
    SpectralReport,
)


class EvidenceBuilder:
    """Converts audit, spectral, recommendation, and papers into evidence."""

    def dataset_split_decision(self, audit: DatasetAudit, train_ratio: float) -> DecisionRecord:
        evidence = [
            EvidenceItem(
                source_type="dataset_audit",
                source_id=audit.dataset_name,
                claim="Dataset split should account for labeled sample availability.",
                support=(
                    f"{audit.dataset_name} has {audit.labeled_pixel_count} labeled pixels "
                    f"across {audit.class_count} classes."
                ),
                confidence=0.85,
            )
        ]
        return DecisionRecord(
            decision_type="dataset_split",
            target=audit.dataset_name,
            choice=f"stratified_random train_ratio={train_ratio}",
            rationale="Use stratification to preserve class coverage while keeping a reproducible seed.",
            evidence=evidence,
            expected_effect="Stable class coverage and reproducible train/test separation.",
            risk="Random split may overestimate performance if spatial autocorrelation is strong.",
        )

    def preprocessing_decision(
        self, audit: DatasetAudit, spectral_report: SpectralReport
    ) -> DecisionRecord:
        evidence = [
            EvidenceItem(
                source_type="spectral_report",
                source_id=audit.dataset_name,
                claim="Band preprocessing should follow detected spectral quality issues.",
                support=(
                    f"Recommended removal bands: {spectral_report.recommended_removed_bands}; "
                    f"mean adjacent correlation: {spectral_report.adjacent_correlation_mean}."
                ),
                confidence=0.75,
            )
        ]
        return DecisionRecord(
            decision_type="preprocessing",
            target="spectral_bands",
            choice=f"remove_bands={spectral_report.recommended_removed_bands}; normalization=standard",
            rationale="Remove obviously weak/noisy bands and standardize each spectral channel.",
            evidence=evidence,
            expected_effect="Reduce noisy spectral dimensions and stabilize baseline optimization.",
            risk="Aggressive band removal may discard useful class-specific absorption signals.",
        )

    def model_decision(
        self, recommendation: ModelRecommendation, papers: Optional[Iterable[LiteraturePaper]] = None
    ) -> DecisionRecord:
        evidence: List[EvidenceItem] = []
        for candidate in recommendation.candidates:
            evidence.append(
                EvidenceItem(
                    source_type="model_recommendation",
                    source_id=candidate.name,
                    claim=f"{candidate.name} is a candidate model.",
                    support=f"score={candidate.score}; rationale={candidate.rationale}",
                    confidence=float(candidate.score),
                )
            )
        for paper in papers or []:
            evidence.append(
                EvidenceItem(
                    source_type="literature",
                    source_id=paper.title,
                    claim="Related literature may motivate model/module choice.",
                    support=paper.abstract[:400] if paper.abstract else paper.title,
                    confidence=0.55,
                    url=paper.url,
                )
            )
        return DecisionRecord(
            decision_type="model_selection",
            target="baseline_model",
            choice=recommendation.recommended_model,
            rationale="Select the highest-scored MVP-supported model before testing larger modules.",
            evidence=evidence,
            expected_effect="Provide a stable baseline for later ablation and module additions.",
            risk="Baseline recommendation does not guarantee state-of-the-art accuracy.",
        )

