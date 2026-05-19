"""Purposeful module proposal tool."""

from typing import Iterable, List

from hyperagent.schemas import (
    DatasetAudit,
    EvidenceItem,
    LiteraturePaper,
    ModuleProposal,
    SpectralReport,
)


class ModulePlanner:
    """Proposes module additions with explicit intent and evidence."""

    def propose(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        papers: Iterable[LiteraturePaper],
        objective: str = "improve_spectral_spatial_modeling",
    ) -> ModuleProposal:
        paper_list = list(papers)
        evidence: List[EvidenceItem] = [
            EvidenceItem(
                source_type="dataset_audit",
                source_id=audit.dataset_name,
                claim="Module design should respect band count and labeled sample size.",
                support=(
                    f"band_count={audit.band_count}; labeled_pixel_count={audit.labeled_pixel_count}; "
                    f"class_count={audit.class_count}"
                ),
                confidence=0.8,
            ),
            EvidenceItem(
                source_type="spectral_report",
                source_id=audit.dataset_name,
                claim="Spectral redundancy motivates a spectral gating or attention module.",
                support=(
                    f"recommended_removed_bands={spectral_report.recommended_removed_bands}; "
                    f"high_corr_pairs={len(spectral_report.highly_correlated_band_pairs)}"
                ),
                confidence=0.72,
            ),
        ]
        for paper in paper_list[:3]:
            evidence.append(
                EvidenceItem(
                    source_type="literature",
                    source_id=paper.title,
                    claim="Literature idea can motivate module design.",
                    support=paper.abstract[:350] if paper.abstract else paper.title,
                    confidence=0.55,
                    url=paper.url,
                )
            )

        if audit.labeled_pixel_count < 2000:
            module_type = "lightweight_spectral_gate"
            name = "EvidenceGuidedSpectralGate"
            summary = (
                "A lightweight per-band gating module before the classifier, regularized to avoid "
                "overfitting on small labeled sets."
            )
            risk = "May overfit if gate parameters are not regularized or validated across seeds."
        else:
            module_type = "spectral_spatial_attention"
            name = "EvidenceGuidedSpectralSpatialAdapter"
            summary = (
                "A compact adapter that combines spectral gating with shallow spatial context "
                "before downstream classification."
            )
            risk = "Adds training complexity and needs ablation against spectral-only baselines."

        return ModuleProposal(
            name=name,
            module_type=module_type,
            insertion_point="models registry: add module-backed classifier factory",
            design_summary=f"{summary} Objective: {objective}.",
            expected_effect="Improve robustness to redundant/noisy bands while preserving reproducible ablation.",
            implementation_steps=[
                "Create a new model factory under hyperagent/models without modifying training runner.",
                "Expose module hyperparameters through ModelConfig.params.",
                "Add an ablation candidate that toggles the module on/off.",
                "Record module rationale in ExperimentPlan.metadata['evidence'].",
            ],
            required_interfaces=["ClassifierFactory", "ExperimentPlan", "ModelConfig"],
            evidence=evidence,
            risks=[risk, "Do not claim improvement without fixed split and multi-seed comparison."],
        )

