"""Purposeful parameter tuning suggestions."""

from typing import List

from hyperagent.schemas import DatasetAudit, ExperimentPlan, ExperimentResult, ParameterProposal
from hyperagent.schemas import EvidenceItem


class ParameterTuner:
    """Suggests next parameter changes from evidence and prior result."""

    def propose(
        self,
        plan: ExperimentPlan,
        result: ExperimentResult,
        audit: DatasetAudit,
    ) -> List[ParameterProposal]:
        proposals: List[ParameterProposal] = []
        oa = result.evaluation.overall_accuracy
        if audit.labeled_pixel_count < 2000 and plan.split.train_ratio < 0.2:
            proposals.append(
                ParameterProposal(
                    parameter="split.train_ratio",
                    old_value=plan.split.train_ratio,
                    new_value=0.2,
                    rationale="Small labeled sets can make training unstable; test sensitivity to a larger train split.",
                    expected_effect="Reduce variance in small-sample baseline at the cost of fewer test samples.",
                    evidence=[
                        EvidenceItem(
                            source_type="dataset_audit",
                            source_id=audit.dataset_name,
                            claim="Labeled sample count is limited.",
                            support=f"labeled_pixel_count={audit.labeled_pixel_count}",
                            confidence=0.8,
                        )
                    ],
                )
            )
        if plan.model.name == "svm" and oa < 0.9:
            proposals.append(
                ParameterProposal(
                    parameter="model.params.C",
                    old_value=plan.model.params.get("C", 10.0),
                    new_value=30.0,
                    rationale="Underperforming RBF SVM may need a stronger margin penalty.",
                    expected_effect="Increase training fit; should be checked for overfitting.",
                    evidence=[
                        EvidenceItem(
                            source_type="experiment_result",
                            source_id=result.experiment_name,
                            claim="Current OA is below the target threshold.",
                            support=f"OA={oa:.4f}",
                            confidence=0.65,
                        )
                    ],
                )
            )
        if plan.model.name == "mlp" and oa < 0.9:
            proposals.append(
                ParameterProposal(
                    parameter="model.params.epochs",
                    old_value=plan.model.params.get("epochs", 30),
                    new_value=60,
                    rationale="Neural baseline under target accuracy may require longer optimization.",
                    expected_effect="Improve convergence; monitor overfitting with repeated seeds.",
                    evidence=[
                        EvidenceItem(
                            source_type="experiment_result",
                            source_id=result.experiment_name,
                            claim="Current OA is below the target threshold.",
                            support=f"OA={oa:.4f}",
                            confidence=0.6,
                        )
                    ],
                )
            )
        if not proposals:
            proposals.append(
                ParameterProposal(
                    parameter="seed",
                    old_value=plan.seed,
                    new_value=plan.seed + 1,
                    rationale="Current result meets basic thresholds; next check should estimate seed stability.",
                    expected_effect="Measure variance instead of changing architecture prematurely.",
                    evidence=[
                        EvidenceItem(
                            source_type="experiment_result",
                            source_id=result.experiment_name,
                            claim="No urgent parameter change was indicated.",
                            support=f"OA={oa:.4f}",
                            confidence=0.7,
                        )
                    ],
                )
            )
        return proposals
