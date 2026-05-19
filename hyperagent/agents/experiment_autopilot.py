"""Autonomous experiment-result analysis and next-experiment runner."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hyperagent.core.bootstrap import bootstrap_default_components
from hyperagent.core.io import write_json, write_yaml
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import (
    DatasetAudit,
    EvidenceItem,
    ExperimentCycle,
    ExperimentDiagnosis,
    ExperimentPlan,
    ExperimentResult,
    ParameterProposal,
)
from hyperagent.tools.parameter_tuner import ParameterTuner
from hyperagent.tools.report_builder import MarkdownReportBuilder
from hyperagent.training.baseline_runner import BaselineRunner


class ExperimentAutopilotAgent:
    """Analyzes completed experiments and launches evidence-backed next runs."""

    def __init__(self) -> None:
        bootstrap_default_components()
        self.tuner = ParameterTuner()
        self.runner = BaselineRunner()
        self.report_builder = MarkdownReportBuilder()

    def diagnose(
        self,
        plan: ExperimentPlan,
        result: ExperimentResult,
        audit: DatasetAudit,
        objective: str = "maximize_oa_with_reproducible_baseline",
        target_oa: float = 0.9,
    ) -> ExperimentDiagnosis:
        evaluation = result.evaluation
        weakest = sorted(
            [
                {"class_id": label, "accuracy": value}
                for label, value in evaluation.per_class_accuracy.items()
            ],
            key=lambda item: float(item["accuracy"]),
        )[:5]
        findings = [
            f"OA={evaluation.overall_accuracy:.4f}, AA={evaluation.average_accuracy:.4f}, Kappa={evaluation.kappa:.4f}.",
            f"Train/test samples={result.train_samples}/{result.test_samples}; train_ratio={plan.split.train_ratio:.3f}.",
        ]
        if weakest:
            findings.append(
                "Weakest classes: "
                + ", ".join(
                    f"{item['class_id']}={float(item['accuracy']):.4f}"
                    for item in weakest
                )
                + "."
            )
        if evaluation.overall_accuracy < target_oa:
            recommendation = "Continue with a targeted parameter change before adding new modules."
            should_continue = True
        else:
            recommendation = "Continue with a seed-stability run before changing architecture."
            should_continue = True
        evidence = [
            EvidenceItem(
                source_type="experiment_result",
                source_id=result.experiment_name,
                claim="Experiment metrics were analyzed for autonomous next-step selection.",
                support=(
                    f"OA={evaluation.overall_accuracy:.4f}; "
                    f"AA={evaluation.average_accuracy:.4f}; "
                    f"Kappa={evaluation.kappa:.4f}"
                ),
                confidence=0.85,
            ),
            EvidenceItem(
                source_type="dataset_audit",
                source_id=audit.dataset_name,
                claim="Dataset scale constrains the next experiment choice.",
                support=(
                    f"labeled_pixel_count={audit.labeled_pixel_count}; "
                    f"class_count={audit.class_count}; band_count={audit.band_count}"
                ),
                confidence=0.75,
            ),
        ]
        return ExperimentDiagnosis(
            experiment_name=result.experiment_name,
            objective=objective,
            overall_accuracy=evaluation.overall_accuracy,
            average_accuracy=evaluation.average_accuracy,
            kappa=evaluation.kappa,
            weakest_classes=weakest,
            findings=findings,
            recommendation=recommendation,
            should_continue=should_continue,
            evidence=evidence,
        )

    def run_cycle(
        self,
        plan: ExperimentPlan,
        result: ExperimentResult,
        audit: DatasetAudit,
        previous_plan_path: Path,
        previous_result_path: Path,
        audit_path: Path,
        output_root: Path,
        objective: str = "maximize_oa_with_reproducible_baseline",
        target_oa: float = 0.9,
        run_next: bool = False,
    ) -> ExperimentCycle:
        cycle_id = self._new_cycle_id()
        cycle_dir = output_root / cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)
        diagnosis = self.diagnose(
            plan,
            result,
            audit,
            objective=objective,
            target_oa=target_oa,
        )
        proposals = self.tuner.propose(plan, result, audit)
        selected = proposals[0] if proposals else None
        next_plan = self.build_next_plan(
            plan,
            selected,
            output_dir=cycle_dir / "run",
            cycle_id=cycle_id,
        )

        diagnosis_path = cycle_dir / "diagnosis.json"
        proposals_path = cycle_dir / "parameter_proposals.json"
        next_plan_path = cycle_dir / "next_experiment.yaml"
        write_json(diagnosis_path, diagnosis)
        write_json(proposals_path, {"proposals": [item.to_dict() for item in proposals]})
        write_yaml(next_plan_path, next_plan)

        next_result_path: Optional[str] = None
        report_path: Optional[str] = None
        warnings: List[str] = []
        status = "planned"
        if run_next:
            next_result = self.runner.run(next_plan)
            report = self.report_builder.write(
                next_result,
                Path(next_result.experiment_dir) / "report.md",
            )
            next_result_path = str(Path(next_result.experiment_dir) / "result.json")
            report_path = str(report)
            warnings.extend(next_result.warnings)
            status = "completed"

        cycle = ExperimentCycle(
            cycle_id=cycle_id,
            created_at=utc_now(),
            status=status,
            previous_plan_path=str(previous_plan_path),
            previous_result_path=str(previous_result_path),
            audit_path=str(audit_path),
            cycle_dir=str(cycle_dir),
            diagnosis_path=str(diagnosis_path),
            proposals_path=str(proposals_path),
            next_plan_path=str(next_plan_path),
            selected_proposal=selected,
            next_result_path=next_result_path,
            report_path=report_path,
            warnings=warnings,
        )
        write_json(cycle_dir / "cycle.json", cycle)
        return cycle

    def build_next_plan(
        self,
        plan: ExperimentPlan,
        proposal: Optional[ParameterProposal],
        output_dir: Path,
        cycle_id: str,
    ) -> ExperimentPlan:
        data = deepcopy(plan.to_dict())
        if proposal is not None:
            self._set_plan_value(data, proposal.parameter, proposal.new_value)
        data["experiment_name"] = self._next_experiment_name(plan, proposal, cycle_id)
        data["output_dir"] = str(output_dir)
        metadata = dict(data.get("metadata", {}))
        metadata["autopilot"] = {
            "cycle_id": cycle_id,
            "parent_experiment": plan.experiment_name,
            "selected_parameter": proposal.parameter if proposal else None,
            "old_value": proposal.old_value if proposal else None,
            "new_value": proposal.new_value if proposal else None,
            "rationale": proposal.rationale if proposal else "No proposal available.",
            "expected_effect": proposal.expected_effect if proposal else "",
        }
        data["metadata"] = metadata
        return ExperimentPlan.from_dict(data)

    def _set_plan_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        target: Dict[str, Any] = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
        if path == "split.train_ratio":
            split = data.setdefault("split", {})
            val_ratio = float(split.get("val_ratio", 0.0))
            train_ratio = float(value)
            split["test_ratio"] = max(0.0, 1.0 - train_ratio - val_ratio)

    def _next_experiment_name(
        self,
        plan: ExperimentPlan,
        proposal: Optional[ParameterProposal],
        cycle_id: str,
    ) -> str:
        if proposal is None:
            suffix = "no_proposal"
        else:
            safe_parameter = "".join(
                char.lower() if char.isalnum() else "_"
                for char in proposal.parameter
            ).strip("_")
            suffix = safe_parameter[:32] or "update"
        return f"{plan.experiment_name}_auto_{suffix}_{cycle_id[-6:]}"

    def _new_cycle_id(self) -> str:
        return f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}"
