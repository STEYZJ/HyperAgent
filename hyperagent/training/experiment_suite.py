"""Multi-run experiment suite runner."""

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from hyperagent.core.bootstrap import bootstrap_default_components
from hyperagent.core.io import write_json, write_yaml
from hyperagent.schemas import ExperimentPlan, ExperimentResult, ExperimentSuiteResult
from hyperagent.training.baseline_runner import BaselineRunner


class ExperimentSuiteRunner:
    """Run one experiment plan across multiple seeds and summarize variance."""

    name = "experiment_suite"

    def __init__(self, runner: Optional[BaselineRunner] = None) -> None:
        bootstrap_default_components()
        self.runner = runner or BaselineRunner()

    def run(
        self,
        plan: ExperimentPlan,
        seeds: Sequence[int],
        output_dir: Optional[Path] = None,
        suite_name: Optional[str] = None,
    ) -> ExperimentSuiteResult:
        normalized_seeds = self._normalize_seeds(seeds)
        suite_name = suite_name or f"{plan.experiment_name}_suite"
        suite_dir = Path(output_dir) if output_dir else Path(plan.output_dir).parent / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)

        base_plan_path = suite_dir / "base_plan.yaml"
        write_yaml(base_plan_path, plan)

        results: List[ExperimentResult] = []
        warnings: List[str] = []
        for seed in normalized_seeds:
            seed_plan = self._build_seed_plan(plan, seed, suite_name, suite_dir)
            result = self.runner.run(seed_plan)
            results.append(result)
            warnings.extend(result.warnings)

        metrics_summary = self._summarize_metrics(results)
        best_result = max(
            results,
            key=lambda item: item.evaluation.overall_accuracy,
        )
        csv_path = self._write_csv(suite_dir / "suite_metrics.csv", results)
        markdown_path = self._write_markdown(
            suite_dir / "suite_report.md",
            suite_name,
            base_plan_path,
            normalized_seeds,
            results,
            metrics_summary,
            best_result,
        )

        suite_result = ExperimentSuiteResult(
            suite_name=suite_name,
            output_dir=str(suite_dir),
            base_plan_path=str(base_plan_path),
            seeds=normalized_seeds,
            run_count=len(results),
            results=results,
            metrics_summary=metrics_summary,
            best_seed=best_result.seed,
            best_result_path=str(Path(best_result.experiment_dir) / "result.json"),
            artifacts=[str(csv_path), str(markdown_path)],
            warnings=sorted(set(warnings)),
        )
        write_json(suite_dir / "suite.json", suite_result)
        return suite_result

    def _normalize_seeds(self, seeds: Sequence[int]) -> List[int]:
        normalized = [int(seed) for seed in seeds]
        if not normalized:
            raise ValueError("At least one seed is required for an experiment suite")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Duplicate seeds are not allowed: {normalized}")
        return normalized

    def _build_seed_plan(
        self,
        plan: ExperimentPlan,
        seed: int,
        suite_name: str,
        suite_dir: Path,
    ) -> ExperimentPlan:
        data = deepcopy(plan.to_dict())
        data["seed"] = seed
        data["experiment_name"] = f"{suite_name}_seed{seed}"
        data["output_dir"] = str(suite_dir / f"seed{seed}")
        metadata = dict(data.get("metadata", {}))
        metadata["suite"] = {
            "suite_name": suite_name,
            "base_experiment": plan.experiment_name,
            "seed": seed,
            "purpose": "Estimate seed variance before claiming parameter or module gains.",
        }
        data["metadata"] = metadata
        return ExperimentPlan.from_dict(data)

    def _summarize_metrics(
        self,
        results: Sequence[ExperimentResult],
    ) -> Dict[str, Dict[str, Any]]:
        metrics = {
            "overall_accuracy": [
                result.evaluation.overall_accuracy for result in results
            ],
            "average_accuracy": [
                result.evaluation.average_accuracy for result in results
            ],
            "kappa": [result.evaluation.kappa for result in results],
            "duration_sec": [result.duration_sec for result in results],
        }
        return {
            name: self._summary(values)
            for name, values in metrics.items()
        }

    def _summary(self, values: Sequence[float]) -> Dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "values": [float(value) for value in array.tolist()],
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    def _write_csv(self, path: Path, results: Sequence[ExperimentResult]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment_name",
                    "seed",
                    "model_name",
                    "train_samples",
                    "test_samples",
                    "overall_accuracy",
                    "average_accuracy",
                    "kappa",
                    "duration_sec",
                    "result_path",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "experiment_name": result.experiment_name,
                        "seed": result.seed,
                        "model_name": result.model_name,
                        "train_samples": result.train_samples,
                        "test_samples": result.test_samples,
                        "overall_accuracy": result.evaluation.overall_accuracy,
                        "average_accuracy": result.evaluation.average_accuracy,
                        "kappa": result.evaluation.kappa,
                        "duration_sec": result.duration_sec,
                        "result_path": str(Path(result.experiment_dir) / "result.json"),
                    }
                )
        return path

    def _write_markdown(
        self,
        path: Path,
        suite_name: str,
        base_plan_path: Path,
        seeds: Sequence[int],
        results: Sequence[ExperimentResult],
        metrics_summary: Dict[str, Dict[str, Any]],
        best_result: ExperimentResult,
    ) -> Path:
        lines = [
            f"# Experiment Suite: {suite_name}",
            "",
            f"- Base plan: {base_plan_path}",
            f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
            f"- Runs: {len(results)}",
            f"- Best seed: {best_result.seed}",
            f"- Best OA: {best_result.evaluation.overall_accuracy:.4f}",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for metric in ("overall_accuracy", "average_accuracy", "kappa"):
            summary = metrics_summary[metric]
            lines.append(
                f"| {metric} | {summary['mean']:.4f} | {summary['std']:.4f} | "
                f"{summary['min']:.4f} | {summary['max']:.4f} |"
            )
        lines.extend(
            [
                "",
                "## Runs",
                "",
                "| Seed | Model | Train | Test | OA | AA | Kappa | Result |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for result in results:
            result_path = Path(result.experiment_dir) / "result.json"
            lines.append(
                f"| {result.seed} | {result.model_name} | {result.train_samples} | "
                f"{result.test_samples} | {result.evaluation.overall_accuracy:.4f} | "
                f"{result.evaluation.average_accuracy:.4f} | {result.evaluation.kappa:.4f} | "
                f"{result_path} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
