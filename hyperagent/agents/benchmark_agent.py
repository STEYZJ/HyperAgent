"""Benchmark matrix orchestration for real HSI datasets."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hyperagent.agents.coordinator import CoordinatorAgent
from hyperagent.core.io import read_yaml, write_json
from hyperagent.training.experiment_suite import ExperimentSuiteRunner


class BenchmarkAgent:
    """Build audit/plan/suite artifacts for multiple catalogued datasets."""

    def __init__(self) -> None:
        self.coordinator = CoordinatorAgent()
        self.suite_runner = ExperimentSuiteRunner()

    def list_catalog(self, catalog_path: Path) -> Dict[str, Any]:
        catalog = read_yaml(catalog_path)
        datasets = catalog.get("datasets", {})
        if not isinstance(datasets, dict):
            raise ValueError(f"Catalog must contain a datasets mapping: {catalog_path}")
        return datasets

    def run_matrix(
        self,
        catalog_path: Path,
        dataset_names: Optional[Iterable[str]],
        reports_root: Path,
        experiments_root: Path,
        seeds: Iterable[int],
        run_suite: bool = False,
    ) -> Dict[str, Any]:
        datasets = self.list_catalog(catalog_path)
        selected_names = list(dataset_names or datasets.keys())
        normalized_seeds = [int(seed) for seed in seeds]
        reports_root.mkdir(parents=True, exist_ok=True)
        experiments_root.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for name in selected_names:
            spec = datasets.get(name)
            if spec is None:
                rows.append(
                    {
                        "dataset": name,
                        "status": "missing_catalog_entry",
                        "warnings": [f"Dataset '{name}' is not in {catalog_path}."],
                    }
                )
                continue
            row = self._run_dataset(
                name,
                spec,
                reports_root,
                experiments_root,
                normalized_seeds,
                run_suite=run_suite,
            )
            rows.append(row)

        matrix = {
            "catalog_path": str(catalog_path),
            "reports_root": str(reports_root),
            "experiments_root": str(experiments_root),
            "run_suite": run_suite,
            "seeds": normalized_seeds,
            "datasets": rows,
        }
        write_json(reports_root / "benchmark_matrix.json", matrix)
        self._write_markdown(reports_root / "benchmark_matrix.md", matrix)
        return matrix

    def _run_dataset(
        self,
        name: str,
        spec: Dict[str, Any],
        reports_root: Path,
        experiments_root: Path,
        seeds: List[int],
        run_suite: bool,
    ) -> Dict[str, Any]:
        dataset_path = Path(str(spec.get("local_example", "")))
        dataset_dir = reports_root / _slug(name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if not dataset_path.exists():
            return {
                "dataset": name,
                "status": "missing_local_path",
                "dataset_path": str(dataset_path),
                "source_url": spec.get("source_url"),
                "warnings": [f"Local path does not exist: {dataset_path}"],
            }

        try:
            audit_path = dataset_dir / "audit.json"
            spectral_path = dataset_dir / "spectral_report.json"
            recommendation_path = dataset_dir / "model_recommendation.json"
            plan_path = dataset_dir / "experiment.yaml"
            audit = self.coordinator.audit(dataset_path, audit_path)
            spectral = self.coordinator.analyze(audit, spectral_path)
            recommendation = self.coordinator.recommend(
                audit,
                spectral,
                recommendation_path,
            )
            plan = self.coordinator.plan(
                audit,
                spectral,
                recommendation,
                plan_path,
                experiments_root / _slug(name),
                seed=seeds[0] if seeds else 42,
            )
            row: Dict[str, Any] = {
                "dataset": name,
                "status": "planned",
                "dataset_path": str(dataset_path),
                "source_url": spec.get("source_url"),
                "audit_path": str(audit_path),
                "spectral_report_path": str(spectral_path),
                "recommendation_path": str(recommendation_path),
                "plan_path": str(plan_path),
                "reader_name": audit.reader_name,
                "cube_shape": audit.cube_shape,
                "band_count": audit.band_count,
                "class_count": audit.class_count,
                "labeled_pixel_count": audit.labeled_pixel_count,
                "recommended_model": recommendation.recommended_model,
                "warnings": list(audit.warnings),
            }
            if run_suite:
                suite = self.suite_runner.run(
                    plan,
                    seeds=seeds,
                    output_dir=experiments_root / f"{_slug(name)}_suite",
                    suite_name=f"{_slug(name)}_{plan.model.name}_suite",
                )
                row.update(
                    {
                        "status": "completed",
                        "suite_path": str(Path(suite.output_dir) / "suite.json"),
                        "suite_report_path": str(Path(suite.output_dir) / "suite_report.md"),
                        "oa_mean": suite.metrics_summary["overall_accuracy"]["mean"],
                        "oa_std": suite.metrics_summary["overall_accuracy"]["std"],
                        "aa_mean": suite.metrics_summary["average_accuracy"]["mean"],
                        "kappa_mean": suite.metrics_summary["kappa"]["mean"],
                        "best_seed": suite.best_seed,
                        "best_result_path": suite.best_result_path,
                        "warnings": sorted(set(row["warnings"] + suite.warnings)),
                    }
                )
            return row
        except Exception as exc:
            return {
                "dataset": name,
                "status": "failed",
                "dataset_path": str(dataset_path),
                "source_url": spec.get("source_url"),
                "warnings": [f"{type(exc).__name__}: {exc}"],
            }

    def _write_markdown(self, path: Path, matrix: Dict[str, Any]) -> Path:
        lines = [
            "# Benchmark Matrix",
            "",
            f"- Catalog: {matrix['catalog_path']}",
            f"- Reports root: {matrix['reports_root']}",
            f"- Experiments root: {matrix['experiments_root']}",
            f"- Run suite: {matrix['run_suite']}",
            f"- Seeds: {', '.join(str(seed) for seed in matrix['seeds'])}",
            "",
            "| Dataset | Status | Reader | Bands | Classes | Labeled | Model | OA Mean | OA Std | Artifacts |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
        for row in matrix["datasets"]:
            artifacts = row.get("suite_report_path") or row.get("plan_path") or ""
            lines.append(
                f"| {row.get('dataset', '')} | {row.get('status', '')} | "
                f"{row.get('reader_name', '')} | {row.get('band_count', '')} | "
                f"{row.get('class_count', '')} | {row.get('labeled_pixel_count', '')} | "
                f"{row.get('recommended_model', '')} | {_format_metric(row.get('oa_mean'))} | "
                f"{_format_metric(row.get('oa_std'))} | {artifacts} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"
