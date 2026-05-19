"""Coordinator agent that orchestrates tools without owning model logic."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from hyperagent.core.bootstrap import bootstrap_default_components
from hyperagent.core.io import write_json, write_yaml
from hyperagent.core.registries import (
    analyzer_registry,
    dataset_reader_registry,
    model_recommender_registry,
)
from hyperagent.schemas import (
    AutoExperimentAgenda,
    DatasetAudit,
    ExperimentPlan,
    ExperimentResult,
    LiteraturePaper,
    LiteratureSearchResult,
    ModuleProposal,
    ModelRecommendation,
    ParameterProposal,
    SpectralReport,
)
from hyperagent.tools.auto_experiment import AutoExperimentDesigner
from hyperagent.tools.dataset_inspector import DatasetInspector
from hyperagent.tools.literature_searcher import LiteratureSearcher
from hyperagent.tools.module_planner import ModulePlanner
from hyperagent.tools.parameter_tuner import ParameterTuner
from hyperagent.tools.planner import ExperimentPlanner
from hyperagent.tools.report_builder import MarkdownReportBuilder
from hyperagent.training.baseline_runner import BaselineRunner


class CoordinatorAgent:
    """Coordinates HSI research workflow steps through decoupled tools."""

    def __init__(self) -> None:
        bootstrap_default_components()
        self.inspector = DatasetInspector()
        self.planner = ExperimentPlanner()
        self.runner = BaselineRunner()
        self.report_builder = MarkdownReportBuilder()
        self.literature_searcher = LiteratureSearcher()
        self.auto_experiment_designer = AutoExperimentDesigner()
        self.parameter_tuner = ParameterTuner()
        self.module_planner = ModulePlanner()

    def audit(
        self,
        data_root: Path,
        output_path: Optional[Path] = None,
        reader_name: Optional[str] = None,
    ) -> DatasetAudit:
        audit = self.inspector.inspect(data_root, reader_name)
        if output_path is not None:
            write_json(output_path, audit)
        return audit

    def analyze(
        self,
        audit: DatasetAudit,
        output_path: Optional[Path] = None,
        analyzer_name: str = "basic",
    ) -> SpectralReport:
        reader = dataset_reader_registry.get(audit.reader_name)
        cube, _, metadata = reader.read(Path(audit.data_root))
        analyzer = analyzer_registry.get(analyzer_name)
        report = analyzer.analyze(cube, audit, metadata.get("wavelengths"))
        if output_path is not None:
            write_json(output_path, report)
        return report

    def recommend(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        output_path: Optional[Path] = None,
        constraints: Optional[Dict[str, Any]] = None,
        recommender_name: str = "basic",
    ) -> ModelRecommendation:
        recommender = model_recommender_registry.get(recommender_name)
        recommendation = recommender.recommend(audit, spectral_report, constraints or {})
        if output_path is not None:
            write_json(output_path, recommendation)
        return recommendation

    def plan(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        recommendation: ModelRecommendation,
        output_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> ExperimentPlan:
        plan = self.planner.build(audit, spectral_report, recommendation, output_dir, seed)
        if output_path is not None:
            write_yaml(output_path, plan)
        return plan

    def run(self, plan: ExperimentPlan) -> ExperimentResult:
        return self.runner.run(plan)

    def write_report(self, result: ExperimentResult, output_path: Path) -> Path:
        return self.report_builder.write(result, output_path)

    def search_literature(
        self,
        query: str,
        output_path: Optional[Path] = None,
        provider_name: str = "arxiv",
        max_results: int = 10,
        year_from: Optional[int] = None,
        sort_by: str = "latest",
    ) -> LiteratureSearchResult:
        return self.literature_searcher.search(
            query,
            output_path,
            provider_name=provider_name,
            max_results=max_results,
            year_from=year_from,
            sort_by=sort_by,
        )

    def design_auto_experiments(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        recommendation: ModelRecommendation,
        objective: str = "maximize_oa_with_reproducible_baseline",
        max_candidates: int = 4,
    ) -> AutoExperimentAgenda:
        return self.auto_experiment_designer.design(
            audit,
            spectral_report,
            recommendation,
            objective=objective,
            max_candidates=max_candidates,
        )

    def propose_parameter_updates(
        self,
        plan: ExperimentPlan,
        result: ExperimentResult,
        audit: DatasetAudit,
    ) -> List[ParameterProposal]:
        return self.parameter_tuner.propose(plan, result, audit)

    def propose_module(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        papers: List[LiteraturePaper],
        objective: str = "improve_spectral_spatial_modeling",
    ) -> ModuleProposal:
        return self.module_planner.propose(
            audit,
            spectral_report,
            papers,
            objective=objective,
        )
