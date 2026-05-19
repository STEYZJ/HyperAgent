"""Hermes adapter functions.

These functions intentionally contain no business logic; they call the
CoordinatorAgent and return schema dictionaries.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from hyperagent.agents import CoordinatorAgent
from hyperagent.core.io import read_json, read_yaml, write_json
from hyperagent.schemas import (
    DatasetAudit,
    ExperimentPlan,
    ExperimentResult,
    LiteratureSearchResult,
    ModelRecommendation,
    SpectralReport,
)


def inspect_hsi_dataset(data_root: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    audit = agent.audit(Path(data_root), Path(output_path) if output_path else None)
    return audit.to_dict()


def analyze_spectral_bands(
    audit_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    audit = DatasetAudit.from_dict(read_json(Path(audit_path)))
    report = agent.analyze(audit, Path(output_path) if output_path else None)
    return report.to_dict()


def recommend_hsi_model(
    audit_path: str,
    spectral_report_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    audit = DatasetAudit.from_dict(read_json(Path(audit_path)))
    spectral_report = SpectralReport.from_dict(read_json(Path(spectral_report_path)))
    recommendation = agent.recommend(audit, spectral_report, Path(output_path) if output_path else None)
    return recommendation.to_dict()


def build_experiment_plan(
    audit_path: str,
    spectral_report_path: str,
    recommendation_path: str,
    output_path: str,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    audit = DatasetAudit.from_dict(read_json(Path(audit_path)))
    spectral_report = SpectralReport.from_dict(read_json(Path(spectral_report_path)))
    recommendation = ModelRecommendation.from_dict(read_json(Path(recommendation_path)))
    plan = agent.plan(
        audit,
        spectral_report,
        recommendation,
        Path(output_path),
        Path(output_dir) if output_dir else None,
        seed,
    )
    return plan.to_dict()


def run_hsi_baseline(plan_path: str) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    plan = ExperimentPlan.from_dict(read_yaml(Path(plan_path)))
    return agent.run(plan).to_dict()


def build_hsi_report(experiment_dir: str, output_path: Optional[str] = None) -> str:
    agent = CoordinatorAgent()
    result = ExperimentResult.from_dict(read_json(Path(experiment_dir) / "result.json"))
    target = Path(output_path) if output_path else Path(experiment_dir) / "report.md"
    return str(agent.write_report(result, target))


def search_hsi_literature(
    query: str,
    output_path: Optional[str] = None,
    provider: str = "arxiv",
    max_results: int = 10,
    year_from: Optional[int] = None,
) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    result = agent.search_literature(
        query,
        Path(output_path) if output_path else None,
        provider_name=provider,
        max_results=max_results,
        year_from=year_from,
    )
    return result.to_dict()


def design_hsi_auto_experiments(
    audit_path: str,
    spectral_report_path: str,
    recommendation_path: str,
    output_path: Optional[str] = None,
    objective: str = "maximize_oa_with_reproducible_baseline",
) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    audit = DatasetAudit.from_dict(read_json(Path(audit_path)))
    spectral_report = SpectralReport.from_dict(read_json(Path(spectral_report_path)))
    recommendation = ModelRecommendation.from_dict(read_json(Path(recommendation_path)))
    agenda = agent.design_auto_experiments(audit, spectral_report, recommendation, objective)
    if output_path:
        write_json(Path(output_path), agenda)
    return agenda.to_dict()


def propose_hsi_parameter_updates(
    plan_path: str,
    result_path: str,
    audit_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    plan = ExperimentPlan.from_dict(read_yaml(Path(plan_path)))
    result = ExperimentResult.from_dict(read_json(Path(result_path)))
    audit = DatasetAudit.from_dict(read_json(Path(audit_path)))
    proposals = agent.propose_parameter_updates(plan, result, audit)
    payload = {"proposals": [item.to_dict() for item in proposals]}
    if output_path:
        write_json(Path(output_path), payload)
    return payload


def propose_hsi_module(
    audit_path: str,
    spectral_report_path: str,
    literature_path: str,
    output_path: Optional[str] = None,
    objective: str = "improve_spectral_spatial_modeling",
) -> Dict[str, Any]:
    agent = CoordinatorAgent()
    audit = DatasetAudit.from_dict(read_json(Path(audit_path)))
    spectral_report = SpectralReport.from_dict(read_json(Path(spectral_report_path)))
    literature = LiteratureSearchResult.from_dict(read_json(Path(literature_path)))
    proposal = agent.propose_module(audit, spectral_report, literature.papers, objective)
    if output_path:
        write_json(Path(output_path), proposal)
    return proposal.to_dict()
