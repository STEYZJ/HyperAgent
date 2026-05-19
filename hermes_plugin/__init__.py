"""Thin Hermes-facing wrappers for HyperAgent tools."""

from hermes_plugin.tools import (
    analyze_spectral_bands,
    build_experiment_plan,
    build_hsi_report,
    design_hsi_auto_experiments,
    inspect_hsi_dataset,
    propose_hsi_module,
    propose_hsi_parameter_updates,
    recommend_hsi_model,
    run_hsi_baseline,
    search_hsi_literature,
)

__all__ = [
    "analyze_spectral_bands",
    "build_experiment_plan",
    "build_hsi_report",
    "design_hsi_auto_experiments",
    "inspect_hsi_dataset",
    "propose_hsi_module",
    "propose_hsi_parameter_updates",
    "recommend_hsi_model",
    "run_hsi_baseline",
    "search_hsi_literature",
]
