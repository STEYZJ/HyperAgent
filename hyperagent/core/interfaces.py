"""Protocol interfaces that keep HyperAgent modules decoupled."""

from pathlib import Path
from typing import Any, Dict, Protocol, Tuple

import numpy as np

from hyperagent.schemas import (
    DatasetAudit,
    EvaluationReport,
    ExperimentPlan,
    ExperimentResult,
    LiteratureSearchResult,
    ModelRecommendation,
    SpectralReport,
)


class DatasetReader(Protocol):
    """Reads HSI data into a cube and label map."""

    name: str

    def can_read(self, data_root: Path) -> bool:
        """Return whether this reader can load the given data root."""

    def read(self, data_root: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return cube, label map, and metadata."""


class SpectralAnalyzer(Protocol):
    """Produces a spectral diagnostic report from an HSI cube."""

    name: str

    def analyze(
        self,
        cube: np.ndarray,
        audit: DatasetAudit,
        wavelengths: Any = None,
    ) -> SpectralReport:
        """Analyze spectral statistics and produce recommendations."""


class ModelRecommender(Protocol):
    """Recommends a model from dataset and spectral diagnostics."""

    name: str

    def recommend(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        constraints: Dict[str, Any],
    ) -> ModelRecommendation:
        """Return model candidates and the selected model."""


class LiteratureProvider(Protocol):
    """Searches literature metadata from a replaceable source."""

    name: str

    def search(
        self,
        query: str,
        max_results: int = 10,
        *,
        year_from: Any = None,
        sort_by: str = "latest",
    ) -> LiteratureSearchResult:
        """Return literature search results."""


class ExperimentRunner(Protocol):
    """Runs an experiment plan and returns persisted results."""

    name: str

    def run(self, plan: ExperimentPlan) -> ExperimentResult:
        """Execute a reproducible experiment plan."""


class Evaluator(Protocol):
    """Computes classification metrics."""

    name: str

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationReport:
        """Compute metrics from true and predicted labels."""


class ReportWriter(Protocol):
    """Writes user-facing experiment reports."""

    name: str

    def write(self, result: ExperimentResult, output_path: Path) -> Path:
        """Write a report and return its path."""


class Classifier(Protocol):
    """Minimal classifier contract used by training runners."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the classifier."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels for feature rows."""


class ClassifierFactory(Protocol):
    """Builds a classifier from model params and a seed."""

    def __call__(self, params: Dict[str, Any], seed: int) -> Classifier:
        """Return a classifier instance."""
