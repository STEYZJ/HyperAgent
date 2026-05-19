"""Default component registration."""

import importlib
from pathlib import Path

_BOOTSTRAPPED = False


def bootstrap_default_components() -> None:
    """Import default modules once so their registry hooks run."""

    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    import hyperagent.data.mat_reader  # noqa: F401
    import hyperagent.evaluation.metrics  # noqa: F401
    import hyperagent.literature.arxiv_provider  # noqa: F401
    import hyperagent.literature.semantic_scholar_provider  # noqa: F401
    import hyperagent.models.mlp  # noqa: F401
    import hyperagent.models.svm  # noqa: F401
    import hyperagent.tools.model_recommender  # noqa: F401
    import hyperagent.tools.spectral_analyzer  # noqa: F401
    _import_generated_models()

    _BOOTSTRAPPED = True


def _import_generated_models() -> None:
    generated_dir = Path(__file__).resolve().parents[1] / "models" / "generated"
    if not generated_dir.exists():
        return
    for path in sorted(generated_dir.glob("*.py")):
        if path.name.startswith("__"):
            continue
        importlib.import_module(f"hyperagent.models.generated.{path.stem}")
