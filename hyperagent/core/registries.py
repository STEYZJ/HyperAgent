"""Global registries for replaceable HyperAgent components."""

from hyperagent.core.interfaces import (
    ClassifierFactory,
    DatasetReader,
    Evaluator,
    LiteratureProvider,
    ModelRecommender,
    SpectralAnalyzer,
)
from hyperagent.core.registry import Registry

dataset_reader_registry: Registry[DatasetReader] = Registry("dataset_reader")
analyzer_registry: Registry[SpectralAnalyzer] = Registry("analyzer")
evaluator_registry: Registry[Evaluator] = Registry("evaluator")
model_registry: Registry[ClassifierFactory] = Registry("model")
model_recommender_registry: Registry[ModelRecommender] = Registry("model_recommender")
literature_provider_registry: Registry[LiteratureProvider] = Registry("literature_provider")
