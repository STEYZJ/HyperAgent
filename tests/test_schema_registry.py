import unittest

from hyperagent.core.registry import Registry
from hyperagent.schemas import (
    DatasetAudit,
    EvaluationReport,
    ExperimentPlan,
    ExperimentResult,
    ModelConfig,
    ModelRecommendation,
    PreprocessingConfig,
    SplitConfig,
)
from hyperagent.schemas.recommendation import ModelCandidate


class SchemaRegistryTest(unittest.TestCase):
    def test_registry_register_and_get(self):
        registry = Registry("unit")
        registry.register("demo", 3)
        self.assertEqual(registry.get("demo"), 3)
        self.assertTrue(registry.has("demo"))

    def test_schema_round_trip(self):
        audit = DatasetAudit(
            data_root="data",
            dataset_name="demo",
            cube_path="cube.mat",
            label_path="label.mat",
            cube_shape=[4, 5, 6],
            label_shape=[4, 5],
            band_count=6,
            class_count=2,
            labeled_pixel_count=10,
            unlabeled_pixel_count=10,
            class_distribution={"1": 5, "2": 5},
            has_nan=False,
            has_inf=False,
            dtype="float32",
            reader_name="mat",
        )
        self.assertEqual(DatasetAudit.from_dict(audit.to_dict()).band_count, 6)

        recommendation = ModelRecommendation(
            recommended_model="svm",
            candidates=[ModelCandidate("svm", 0.9, "small sample", {"C": 10})],
        )
        self.assertEqual(
            ModelRecommendation.from_dict(recommendation.to_dict()).recommended_model,
            "svm",
        )

        plan = ExperimentPlan(
            experiment_name="demo",
            dataset_root="data",
            output_dir="out",
            seed=42,
            reader_name="mat",
            split=SplitConfig(),
            preprocessing=PreprocessingConfig(),
            model=ModelConfig(),
        )
        self.assertEqual(ExperimentPlan.from_dict(plan.to_dict()).model.name, "svm")

        evaluation = EvaluationReport(
            overall_accuracy=1.0,
            average_accuracy=1.0,
            kappa=1.0,
            labels=[1],
            per_class_accuracy={"1": 1.0},
            confusion_matrix=[[3]],
        )
        result = ExperimentResult(
            experiment_name="demo",
            experiment_dir="out",
            model_name="svm",
            seed=42,
            train_samples=3,
            test_samples=3,
            evaluation=evaluation,
        )
        self.assertEqual(ExperimentResult.from_dict(result.to_dict()).evaluation.kappa, 1.0)


if __name__ == "__main__":
    unittest.main()

