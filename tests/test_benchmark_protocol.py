import json
import tempfile
import unittest
from pathlib import Path

from hyperagent.agents import BenchmarkAgent
from hyperagent.core.bootstrap import bootstrap_default_components
from hyperagent.core.io import read_json, write_yaml
from hyperagent.core.registries import model_registry
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.training.benchmark_protocol import BenchmarkProtocolStore


class BenchmarkProtocolTest(unittest.TestCase):
    def test_protocol_records_fixed_split_fingerprints_without_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_a = root / "datasets" / "SyntheticA"
            data_b = root / "datasets" / "SyntheticB"
            write_synthetic_mat(data_a, seed=71)
            write_synthetic_mat(data_b, seed=72)
            catalog_path = root / "datasets.yaml"
            write_yaml(
                catalog_path,
                {
                    "datasets": {
                        "SyntheticA": {
                            "local_example": str(data_a),
                            "source_url": "https://example.org/a",
                        },
                        "SyntheticB": {
                            "local_example": str(data_b),
                            "source_url": "https://example.org/b",
                        },
                    }
                },
            )
            store = BenchmarkProtocolStore(root / "reports" / "protocol")

            protocol = store.create(
                catalog_path=catalog_path,
                dataset_names=["SyntheticA", "SyntheticB"],
                seeds=[11, 12],
                baselines=["svm", "knn"],
            )
            again = store.create(
                catalog_path=catalog_path,
                dataset_names=["SyntheticA", "SyntheticB"],
                seeds=[11, 12],
                baselines=["svm", "knn"],
            )

            self.assertEqual(protocol["baselines"], ["svm", "knn"])
            self.assertEqual(len(protocol["datasets"]), 2)
            self.assertEqual(protocol["datasets"][0]["splits"][0]["fingerprint"], again["datasets"][0]["splits"][0]["fingerprint"])
            self.assertNotIn("train_indices", json.dumps(protocol))
            self.assertNotIn("test_indices", json.dumps(protocol))
            self.assertTrue(store.path.exists())
            self.assertTrue((store.root / "benchmark_protocol.md").exists())

            matrix = BenchmarkAgent().run_matrix(
                catalog_path,
                dataset_names=None,
                reports_root=root / "reports" / "matrix",
                experiments_root=root / "experiments" / "matrix",
                seeds=[999],
                run_suite=False,
                protocol_path=store.path,
            )

            self.assertEqual(len(matrix["datasets"]), 4)
            self.assertTrue(all(row["status"] == "planned" for row in matrix["datasets"]))
            self.assertTrue(all(row["split_fingerprints"] for row in matrix["datasets"]))
            self.assertIn("knn", (root / "reports" / "matrix" / "benchmark_matrix.md").read_text(encoding="utf-8"))

    def test_protocol_matrix_can_run_a_baseline_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "datasets" / "Synthetic"
            write_synthetic_mat(data_root, seed=81)
            catalog_path = root / "datasets.yaml"
            write_yaml(
                catalog_path,
                {
                    "datasets": {
                        "Synthetic": {
                            "local_example": str(data_root),
                            "source_url": "https://example.org/synthetic",
                        }
                    }
                },
            )
            store = BenchmarkProtocolStore(root / "reports" / "protocol")
            store.create(
                catalog_path=catalog_path,
                dataset_names=["Synthetic"],
                seeds=[81, 82],
                baselines=["svm"],
            )

            matrix = BenchmarkAgent().run_matrix(
                catalog_path,
                dataset_names=None,
                reports_root=root / "reports" / "matrix",
                experiments_root=root / "experiments" / "matrix",
                seeds=[81, 82],
                run_suite=True,
                protocol_path=store.path,
            )

            row = matrix["datasets"][0]
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["baseline"], "svm")
            self.assertTrue(Path(row["suite_path"]).exists())
            self.assertIn("oa_mean", row)

    def test_strong_sklearn_baselines_are_registered(self):
        bootstrap_default_components()

        self.assertTrue(model_registry.has("random_forest"))
        self.assertTrue(model_registry.has("knn"))


if __name__ == "__main__":
    unittest.main()
