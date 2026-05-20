import tempfile
import unittest
from pathlib import Path

from hyperagent.agents import BenchmarkAgent
from hyperagent.core.io import read_json, write_yaml
from hyperagent.data.synthetic import write_synthetic_mat


class BenchmarkAgentTest(unittest.TestCase):
    def test_benchmark_matrix_runs_catalogued_dataset_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "datasets" / "Synthetic"
            write_synthetic_mat(data_root, seed=61)
            catalog_path = root / "datasets.yaml"
            write_yaml(
                catalog_path,
                {
                    "datasets": {
                        "Synthetic": {
                            "local_example": str(data_root),
                            "source_url": "https://example.org/synthetic",
                            "notes": "unit test dataset",
                        }
                    }
                },
            )

            matrix = BenchmarkAgent().run_matrix(
                catalog_path,
                dataset_names=["Synthetic"],
                reports_root=root / "reports",
                experiments_root=root / "experiments",
                seeds=[61, 62],
                run_suite=True,
            )

            self.assertEqual(len(matrix["datasets"]), 1)
            row = matrix["datasets"][0]
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["dataset"], "Synthetic")
            self.assertEqual(row["reader_name"], "mat")
            self.assertTrue(Path(row["audit_path"]).exists())
            self.assertTrue(Path(row["plan_path"]).exists())
            self.assertTrue(Path(row["suite_path"]).exists())
            self.assertTrue(Path(row["suite_report_path"]).exists())
            self.assertIn("oa_mean", row)

            saved = read_json(root / "reports" / "benchmark_matrix.json")
            self.assertEqual(saved["datasets"][0]["status"], "completed")
            self.assertTrue((root / "reports" / "benchmark_matrix.md").exists())


if __name__ == "__main__":
    unittest.main()
