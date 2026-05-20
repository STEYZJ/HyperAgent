import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.agents import CoordinatorAgent
from hyperagent.cli import main
from hyperagent.core.io import read_json
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.schemas import ExperimentSuiteResult
from hyperagent.training.experiment_suite import ExperimentSuiteRunner


class ExperimentSuiteTest(unittest.TestCase):
    def test_suite_runner_executes_multiple_seeds_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=41)
            plan = self._build_plan(root, data_root, seed=41)

            suite = ExperimentSuiteRunner().run(
                plan,
                seeds=[41, 42],
                output_dir=root / "suite",
                suite_name="synthetic_suite",
            )

            self.assertEqual(suite.run_count, 2)
            self.assertEqual(suite.seeds, [41, 42])
            self.assertEqual(
                len(suite.metrics_summary["overall_accuracy"]["values"]),
                2,
            )
            self.assertTrue((root / "suite" / "suite.json").exists())
            self.assertTrue((root / "suite" / "suite_metrics.csv").exists())
            self.assertTrue((root / "suite" / "suite_report.md").exists())
            self.assertTrue(Path(suite.best_result_path).exists())

            loaded = ExperimentSuiteResult.from_dict(
                read_json(root / "suite" / "suite.json")
            )
            self.assertEqual(loaded.run_count, 2)
            self.assertGreaterEqual(
                loaded.metrics_summary["overall_accuracy"]["mean"],
                0.0,
            )

    def test_run_suite_cli(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=51)
            self._build_plan(root, data_root, seed=51)
            os.chdir(root)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "run-suite",
                                "--config",
                                str(root / "plan.yaml"),
                                "--seeds",
                                "51,52",
                                "--output-dir",
                                str(root / "cli_suite"),
                            ]
                        ),
                        0,
                    )
                output = buffer.getvalue()
                self.assertIn("Wrote suite:", output)
                self.assertIn("oa_mean:", output)
                self.assertTrue((root / "cli_suite" / "suite.json").exists())
            finally:
                os.chdir(old_cwd)

    def _build_plan(self, root: Path, data_root: Path, seed: int):
        agent = CoordinatorAgent()
        audit = agent.audit(data_root, root / "audit.json")
        spectral = agent.analyze(audit, root / "spectral.json")
        recommendation = agent.recommend(audit, spectral, root / "recommendation.json")
        return agent.plan(
            audit,
            spectral,
            recommendation,
            root / "plan.yaml",
            root / "run",
            seed=seed,
        )


if __name__ == "__main__":
    unittest.main()
