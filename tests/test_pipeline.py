import tempfile
import unittest
from pathlib import Path

from hyperagent.agents import CoordinatorAgent
from hyperagent.data.synthetic import write_synthetic_mat


class PipelineTest(unittest.TestCase):
    def test_synthetic_pipeline_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=7)

            agent = CoordinatorAgent()
            audit = agent.audit(data_root, root / "audit.json")
            self.assertGreater(audit.band_count, 1)
            self.assertGreater(audit.class_count, 1)

            spectral = agent.analyze(audit, root / "spectral.json")
            recommendation = agent.recommend(audit, spectral, root / "recommendation.json")
            plan = agent.plan(
                audit,
                spectral,
                recommendation,
                root / "experiment.yaml",
                root / "run",
                seed=7,
            )
            result = agent.run(plan)
            report_path = agent.write_report(result, Path(result.experiment_dir) / "report.md")

            self.assertTrue((root / "audit.json").exists())
            self.assertTrue((root / "experiment.yaml").exists())
            self.assertTrue((Path(result.experiment_dir) / "result.json").exists())
            self.assertTrue(report_path.exists())
            self.assertGreaterEqual(result.evaluation.overall_accuracy, 0.0)
            self.assertLessEqual(result.evaluation.overall_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()

