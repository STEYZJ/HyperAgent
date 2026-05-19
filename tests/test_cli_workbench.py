import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.cli import main
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.runtime.workspace import HyperAgentWorkspace


class CLIWorkbenchTest(unittest.TestCase):
    def test_cli_task_workbench(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets"
            write_synthetic_mat(dataset_root / "Synthetic", seed=13)
            os.chdir(root)
            try:
                self.assertEqual(
                    main(["init", "--dataset-root", str(dataset_root)]),
                    0,
                )
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "task-create",
                                "--goal",
                                "build a reproducible HSI baseline",
                                "--dataset",
                                "Synthetic",
                                "--keywords",
                                "hyperspectral,baseline",
                            ]
                        ),
                        0,
                    )
                task_id = buffer.getvalue().strip().splitlines()[-1]
                self.assertTrue(task_id)
                self.assertEqual(main(["task-run", "--task-id", task_id]), 0)

                workspace = HyperAgentWorkspace(root)
                task = workspace.load_task(task_id)
                self.assertEqual(task.status, "completed")
                self.assertIn("audit", task.artifacts)
                self.assertIn("auto_experiment_agenda", task.artifacts)
                self.assertTrue(Path(task.artifacts["experiment_plan"]).exists())

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(main(["task-show", "--task-id", task_id]), 0)
                self.assertIn("build a reproducible HSI baseline", buffer.getvalue())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()

