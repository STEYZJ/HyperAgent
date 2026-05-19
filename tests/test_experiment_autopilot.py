import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.agents import (
    CoordinatorAgent,
    ExperimentAutopilotAgent,
    ExperimentCouncilAgent,
)
from hyperagent.cli import main
from hyperagent.core.io import read_json, read_yaml, write_json
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.schemas import ExperimentCycle, ExperimentPlan, ParameterProposal


class ExperimentAutopilotTest(unittest.TestCase):
    def test_autopilot_builds_diagnosis_and_next_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=23)

            agent = CoordinatorAgent()
            audit = agent.audit(data_root, root / "audit.json")
            spectral = agent.analyze(audit, root / "spectral.json")
            recommendation = agent.recommend(audit, spectral, root / "recommendation.json")
            plan = agent.plan(
                audit,
                spectral,
                recommendation,
                root / "plan.yaml",
                root / "run",
                seed=23,
            )
            result = agent.run(plan)

            cycle = ExperimentAutopilotAgent().run_cycle(
                plan,
                result,
                audit,
                previous_plan_path=root / "plan.yaml",
                previous_result_path=Path(result.experiment_dir) / "result.json",
                audit_path=root / "audit.json",
                output_root=root / "cycles",
                run_next=False,
            )

            self.assertEqual(cycle.status, "planned")
            self.assertTrue(Path(cycle.diagnosis_path).exists())
            self.assertTrue(Path(cycle.proposals_path).exists())
            self.assertTrue(Path(cycle.council_path).exists())
            self.assertTrue(Path(cycle.next_plan_path).exists())
            self.assertIsNotNone(cycle.selected_proposal)

            diagnosis = read_json(Path(cycle.diagnosis_path))
            self.assertIn("findings", diagnosis)
            self.assertGreaterEqual(len(diagnosis["findings"]), 1)

            next_plan = ExperimentPlan.from_dict(read_yaml(Path(cycle.next_plan_path)))
            self.assertIn("autopilot", next_plan.metadata)
            self.assertNotEqual(next_plan.output_dir, plan.output_dir)

            council = read_json(Path(cycle.council_path))
            self.assertIn("votes", council)
            self.assertGreaterEqual(len(council["votes"]), 3)

    def test_experiment_cycle_cli_runs_next_experiment(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=29)
            os.chdir(root)
            try:
                agent = CoordinatorAgent()
                audit = agent.audit(data_root, root / "audit.json")
                spectral = agent.analyze(audit, root / "spectral.json")
                recommendation = agent.recommend(audit, spectral, root / "recommendation.json")
                plan = agent.plan(
                    audit,
                    spectral,
                    recommendation,
                    root / "plan.yaml",
                    root / "run",
                    seed=29,
                )
                result = agent.run(plan)
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "experiment-cycle",
                                "--plan",
                                str(root / "plan.yaml"),
                                "--result",
                                str(Path(result.experiment_dir) / "result.json"),
                                "--audit",
                                str(root / "audit.json"),
                                "--output-root",
                                str(root / "cycles"),
                                "--run-next",
                            ]
                        ),
                        0,
                    )
                output = buffer.getvalue()
                self.assertIn("cycle_id:", output)
                self.assertIn("next_result:", output)
                self.assertIn("council:", output)
                cycle_dirs = sorted((root / "cycles").iterdir())
                self.assertEqual(len(cycle_dirs), 1)
                cycle = ExperimentCycle.from_dict(read_json(cycle_dirs[0] / "cycle.json"))
                self.assertEqual(cycle.status, "completed")
                self.assertTrue(Path(cycle.next_result_path).exists())
                self.assertTrue(Path(cycle.report_path).exists())
            finally:
                os.chdir(old_cwd)

    def test_council_rejects_repeated_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=31)

            agent = CoordinatorAgent()
            audit = agent.audit(data_root, root / "audit.json")
            spectral = agent.analyze(audit, root / "spectral.json")
            recommendation = agent.recommend(audit, spectral, root / "recommendation.json")
            plan = agent.plan(
                audit,
                spectral,
                recommendation,
                root / "plan.yaml",
                root / "run",
                seed=31,
            )
            result = agent.run(plan)
            autopilot = ExperimentAutopilotAgent()
            diagnosis = autopilot.diagnose(plan, result, audit)
            proposal = ParameterProposal(
                parameter="seed",
                old_value=31,
                new_value=32,
                rationale="Repeat seed check.",
                expected_effect="Estimate variance.",
            )
            history = []
            for index in range(2):
                cycle = ExperimentCycle(
                    cycle_id=f"old-{index}",
                    created_at="now",
                    status="completed",
                    previous_plan_path="plan.yaml",
                    previous_result_path="result.json",
                    audit_path="audit.json",
                    cycle_dir=str(root / f"old-{index}"),
                    diagnosis_path="diagnosis.json",
                    proposals_path="proposals.json",
                    next_plan_path="next.yaml",
                    selected_proposal=proposal,
                )
                history.append(cycle)

            decision = ExperimentCouncilAgent().review(
                diagnosis,
                [proposal],
                audit,
                plan,
                history,
                max_repeated_parameter=2,
            )

            self.assertEqual(decision.action, "pause")
            self.assertIn("seed", decision.rejected_parameters)
            self.assertTrue(decision.warnings)

    def test_cycle_pauses_when_history_repeats_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=37)

            agent = CoordinatorAgent()
            audit = agent.audit(data_root, root / "audit.json")
            spectral = agent.analyze(audit, root / "spectral.json")
            recommendation = agent.recommend(audit, spectral, root / "recommendation.json")
            plan = agent.plan(
                audit,
                spectral,
                recommendation,
                root / "plan.yaml",
                root / "run",
                seed=37,
            )
            result = agent.run(plan)
            output_root = root / "cycles"
            proposal = ParameterProposal(
                parameter="seed",
                old_value=37,
                new_value=38,
                rationale="Repeat seed check.",
                expected_effect="Estimate variance.",
            )
            for index in range(2):
                cycle_dir = output_root / f"old-{index}"
                cycle_dir.mkdir(parents=True)
                write_json(
                    cycle_dir / "cycle.json",
                    ExperimentCycle(
                        cycle_id=f"old-{index}",
                        created_at="now",
                        status="completed",
                        previous_plan_path="plan.yaml",
                        previous_result_path="result.json",
                        audit_path="audit.json",
                        cycle_dir=str(cycle_dir),
                        diagnosis_path="diagnosis.json",
                        proposals_path="proposals.json",
                        next_plan_path="next.yaml",
                        selected_proposal=proposal,
                    ),
                )

            cycle = ExperimentAutopilotAgent().run_cycle(
                plan,
                result,
                audit,
                previous_plan_path=root / "plan.yaml",
                previous_result_path=Path(result.experiment_dir) / "result.json",
                audit_path=root / "audit.json",
                output_root=output_root,
                run_next=True,
                max_repeated_parameter=2,
            )

            self.assertEqual(cycle.status, "paused")
            self.assertIsNone(cycle.next_result_path)
            council = read_json(Path(cycle.council_path))
            self.assertEqual(council["action"], "pause")


if __name__ == "__main__":
    unittest.main()
