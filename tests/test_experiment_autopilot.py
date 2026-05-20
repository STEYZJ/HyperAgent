import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.agents import (
    CoordinatorAgent,
    ExecutableExperimentCouncilAgent,
    ExperimentAutopilotAgent,
    ExperimentCouncilAgent,
)
from hyperagent.cli import main
from hyperagent.core.io import read_json, read_yaml, write_json
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.schemas import (
    ExperimentCouncilRun,
    ExperimentCycle,
    ExperimentDiagnosis,
    ExperimentPlan,
    LLMResponse,
    ParameterProposal,
)


class _FakeCouncilLLMClient:
    def __init__(self):
        self.calls = 0

    def send(self, spec, messages, model=None, **kwargs):
        self.calls += 1
        return LLMResponse(
            provider=spec.name,
            model=model or spec.default_model,
            content=(
                '{"decision":"llm_reviewed","rationale":"LLM checked the role vote.",'
                '"confidence":0.66,"warnings":[]}'
            ),
            usage={
                "prompt_tokens": 9,
                "completion_tokens": 4,
                "total_tokens": 13,
                "prompt_cache_hit_tokens": 5,
                "prompt_cache_miss_tokens": 4,
            },
        )


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
            self.assertTrue(Path(cycle.council_run_path).exists())
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
            self.assertEqual(len(council["votes"]), 5)
            council_run = ExperimentCouncilRun.from_dict(
                read_json(Path(cycle.council_run_path))
            )
            self.assertEqual(len(council_run.role_runs), 5)
            self.assertFalse(council_run.llm_enabled)

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
                self.assertIn("council_run:", output)
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

    def test_static_council_mode_keeps_legacy_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=41)

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
                seed=41,
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
                council_mode="static",
            )

            self.assertIsNone(cycle.council_run_path)
            self.assertTrue(Path(cycle.council_path).exists())

    def test_executable_council_registry_override_and_repeated_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace_dir = root / ".hyperagent"
            RuntimeExtensionStore(workspace_dir).add_subagent(
                "MetricLead",
                "result_analyst",
                tools=["read_result"],
                profile="reasonix-deep",
            )
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=43)
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
                seed=43,
            )
            result = agent.run(plan)
            diagnosis = ExperimentAutopilotAgent().diagnose(plan, result, audit)
            proposal = ParameterProposal(
                parameter="split.train_ratio",
                old_value=0.05,
                new_value=0.08,
                rationale="Increase train ratio to improve weak classes.",
                expected_effect="More labeled samples should improve stability.",
            )
            old_cycle = ExperimentCycle(
                cycle_id="old",
                created_at="now",
                status="completed",
                previous_plan_path="plan.yaml",
                previous_result_path="result.json",
                audit_path="audit.json",
                cycle_dir=str(root / "old"),
                diagnosis_path="diagnosis.json",
                proposals_path="proposals.json",
                next_plan_path="next.yaml",
                selected_proposal=proposal,
            )

            council_run = ExecutableExperimentCouncilAgent(
                workspace_dir=workspace_dir,
            ).review(
                diagnosis,
                [proposal],
                audit,
                plan,
                [old_cycle],
                max_repeated_parameter=5,
            )

            self.assertEqual(council_run.role_runs[0].agent_name, "MetricLead")
            self.assertEqual(council_run.role_runs[0].profile, "reasonix-deep")
            self.assertEqual(council_run.final_decision.action, "pause")
            self.assertIn("split.train_ratio", council_run.final_decision.rejected_parameters)

    def test_executable_council_high_oa_without_seed_pauses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=47)
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
                seed=47,
            )
            diagnosis = ExperimentDiagnosis(
                experiment_name=plan.experiment_name,
                objective="maximize_oa",
                overall_accuracy=0.96,
                average_accuracy=0.95,
                kappa=0.94,
            )
            proposal = ParameterProposal(
                parameter="model.hidden_layers",
                old_value=[64],
                new_value=[128],
                rationale="Increase model capacity.",
                expected_effect="May fit harder classes.",
            )

            council_run = ExecutableExperimentCouncilAgent().review(
                diagnosis,
                [proposal],
                audit,
                plan,
                [],
                target_oa=0.9,
            )

            self.assertEqual(council_run.final_decision.action, "pause")
            self.assertIsNone(council_run.final_decision.selected_parameter)

    def test_llm_council_budget_and_usage_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace_dir = root / ".hyperagent"
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=53)
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
                seed=53,
            )
            result = agent.run(plan)
            diagnosis = ExperimentAutopilotAgent().diagnose(plan, result, audit)
            proposal = ParameterProposal(
                parameter="seed",
                old_value=53,
                new_value=54,
                rationale="Estimate variance.",
                expected_effect="Check stability.",
                evidence=diagnosis.evidence,
            )
            fake_client = _FakeCouncilLLMClient()

            council_run = ExecutableExperimentCouncilAgent(
                workspace_dir=workspace_dir,
                llm_store=LLMProviderStore(workspace_dir),
                llm_client=fake_client,
            ).review(
                diagnosis,
                [proposal],
                audit,
                plan,
                [],
                llm_enabled=True,
                llm_budget=1,
            )

            self.assertEqual(fake_client.calls, 1)
            self.assertEqual(council_run.budget_used, 1)
            self.assertTrue(council_run.role_runs[0].llm_used)
            self.assertFalse(council_run.role_runs[1].llm_used)
            usage_path = workspace_dir / "usage" / "llm_usage.jsonl"
            self.assertTrue(usage_path.exists())


if __name__ == "__main__":
    unittest.main()
