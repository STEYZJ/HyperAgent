import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.cli import main
from hyperagent.core.io import read_json
from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.action_repair import ActionRepairPipeline
from hyperagent.runtime.checkpoints import CheckpointStore, paths_from_unified_diff
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.events import RuntimeEventLog
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.semantic_index import SemanticIndexStore
from hyperagent.runtime.skills import SkillStore
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import AgentActionRun, LLMResponse


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def send(self, spec, messages, model=None, **kwargs):
        del messages, kwargs
        response = self.responses.pop(0)
        if isinstance(response, LLMResponse):
            return response
        return LLMResponse(
            provider=spec.name,
            model=model or spec.default_model,
            content=str(response),
        )


class ReasonixRuntimeTest(unittest.TestCase):
    def test_native_tool_call_and_cache_first_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            (root / "notes.md").write_text("cache first works\n", encoding="utf-8")
            (root / "other.md").write_text("parallel safe works\n", encoding="utf-8")
            sessions = ConversationStore(workspace.workspace_dir)
            session = sessions.new("native")
            providers = LLMProviderStore(workspace.workspace_dir)
            fake = FakeLLMClient(
                [
                    LLMResponse(
                        provider="deepseek",
                        model="deepseek-v4-flash",
                        content="",
                        tool_calls=[
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "notes.md", "max_lines": 1}',
                                }
                            },
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"path": "other.md", "max_lines": 1},
                                }
                            }
                        ],
                        usage={
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                            "prompt_cache_hit_tokens": 5,
                            "prompt_cache_miss_tokens": 5,
                        },
                    ),
                    '{"action": "final", "final": "done"}',
                ]
            )

            run = AgentActionLoop(sessions, providers, workspace, llm_client=fake).run(
                session.session_id,
                "deepseek",
                "inspect notes",
                max_steps=2,
                loop_mode="cache-first",
            )

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.steps[0].parse_source, "native_tool_calls")
            self.assertEqual(run.steps[0].tool_name, "read_file")
            self.assertEqual(len([step for step in run.steps if step.action == "tool"]), 2)
            self.assertTrue(run.stable_prefix_hash)
            self.assertTrue(Path(run.event_log_path).exists())
            loaded = AgentActionRun.from_dict(read_json(Path(run.run_dir) / "action_run.json"))
            self.assertEqual(loaded.loop_mode, "cache-first")
            self.assertEqual(loaded.usage_event_ids, run.usage_event_ids)
            self.assertEqual(loaded.budget_used_tokens, 12)
            self.assertEqual(loaded.prompt_cache_hit_tokens, 5)
            self.assertEqual(loaded.prompt_cache_miss_tokens, 5)
            self.assertEqual(loaded.cache_hit_ratio, 0.5)
            events = RuntimeEventLog(workspace.workspace_dir).records(run_id=run.run_id)
            self.assertTrue(any(event.event_type == "action_loop.response" for event in events))
            self.assertTrue(any(event.event_type == "action_loop.step" for event in events))

    def test_cumulative_token_budget_pauses_action_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            sessions = ConversationStore(workspace.workspace_dir)
            session = sessions.new("budget")
            providers = LLMProviderStore(workspace.workspace_dir)
            fake = FakeLLMClient(
                [
                    LLMResponse(
                        provider="deepseek",
                        model="deepseek-v4-flash",
                        content='{"action": "tool", "tool_name": "search_code", "args": {"query": "missing"}}',
                        usage={"total_tokens": 6},
                    ),
                    LLMResponse(
                        provider="deepseek",
                        model="deepseek-v4-flash",
                        content='{"action": "final", "final": "done"}',
                        usage={"total_tokens": 6},
                    ),
                ]
            )
            run = AgentActionLoop(sessions, providers, workspace, llm_client=fake).run(
                session.session_id,
                "deepseek",
                "inspect budget",
                max_steps=2,
                token_budget=10,
            )

            self.assertEqual(run.status, "paused")
            self.assertTrue(run.budget_exhausted)
            self.assertEqual(run.budget_used_tokens, 12)
            self.assertEqual(len(run.usage_event_ids), 2)
            events = RuntimeEventLog(workspace.workspace_dir).records(run_id=run.run_id)
            self.assertTrue(any(event.event_type == "action_loop.paused" for event in events))

    def test_cache_first_stable_prefix_hash_ignores_instruction_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            sessions = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            fake = FakeLLMClient(
                [
                    '{"action": "final", "final": "first"}',
                    '{"action": "final", "final": "second"}',
                ]
            )
            first = AgentActionLoop(sessions, providers, workspace, llm_client=fake).run(
                sessions.new("first").session_id,
                "deepseek",
                "inspect file A",
                max_steps=1,
                loop_mode="cache-first",
            )
            second = AgentActionLoop(sessions, providers, workspace, llm_client=fake).run(
                sessions.new("second").session_id,
                "deepseek",
                "inspect file B with different words",
                max_steps=1,
                loop_mode="cache-first",
            )

            self.assertTrue(first.stable_prefix_hash)
            self.assertEqual(first.stable_prefix_hash, second.stable_prefix_hash)

    def test_reasoning_scavenge_and_storm_breaker(self):
        response = LLMResponse(
            provider="deepseek",
            model="deepseek-v4-flash",
            content="I will use a tool.",
            reasoning_content='{"action": "tool", "tool_name": "search_code", "args": {"query": "abc"}}',
        )
        parsed = ActionRepairPipeline().parse(response)
        self.assertEqual(parsed.source, "reasoning_content")
        self.assertEqual(parsed.action["tool_name"], "search_code")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            sessions = ConversationStore(workspace.workspace_dir)
            session = sessions.new("storm")
            providers = LLMProviderStore(workspace.workspace_dir)
            repeated = '{"action": "tool", "tool_name": "search_code", "args": {"query": "missing"}}'
            fake = FakeLLMClient([repeated, repeated, repeated])
            run = AgentActionLoop(sessions, providers, workspace, llm_client=fake).run(
                session.session_id,
                "deepseek",
                "repeat search",
                max_steps=3,
                storm_max_repeats=1,
            )
            self.assertEqual(run.steps[1].tool_result.status, "blocked")
            self.assertIn("storm", " ".join(run.steps[1].tool_result.warnings))

    def test_repair_events_for_reasoning_and_direct_tool_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            sessions = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            fake = FakeLLMClient(
                [
                    LLMResponse(
                        provider="deepseek",
                        model="deepseek-v4-flash",
                        content="I will use a tool.",
                        reasoning_content='{"action": "tool", "tool_name": "search_code", "args": {"query": "abc"}}',
                    ),
                    '{"action": "framework_command", "command": "status"}',
                ]
            )
            reasoning_run = AgentActionLoop(sessions, providers, workspace, llm_client=fake).run(
                sessions.new("reasoning").session_id,
                "deepseek",
                "use reasoning tool",
                max_steps=1,
            )
            direct_run = AgentActionLoop(sessions, providers, workspace, llm_client=fake).run(
                sessions.new("direct").session_id,
                "deepseek",
                "check status",
                max_steps=1,
            )

            reasoning_events = RuntimeEventLog(workspace.workspace_dir).records(run_id=reasoning_run.run_id)
            direct_events = RuntimeEventLog(workspace.workspace_dir).records(run_id=direct_run.run_id)
            self.assertTrue(any(event.event_type == "action_loop.repair" and "scavenged" in event.message for event in reasoning_events))
            self.assertTrue(any(event.event_type == "action_loop.repair" and "Normalized direct tool action" in event.message for event in direct_events))

    def test_cli_replay_and_stats_json_include_runtime_summaries(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            RuntimeEventLog(workspace.workspace_dir).append(
                "action_loop.step",
                source="action_loop",
                run_id="run-1",
                tool_name="read_file",
                status="ok",
                payload={"step_index": 1, "tool_risk_level": "read"},
            )
            os.chdir(root)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(main(["replay", "--run-id", "run-1", "--json"]), 0)
                replay = json.loads(buffer.getvalue())
                self.assertEqual(replay["summary"]["run_count"], 1)
                self.assertEqual(replay["summary"]["by_tool"]["read_file"], 1)

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(main(["stats", "--json"]), 0)
                stats = json.loads(buffer.getvalue())
                self.assertEqual(stats["run_count"], 1)
                self.assertEqual(stats["by_tool"]["read_file"], 1)
                self.assertEqual(stats["by_source"]["action_loop"], 1)
            finally:
                os.chdir(old_cwd)

    def test_skill_frontmatter_and_checkpoint_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            skill_dir = workspace.workspace_dir / "skills" / "demo"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo\ndescription: Demo skill\nrunAs: inline\nallowed-tools: read_file\n---\nHello $ARGUMENTS",
                encoding="utf-8",
            )
            skill = SkillStore([workspace.workspace_dir / "skills"]).render("demo", "HSI")
            self.assertEqual(skill.run_as, "inline")
            self.assertIn("Hello HSI", skill.body)
            self.assertEqual(skill.allowed_tools, ["read_file"])

            target = root / "a.txt"
            target.write_text("before\n", encoding="utf-8")
            store = CheckpointStore(root, workspace.workspace_dir)
            checkpoint = store.create(["a.txt"], reason="test")
            target.write_text("after\n", encoding="utf-8")
            store.restore(checkpoint.checkpoint_id)
            self.assertEqual(target.read_text(encoding="utf-8"), "before\n")
            patch_paths = paths_from_unified_diff("--- a/a.txt\n+++ b/a.txt\n@@\n-before\n+after")
            self.assertEqual(patch_paths, ["a.txt"])

            (root / "paper.md").write_text("spectral mamba benchmark", encoding="utf-8")
            index = SemanticIndexStore(root, workspace.workspace_dir)
            payload = index.build(["paper.md"])
            self.assertEqual(len(payload["documents"]), 1)
            results = index.search("spectral benchmark")
            self.assertEqual(results[0]["path"], "paper.md")


if __name__ == "__main__":
    unittest.main()
