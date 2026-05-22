import tempfile
import unittest
from pathlib import Path

from hyperagent.core.io import read_json
from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import AgentActionRun, LLMResponse


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def send(
        self,
        spec,
        messages,
        model=None,
        temperature=0.2,
        max_tokens=None,
        **kwargs,
    ):
        del temperature, max_tokens, kwargs
        self.messages.append(list(messages))
        return LLMResponse(
            provider=spec.name,
            model=model or spec.default_model,
            content=self.responses.pop(0),
        )


class AgentActionLoopTest(unittest.TestCase):
    def test_action_loop_executes_tool_and_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            (root / "notes.md").write_text("benchmark matrix ready\n", encoding="utf-8")
            session_store = ConversationStore(workspace.workspace_dir)
            session = session_store.new("action")
            llm_store = LLMProviderStore(workspace.workspace_dir)
            fake_client = FakeLLMClient(
                [
                    (
                        '{"thought": "inspect note", "action": "tool", '
                        '"tool_name": "read_file", '
                        '"args": {"path": "notes.md", "max_lines": 1}}'
                    ),
                    (
                        '{"thought": "done", "action": "final", '
                        '"final": "The benchmark note was inspected."}'
                    ),
                ]
            )

            run = AgentActionLoop(
                session_store,
                llm_store,
                workspace,
                llm_client=fake_client,
            ).run(
                session.session_id,
                provider="deepseek",
                instruction="Inspect the benchmark note.",
                max_steps=2,
            )

            self.assertEqual(run.status, "completed")
            self.assertEqual(len(run.steps), 2)
            self.assertEqual(run.steps[0].tool_name, "read_file")
            self.assertEqual(run.steps[0].tool_result.status, "ok")
            self.assertIn("benchmark matrix ready", run.steps[0].tool_result.content)
            self.assertEqual(run.final_response, "The benchmark note was inspected.")
            self.assertTrue((Path(run.run_dir) / "action_run.json").exists())

            loaded = AgentActionRun.from_dict(read_json(Path(run.run_dir) / "action_run.json"))
            self.assertEqual(loaded.status, "completed")
            saved_session = session_store.load(session.session_id)
            self.assertTrue(any(message.role == "tool" for message in saved_session.messages))

    def test_action_loop_treats_plain_text_as_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            session_store = ConversationStore(workspace.workspace_dir)
            session = session_store.new("plain")
            llm_store = LLMProviderStore(workspace.workspace_dir)
            run = AgentActionLoop(
                session_store,
                llm_store,
                workspace,
                llm_client=FakeLLMClient(["plain final answer"]),
            ).run(
                session.session_id,
                provider="deepseek",
                instruction="Answer directly.",
                max_steps=1,
            )

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.final_response, "plain final answer")
            self.assertIn("not valid JSON", run.steps[0].warnings[0])

    def test_action_loop_executes_framework_command_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            session_store = ConversationStore(workspace.workspace_dir)
            session = session_store.new("framework")
            llm_store = LLMProviderStore(workspace.workspace_dir)
            fake_client = FakeLLMClient(
                [
                    (
                        '{"thought": "inspect HyperAgent status", "action": "tool", '
                        '"tool_name": "framework_command", '
                        '"args": {"command": "status"}}'
                    ),
                    (
                        '{"thought": "status inspected", "action": "final", '
                        '"final": "HyperAgent status was queried."}'
                    ),
                ]
            )

            run = AgentActionLoop(
                session_store,
                llm_store,
                workspace,
                llm_client=fake_client,
            ).run(
                session.session_id,
                provider="deepseek",
                instruction="Can you check the framework status?",
                max_steps=2,
            )

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.steps[0].tool_name, "framework_command")
            self.assertEqual(run.steps[0].tool_result.status, "ok")
            self.assertIn('"initialized": true', run.steps[0].tool_result.content)
            self.assertEqual(run.final_response, "HyperAgent status was queried.")


if __name__ == "__main__":
    unittest.main()
