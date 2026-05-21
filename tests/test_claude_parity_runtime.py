import tempfile
import unittest
from pathlib import Path

from hyperagent.core.io import write_json
from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.commands import SlashCommandStore
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.hooks import HookEngine
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.todos import TodoStore
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import LLMResponse, MultiAgentTaskRun


class _FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def send(self, spec, messages, model=None, **kwargs):
        return LLMResponse(
            provider=spec.name,
            model=model or spec.default_model,
            content=self.responses.pop(0),
        )


class ClaudeParityRuntimeTest(unittest.TestCase):
    def test_slash_command_discovery_and_argument_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            command_dir = workspace.workspace_dir / "commands"
            command_dir.mkdir(parents=True)
            (command_dir / "demo.md").write_text(
                "---\n"
                "description: Demo command\n"
                "argument-hint: '<topic>'\n"
                "allowed-tools: [read_file]\n"
                "---\n"
                "Investigate $ARGUMENTS.",
                encoding="utf-8",
            )

            store = SlashCommandStore(root, workspace.workspace_dir)
            names = [command.name for command in store.discover()]
            self.assertIn("demo", names)
            self.assertIn("feature-dev", names)

            rendered = store.render("demo", "spectral bands")
            self.assertEqual(rendered.prompt, "Investigate spectral bands.")
            self.assertEqual(rendered.spec.allowed_tools, ["read_file"])

    def test_todo_store_roundtrip_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            store = TodoStore(workspace.workspace_dir)

            todo_list = store.replace(
                "session-1",
                [
                    {
                        "content": "inspect command loader",
                        "status": "in_progress",
                        "priority": "high",
                    }
                ],
            )

            self.assertEqual(todo_list.items[0].content, "inspect command loader")
            self.assertEqual(store.load("session-1").items[0].status, "in_progress")
            export_path = store.export_markdown("session-1")
            self.assertIn("inspect command loader", export_path.read_text(encoding="utf-8"))

    def test_hook_engine_blocks_pre_tool_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            write_json(
                workspace.workspace_dir / "runtime_extensions" / "hooks.json",
                {
                    "hooks": [
                        {
                            "id": "hook-block-run",
                            "name": "block run",
                            "event": "PreToolUse",
                            "tool_name": "run_command",
                            "action": "block",
                            "message": "shell disabled in this test",
                            "enabled": True,
                        }
                    ]
                },
            )

            result = SafeAgentToolExecutor(
                root,
                workspace.workspace_dir,
                permission_policy="auto",
                hook_engine=HookEngine(workspace.workspace_dir),
            ).run_command(["git", "status"])

            self.assertEqual(result.status, "blocked")
            self.assertIn("shell disabled", result.content)

    def test_action_loop_task_tool_runs_registered_subagent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            RuntimeExtensionStore(workspace.workspace_dir).add_subagent(
                "reviewer",
                "code_review",
                tools=["read_file"],
            )
            conversations = ConversationStore(workspace.workspace_dir)
            session = conversations.new("parent")
            providers = LLMProviderStore(workspace.workspace_dir)
            fake = _FakeLLMClient(
                [
                    (
                        '{"action":"tool","tool_name":"task","args":'
                        '{"agents":["reviewer"],"instruction":"review this","mode":"sequential","max_steps":1}}'
                    ),
                    '{"action":"final","final":"subagent reviewed"}',
                    '{"action":"final","final":"parent done"}',
                ]
            )

            run = AgentActionLoop(
                conversations,
                providers,
                workspace,
                llm_client=fake,
            ).run(
                session.session_id,
                provider="deepseek",
                instruction="delegate review",
                max_steps=2,
            )

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.steps[0].tool_name, "task")
            self.assertIn("subagent reviewed", run.steps[0].tool_result.content)
            task_path = Path(run.steps[0].tool_result.artifact_path)
            self.assertTrue(task_path.exists())
            loaded = MultiAgentTaskRun.from_dict(__import__("json").loads(task_path.read_text(encoding="utf-8")))
            self.assertEqual(loaded.role_runs[0].agent_name, "reviewer")


if __name__ == "__main__":
    unittest.main()
