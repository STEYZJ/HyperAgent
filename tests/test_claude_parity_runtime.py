import tempfile
import unittest
from pathlib import Path

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.background_jobs import BackgroundJobStore
from hyperagent.runtime.commands import SlashCommandStore
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.hooks import HookEngine
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.multi_agent import MultiAgentTaskRunner
from hyperagent.runtime.slash_registry import (
    command_names,
    gateway_command_names,
    grouped_help,
    resolve_command,
)
from hyperagent.runtime.command_aliases import normalize_hyperagent_args
from hyperagent.runtime.skills import SkillStore
from hyperagent.runtime.subagents import SubagentRuntimeRegistry
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
    def test_central_slash_registry_drives_help_and_gateway_allowlist(self):
        self.assertIn("agents", command_names())
        self.assertIn("background", command_names())
        self.assertEqual(resolve_command("channels").cli_command, "channel-list")
        self.assertIn("forget", resolve_command("permissions").args_hint)
        self.assertIn("status", gateway_command_names())
        help_text = grouped_help()
        self.assertIn("/hsi", help_text)
        self.assertIn("/permissions [list|forget", help_text)
        self.assertIn("/snapshot", help_text)
        self.assertEqual(
            normalize_hyperagent_args(["/agents", "pause", "maintenance"]),
            ["agent-pause", "maintenance"],
        )
        self.assertEqual(
            normalize_hyperagent_args(["/skills", "bundles"]),
            ["skill-bundles"],
        )

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

    def test_skill_store_search_bundles_and_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "spectral-reviewer"
            source.mkdir(parents=True)
            (source / "SKILL.md").write_text(
                "---\n"
                "name: spectral-reviewer\n"
                "description: Review HSI spectral bands\n"
                "bundle: hsi\n"
                "runAs: subagent\n"
                "allowed-tools: [read_file]\n"
                "---\n"
                "Inspect $ARGUMENTS for spectral redundancy.",
                encoding="utf-8",
            )
            store = SkillStore([root / "source"])

            self.assertEqual(store.search("redundancy")[0].name, "spectral-reviewer")
            self.assertIn("hsi", store.bundles())

            installed = store.install(source, root / "installed")
            self.assertEqual(installed.run_as, "subagent")
            self.assertTrue(
                (root / "installed" / "spectral-reviewer" / "SKILL.md").exists()
            )

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

    def test_action_loop_emits_task_complete_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            write_json(
                workspace.workspace_dir / "runtime_extensions" / "hooks.json",
                {
                    "hooks": [
                        {
                            "id": "hook-task-complete",
                            "name": "task complete",
                            "event": "TaskComplete",
                            "action": "warn",
                            "message": "task finished hook",
                            "enabled": True,
                        }
                    ]
                },
            )
            conversations = ConversationStore(workspace.workspace_dir)
            session = conversations.new("task complete")
            providers = LLMProviderStore(workspace.workspace_dir)
            fake = _FakeLLMClient(['{"action":"final","final":"done"}'])

            run = AgentActionLoop(
                conversations,
                providers,
                workspace,
                llm_client=fake,
                permission_policy="auto",
            ).run(session.session_id, "deepseek", "finish")

            self.assertEqual(run.status, "completed")
            self.assertIn("TaskComplete hook: task finished hook", run.warnings)
            hook_runs = list((workspace.workspace_dir / "hook_runs").glob("*TaskComplete.json"))
            self.assertEqual(len(hook_runs), 1)
            payload = read_json(hook_runs[0])
            self.assertEqual(payload["payload"]["status"], "completed")
            self.assertEqual(payload["payload"]["final_response"], "done")

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
            loaded = MultiAgentTaskRun.from_dict(
                __import__("json").loads(task_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(loaded.role_runs[0].agent_name, "reviewer")
            self.assertEqual(loaded.max_depth, 1)
            self.assertTrue(loaded.active_registry_path.endswith("active_subagents.json"))

    def test_subagent_registry_pause_stop_and_depth_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            session = conversations.new("parent")
            providers = LLMProviderStore(workspace.workspace_dir)
            RuntimeExtensionStore(workspace.workspace_dir).add_subagent(
                "reviewer",
                "code_review",
                tools=["read_file"],
            )
            registry = SubagentRuntimeRegistry(workspace.workspace_dir)
            registry.pause("maintenance")

            paused = MultiAgentTaskRunner(workspace, conversations, providers).run(
                session_id=session.session_id,
                provider="deepseek",
                instruction="review",
                agents=["reviewer"],
            )

            self.assertEqual(paused.status, "blocked")
            self.assertTrue(paused.paused)
            self.assertIn("paused", paused.warnings[0].lower())

            registry.resume()
            depth_blocked = MultiAgentTaskRunner(
                workspace,
                conversations,
                providers,
            ).run(
                session_id=session.session_id,
                provider="deepseek",
                instruction="review",
                agents=["reviewer"],
                depth=1,
                max_depth=0,
            )
            self.assertEqual(depth_blocked.status, "blocked")
            self.assertIn("max_depth", depth_blocked.warnings[0])

            state = registry.register(
                subagent_id="sa-test",
                agent_name="reviewer",
                role="code_review",
                instruction="review",
                run_id="run-1",
            )
            self.assertEqual(state.status, "running")
            registry.stop("sa-test")
            stopped = registry.list(include_completed=True)[-1]
            self.assertEqual(stopped.status, "stopped")
            self.assertTrue(registry.should_stop("sa-test"))

    def test_background_job_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            store = BackgroundJobStore(workspace.workspace_dir)

            job = store.create(
                kind="prompt",
                instruction="review experiments",
                session_id="session-1",
            )
            store.update(job.job_id, status="running", run_path="runs/1.json")
            store.update(job.job_id, warning="still in progress")

            loaded = store.list()[0]
            self.assertEqual(loaded.status, "running")
            self.assertEqual(loaded.run_path, "runs/1.json")
            self.assertEqual(loaded.warnings, ["still in progress"])


if __name__ == "__main__":
    unittest.main()
