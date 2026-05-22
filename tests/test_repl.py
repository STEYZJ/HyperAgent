import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.wait_indicator import NullWaitIndicator
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import AgentActionRun, AgentTurnResult, AgentTurnTiming, LLMResponse


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "hyperagent" / "prompts"


class HyperAgentReplTest(unittest.TestCase):
    def test_repl_status_context_tools_and_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            lines = iter(["/status", "/context", "/tools", "/exit"])
            outputs = []

            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=conversations,
                providers=providers,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                input_func=lambda prompt: next(lines),
                output_func=outputs.append,
            )

            self.assertEqual(repl.run(), 0)
            text = "\n".join(outputs)
            self.assertIn("HyperAgent interactive mode", text)
            self.assertIn("initialized: true", text)
            self.assertIn("should_compress:", text)
            self.assertIn("Available local tools", text)

    def test_repl_memory_extensions_clear_and_rewind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            lines = iter(
                [
                    "/init",
                    "/memory add project prefer Chinese responses",
                    "/memory show project",
                    "/agents add reviewer code_quality read_file,search_code",
                    "/agents",
                    "/hooks add precommit before_commit python -m unittest discover -s tests",
                    "/hooks",
                    "/plugin add commit-commands git workflow helper",
                    "/plugin",
                    "/rewind save",
                    "/rewind",
                    "/usage",
                    "/reasonix",
                    "/reasonix reasonix-deep",
                    "/clear",
                    "/simplify",
                    "/exit",
                ]
            )
            outputs = []
            session = conversations.new("stateful")
            conversations.add_message(session.session_id, "user", "old context")

            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=conversations,
                providers=providers,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                session_id=session.session_id,
                input_func=lambda prompt: next(lines),
                output_func=outputs.append,
            )

            self.assertEqual(repl.run(), 0)
            text = "\n".join(outputs)
            self.assertTrue((root / "HyperAgent.md").exists())
            self.assertIn("prefer Chinese responses", text)
            self.assertIn("subagent added:", text)
            self.assertIn("hook added:", text)
            self.assertIn("plugin added:", text)
            self.assertIn("rewind snapshot:", text)
            self.assertIn("llm usage:", text)
            self.assertIn("reasonix profiles:", text)
            self.assertIn("reasonix-deep", text)
            self.assertIn("cleared: messages=0 summaries=0", text)
            self.assertIn("simplify council", text)

    def test_repl_tool_permission_denial_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            package = root / "hyperagent"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")

            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            command = (
                "/tool run "
                + sys.executable
                + " -m compileall -q hyperagent"
            )
            lines = iter([command, "n", "/exit"])
            outputs = []

            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=conversations,
                providers=providers,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                permission_policy="ask",
                input_func=lambda prompt: next(lines),
                output_func=outputs.append,
            )

            self.assertEqual(repl.run(), 0)
            text = "\n".join(outputs)
            self.assertIn("permission requested: run_command", text)
            self.assertIn("status: blocked", text)
            self.assertIn("permission denied by user", text)

    def test_plain_chat_auto_routes_tool_intent_to_action_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            outputs = []
            calls = []

            class FakeActionLoop:
                def __init__(self, *args, **kwargs):
                    calls.append({"init": kwargs})

                def run(self, session_id, provider, instruction, **kwargs):
                    calls.append(
                        {
                            "session_id": session_id,
                            "provider": provider,
                            "instruction": instruction,
                            "kwargs": kwargs,
                        }
                    )
                    return AgentActionRun(
                        run_id="run-1",
                        session_id=session_id,
                        provider=provider,
                        model="deepseek-chat",
                        instruction=instruction,
                        created_at="2026-05-22T00:00:00Z",
                        run_dir=str(workspace.workspace_dir / "agent_action_runs" / "run-1"),
                        status="completed",
                        final_response="已通过受控工具闭环处理。",
                    )

            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=conversations,
                providers=providers,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                permission_policy="session-ask",
                output_func=outputs.append,
                wait_indicator_factory=NullWaitIndicator,
            )

            with patch("hyperagent.runtime.repl.AgentActionLoop", FakeActionLoop):
                repl._chat("请联网搜索最新高光谱分类论文")

            text = "\n".join(outputs)
            self.assertIn("Detected a tool-capable request", text)
            self.assertIn("[action-run] run-1", text)
            self.assertEqual(calls[1]["instruction"], "请联网搜索最新高光谱分类论文")
            self.assertTrue(calls[0]["init"]["tool_executor"].allow_arbitrary_commands)

    def test_plain_chat_without_tool_intent_uses_text_agent_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            outputs = []
            calls = []

            class FakeAgentLoop:
                def __init__(self, *args, **kwargs):
                    pass

                def run(self, **kwargs):
                    calls.append(kwargs)
                    return AgentTurnResult(
                        session_id=kwargs["session_id"],
                        provider="deepseek",
                        model="deepseek-chat",
                        mode="research",
                        task_id=None,
                        response=LLMResponse(
                            provider="deepseek",
                            model="deepseek-chat",
                            content="普通回答",
                            reasoning_content="",
                        ),
                        context_message_count=2,
                        context_chars=12,
                    )

            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=conversations,
                providers=providers,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                output_func=outputs.append,
                wait_indicator_factory=NullWaitIndicator,
            )

            with patch("hyperagent.runtime.repl.AgentLoop", FakeAgentLoop):
                repl._chat("解释一下 OA 和 Kappa 的区别")

            self.assertEqual(len(calls), 1)
            self.assertIn("普通回答", "\n".join(outputs))
            self.assertNotIn("controlled action loop", "\n".join(outputs))

    def test_context_status_triggers_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            session = conversations.new("long")
            for index in range(10):
                conversations.add_message(
                    session.session_id,
                    "user",
                    f"message {index} " + "x" * 100,
                )

            status = conversations.context_status(
                session.session_id,
                max_chars=200,
                keep_last=4,
                trigger_ratio=0.8,
            )
            self.assertTrue(status.should_compress)

            compressed = conversations.auto_compress(
                session.session_id,
                max_chars=200,
                keep_last=4,
                trigger_ratio=0.8,
            )
            self.assertEqual(len(compressed.messages), 4)
            self.assertEqual(len(compressed.summaries), 1)

    def test_thinking_toggle_controls_reasoning_display_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            outputs = []
            calls = []

            class FakeAgentLoop:
                def __init__(self, *args, **kwargs):
                    pass

                def run(self, **kwargs):
                    calls.append(kwargs)
                    return AgentTurnResult(
                        session_id=kwargs["session_id"],
                        provider="deepseek",
                        model="deepseek-chat",
                        mode="research",
                        task_id=None,
                        response=LLMResponse(
                            provider="deepseek",
                            model="deepseek-chat",
                            content="final answer",
                            reasoning_content="reasoning trace",
                        ),
                        context_message_count=2,
                        context_chars=12,
                        timing=AgentTurnTiming(
                            turn_started_at="2026-05-20T00:00:00Z",
                            turn_completed_at="2026-05-20T00:00:01Z",
                            model_wait_elapsed_sec=1.0,
                        ),
                    )

            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=conversations,
                providers=providers,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                llm_kwargs={"thinking": {"type": "disabled"}},
                output_func=outputs.append,
                wait_indicator_factory=NullWaitIndicator,
            )

            with patch("hyperagent.runtime.repl.AgentLoop", FakeAgentLoop):
                repl.handle_line("/thinking status")
                repl._chat("hello")
                repl.handle_line("/thinking on")
                repl._chat("hello again")

            text = "\n".join(outputs)
            self.assertIn("model thinking: disabled", text)
            self.assertIn("reasoning display: collapsed", text)
            self.assertIn("【模型思考内容已折叠，可用 /thinking on 展开】", text)
            self.assertIn("reasoning display: expanded", text)
            self.assertIn("【模型思考内容】", text)
            self.assertIn("reasoning trace", text)
            self.assertFalse(calls[0]["thinking_displayed"])
            self.assertFalse(calls[0]["reasoning_content_expanded"])
            self.assertTrue(calls[1]["thinking_displayed"])
            self.assertTrue(calls[1]["reasoning_content_expanded"])
            self.assertEqual(repl.llm_kwargs["thinking"], {"type": "disabled"})

    def test_chat_without_reasoning_content_does_not_show_folded_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            outputs = []

            class FakeAgentLoop:
                def __init__(self, *args, **kwargs):
                    pass

                def run(self, **kwargs):
                    return AgentTurnResult(
                        session_id=kwargs["session_id"],
                        provider="deepseek",
                        model="deepseek-chat",
                        mode="research",
                        task_id=None,
                        response=LLMResponse(
                            provider="deepseek",
                            model="deepseek-chat",
                            content="plain answer",
                            reasoning_content="",
                        ),
                        context_message_count=2,
                        context_chars=12,
                    )

            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=conversations,
                providers=providers,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                output_func=outputs.append,
                wait_indicator_factory=NullWaitIndicator,
            )

            with patch("hyperagent.runtime.repl.AgentLoop", FakeAgentLoop):
                repl._chat("hello")

            text = "\n".join(outputs)
            self.assertIn("plain answer", text)
            self.assertNotIn("模型思考内容已折叠", text)


if __name__ == "__main__":
    unittest.main()
