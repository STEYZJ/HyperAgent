import sys
import tempfile
import unittest
from pathlib import Path

from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.workspace import HyperAgentWorkspace


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
            self.assertIn("initialized: True", text)
            self.assertIn("should_compress:", text)
            self.assertIn("Available local tools", text)

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


if __name__ == "__main__":
    unittest.main()
