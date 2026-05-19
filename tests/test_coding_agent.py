import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hyperagent.cli import main
from hyperagent.runtime.coding_agent import CodingAgent
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repo_context import RepoContextBuilder
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import LLMResponse


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "hyperagent" / "prompts"


class _FakeLLMClient:
    def send(
        self,
        spec,
        messages,
        model=None,
        temperature=0.2,
        max_tokens=None,
    ):
        return LLMResponse(
            provider=spec.name,
            model=model or spec.default_model,
            content="# Plan\n\n- Edit runtime files.\n- Run tests.",
        )


class CodingAgentTest(unittest.TestCase):
    def test_repo_context_builder_selects_relevant_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hyperagent" / "runtime").mkdir(parents=True)
            (root / "hyperagent" / "runtime" / "agent_loop.py").write_text(
                "class AgentLoop:\n    pass\n",
                encoding="utf-8",
            )
            (root / "experiments").mkdir()
            (root / "experiments" / "large.txt").write_text("ignore", encoding="utf-8")

            snapshot = RepoContextBuilder(root).build(
                query="agent loop",
                max_files=5,
                max_preview_chars=80,
            )

            paths = [item.path for item in snapshot.selected_files]
            self.assertIn("hyperagent/runtime/agent_loop.py", paths)
            self.assertNotIn("experiments/large.txt", paths)
            self.assertFalse(snapshot.is_git_repo)

    def test_coding_agent_plan_saves_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hyperagent").mkdir()
            (root / "hyperagent" / "cli.py").write_text("def main(): pass\n", encoding="utf-8")
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            session = conversations.new("code run")

            run = CodingAgent(
                workspace,
                conversations,
                providers,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                llm_client=_FakeLLMClient(),
            ).plan(
                session_id=session.session_id,
                provider="deepseek",
                instruction="add a command",
                max_files=5,
            )

            self.assertEqual(run.status, "planned")
            self.assertTrue(Path(run.repo_context_path).exists())
            self.assertTrue(Path(run.repo_context_markdown_path).exists())
            self.assertTrue(Path(run.response_path).exists())
            self.assertTrue(Path(run.plan_path).exists())
            self.assertIn("Edit runtime files", Path(run.plan_path).read_text(encoding="utf-8"))

    def test_agent_context_and_plan_cli(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hyperagent").mkdir()
            (root / "hyperagent" / "cli.py").write_text("def main(): pass\n", encoding="utf-8")
            os.chdir(root)
            try:
                self.assertEqual(main(["init", "--dataset-root", str(root / "datasets")]), 0)
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "agent-context",
                                "--query",
                                "cli",
                                "--max-files",
                                "3",
                            ]
                        ),
                        0,
                    )
                self.assertIn("Repository Context", buffer.getvalue())
                self.assertIn("hyperagent/cli.py", buffer.getvalue())

                with patch(
                    "hyperagent.runtime.agent_loop.LLMClient.send",
                    return_value=LLMResponse(
                        provider="deepseek",
                        model="deepseek-chat",
                        content="planned from cli",
                    ),
                ):
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        self.assertEqual(
                            main(
                                [
                                    "agent-plan",
                                    "--provider",
                                    "deepseek",
                                    "--instruction",
                                    "add a repo-aware command",
                                    "--max-files",
                                    "3",
                                    "--max-tokens",
                                    "64",
                                ]
                            ),
                            0,
                        )
                output = buffer.getvalue()
                self.assertIn("run_id:", output)
                self.assertIn("plan:", output)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
