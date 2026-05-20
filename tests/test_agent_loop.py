import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hyperagent.cli import main
from hyperagent.core.io import write_json
from hyperagent.runtime.agent_loop import AgentLoop
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import LLMResponse


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "hyperagent" / "prompts"


class _FakeLLMClient:
    def __init__(self):
        self.calls = []

    def send(
        self,
        spec,
        messages,
        model=None,
        temperature=0.2,
        max_tokens=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "spec": spec,
                "messages": list(messages),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        return LLMResponse(
            provider=spec.name,
            model=model or spec.default_model,
            content="assistant answer",
        )


class AgentLoopTest(unittest.TestCase):
    def test_agent_loop_persists_user_and_assistant_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            session = conversations.new("demo")
            fake = _FakeLLMClient()

            result = AgentLoop(
                conversations,
                providers,
                workspace,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                llm_client=fake,
            ).run(
                session.session_id,
                provider="deepseek",
                user_message="帮我设计下一步实验",
                max_tokens=32,
            )

            saved = conversations.load(session.session_id)
            self.assertEqual(result.response.content, "assistant answer")
            self.assertEqual([message.role for message in saved.messages], ["user", "assistant"])
            self.assertEqual(saved.messages[-1].content, "assistant answer")
            self.assertIsNotNone(result.timing)
            self.assertGreaterEqual(result.timing.model_wait_elapsed_sec, 0.0)
            self.assertIn("turn_started_at", saved.messages[-1].metadata)
            self.assertIn("turn_completed_at", saved.messages[-1].metadata)
            self.assertIn("model_wait_elapsed_sec", saved.messages[-1].metadata)
            self.assertEqual(saved.messages[-1].metadata["provider"], "deepseek")
            self.assertEqual(saved.messages[-1].metadata["mode"], "research")
            self.assertIn("帮我设计下一步实验", fake.calls[0]["messages"][-1].content)
            self.assertGreater(result.context_message_count, 1)

    def test_conversation_message_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            session = conversations.new("metadata")

            conversations.add_message(
                session.session_id,
                "assistant",
                "answer",
                created_at="2026-05-20T00:00:00Z",
                metadata={"model_wait_elapsed_sec": 1.25, "provider": "deepseek"},
            )

            saved = conversations.load(session.session_id)
            self.assertEqual(saved.messages[-1].created_at, "2026-05-20T00:00:00Z")
            self.assertEqual(saved.messages[-1].metadata["model_wait_elapsed_sec"], 1.25)
            self.assertEqual(saved.messages[-1].metadata["provider"], "deepseek")

    def test_agent_loop_injects_task_and_artifact_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            task = workspace.create_task(
                goal="design a spectral-spatial module",
                dataset="Synthetic",
                objective="maximize_oa_with_ablation",
                keywords=["hsi", "adapter"],
            )
            artifact = workspace.task_artifact_dir(task.task_id) / "audit.json"
            write_json(artifact, {"dataset_name": "Synthetic", "band_count": 16})
            task.artifacts["audit"] = str(artifact)
            workspace.save_task(task)

            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            session = conversations.new("algorithm")
            messages = AgentLoop(
                conversations,
                providers,
                workspace,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                llm_client=_FakeLLMClient(),
            ).build_messages(session, mode="algorithm", task_id=task.task_id)

            text = "\n".join(message.content for message in messages)
            self.assertIn("Algorithm Designer", text)
            self.assertIn("design a spectral-spatial module", text)
            self.assertIn("audit.json", text)
            self.assertIn("band_count", text)

    def test_agent_loop_auto_compresses_long_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)
            session = conversations.new("long")
            for index in range(8):
                role = "user" if index % 2 == 0 else "assistant"
                conversations.add_message(session.session_id, role, f"message {index} " + "x" * 300)

            AgentLoop(
                conversations,
                providers,
                workspace,
                prompt_library=PromptLibrary([PROMPT_ROOT]),
                llm_client=_FakeLLMClient(),
            ).run(
                session.session_id,
                provider="deepseek",
                user_message="继续",
                max_context_chars=500,
            )

            saved = conversations.load(session.session_id)
            self.assertGreaterEqual(len(saved.summaries), 1)
            self.assertEqual(saved.messages[-2].content, "继续")
            self.assertEqual(saved.messages[-1].content, "assistant answer")

    def test_agent_chat_cli_uses_saved_session(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chdir(root)
            try:
                self.assertEqual(main(["init", "--dataset-root", str(root / "datasets")]), 0)
                with patch(
                    "hyperagent.runtime.agent_loop.LLMClient.send",
                    return_value=LLMResponse(
                        provider="deepseek",
                        model="deepseek-chat",
                        content="cli answer",
                    ),
                ):
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        self.assertEqual(
                            main(
                                [
                                    "agent-chat",
                                    "--provider",
                                    "deepseek",
                                    "--new-title",
                                    "cli demo",
                                    "--message",
                                    "hello",
                                    "--max-tokens",
                                    "16",
                                ]
                            ),
                            0,
                        )
                output = buffer.getvalue()
                self.assertIn("session_id:", output)
                self.assertIn("cli answer", output)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
