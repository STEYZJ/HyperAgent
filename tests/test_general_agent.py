import json
import tempfile
import unittest
from pathlib import Path

from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.general_agent import GeneralAgentRunner
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import GeneralAgentRun, LLMResponse


class _FinalLLMClient:
    def send(self, spec, messages, model=None, **kwargs):
        return LLMResponse(
            provider=spec.name,
            model=model or spec.default_model,
            content='{"action":"final","final":"done"}',
        )


class GeneralAgentRunnerTest(unittest.TestCase):
    def test_registered_agent_runs_and_persists_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            RuntimeExtensionStore(workspace.workspace_dir).add_subagent(
                "reviewer",
                "code_quality",
                tools=["read_file"],
                profile="reasonix-balanced",
            )
            conversations = ConversationStore(workspace.workspace_dir)
            providers = LLMProviderStore(workspace.workspace_dir)

            run = GeneralAgentRunner(
                workspace,
                conversations,
                providers,
                permission_policy="deny",
                llm_client=_FinalLLMClient(),
            ).run("reviewer", "summarize status")

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.agent_name, "reviewer")
            self.assertTrue(Path(run.action_run_path).exists())
            saved = GeneralAgentRun.from_dict(
                json.loads((Path(run.run_dir) / "agent_run.json").read_text(encoding="utf-8"))
            )
            self.assertEqual(saved.run_id, run.run_id)


if __name__ == "__main__":
    unittest.main()
