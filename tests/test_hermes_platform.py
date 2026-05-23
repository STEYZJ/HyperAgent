import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperagent.runtime.channels import ChannelConfigStore
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.platform_runtime import (
    PlatformStatusReporter,
    SessionSearchIndex,
    SkillTelemetryStore,
    summarize_skill_bundles,
)
from hyperagent.runtime.slash_registry import resolve_command
from hyperagent.runtime.workspace import HyperAgentWorkspace, utc_now
from hyperagent.schemas import ConversationSummary


class HermesPlatformRuntimeTest(unittest.TestCase):
    def _workspace(self, root: Path):
        workspace = HyperAgentWorkspace(root)
        workspace.init(root / "datasets")
        return workspace, ConversationStore(workspace.workspace_dir), LLMProviderStore(workspace.workspace_dir)

    def test_platform_status_aggregates_local_health_without_secret_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, conversations, providers = self._workspace(root)
            channel_store = ChannelConfigStore(workspace.workspace_dir)
            old_deepseek = os.environ.get("DEEPSEEK_API_KEY")
            old_feishu = os.environ.get("FEISHU_VERIFICATION_TOKEN")
            os.environ["DEEPSEEK_API_KEY"] = "fake-deepseek-value"
            os.environ["FEISHU_VERIFICATION_TOKEN"] = "fake-feishu-token"
            try:
                report = PlatformStatusReporter(
                    workspace,
                    conversations,
                    providers,
                    channel_store=channel_store,
                    skill_roots=[root / "skills"],
                ).report()
            finally:
                if old_deepseek is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek
                if old_feishu is None:
                    os.environ.pop("FEISHU_VERIFICATION_TOKEN", None)
                else:
                    os.environ["FEISHU_VERIFICATION_TOKEN"] = old_feishu

            text = json.dumps(report, ensure_ascii=False)
            self.assertIn("providers", report)
            self.assertIn("channels", report)
            self.assertIn("sessions", report)
            self.assertIn("skills", report)
            self.assertIn("channel_delivery", report)
            self.assertIn("DEEPSEEK_API_KEY", text)
            self.assertIn("FEISHU_VERIFICATION_TOKEN", text)
            self.assertNotIn("fake-deepseek-value", text)
            self.assertNotIn("fake-feishu-token", text)
            self.assertTrue(all(channel["chat_query_only"] for channel in report["channels"]))
            self.assertTrue(all(channel["live"]["checked"] is False for channel in report["channels"]))

    def test_platform_status_live_probe_is_opt_in_and_secret_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, conversations, providers = self._workspace(root)
            old_deepseek = os.environ.get("DEEPSEEK_API_KEY")
            os.environ["DEEPSEEK_API_KEY"] = "fake-deepseek-value"
            try:
                with patch.object(
                    PlatformStatusReporter,
                    "_probe_url",
                    return_value={"checked": True, "reachable": True, "host": "example.test"},
                ):
                    report = PlatformStatusReporter(
                        workspace,
                        conversations,
                        providers,
                        skill_roots=[root / "skills"],
                    ).report(live=True, timeout_sec=0.001)
            finally:
                if old_deepseek is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek

            text = json.dumps(report, ensure_ascii=False)
            self.assertTrue(report["live"])
            self.assertTrue(all(provider["live"]["checked"] for provider in report["providers"]))
            self.assertIn("DEEPSEEK_API_KEY", text)
            self.assertNotIn("fake-deepseek-value", text)

    def test_session_search_hits_title_message_summary_and_writes_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, conversations, _providers = self._workspace(root)
            session = conversations.new("Spectral platform diagnosis")
            conversations.add_message(
                session.session_id,
                "user",
                "Hermes status should locate this calibration message.",
            )
            saved = conversations.load(session.session_id)
            saved.summaries.append(
                ConversationSummary(
                    summary_id="summary-hermes",
                    created_at=utc_now(),
                    message_count=1,
                    content="Summary mentions session search telemetry and platform health.",
                    method="manual",
                )
            )
            conversations.save(saved)

            index = SessionSearchIndex(workspace.workspace_dir)
            title_results = index.search(conversations, "Spectral", limit=5)
            message_results = index.search(conversations, "calibration", limit=5)
            summary_results = index.search(conversations, "telemetry", limit=5)

            self.assertEqual(title_results[0].session_id, session.session_id)
            self.assertEqual(message_results[0].session_id, session.session_id)
            self.assertEqual(summary_results[0].session_id, session.session_id)
            self.assertLessEqual(len(summary_results[0].snippet), 240)
            self.assertTrue(index.path.exists())

    def test_skill_usage_summarize_and_curate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, _conversations, _providers = self._workspace(root)
            skills_root = root / "skills"
            demo = skills_root / "demo-skill"
            quiet = skills_root / "quiet-skill"
            demo.mkdir(parents=True)
            quiet.mkdir(parents=True)
            (demo / "SKILL.md").write_text(
                "---\n"
                "name: demo-skill\n"
                "description: Demo skill\n"
                "bundle: hsi\n"
                "---\n"
                "Run demo.",
                encoding="utf-8",
            )
            (quiet / "SKILL.md").write_text(
                "---\n"
                "name: quiet-skill\n"
                "description: No bundle metadata\n"
                "---\n"
                "Stay quiet.",
                encoding="utf-8",
            )

            from hyperagent.runtime.skills import SkillStore

            skill_store = SkillStore([skills_root])
            telemetry = SkillTelemetryStore(workspace.workspace_dir)
            telemetry.record("list", source="test", metadata={"token": "fake-token-value"})
            telemetry.record("run", skill="demo-skill", bundle="hsi", source="test")
            telemetry.record("install", skill="demo-skill", bundle="hsi", source="test")
            summary = telemetry.summarize()
            curator = telemetry.curate(skill_store.list())
            bundle_summary = summarize_skill_bundles(skill_store.list())

            self.assertEqual(summary["total_events"], 3)
            self.assertEqual(summary["by_action"]["run"], 1)
            self.assertEqual(summary["by_skill"]["demo-skill"], 2)
            self.assertEqual(summary["by_bundle"]["hsi"], 2)
            self.assertIn("quiet-skill", curator["unused_skills"])
            self.assertIn("quiet-skill", curator["missing_bundle_metadata"])
            self.assertEqual(bundle_summary["bundles"]["hsi"]["owners"], [])
            self.assertIn("quiet-skill", bundle_summary["bundles"]["skills"]["skills"])
            self.assertNotIn("fake-token-value", json.dumps(summary, ensure_ascii=False))

    def test_slash_registry_exposes_hermes_commands(self):
        self.assertEqual(resolve_command("platforms").cli_command, "platform-status")
        self.assertEqual(resolve_command("session-search").cli_command, "session-search")
        self.assertEqual(resolve_command("skill-usage").cli_command, "skill-usage")


if __name__ == "__main__":
    unittest.main()
