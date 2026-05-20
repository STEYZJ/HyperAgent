import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.launcher import main as launcher_main
from hyperagent.runtime.command_aliases import normalize_hyperagent_args


class HyperAgentLauncherTest(unittest.TestCase):
    def test_empty_args_start_repl(self):
        self.assertEqual(normalize_hyperagent_args([]), ["repl"])

    def test_plain_prompt_maps_to_agent_chat(self):
        self.assertEqual(
            normalize_hyperagent_args(["analyze", "the", "latest", "result"]),
            ["agent-chat", "--message", "analyze the latest result"],
        )

    def test_prompt_keeps_provider_options(self):
        self.assertEqual(
            normalize_hyperagent_args(
                [
                    "--model",
                    "deepseek-v4-pro",
                    "--thinking",
                    "enabled",
                    "design",
                    "next",
                    "experiment",
                ]
            ),
            [
                "agent-chat",
                "--model",
                "deepseek-v4-pro",
                "--thinking",
                "enabled",
                "--message",
                "design next experiment",
            ],
        )

    def test_plan_alias_does_not_break_canonical_experiment_plan(self):
        self.assertEqual(
            normalize_hyperagent_args(["plan", "implement", "module"]),
            ["agent-plan", "--instruction", "implement module"],
        )
        self.assertEqual(
            normalize_hyperagent_args(["plan", "--audit", "reports/audit.json"]),
            ["plan", "--audit", "reports/audit.json"],
        )

    def test_slash_aliases(self):
        self.assertEqual(normalize_hyperagent_args(["/status"]), ["status"])
        self.assertEqual(normalize_hyperagent_args(["/model"]), ["llm-providers"])
        self.assertEqual(normalize_hyperagent_args(["/reasonix"]), ["llm-profile"])
        self.assertEqual(normalize_hyperagent_args(["/usage"]), ["llm-usage"])
        self.assertEqual(normalize_hyperagent_args(["/repl"]), ["repl"])
        self.assertEqual(
            normalize_hyperagent_args(["/resume", "sid-1", "continue", "work"]),
            ["agent-chat", "--session-id", "sid-1", "--message", "continue work"],
        )
        self.assertEqual(
            normalize_hyperagent_args(["/compact", "sid-1", "--keep-last", "3"]),
            ["session-compress", "--session-id", "sid-1", "--keep-last", "3"],
        )

    def test_launcher_help_prints_hyperagent_commands(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(Path(tmp))
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(launcher_main(["/help"]), 0)
            finally:
                os.chdir(old_cwd)
        output = buffer.getvalue()
        self.assertIn("HyperAgent command format", output)
        self.assertIn('HyperAgent "analyze the latest HSI result', output)


if __name__ == "__main__":
    unittest.main()
