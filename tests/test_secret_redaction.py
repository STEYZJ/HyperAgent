import subprocess
import tempfile
import unittest
from pathlib import Path

from hyperagent.core.worklog import append_worklog, redact_secrets


SECRET_PREFIX = "sk-"


class SecretRedactionTest(unittest.TestCase):
    def test_redacts_provider_tokens_and_env_assignments(self):
        token = SECRET_PREFIX + ("a" * 32)

        redacted = redact_secrets(f"DEEPSEEK_API_KEY={token} plain={token}")

        self.assertNotIn(token, redacted)
        self.assertIn("[REDACTED_SECRET]", redacted)

    def test_does_not_redact_task_branch_names(self):
        branch = "task-tui-shell-prompt-native-selection"

        self.assertEqual(redact_secrets(branch), branch)

    def test_append_worklog_never_persists_obvious_secret(self):
        token = SECRET_PREFIX + ("b" * 32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worklog.md"

            append_worklog(
                "secret check",
                "none",
                f"configured token {token}",
                "avoid leaks",
                f"DEEPSEEK_API_KEY={token}",
                "redacted",
                "continue",
                path=path,
            )

            content = path.read_text(encoding="utf-8")
            self.assertNotIn(token, content)
            self.assertIn("[REDACTED_SECRET]", content)

    def test_tracked_files_do_not_contain_provider_secret_shape(self):
        pattern = "(^|[^A-Za-z0-9_])" + SECRET_PREFIX + "[A-Za-z0-9_-]{20,}"
        result = subprocess.run(
            ["git", "grep", "-n", "-E", pattern, "--", "."],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
