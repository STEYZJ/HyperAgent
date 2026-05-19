import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.cli import main
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import AgentToolResult


class AgentToolsTest(unittest.TestCase):
    def test_read_search_and_blocked_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            source = root / "hyperagent" / "demo.py"
            source.parent.mkdir()
            source.write_text("alpha = 1\nbeta = alpha + 1\n", encoding="utf-8")

            executor = SafeAgentToolExecutor(root, workspace.workspace_dir)
            read_result = executor.read_file("hyperagent/demo.py", max_lines=1)
            self.assertEqual(read_result.status, "ok")
            self.assertIn("1: alpha = 1", read_result.content)
            self.assertTrue(Path(read_result.artifact_path).exists())

            search_result = executor.search_code("beta")
            self.assertEqual(search_result.status, "ok")
            self.assertIn("hyperagent/demo.py:2", search_result.content)

            blocked = executor.read_file("../outside.txt")
            self.assertEqual(blocked.status, "blocked")
            self.assertIn("project root", blocked.content)

    def test_run_command_allowlist_and_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            package = root / "hyperagent"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")

            executor = SafeAgentToolExecutor(root, workspace.workspace_dir)
            allowed = executor.run_command(
                [sys.executable, "-m", "compileall", "-q", "hyperagent"]
            )
            self.assertEqual(allowed.status, "ok")
            self.assertEqual(allowed.exit_code, 0)

            blocked = executor.run_command(["bash", "-lc", "echo no"])
            self.assertEqual(blocked.status, "blocked")
            self.assertIn("allowlist", blocked.warnings[0])

    def test_check_and_apply_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            target = root / "demo.txt"
            target.write_text("old\n", encoding="utf-8")
            patch = (
                "diff --git a/demo.txt b/demo.txt\n"
                "--- a/demo.txt\n"
                "+++ b/demo.txt\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            )

            executor = SafeAgentToolExecutor(root, workspace.workspace_dir)
            checked = executor.check_patch(patch)
            self.assertEqual(checked.status, "ok")
            self.assertIn("patch check passed", checked.content)

            applied = executor.apply_patch(patch)
            self.assertEqual(applied.status, "ok")
            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")

    def test_agent_tool_cli(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hyperagent").mkdir()
            (root / "hyperagent" / "__init__.py").write_text("", encoding="utf-8")
            (root / "hyperagent" / "demo.py").write_text(
                "needle = 'value'\n",
                encoding="utf-8",
            )
            os.chdir(root)
            try:
                self.assertEqual(main(["init", "--dataset-root", str(root / "datasets")]), 0)
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "agent-tool",
                                "read-file",
                                "--path",
                                "hyperagent/demo.py",
                                "--max-lines",
                                "1",
                            ]
                        ),
                        0,
                    )
                self.assertIn("needle", buffer.getvalue())

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "agent-tool",
                                "run-command",
                                "--",
                                sys.executable,
                                "-m",
                                "compileall",
                                "-q",
                                "hyperagent",
                            ]
                        ),
                        0,
                    )
                self.assertIn("status: ok", buffer.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_schema_roundtrip(self):
        result = AgentToolResult(
            call_id="call",
            tool_name="read_file",
            status="ok",
            created_at="now",
            content="content",
        )
        loaded = AgentToolResult.from_dict(result.to_dict())
        self.assertEqual(loaded.tool_name, "read_file")
        self.assertEqual(loaded.content, "content")


if __name__ == "__main__":
    unittest.main()
