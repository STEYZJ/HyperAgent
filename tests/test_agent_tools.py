import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.agents import CoordinatorAgent
from hyperagent.cli import main
from hyperagent.core.io import read_json
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.mcp import MCPServerStore
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import AgentToolResult, MCPServerSpec


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

    def test_run_command_permission_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            package = root / "hyperagent"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")

            denied = SafeAgentToolExecutor(
                root,
                workspace.workspace_dir,
                permission_policy="ask",
                permission_callback=lambda request: False,
            ).run_command([sys.executable, "-m", "compileall", "-q", "hyperagent"])
            self.assertEqual(denied.status, "blocked")
            self.assertIn("permission denied", denied.warnings[0])

            allowed = SafeAgentToolExecutor(
                root,
                workspace.workspace_dir,
                permission_policy="ask",
                permission_callback=lambda request: True,
            ).run_command([sys.executable, "-m", "compileall", "-q", "hyperagent"])
            self.assertEqual(allowed.status, "ok")

    def test_session_ask_caches_same_risk_and_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            package = root / "hyperagent"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            approvals = []
            cache = {}

            executor = SafeAgentToolExecutor(
                root,
                workspace.workspace_dir,
                permission_policy="session-ask",
                permission_callback=lambda request: approvals.append(request) or True,
                session_permission_cache=cache,
            )
            first = executor.run_command([sys.executable, "-m", "compileall", "-q", "hyperagent"])
            second = executor.run_command([sys.executable, "-m", "compileall", "-q", "hyperagent"])

            self.assertEqual(first.status, "ok")
            self.assertEqual(second.status, "ok")
            self.assertEqual(len(approvals), 1)

    def test_arbitrary_command_requires_explicit_executor_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")

            blocked = SafeAgentToolExecutor(root, workspace.workspace_dir).run_command(
                [sys.executable, "-c", "print('blocked')"]
            )
            self.assertEqual(blocked.status, "blocked")

            allowed = SafeAgentToolExecutor(
                root,
                workspace.workspace_dir,
                permission_policy="ask",
                permission_callback=lambda request: True,
                allow_arbitrary_commands=True,
            ).run_command([sys.executable, "-c", "print('allowed')"])
            self.assertEqual(allowed.status, "ok")
            self.assertIn("allowed", allowed.content)

    def test_run_command_can_use_skill_directory_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            codex_home = Path(tmp) / "codex-home"
            skill_dir = codex_home / "skills" / "demo-skill"
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True)
            (scripts_dir / "show.py").write_text("print('skill cwd ok')\n", encoding="utf-8")
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            old_codex_home = os.environ.get("CODEX_HOME")
            os.environ["CODEX_HOME"] = str(codex_home)
            try:
                executor = SafeAgentToolExecutor(
                    root,
                    workspace.workspace_dir,
                    permission_policy="ask",
                    permission_callback=lambda request: True,
                    allow_arbitrary_commands=True,
                )
                result = executor.run_command(
                    [sys.executable, "scripts/show.py"],
                    cwd=str(skill_dir),
                )
                blocked = executor.run_command(
                    [sys.executable, "-c", "print('bad cwd')"],
                    cwd=str(Path(tmp)),
                )
            finally:
                if old_codex_home is None:
                    os.environ.pop("CODEX_HOME", None)
                else:
                    os.environ["CODEX_HOME"] = old_codex_home

            self.assertEqual(result.status, "ok")
            self.assertIn("skill cwd ok", result.content)
            self.assertEqual(blocked.status, "blocked")
            self.assertIn("cwd", blocked.warnings[0])

    def test_run_experiment_tool_runs_synthetic_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=71)
            agent = CoordinatorAgent()
            audit = agent.audit(data_root, root / "audit.json")
            spectral = agent.analyze(audit, root / "spectral.json")
            recommendation = agent.recommend(audit, spectral, root / "recommendation.json")
            plan = agent.plan(
                audit,
                spectral,
                recommendation,
                root / "plan.yaml",
                root / "run",
                seed=71,
            )

            result = SafeAgentToolExecutor(root, workspace.workspace_dir).run_experiment(
                "plan.yaml"
            )

            self.assertEqual(result.status, "ok")
            self.assertIn("result_path", result.content)
            self.assertTrue((Path(plan.output_dir) / "result.json").exists())

    def test_framework_command_status_and_unsupported_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            executor = SafeAgentToolExecutor(root, workspace.workspace_dir)

            status = executor.framework_command("status")
            self.assertEqual(status.status, "ok")
            payload = read_json(Path(status.artifact_path))["result"]["content"]
            self.assertIn('"initialized": true', payload)
            self.assertIn('"task_count"', payload)

            blocked = executor.framework_command("delete everything")
            self.assertEqual(blocked.status, "blocked")
            self.assertIn("unsupported framework command", blocked.warnings)

    def test_framework_command_does_not_leak_search_provider_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            secret = "SECRET_SHOULD_NOT_LEAK"
            old_value = os.environ.get("BRAVE_SEARCH_API_KEY")
            os.environ["BRAVE_SEARCH_API_KEY"] = secret
            try:
                result = SafeAgentToolExecutor(
                    root,
                    workspace.workspace_dir,
                ).framework_command("web status")
            finally:
                if old_value is None:
                    os.environ.pop("BRAVE_SEARCH_API_KEY", None)
                else:
                    os.environ["BRAVE_SEARCH_API_KEY"] = old_value

            self.assertEqual(result.status, "ok")
            self.assertIn('"brave": true', result.content)
            self.assertNotIn(secret, result.content)

    def test_framework_command_does_not_leak_mcp_env_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            secret = "SECRET_SHOULD_NOT_LEAK"
            MCPServerStore(workspace.workspace_dir).upsert(
                MCPServerSpec(
                    name="demo",
                    command="demo-mcp",
                    args=["--safe"],
                    env={"TOKEN": secret},
                    enabled=True,
                    description="demo server",
                )
            )

            result = SafeAgentToolExecutor(
                root,
                workspace.workspace_dir,
            ).framework_command("mcp status")

            self.assertEqual(result.status, "ok")
            self.assertIn('"env_keys"', result.content)
            self.assertIn('"TOKEN"', result.content)
            self.assertNotIn(secret, result.content)

    def test_deny_write_blocks_apply_patch(self):
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

            executor = SafeAgentToolExecutor(
                root,
                workspace.workspace_dir,
                permission_policy="deny-write",
            )
            checked = executor.check_patch(patch)
            self.assertEqual(checked.status, "ok")

            applied = executor.apply_patch(patch)
            self.assertEqual(applied.status, "blocked")
            self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

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
                self.assertIn("状态: ok", buffer.getvalue())
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
