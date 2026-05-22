import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.cli import main
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.skill_installer import (
    SkillInstaller,
    default_user_skill_root,
    parse_github_skill_url,
)
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.skills import SkillStore
from hyperagent.runtime.workspace import HyperAgentWorkspace


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "hyperagent" / "prompts"


def _write_skill(root: Path, name: str = "academic-research-skill", body: str = "Use $ARGUMENTS.") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Academic research helper\n"
        "runAs: inline\n"
        "allowed-tools: [read_file]\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return skill_dir


class SkillInstallerTest(unittest.TestCase):
    def _with_codex_home(self, root: Path):
        class _Env:
            def __enter__(inner_self):
                inner_self.old = os.environ.get("CODEX_HOME")
                os.environ["CODEX_HOME"] = str(root)
                return inner_self

            def __exit__(inner_self, exc_type, exc, tb):
                if inner_self.old is None:
                    os.environ.pop("CODEX_HOME", None)
                else:
                    os.environ["CODEX_HOME"] = inner_self.old

        return _Env()

    def test_default_root_uses_codex_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_home = Path(tmp) / "codex"
            with self._with_codex_home(codex_home):
                self.assertEqual(default_user_skill_root(), codex_home / "skills")

    def test_local_directory_install_and_store_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_skill(root / "source")
            codex_home = root / "codex"
            with self._with_codex_home(codex_home):
                result = SkillInstaller().install_from_path(source)
                self.assertEqual(result.status, "installed")
                self.assertTrue((codex_home / "skills" / "academic-research-skill" / "SKILL.md").exists())
                self.assertIsNotNone(SkillStore([codex_home / "skills"]).get("academic-research-skill"))

    def test_local_skill_file_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_skill(root / "source", name="single-file-skill")
            codex_home = root / "codex"
            with self._with_codex_home(codex_home):
                result = SkillInstaller().install_from_path(source / "SKILL.md")
                self.assertEqual(result.status, "installed")
                self.assertTrue((codex_home / "skills" / "single-file-skill" / "SKILL.md").exists())

    def test_existing_skill_refuses_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_skill(root / "source", body="first")
            source_2 = _write_skill(root / "source2", body="second")
            codex_home = root / "codex"
            with self._with_codex_home(codex_home):
                first = SkillInstaller().install_from_path(source)
                second = SkillInstaller().install_from_path(source_2)
                forced = SkillInstaller().install_from_path(source_2, force=True)
                self.assertEqual(first.status, "installed")
                self.assertEqual(second.status, "blocked")
                self.assertIn("already exists", second.plan.blocked_reasons[0])
                self.assertEqual(forced.status, "installed")

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_skill(root / "source", name="dry-run-skill")
            codex_home = root / "codex"
            with self._with_codex_home(codex_home):
                result = SkillInstaller().install_from_path(source, dry_run=True)
                self.assertEqual(result.status, "planned")
                self.assertFalse((codex_home / "skills" / "dry-run-skill").exists())

    def test_github_url_parser(self):
        parsed = parse_github_skill_url(
            "https://github.com/owner/repo/tree/main/skills/academic-research"
        )
        self.assertEqual(parsed.owner, "owner")
        self.assertEqual(parsed.repo, "repo")
        self.assertEqual(parsed.ref, "main")
        self.assertEqual(parsed.skill_path, "skills/academic-research")

    def test_github_repo_root_url_parser(self):
        parsed = parse_github_skill_url("https://github.com/owner/repo")

        self.assertEqual(parsed.owner, "owner")
        self.assertEqual(parsed.repo, "repo")
        self.assertEqual(parsed.ref, "main")
        self.assertEqual(parsed.skill_path, "")

    def test_github_repo_root_autodetects_single_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            skill = _write_skill(repo / "skills", name="nested-skill")

            selected = SkillInstaller(install_root=root / "installed")._select_github_skill_source(
                repo,
                parse_github_skill_url("https://github.com/owner/repo"),
            )

            self.assertEqual(selected, skill)

    def test_github_repo_root_reports_claude_plugin_without_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            (repo / ".claude-plugin").mkdir(parents=True)
            (repo / ".claude-plugin" / "plugin.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as ctx:
                SkillInstaller(install_root=root / "installed")._select_github_skill_source(
                    repo,
                    parse_github_skill_url("https://github.com/owner/repo"),
                )

            self.assertIn("Claude plugin", str(ctx.exception))

    def test_github_repo_root_requires_path_when_multiple_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _write_skill(repo / "skills", name="one")
            _write_skill(repo / "more", name="two")

            with self.assertRaises(ValueError) as ctx:
                SkillInstaller(install_root=root / "installed")._select_github_skill_source(
                    repo,
                    parse_github_skill_url("https://github.com/owner/repo"),
                )

            self.assertIn("multiple SKILL.md", str(ctx.exception))

    def test_risk_warnings_and_secret_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = _write_skill(root / "source", name="risky-skill")
            (skill / "scripts").mkdir()
            (skill / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")
            (skill / "SKILL.md").write_text(
                "---\n"
                "name: risky-skill\n"
                "description: risky\n"
                "allowed-tools: [run_command, web_fetch]\n"
                "---\n"
                "token = " + "sk-" + ("A" * 24) + "\n",
                encoding="utf-8",
            )
            result = SkillInstaller(install_root=root / "installed").install_from_path(skill, dry_run=True)
            self.assertEqual(result.status, "blocked")
            self.assertTrue(any("high-risk" in item for item in result.plan.warnings))
            self.assertTrue(any("possible API key" in item for item in result.plan.blocked_reasons))

    def test_cli_skill_install_path_yes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            source = _write_skill(root / "source", name="cli-skill")
            codex_home = root / "codex"
            old_cwd = Path.cwd()
            with self._with_codex_home(codex_home):
                try:
                    os.chdir(project)
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        code = main(["skill-install", "--path", str(source), "--yes"])
                finally:
                    os.chdir(old_cwd)
            self.assertEqual(code, 0)
            self.assertTrue((codex_home / "skills" / "cli-skill" / "SKILL.md").exists())
            self.assertIn("installed", stdout.getvalue())

    def test_agent_tool_install_skill_permission_and_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            source = _write_skill(root / "source", name="tool-skill")
            workspace = HyperAgentWorkspace(project)
            workspace.init(project / "datasets")
            codex_home = root / "codex"
            with self._with_codex_home(codex_home):
                blocked = SafeAgentToolExecutor(
                    project,
                    workspace.workspace_dir,
                    permission_policy="ask",
                    permission_callback=lambda request: False,
                ).install_skill(path=str(source), dry_run=False)
                allowed = SafeAgentToolExecutor(
                    project,
                    workspace.workspace_dir,
                    permission_policy="ask",
                    permission_callback=lambda request: True,
                ).install_skill(path=str(source), dry_run=False)
            self.assertEqual(blocked.status, "blocked")
            self.assertEqual(allowed.status, "ok")
            self.assertTrue((codex_home / "skills" / "tool-skill" / "SKILL.md").exists())

    def test_skill_installer_slash_alias_reports_existing_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            codex_home = root / "codex"
            _write_skill(codex_home / "skills", name="nature-skill")
            workspace = HyperAgentWorkspace(project)
            workspace.init(project / "datasets")
            outputs = []
            with self._with_codex_home(codex_home):
                repl = HyperAgentRepl(
                    workspace=workspace,
                    conversations=ConversationStore(workspace.workspace_dir),
                    providers=LLMProviderStore(workspace.workspace_dir),
                    prompt_library=PromptLibrary([PROMPT_ROOT]),
                    output_func=outputs.append,
                )
                handled = repl._skill_slash_command("/skill-installer", ["安装", "Nature-skill"])

            self.assertTrue(handled)
            text = "\n".join(outputs)
            self.assertIn("already_installed", text)
            self.assertIn("nature-skill", text)


if __name__ == "__main__":
    unittest.main()
