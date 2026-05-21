import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hyperagent.cli import main
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.command_aliases import command_help_text
from hyperagent.runtime.i18n import I18nStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.workspace import HyperAgentWorkspace


class I18nTest(unittest.TestCase):
    def test_translator_defaults_to_chinese_and_falls_back_to_english(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = I18nStore(root)
            translator = store.translator(store.resolve_locale([]))

            self.assertEqual(translator.locale, "zh-CN")
            self.assertIn("HyperAgent", translator.t("cli.description"))
            self.assertEqual(
                translator.t("missing.key", default="fallback {value}", value=3),
                "fallback 3",
            )

    def test_user_language_pack_can_override_builtin_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            store = I18nStore(root, workspace.workspace_dir)
            pack = root / "custom.json"
            pack.write_text(
                json.dumps(
                    {
                        "locale": "zh-CN",
                        "translations": {"cli.description": "自定义 HyperAgent"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            store.install(pack)
            self.assertEqual(store.translator("zh-CN").t("cli.description"), "自定义 HyperAgent")

    def test_builtin_chinese_and_english_packs_have_same_keys(self):
        store = I18nStore(Path("."))
        packs = {pack.locale: pack.translations for pack in store.list_packs()}

        self.assertEqual(set(packs["zh-CN"]), set(packs["en"]))

    def test_cli_language_commands_and_config_locale(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chdir(root)
            try:
                self.assertEqual(main(["init", "--dataset-root", str(root / "datasets")]), 0)
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(main(["language-list"]), 0)
                self.assertIn("zh-CN", buffer.getvalue())
                self.assertIn("en", buffer.getvalue())

                self.assertEqual(main(["language-set", "en"]), 0)
                config = HyperAgentWorkspace(root).load_config()
                self.assertEqual(config.metadata["locale"], "en")
            finally:
                os.chdir(old_cwd)

    def test_cli_help_defaults_to_chinese_and_lang_en_switches_back(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(Path(tmp))
            try:
                with patch.dict(os.environ, {}, clear=True):
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        with self.assertRaises(SystemExit) as ctx:
                            main(["status", "--help"])
                    self.assertEqual(ctx.exception.code, 0)
                    self.assertIn("显示 HyperAgent 工作区状态", buffer.getvalue())

                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        with self.assertRaises(SystemExit) as ctx:
                            main(["--lang", "en", "status", "--help"])
                    self.assertEqual(ctx.exception.code, 0)
                    self.assertIn("Show HyperAgent workspace status", buffer.getvalue())

                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        with self.assertRaises(SystemExit) as ctx:
                            main(["tui", "--help"])
                    self.assertEqual(ctx.exception.code, 0)
                    self.assertIn("启动全屏 curses HyperAgent 界面", buffer.getvalue())
                    self.assertIn("大模型供应商名称", buffer.getvalue())
                    self.assertIn("用法:", buffer.getvalue())
                    self.assertIn("可选参数:", buffer.getvalue())
                    self.assertIn("显示此帮助信息并退出", buffer.getvalue())
                    self.assertNotIn("optional arguments:", buffer.getvalue())
                    self.assertNotIn("show this help message and exit", buffer.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_env_language_switches_help_to_english(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(Path(tmp))
            try:
                with patch.dict(os.environ, {"HYPERAGENT_LANG": "en"}, clear=True):
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        with self.assertRaises(SystemExit) as ctx:
                            main(["status", "--help"])
                    self.assertEqual(ctx.exception.code, 0)
                    self.assertIn("Show HyperAgent workspace status", buffer.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_repl_and_launcher_help_translate_command_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            translator = I18nStore(root, workspace.workspace_dir).translator("zh-CN")

            launcher_help = command_help_text(translator)
            self.assertIn("已注册斜杠命令", launcher_help)
            self.assertIn("智能体:", launcher_help)
            self.assertIn("/agents", launcher_help)
            self.assertIn("列出、运行、暂停、恢复或停止子智能体", launcher_help)
            self.assertNotIn("Registered slash commands", launcher_help)
            self.assertNotIn("Central command registry", launcher_help)
            self.assertNotIn("Show available commands", launcher_help)

            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=ConversationStore(workspace.workspace_dir),
                providers=LLMProviderStore(workspace.workspace_dir),
                prompt_library=PromptLibrary([]),
                translator=translator,
            )
            repl_help = repl._help()
            self.assertIn("中央命令注册表", repl_help)
            self.assertIn("显示可用命令", repl_help)
            self.assertNotIn("Central command registry", repl_help)
            self.assertNotIn("Show available commands", repl_help)


if __name__ == "__main__":
    unittest.main()
