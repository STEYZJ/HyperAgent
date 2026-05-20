import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hyperagent.cli import main
from hyperagent.runtime.i18n import I18nStore
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


if __name__ == "__main__":
    unittest.main()
