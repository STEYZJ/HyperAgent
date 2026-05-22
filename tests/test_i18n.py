import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hyperagent.cli import main
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.command_aliases import command_help_text
from hyperagent.runtime.i18n import I18nStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.slash_registry import public_commands
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

    def test_chinese_slash_args_hints_are_complete(self):
        store = I18nStore(Path("."))
        packs = {pack.locale: pack.translations for pack in store.list_packs()}
        zh = packs["zh-CN"]

        missing = []
        untranslated = []
        for command in public_commands():
            if not command.args_hint:
                continue
            key = f"slash.command.{command.name}.args_hint"
            if key not in zh:
                missing.append(key)
                continue
            if zh[key] == command.args_hint:
                untranslated.append(key)

        self.assertEqual(missing, [])
        self.assertEqual(untranslated, [])

    def test_chinese_command_help_does_not_leak_common_english_help(self):
        translator = I18nStore(Path(".")).translator("zh-CN")
        help_text = command_help_text(translator)

        forbidden = [
            "Show available commands",
            "Run a prompt",
            "List saved",
            "Generate or edit",
            "[list|run|status|pause|resume|stop]",
            "<session_id> [message]",
        ]
        for phrase in forbidden:
            self.assertNotIn(phrase, help_text)

        self.assertIn("/agents [list列表|run运行|status状态|pause暂停|resume恢复|stop停止]", help_text)
        self.assertIn("/web <status状态|search搜索|fetch抓取|cite引用>", help_text)

    def test_repl_help_uses_chinese_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            translator = I18nStore(root, workspace.workspace_dir).translator("zh-CN")
            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=ConversationStore(workspace.workspace_dir),
                providers=LLMProviderStore(workspace.workspace_dir),
                prompt_library=PromptLibrary([]),
                translator=translator,
            )

            repl_help = repl._help()
            forbidden = [
                "<question>",
                "<instruction>",
                "[title]",
                "<path>",
                "[limit]",
                "<session_id>",
            ]
            for phrase in forbidden:
                self.assertNotIn(phrase, repl_help)
            self.assertIn("/btw <问题>", repl_help)
            self.assertIn("/plan <任务说明>", repl_help)
            self.assertIn("/restore <检查点ID>", repl_help)

    def test_common_cli_command_outputs_are_localized_to_chinese(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chdir(root)
            try:
                self.assertEqual(main(["init", "--dataset-root", str(root / "datasets")]), 0)
                commands = [
                    (["status"], ["是否初始化", "工作区", "数据集根目录"], ["initialized:", "workspace:", "dataset_root:"]),
                    (["web", "status"], ["搜索已配置", "可抓取网页"], ["search_configured:", "fetch_available:"]),
                    (["image", "status"], ["供应商", "需要的环境变量", "是否已配置"], ["provider:", "required_env:", "configured:"]),
                    (["ide-context", "status"], ["是否启用", "打开文件", "更新时间"], ["enabled:", "open_files:", "updated_at:"]),
                    (["plan-mode", "status"], ["是否启用"], ["enabled:"]),
                    (["personality", "status"], ["无交互风格备注"], ["no personality note"]),
                    (["feedback", "list"], ["无反馈"], ["no feedback"]),
                    (["todos"], ["无待办"], ["no todos"]),
                    (["doctor"], ["HyperAgent 自检", "是否初始化"], ["initialized:", "workspace:"]),
                    (["llm-profile", "--profile", "reasonix-cheap"], ["模型=", "thinking=", "推理强度="], ["model=", "effort="]),
                ]
                with patch.dict(os.environ, {}, clear=True):
                    for argv, expected, forbidden in commands:
                        buffer = io.StringIO()
                        with redirect_stdout(buffer):
                            self.assertEqual(main(argv), 0, argv)
                        output = buffer.getvalue()
                        for phrase in expected:
                            self.assertIn(phrase, output, argv)
                        for phrase in forbidden:
                            self.assertNotIn(phrase, output, argv)
            finally:
                os.chdir(old_cwd)

    def test_common_repl_command_outputs_are_localized_to_chinese(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            translator = I18nStore(root, workspace.workspace_dir).translator("zh-CN")
            outputs = []
            repl = HyperAgentRepl(
                workspace=workspace,
                conversations=ConversationStore(workspace.workspace_dir),
                providers=LLMProviderStore(workspace.workspace_dir),
                prompt_library=PromptLibrary([]),
                translator=translator,
                output_func=outputs.append,
            )

            for line in ["/status", "/web status", "/image status", "/feedback list", "/jobs", "/new 中文会话"]:
                repl.handle_line(line)

            output = "\n".join(outputs)
            self.assertIn("是否初始化", output)
            self.assertIn("搜索已配置", output)
            self.assertIn("可抓取网页", output)
            self.assertIn("供应商", output)
            self.assertIn("是否已配置", output)
            self.assertIn("无反馈", output)
            self.assertIn("无后台任务", output)
            self.assertIn("新会话", output)
            forbidden = ["initialized:", "search_configured:", "fetch_available:", "no feedback", "no background jobs", "new session:"]
            for phrase in forbidden:
                self.assertNotIn(phrase, output)

    def test_argparse_error_prefix_is_localized_to_chinese(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(Path(tmp))
            try:
                with patch.dict(os.environ, {}, clear=True):
                    buffer = io.StringIO()
                    with redirect_stderr(buffer):
                        with self.assertRaises(SystemExit) as ctx:
                            main(["llm-profile", "reasonix-cheap"])
                    self.assertEqual(ctx.exception.code, 2)
                    output = buffer.getvalue()
                    self.assertIn("错误:", output)
                    self.assertIn("无法识别的参数", output)
                    self.assertNotIn("error:", output)
                    self.assertNotIn("unrecognized arguments", output)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
