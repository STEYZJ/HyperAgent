import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hyperagent.runtime.i18n import I18nStore
from hyperagent.runtime.tui import HyperAgentTui, TuiLine
from hyperagent.runtime.workspace import HyperAgentWorkspace


class HyperAgentTuiTest(unittest.TestCase):
    def _tui(self):
        return HyperAgentTui(
            workspace=None,
            conversations=None,
            providers=None,
            prompt_library=None,
        )

    def test_wraps_long_dialog_lines_to_content_width(self):
        tui = self._tui()
        wrapped = tui._wrap_lines(
            [
                "abcdefghij",
                "中文中文中文",
                "artifact: /tmp/hyperagent/some/really/long/path/result.json",
            ],
            width=6,
        )

        self.assertEqual(wrapped[0], "abcdef")
        self.assertIn("ghij", wrapped)
        self.assertTrue(all(tui._display_width(line) <= 6 for line in wrapped))
        self.assertGreater(len(wrapped), 4)

    def test_semantic_wrap_keeps_kind_and_role_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = I18nStore(Path(tmp)).translator("zh-CN")
            tui = HyperAgentTui(
                workspace=None,
                conversations=None,
                providers=None,
                prompt_library=None,
                translator=translator,
            )
            wrapped = tui._wrap_lines(
                [TuiLine("assistant", "高光谱图像分类实验结果需要继续分析")],
                width=18,
                role_labels=True,
            )

            self.assertGreater(len(wrapped), 1)
            self.assertTrue(all(line.kind == "assistant" for line in wrapped))
            self.assertTrue(wrapped[0].text.startswith("助手 │ "))
            self.assertTrue(wrapped[1].text.startswith(" " * tui._display_width("助手 │ ")))
            self.assertTrue(all(tui._display_width(line.text) <= 18 for line in wrapped))

    def test_multiline_output_event_labels_first_line_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = I18nStore(Path(tmp)).translator("zh-CN")
            tui = HyperAgentTui(
                workspace=None,
                conversations=None,
                providers=None,
                prompt_library=None,
                translator=translator,
            )

            tui._append_output_event("tool", "步骤 1: action=tool\n工具：framework_command\n状态：ok")
            wrapped = tui._wrap_lines(tui.lines, width=40, role_labels=True)

            self.assertEqual(wrapped[0].text, "工具 │ 步骤 1: action=tool")
            self.assertEqual(
                wrapped[1].text,
                " " * tui._display_width("工具 │ ") + "工具：framework_command",
            )
            self.assertEqual(
                wrapped[2].text,
                " " * tui._display_width("工具 │ ") + "状态：ok",
            )
            self.assertEqual(sum(1 for line in wrapped if line.text.startswith("工具 │ ")), 1)

    def test_consecutive_assistant_events_share_one_role_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = I18nStore(Path(tmp)).translator("zh-CN")
            tui = HyperAgentTui(
                workspace=None,
                conversations=None,
                providers=None,
                prompt_library=None,
                translator=translator,
            )

            tui._append_output_event("assistant", "\n\n根据技能清单：")
            tui._append_output_event("assistant", "**open-design** - UI 原型设计")
            wrapped = tui._wrap_lines(tui.lines, width=42, role_labels=True)

            self.assertEqual(sum(1 for line in wrapped if line.text.startswith("助手 │ ")), 1)
            self.assertIn("open-design - UI 原型设计", "\n".join(line.text for line in wrapped))
            self.assertNotIn("**", "\n".join(line.text for line in wrapped))

    def test_assistant_display_strips_leading_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = I18nStore(Path(tmp)).translator("zh-CN")
            tui = HyperAgentTui(
                workspace=None,
                conversations=None,
                providers=None,
                prompt_library=None,
                translator=translator,
            )

            tui._append_output_event("assistant", "\n\n你好")
            wrapped = tui._wrap_lines(tui.lines, width=20, role_labels=True)

            self.assertEqual(wrapped[0].text, "助手 │ 你好")
            self.assertEqual(len(wrapped), 1)

    def test_role_labels_are_localized(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = I18nStore(Path(tmp)).translator("zh-CN")
            tui = HyperAgentTui(
                workspace=None,
                conversations=None,
                providers=None,
                prompt_library=None,
                translator=translator,
            )

            self.assertEqual(tui._role_label("user"), "你")
            self.assertEqual(tui._role_label("assistant"), "助手")
            self.assertEqual(tui._role_label("reasoning"), "思考")
            self.assertEqual(tui._role_label("tool"), "工具")
            self.assertEqual(tui._role_label("command"), "命令")

    def test_clip_respects_wide_char_width(self):
        tui = self._tui()
        self.assertEqual(tui._clip_to_width("中文abc", 4), "中文")
        self.assertEqual(tui._clip_to_width("中文abc", 5), "中文a")

    def test_addstr_writes_wide_text_as_one_curses_string(self):
        tui = self._tui()

        class FakeWindow:
            def __init__(self):
                self.calls = []

            def getmaxyx(self):
                return (5, 20)

            def addstr(self, y, x, text, attr=0):
                self.calls.append((y, x, text, attr))

        fake = FakeWindow()
        tui.stdscr = fake

        tui._addstr(1, 3, "你好A")

        self.assertEqual(fake.calls, [(1, 3, "你好A", 0)])

    def test_addstr_clips_wide_text_before_right_edge(self):
        tui = self._tui()

        class FakeWindow:
            def __init__(self):
                self.calls = []

            def getmaxyx(self):
                return (5, 8)

            def addstr(self, y, x, text, attr=0):
                self.calls.append((y, x, text, attr))

        fake = FakeWindow()
        tui.stdscr = fake

        tui._addstr(1, 3, "你好ABC")

        self.assertEqual(fake.calls, [(1, 3, "你好A", 0)])

    def test_draw_uses_kind_specific_attrs(self):
        tui = self._tui()
        tui.lines = [
            TuiLine("user", "你好"),
            TuiLine("assistant", "收到"),
            TuiLine("tool", "工件: /tmp/result.json"),
        ]
        tui.styles = {
            "user": 11,
            "assistant": 22,
            "tool": 33,
            "system": 44,
            "panel": 55,
        }

        class FakeWindow:
            def __init__(self):
                self.calls = []

            def getmaxyx(self):
                return (10, 80)

            def erase(self):
                pass

            def refresh(self):
                pass

            def move(self, y, x):
                pass

            def addstr(self, y, x, text, attr=0):
                self.calls.append((y, x, text, attr))

        fake = FakeWindow()
        tui.stdscr = fake

        tui._draw("HyperAgent > ")

        attrs_by_text = {call[2]: call[3] for call in fake.calls}
        self.assertEqual(attrs_by_text["You │ 你好"], 11)
        self.assertEqual(attrs_by_text["Assistant │ 收到"], 22)
        self.assertEqual(attrs_by_text["Tool │ 工件: /tmp/result.json"], 33)

    def test_style_for_kind_falls_back_without_colors(self):
        tui = self._tui()
        tui.styles = {}

        style = tui._style_for_kind("error")

        self.assertIsInstance(style, int)

    def test_wrap_and_render_keeps_chinese_content(self):
        tui = self._tui()
        paragraph = (
            "你好！我是HyperAgent，一个专注于高光谱图像分类的持续性研究代理。"
            "请提供任务ID、数据集、目标和目的。"
        )

        class FakeWindow:
            def __init__(self):
                self.calls = []

            def addstr(self, y, x, text, attr=0):
                self.calls.append((y, x, text, attr))

        wrapped = tui._wrap_lines([paragraph], width=30)
        fake = FakeWindow()
        tui.stdscr = fake
        for row, line in enumerate(wrapped):
            tui._addstr(row, 0, tui._clip_to_width(line, 30))

        rendered = "".join(call[2] for call in fake.calls)
        self.assertEqual(rendered, "".join(str(line) for line in wrapped))
        self.assertIn("高光谱图像分类", rendered)
        self.assertIn("持续性研究代理", rendered)

    def test_init_locale_warns_for_non_utf8_terminal(self):
        tui = self._tui()

        with patch("hyperagent.runtime.tui.locale.setlocale"), patch(
            "hyperagent.runtime.tui.locale.getpreferredencoding",
            return_value="ANSI_X3.4-1968",
        ):
            warning = tui._init_locale()

        self.assertIn("UTF-8", warning)

    def test_scroll_visible_lines_and_clamp(self):
        tui = self._tui()
        lines = [f"line {index}" for index in range(10)]

        self.assertEqual(tui._visible_lines(lines, height=3, scroll_offset=0), lines[-3:])
        self.assertEqual(tui._visible_lines(lines, height=3, scroll_offset=2), lines[-5:-2])
        self.assertEqual(tui._clamp_scroll_offset(100, line_count=10, viewport_height=3), 7)
        self.assertEqual(tui._clamp_scroll_offset(-5, line_count=10, viewport_height=3), 0)

    def test_mouse_scroll_delta_supports_wheel_down_fallback(self):
        tui = self._tui()
        button4_pressed = 2 << 15
        button5_pressed = 2 << 20
        button5_clicked = 4 << 20

        self.assertEqual(tui._mouse_scroll_delta(button4_pressed), 3)
        self.assertEqual(tui._mouse_scroll_delta(button5_pressed), -3)
        self.assertEqual(tui._mouse_scroll_delta(button5_clicked), -3)
        self.assertEqual(tui._mouse_scroll_delta(0), 0)

    def test_panel_uses_reasoning_display_language(self):
        tui = self._tui()
        tui.repl = SimpleNamespace(
            session_id="session-1",
            _reasoning_display_mode=lambda: "expanded",
        )

        lines = tui._panel_lines()
        self.assertIn("reasoning: expanded", lines)
        self.assertIn("mouse: interactive", lines)
        self.assertNotIn("thinking: on", "\n".join(lines))

    def test_panel_translates_common_status_labels_to_chinese(self):
        with tempfile.TemporaryDirectory() as tmp:
            translator = I18nStore(Path(tmp)).translator("zh-CN")
            tui = HyperAgentTui(
                workspace=None,
                conversations=None,
                providers=None,
                prompt_library=None,
                translator=translator,
            )
            tui.repl = SimpleNamespace(
                session_id="session-1",
                _reasoning_display_mode=lambda: "expanded",
            )

            text = "\n".join(tui._panel_lines())

            self.assertIn("会话: session-1", text)
            self.assertIn("思考显示: 展开", text)
            self.assertIn("鼠标: 交互", text)
            self.assertIn("活跃子智能体:", text)
            self.assertIn("命令建议:", text)
            self.assertNotIn("reasoning:", text)
            self.assertNotIn("mouse:", text)

    def test_main_prompt_uses_cwd_and_hyperagent_marker(self):
        tui = self._tui()

        with patch(
            "hyperagent.runtime.tui.Path.cwd",
            return_value=Path("/data2/lzj/HyperAgent"),
        ):
            self.assertEqual(
                tui._main_prompt(),
                "/data2/lzj/HyperAgent HyperAgent > ",
            )

    def test_main_prompt_left_elides_cwd_when_narrow(self):
        tui = self._tui()
        prompt = "/data2/lzj/some/deep/HyperAgent HyperAgent > "

        fitted = tui._fit_input_prompt(prompt, width=42)

        self.assertLessEqual(tui._display_width(fitted), 41)
        self.assertIn("...", fitted)
        self.assertTrue(fitted.endswith("HyperAgent > "))

    def test_main_prompt_falls_back_for_tiny_width(self):
        tui = self._tui()

        self.assertEqual(
            tui._fit_input_prompt("/very/long/path HyperAgent > ", width=14),
            "HyperAgent > ",
        )
        self.assertEqual(
            tui._fit_input_prompt("/very/long/path HyperAgent > ", width=4),
            "> ",
        )

    def test_non_shell_prompt_is_clipped_not_replaced(self):
        tui = self._tui()

        self.assertEqual(tui._fit_input_prompt("allow? [y/N] ", width=8), "allow? ")

    def test_input_edit_helpers_preserve_cursor_position(self):
        tui = self._tui()

        buffer, cursor = tui._insert_text("abcd", 2, "X")
        self.assertEqual((buffer, cursor), ("abXcd", 3))

        buffer, cursor = tui._backspace_text(buffer, cursor)
        self.assertEqual((buffer, cursor), ("abcd", 2))

        buffer, cursor = tui._delete_text(buffer, cursor)
        self.assertEqual((buffer, cursor), ("abd", 2))

        buffer, cursor = tui._delete_previous_word("alpha beta  gamma", 18)
        self.assertEqual((buffer, cursor), ("alpha beta  ", 12))

        buffer, cursor = tui._delete_previous_word("中文 实验", 5)
        self.assertEqual((buffer, cursor), ("中文 ", 3))

    def test_read_line_ctrl_u_clears_current_input(self):
        tui = self._tui()

        class FakeWindow:
            def __init__(self):
                self.keys = list("abc") + ["\x15"] + list("中文") + ["\n"]

            def getmaxyx(self):
                return (10, 80)

            def get_wch(self):
                return self.keys.pop(0)

            def erase(self):
                pass

            def refresh(self):
                pass

            def move(self, y, x):
                pass

            def addstr(self, y, x, text, attr=0):
                pass

        tui.stdscr = FakeWindow()

        line = tui._read_line("/tmp/project HyperAgent > ")

        self.assertEqual(line, "中文")
        self.assertEqual(tui.lines[-1].kind, "user")
        self.assertEqual(tui.lines[-1].text, "中文")

    def test_input_click_maps_wide_char_columns_to_cursor_index(self):
        tui = self._tui()
        prompt = "P> "
        buffer = "a中b"
        prompt_width = tui._display_width(prompt)

        self.assertEqual(
            tui._cursor_index_from_input_x(prompt, buffer, 0, prompt_width),
            0,
        )
        self.assertEqual(
            tui._cursor_index_from_input_x(prompt, buffer, 0, prompt_width + 1),
            1,
        )
        self.assertEqual(
            tui._cursor_index_from_input_x(prompt, buffer, 0, prompt_width + 2),
            2,
        )
        self.assertEqual(
            tui._cursor_index_from_input_x(prompt, buffer, 0, prompt_width + 4),
            3,
        )

    def test_input_prompt_view_keeps_cursor_visible_for_long_input(self):
        tui = self._tui()
        line, cursor_x, view_start = tui._input_prompt_view(
            "> ",
            "abcdef",
            cursor_index=6,
            width=5,
        )

        self.assertEqual(line, "> def")
        self.assertEqual(cursor_x, 5)
        self.assertEqual(view_start, 3)

    def test_tui_history_persists_only_main_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            tui = HyperAgentTui(
                workspace=workspace,
                conversations=None,
                providers=None,
                prompt_library=None,
            )
            prompt = "/tmp/project HyperAgent > "

            tui._record_history(prompt, "/status")
            tui._record_history("allow? [y/N] ", "y")
            tui._record_history(prompt, "/context")
            tui._record_history(prompt, "/exit")

            reloaded = HyperAgentTui(
                workspace=workspace,
                conversations=None,
                providers=None,
                prompt_library=None,
            )
            self.assertEqual(reloaded.command_history, ["/status", "/context"])
            self.assertEqual(reloaded._history_previous("draft"), "/context")
            self.assertEqual(reloaded._history_previous("/context"), "/status")
            self.assertEqual(reloaded._history_next(), "/context")
            self.assertEqual(reloaded._history_next(), "draft")

    def test_mouse_mode_commands_switch_tui_behavior(self):
        tui = self._tui()

        self.assertEqual(tui.mouse_mode, "interactive")
        self.assertTrue(tui._handle_tui_command("/mouse select"))
        self.assertEqual(tui.mouse_mode, "selection")
        self.assertIn("selection", tui.lines[-1].text)

        self.assertTrue(tui._handle_tui_command("/mouse interactive"))
        self.assertEqual(tui.mouse_mode, "interactive")
        self.assertIn("interactive", tui.lines[-1].text)

        self.assertTrue(tui._handle_tui_command("/mouse toggle"))
        self.assertEqual(tui.mouse_mode, "selection")

    def test_mouse_event_is_ignored_in_selection_mode(self):
        tui = self._tui()
        tui.mouse_mode = "selection"
        tui.main_scroll_offset = 0
        tui.stdscr = SimpleNamespace(getmaxyx=lambda: (20, 100))

        tui._handle_mouse_event(0, 2, 2 << 15)

        self.assertEqual(tui.main_scroll_offset, 0)

    def test_tui_suggests_builtin_and_markdown_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            tui = HyperAgentTui(
                workspace=workspace,
                conversations=None,
                providers=None,
                prompt_library=None,
            )
            tui.repl = SimpleNamespace(
                command_store=__import__(
                    "hyperagent.runtime.commands",
                    fromlist=["SlashCommandStore"],
                ).SlashCommandStore(root, workspace.workspace_dir),
            )

            self.assertIn("/help", tui._suggest_commands("/"))
            suggestions = tui._suggest_commands("/fea")

            self.assertIn("/feature-dev", suggestions)
            self.assertIn("/background", tui._suggest_commands("/back"))

    def test_tui_suggests_skill_slash_commands(self):
        tui = self._tui()
        tui.repl = SimpleNamespace(
            command_store=SimpleNamespace(discover=lambda: []),
            skill_names=lambda: ["open-design", "spectral-critic"],
        )

        self.assertIn("/skill open-design", tui._suggest_commands("/open"))
        self.assertIn("/skills", tui._suggest_commands("/"))


if __name__ == "__main__":
    unittest.main()
