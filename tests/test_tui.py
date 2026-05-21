import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hyperagent.runtime.tui import HyperAgentTui
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

    def test_clip_respects_wide_char_width(self):
        tui = self._tui()
        self.assertEqual(tui._clip_to_width("中文abc", 4), "中文")
        self.assertEqual(tui._clip_to_width("中文abc", 5), "中文a")

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

    def test_shell_prompt_uses_env_user_host_and_cwd(self):
        tui = self._tui()

        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "HyperAgent"}, clear=True), patch(
            "hyperagent.runtime.tui.getpass.getuser",
            return_value="lzj",
        ), patch(
            "hyperagent.runtime.tui.socket.gethostname",
            return_value="nwafu-406.example",
        ), patch(
            "hyperagent.runtime.tui.Path.cwd",
            return_value=Path("/data2/lzj/HyperAgent"),
        ):
            self.assertEqual(
                tui._main_prompt(),
                "(HyperAgent) lzj@nwafu-406:/data2/lzj/HyperAgent$ ",
            )

    def test_shell_prompt_left_elides_cwd_when_narrow(self):
        tui = self._tui()
        prompt = "(HyperAgent) lzj@nwafu-406:/data2/lzj/some/deep/HyperAgent$ "

        fitted = tui._fit_input_prompt(prompt, width=42)

        self.assertLessEqual(tui._display_width(fitted), 41)
        self.assertIn("...", fitted)
        self.assertTrue(fitted.endswith("$ "))

    def test_shell_prompt_falls_back_for_tiny_width(self):
        tui = self._tui()

        self.assertEqual(
            tui._fit_input_prompt("(HyperAgent) user@host:/very/long/path$ ", width=4),
            "$ ",
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
            prompt = "(HyperAgent) user@host:/tmp/project$ "

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
        self.assertIn("selection", tui.lines[-1])

        self.assertTrue(tui._handle_tui_command("/mouse interactive"))
        self.assertEqual(tui.mouse_mode, "interactive")
        self.assertIn("interactive", tui.lines[-1])

        self.assertTrue(tui._handle_tui_command("/mouse toggle"))
        self.assertEqual(tui.mouse_mode, "selection")

    def test_mouse_event_is_ignored_in_selection_mode(self):
        tui = self._tui()
        tui.mouse_mode = "selection"
        tui.main_scroll_offset = 0
        tui.stdscr = SimpleNamespace(getmaxyx=lambda: (20, 100))

        tui._handle_mouse_event(0, 2, 2 << 15)

        self.assertEqual(tui.main_scroll_offset, 0)


if __name__ == "__main__":
    unittest.main()
