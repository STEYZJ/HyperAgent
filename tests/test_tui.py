import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        self.assertNotIn("thinking: on", "\n".join(lines))

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

            tui._record_history("HyperAgent> ", "/status")
            tui._record_history("allow? [y/N] ", "y")
            tui._record_history("HyperAgent> ", "/context")
            tui._record_history("HyperAgent> ", "/exit")

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


if __name__ == "__main__":
    unittest.main()
