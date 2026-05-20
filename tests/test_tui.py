import unittest

from hyperagent.runtime.tui import HyperAgentTui


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


if __name__ == "__main__":
    unittest.main()
