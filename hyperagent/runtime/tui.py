"""Stdlib curses fullscreen interface for HyperAgent."""

import unicodedata
from typing import Dict, List, Optional

from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.workspace import HyperAgentWorkspace

try:  # pragma: no cover - import availability depends on platform.
    import curses
except Exception:  # pragma: no cover
    curses = None


class HyperAgentTui:
    """Small curses wrapper around the existing REPL command handler."""

    def __init__(
        self,
        *,
        workspace: HyperAgentWorkspace,
        conversations: ConversationStore,
        providers: LLMProviderStore,
        prompt_library: PromptLibrary,
        provider: str = "deepseek",
        model: Optional[str] = None,
        mode: str = "research",
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        new_title: Optional[str] = None,
        permission_policy: str = "session-ask",
        max_context_chars: int = 12000,
        keep_last: int = 6,
        llm_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        self.workspace = workspace
        self.conversations = conversations
        self.providers = providers
        self.prompt_library = prompt_library
        self.provider = provider
        self.model = model
        self.mode = mode
        self.task_id = task_id
        self.session_id = session_id
        self.new_title = new_title
        self.permission_policy = permission_policy
        self.max_context_chars = max_context_chars
        self.keep_last = keep_last
        self.llm_kwargs = dict(llm_kwargs or {})
        self.lines: List[str] = []
        self.stdscr = None
        self.repl: Optional[HyperAgentRepl] = None

    def run(self) -> int:
        if curses is None:
            print("error: Python curses module is not available on this platform.")
            return 2
        return curses.wrapper(self._run)

    def _run(self, stdscr) -> int:
        self.stdscr = stdscr
        curses.curs_set(1)
        stdscr.keypad(True)
        self.repl = HyperAgentRepl(
            workspace=self.workspace,
            conversations=self.conversations,
            providers=self.providers,
            prompt_library=self.prompt_library,
            provider=self.provider,
            model=self.model,
            mode=self.mode,
            task_id=self.task_id,
            session_id=self.session_id,
            new_title=self.new_title,
            permission_policy=self.permission_policy,
            max_context_chars=self.max_context_chars,
            keep_last=self.keep_last,
            llm_kwargs=self.llm_kwargs,
            input_func=self._prompt_dialog,
            output_func=self._append_output,
        )
        self._append_output(self.repl._banner())
        while True:
            self._draw()
            line = self._read_line("HyperAgent> ")
            if not line.strip():
                continue
            assert self.repl is not None
            keep_running = self.repl.handle_line(line.strip())
            if not keep_running:
                self._draw()
                return 0

    def _append_output(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
        self.lines = self.lines[-1000:]
        if self.stdscr is not None:
            self._draw()

    def _prompt_dialog(self, prompt: str) -> str:
        self._append_output(prompt)
        return self._read_line(prompt)

    def _read_line(self, prompt: str) -> str:
        assert self.stdscr is not None
        buffer = ""
        while True:
            self._draw(prompt + buffer)
            key = self.stdscr.get_wch()
            if key in ("\n", "\r"):
                self._append_output(prompt + buffer)
                return buffer
            if key in ("\x1b",):
                return "/exit"
            if key in ("\b", "\x7f") or key == curses.KEY_BACKSPACE:
                buffer = buffer[:-1]
                continue
            if isinstance(key, str) and key.isprintable():
                buffer += key

    def _draw(self, prompt_line: str = "HyperAgent> ") -> None:
        assert self.stdscr is not None
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        side_width = max(28, min(44, width // 3)) if width >= 90 else 0
        main_width = width - side_width - (1 if side_width else 0)
        status = (
            f" HyperAgent TUI | provider={self.provider} "
            f"| permission={self.permission_policy} | /help /exit "
        )
        self._addstr(0, 0, status[: max(width - 1, 0)], curses.A_REVERSE)
        log_height = max(1, height - 3)
        main_content_width = max(main_width - 1, 1)
        visible = self._wrap_lines(self.lines, main_content_width)[-log_height:]
        for index, line in enumerate(visible, start=1):
            self._addstr(index, 0, self._clip_to_width(line, main_content_width))
        if side_width:
            separator_x = main_width
            for row in range(1, height - 1):
                self._addstr(row, separator_x, "|")
            panel_x = separator_x + 1
            panel_width = max(side_width - 1, 1)
            self._addstr(
                1,
                panel_x,
                self._clip_to_width("Agent/Tool Panel", panel_width),
                curses.A_BOLD,
            )
            panel_lines = self._wrap_lines(self._panel_lines(), panel_width)
            for offset, line in enumerate(panel_lines[: max(height - 4, 0)], start=3):
                self._addstr(offset, panel_x, self._clip_to_width(line, panel_width))
        self._addstr(height - 2, 0, "-" * max(width - 1, 0))
        self._addstr(height - 1, 0, self._clip_to_width(prompt_line, max(width - 1, 1)))
        self.stdscr.refresh()

    def _panel_lines(self) -> List[str]:
        session = self.repl.session_id if self.repl is not None else self.session_id
        lines = [
            f"session: {session or ''}",
            f"mode: {self.mode}",
            f"model: {self.model or 'profile/default'}",
            "",
            "recent artifacts:",
        ]
        artifact_lines = [
            line
            for line in self.lines[-80:]
            if "artifact:" in line or "agent_run:" in line or "action_run:" in line
        ]
        lines.extend(artifact_lines[-10:] or ["none"])
        return lines

    def _addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        assert self.stdscr is not None
        try:
            self.stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def _wrap_lines(self, lines: List[str], width: int) -> List[str]:
        wrapped: List[str] = []
        for line in lines:
            wrapped.extend(self._wrap_line(line, width))
        return wrapped

    def _wrap_line(self, line: str, width: int) -> List[str]:
        width = max(int(width), 1)
        expanded = str(line).expandtabs(4)
        if not expanded:
            return [""]
        parts: List[str] = []
        current: List[str] = []
        current_width = 0
        for char in expanded:
            char_width = self._char_width(char)
            if current and current_width + char_width > width:
                parts.append("".join(current))
                current = []
                current_width = 0
            if char_width > width:
                parts.append(char)
                continue
            current.append(char)
            current_width += char_width
        if current:
            parts.append("".join(current))
        return parts or [""]

    def _clip_to_width(self, text: str, width: int) -> str:
        width = max(int(width), 0)
        if width <= 0:
            return ""
        current_width = 0
        output: List[str] = []
        for char in str(text).expandtabs(4):
            char_width = self._char_width(char)
            if current_width + char_width > width:
                break
            output.append(char)
            current_width += char_width
        return "".join(output)

    def _display_width(self, text: str) -> int:
        return sum(self._char_width(char) for char in str(text).expandtabs(4))

    def _char_width(self, char: str) -> int:
        if not char:
            return 0
        if unicodedata.combining(char):
            return 0
        if unicodedata.category(char) in {"Cc", "Cf"}:
            return 0
        return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
