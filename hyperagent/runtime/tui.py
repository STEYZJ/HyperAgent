"""Stdlib curses fullscreen interface for HyperAgent."""

import json
from pathlib import Path
import unicodedata
from typing import Dict, List, Optional

from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.wait_indicator import WaitIndicator
from hyperagent.runtime.workspace import HyperAgentWorkspace

try:  # pragma: no cover - import availability depends on platform.
    import curses
except Exception:  # pragma: no cover
    curses = None


class TuiWaitIndicator(WaitIndicator):
    """Status-line wait indicator for the fullscreen TUI."""

    def __init__(self, tui: "HyperAgentTui") -> None:
        self.tui = tui

    def start(self, elapsed_sec: float = 0.0) -> None:
        self.tui._set_wait_status(f"思考中...... {elapsed_sec:.0f}s")

    def update(self, elapsed_sec: float) -> None:
        self.tui._set_wait_status(f"思考中...... {elapsed_sec:.0f}s")

    def finish(self, elapsed_sec: float, *, failed: bool = False) -> None:
        label = "思考失败" if failed else "思考完成"
        self.tui._set_wait_status(f"{label}，用时 {elapsed_sec:.1f}s")


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
        self.main_scroll_offset = 0
        self.panel_scroll_offset = 0
        self.wait_status = ""
        self.history_limit = 500
        self.history_cursor: Optional[int] = None
        self.history_draft = ""
        self.history_path = self._history_path()
        self.command_history = self._load_history()

    def run(self) -> int:
        if curses is None:
            print("error: Python curses module is not available on this platform.")
            return 2
        return curses.wrapper(self._run)

    def _run(self, stdscr) -> int:
        self.stdscr = stdscr
        curses.curs_set(1)
        stdscr.keypad(True)
        try:
            mouse_mask = curses.ALL_MOUSE_EVENTS | getattr(
                curses,
                "REPORT_MOUSE_POSITION",
                0,
            )
            curses.mousemask(mouse_mask)
        except curses.error:
            pass
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
            wait_indicator_factory=lambda: TuiWaitIndicator(self),
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
        self.history_cursor = None
        self.history_draft = ""
        while True:
            self._draw(prompt + buffer)
            key = self.stdscr.get_wch()
            if key in ("\n", "\r"):
                self._append_output(prompt + buffer)
                self._record_history(prompt, buffer)
                return buffer
            if key in ("\x1b",):
                return "/exit"
            if key in ("\b", "\x7f") or key == curses.KEY_BACKSPACE:
                buffer = buffer[:-1]
                continue
            if key == curses.KEY_UP:
                buffer = self._history_previous(buffer)
                continue
            if key == curses.KEY_DOWN:
                buffer = self._history_next(buffer)
                continue
            if key in {curses.KEY_PPAGE, curses.KEY_NPAGE, curses.KEY_HOME, curses.KEY_END}:
                self._handle_scroll_key(key)
                continue
            if key == curses.KEY_MOUSE:
                self._handle_mouse()
                continue
            if isinstance(key, str) and key.isprintable():
                buffer += key

    def _draw(self, prompt_line: str = "HyperAgent> ") -> None:
        assert self.stdscr is not None
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        side_width = max(28, min(44, width // 3)) if width >= 90 else 0
        main_width = width - side_width - (1 if side_width else 0)
        reasoning = "collapsed"
        if self.repl is not None:
            reasoning = self.repl._reasoning_display_mode()
        status = (
            f" HyperAgent TUI | provider={self.provider} "
            f"| permission={self.permission_policy} | reasoning={reasoning} "
            f"| /help /exit "
        )
        if self.wait_status:
            status += f"| {self.wait_status} "
        self._addstr(
            0,
            0,
            self._clip_to_width(status, max(width - 1, 0)),
            curses.A_REVERSE,
        )
        log_height = max(1, height - 3)
        main_content_width = max(main_width - 1, 1)
        wrapped_main = self._wrap_lines(self.lines, main_content_width)
        self.main_scroll_offset = self._clamp_scroll_offset(
            self.main_scroll_offset,
            len(wrapped_main),
            log_height,
        )
        visible = self._visible_lines(
            wrapped_main,
            log_height,
            self.main_scroll_offset,
        )
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
            panel_height = max(height - 4, 0)
            self.panel_scroll_offset = self._clamp_scroll_offset(
                self.panel_scroll_offset,
                len(panel_lines),
                panel_height,
            )
            visible_panel = self._visible_lines(
                panel_lines,
                panel_height,
                self.panel_scroll_offset,
            )
            for offset, line in enumerate(visible_panel, start=3):
                self._addstr(offset, panel_x, self._clip_to_width(line, panel_width))
        self._addstr(height - 2, 0, "-" * max(width - 1, 0))
        self._addstr(height - 1, 0, self._clip_to_width(prompt_line, max(width - 1, 1)))
        self.stdscr.refresh()

    def _panel_lines(self) -> List[str]:
        session = self.repl.session_id if self.repl is not None else self.session_id
        reasoning = "collapsed"
        if self.repl is not None:
            reasoning = self.repl._reasoning_display_mode()
        lines = [
            f"session: {session or ''}",
            f"mode: {self.mode}",
            f"model: {self.model or 'profile/default'}",
            f"reasoning: {reasoning}",
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

    def _visible_lines(
        self,
        lines: List[str],
        height: int,
        scroll_offset: int,
    ) -> List[str]:
        if height <= 0:
            return []
        if not lines:
            return []
        end = max(len(lines) - max(scroll_offset, 0), 0)
        start = max(end - height, 0)
        return lines[start:end]

    def _clamp_scroll_offset(
        self,
        scroll_offset: int,
        line_count: int,
        viewport_height: int,
    ) -> int:
        if viewport_height <= 0:
            return 0
        return max(0, min(int(scroll_offset), max(line_count - viewport_height, 0)))

    def _handle_scroll_key(self, key: int) -> None:
        assert self.stdscr is not None
        height, _ = self.stdscr.getmaxyx()
        page = max(height - 5, 1)
        if key == curses.KEY_PPAGE:
            self.main_scroll_offset += page
        elif key == curses.KEY_NPAGE:
            self.main_scroll_offset -= page
        elif key == curses.KEY_HOME:
            self.main_scroll_offset = 10**9
        elif key == curses.KEY_END:
            self.main_scroll_offset = 0

    def _handle_mouse(self) -> None:
        if curses is None:
            return
        try:
            _, x, _, _, state = curses.getmouse()
        except curses.error:
            return
        delta = self._mouse_scroll_delta(state)
        if delta == 0:
            return
        _, width = self.stdscr.getmaxyx()
        side_width = max(28, min(44, width // 3)) if width >= 90 else 0
        main_width = width - side_width - (1 if side_width else 0)
        if side_width and x >= main_width:
            self.panel_scroll_offset += delta
        else:
            self.main_scroll_offset += delta

    def _mouse_scroll_delta(self, state: int) -> int:
        if state & self._mouse_button_event_mask(4):
            return 3
        if state & self._mouse_button_event_mask(5):
            return -3
        return 0

    def _mouse_button_event_mask(self, button: int) -> int:
        # ncurses stores five event bits per button. Python builds may omit
        # BUTTON5_* constants even though terminals still report wheel-down.
        event_bits_without_release = 0b11110
        mask = event_bits_without_release << ((button - 1) * 5)
        if curses is not None:
            pressed = getattr(curses, f"BUTTON{button}_PRESSED", None)
            if pressed:
                mask |= int(pressed)
        return mask

    def _history_path(self) -> Optional[Path]:
        workspace_dir = getattr(self.workspace, "workspace_dir", None)
        if workspace_dir is None:
            return None
        return Path(workspace_dir) / "history" / "tui_history.jsonl"

    def _load_history(self) -> List[str]:
        if self.history_path is None or not self.history_path.exists():
            return []
        history: List[str] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                command = str(item.get("command", "")).strip()
            except json.JSONDecodeError:
                command = line.strip()
            if command:
                history.append(command)
        return history[-self.history_limit :]

    def _record_history(self, prompt: str, command: str) -> None:
        command = command.strip()
        if not self._should_record_history(prompt, command):
            return
        if self.command_history and self.command_history[-1] == command:
            return
        self.command_history.append(command)
        self.command_history = self.command_history[-self.history_limit :]
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")

    def _should_record_history(self, prompt: str, command: str) -> bool:
        if not command:
            return False
        if command in {"/exit", "/quit", "exit", "quit"}:
            return False
        return prompt.startswith("HyperAgent> ")

    def _history_previous(self, current_buffer: str) -> str:
        if not self.command_history:
            return current_buffer
        if self.history_cursor is None:
            self.history_draft = current_buffer
            self.history_cursor = len(self.command_history) - 1
        else:
            self.history_cursor = max(0, self.history_cursor - 1)
        return self.command_history[self.history_cursor]

    def _history_next(self, current_buffer: str = "") -> str:
        if self.history_cursor is None:
            return current_buffer
        if self.history_cursor >= len(self.command_history) - 1:
            self.history_cursor = None
            return self.history_draft
        self.history_cursor += 1
        return self.command_history[self.history_cursor]

    def _set_wait_status(self, text: str) -> None:
        self.wait_status = text
        if self.stdscr is not None:
            self._draw()
