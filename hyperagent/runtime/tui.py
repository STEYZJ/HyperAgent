"""Stdlib curses fullscreen interface for HyperAgent."""

import getpass
import json
import os
from pathlib import Path
import socket
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.i18n import Translator
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
        translator: Optional[Translator] = None,
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
        self.translator = translator
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
        self.mouse_mode = "interactive"
        self._main_prompt_suffix = "$ "
        self.command_suggestions: List[str] = []

    def run(self) -> int:
        if curses is None:
            print(
                self._t(
                    "tui.curses_missing",
                    "error: Python curses module is not available on this platform.",
                )
            )
            return 2
        return curses.wrapper(self._run)

    def _run(self, stdscr) -> int:
        self.stdscr = stdscr
        curses.curs_set(1)
        stdscr.keypad(True)
        self._set_mouse_mode("interactive")
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
            translator=self.translator,
            input_func=self._prompt_dialog,
            output_func=self._append_output,
            wait_indicator_factory=lambda: TuiWaitIndicator(self),
        )
        self._append_output(self.repl._banner())
        while True:
            self._draw()
            line = self._read_line(self._main_prompt())
            if not line.strip():
                continue
            if self._handle_tui_command(line.strip()):
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
        cursor_index = 0
        self.history_cursor = None
        self.history_draft = ""
        while True:
            _, width = self.stdscr.getmaxyx()
            display_prompt = self._fit_input_prompt(prompt, max(width - 1, 1))
            prompt_line, cursor_x, view_start = self._input_prompt_view(
                display_prompt,
                buffer,
                cursor_index,
                max(width - 1, 1),
            )
            self.command_suggestions = self._suggest_commands(buffer)
            self._draw(prompt_line, cursor_x=cursor_x)
            key = self.stdscr.get_wch()
            if key in ("\n", "\r"):
                self._append_output(prompt + buffer)
                self._record_history(prompt, buffer)
                return buffer
            if key in ("\x1b",):
                return "/exit"
            if key in ("\b", "\x7f") or key == curses.KEY_BACKSPACE:
                buffer, cursor_index = self._backspace_text(buffer, cursor_index)
                self._reset_history_browse()
                continue
            if key == curses.KEY_DC:
                buffer, cursor_index = self._delete_text(buffer, cursor_index)
                self._reset_history_browse()
                continue
            if key == curses.KEY_LEFT:
                cursor_index = max(0, cursor_index - 1)
                continue
            if key == curses.KEY_RIGHT:
                cursor_index = min(len(buffer), cursor_index + 1)
                continue
            if key == curses.KEY_UP:
                buffer = self._history_previous(buffer)
                cursor_index = len(buffer)
                continue
            if key == curses.KEY_DOWN:
                buffer = self._history_next(buffer)
                cursor_index = len(buffer)
                continue
            if key in {curses.KEY_PPAGE, curses.KEY_NPAGE, curses.KEY_HOME, curses.KEY_END}:
                self._handle_scroll_key(key)
                continue
            if self._is_f2_key(key):
                self._toggle_mouse_mode()
                continue
            if key == curses.KEY_MOUSE:
                if self.mouse_mode == "selection":
                    continue
                event = self._read_mouse_event()
                if event is None:
                    continue
                x, y, state = event
                if self._mouse_scroll_delta(state):
                    self._handle_mouse_event(x, y, state)
                    continue
                if self._mouse_targets_input(y, state):
                    cursor_index = self._cursor_index_from_input_x(
                        display_prompt,
                        buffer,
                        view_start,
                        x,
                    )
                    self._reset_history_browse()
                    continue
                self._handle_mouse_event(x, y, state)
                continue
            if isinstance(key, str) and key.isprintable():
                buffer, cursor_index = self._insert_text(buffer, cursor_index, key)
                self._reset_history_browse()

    def _main_prompt(self) -> str:
        return f"({self._environment_name()}) {self._user_name()}@{self._host_name()}:{Path.cwd()}$ "

    def _environment_name(self) -> str:
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
        if env_name:
            return env_name
        prefix_name = Path(sys.prefix).name if sys.prefix else ""
        if prefix_name and prefix_name not in {"", "usr", "local"}:
            return prefix_name
        return "HyperAgent"

    def _user_name(self) -> str:
        return getpass.getuser() or "user"

    def _host_name(self) -> str:
        hostname = socket.gethostname().split(".", 1)[0].strip()
        return hostname or "localhost"

    def _fit_input_prompt(
        self,
        prompt: str,
        width: int,
        *,
        min_input_width: int = 1,
    ) -> str:
        prompt_budget = max(int(width) - max(int(min_input_width), 0), 1)
        if self._display_width(prompt) <= prompt_budget:
            return prompt
        if not (prompt.endswith("$ ") and ":" in prompt):
            return self._clip_to_width(prompt, prompt_budget)
        if prompt.endswith("$ ") and ":" in prompt:
            prefix, cwd_with_suffix = prompt.rsplit(":", 1)
            cwd_text = cwd_with_suffix[:-2]
            shell_prefix = f"{prefix}:"
            shell_suffix = "$ "
            cwd_budget = prompt_budget - self._display_width(shell_prefix + shell_suffix)
            if cwd_budget > 0:
                candidate = f"{shell_prefix}{self._left_ellipsis(cwd_text, cwd_budget)}{shell_suffix}"
                if self._display_width(candidate) <= prompt_budget:
                    return candidate
        for fallback in ("HyperAgent$ ", "$ ", "$"):
            if self._display_width(fallback) <= prompt_budget:
                return fallback
        return ""

    def _left_ellipsis(self, text: str, width: int) -> str:
        width = max(int(width), 0)
        if self._display_width(text) <= width:
            return text
        marker = "..."
        marker_width = self._display_width(marker)
        if width <= marker_width:
            return self._clip_to_width(marker, width)
        suffix_budget = width - marker_width
        suffix: List[str] = []
        current_width = 0
        for char in reversed(str(text)):
            char_width = self._char_width(char)
            if current_width + char_width > suffix_budget:
                break
            suffix.append(char)
            current_width += char_width
        return marker + "".join(reversed(suffix))

    def _draw(self, prompt_line: str = "HyperAgent> ", cursor_x: Optional[int] = None) -> None:
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
        status += f"| mouse={self.mouse_mode} "
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
                self._clip_to_width(
                    self._t("tui.panel.title", "Agent/Tool Panel"),
                    panel_width,
                ),
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
        if cursor_x is not None:
            try:
                self.stdscr.move(height - 1, max(0, min(int(cursor_x), max(width - 1, 0))))
            except curses.error:
                pass
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
            f"permission: {self.permission_policy}",
            f"reasoning: {reasoning}",
            f"mouse: {self.mouse_mode}",
        ]
        if self.repl is not None:
            try:
                usage = self.repl.usage.summarize()
                lines.extend(
                    [
                        f"llm_requests: {usage['request_count']}",
                        f"tokens: {usage['total_tokens']}",
                        f"cache_hit: {usage['cache_hit_ratio']}",
                        f"loop: {self.repl.action_loop_mode}",
                        f"budget: {self.repl.action_token_budget}",
                    ]
                )
            except Exception:
                pass
        lines.extend(["", self._t("tui.panel.commands", "command suggestions:")])
        lines.extend(self.command_suggestions[:6] or [self._t("tui.panel.none", "none")])
        lines.append("")
        lines.append(self._t("tui.panel.todos", "todos:"))
        lines.extend(self._todo_panel_lines(session or "project"))
        lines.extend([
            "",
            self._t("tui.panel.recent_artifacts", "recent artifacts:"),
        ])
        artifact_lines = [
            line
            for line in self.lines[-80:]
            if "artifact:" in line or "agent_run:" in line or "action_run:" in line
        ]
        lines.extend(artifact_lines[-10:] or [self._t("tui.panel.none", "none")])
        return lines

    def _todo_panel_lines(self, owner: str) -> List[str]:
        if self.repl is None:
            return [self._t("tui.panel.none", "none")]
        try:
            todo_list = self.repl.todos.load(owner)
        except Exception:
            return [self._t("tui.panel.none", "none")]
        if not todo_list.items:
            return [self._t("tui.panel.none", "none")]
        return [
            f"- {item.status}: {item.content}"
            for item in todo_list.items[:6]
        ]

    def _suggest_commands(self, buffer: str) -> List[str]:
        if self.repl is None:
            return []
        text = buffer.strip()
        if not text.startswith("/"):
            return []
        query = text[1:].split()[0].lower()
        builtins = [
            "help",
            "status",
            "context",
            "usage",
            "agents",
            "commands",
            "todos",
            "hooks",
            "permissions",
            "export",
            "doctor",
            "tool",
            "act",
            "plan",
            "cost",
            "stats",
            "skill",
            "checkpoint",
            "restore",
            "budget",
            "pro",
            "logs",
            "mouse",
            "exit",
        ]
        custom = [command.name for command in self.repl.command_store.discover()]
        matches = sorted({name for name in builtins + custom if name.startswith(query)})
        return ["/" + name for name in matches[:8]]

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

    def _input_prompt_view(
        self,
        prompt: str,
        buffer: str,
        cursor_index: int,
        width: int,
    ) -> Tuple[str, int, int]:
        width = max(int(width), 1)
        cursor_index = max(0, min(int(cursor_index), len(buffer)))
        prompt_width = self._display_width(prompt)
        input_width = max(width - prompt_width, 1)
        view_start = 0
        while (
            view_start < cursor_index
            and self._display_width(buffer[view_start:cursor_index]) > input_width
        ):
            view_start += 1
        visible_buffer = buffer[view_start:]
        cursor_x = prompt_width + self._display_width(buffer[view_start:cursor_index])
        return self._clip_to_width(prompt + visible_buffer, width), cursor_x, view_start

    def _cursor_index_from_input_x(
        self,
        prompt: str,
        buffer: str,
        view_start: int,
        x: int,
    ) -> int:
        prompt_width = self._display_width(prompt)
        target = int(x) - prompt_width
        if target <= 0:
            return max(0, min(view_start, len(buffer)))
        return self._buffer_index_from_display_col(buffer, view_start, target)

    def _buffer_index_from_display_col(
        self,
        buffer: str,
        view_start: int,
        display_col: int,
    ) -> int:
        current = 0
        start = max(0, min(int(view_start), len(buffer)))
        target = max(int(display_col), 0)
        for index in range(start, len(buffer)):
            char_width = max(self._char_width(buffer[index]), 1)
            if target <= current:
                return index
            if target < current + char_width:
                halfway = current + (char_width / 2.0)
                return index if target < halfway else index + 1
            current += char_width
        return len(buffer)

    def _insert_text(self, buffer: str, cursor_index: int, text: str) -> Tuple[str, int]:
        cursor = max(0, min(int(cursor_index), len(buffer)))
        return buffer[:cursor] + text + buffer[cursor:], cursor + len(text)

    def _backspace_text(self, buffer: str, cursor_index: int) -> Tuple[str, int]:
        cursor = max(0, min(int(cursor_index), len(buffer)))
        if cursor <= 0:
            return buffer, cursor
        return buffer[: cursor - 1] + buffer[cursor:], cursor - 1

    def _delete_text(self, buffer: str, cursor_index: int) -> Tuple[str, int]:
        cursor = max(0, min(int(cursor_index), len(buffer)))
        if cursor >= len(buffer):
            return buffer, cursor
        return buffer[:cursor] + buffer[cursor + 1 :], cursor

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
        if self.mouse_mode == "selection":
            return
        event = self._read_mouse_event()
        if event is None:
            return
        x, y, state = event
        self._handle_mouse_event(x, y, state)

    def _read_mouse_event(self) -> Optional[Tuple[int, int, int]]:
        if curses is None:
            return None
        try:
            _, x, y, _, state = curses.getmouse()
        except curses.error:
            return None
        return int(x), int(y), int(state)

    def _handle_mouse_event(self, x: int, y: int, state: int) -> None:
        if self.mouse_mode == "selection":
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

    def _mouse_targets_input(self, y: int, state: int) -> bool:
        if self.stdscr is None:
            return False
        height, _ = self.stdscr.getmaxyx()
        return y == height - 1 and bool(state & self._mouse_button_event_mask(1))

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

    def _interactive_mouse_mask(self) -> int:
        if curses is None:
            return 0
        return int(curses.ALL_MOUSE_EVENTS | getattr(curses, "REPORT_MOUSE_POSITION", 0))

    def _set_mouse_mode(self, mode: str) -> None:
        if mode not in {"interactive", "selection"}:
            raise ValueError(f"unsupported mouse mode: {mode}")
        self.mouse_mode = mode
        if curses is None:
            return
        try:
            curses.mousemask(0 if mode == "selection" else self._interactive_mouse_mask())
        except curses.error:
            pass

    def _toggle_mouse_mode(self) -> str:
        mode = "selection" if self.mouse_mode == "interactive" else "interactive"
        self._set_mouse_mode(mode)
        self._append_output(self._mouse_mode_message(mode))
        return mode

    def _mouse_mode_message(self, mode: Optional[str] = None) -> str:
        mode = mode or self.mouse_mode
        if mode == "selection":
            return self._t(
                "tui.mouse.selection",
                "mouse mode: selection. Terminal-native drag selection is enabled; press F2 or run /mouse interactive to restore TUI mouse controls.",
            )
        return self._t(
            "tui.mouse.interactive",
            "mouse mode: interactive. TUI handles wheel scrolling and input clicks; press F2 or run /mouse select for terminal-native selection.",
        )

    def _handle_tui_command(self, line: str) -> bool:
        parts = line.strip().split()
        if not parts or parts[0] != "/mouse":
            return False
        action = parts[1] if len(parts) > 1 else "status"
        if action == "status":
            self._append_output(self._mouse_mode_message())
        elif action == "select":
            self._set_mouse_mode("selection")
            self._append_output(self._mouse_mode_message("selection"))
        elif action == "interactive":
            self._set_mouse_mode("interactive")
            self._append_output(self._mouse_mode_message("interactive"))
        elif action == "toggle":
            self._toggle_mouse_mode()
        else:
            self._append_output(
                self._t(
                    "tui.mouse.help",
                    "usage: /mouse status|select|interactive|toggle",
                )
            )
        return True

    def _is_f2_key(self, key: object) -> bool:
        if curses is None:
            return False
        candidates = {getattr(curses, "KEY_F2", None)}
        try:
            candidates.add(curses.KEY_F(2))
        except Exception:
            pass
        return isinstance(key, int) and key in {value for value in candidates if value is not None}

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
        return prompt.endswith(self._main_prompt_suffix)

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

    def _reset_history_browse(self) -> None:
        self.history_cursor = None
        self.history_draft = ""

    def _set_wait_status(self, text: str) -> None:
        self.wait_status = text
        if self.stdscr is not None:
            self._draw()

    def _t(self, key: str, default: str, **kwargs) -> str:
        if self.translator is None:
            try:
                return default.format(**kwargs)
            except (KeyError, ValueError):
                return default
        return self.translator.t(key, default=default, **kwargs)
