"""Stdlib curses fullscreen interface for HyperAgent."""

import json
import locale
import os
import re
from dataclasses import dataclass
from pathlib import Path
import unicodedata
from typing import Dict, List, Optional, Tuple

from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.background_jobs import BackgroundJobStore
from hyperagent.runtime.extensions import PluginBundleStore
from hyperagent.runtime.feature_state import (
    IDEContextStore,
    PlanModeStore,
    image_status,
    web_status,
)
from hyperagent.runtime.i18n import Translator
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.mcp import MCPServerStore
from hyperagent.runtime.permissions import RememberedPermissionStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.slash_registry import command_names, public_commands
from hyperagent.runtime.subagents import SubagentRuntimeRegistry
from hyperagent.runtime.wait_indicator import WaitIndicator
from hyperagent.runtime.workspace import HyperAgentWorkspace

try:  # pragma: no cover - import availability depends on platform.
    import curses
except Exception:  # pragma: no cover
    curses = None


@dataclass
class TuiLine:
    kind: str
    text: str
    show_label: bool = True

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, TuiLine):
            return self.kind == other.kind and self.text == other.text
        return False


@dataclass(frozen=True)
class CommandPaletteEntry:
    command: str
    title: str
    category: str
    description: str = ""
    source: str = "builtin"


def build_command_palette_entries(
    repl: Optional[object] = None,
    workspace: Optional[HyperAgentWorkspace] = None,
) -> List[CommandPaletteEntry]:
    entries: List[CommandPaletteEntry] = []
    seen = set()

    def add(entry: CommandPaletteEntry) -> None:
        if entry.command in seen:
            return
        seen.add(entry.command)
        entries.append(entry)

    for command in public_commands():
        add(
            CommandPaletteEntry(
                command="/" + command.name,
                title="/" + command.name,
                category=command.category,
                description=command.description,
                source="builtin",
            )
        )
    command_store = getattr(repl, "command_store", None)
    if command_store is not None:
        try:
            for command in command_store.discover():
                add(
                    CommandPaletteEntry(
                        command="/" + command.name,
                        title="/" + command.name,
                        category="Markdown",
                        description=getattr(command, "description", ""),
                        source=getattr(command, "source", "markdown"),
                    )
                )
        except Exception:
            pass
    try:
        skills = repl.skill_names() if repl is not None and hasattr(repl, "skill_names") else []
    except Exception:
        skills = []
    for skill in skills:
        add(
            CommandPaletteEntry(
                command="/skill " + skill,
                title=skill,
                category="Skills",
                description="Run skill",
                source="skill",
            )
        )
    workspace_obj = workspace or getattr(repl, "workspace", None)
    if workspace_obj is not None:
        try:
            for bundle in PluginBundleStore(
                workspace_obj.workspace_dir,
                workspace_obj.project_root,
            ).list():
                add(
                    CommandPaletteEntry(
                        command="/plugin bundles " + bundle.id,
                        title=bundle.name,
                        category="Plugin Bundles",
                        description=bundle.description,
                        source=bundle.source,
                    )
                )
        except Exception:
            pass
    return entries


def filter_command_palette(
    entries: List[CommandPaletteEntry],
    query: str,
    *,
    limit: int = 12,
) -> List[CommandPaletteEntry]:
    needle = str(query or "").strip().lower().lstrip("/")
    if not needle:
        return entries[: max(limit, 0)]
    terms = [term for term in needle.split() if term]
    scored = []
    for index, entry in enumerate(entries):
        haystack = " ".join(
            [entry.command, entry.title, entry.category, entry.description, entry.source]
        ).lower()
        if not all(term in haystack for term in terms):
            continue
        command = entry.command.lower().lstrip("/")
        title = entry.title.lower()
        score = 0
        if command.startswith(needle):
            score += 50
        if title.startswith(needle):
            score += 30
        score += sum(haystack.count(term) for term in terms)
        scored.append((-score, index, entry))
    return [entry for _, _, entry in sorted(scored)[: max(limit, 0)]]


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
        self.lines: List[TuiLine] = []
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
        self._main_prompt_suffix = "HyperAgent > "
        self.command_suggestions: List[str] = []
        self.locale_warning = ""
        self.color_enabled = False
        self.styles: Dict[str, int] = {}

    def run(self) -> int:
        if curses is None:
            print(
                self._t(
                    "tui.curses_missing",
                    "error: Python curses module is not available on this platform.",
                )
            )
            return 2
        self.locale_warning = self._init_locale()
        return curses.wrapper(self._run)

    def _init_locale(self) -> str:
        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error as exc:
            return self._t(
                "tui.locale.set_failed",
                "locale initialization failed: {error}",
                error=str(exc),
            )
        encoding = locale.getpreferredencoding(False)
        if "UTF" not in encoding.upper():
            return self._t(
                "tui.locale.non_utf8",
                "terminal locale is not UTF-8: {encoding}",
                encoding=encoding,
            )
        return ""

    def _init_colors(self) -> None:
        self.color_enabled = False
        self.styles = {}
        if curses is None:
            return
        try:
            if not curses.has_colors():
                self.styles = self._fallback_styles()
                return
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = curses.COLOR_BLACK
            pairs = {
                "user": (1, curses.COLOR_CYAN, background),
                "command": (2, curses.COLOR_BLUE, background),
                "assistant": (3, curses.COLOR_GREEN, background),
                "reasoning": (4, curses.COLOR_YELLOW, background),
                "tool": (5, curses.COLOR_MAGENTA, background),
                "system": (6, curses.COLOR_WHITE, background),
                "warning": (7, curses.COLOR_YELLOW, background),
                "error": (8, curses.COLOR_RED, background),
                "panel": (9, curses.COLOR_WHITE, background),
            }
            for _, (pair_id, fg, bg) in pairs.items():
                curses.init_pair(pair_id, fg, bg)
            self.styles = {
                "user": curses.color_pair(1) | curses.A_BOLD,
                "command": curses.color_pair(2) | curses.A_BOLD,
                "assistant": curses.color_pair(3),
                "reasoning": curses.color_pair(4),
                "tool": curses.color_pair(5) | curses.A_BOLD,
                "system": curses.color_pair(6),
                "warning": curses.color_pair(7) | curses.A_BOLD,
                "error": curses.color_pair(8) | curses.A_BOLD,
                "panel": curses.color_pair(9),
            }
            self.color_enabled = True
        except curses.error:
            self.styles = self._fallback_styles()

    def _fallback_styles(self) -> Dict[str, int]:
        if curses is None:
            return {}
        return {
            "user": curses.A_BOLD,
            "command": curses.A_BOLD,
            "assistant": 0,
            "reasoning": curses.A_BOLD,
            "tool": curses.A_BOLD,
            "system": 0,
            "warning": curses.A_BOLD,
            "error": curses.A_REVERSE | curses.A_BOLD,
            "panel": 0,
        }

    def _style_for_kind(self, kind: str) -> int:
        if not self.styles:
            self.styles = self._fallback_styles()
        return self.styles.get(str(kind), self.styles.get("system", 0))

    def _role_label(self, kind: str) -> str:
        normalized = str(kind or "system").strip().lower()
        defaults = {
            "user": "You",
            "command": "Command",
            "assistant": "Assistant",
            "reasoning": "Reasoning",
            "tool": "Tool",
            "system": "System",
            "warning": "Warning",
            "error": "Error",
        }
        if normalized == "panel":
            return ""
        return self._t(f"tui.role.{normalized}", defaults.get(normalized, "System"))

    def _run(self, stdscr) -> int:
        self.stdscr = stdscr
        curses.curs_set(1)
        stdscr.keypad(True)
        self._init_colors()
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
            output_event_func=self._append_output_event,
            wait_indicator_factory=lambda: TuiWaitIndicator(self),
        )
        self._append_output(self.repl._banner(), kind="system")
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

    def _append_output(self, text: str, kind: str = "system") -> None:
        self._append_output_event(kind, text)

    def _append_output_event(self, kind: str, text: str) -> None:
        event_lines = self._display_lines_for_event(kind, text)
        if not event_lines:
            return
        show_first_label = self._should_show_role_label(kind)
        for index, line in enumerate(event_lines):
            self.lines.append(TuiLine(kind, line, show_label=show_first_label and index == 0))
        self.lines = self.lines[-1000:]
        if self.stdscr is not None:
            self._draw()

    def _display_lines_for_event(self, kind: str, text: str) -> List[str]:
        display_text = str(text).replace("\r\n", "\n").replace("\r", "\n")
        if kind in {"assistant", "reasoning"}:
            display_text = self._strip_display_markdown(display_text)
        lines = display_text.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        collapsed: List[str] = []
        previous_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            collapsed.append(line)
            previous_blank = is_blank
        return collapsed or ([""] if kind in {"user", "command", "warning", "error"} else [])

    def _should_show_role_label(self, kind: str) -> bool:
        normalized = str(kind or "system").strip().lower()
        if normalized in {"user", "command", "warning", "error"}:
            return True
        for line in reversed(self.lines):
            if not line.text.strip():
                continue
            return line.kind != normalized
        return True

    def _strip_display_markdown(self, text: str) -> str:
        # TUI is not a Markdown renderer. Strip common inline emphasis so narrow
        # terminals do not break **skill** into unreadable star fragments.
        cleaned = re.sub(r"(?<!\*)\*\*([^*\n]+?)\*\*(?!\*)", r"\1", str(text))
        cleaned = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", cleaned)
        return cleaned

    def _prompt_dialog(self, prompt: str) -> str:
        self._append_output(prompt, kind="warning" if "?" in prompt else "system")
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
                kind = "command" if buffer.strip().startswith("/") else "user"
                self._append_output(buffer, kind=kind)
                self._record_history(prompt, buffer)
                return buffer
            if key in ("\x1b",):
                return "/exit"
            if key == "\x01":  # Ctrl-A
                cursor_index = 0
                continue
            if key == "\x05":  # Ctrl-E
                cursor_index = len(buffer)
                continue
            if key == "\x0b":  # Ctrl-K
                buffer = buffer[:cursor_index]
                self._reset_history_browse()
                continue
            if key == "\x15":  # Ctrl-U
                buffer = ""
                cursor_index = 0
                self._reset_history_browse()
                continue
            if key == "\x17":  # Ctrl-W
                buffer, cursor_index = self._delete_previous_word(buffer, cursor_index)
                self._reset_history_browse()
                continue
            if key == "\x10":  # Ctrl-P
                selected = self._command_palette_dialog(buffer)
                if selected:
                    buffer = selected
                    cursor_index = len(buffer)
                    self._reset_history_browse()
                continue
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
        return f"{Path.cwd()} {self._main_prompt_suffix}"

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
        if prompt.endswith(self._main_prompt_suffix):
            cwd_text = prompt[: -len(self._main_prompt_suffix)].rstrip()
            suffix = self._main_prompt_suffix
            cwd_budget = prompt_budget - self._display_width(" " + suffix)
            if cwd_budget > 0:
                candidate = f"{self._left_ellipsis(cwd_text, cwd_budget)} {suffix}"
                if self._display_width(candidate) <= prompt_budget:
                    return candidate
            for fallback in (suffix, "HyperAgent> ", "> ", ">"):
                if self._display_width(fallback) <= prompt_budget:
                    return fallback
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

    def _draw(self, prompt_line: str = "HyperAgent > ", cursor_x: Optional[int] = None) -> None:
        assert self.stdscr is not None
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        side_width = max(28, min(44, width // 3)) if width >= 90 else 0
        main_width = width - side_width - (1 if side_width else 0)
        self._addstr(
            0,
            0,
            self._build_status_line(width),
            curses.A_REVERSE,
        )
        log_height = max(1, height - 3)
        main_content_width = max(main_width - 1, 1)
        wrapped_main = self._wrap_lines(
            self.lines,
            main_content_width,
            role_labels=True,
            default_kind="system",
        )
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
            self._addstr(
                index,
                0,
                self._clip_to_width(line.text, main_content_width),
                self._style_for_kind(line.kind),
            )
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
            panel_lines = self._wrap_lines(
                self._panel_lines(),
                panel_width,
                role_labels=False,
                default_kind="panel",
            )
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
                self._addstr(
                    offset,
                    panel_x,
                    self._clip_to_width(line.text, panel_width),
                    self._style_for_kind(line.kind),
                )
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
            self._panel_kv("session", "session", session or ""),
            self._panel_kv("mode", "mode", self.mode),
            self._panel_kv(
                "model",
                "model",
                self.model or self._t("tui.value.profile_default", "profile/default"),
            ),
            self._panel_kv("permission", "permission", self.permission_policy),
            self._panel_kv("reasoning", "reasoning", self._reasoning_label(reasoning)),
            self._panel_kv("mouse", "mouse", self._mouse_label(self.mouse_mode)),
        ]
        if self.repl is not None:
            try:
                usage = self.repl.usage.summarize()
                context = self._context_status()
                ide = IDEContextStore(self.workspace.workspace_dir).load()
                plan_mode = PlanModeStore(self.workspace.workspace_dir).load()
                web = web_status()
                image = image_status()
                permission_counts = self._permission_counts()
                lines.extend(
                    [
                        self._panel_kv("llm_requests", "llm_requests", usage["request_count"]),
                        self._panel_kv("tokens", "tokens", usage["total_tokens"]),
                        self._panel_kv("cache_hit", "cache_hit", usage["cache_hit_ratio"]),
                        self._panel_kv("context", "context", self._format_context_meter(context)),
                        self._panel_kv("summaries", "summaries", context.get("summary_count", 0)),
                        self._panel_kv("session_grants", "session grants", permission_counts["session_grants"]),
                        self._panel_kv("remembered_grants", "remembered grants", permission_counts["remembered_grants"]),
                        self._panel_kv("loop", "loop", self.repl.action_loop_mode),
                        self._panel_kv("budget", "budget", self.repl.action_token_budget),
                        self._panel_kv("ide_context", "IDE context", self._on_off(ide.get("enabled"))),
                        self._panel_kv("plan_mode", "plan mode", self._on_off(plan_mode.get("enabled"))),
                        self._panel_kv("web", "web", self._t("tui.value.configured", "configured") if web["search_configured"] else self._t("tui.value.fetch_only", "fetch-only")),
                        self._panel_kv("mcp", "MCP", len(MCPServerStore(self.workspace.workspace_dir).list())),
                        self._panel_kv("image", "image", self._t("tui.value.configured", "configured") if image["configured"] else self._t("tui.value.missing_key", "missing key")),
                    ]
                )
                lines.extend(["", self._t("tui.panel.permissions", "permissions:")])
                lines.extend(self._permission_detail_lines())
            except Exception:
                pass
        lines.extend(["", self._t("tui.panel.subagents", "active subagents:")])
        lines.extend(self._subagent_panel_lines())
        lines.extend(["", self._t("tui.panel.jobs", "background jobs:")])
        lines.extend(self._job_panel_lines())
        lines.extend(["", self._t("tui.panel.commands", "command suggestions:")])
        lines.extend(self.command_suggestions[:6] or [self._t("tui.panel.none", "none")])
        lines.append("")
        lines.append(self._t("tui.panel.todos", "todos:"))
        lines.extend(self._todo_panel_lines(session or "project"))
        lines.extend([
            "",
            self._t("tui.panel.recent_artifacts", "recent artifacts:"),
        ])
        artifact_markers = (
            "artifact:",
            "agent_run:",
            "action_run:",
            "工件:",
            "智能体运行:",
            "动作运行:",
        )
        artifact_lines = [
            line
            for line in self.lines[-80:]
            if line.kind == "tool" or any(marker in line.text for marker in artifact_markers)
        ]
        lines.extend([line.text for line in artifact_lines[-10:]] or [self._t("tui.panel.none", "none")])
        return lines

    def _build_status_line(self, width: int) -> str:
        reasoning = "collapsed"
        if self.repl is not None:
            reasoning = self.repl._reasoning_display_mode()
        usage = self._usage_status()
        context = self._context_status()
        model = self.model or self._t("tui.value.profile_default", "profile/default")
        session = self._short_session_id()
        parts = [
            " HyperAgent TUI",
            f"{self._t('tui.status.provider', 'provider')}={self.provider}",
            f"{self._t('tui.status.model', 'model')}={model}",
            f"{self._t('tui.status.session', 'session')}={session}",
            f"{self._t('tui.status.permission', 'permission')}={self.permission_policy}",
            f"{self._t('tui.status.context', 'context')}={self._format_context_meter(context)}",
            f"{self._t('tui.status.tokens', 'tokens')}={usage.get('total_tokens', 0)}",
            f"{self._t('tui.status.cache', 'cache')}={self._format_cache_ratio(usage.get('cache_hit_ratio'))}",
            f"{self._t('tui.status.reasoning', 'reasoning')}={self._reasoning_label(reasoning)}",
            f"{self._t('tui.status.mouse', 'mouse')}={self._mouse_label(self.mouse_mode)}",
            "/help /exit",
        ]
        if self.wait_status:
            parts.append(self.wait_status)
        if self.locale_warning:
            parts.append(self.locale_warning)
        return self._clip_to_width(" | ".join(parts) + " ", max(width - 1, 0))

    def _panel_kv(self, key: str, default: str, value: object) -> str:
        return f"{self._t(f'tui.panel.{key}', default)}: {value}"

    def _context_status(self) -> Dict[str, object]:
        session_id = self.repl.session_id if self.repl is not None else self.session_id
        if not session_id or self.conversations is None:
            return {
                "current_chars": 0,
                "max_chars": self.max_context_chars,
                "summary_count": 0,
            }
        try:
            status = self.conversations.context_status(
                session_id,
                max_chars=self.max_context_chars,
                keep_last=self.keep_last,
            )
            return {
                "current_chars": status.current_chars,
                "max_chars": status.max_chars,
                "summary_count": status.summary_count,
            }
        except Exception:
            return {
                "current_chars": 0,
                "max_chars": self.max_context_chars,
                "summary_count": 0,
            }

    def _usage_status(self) -> Dict[str, object]:
        if self.repl is None:
            return {"total_tokens": 0, "cache_hit_ratio": None}
        try:
            return self.repl.usage.summarize()
        except Exception:
            return {"total_tokens": 0, "cache_hit_ratio": None}

    def _permission_counts(self) -> Dict[str, int]:
        session_grants = len(getattr(self.repl, "permission_cache", {}) or {})
        store = getattr(self.repl, "remembered_permissions", None)
        if store is None and self.workspace is not None:
            store = RememberedPermissionStore(self.workspace.workspace_dir)
        remembered_grants = 0
        if store is not None:
            try:
                remembered_grants = int(store.summary().get("count", 0))
            except Exception:
                remembered_grants = 0
        return {
            "session_grants": session_grants,
            "remembered_grants": remembered_grants,
        }

    def _permission_detail_lines(self) -> List[str]:
        if self.repl is None:
            return [self._t("tui.panel.none", "none")]
        lines: List[str] = []
        permission_cache = getattr(self.repl, "permission_cache", {}) or {}
        for key, allowed in sorted(permission_cache.items())[-3:]:
            lines.append(f"- session {key}={allowed}")
        store = getattr(self.repl, "remembered_permissions", None)
        if store is None and self.workspace is not None:
            store = RememberedPermissionStore(self.workspace.workspace_dir)
        if store is not None:
            try:
                for rule in store.list_rules()[-3:]:
                    lines.append(
                        f"- remembered {rule.tool_name} {rule.risk_level} "
                        f"fp={rule.args_fingerprint} uses={rule.uses}"
                    )
            except Exception:
                pass
        return lines or [self._t("tui.panel.none", "none")]

    def _short_session_id(self) -> str:
        session = self.repl.session_id if self.repl is not None else self.session_id
        if not session:
            return "-"
        text = str(session)
        return text[:10]

    def _format_context_meter(self, context: Dict[str, object]) -> str:
        current = int(context.get("current_chars") or 0)
        maximum = int(context.get("max_chars") or self.max_context_chars or 0)
        return f"{current}/{maximum}"

    def _format_cache_ratio(self, value: object) -> str:
        if value in {None, ""}:
            return "n/a"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    def _reasoning_label(self, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized == "expanded":
            return self._t("tui.value.expanded", "expanded")
        return self._t("tui.value.collapsed", "collapsed")

    def _mouse_label(self, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized == "selection":
            return self._t("tui.value.selection", "selection")
        return self._t("tui.value.interactive", "interactive")

    def _on_off(self, value: object) -> str:
        return self._t("tui.value.on", "on") if value else self._t("tui.value.off", "off")

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

    def _subagent_panel_lines(self) -> List[str]:
        try:
            registry = SubagentRuntimeRegistry(self.workspace.workspace_dir)
            active = registry.list(include_completed=False)
        except Exception:
            return [self._t("tui.panel.none", "none")]
        if not active:
            return [self._t("tui.panel.none", "none")]
        return [
            f"- d{item.depth} {item.status}: {item.agent_name} {item.subagent_id}"
            for item in active[:6]
        ]

    def _job_panel_lines(self) -> List[str]:
        try:
            jobs = BackgroundJobStore(self.workspace.workspace_dir).list()
        except Exception:
            return [self._t("tui.panel.none", "none")]
        if not jobs:
            return [self._t("tui.panel.none", "none")]
        return [
            f"- {job.status}: {job.kind} {job.job_id}"
            for job in jobs[-6:]
        ]

    def _suggest_commands(self, buffer: str) -> List[str]:
        if self.repl is None:
            return []
        text = buffer.strip()
        if not text.startswith("/"):
            return []
        builtins = command_names(include_aliases=True) + ["mouse", "act", "plan"]
        custom = [command.name for command in self.repl.command_store.discover()]
        skills = self.repl.skill_names() if hasattr(self.repl, "skill_names") else []
        body = text[1:].strip()
        if not body:
            common = ["help", "status", "agents", "commands", "skills", "todos", "doctor", "exit"]
            available = set(builtins + custom)
            return ["/" + name for name in common if name in available][:8]
        query = body.split()[0].lower()
        command_matches = sorted({name for name in builtins + custom if name.startswith(query)})
        skill_matches = sorted({name for name in skills if name.startswith(query)})
        suggestions = ["/" + name for name in command_matches]
        suggestions.extend("/skill " + name for name in skill_matches)
        return suggestions[:8]

    def _command_palette_entries(self) -> List[CommandPaletteEntry]:
        return build_command_palette_entries(self.repl, self.workspace)

    def _command_palette_dialog(self, current_buffer: str = "") -> Optional[str]:
        if self.stdscr is None:
            return None
        query = current_buffer.strip()
        if query.startswith("/"):
            query = query[1:]
        selected = 0
        while True:
            entries = filter_command_palette(self._command_palette_entries(), query, limit=8)
            if selected >= len(entries):
                selected = max(0, len(entries) - 1)
            self.command_suggestions = [
                ("› " if index == selected else "  ") + entry.command
                for index, entry in enumerate(entries)
            ] or [self._t("tui.panel.none", "none")]
            self._draw("palette> " + query)
            key = self.stdscr.get_wch()
            if key in ("\n", "\r"):
                return entries[selected].command if entries else None
            if key in ("\x1b",):
                return None
            if key in ("\b", "\x7f") or key == curses.KEY_BACKSPACE:
                query = query[:-1]
                continue
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
                continue
            if key == curses.KEY_DOWN:
                selected = min(max(len(entries) - 1, 0), selected + 1)
                continue
            if isinstance(key, str) and key.isprintable():
                query += key

    def _addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        assert self.stdscr is not None
        row = int(y)
        col = max(int(x), 0)
        try:
            height, width = self.stdscr.getmaxyx()
        except Exception:
            height, width = row + 1, col + self._display_width(text) + 1
        if row < 0 or row >= height or col >= width:
            return
        safe_text = self._clip_to_width(str(text).expandtabs(4), max(width - col, 0))
        if not safe_text:
            return
        try:
            self.stdscr.addstr(row, col, safe_text, attr)
            return
        except curses.error:
            pass
        # Some curses implementations raise when a string reaches the right edge.
        # Retry with progressively shorter whole strings instead of writing CJK
        # characters one by one; per-character wide writes are unreliable in
        # several terminal/ncurses combinations and can render as blanks.
        while safe_text:
            safe_text = safe_text[:-1]
            if not safe_text:
                return
            try:
                self.stdscr.addstr(row, col, safe_text, attr)
                return
            except curses.error:
                continue

    def _wrap_lines(
        self,
        lines: List[object],
        width: int,
        *,
        role_labels: bool = False,
        default_kind: str = "system",
    ) -> List[TuiLine]:
        wrapped: List[TuiLine] = []
        for line in lines:
            item = line if isinstance(line, TuiLine) else TuiLine(default_kind, str(line))
            if role_labels:
                wrapped.extend(self._wrap_semantic_line(item, width))
            else:
                wrapped.extend(TuiLine(item.kind, part) for part in self._wrap_line(item.text, width))
        return wrapped

    def _wrap_semantic_line(self, line: TuiLine, width: int) -> List[TuiLine]:
        width = max(int(width), 1)
        label = self._role_label(line.kind)
        label_prefix = f"{label} │ " if label else ""
        label_width = self._display_width(label_prefix)
        prefix = label_prefix if line.show_label else " " * label_width
        prefix_width = self._display_width(prefix)
        if prefix_width >= width:
            prefix = self._clip_to_width(prefix, max(width - 1, 1))
            prefix_width = self._display_width(prefix)
        content_width = max(width - prefix_width, 1)
        parts = self._wrap_line(line.text, content_width)
        continuation = " " * prefix_width
        wrapped: List[TuiLine] = []
        for index, part in enumerate(parts):
            wrapped.append(
                TuiLine(
                    line.kind,
                    (prefix if index == 0 else continuation) + part,
                    show_label=line.show_label and index == 0,
                )
            )
        return wrapped or [TuiLine(line.kind, prefix, show_label=line.show_label)]

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

    def _delete_previous_word(self, buffer: str, cursor_index: int) -> Tuple[str, int]:
        cursor = max(0, min(int(cursor_index), len(buffer)))
        if cursor <= 0:
            return buffer, 0
        index = cursor
        while index > 0 and buffer[index - 1].isspace():
            index -= 1
        while index > 0 and not buffer[index - 1].isspace():
            index -= 1
        return buffer[:index] + buffer[cursor:], index

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
