"""Central slash command registry shared by launcher, REPL, TUI, and channels."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SlashCommandDef:
    name: str
    description: str
    category: str
    aliases: Tuple[str, ...] = ()
    args_hint: str = ""
    cli_command: str = ""
    gateway_allowed: bool = False
    hidden: bool = False

    def usage(self) -> str:
        return f"/{self.name} {self.args_hint}".rstrip()


COMMAND_REGISTRY: List[SlashCommandDef] = [
    SlashCommandDef("help", "Show available commands", "Info", cli_command="hyperagent-commands", gateway_allowed=True),
    SlashCommandDef("status", "Show workspace and session status", "Info", cli_command="status", gateway_allowed=True),
    SlashCommandDef("sessions", "List or search saved conversation sessions", "Session", args_hint="[search <query>]", cli_command="session-list"),
    SlashCommandDef("session-search", "Search saved conversation sessions", "Session", args_hint="<query>", cli_command="session-search"),
    SlashCommandDef("resume", "Resume a saved session", "Session", args_hint="<session_id> [message]"),
    SlashCommandDef("compact", "Compress a saved conversation", "Session", aliases=("compress",), args_hint="<session_id>"),
    SlashCommandDef("new", "Create a new REPL/TUI session", "Session", args_hint="[title]"),
    SlashCommandDef("undo", "Remove the last user/assistant exchange", "Session"),
    SlashCommandDef("branch", "Branch the current conversation", "Session", aliases=("fork",), args_hint="[title]"),
    SlashCommandDef("queue", "Queue a prompt for the next turn", "Session", aliases=("q",), args_hint="<prompt>"),
    SlashCommandDef("steer", "Inject a steering note for the next action", "Session", args_hint="<prompt>"),
    SlashCommandDef("goal", "Set or inspect a standing research goal", "Session", args_hint="[text|pause|resume|clear|status]"),
    SlashCommandDef("subgoal", "Manage extra criteria for the active goal", "Session", args_hint="[text|remove N|clear]"),
    SlashCommandDef("background", "Run a prompt as a background job", "Session", aliases=("bg", "btw"), args_hint="<prompt>"),
    SlashCommandDef("agents", "List, run, pause, resume, or stop subagents", "Agents", aliases=("tasks",), args_hint="[list|run|status|pause|resume|stop]"),
    SlashCommandDef("tools", "Show available local tools", "Tools"),
    SlashCommandDef("tool", "Run a local tool manually", "Tools", args_hint="<name> <json>"),
    SlashCommandDef("commands", "List or render Markdown slash commands", "Tools", aliases=("command",), cli_command="command-list"),
    SlashCommandDef("todos", "Show or update TodoWrite state", "Tools", cli_command="todos"),
    SlashCommandDef("hooks", "List or manage runtime hooks", "Tools"),
    SlashCommandDef("permissions", "Show or manage session and remembered permissions", "Safety", args_hint="[list|show <id|key>|forget <id|key>|clear]"),
    SlashCommandDef("rollback", "Alias for checkpoint restore/list", "Safety", cli_command="checkpoint"),
    SlashCommandDef("snapshot", "Create a file checkpoint", "Safety", cli_command="checkpoint"),
    SlashCommandDef("checkpoint", "List or create file checkpoints", "Safety", cli_command="checkpoint"),
    SlashCommandDef("restore", "Restore files from a checkpoint", "Safety", cli_command="restore"),
    SlashCommandDef("model", "Show configured LLM providers", "LLM", cli_command="llm-providers"),
    SlashCommandDef("reasonix", "Show DeepSeek/Reasonix profiles", "LLM", aliases=("deepseek",), cli_command="llm-profile"),
    SlashCommandDef("usage", "Show token usage", "LLM", cli_command="llm-usage", gateway_allowed=True),
    SlashCommandDef("cost", "Show token/cost ledger", "LLM", cli_command="llm-usage"),
    SlashCommandDef("web", "Search or fetch the web through controlled tools", "Tools", args_hint="<status|search|fetch|cite>", cli_command="web"),
    SlashCommandDef("image", "Generate or edit image request artifacts", "Tools", args_hint="<status|generate|edit>", cli_command="image"),
    SlashCommandDef("ide-context", "Show or manage manually supplied IDE context", "Info", args_hint="<status|on|off|set-open-files|clear>", cli_command="ide-context"),
    SlashCommandDef("personality", "Show or update the local interaction personality note", "Info", args_hint="<status|set|clear>", cli_command="personality"),
    SlashCommandDef("feedback", "Record or list local feedback notes", "Info", args_hint="<add|list>", cli_command="feedback"),
    SlashCommandDef("plan-mode", "Toggle plan-only mode that blocks tool execution", "Safety", args_hint="<status|on|off>", cli_command="plan-mode"),
    SlashCommandDef("worktree", "Show git worktree status without modifying files", "Safety", cli_command="worktree"),
    SlashCommandDef("preset", "Select action-loop preset", "LLM", args_hint="<flash|auto|pro|standard>"),
    SlashCommandDef("pro", "Use stronger model/profile for one turn", "LLM"),
    SlashCommandDef("budget", "Set or inspect token budget", "LLM", args_hint="[tokens|clear]"),
    SlashCommandDef("thinking", "Expand or collapse returned reasoning_content", "LLM", args_hint="[on|off|toggle|status]"),
    SlashCommandDef("skills", "List, browse, inspect, install, run, or inspect usage for skills", "Skills", aliases=("skill",), cli_command="skill-list"),
    SlashCommandDef("bundles", "List skill bundles", "Skills", cli_command="skill-bundles"),
    SlashCommandDef("skill-usage", "Summarize skill usage telemetry and curator hints", "Skills", cli_command="skill-usage"),
    SlashCommandDef("mcp", "List MCP server specs", "Integrations", cli_command="mcp-list"),
    SlashCommandDef("plugin", "List or manage project plugins and bundles", "Integrations", aliases=("plugins",), args_hint="[list|add|bundles]"),
    SlashCommandDef("plugin-bundles", "List local plugin bundle manifests", "Integrations", cli_command="plugin-bundles"),
    SlashCommandDef("browser", "Show browser-tool integration status", "Integrations"),
    SlashCommandDef("platforms", "Show Hermes-style platform health", "Integrations", cli_command="platform-status", gateway_allowed=True),
    SlashCommandDef("channels", "List configured bot channels", "Integrations", aliases=("channel",), cli_command="channel-list"),
    SlashCommandDef("events", "List runtime event records", "Runtime", cli_command="events"),
    SlashCommandDef("replay", "Replay runtime event records", "Runtime", cli_command="replay"),
    SlashCommandDef("diff", "Show artifact diff", "Runtime", cli_command="diff"),
    SlashCommandDef("stats", "Show runtime stats", "Runtime", cli_command="stats"),
    SlashCommandDef("jobs", "Show or control background jobs", "Runtime"),
    SlashCommandDef("logs", "Show recent runtime/worklog paths", "Runtime"),
    SlashCommandDef("doctor", "Run local self-check", "Info", cli_command="doctor"),
    SlashCommandDef("export", "Export the current session", "Session"),
    SlashCommandDef("copy", "Export/copy last assistant response", "Session"),
    SlashCommandDef("hsi", "Run HSI-specific workflows", "HSI", args_hint="<audit|experiment-cycle|review-experiment|literature|materialize-module>"),
    SlashCommandDef("tui", "Start fullscreen TUI", "UI", cli_command="tui"),
    SlashCommandDef("repl", "Start interactive REPL", "UI", cli_command="repl"),
    SlashCommandDef("exit", "Exit interactive mode", "Exit", aliases=("quit",)),
]


def command_lookup() -> Dict[str, SlashCommandDef]:
    lookup: Dict[str, SlashCommandDef] = {}
    for command in COMMAND_REGISTRY:
        lookup[command.name] = command
        for alias in command.aliases:
            lookup[alias] = command
    return lookup


def resolve_command(name: str) -> Optional[SlashCommandDef]:
    return command_lookup().get(name.lower().lstrip("/"))


def public_commands() -> List[SlashCommandDef]:
    return [command for command in COMMAND_REGISTRY if not command.hidden]


def command_names(include_aliases: bool = False) -> List[str]:
    if include_aliases:
        return sorted(command_lookup())
    return sorted(command.name for command in public_commands())


def gateway_command_names() -> List[str]:
    return sorted(command.name for command in public_commands() if command.gateway_allowed)


def grouped_help(
    commands: Iterable[SlashCommandDef] = (),
    translator: Optional[Any] = None,
) -> str:
    selected = list(commands) or public_commands()
    by_category: Dict[str, List[SlashCommandDef]] = {}
    for command in selected:
        by_category.setdefault(command.category, []).append(command)
    lines: List[str] = []
    for category in sorted(by_category):
        category_label = _translate(
            translator,
            f"slash.category.{category}",
            category,
        )
        lines.append(f"{category_label}:")
        for command in sorted(by_category[category], key=lambda item: item.name):
            alias = ""
            if command.aliases:
                alias_label = _translate(translator, "slash.aliases", "aliases")
                alias = f" {alias_label}={','.join(command.aliases)}"
            hint_text = _translate(
                translator,
                f"slash.command.{command.name}.args_hint",
                command.args_hint,
            )
            description = _translate(
                translator,
                f"slash.command.{command.name}.description",
                command.description,
            )
            hint = f" {hint_text}" if hint_text else ""
            lines.append(f"  /{command.name}{hint} - {description}{alias}")
    return "\n".join(lines)


def _translate(translator: Optional[Any], key: str, default: str) -> str:
    if translator is None:
        return default
    return str(translator.t(key, default=default))
