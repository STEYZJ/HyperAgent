"""Claude-Code-like command aliases for the HyperAgent launcher."""

from typing import Iterable, List, Sequence, Tuple

from hyperagent.runtime.slash_registry import grouped_help


EXISTING_COMMANDS = {
    "audit",
    "plan",
    "run-baseline",
    "run-suite",
    "benchmark-list",
    "benchmark-matrix",
    "report",
    "demo",
    "init",
    "status",
    "task-create",
    "task-list",
    "task-show",
    "task-run",
    "literature",
    "auto-experiment",
    "tune-next",
    "experiment-cycle",
    "propose-module",
    "llm-providers",
    "llm-profile",
    "llm-usage",
    "web",
    "image",
    "ide-context",
    "plan-mode",
    "personality",
    "feedback",
    "worktree",
    "llm-dry-run",
    "llm-send",
    "agent-chat",
    "run",
    "repl",
    "agent-context",
    "agent-plan",
    "agent-act",
    "agent-run",
    "agent-status",
    "agent-pause",
    "agent-resume",
    "agent-stop",
    "agent-tool",
    "command-list",
    "command-render",
    "todos",
    "doctor",
    "events",
    "replay",
    "diff",
    "stats",
    "platform-status",
    "prune-sessions",
    "checkpoint",
    "restore",
    "tui",
    "session-new",
    "session-add",
    "session-list",
    "session-search",
    "session-show",
    "session-archive",
    "session-delete",
    "session-compress",
    "skill-list",
    "skill-search",
    "skill-inspect",
    "skill-install",
    "skill-bundles",
    "skill-usage",
    "skill-run",
    "mcp-add",
    "mcp-list",
    "mcp-export",
    "mcp-inspect",
    "mcp-health",
    "index",
    "obsidian-index",
    "obsidian-search",
    "prompt-list",
    "prompt-render",
    "materialize-module",
    "channel-init",
    "channel-list",
    "channel-run",
    "channel-test",
    "language-list",
    "language-set",
    "language-install",
    "language-export",
    "hyperagent-commands",
}

BOOLEAN_FLAGS = {
    "--json-output",
    "--no-auto-compress",
    "--with-literature",
    "--run-baseline",
    "--run-next",
    "--force",
    "--synthetic",
    "--json",
    "--include-archived",
    "--hard",
    "--dry-run",
    "--yes",
    "--list",
    "--all",
}


def normalize_hyperagent_args(argv: Sequence[str]) -> List[str]:
    """Translate `HyperAgent ...` shorthand into the canonical CLI argv."""

    global_options, args = _extract_global_options(list(argv))
    if not args:
        return global_options + ["repl"]

    first = args[0]
    if first in {"-h", "--help", "help"}:
        return global_options + ["hyperagent-commands"]
    if first == "plan" and not _contains_flag(args[1:], "--audit"):
        return global_options + _prompt_command("agent-plan", "--instruction", args[1:])
    if first in EXISTING_COMMANDS:
        return global_options + args
    if first.startswith("/") and len(first) > 1:
        return global_options + _normalize_slash_command(first[1:], args[1:])

    alias = first.lower()
    rest = args[1:]
    if alias in {"chat", "ask"}:
        return global_options + _prompt_command("agent-chat", "--message", rest)
    if alias == "research":
        return global_options + _prompt_command("agent-chat", "--message", rest, defaults=["--mode", "research"])
    if alias == "algorithm":
        return global_options + _prompt_command("agent-chat", "--message", rest, defaults=["--mode", "algorithm"])
    if alias in {"plan", "code"}:
        return global_options + _prompt_command("agent-plan", "--instruction", rest)
    if alias in {"act", "do"}:
        return global_options + _prompt_command("agent-act", "--message", rest)
    if alias == "send":
        return global_options + _prompt_command(
            "llm-send",
            "--user",
            rest,
            defaults=_default_provider(rest),
        )
    if alias == "dry":
        return global_options + _prompt_command(
            "llm-dry-run",
            "--user",
            rest,
            defaults=_default_provider(rest),
        )
    if alias == "sessions":
        return global_options + _sessions_command(rest)
    if alias == "resume":
        return global_options + _resume_command(rest)
    if alias == "compact":
        return global_options + _compact_command(rest)
    if alias == "mcp":
        return global_options + _mcp_command(rest)
    if alias in {"skills", "skill"}:
        return global_options + _skills_command(rest)
    if alias in {"prompts", "prompt"}:
        return global_options + ["prompt-list"] + rest
    if alias == "model":
        return global_options + ["llm-providers"] + rest
    if alias in {"reasonix", "deepseek"}:
        return global_options + ["llm-profile"] + rest
    if alias == "usage":
        return global_options + ["llm-usage"] + rest
    if alias == "cost":
        return global_options + ["llm-usage"] + rest
    if alias in {"web", "image", "ide-context", "plan-mode", "personality", "feedback", "worktree"}:
        return global_options + [alias] + rest
    if alias in {"events", "replay", "diff", "stats", "checkpoint", "restore", "index"}:
        return global_options + [alias] + rest
    if alias in {"commands", "command"}:
        return global_options + ["command-list"] + rest
    if alias == "todos":
        return global_options + ["todos"] + rest
    if alias == "doctor":
        return global_options + ["doctor"] + rest
    if alias == "language":
        return global_options + ["language-list"] + rest
    if alias in {"channels", "channel"}:
        return global_options + ["channel-list"] + rest
    if alias == "platforms":
        return global_options + ["platform-status"] + rest
    if alias == "agents":
        return global_options + _agents_command(rest)

    return global_options + _prompt_command("agent-chat", "--message", args)


def command_help_text(translator=None) -> str:
    default = """HyperAgent command format

Use the Claude-Code-like launcher:
  HyperAgent "analyze the latest HSI result and propose the next experiment"
  HyperAgent
  HyperAgent chat "continue this research session"
  HyperAgent plan "implement an ablation config generator"
  HyperAgent act "inspect reports and choose the next safe command"

Slash-style aliases:
  HyperAgent /status
  HyperAgent /sessions
  HyperAgent /resume <session_id> "continue from here"
  HyperAgent /compact <session_id> --keep-last 4
  HyperAgent /model
  HyperAgent /reasonix
  HyperAgent /usage
  HyperAgent /cost
  HyperAgent /events
  HyperAgent /stats
  HyperAgent /commands
  HyperAgent /todos
  HyperAgent /doctor
  HyperAgent /web
  HyperAgent /image
  HyperAgent /ide-context
  HyperAgent /plan-mode
  HyperAgent /worktree
  HyperAgent /mcp
  HyperAgent /skills
  HyperAgent /prompts
  HyperAgent /channels
  HyperAgent /tui
  HyperAgent /repl

Inside REPL:
  /context, /compact, /clear, /usage, /cost, /stats, /budget, /pro, /skill, /checkpoint, /restore, /logs, /init, /memory, /agents, /agents run, /commands, /todos, /hooks, /permissions, /export, /doctor, /plugin, /rewind, /reasonix, /btw, /simplify, /thinking, /web, /image, /ide-context, /plan-mode, /feedback, /worktree, /mcp status

Provider options can be mixed into prompt commands:
  HyperAgent --model deepseek-v4-pro --thinking enabled --reasoning-effort max "design the next experiment"
  HyperAgent --reasonix-profile reasonix-deep "diagnose the failed experiment"
  HyperAgent send --model deepseek-v4-flash --json-output "return a JSON plan"

Canonical commands remain available:
  HyperAgent demo --synthetic
  HyperAgent run-suite --config configs/experiment.yaml --seeds 42,43,44
  HyperAgent channel-run --host 0.0.0.0 --port 8765
"""
    if translator is None:
        heading = "Registered slash commands"
        return default + f"\n{heading}:\n" + grouped_help()
    heading = translator.t(
        "launcher.registered_commands",
        default="Registered slash commands",
    )
    return (
        translator.t("launcher.help", default=default)
        + f"\n{heading}:\n"
        + grouped_help(translator=translator)
    )


def _normalize_slash_command(command: str, rest: Sequence[str]) -> List[str]:
    alias = command.lower()
    if alias == "help":
        return ["hyperagent-commands"]
    if alias in {"commands", "command"}:
        return ["command-list"] + list(rest)
    if alias == "repl":
        return ["repl"] + list(rest)
    if alias == "tui":
        return ["tui"] + list(rest)
    if alias == "status":
        return ["status"] + list(rest)
    if alias == "sessions":
        return _sessions_command(rest)
    if alias == "resume":
        return _resume_command(rest)
    if alias in {"compact", "compress"}:
        return _compact_command(rest)
    if alias == "model":
        return ["llm-providers"] + list(rest)
    if alias in {"reasonix", "deepseek"}:
        return ["llm-profile"] + list(rest)
    if alias == "usage":
        return ["llm-usage"] + list(rest)
    if alias == "cost":
        return ["llm-usage"] + list(rest)
    if alias in {"web", "image", "ide-context", "plan-mode", "personality", "feedback", "worktree"}:
        return [alias] + list(rest)
    if alias in {"rollback", "snapshot"}:
        return ["checkpoint"] + list(rest)
    if alias == "session-search":
        query = " ".join(rest).strip()
        return ["session-search", "--query", query] if query else ["session-search"]
    if alias in {
        "events",
        "replay",
        "diff",
        "stats",
        "platform-status",
        "skill-usage",
        "checkpoint",
        "restore",
        "index",
    }:
        return [alias] + list(rest)
    if alias in {"commands", "command"}:
        return ["command-list"] + list(rest)
    if alias == "todos":
        return ["todos"] + list(rest)
    if alias == "doctor":
        return ["doctor"] + list(rest)
    if alias == "mcp":
        return _mcp_command(rest)
    if alias in {"skills", "skill"}:
        return _skills_command(rest)
    if alias in {"prompts", "prompt"}:
        return ["prompt-list"] + list(rest)
    if alias == "language":
        return ["language-list"] + list(rest)
    if alias in {"channels", "channel"}:
        return ["channel-list"] + list(rest)
    if alias == "platforms":
        return ["platform-status"] + list(rest)
    if alias == "agents":
        return _agents_command(rest)
    if alias == "hsi":
        return _hsi_command(rest)
    if alias in {"chat", "ask"}:
        return _prompt_command("agent-chat", "--message", rest)
    if alias == "plan":
        return _prompt_command("agent-plan", "--instruction", rest)
    if alias in {"act", "do"}:
        return _prompt_command("agent-act", "--message", rest)
    return _prompt_command("agent-chat", "--message", ["/" + command] + list(rest))


def _agents_command(rest: Sequence[str]) -> List[str]:
    if not rest:
        return ["agent-status"]
    action = rest[0].lower()
    tail = list(rest[1:])
    if action == "status":
        return ["agent-status"] + tail
    if action == "pause":
        return ["agent-pause"] + tail
    if action == "resume":
        return ["agent-resume"] + tail
    if action == "stop":
        return ["agent-stop"] + tail
    return ["agent-status"] + list(rest)


def _sessions_command(rest: Sequence[str]) -> List[str]:
    if not rest:
        return ["session-list"]
    action = rest[0].lower()
    tail = list(rest[1:])
    if action == "search":
        query = " ".join(tail).strip()
        return ["session-search", "--query", query] if query else ["session-search"]
    return ["session-list"] + list(rest)


def _skills_command(rest: Sequence[str]) -> List[str]:
    if not rest:
        return ["skill-list"]
    action = rest[0].lower()
    tail = list(rest[1:])
    if action in {"browse", "list"}:
        return ["skill-list"] + tail
    if action == "search":
        return ["skill-search"] + tail
    if action == "inspect":
        return ["skill-inspect"] + tail
    if action == "install":
        return ["skill-install"] + tail
    if action == "bundles":
        return ["skill-bundles"] + tail
    if action in {"usage", "telemetry"}:
        return ["skill-usage"] + tail
    if action == "run":
        return ["skill-run"] + tail
    return ["skill-search"] + list(rest)


def _mcp_command(rest: Sequence[str]) -> List[str]:
    if not rest:
        return ["mcp-list"]
    action = rest[0].lower()
    tail = list(rest[1:])
    if action in {"status", "list", "tools"}:
        return ["mcp-list"] + tail
    if action in {"health"}:
        return ["mcp-health"] + tail
    if action in {"inspect", "browse"}:
        return ["mcp-inspect"] + tail
    return ["mcp-list"] + list(rest)


def _hsi_command(rest: Sequence[str]) -> List[str]:
    if not rest:
        return ["hyperagent-commands"]
    action = rest[0].lower()
    tail = list(rest[1:])
    mapping = {
        "audit": "audit",
        "plan": "plan",
        "run-baseline": "run-baseline",
        "experiment-cycle": "experiment-cycle",
        "review-experiment": "command-render",
        "literature": "literature",
        "materialize-module": "materialize-module",
    }
    target = mapping.get(action)
    if target is None:
        return _prompt_command("agent-chat", "--message", [" ".join(["/hsi"] + list(rest))])
    if action == "review-experiment":
        return [target, "review-experiment"] + tail
    return [target] + tail


def _extract_global_options(args: List[str]) -> Tuple[List[str], List[str]]:
    global_options: List[str] = []
    remaining: List[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            remaining.extend(args[index:])
            break
        if token == "--lang":
            global_options.append(token)
            if index + 1 < len(args):
                global_options.append(args[index + 1])
                index += 2
            else:
                index += 1
            continue
        if token.startswith("--lang="):
            global_options.append(token)
            index += 1
            continue
        remaining.append(token)
        index += 1
    return global_options, remaining


def _prompt_command(
    command: str,
    prompt_flag: str,
    args: Sequence[str],
    defaults: Sequence[str] = (),
) -> List[str]:
    if _contains_flag(args, prompt_flag):
        return [command] + list(defaults) + list(args)
    options, text_parts = _split_options_and_text(args)
    if not text_parts:
        return [command] + list(defaults) + options
    return [command] + list(defaults) + options + [prompt_flag, " ".join(text_parts)]


def _resume_command(args: Sequence[str]) -> List[str]:
    if not args:
        return ["session-list"]
    session_id = args[0]
    rest = list(args[1:])
    options, text_parts = _split_options_and_text(rest)
    if not text_parts:
        return ["session-show", "--session-id", session_id] + options
    return [
        "agent-chat",
        "--session-id",
        session_id,
    ] + options + ["--message", " ".join(text_parts)]


def _compact_command(args: Sequence[str]) -> List[str]:
    if not args:
        return ["session-list"]
    return ["session-compress", "--session-id", args[0]] + list(args[1:])


def _default_provider(args: Sequence[str]) -> List[str]:
    if _contains_flag(args, "--provider"):
        return []
    return ["--provider", "deepseek"]


def _contains_flag(args: Sequence[str], flag: str) -> bool:
    prefix = flag + "="
    return any(item == flag or item.startswith(prefix) for item in args)


def _split_options_and_text(args: Sequence[str]) -> Tuple[List[str], List[str]]:
    options: List[str] = []
    text: List[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            text.extend(args[index + 1 :])
            break
        if token.startswith("--"):
            options.append(token)
            if "=" in token or token in BOOLEAN_FLAGS:
                index += 1
                continue
            if index + 1 < len(args):
                options.append(args[index + 1])
                index += 2
                continue
            index += 1
            continue
        text.append(token)
        index += 1
    return options, text
