"""Claude-Code-like command aliases for the HyperAgent launcher."""

from typing import Iterable, List, Sequence, Tuple


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
    "llm-dry-run",
    "llm-send",
    "agent-chat",
    "repl",
    "agent-context",
    "agent-plan",
    "agent-act",
    "agent-tool",
    "session-new",
    "session-add",
    "session-list",
    "session-show",
    "session-archive",
    "session-delete",
    "session-compress",
    "skill-list",
    "mcp-add",
    "mcp-list",
    "mcp-export",
    "obsidian-index",
    "obsidian-search",
    "prompt-list",
    "prompt-render",
    "materialize-module",
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
}


def normalize_hyperagent_args(argv: Sequence[str]) -> List[str]:
    """Translate `HyperAgent ...` shorthand into the canonical CLI argv."""

    args = list(argv)
    if not args:
        return ["repl"]

    first = args[0]
    if first in {"-h", "--help", "help"}:
        return ["hyperagent-commands"]
    if first == "plan" and not _contains_flag(args[1:], "--audit"):
        return _prompt_command("agent-plan", "--instruction", args[1:])
    if first in EXISTING_COMMANDS:
        return args
    if first.startswith("/") and len(first) > 1:
        return _normalize_slash_command(first[1:], args[1:])

    alias = first.lower()
    rest = args[1:]
    if alias in {"chat", "ask"}:
        return _prompt_command("agent-chat", "--message", rest)
    if alias == "research":
        return _prompt_command("agent-chat", "--message", rest, defaults=["--mode", "research"])
    if alias == "algorithm":
        return _prompt_command("agent-chat", "--message", rest, defaults=["--mode", "algorithm"])
    if alias in {"plan", "code"}:
        return _prompt_command("agent-plan", "--instruction", rest)
    if alias in {"act", "do"}:
        return _prompt_command("agent-act", "--message", rest)
    if alias == "send":
        return _prompt_command(
            "llm-send",
            "--user",
            rest,
            defaults=_default_provider(rest),
        )
    if alias == "dry":
        return _prompt_command(
            "llm-dry-run",
            "--user",
            rest,
            defaults=_default_provider(rest),
        )
    if alias == "sessions":
        return ["session-list"] + rest
    if alias == "resume":
        return _resume_command(rest)
    if alias == "compact":
        return _compact_command(rest)
    if alias == "mcp":
        return ["mcp-list"] + rest
    if alias in {"skills", "skill"}:
        return ["skill-list"] + rest
    if alias in {"prompts", "prompt"}:
        return ["prompt-list"] + rest
    if alias == "model":
        return ["llm-providers"] + rest

    return _prompt_command("agent-chat", "--message", args)


def command_help_text() -> str:
    return """HyperAgent command format

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
  HyperAgent /mcp
  HyperAgent /skills
  HyperAgent /prompts
  HyperAgent /repl

Inside REPL:
  /context, /compact, /clear, /init, /memory, /agents, /hooks, /plugin, /rewind, /btw, /simplify

Provider options can be mixed into prompt commands:
  HyperAgent --model deepseek-v4-pro --thinking enabled --reasoning-effort max "design the next experiment"
  HyperAgent send --model deepseek-v4-flash --json-output "return a JSON plan"

Canonical commands remain available:
  HyperAgent demo --synthetic
  HyperAgent run-suite --config configs/experiment.yaml --seeds 42,43,44
"""


def _normalize_slash_command(command: str, rest: Sequence[str]) -> List[str]:
    alias = command.lower()
    if alias in {"help", "commands"}:
        return ["hyperagent-commands"]
    if alias == "repl":
        return ["repl"] + list(rest)
    if alias == "status":
        return ["status"] + list(rest)
    if alias == "sessions":
        return ["session-list"] + list(rest)
    if alias == "resume":
        return _resume_command(rest)
    if alias in {"compact", "compress"}:
        return _compact_command(rest)
    if alias == "model":
        return ["llm-providers"] + list(rest)
    if alias == "mcp":
        return ["mcp-list"] + list(rest)
    if alias in {"skills", "skill"}:
        return ["skill-list"] + list(rest)
    if alias in {"prompts", "prompt"}:
        return ["prompt-list"] + list(rest)
    if alias in {"chat", "ask"}:
        return _prompt_command("agent-chat", "--message", rest)
    if alias == "plan":
        return _prompt_command("agent-plan", "--instruction", rest)
    if alias in {"act", "do"}:
        return _prompt_command("agent-act", "--message", rest)
    return _prompt_command("agent-chat", "--message", ["/" + command] + list(rest))


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
