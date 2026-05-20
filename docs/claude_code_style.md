# HyperAgent Claude-Code-Like Command Layer

HyperAgent keeps the core HSI workflow commands intact, then adds a thin launcher
that translates Claude-Code-like input into canonical CLI commands.

## Entry Format

All user-facing commands should start with `HyperAgent`:

```bash
HyperAgent
HyperAgent "analyze the last HSI experiment and propose the next run"
HyperAgent chat "continue the research session"
HyperAgent plan "implement a new ablation generator"
HyperAgent act "inspect reports and choose the next safe command"
HyperAgent repl --permission ask
```

Repository-local usage:

```bash
./HyperAgent /help
```

Editable-install usage:

```bash
HyperAgent /help
```

## Alias Mapping

| HyperAgent command | Canonical command |
| --- | --- |
| `HyperAgent` | `repl` |
| `HyperAgent repl` | `repl` |
| `HyperAgent "prompt"` | `agent-chat --message "prompt"` |
| `HyperAgent chat "prompt"` | `agent-chat --message "prompt"` |
| `HyperAgent plan "task"` | `agent-plan --instruction "task"` |
| `HyperAgent act "task"` | `agent-act --message "task"` |
| `HyperAgent send "prompt"` | `llm-send --provider deepseek --user "prompt"` |
| `HyperAgent dry "prompt"` | `llm-dry-run --provider deepseek --user "prompt"` |
| `HyperAgent /status` | `status` |
| `HyperAgent /sessions` | `session-list` |
| `HyperAgent /resume <id> "prompt"` | `agent-chat --session-id <id> --message "prompt"` |
| `HyperAgent /compact <id>` | `session-compress --session-id <id>` |
| `HyperAgent /model` | `llm-providers` |
| `HyperAgent /reasonix` | `llm-profile` |
| `HyperAgent /usage` | `llm-usage` |
| `HyperAgent /mcp` | `mcp-list` |
| `HyperAgent /skills` | `skill-list` |
| `HyperAgent /prompts` | `prompt-list` |
| `HyperAgent /repl` | `repl` |

Canonical commands still pass through unchanged:

```bash
HyperAgent plan --audit reports/audit.json --output configs/experiment.yaml
HyperAgent run-suite --config configs/experiment.yaml --seeds 42,43,44
```

## Interactive REPL

The REPL preserves one active conversation session and accepts plain text as
agent-chat turns. Slash commands handle local operations:

```text
/help                 show commands
/status               show workspace status
/session              show current session
/sessions             list sessions
/new [title]          create a new session
/resume <session_id>  switch session
/context              show context compression status
/usage [limit]        summarize LLM usage and cache-hit ledger
/compact [keep_last]  compress current session
/clear                clear context after saving a rewind snapshot
/init                 create project HyperAgent.md memory
/memory ...           list/show/add project, user, or auto memory
/agents ...           list/add project subagents
/hooks ...            list/add project hooks
/plugin ...           list/add project plugins
/rewind [save]        list or save rewind snapshots
/reasonix [profile]   show DeepSeek Reasonix-inspired profiles
/btw <question>       ask an isolated temporary question
/simplify             show the three-agent simplification council
/model                list LLM providers
/mcp                  list MCP servers
/skills               list skills
/tools                list local tools
/tool ...             run a local tool with permission policy
/plan <instruction>   generate a coding/algorithm plan
/act <instruction>    run controlled LLM tool loop
/exit                 quit
```

Permission modes:

- `auto`: run allowlisted tools without asking.
- `ask`: confirm execute/write operations interactively.
- `deny-write`: allow read/check operations and block patch application.
- `deny`: block execute/write operations.

Tool result panels show status, exit code, artifact path, warnings, and output
preview so experiment and code actions remain auditable.

## DeepSeek Reasonix Notes

HyperAgent now exposes Reasonix-inspired DeepSeek presets without coupling the
core agent loop to one vendor:

```bash
HyperAgent /reasonix
HyperAgent --reasonix-profile reasonix-balanced "analyze the failed run"
HyperAgent --reasonix-profile reasonix-deep "design the next HSI module"
HyperAgent /usage
```

The runtime also records an append-only LLM usage ledger under
`.hyperagent/usage/llm_usage.jsonl`, including provider usage fields such as
`prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` when the provider
returns them.

## Boundary

This layer mimics command ergonomics, not private implementation. Core modules
remain decoupled:

- `hyperagent.runtime.command_aliases` only translates arguments.
- `hyperagent.launcher` calls the existing CLI.
- `hyperagent.runtime.repl` orchestrates conversation, tools, permissions, and panels.
- HSI data, models, training, evaluation, and tools do not import the launcher.
