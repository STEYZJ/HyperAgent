# HyperAgent Claude-Code-Like Command Layer

HyperAgent keeps the core HSI workflow commands intact, then adds a thin launcher
that translates Claude-Code-like input into canonical CLI commands.

## Entry Format

All user-facing commands should start with `HyperAgent`:

```bash
HyperAgent "analyze the last HSI experiment and propose the next run"
HyperAgent chat "continue the research session"
HyperAgent plan "implement a new ablation generator"
HyperAgent act "inspect reports and choose the next safe command"
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
| `HyperAgent /mcp` | `mcp-list` |
| `HyperAgent /skills` | `skill-list` |
| `HyperAgent /prompts` | `prompt-list` |

Canonical commands still pass through unchanged:

```bash
HyperAgent plan --audit reports/audit.json --output configs/experiment.yaml
HyperAgent run-suite --config configs/experiment.yaml --seeds 42,43,44
```

## Boundary

This layer mimics command ergonomics, not private implementation. Core modules
remain decoupled:

- `hyperagent.runtime.command_aliases` only translates arguments.
- `hyperagent.launcher` calls the existing CLI.
- HSI data, models, training, evaluation, and tools do not import the launcher.
