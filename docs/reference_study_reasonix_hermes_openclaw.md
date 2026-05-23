# Reference Study: Reasonix, Hermes Agent, OpenClaw

This note records the design ideas HyperAgent should absorb without copying
external project code.

## DeepSeek Reasonix

Source: <https://github.com/esengine/deepseek-reasonix>

Useful ideas for HyperAgent:

- Cache-first prompt layout: keep stable system rules, project memory, dataset
  cards, spectral rules, and tool schemas before volatile user/tool output.
- DeepSeek-specific runtime presets: use a fast model for routing and summaries,
  and a stronger reasoning model for algorithm design and experiment diagnosis.
- Tool-call robustness: normalize provider tool-call payloads before passing them
  into the local tool layer.
- Cost/cache visibility: record usage fields such as cache-hit and cache-miss
  tokens when the provider returns them.
- Replayable events: keep append-only runtime records so experiment-agent
  decisions can be audited later.

Implemented in this phase:

- `hyperagent.runtime.deepseek_reasonix`
- `hyperagent.runtime.llm_usage`
- `--reasonix-profile`
- `llm-profile`
- `llm-usage`
- REPL `/reasonix` and `/usage`
- DeepSeek tool-call argument normalization

## Hermes Agent

Source: <https://github.com/NousResearch/hermes-agent>

Useful ideas for HyperAgent:

- Central command registry and one launcher entrypoint.
- Persistent memory and skill lifecycle.
- Usage/insight commands, context compression, and stable context-file behavior.
- Tool permission boundaries and auditable tool output.
- Subagents with explicit iteration budgets and anti-spin constraints.

Already partially present:

- `HyperAgent` launcher
- REPL slash commands
- memory, skills, MCP, Obsidian, permission modes
- worklog and session persistence

Still worth implementing:

- broader subagent execution budgets outside experiment council
- richer live provider checks and channel retry/fallback policies

Implemented in Hermes first round:

- centralized slash-command registry entries for platform/session/skill usage
- `PlatformStatusReporter.report()` for providers, channels, sessions, skills,
  runtime events, and skill usage
- `HyperAgent platform-status --json` and gateway `GET /status`
- `SessionSearchIndex` with `HyperAgent session-search`
- `SkillTelemetryStore` with `HyperAgent skill-usage` and curator summary
- skill usage events from CLI, REPL, framework commands, action-loop
  `run_skill`, and `install_skill`

## OpenClaw

Source: <https://github.com/openclaw/openclaw>

Useful ideas for HyperAgent:

- local-first agent gateway
- multiple model/channel adapters
- explicit security and permission posture
- migration-friendly local state

Still worth implementing:

- gateway process for IDE/terminal/web UI clients
- live provider health checks and route fallback
- config migration utilities
- safer command allowlists with persistent approvals
