# HyperAgent

[中文文档](README.zh-CN.md)

HyperAgent is a decoupled MVP framework for hyperspectral image classification research workflows.

The first version focuses on a reproducible loop:

```text
dataset audit -> spectral diagnosis -> model recommendation -> experiment plan -> baseline training -> report
```

## Design

- `agents/` orchestrates tools and runners.
- `tools/` provides HSI research operations.
- `schemas/` defines all cross-module data contracts.
- `core/` owns protocols, registries, IO, bootstrap, and worklog helpers.
- `data/`, `models/`, `training/`, and `evaluation/` are replaceable implementation layers.
- `hermes_plugin/` is a thin adapter and contains no model or training logic.

Generated architecture and comparison diagrams are kept under `描述文件/设计结构/`.
The comparison set in `描述文件/设计结构/对比/` contrasts HyperAgent with
Hermes Agent, Claude Code references, and DeepSeek Reasonix across topology,
workflow, use-case coverage, and module communication.
The Reasonix cache-hit study in `docs/reasonix_cache_hit_study.md` explains how
DeepSeek prefix-cache stability, append-only logs, repair, and telemetry produce
the reported high cache-hit ratios and which pieces are transferable to
HyperAgent.
The feature design summary in `docs/hyperagent_feature_design_summary.md`
describes the planned product capabilities, implementation phases, acceptance
criteria, and repository governance rules.
The production potential analysis in `docs/hyperagent_production_potential.md`
maps the framework to concrete lab ResearchOps, reproducible benchmarking,
team bot, and controlled-agent runtime deployment paths.

## Quick Start

Use the project Python environment:

```bash
conda activate HyperAgent
HyperAgent demo --synthetic
```

Environment maintenance is documented in `docs/environment_maintenance.md`.
Use `environment.yml` for a compatible setup and `environment.txt` for the exact
verified package snapshot.

The demo writes artifacts under `experiments/synthetic_demo/` and appends a worklog entry under `logs/worklog/`.

## Secret Handling

Store API keys only in environment variables or the local `.env` file, which is ignored by git. Do not paste raw API keys into README files, worklogs, config examples, prompts, or committed conversation artifacts. HyperAgent redacts obvious secret shapes before writing worklogs, and the test suite includes a tracked-file secret scan.

## Bot Channels

HyperAgent can run a FastAPI webhook gateway for official Feishu and QQ bots. The v1 channel path is chat/query only: it routes text messages into the persistent AgentLoop and does not expose shell, training, file-writing, or general agent tools to external platforms. WeChat is intentionally reserved for a later adapter.

```bash
HyperAgent channel-init --provider feishu
HyperAgent channel-init --provider qq
HyperAgent channel-list
HyperAgent channel-test --provider feishu --text "Plan the next HSI experiment"
HyperAgent channel-run --host 0.0.0.0 --port 8765
```

Configure platform credentials through local environment variables or `.env`, then point the official platform callback URL at:

```text
GET /status
POST /webhooks/feishu
POST /webhooks/qq
```

The generated `.hyperagent/channels.yaml` stores only environment variable names, never secret values.

## Hermes-Style Platform Runtime

HyperAgent now exposes a local platform view inspired by Hermes Agent while keeping external channels chat/query only. `HyperAgent platform-status` aggregates provider config health, channel env health, session counts, skill/bundle counts, runtime event summaries, channel delivery status, and skill usage totals without making live provider calls or storing secret values. Live reachability is explicit with `--live`, and the channel gateway serves the same read-only payload at `GET /status`.

```bash
HyperAgent platform-status
HyperAgent platform-status --json
HyperAgent platform-status --live --json
HyperAgent channel-retry --provider feishu --limit 5
HyperAgent session-search --query "spectral result" --limit 5
HyperAgent skill-usage --json
HyperAgent /platforms
HyperAgent /sessions search "spectral result"
HyperAgent /skills usage
```

Session search writes a lightweight local index to `.hyperagent/sessions/session_index.json` and returns only short snippets, not full conversation contents. Channel delivery telemetry is append-only under `.hyperagent/channels/delivery.jsonl`; retry re-sends the already generated outbound message and does not rerun the LLM. Skill usage telemetry is append-only under `.hyperagent/skills/usage.jsonl`; the curator summary highlights unused skills, high-frequency skills, and skills missing bundle metadata without rewriting skill files.

## Controlled Web And Feature Panel

HyperAgent gives the model web access through local audited tools, not through an opaque provider-side browser. Search requires at least one provider key in local `.env` or environment variables: `BRAVE_SEARCH_API_KEY`, `TAVILY_API_KEY`, `SERPAPI_API_KEY`, or `SEARXNG_BASE_URL`. Direct fetch works for user-provided public HTTP(S) URLs without a search provider.

```bash
HyperAgent web status
HyperAgent web search --query "latest hyperspectral image classification agent"
HyperAgent web fetch --url https://example.org/paper
HyperAgent /web
```

The web tools reject `file:`, `data:`, `javascript:`, localhost, and private IP URLs. Results are saved under `.hyperagent/web_runs/`; conversations receive summaries, source URLs, fetch time, and citation ids. Image commands currently create permission-gated request artifacts under `.hyperagent/image_runs/`:

```bash
HyperAgent image status
HyperAgent image generate --prompt "HSI experiment workflow diagram"
```

Claude/Codex-style feature entries are available through CLI, REPL, TUI, and slash commands:

```bash
HyperAgent ide-context status
HyperAgent ide-context set-open-files hyperagent/cli.py hyperagent/runtime/repl.py
HyperAgent plan-mode on "design only"
HyperAgent plan-mode off
HyperAgent personality status
HyperAgent feedback add "TUI panel should emphasize web status"
HyperAgent worktree
HyperAgent /mcp status
```

Plan mode pauses action-loop tool execution (`run`, `agent-act`, `agent-run`, REPL `/act`, and manual `/tool`) until it is turned off.

## Third-Party Licenses

Known open-source dependencies and reference projects are tracked in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Update that file before adding new runtime dependencies or copying any third-party source code.

The verified environment path is:

```bash
/home/lzj/miniconda3/envs/HyperAgent
```

If the package is not installed as an editable project yet, use the repository-local launcher:

```bash
./HyperAgent demo --synthetic
```

## CLI

```bash
HyperAgent init --dataset-root /data2/lzj/lab/Mamba_test/dataset
HyperAgent status
HyperAgent task-create --goal "build reproducible Indian Pines baseline" --dataset Indian_pines --keywords "hyperspectral image classification,mamba"
HyperAgent task-run --task-id <task_id>
HyperAgent task-list
HyperAgent task-show --task-id <task_id>
HyperAgent audit --data-root <path> --output reports/audit.json
HyperAgent plan --audit reports/audit.json --output configs/experiment.yaml
HyperAgent run-baseline --config configs/experiment.yaml
HyperAgent run-suite --config configs/experiment.yaml --seeds 42,43,44 --output-dir experiments/suite
HyperAgent benchmark-list --catalog dataset/datasets.yaml
HyperAgent benchmark-protocol --catalog dataset/datasets.yaml --datasets Indian_pines,PaviaU --seeds 42,43 --baselines svm,mlp,random_forest,knn
HyperAgent benchmark-matrix --protocol reports/benchmark_protocol/benchmark_protocol.json --run-suite
HyperAgent benchmark-matrix --catalog dataset/datasets.yaml --datasets Indian_pines,PaviaU --run-suite --seeds 42,43
HyperAgent report --experiment <experiment_dir>
HyperAgent demo --synthetic
HyperAgent literature --query "hyperspectral image classification mamba" --output reports/literature.json
HyperAgent auto-experiment --audit reports/audit.json --spectral reports/spectral_report.json --recommendation reports/model_recommendation.json --output reports/agenda.json
HyperAgent tune-next --plan experiments/run/plan.yaml --result experiments/run/result.json --audit reports/audit.json --output reports/tuning.json
HyperAgent experiment-cycle --plan experiments/run/plan.yaml --result experiments/run/result.json --audit reports/audit.json --output-root experiments/autopilot --max-repeated-parameter 2
HyperAgent experiment-cycle --plan experiments/run/plan.yaml --result experiments/run/result.json --audit reports/audit.json --output-root experiments/autopilot --llm-council --council-profile reasonix-balanced
HyperAgent propose-module --audit reports/audit.json --spectral reports/spectral_report.json --literature reports/literature.json --output reports/module_proposal.json
HyperAgent llm-providers
HyperAgent llm-profile
HyperAgent llm-usage
HyperAgent web status
HyperAgent web search --query "latest hyperspectral image classification agent"
HyperAgent web fetch --url https://example.org
HyperAgent image status
HyperAgent ide-context status
HyperAgent plan-mode status
HyperAgent feedback list
HyperAgent worktree
HyperAgent llm-dry-run --provider openai --user "Plan an HSI experiment"
HyperAgent llm-send --provider deepseek --user "Plan a small HSI baseline experiment"
HyperAgent llm-dry-run --provider deepseek --model deepseek-v4-pro --thinking enabled --reasoning-effort max --json-output --user "Return a JSON experiment plan"
HyperAgent llm-dry-run --provider deepseek --reasonix-profile reasonix-deep --user "Diagnose the failed experiment"
HyperAgent llm-send --provider deepseek --model deepseek-v4-flash --thinking disabled --top-p 0.9 --user "Plan the next HSI baseline"
HyperAgent "Plan the next experiment"
HyperAgent chat --provider deepseek --new-title "HSI research" --mode research "Plan the next experiment"
HyperAgent agent-context --query "agent plan" --max-files 12
HyperAgent run --loop-mode cache-first --token-budget 4096 "inspect reports and decide the next action"
HyperAgent plan --provider deepseek --mode code "Make HyperAgent more like Claude Code"
HyperAgent act --provider deepseek --new-title "Action loop" --max-steps 3 "Inspect benchmark matrix and choose the next safe step"
HyperAgent events --limit 20
HyperAgent replay --run-id <run_id>
HyperAgent stats
HyperAgent platform-status
HyperAgent diff --left reports/a.json --right reports/b.json
HyperAgent checkpoint --path hyperagent/runtime/action_loop.py --reason "before repair change"
HyperAgent restore --checkpoint-id <checkpoint_id>
HyperAgent index --root hyperagent --root tests
HyperAgent index --query "spectral experiment council"
HyperAgent skill-list
HyperAgent skill-usage
HyperAgent skill-run --name review-experiment --arguments "review the latest result"
HyperAgent agent-tool read-file --path hyperagent/cli.py --max-lines 80
HyperAgent agent-tool search-code --query "AgentLoop" --path hyperagent
HyperAgent agent-tool run-command -- python -m unittest discover -s tests
HyperAgent session-new --title "Indian Pines research"
HyperAgent session-add --session-id <session_id> --role user --content "Next experiment?"
HyperAgent session-search --query "Indian Pines"
HyperAgent /compact <session_id> --keep-last 4
HyperAgent session-archive --session-id <session_id>
HyperAgent session-delete --session-id <session_id>
HyperAgent /skills
HyperAgent mcp-add --name demo --command python --arg=-m --arg demo_server
HyperAgent /mcp
HyperAgent obsidian-index --vault <obsidian_vault>
HyperAgent obsidian-search --query "spectral gate"
HyperAgent /prompts
HyperAgent prompt-render --name hsi_research_copilot --var dataset=Indian_pines --var objective=OA
HyperAgent materialize-module --proposal reports/indian_pines/module_proposal.json --base-plan reports/indian_pines/experiment.yaml --ablation-output configs/ablations/indian_pines_evidence_adapter --force
HyperAgent channel-run --host 0.0.0.0 --port 8765
```

The local dataset root used in this workspace is recorded in `configs/hyperagent_local.yaml`:

```text
/data2/lzj/lab/Mamba_test/dataset
```

## LLM Runtime Options

OpenAI-compatible providers share one runtime path. DeepSeek currently supports explicit model selection, thinking mode, reasoning effort, JSON output, and raw request-body extensions:

```bash
HyperAgent llm-send \
  --provider deepseek \
  --model deepseek-v4-pro \
  --thinking enabled \
  --reasoning-effort max \
  --json-output \
  --user "Return one JSON object with the next HSI experiment."
```

Useful flags:

- `--model`: choose a provider model such as `deepseek-v4-flash` or `deepseek-v4-pro`.
- `--reasonix-profile reasonix-cheap|reasonix-balanced|reasonix-deep`: choose a DeepSeek Reasonix-inspired preset for model, thinking, and reasoning strength.
- `--thinking enabled|disabled`: switch DeepSeek thinking mode for supported models.
- `--reasoning-effort high|max`: choose thinking strength. Compatibility aliases `low`, `medium`, and `xhigh` are accepted by the CLI.
- `--json-output`: sends `response_format={"type":"json_object"}`.
- `--extra-body-json`: merges raw JSON into the request body, so provider features such as `tools` and `tool_choice` can be used without changing the core agent code.
- `--top-p` and `--user-id`: pass common provider options when supported.

The Reasonix-inspired path keeps long stable context before volatile user/tool
output and records LLM usage in `.hyperagent/usage/llm_usage.jsonl`. Use
`HyperAgent llm-usage` or REPL `/usage` to inspect token counts and provider
cache-hit fields when available.

## Claude-Code-Like Launcher

HyperAgent now has a Claude-Code-like launcher and an interactive REPL. The command format starts with `HyperAgent`:

```bash
HyperAgent
HyperAgent "analyze the last report and propose the next experiment"
HyperAgent --model deepseek-v4-pro --thinking enabled --reasoning-effort max "design an evidence-backed ablation"
HyperAgent plan "materialize module_proposal.json into a model factory"
HyperAgent act "inspect reports and choose the next safe local tool"
HyperAgent repl --permission ask
HyperAgent /resume <session_id> "continue from the last result"
HyperAgent /status
HyperAgent /sessions
HyperAgent /platforms
HyperAgent /session-search "spectral result"
HyperAgent /model
HyperAgent /reasonix
HyperAgent /usage
HyperAgent /commands
HyperAgent /todos
HyperAgent /doctor
HyperAgent /help
```

Inside the REPL, use slash commands such as `/context`, `/compact`, `/clear`, `/usage`, `/init`, `/memory`, `/agents`, `/agents run`, `/commands`, `/todos`, `/hooks`, `/permissions`, `/export`, `/doctor`, `/plugin`, `/plugin bundles`, `/rewind`, `/reasonix`, `/btw`, `/platforms`, `/sessions search ...`, `/skills usage`, `/ide-context`, `/plan-mode`, `/web`, `/image`, `/feedback`, `/worktree`, `/mcp status`, `/tools`, `/tool read hyperagent/cli.py`, `/tool web-fetch https://example.org`, `/tool run python -m unittest discover -s tests`, `/plan ...`, and `/act ...`. Risky local tools can require confirmation with `--permission ask`; answer `a` at the prompt to remember the exact tool/risk/args approval under `.hyperagent/permissions/remembered.json`, inspect it with `/permissions show <id|key>`, and remove it with `/permissions forget <id|key>` or `/permissions clear`. Write operations can be blocked with `--permission deny-write`.

The fullscreen TUI shares the same REPL runtime and now exposes a Claude-style status line with provider/model, short session id, permission mode, context meter, token usage, cache-hit ratio, and wait status. Press `Ctrl-P` to open a command palette over built-in slash commands, Markdown commands, skills, and plugin bundles. Its side panel also shows context summaries plus session and remembered permission details. Action-loop terminal states emit a lightweight `TaskComplete` hook, so local hook rules can react to completed, paused, or failed agent tasks without expanding commit/test automation.

Canonical subcommands still work, so automation scripts can keep using explicit names such as `HyperAgent run-suite` or `HyperAgent experiment-cycle`.

## Commands, Agents, Hooks, and Todos

HyperAgent supports Claude-Code-inspired Markdown commands and built-in research subagents without copying Claude Code source. Built-in commands include `/feature-dev`, `/code-review`, `/review-experiment`, `/commit`, `/commit-push-pr`, `/doctor`, `/permissions`, `/export`, and `/bug`.

```bash
HyperAgent command-list
HyperAgent command-render --name feature-dev --arguments "add a result reviewer"
HyperAgent plugin-bundles --json
HyperAgent agent-run --agent code-reviewer --instruction "review the current diff"
HyperAgent agent-tool todo-write --owner project --item "inspect experiment report"
HyperAgent todos --owner project
HyperAgent doctor
```

Project commands can be added under `.hyperagent/commands/*.md`; project agents can be added under `.hyperagent/agents/*.md`; hooks can be added through `/hooks add` or `.hyperagent/hooks/*.md`. External Bot channels remain chat/query only and cannot trigger shell, training, write, or general agent tools.

## Reasonix-Inspired Runtime

HyperAgent reimplements selected DeepSeek-Reasonix ideas in Python without copying its TypeScript source. The current slice adds:

- `--loop-mode cache-first` for action-loop runs, which records a stable prefix hash over the stable system/cache guidance prefix while keeping repo, task, session, and tool output in later volatile messages.
- Native `tool_calls` parsing plus JSON list/actions/tool_calls repair, reasoning-content repair, direct tool-action normalization, and a repeated tool-call storm breaker.
- Runtime event logs under `.hyperagent/events/runtime_events.jsonl`, inspectable with `HyperAgent events`, `HyperAgent replay`, and `HyperAgent stats`; replay now shows response, repair, tool step, final, paused, and max-step milestones.
- Cumulative action-run token budgets, usage event ids, cache-hit/miss token totals, and cache-hit ratio metadata in `action_run.json`.
- Reversible file checkpoints under `.hyperagent/checkpoints`, usable through `HyperAgent checkpoint`, `HyperAgent restore`, `/checkpoint`, and `/restore`.
- Built-in HSI skills under `hyperagent/skills`, including `review-experiment`, `spectral-critic`, and `paper-method-extractor`.
- A lightweight lexical `HyperAgent index` command as the placeholder interface for later embedding-backed semantic retrieval.

This is not a full Reasonix clone yet: live MCP clients, rich dashboard/desktop parity, edit gates, and background job management remain staged work.

`experiment-cycle` now uses an executable multi-agent council by default. It
writes `council_run.json` plus the compatible `council_decision.json`; pass
`--council-mode static` to use the older fixed-rule council. The command only
runs the next experiment when `--run-next` is explicit.

`benchmark-protocol` turns the benchmark catalog into a paper-style empirical
plan: multiple datasets, multiple seeds, fixed split fingerprints, and a
baseline list. `benchmark-matrix --protocol ...` then expands the protocol into
dataset x baseline rows and can optionally run each row as a multi-seed suite.
The protocol records split hashes and sample counts, not full train/test index
lists.

## MVP Limits

- Dataset readers currently support `.mat`, basic TIFF `.tif/.tiff`, and lightweight ENVI `.hdr` plus raw binary files (`bsq`, `bil`, `bip`). `.mat` remains the most mature benchmark path.
- Baselines are limited to SVM and a lightweight MLP.
- Complex HSI models such as SSRN, SpectralFormer, Mamba, and GCN are reserved for later phases.
