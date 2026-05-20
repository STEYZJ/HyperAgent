# HyperAgent

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

## Quick Start

Use the project Python environment:

```bash
conda activate HyperAgent
HyperAgent demo --synthetic
```

The demo writes artifacts under `experiments/synthetic_demo/` and appends a worklog entry under `logs/worklog/`.

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
HyperAgent benchmark-matrix --catalog dataset/datasets.yaml --datasets Indian_pines,PaviaU --run-suite --seeds 42,43
HyperAgent report --experiment <experiment_dir>
HyperAgent demo --synthetic
HyperAgent literature --query "hyperspectral image classification mamba" --output reports/literature.json
HyperAgent auto-experiment --audit reports/audit.json --spectral reports/spectral_report.json --recommendation reports/model_recommendation.json --output reports/agenda.json
HyperAgent tune-next --plan experiments/run/plan.yaml --result experiments/run/result.json --audit reports/audit.json --output reports/tuning.json
HyperAgent experiment-cycle --plan experiments/run/plan.yaml --result experiments/run/result.json --audit reports/audit.json --output-root experiments/autopilot --run-next --max-repeated-parameter 2
HyperAgent propose-module --audit reports/audit.json --spectral reports/spectral_report.json --literature reports/literature.json --output reports/module_proposal.json
HyperAgent llm-providers
HyperAgent llm-profile
HyperAgent llm-usage
HyperAgent llm-dry-run --provider openai --user "Plan an HSI experiment"
HyperAgent llm-send --provider deepseek --user "Plan a small HSI baseline experiment"
HyperAgent llm-dry-run --provider deepseek --model deepseek-v4-pro --thinking enabled --reasoning-effort max --json-output --user "Return a JSON experiment plan"
HyperAgent llm-dry-run --provider deepseek --reasonix-profile reasonix-deep --user "Diagnose the failed experiment"
HyperAgent llm-send --provider deepseek --model deepseek-v4-flash --thinking disabled --top-p 0.9 --user "Plan the next HSI baseline"
HyperAgent "Plan the next experiment"
HyperAgent chat --provider deepseek --new-title "HSI research" --mode research "Plan the next experiment"
HyperAgent agent-context --query "agent plan" --max-files 12
HyperAgent plan --provider deepseek --mode code "Make HyperAgent more like Claude Code"
HyperAgent act --provider deepseek --new-title "Action loop" --max-steps 3 "Inspect benchmark matrix and choose the next safe step"
HyperAgent agent-tool read-file --path hyperagent/cli.py --max-lines 80
HyperAgent agent-tool search-code --query "AgentLoop" --path hyperagent
HyperAgent agent-tool run-command -- python -m unittest discover -s tests
HyperAgent session-new --title "Indian Pines research"
HyperAgent session-add --session-id <session_id> --role user --content "Next experiment?"
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
HyperAgent /model
HyperAgent /reasonix
HyperAgent /usage
HyperAgent /help
```

Inside the REPL, use slash commands such as `/context`, `/compact`, `/clear`, `/usage`, `/init`, `/memory`, `/agents`, `/hooks`, `/plugin`, `/rewind`, `/reasonix`, `/btw`, `/tools`, `/tool read hyperagent/cli.py`, `/tool run python -m unittest discover -s tests`, `/plan ...`, and `/act ...`. Risky local tools can require confirmation with `--permission ask`; write operations can be blocked with `--permission deny-write`.

Canonical subcommands still work, so automation scripts can keep using explicit names such as `HyperAgent run-suite` or `HyperAgent experiment-cycle`.

## MVP Limits

- Dataset readers currently support `.mat`, basic TIFF `.tif/.tiff`, and lightweight ENVI `.hdr` plus raw binary files (`bsq`, `bil`, `bip`). `.mat` remains the most mature benchmark path.
- Baselines are limited to SVM and a lightweight MLP.
- Complex HSI models such as SSRN, SpectralFormer, Mamba, and GCN are reserved for later phases.
