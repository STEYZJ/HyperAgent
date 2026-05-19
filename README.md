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
python -m hyperagent.cli demo --synthetic
```

The demo writes artifacts under `experiments/synthetic_demo/` and appends a worklog entry under `logs/worklog/`.

The verified environment path is:

```bash
/home/lzj/miniconda3/envs/HyperAgent
```

## CLI

```bash
python -m hyperagent.cli init --dataset-root /data2/lzj/lab/Mamba_test/dataset
python -m hyperagent.cli status
python -m hyperagent.cli task-create --goal "build reproducible Indian Pines baseline" --dataset Indian_pines --keywords "hyperspectral image classification,mamba"
python -m hyperagent.cli task-run --task-id <task_id>
python -m hyperagent.cli task-list
python -m hyperagent.cli task-show --task-id <task_id>
python -m hyperagent.cli audit --data-root <path> --output reports/audit.json
python -m hyperagent.cli plan --audit reports/audit.json --output configs/experiment.yaml
python -m hyperagent.cli run-baseline --config configs/experiment.yaml
python -m hyperagent.cli report --experiment <experiment_dir>
python -m hyperagent.cli demo --synthetic
python -m hyperagent.cli literature --query "hyperspectral image classification mamba" --output reports/literature.json
python -m hyperagent.cli auto-experiment --audit reports/audit.json --spectral reports/spectral_report.json --recommendation reports/model_recommendation.json --output reports/agenda.json
python -m hyperagent.cli tune-next --plan experiments/run/plan.yaml --result experiments/run/result.json --audit reports/audit.json --output reports/tuning.json
python -m hyperagent.cli propose-module --audit reports/audit.json --spectral reports/spectral_report.json --literature reports/literature.json --output reports/module_proposal.json
python -m hyperagent.cli llm-providers
python -m hyperagent.cli llm-dry-run --provider openai --user "Plan an HSI experiment"
python -m hyperagent.cli session-new --title "Indian Pines research"
python -m hyperagent.cli session-add --session-id <session_id> --role user --content "Next experiment?"
python -m hyperagent.cli session-compress --session-id <session_id> --keep-last 4
python -m hyperagent.cli session-archive --session-id <session_id>
python -m hyperagent.cli session-delete --session-id <session_id>
python -m hyperagent.cli skill-list
python -m hyperagent.cli mcp-add --name demo --command python --arg=-m --arg demo_server
python -m hyperagent.cli mcp-export
python -m hyperagent.cli obsidian-index --vault <obsidian_vault>
python -m hyperagent.cli obsidian-search --query "spectral gate"
python -m hyperagent.cli prompt-list
python -m hyperagent.cli prompt-render --name hsi_research_copilot --var dataset=Indian_pines --var objective=OA
python -m hyperagent.cli materialize-module --proposal reports/indian_pines/module_proposal.json --base-plan reports/indian_pines/experiment.yaml --ablation-output configs/ablations/indian_pines_evidence_adapter --force
```

The local dataset root used in this workspace is recorded in `configs/hyperagent_local.yaml`:

```text
/data2/lzj/lab/Mamba_test/dataset
```

## MVP Limits

- `.mat` is the default supported dataset format.
- Baselines are limited to SVM and a lightweight MLP.
- Complex HSI models such as SSRN, SpectralFormer, Mamba, and GCN are reserved for later phases.
