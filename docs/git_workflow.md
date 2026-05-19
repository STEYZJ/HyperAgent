# Git Workflow

HyperAgent uses Git by stage and task.

## Commit Cadence

Commit after each coherent task or stage:

- feature implementation is complete
- tests for the touched area pass
- generated source/config artifacts that should be reproducible are reviewed
- runtime outputs are not included

Suggested message style:

```text
stage/task: concise summary
```

Examples:

```text
runtime: add conversation persistence commands
research: materialize module proposal into model factory
experiments: add Indian Pines ablation configs
```

## Push Cadence

Push after a stable task batch or phase:

```bash
git push origin main
```

Before pushing, run:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/lzj/miniconda3/envs/HyperAgent/bin/python -m compileall -q hyperagent hermes_plugin tests
MPLCONFIGDIR=/tmp/matplotlib /home/lzj/miniconda3/envs/HyperAgent/bin/python -m unittest discover -s tests
```

## What To Commit

Commit:

- source code under `hyperagent/` and `hermes_plugin/`
- tests under `tests/`
- stable configs under `configs/`
- docs and prompt templates
- `environment.yml`, `pyproject.toml`, `.gitignore`

Do not commit:

- `.hyperagent/`
- `experiments/`
- `reports/`
- `__pycache__/`
- temporary logs or cache files

