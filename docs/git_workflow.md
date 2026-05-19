# Git Workflow

HyperAgent uses Git by stage and task.

## Branch Policy

Keep `main` stable. Use short-lived branches for implementation:

```text
task/<short-topic>
stage/<stage-name>
experiment/<dataset-or-module>
```

Examples:

```text
task/mcp-client
stage/hsi-agent-v2
experiment/indian-pines-adapter
```

Default flow:

```bash
git switch main
git pull --ff-only origin main
git switch -c task/<short-topic>
# implement and test
git add <files>
git commit -m "task/<short-topic>: concise summary"
git switch main
git merge --ff-only task/<short-topic>
git push origin main
```

For longer experimental work, push the branch first:

```bash
git push -u origin experiment/<dataset-or-module>
```

Do not develop directly on `main` for non-trivial changes unless the change is documentation-only or a small maintenance edit.

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

If dependencies changed, update the environment snapshot before committing:

```bash
bash scripts/update_environment_txt.sh HyperAgent environment.txt
```

## What To Commit

Commit:

- source code under `hyperagent/` and `hermes_plugin/`
- tests under `tests/`
- stable configs under `configs/`
- docs and prompt templates
- `environment.yml`, `environment.txt`, `pyproject.toml`, `.gitignore`

Do not commit:

- `.hyperagent/`
- `experiments/`
- `reports/`
- `__pycache__/`
- temporary logs or cache files
