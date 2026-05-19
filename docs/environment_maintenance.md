# Environment Maintenance

HyperAgent keeps two environment files:

- `environment.yml`: human-readable dependency intent for creating a compatible environment.
- `environment.txt`: exact conda package snapshot for the current verified environment.

Update `environment.txt` after dependency changes and before stage commits:

```bash
bash scripts/update_environment_txt.sh HyperAgent environment.txt
```

Then verify:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/lzj/miniconda3/envs/HyperAgent/bin/python -m compileall -q hyperagent hermes_plugin tests
MPLCONFIGDIR=/tmp/matplotlib /home/lzj/miniconda3/envs/HyperAgent/bin/python -m unittest discover -s tests
```

Use `environment.yml` for ordinary setup. Use `environment.txt` when exact package versions matter for reproducibility.
