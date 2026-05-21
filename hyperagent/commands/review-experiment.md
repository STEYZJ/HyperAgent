---
description: Review an HSI experiment result and propose the next grounded run.
argument-hint: "<result/report path or question>"
allowed-tools: ["read_file", "search_code", "task", "todo_write", "run_experiment"]
profile: reasonix-deep
---
Review this hyperspectral image classification experiment context:

$ARGUMENTS

Ground the diagnosis in saved artifacts. Use experiment-analyst for metrics,
spectral-critic for band/data reasoning, and reproducibility-reviewer for split,
seed, and evidence checks when those subagents are available. Prefer a small,
purposeful next experiment over broad search.
