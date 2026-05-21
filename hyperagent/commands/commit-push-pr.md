---
description: Commit the current task, push the branch, and prepare PR text.
argument-hint: "[message hint]"
allowed-tools: ["run_command"]
profile: reasonix-balanced
---
Complete the Git workflow for this task.

Message hint:
$ARGUMENTS

Check status and diff, avoid unrelated local changes, run the project verification
commands, commit the task, push the branch, and produce a concise PR summary.
Do not push secrets or large experiment artifacts.
