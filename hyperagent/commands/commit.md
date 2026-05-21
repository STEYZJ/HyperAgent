---
description: Prepare a focused commit after status, diff, tests, and secret checks.
argument-hint: "[message hint]"
allowed-tools: ["run_command"]
profile: reasonix-balanced
---
Prepare a focused Git commit for the current task.

Message hint:
$ARGUMENTS

Inspect `git status`, relevant diffs, and test state. Do not include unrelated
local changes. Run a secret scan before committing and explain any files that
must remain untracked.
