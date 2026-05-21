---
description: Multi-agent code review focused on bugs, regressions, and tests.
argument-hint: "[scope]"
allowed-tools: ["read_file", "search_code", "task", "run_command"]
profile: reasonix-balanced
---
Review the current repository changes or requested scope:

$ARGUMENTS

Prioritize concrete defects, behavioral regressions, unsafe assumptions, and
missing tests. Use task subagents when useful: code-reviewer for implementation
risks, reproducibility-reviewer for tests and repeatability, and code-explorer
for locating relevant files. Report findings first with file/line references.
