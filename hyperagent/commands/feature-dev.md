---
description: Guided multi-agent feature development workflow.
argument-hint: "<feature request>"
allowed-tools: ["read_file", "search_code", "task", "todo_write", "check_patch", "apply_patch", "run_command"]
profile: reasonix-balanced
---
Run a HyperAgent feature-development workflow for:

$ARGUMENTS

Use TodoWrite first to track discovery, design, implementation, tests, and review.
Explore the repository before editing. If specialized subagents are available, use
the task tool to ask code-explorer, code-architect, and code-reviewer for focused
input. Keep changes scoped and validate with tests before the final answer.
