---
description: Diagnose HyperAgent workspace, LLM, command, hook, and runtime status.
argument-hint: ""
allowed-tools: ["read_file", "search_code", "run_command"]
profile: reasonix-balanced
---
Run a HyperAgent self-check.

$ARGUMENTS

Check workspace initialization, provider configuration, command/plugin/agent
registration, hooks, MCP/channel status, Git state, and obvious secret leaks.
Return actionable failures first.
