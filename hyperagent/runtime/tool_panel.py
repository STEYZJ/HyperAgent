"""Text panels for local tool calls and agent action runs."""

from typing import Iterable, List

from hyperagent.schemas import AgentActionRun, AgentToolResult


def render_tool_result(result: AgentToolResult, max_chars: int = 1800) -> str:
    lines = [
        f"[tool] {result.tool_name}",
        f"status: {result.status}",
    ]
    if result.exit_code is not None:
        lines.append(f"exit_code: {result.exit_code}")
    if result.artifact_path:
        lines.append(f"artifact: {result.artifact_path}")
    if result.warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.content:
        lines.append("output:")
        lines.append(_preview(result.content, max_chars=max_chars))
    return "\n".join(lines)


def render_action_run(run: AgentActionRun, max_chars: int = 1200) -> str:
    lines = [
        f"[action-run] {run.run_id}",
        f"status: {run.status}",
        f"steps: {len(run.steps)}",
        f"artifact: {run.run_dir}/action_run.json",
    ]
    if run.warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in run.warnings)
    for step in run.steps:
        lines.append("")
        lines.append(
            f"step {step.step_index}: action={step.action} status={step.status}"
        )
        if step.tool_name:
            lines.append(f"tool: {step.tool_name}")
        if step.warnings:
            lines.append("step_warnings:")
            lines.extend(f"- {warning}" for warning in step.warnings)
        if step.tool_result:
            lines.append(_indent(render_tool_result(step.tool_result, max_chars=max_chars)))
    if run.final_response:
        lines.append("")
        lines.append("final:")
        lines.append(_preview(run.final_response, max_chars=max_chars))
    return "\n".join(lines)


def render_tool_catalog(tool_names: Iterable[str]) -> str:
    names: List[str] = list(tool_names)
    lines = ["Available local tools:"]
    lines.extend(f"- {name}" for name in names)
    return "\n".join(lines)


def _preview(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())
