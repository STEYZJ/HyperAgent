"""Text panels for local tool calls and agent action runs."""

import json
from typing import Iterable, List, Optional

from hyperagent.runtime.i18n import Translator
from hyperagent.schemas import AgentActionRun, AgentToolResult


def render_tool_result(
    result: AgentToolResult,
    max_chars: int = 1800,
    translator: Optional[Translator] = None,
) -> str:
    lines = [
        f"[tool] {result.tool_name}",
        f"{_t(translator, 'tool_panel.status', 'status')}: {result.status}",
    ]
    if result.exit_code is not None:
        lines.append(f"{_t(translator, 'tool_panel.exit_code', 'exit_code')}: {result.exit_code}")
    if result.artifact_path:
        lines.append(f"{_t(translator, 'tool_panel.artifact', 'artifact')}: {result.artifact_path}")
    if result.warnings:
        lines.append(f"{_t(translator, 'tool_panel.warnings', 'warnings')}:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.content:
        lines.append(f"{_t(translator, 'tool_panel.output', 'output')}:")
        lines.append(_preview_tool_content(result, max_chars=max_chars))
    return "\n".join(lines)


def render_action_run(
    run: AgentActionRun,
    max_chars: int = 1200,
    translator: Optional[Translator] = None,
) -> str:
    lines = [
        f"[action-run] {run.run_id}",
        f"{_t(translator, 'tool_panel.status', 'status')}: {run.status}",
        f"{_t(translator, 'tool_panel.steps', 'steps')}: {len(run.steps)}",
        f"{_t(translator, 'tool_panel.artifact', 'artifact')}: {run.run_dir}/action_run.json",
    ]
    if run.warnings:
        lines.append(f"{_t(translator, 'tool_panel.warnings', 'warnings')}:")
        lines.extend(f"- {warning}" for warning in run.warnings)
    for step in run.steps:
        lines.append("")
        lines.append(
            f"{_t(translator, 'tool_panel.step', 'step')} {step.step_index}: "
            f"action={step.action} status={step.status}"
        )
        if step.tool_name:
            lines.append(f"{_t(translator, 'tool_panel.tool', 'tool')}: {step.tool_name}")
        if step.warnings:
            lines.append(f"{_t(translator, 'tool_panel.step_warnings', 'step_warnings')}:")
            lines.extend(f"- {warning}" for warning in step.warnings)
        if step.tool_result:
            lines.append(_indent(render_tool_result(step.tool_result, max_chars=max_chars, translator=translator)))
    if run.final_response:
        lines.append("")
        lines.append(f"{_t(translator, 'tool_panel.final', 'final')}:")
        lines.append(_preview(run.final_response, max_chars=max_chars))
    return "\n".join(lines)


def render_tool_catalog(tool_names: Iterable[str], title: str = "Available local tools") -> str:
    names: List[str] = list(tool_names)
    lines = [title.rstrip(":") + ":"]
    lines.extend(f"- {name}" for name in names)
    return "\n".join(lines)


def _preview(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _preview_tool_content(result: AgentToolResult, max_chars: int) -> str:
    if result.tool_name != "framework_command":
        return _preview(result.content, max_chars=max_chars)
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        return _preview(result.content, max_chars=max_chars)
    summary = _summarize_framework_payload(payload)
    return _preview(summary or result.content, max_chars=max_chars)


def _summarize_framework_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    if "providers" in payload:
        providers = payload.get("providers", {})
        if isinstance(providers, dict):
            configured = [
                name
                for name, value in providers.items()
                if bool(value)
            ]
            return "\n".join(
                [
                    f"search_configured: {bool(payload.get('search_configured'))}",
                    "providers: " + (", ".join(configured) if configured else "none"),
                    f"fetch_available: {bool(payload.get('fetch_available'))}",
                ]
            )
    if "skills" in payload:
        skills = payload.get("skills", [])
        if isinstance(skills, list):
            names = [
                str(item.get("name", ""))
                for item in skills
                if isinstance(item, dict) and item.get("name")
            ]
            visible = names[:20]
            suffix = f"\n... {len(names) - len(visible)} more" if len(names) > len(visible) else ""
            return "skills:\n" + "\n".join(f"- {name}" for name in visible) + suffix
    if "request_count" in payload:
        return "\n".join(
            [
                f"request_count: {payload.get('request_count')}",
                f"total_tokens: {payload.get('total_tokens')}",
                f"cache_hit_ratio: {payload.get('cache_hit_ratio')}",
                f"ledger_path: {payload.get('ledger_path')}",
            ]
        )
    if "initialized" in payload:
        return "\n".join(
            [
                f"initialized: {payload.get('initialized')}",
                f"workspace: {payload.get('workspace')}",
                f"dataset_root: {payload.get('dataset_root')}",
                f"task_count: {payload.get('task_count')}",
            ]
        )
    if "servers" in payload:
        servers = payload.get("servers", [])
        names = [
            str(item.get("name", ""))
            for item in servers
            if isinstance(item, dict) and item.get("name")
        ] if isinstance(servers, list) else []
        return "mcp_servers: " + (", ".join(names) if names else "none")
    return ""


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _t(translator: Optional[Translator], key: str, default: str) -> str:
    return translator.t(key, default=default) if translator is not None else default
