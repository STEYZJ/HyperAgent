"""Controlled local tools for Claude-Code-like agent workflows."""

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import uuid4

from hyperagent.core.io import read_yaml, write_json
from hyperagent.runtime.checkpoints import CheckpointStore, paths_from_unified_diff
from hyperagent.runtime.events import RuntimeEventLog
from hyperagent.runtime.repo_context import SKIP_DIRS, TEXT_SUFFIXES
from hyperagent.runtime.web_tools import (
    configured_search_provider,
    recent_citations,
    web_fetch as controlled_web_fetch,
    web_search as controlled_web_search,
    write_web_artifact,
)
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import AgentToolCall, AgentToolResult, ExperimentPlan


ALLOWED_GIT_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "branch",
    "rev-parse",
    "ls-files",
}

ALLOWED_PYTHON_MODULES = {
    "compileall",
    "unittest",
    "hyperagent.cli",
}


@dataclass
class ToolPermissionRequest:
    tool_name: str
    args: dict
    risk_level: str
    reason: str
    run_id: Optional[str] = None


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    risk_level: str = "read"
    mutating: bool = False
    parallel_safe: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TOOL_METADATA: Dict[str, ToolMetadata] = {
    "read_file": ToolMetadata(
        name="read_file",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Read a text file inside the project root.",
    ),
    "search_code": ToolMetadata(
        name="search_code",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Search text inside project files.",
    ),
    "run_command": ToolMetadata(
        name="run_command",
        risk_level="execute",
        mutating=True,
        parallel_safe=False,
        description="Run an allowlisted or user-authorized shell command.",
    ),
    "run_experiment": ToolMetadata(
        name="run_experiment",
        risk_level="training",
        mutating=True,
        parallel_safe=False,
        description="Run a HyperAgent experiment YAML.",
    ),
    "task": ToolMetadata(
        name="task",
        risk_level="agent",
        mutating=False,
        parallel_safe=True,
        description="Run one or more subagents and aggregate their output.",
    ),
    "run_skill": ToolMetadata(
        name="run_skill",
        risk_level="agent",
        mutating=False,
        parallel_safe=True,
        description="Run a SKILL.md skill inline or as a subagent.",
    ),
    "framework_command": ToolMetadata(
        name="framework_command",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Query safe HyperAgent framework status and registry features.",
    ),
    "todo_write": ToolMetadata(
        name="todo_write",
        risk_level="write",
        mutating=True,
        parallel_safe=False,
        description="Replace TodoWrite state.",
    ),
    "check_patch": ToolMetadata(
        name="check_patch",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Validate a unified diff without applying it.",
    ),
    "apply_patch": ToolMetadata(
        name="apply_patch",
        risk_level="write",
        mutating=True,
        parallel_safe=False,
        description="Apply a unified diff after permission and checkpoint.",
    ),
    "web_search": ToolMetadata(
        name="web_search",
        risk_level="network",
        mutating=False,
        parallel_safe=True,
        description="Search the web through a configured provider and save citations.",
    ),
    "web_fetch": ToolMetadata(
        name="web_fetch",
        risk_level="network",
        mutating=False,
        parallel_safe=True,
        description="Fetch a public HTTP(S) page, extract text, and save an artifact.",
    ),
    "web_extract": ToolMetadata(
        name="web_extract",
        risk_level="network",
        mutating=False,
        parallel_safe=True,
        description="Fetch and extract readable text from a public HTTP(S) page.",
    ),
    "web_cite": ToolMetadata(
        name="web_cite",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="List recent web citations saved by HyperAgent.",
    ),
    "image_generate": ToolMetadata(
        name="image_generate",
        risk_level="image",
        mutating=True,
        parallel_safe=False,
        description="Create an image-generation request artifact.",
    ),
    "image_edit": ToolMetadata(
        name="image_edit",
        risk_level="image",
        mutating=True,
        parallel_safe=False,
        description="Create an image-edit request artifact.",
    ),
    "research_pattern_search": ToolMetadata(
        name="research_pattern_search",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Search extracted novelty, gap, contribution, and reviewer-expectation research patterns.",
    ),
    "experiment_strategy_search": ToolMetadata(
        name="experiment_strategy_search",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Search extracted baseline, ablation, control-variable, robustness, and visualization strategies.",
    ),
    "storytelling_search": ToolMetadata(
        name="storytelling_search",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Search extracted scientific storytelling and reviewer-persuasion strategies.",
    ),
    "research_taste_search": ToolMetadata(
        name="research_taste_search",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Search cross-paper research taste and field-timing lessons.",
    ),
    "extract_research_pattern": ToolMetadata(
        name="extract_research_pattern",
        risk_level="write",
        mutating=True,
        parallel_safe=False,
        description="Extract research-pattern lessons from one paper and write a HyperVault strategy card.",
    ),
    "extract_experiment_strategy": ToolMetadata(
        name="extract_experiment_strategy",
        risk_level="write",
        mutating=True,
        parallel_safe=False,
        description="Extract experiment-strategy lessons from one paper and write a HyperVault strategy card.",
    ),
    "extract_storytelling": ToolMetadata(
        name="extract_storytelling",
        risk_level="write",
        mutating=True,
        parallel_safe=False,
        description="Extract scientific-storytelling lessons from one paper and write a HyperVault strategy card.",
    ),
    "extract_research_taste": ToolMetadata(
        name="extract_research_taste",
        risk_level="write",
        mutating=True,
        parallel_safe=False,
        description="Extract or mark underdetermined research-taste lessons for one paper.",
    ),
    "paper_strategy_compare": ToolMetadata(
        name="paper_strategy_compare",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Compare research strategies across papers without writing derived memory.",
    ),
    "compare_paper_strategies": ToolMetadata(
        name="compare_paper_strategies",
        risk_level="read",
        mutating=False,
        parallel_safe=True,
        description="Alias for paper_strategy_compare.",
    ),
    "research_experience_consolidate": ToolMetadata(
        name="research_experience_consolidate",
        risk_level="write",
        mutating=True,
        parallel_safe=False,
        description="Write long-term research-experience memory from paper strategy cards.",
    ),
    "consolidate_research_experience": ToolMetadata(
        name="consolidate_research_experience",
        risk_level="write",
        mutating=True,
        parallel_safe=False,
        description="Alias for research_experience_consolidate.",
    ),
}


def tool_metadata(tool_name: str) -> ToolMetadata:
    return TOOL_METADATA.get(
        tool_name,
        ToolMetadata(
            name=tool_name,
            risk_level="unknown",
            mutating=True,
            parallel_safe=False,
            description="Unknown tool.",
        ),
    )


def tool_catalog() -> List[Dict[str, Any]]:
    return [TOOL_METADATA[name].to_dict() for name in sorted(TOOL_METADATA)]


class SafeAgentToolExecutor:
    """Executes a small set of auditable local tools."""

    def __init__(
        self,
        project_root: Path,
        workspace_dir: Path,
        permission_policy: str = "auto",
        permission_callback: Optional[Callable[[ToolPermissionRequest], bool]] = None,
        session_permission_cache: Optional[Dict[str, bool]] = None,
        allow_arbitrary_commands: bool = False,
        hook_engine: Optional[Any] = None,
        event_log: Optional[RuntimeEventLog] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.workspace_dir = workspace_dir.resolve()
        self.tool_runs_dir = self.workspace_dir / "tool_runs"
        self.permission_policy = permission_policy
        self.permission_callback = permission_callback
        self.session_permission_cache = (
            session_permission_cache if session_permission_cache is not None else {}
        )
        self.allow_arbitrary_commands = allow_arbitrary_commands
        self.hook_engine = hook_engine
        self.event_log = event_log or RuntimeEventLog(self.workspace_dir)

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        max_lines: int = 200,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "read_file",
            {
                "path": path,
                "start_line": start_line,
                "max_lines": max_lines,
            },
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        try:
            target = self._resolve_safe_path(path)
        except ValueError as exc:
            return self._record(
                call,
                "blocked",
                str(exc),
                warnings=["read_file path must stay inside project root"],
            )
        if not target.exists() or not target.is_file():
            return self._record(
                call,
                "error",
                f"File not found: {path}",
                warnings=["read_file target must be an existing file"],
            )
        if target.suffix.lower() not in TEXT_SUFFIXES:
            return self._record(
                call,
                "error",
                f"Refusing to read non-text file: {path}",
                warnings=["read_file only supports text-like project files"],
            )
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(start_line, 1)
        end = min(start + max(max_lines, 1) - 1, len(lines))
        content = "\n".join(
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(start, end + 1)
        )
        return self._record(call, "ok", content)

    def search_code(
        self,
        query: str,
        path: str = ".",
        max_results: int = 50,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "search_code",
            {"query": query, "path": path, "max_results": max_results},
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        if not query.strip():
            return self._record(
                call,
                "error",
                "Empty query",
                warnings=["search_code requires a non-empty query"],
            )
        try:
            root = self._resolve_safe_path(path)
        except ValueError as exc:
            return self._record(
                call,
                "blocked",
                str(exc),
                warnings=["search_code path must stay inside project root"],
            )
        if not root.exists():
            return self._record(
                call,
                "error",
                f"Path not found: {path}",
                warnings=["search_code path must exist"],
            )
        files = [root] if root.is_file() else root.rglob("*")
        needle = query.lower()
        matches: List[str] = []
        for candidate in files:
            if not candidate.is_file() or self._is_skipped(candidate):
                continue
            if candidate.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle in line.lower():
                    relative = candidate.relative_to(self.project_root)
                    matches.append(f"{relative}:{line_number}: {line.strip()}")
                    if len(matches) >= max(max_results, 1):
                        return self._record(call, "ok", "\n".join(matches))
        return self._record(call, "ok", "\n".join(matches))

    def run_command(
        self,
        argv: Sequence[str],
        timeout_sec: int = 60,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "run_command",
            {"argv": list(argv), "timeout_sec": timeout_sec},
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        allowed, reason = self._command_allowed(argv)
        if not allowed:
            return self._record(
                call,
                "blocked",
                reason,
                warnings=["command is outside the HyperAgent agent-tool allowlist"],
            )
        permission_reason = (
            "run a user-authorized arbitrary command"
            if self.allow_arbitrary_commands
            else "run a local allowlisted command"
        )
        permission = self._check_permission(
            call,
            risk_level="execute",
            reason=permission_reason,
        )
        if permission is not None:
            return permission
        try:
            completed = subprocess.run(
                list(argv),
                cwd=str(self.project_root),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return self._record(
                call,
                "timeout",
                str(exc),
                exit_code=None,
                warnings=["command timed out"],
            )
        except OSError as exc:
            return self._record(
                call,
                "error",
                str(exc),
                warnings=["command failed to start"],
            )
        content = self._join_process_output(completed.stdout, completed.stderr)
        status = "ok" if completed.returncode == 0 else "error"
        return self._record(
            call,
            status,
            content,
            exit_code=completed.returncode,
        )

    def run_experiment(
        self,
        plan_path: str,
        seeds: Optional[Sequence[int]] = None,
        output_dir: Optional[str] = None,
        suite_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "run_experiment",
            {
                "plan_path": plan_path,
                "seeds": list(seeds or []),
                "output_dir": output_dir,
                "suite_name": suite_name,
            },
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        try:
            target = self._resolve_safe_path(plan_path)
        except ValueError as exc:
            return self._record(
                call,
                "blocked",
                str(exc),
                warnings=["run_experiment plan path must stay inside project root"],
            )
        if not target.exists() or not target.is_file():
            return self._record(
                call,
                "error",
                f"Experiment plan not found: {plan_path}",
                warnings=["run_experiment requires an existing YAML plan"],
            )
        permission = self._check_permission(
            call,
            risk_level="training",
            reason="run a HyperAgent experiment plan",
        )
        if permission is not None:
            return permission
        try:
            from hyperagent.tools.report_builder import MarkdownReportBuilder
            from hyperagent.training.baseline_runner import BaselineRunner
            from hyperagent.training.experiment_suite import ExperimentSuiteRunner

            plan = ExperimentPlan.from_dict(read_yaml(target))
            artifacts: List[str] = []
            if seeds:
                suite = ExperimentSuiteRunner().run(
                    plan,
                    seeds=[int(seed) for seed in seeds],
                    output_dir=(
                        self._resolve_safe_path(output_dir)
                        if output_dir
                        else None
                    ),
                    suite_name=suite_name,
                )
                artifacts = [str(Path(suite.output_dir) / "suite.json")] + list(
                    suite.artifacts
                )
                content = {
                    "mode": "suite",
                    "suite_name": suite.suite_name,
                    "output_dir": suite.output_dir,
                    "suite_path": str(Path(suite.output_dir) / "suite.json"),
                    "report_path": str(Path(suite.output_dir) / "suite_report.md"),
                    "oa_mean": suite.metrics_summary["overall_accuracy"]["mean"],
                    "oa_std": suite.metrics_summary["overall_accuracy"]["std"],
                }
            else:
                result = BaselineRunner().run(plan)
                report = MarkdownReportBuilder().write(
                    result,
                    Path(result.experiment_dir) / "report.md",
                )
                artifacts = [
                    str(Path(result.experiment_dir) / "result.json"),
                    str(report),
                ] + list(result.artifacts)
                content = {
                    "mode": "single",
                    "experiment_name": result.experiment_name,
                    "result_path": str(Path(result.experiment_dir) / "result.json"),
                    "report_path": str(report),
                    "overall_accuracy": result.evaluation.overall_accuracy,
                    "average_accuracy": result.evaluation.average_accuracy,
                    "kappa": result.evaluation.kappa,
                }
            return self._record(
                call,
                "ok",
                json.dumps(content, ensure_ascii=False, indent=2),
                warnings=[f"artifact: {path}" for path in artifacts],
            )
        except Exception as exc:
            return self._record(
                call,
                "error",
                f"{type(exc).__name__}: {exc}",
                warnings=["run_experiment failed"],
            )

    def check_patch(
        self,
        patch_text: str,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._git_apply(patch_text, apply=False, run_id=run_id)

    def apply_patch(
        self,
        patch_text: str,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._git_apply(patch_text, apply=True, run_id=run_id)

    def _git_apply(
        self,
        patch_text: str,
        apply: bool,
        run_id: Optional[str],
    ) -> AgentToolResult:
        tool_name = "apply_patch" if apply else "check_patch"
        call = self._call(tool_name, {"apply": apply}, run_id=run_id)
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        if not patch_text.strip():
            return self._record(
                call,
                "error",
                "Empty patch",
                warnings=[f"{tool_name} requires a non-empty unified diff"],
            )
        if apply:
            touched_paths = paths_from_unified_diff(patch_text)
            if touched_paths:
                checkpoint = CheckpointStore(
                    self.project_root,
                    self.workspace_dir,
                ).create(touched_paths, reason=f"before {tool_name}")
                call.args["checkpoint_id"] = checkpoint.checkpoint_id
            permission = self._check_permission(
                call,
                risk_level="write",
                reason="apply a patch to project files",
            )
            if permission is not None:
                return permission
        argv = ["git", "apply"] if apply else ["git", "apply", "--check"]
        try:
            completed = subprocess.run(
                argv,
                cwd=str(self.project_root),
                input=patch_text,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            return self._record(
                call,
                "timeout",
                str(exc),
                warnings=[f"{tool_name} timed out"],
            )
        content = self._join_process_output(completed.stdout, completed.stderr)
        status = "ok" if completed.returncode == 0 else "error"
        return self._record(
            call,
            status,
            content or ("patch applied" if apply else "patch check passed"),
            exit_code=completed.returncode,
        )

    def todo_write(
        self,
        items: Sequence[Dict[str, object]],
        owner: str = "project",
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "todo_write",
            {"owner": owner, "items": list(items)},
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        try:
            from hyperagent.runtime.todos import TodoStore

            store = TodoStore(self.workspace_dir)
            todo_list = store.replace(owner, items)
            path = store.path_for(todo_list.owner)
            return self._record(
                call,
                "ok",
                json.dumps(todo_list.to_dict(), ensure_ascii=False, indent=2),
                warnings=[f"artifact: {path}"],
            )
        except Exception as exc:
            return self._record(
                call,
                "error",
                f"{type(exc).__name__}: {exc}",
                warnings=["todo_write failed"],
            )

    def framework_command(
        self,
        command: str,
        args: Optional[Sequence[str]] = None,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        normalized_args = [str(item) for item in (args or [])]
        call = self._call(
            "framework_command",
            {"command": command, "args": normalized_args},
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        try:
            payload = self._framework_command_payload(command, normalized_args)
            return self._record(
                call,
                "ok",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
        except KeyError as exc:
            return self._record(
                call,
                "blocked",
                str(exc),
                warnings=["unsupported framework command"],
            )
        except Exception as exc:
            return self._record(
                call,
                "error",
                f"{type(exc).__name__}: {exc}",
                warnings=["framework_command failed"],
            )

    def web_search(
        self,
        query: str,
        provider: str = "auto",
        max_results: int = 5,
        timeout_sec: int = 20,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "web_search",
            {
                "query": query,
                "provider": provider,
                "max_results": max_results,
                "timeout_sec": timeout_sec,
            },
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        if not str(query or "").strip():
            return self._record(
                call,
                "error",
                "query is required",
                warnings=["web_search requires a non-empty query"],
            )
        if not configured_search_provider(provider):
            return self._record(
                call,
                "error",
                "No web search provider configured. Set BRAVE_SEARCH_API_KEY, TAVILY_API_KEY, SERPAPI_API_KEY, or SEARXNG_BASE_URL.",
                warnings=["web search provider missing"],
            )
        permission = self._check_permission(
            call,
            risk_level="network",
            reason="search the public web through a configured provider",
        )
        if permission is not None:
            return permission
        try:
            payload = controlled_web_search(
                query,
                provider=provider,
                max_results=max_results,
                timeout_sec=timeout_sec,
            )
            artifact = write_web_artifact(self.workspace_dir, payload, run_id=run_id)
            content = json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)
            return self._record(
                call,
                "ok",
                content,
                warnings=[f"artifact: {artifact}"],
            )
        except Exception as exc:
            return self._record(
                call,
                "error",
                f"{type(exc).__name__}: {exc}",
                warnings=["web_search failed"],
            )

    def web_fetch(
        self,
        url: str,
        max_chars: int = 12000,
        timeout_sec: int = 20,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "web_fetch",
            {"url": url, "max_chars": max_chars, "timeout_sec": timeout_sec},
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        permission = self._check_permission(
            call,
            risk_level="network",
            reason="fetch a public HTTP(S) URL and save a cited artifact",
        )
        if permission is not None:
            return permission
        try:
            payload = controlled_web_fetch(
                url,
                max_chars=max_chars,
                timeout_sec=timeout_sec,
            )
            artifact = write_web_artifact(self.workspace_dir, payload, run_id=run_id)
            content = json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)
            return self._record(
                call,
                "ok",
                content,
                warnings=[f"artifact: {artifact}", f"citation: {payload.citation_id}"],
            )
        except Exception as exc:
            return self._record(
                call,
                "error",
                f"{type(exc).__name__}: {exc}",
                warnings=["web_fetch failed"],
            )

    def web_extract(
        self,
        url: str,
        selector: str = "",
        max_chars: int = 12000,
        timeout_sec: int = 20,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "web_extract",
            {
                "url": url,
                "selector": selector,
                "max_chars": max_chars,
                "timeout_sec": timeout_sec,
            },
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        permission = self._check_permission(
            call,
            risk_level="network",
            reason="extract readable text from a public HTTP(S) URL",
        )
        if permission is not None:
            return permission
        try:
            payload = controlled_web_fetch(
                url,
                max_chars=max_chars,
                timeout_sec=timeout_sec,
            )
            payload.kind = "web_extract"
            if selector:
                payload.metadata["selector"] = selector
                payload.warnings.append(
                    "selector is recorded but not applied in v1; full-page readable text was extracted"
                )
            artifact = write_web_artifact(self.workspace_dir, payload, run_id=run_id)
            content = json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)
            return self._record(
                call,
                "ok",
                content,
                warnings=[f"artifact: {artifact}", f"citation: {payload.citation_id}"] + payload.warnings,
            )
        except Exception as exc:
            return self._record(
                call,
                "error",
                f"{type(exc).__name__}: {exc}",
                warnings=["web_extract failed"],
            )

    def web_cite(
        self,
        citation_id: str = "",
        limit: int = 20,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "web_cite",
            {"citation_id": citation_id, "limit": limit},
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        citations = recent_citations(
            self.workspace_dir,
            citation_id_filter=citation_id,
            limit=limit,
        )
        return self._record(
            call,
            "ok",
            json.dumps({"citations": citations}, ensure_ascii=False, indent=2),
        )

    def image_generate(
        self,
        prompt: str,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call("image_generate", {"prompt": prompt}, run_id=run_id)
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        permission = self._check_permission(
            call,
            risk_level="image",
            reason="create an image-generation request artifact",
        )
        if permission is not None:
            return permission
        return self._image_request_artifact(call, "generate", {"prompt": prompt})

    def image_edit(
        self,
        image_path: str,
        instruction: str,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call(
            "image_edit",
            {"image_path": image_path, "instruction": instruction},
            run_id=run_id,
        )
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        permission = self._check_permission(
            call,
            risk_level="image",
            reason="create an image-edit request artifact",
        )
        if permission is not None:
            return permission
        return self._image_request_artifact(
            call,
            "edit",
            {"image_path": image_path, "instruction": instruction},
        )


    def research_pattern_search(
        self,
        query: str,
        field: str = "",
        top_k: int = 8,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._research_strategy_search("research_pattern", "research_pattern_search", query, field, top_k, run_id)

    def experiment_strategy_search(
        self,
        query: str,
        field: str = "",
        top_k: int = 8,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._research_strategy_search("experiment_strategy", "experiment_strategy_search", query, field, top_k, run_id)

    def storytelling_search(
        self,
        query: str,
        field: str = "",
        top_k: int = 8,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._research_strategy_search("scientific_storytelling", "storytelling_search", query, field, top_k, run_id)

    def research_taste_search(
        self,
        query: str,
        field: str = "",
        top_k: int = 8,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._research_strategy_search("research_taste", "research_taste_search", query, field, top_k, run_id)

    def extract_research_pattern(
        self,
        paper: str,
        provider: str = "",
        model: Optional[str] = None,
        field: str = "",
        write: bool = True,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._extract_research_section("research_pattern", "extract_research_pattern", paper, provider, model, field, write, run_id)

    def extract_experiment_strategy(
        self,
        paper: str,
        provider: str = "",
        model: Optional[str] = None,
        field: str = "",
        write: bool = True,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._extract_research_section("experiment_strategy", "extract_experiment_strategy", paper, provider, model, field, write, run_id)

    def extract_storytelling(
        self,
        paper: str,
        provider: str = "",
        model: Optional[str] = None,
        field: str = "",
        write: bool = True,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._extract_research_section("scientific_storytelling", "extract_storytelling", paper, provider, model, field, write, run_id)

    def extract_research_taste(
        self,
        paper: str,
        provider: str = "",
        model: Optional[str] = None,
        field: str = "",
        write: bool = True,
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        return self._extract_research_section("research_taste", "extract_research_taste", paper, provider, model, field, write, run_id)

    def paper_strategy_compare(
        self,
        papers: Sequence[str],
        provider: str = "",
        model: Optional[str] = None,
        field: str = "",
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call("paper_strategy_compare", {"papers": list(papers), "provider": provider, "model": model, "field": field}, run_id=run_id)
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        try:
            payload = self._research_agent().compare_paper_strategies(papers, provider=provider, model=model, field=field)
            return self._record(call, "ok", json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            return self._record(call, "error", f"{type(exc).__name__}: {exc}", warnings=["paper strategy comparison failed"])

    def compare_paper_strategies(self, *args, **kwargs) -> AgentToolResult:
        return self.paper_strategy_compare(*args, **kwargs)

    def research_experience_consolidate(
        self,
        topic: str,
        papers: Optional[Sequence[str]] = None,
        provider: str = "",
        model: Optional[str] = None,
        field: str = "",
        run_id: Optional[str] = None,
    ) -> AgentToolResult:
        call = self._call("research_experience_consolidate", {"topic": topic, "papers": list(papers or []), "provider": provider, "model": model, "field": field}, run_id=run_id)
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        permission = self._check_permission(call, risk_level="write", reason="write consolidated research-experience memory into HyperVault")
        if permission is not None:
            return permission
        try:
            payload = self._research_agent().consolidate_research_experience(topic, papers=papers, provider=provider, model=model, field=field)
            return self._record(call, "ok", json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            return self._record(call, "error", f"{type(exc).__name__}: {exc}", warnings=["research experience consolidation failed"])

    def consolidate_research_experience(self, *args, **kwargs) -> AgentToolResult:
        return self.research_experience_consolidate(*args, **kwargs)

    def _research_strategy_search(
        self,
        family: str,
        tool_name: str,
        query: str,
        field: str,
        top_k: int,
        run_id: Optional[str],
    ) -> AgentToolResult:
        call = self._call(tool_name, {"query": query, "field": field, "top_k": top_k}, run_id=run_id)
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        try:
            payload = {
                "family": family,
                "results": [
                    self._research_agent().search_dimension(dimension, query, top_k=top_k, field=field)
                    for dimension in self._research_dimensions(family)
                ],
            }
            return self._record(call, "ok", json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            return self._record(call, "error", f"{type(exc).__name__}: {exc}", warnings=["research strategy search failed"])

    def _extract_research_section(
        self,
        section_type: str,
        tool_name: str,
        paper: str,
        provider: str,
        model: Optional[str],
        field: str,
        write: bool,
        run_id: Optional[str],
    ) -> AgentToolResult:
        call = self._call(tool_name, {"paper": paper, "provider": provider, "model": model, "field": field, "write": write}, run_id=run_id)
        pre_hook = self._pre_tool_check(call)
        if pre_hook is not None:
            return pre_hook
        if write:
            permission = self._check_permission(call, risk_level="write", reason="write extracted paper strategy card into HyperVault")
            if permission is not None:
                return permission
        try:
            payload = self._research_agent().extract_section(section_type, paper, provider=provider, model=model, field=field, write=write)
            return self._record(call, "ok", json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            return self._record(call, "error", f"{type(exc).__name__}: {exc}", warnings=["research strategy extraction failed"])

    def _research_agent(self):
        from hyperagent.agents.research_experience_agent import ResearchExperienceAgent

        return ResearchExperienceAgent(self.project_root, self.workspace_dir)

    def _research_dimensions(self, family: str) -> List[str]:
        from hyperagent.schemas.research_experience import RESEARCH_EXPERIENCE_DIMENSIONS

        return list(RESEARCH_EXPERIENCE_DIMENSIONS.get(family, []))

    def _call(
        self,
        tool_name: str,
        args: dict,
        run_id: Optional[str] = None,
    ) -> AgentToolCall:
        return AgentToolCall(
            call_id=f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}",
            tool_name=tool_name,
            args=args,
            created_at=utc_now(),
            run_id=run_id,
        )

    def _check_permission(
        self,
        call: AgentToolCall,
        risk_level: str,
        reason: str,
    ) -> Optional[AgentToolResult]:
        if self.permission_policy == "auto" or risk_level == "read":
            return None
        if self.permission_policy == "deny-write" and risk_level not in {
            "write",
            "training",
        }:
            return None
        request = ToolPermissionRequest(
            tool_name=call.tool_name,
            args=call.args,
            risk_level=risk_level,
            reason=reason,
            run_id=call.run_id,
        )
        if self.permission_policy == "deny" or (
            self.permission_policy == "deny-write"
            and risk_level in {"write", "training"}
        ):
            return self._record(
                call,
                "blocked",
                f"Permission policy blocked {call.tool_name}: {reason}",
                warnings=[f"permission policy: {self.permission_policy}"],
            )
        if self.permission_policy in {"ask", "session-ask"}:
            cache_key = self._permission_cache_key(request)
            if (
                self.permission_policy == "session-ask"
                and self.session_permission_cache.get(cache_key)
            ):
                return None
            if self.permission_callback is None:
                return self._record(
                    call,
                    "blocked",
                    f"Permission required for {call.tool_name}, but no confirmation callback is configured.",
                    warnings=["permission confirmation is required"],
                )
            approved = bool(self.permission_callback(request))
            if not approved:
                return self._record(
                    call,
                    "blocked",
                    f"User denied permission for {call.tool_name}: {reason}",
                    warnings=["permission denied by user"],
                )
            if self.permission_policy == "session-ask":
                self.session_permission_cache[cache_key] = True
            return None
        return self._record(
            call,
            "blocked",
            f"Unknown permission policy: {self.permission_policy}",
            warnings=["invalid permission policy"],
        )

    def _image_request_artifact(
        self,
        call: AgentToolCall,
        action: str,
        payload: Dict[str, object],
    ) -> AgentToolResult:
        root = self.workspace_dir / "image_runs" / (call.run_id or call.call_id)
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"{action}.json"
        configured = bool(os.environ.get("OPENAI_API_KEY"))
        write_json(
            output,
            {
                "action": action,
                "created_at": utc_now(),
                "provider": "openai",
                "configured": configured,
                "request": payload,
                "note": (
                    "Image provider execution is not enabled in this local v1. "
                    "The request is saved for a user-approved image runtime."
                ),
            },
        )
        status = "ok" if configured else "blocked"
        message = (
            f"image request artifact: {output}"
            if configured
            else "OPENAI_API_KEY is not configured; image request was saved but not sent."
        )
        return self._record(
            call,
            status,
            message,
            warnings=[f"artifact: {output}", "image provider call not executed in v1"],
        )

    def _pre_tool_check(self, call: AgentToolCall) -> Optional[AgentToolResult]:
        if self.hook_engine is None:
            return None
        result = self.hook_engine.run(
            "PreToolUse",
            {
                "tool_name": call.tool_name,
                "args": call.args,
                "run_id": call.run_id or "",
            },
        )
        if result.blocked:
            return self._record(
                call,
                "blocked",
                "\n".join(result.warnings) or f"Hook blocked tool: {call.tool_name}",
                warnings=result.warnings
                + [f"hook matched: {rule}" for rule in result.matched_rules],
            )
        return None

    def _framework_command_payload(self, command: str, args: Sequence[str]) -> Dict[str, Any]:
        from hyperagent.runtime.commands import SlashCommandStore
        from hyperagent.runtime.conversations import ConversationStore
        from hyperagent.runtime.events import RuntimeEventLog
        from hyperagent.runtime.extensions import RuntimeExtensionStore
        from hyperagent.runtime.feature_state import (
            FeedbackStore,
            IDEContextStore,
            PersonalityStore,
            PlanModeStore,
            image_status,
            web_status,
            worktree_status,
        )
        from hyperagent.runtime.hooks import HookEngine
        from hyperagent.runtime.llm_usage import LLMUsageLedger
        from hyperagent.runtime.mcp import MCPServerStore
        from hyperagent.runtime.skills import SkillStore
        from hyperagent.runtime.todos import TodoStore
        from hyperagent.runtime.workspace import HyperAgentWorkspace

        command_text = " ".join([str(command or ""), *[str(arg) for arg in args]]).strip()
        normalized_text = command_text.lower().replace("_", "-").strip()
        alias_tokens = {
            "commands-supported": ["help"],
            "web-status": ["web", "status"],
            "image-status": ["image", "status"],
            "mcp-status": ["mcp", "status"],
            "ide-context-status": ["ide-context", "status"],
            "plan-mode-status": ["plan-mode", "status"],
            "personality-status": ["personality", "status"],
            "feedback-list": ["feedback", "list"],
            "skills-list": ["skills", "list"],
            "skills-search": ["skills", "search"],
            "commands-list": ["commands", "list"],
            "agents-list": ["agents", "list"],
            "hooks-list": ["hooks", "list"],
            "research-strategy-status": ["research", "strategy", "status"],
        }
        tokens = alias_tokens.get(normalized_text, normalized_text.split())
        if tokens and tokens[0] in alias_tokens:
            tokens = alias_tokens[tokens[0]] + tokens[1:]
        key = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "help")
        workspace = HyperAgentWorkspace(self.project_root)

        supported = [
            "help",
            "status",
            "usage",
            "cost",
            "stats",
            "web status",
            "image status",
            "ide-context status",
            "plan-mode status",
            "personality status",
            "feedback list",
            "worktree",
            "mcp status",
            "skills list",
            "skills search",
            "commands list",
            "todos",
            "sessions",
            "agents list",
            "hooks list",
            "research strategy status",
            "research status",
        ]

        if key in {"help", "commands"} and (not tokens or tokens[0] == "help"):
            return {"supported_framework_commands": supported}
        if key == "status":
            status = workspace.status()
            return {
                "initialized": status.initialized,
                "workspace": str(status.workspace_dir),
                "dataset_root": str(status.dataset_root),
                "task_count": status.task_count,
                "tasks_by_status": status.tasks_by_status,
            }
        if key in {"usage", "cost"}:
            return LLMUsageLedger(self.workspace_dir).summarize(limit=20)
        if key == "stats":
            return {
                "events": RuntimeEventLog(self.workspace_dir).summarize(),
                "llm_usage": LLMUsageLedger(self.workspace_dir).summarize(limit=20),
                "tools": tool_catalog(),
            }
        if key == "web status" or tokens[:1] == ["web"]:
            return web_status()
        if key == "image status" or tokens[:1] == ["image"]:
            return image_status()
        if key == "ide-context status" or tokens[:1] == ["ide-context"]:
            return IDEContextStore(self.workspace_dir).load()
        if key == "plan-mode status" or tokens[:1] == ["plan-mode"]:
            return PlanModeStore(self.workspace_dir).load()
        if key == "personality status" or tokens[:1] == ["personality"]:
            return PersonalityStore(self.workspace_dir).load()
        if key == "feedback list" or tokens[:1] == ["feedback"]:
            return {"feedback": FeedbackStore(self.workspace_dir).list(limit=20)}
        if key == "worktree":
            return worktree_status(self.project_root)
        if key == "mcp status" or tokens[:1] == ["mcp"]:
            servers = []
            for server in MCPServerStore(self.workspace_dir).list():
                servers.append(
                    {
                        "name": server.name,
                        "enabled": server.enabled,
                        "command": server.command,
                        "args": server.args,
                        "env_keys": sorted(server.env.keys()),
                        "description": server.description,
                        "runtime_client": "not_connected",
                        "health": "registered",
                    }
                )
            return {"servers": servers}

        if key in {"research strategy", "research status"} or tokens[:1] == ["research"]:
            from hyperagent.runtime.hypervault import HyperVaultClient
            from hyperagent.schemas.research_experience import RESEARCH_EXPERIENCE_DIMENSIONS

            status = HyperVaultClient().status().to_dict()
            return {
                "hypervault": status,
                "dimensions": RESEARCH_EXPERIENCE_DIMENSIONS,
                "tools": [
                    "research_pattern_search",
                    "experiment_strategy_search",
                    "storytelling_search",
                    "research_taste_search",
                    "extract_research_pattern",
                    "extract_experiment_strategy",
                    "extract_storytelling",
                    "extract_research_taste",
                    "paper_strategy_compare",
                    "research_experience_consolidate",
                ],
                "note": "HyperAgent extracts research strategy; HyperVault stores papers, strategy cards, and long-term research memory.",
            }

        if key in {"skills list", "skills search"} or tokens[:1] in (["skills"], ["skill"]):
            roots = [
                Path(__file__).resolve().parents[1] / "skills",
                self.project_root / "skills",
                self.workspace_dir / "skills",
            ]
            codex_home = os.environ.get("CODEX_HOME")
            if codex_home:
                roots.append(Path(codex_home) / "skills")
            else:
                roots.append(Path.home() / ".codex" / "skills")
            store = SkillStore(roots)
            query = " ".join(tokens[2:]) if key == "skills search" else ""
            skills = store.search(query) if query else store.list()
            return {
                "skills": [
                    {
                        "name": skill.name,
                        "description": skill.description,
                        "run_as": skill.run_as,
                        "allowed_tools": skill.allowed_tools,
                        "path": skill.path,
                    }
                    for skill in skills
                ]
            }
        if key == "commands list" or tokens[:1] == ["commands"]:
            commands = SlashCommandStore(self.project_root, self.workspace_dir).discover()
            return {
                "commands": [
                    {
                        "name": item.name,
                        "description": item.description,
                        "argument_hint": item.argument_hint,
                        "allowed_tools": item.allowed_tools,
                        "source": item.source,
                    }
                    for item in commands
                ]
            }
        if key == "todos" or tokens[:1] == ["todos"]:
            owner = tokens[1] if len(tokens) > 1 else "project"
            return TodoStore(self.workspace_dir).load(owner).to_dict()
        if key == "sessions":
            sessions = ConversationStore(self.workspace_dir).list(include_archived=True)
            return {
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "title": session.title,
                        "status": session.status,
                        "messages": len(session.messages),
                        "summaries": len(session.summaries),
                        "updated_at": session.updated_at,
                    }
                    for session in sessions[-20:]
                ]
            }
        if key == "agents list" or tokens[:1] == ["agents"]:
            return {
                "agents": [
                    {
                        "id": item.get("id", ""),
                        "name": item.get("name", ""),
                        "role": item.get("role", ""),
                        "description": item.get("description", ""),
                        "tools": item.get("tools", []),
                        "model": item.get("model", ""),
                        "profile": item.get("profile", ""),
                        "color": item.get("color", ""),
                        "source": item.get("source", ""),
                        "created_at": item.get("created_at", ""),
                    }
                    for item in RuntimeExtensionStore(self.workspace_dir).list_subagents()
                ]
            }
        if key == "hooks list" or tokens[:1] == ["hooks"]:
            return {
                "hooks": [
                    {
                        "id": rule.id,
                        "name": rule.name,
                        "event": rule.event,
                        "action": rule.action,
                        "message": rule.message,
                        "pattern": rule.pattern,
                        "tool_name": rule.tool_name,
                        "command_configured": bool(rule.command),
                        "enabled": rule.enabled,
                        "source": rule.source,
                    }
                    for rule in HookEngine(self.workspace_dir).list_rules()
                ]
            }
        raise KeyError(
            "Unsupported framework command. Supported commands: " + ", ".join(supported)
        )

    def _record(
        self,
        call: AgentToolCall,
        status: str,
        content: str,
        exit_code: Optional[int] = None,
        warnings: Optional[List[str]] = None,
    ) -> AgentToolResult:
        result = AgentToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=status,
            created_at=utc_now(),
            content=content,
            exit_code=exit_code,
            warnings=warnings or [],
        )
        self.tool_runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.tool_runs_dir / f"{call.call_id}-{call.tool_name}.json"
        result.artifact_path = str(path)
        write_json(path, {"call": call.to_dict(), "result": result.to_dict()})
        if self.hook_engine is not None:
            hook_result = self.hook_engine.run(
                "PostToolUse",
                {
                    "tool_name": call.tool_name,
                    "status": status,
                    "artifact_path": str(path),
                    "run_id": call.run_id or "",
                },
            )
            result.warnings.extend(hook_result.warnings)
        if self.event_log is not None:
            self.event_log.append(
                "tool.result",
                source="agent_tool",
                run_id=call.run_id,
                tool_name=call.tool_name,
                status=status,
                message=content[:500],
                payload={
                    "call_id": call.call_id,
                    "args": call.args,
                    "artifact_path": str(path),
                    "exit_code": exit_code,
                    "warnings": warnings or [],
                    "metadata": tool_metadata(call.tool_name).to_dict(),
                },
            )
        return result

    def _resolve_safe_path(self, path: str) -> Path:
        target = (self.project_root / path).resolve()
        try:
            target.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"Path escapes project root: {path}") from exc
        return target

    def _is_skipped(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.project_root)
        except ValueError:
            return True
        return any(part in SKIP_DIRS for part in relative.parts)

    def _command_allowed(self, argv: Sequence[str]) -> tuple:
        if not argv:
            return False, "Empty command"
        if self.allow_arbitrary_commands:
            return True, ""
        executable = Path(argv[0]).name
        if executable == "git":
            if len(argv) >= 2 and argv[1] in ALLOWED_GIT_SUBCOMMANDS:
                return True, ""
            return False, "Only selected read-only git subcommands are allowed"
        if executable.startswith("python"):
            if len(argv) >= 3 and argv[1] == "-m" and argv[2] in ALLOWED_PYTHON_MODULES:
                return True, ""
            return False, "Python commands must use -m with an allowed module"
        return False, f"Command executable is not allowed: {argv[0]}"

    def _join_process_output(self, stdout: str, stderr: str) -> str:
        parts = []
        if stdout:
            parts.append(stdout.rstrip())
        if stderr:
            parts.append("stderr:\n" + stderr.rstrip())
        return "\n\n".join(parts)

    def _permission_cache_key(self, request: ToolPermissionRequest) -> str:
        return f"{request.risk_level}:{request.tool_name}"
