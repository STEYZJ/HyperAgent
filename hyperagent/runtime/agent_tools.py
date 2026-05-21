"""Controlled local tools for Claude-Code-like agent workflows."""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import uuid4

from hyperagent.core.io import read_yaml, write_json
from hyperagent.runtime.checkpoints import CheckpointStore, paths_from_unified_diff
from hyperagent.runtime.events import RuntimeEventLog
from hyperagent.runtime.repo_context import SKIP_DIRS, TEXT_SUFFIXES
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
