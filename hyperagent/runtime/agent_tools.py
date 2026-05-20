"""Controlled local tools for Claude-Code-like agent workflows."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.runtime.repo_context import SKIP_DIRS, TEXT_SUFFIXES
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import AgentToolCall, AgentToolResult


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


class SafeAgentToolExecutor:
    """Executes a small set of auditable local tools."""

    def __init__(
        self,
        project_root: Path,
        workspace_dir: Path,
        permission_policy: str = "auto",
        permission_callback: Optional[Callable[[ToolPermissionRequest], bool]] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.workspace_dir = workspace_dir.resolve()
        self.tool_runs_dir = self.workspace_dir / "tool_runs"
        self.permission_policy = permission_policy
        self.permission_callback = permission_callback

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
        allowed, reason = self._command_allowed(argv)
        if not allowed:
            return self._record(
                call,
                "blocked",
                reason,
                warnings=["command is outside the HyperAgent agent-tool allowlist"],
            )
        permission = self._check_permission(
            call,
            risk_level="execute",
            reason="run a local allowlisted command",
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
        if not patch_text.strip():
            return self._record(
                call,
                "error",
                "Empty patch",
                warnings=[f"{tool_name} requires a non-empty unified diff"],
            )
        if apply:
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
        request = ToolPermissionRequest(
            tool_name=call.tool_name,
            args=call.args,
            risk_level=risk_level,
            reason=reason,
            run_id=call.run_id,
        )
        if self.permission_policy == "deny" or (
            self.permission_policy == "deny-write" and risk_level == "write"
        ):
            return self._record(
                call,
                "blocked",
                f"Permission policy blocked {call.tool_name}: {reason}",
                warnings=[f"permission policy: {self.permission_policy}"],
            )
        if self.permission_policy == "ask":
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
            return None
        return self._record(
            call,
            "blocked",
            f"Unknown permission policy: {self.permission_policy}",
            warnings=["invalid permission policy"],
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
