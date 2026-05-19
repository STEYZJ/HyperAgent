"""Repository context collection for local coding-agent workflows."""

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import RepoFileContext, RepoSnapshot


SKIP_DIRS = {
    ".git",
    ".hyperagent",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "experiments",
    "reports",
    "dataset",
    "datasets",
    "data",
    "outputs",
    "checkpoints",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".sql",
    ".html",
    ".css",
    ".js",
    ".ts",
}

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "shell",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
}


class RepoContextBuilder:
    """Builds a compact, text-only repository snapshot."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def build(
        self,
        query: str = "",
        max_files: int = 20,
        max_preview_chars: int = 1200,
    ) -> RepoSnapshot:
        warnings: List[str] = []
        is_git_repo = self._git_ok()
        branch = self._git_text(["branch", "--show-current"]) if is_git_repo else ""
        commit = self._git_text(["rev-parse", "--short", "HEAD"]) if is_git_repo else ""
        dirty_files = self._dirty_files() if is_git_repo else []
        paths, path_warnings = self._candidate_paths(is_git_repo)
        warnings.extend(path_warnings)
        selected = self._select_paths(paths, query=query, limit=max_files)
        file_contexts = [
            self._file_context(path, max_preview_chars=max_preview_chars)
            for path in selected
        ]
        return RepoSnapshot(
            project_root=str(self.project_root),
            generated_at=utc_now(),
            query=query,
            is_git_repo=is_git_repo,
            branch=branch,
            commit=commit,
            dirty_files=dirty_files,
            file_count=len(paths),
            selected_files=file_contexts,
            warnings=warnings,
        )

    def to_markdown(self, snapshot: RepoSnapshot) -> str:
        lines = [
            "# Repository Context",
            "",
            f"- project_root: `{snapshot.project_root}`",
            f"- generated_at: `{snapshot.generated_at}`",
            f"- query: `{snapshot.query}`",
            f"- git: `{snapshot.is_git_repo}`",
            f"- branch: `{snapshot.branch}`",
            f"- commit: `{snapshot.commit}`",
            f"- tracked_or_visible_files: `{snapshot.file_count}`",
        ]
        if snapshot.dirty_files:
            lines.append("- dirty_files:")
            lines.extend(f"  - `{path}`" for path in snapshot.dirty_files[:50])
        if snapshot.warnings:
            lines.append("- warnings:")
            lines.extend(f"  - {warning}" for warning in snapshot.warnings)
        lines.append("")
        lines.append("## Selected Files")
        for file_context in snapshot.selected_files:
            lines.extend(
                [
                    "",
                    f"### `{file_context.path}`",
                    "",
                    f"- language: `{file_context.language}`",
                    f"- size_bytes: `{file_context.size_bytes}`",
                    "",
                    "```" + file_context.language,
                    file_context.preview,
                    "```",
                ]
            )
        return "\n".join(lines) + "\n"

    def _git_ok(self) -> bool:
        return self._run_git(["rev-parse", "--is-inside-work-tree"])[0] == 0

    def _git_text(self, args: Sequence[str]) -> str:
        code, stdout, _ = self._run_git(args)
        return stdout.strip() if code == 0 else ""

    def _dirty_files(self) -> List[str]:
        code, stdout, _ = self._run_git(["status", "--short"])
        if code != 0:
            return []
        return [line.strip() for line in stdout.splitlines() if line.strip()]

    def _candidate_paths(self, is_git_repo: bool) -> Tuple[List[Path], List[str]]:
        warnings: List[str] = []
        if is_git_repo:
            code, stdout, stderr = self._run_git(["ls-files"])
            if code == 0:
                return self._filter_paths(
                    self.project_root / line.strip()
                    for line in stdout.splitlines()
                    if line.strip()
                ), warnings
            warnings.append(f"git ls-files failed: {stderr.strip()[:240]}")
        return self._fallback_paths(), warnings

    def _fallback_paths(self) -> List[Path]:
        paths: List[Path] = []
        for path in self.project_root.rglob("*"):
            if path.is_file() and not self._is_skipped(path):
                paths.append(path)
        return self._filter_paths(paths)

    def _filter_paths(self, paths: Iterable[Path]) -> List[Path]:
        filtered = []
        for path in paths:
            if self._is_skipped(path) or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > 500_000:
                    continue
            except OSError:
                continue
            filtered.append(path)
        return sorted(filtered, key=lambda value: str(value.relative_to(self.project_root)))

    def _is_skipped(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.project_root)
        except ValueError:
            return True
        return any(part in SKIP_DIRS for part in relative.parts)

    def _select_paths(self, paths: List[Path], query: str, limit: int) -> List[Path]:
        terms = [part.lower() for part in query.replace("/", " ").split() if part]

        def score(path: Path) -> Tuple[int, str]:
            relative = str(path.relative_to(self.project_root)).lower()
            value = 0
            for term in terms:
                if term in relative:
                    value += 10
            if relative.startswith("hyperagent/runtime"):
                value += 4
            if relative.startswith("hyperagent/schemas"):
                value += 3
            if relative.startswith("hyperagent/tools") or relative.startswith("hyperagent/agents"):
                value += 2
            if relative.startswith("tests"):
                value += 2
            if path.name in {"README.md", "pyproject.toml", "environment.yml"}:
                value += 1
            return (-value, relative)

        return sorted(paths, key=score)[: max(limit, 1)]

    def _file_context(self, path: Path, max_preview_chars: int) -> RepoFileContext:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            text = f"<unreadable: {exc}>"
        if len(text) > max_preview_chars:
            text = text[:max_preview_chars] + "\n...<truncated>"
        relative = path.relative_to(self.project_root)
        suffix = path.suffix.lower()
        return RepoFileContext(
            path=str(relative),
            size_bytes=size_bytes,
            language=LANGUAGE_BY_SUFFIX.get(suffix, suffix.lstrip(".") or "text"),
            preview=text,
        )

    def _run_git(self, args: Sequence[str]) -> Tuple[int, str, str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(self.project_root),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 1, "", str(exc)
        return completed.returncode, completed.stdout, completed.stderr
