"""Project and user memory helpers for HyperAgent."""

from pathlib import Path
from typing import List


PROJECT_MEMORY = "HyperAgent.md"
USER_MEMORY = "USER_MEMORY.md"
AUTO_MEMORY = "AUTO_MEMORY.md"


class MemoryStore:
    def __init__(self, project_root: Path, workspace_dir: Path) -> None:
        self.project_root = project_root.resolve()
        self.workspace_dir = workspace_dir.resolve()
        self.memory_dir = self.workspace_dir / "memory"

    def ensure_project_memory(self) -> Path:
        path = self.project_root / PROJECT_MEMORY
        if not path.exists():
            path.write_text(self._project_template(), encoding="utf-8")
        return path

    def append(self, scope: str, text: str) -> Path:
        path = self._path(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text.strip() + "\n")
        return path

    def read(self, scope: str) -> str:
        path = self._path(scope)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def list(self) -> List[str]:
        items = []
        for scope in ("project", "user", "auto"):
            path = self._path(scope)
            items.append(f"{scope}\t{path}\t{'exists' if path.exists() else 'missing'}")
        return items

    def _path(self, scope: str) -> Path:
        normalized = scope.strip().lower()
        if normalized == "project":
            return self.project_root / PROJECT_MEMORY
        if normalized == "user":
            return self.memory_dir / USER_MEMORY
        if normalized == "auto":
            return self.memory_dir / AUTO_MEMORY
        raise ValueError(f"Unknown memory scope: {scope}")

    def _project_template(self) -> str:
        return """# HyperAgent Project Memory

## Project Rules
- Use `HyperAgent` as the user-facing command prefix.
- Keep HSI data/model/training/evaluation modules decoupled from runtime UI.
- Record meaningful work in `logs/worklog/`.

## Research Context
- Goal: autonomous multi-agent framework for hyperspectral image classification research.
- Default dataset root: `/data2/lzj/lab/Mamba_test/dataset`.

## Preferred Workflow
1. Inspect artifacts and prior conversation.
2. Propose an evidence-backed plan.
3. Ask for permission before risky local actions.
4. Run focused tests.
5. Commit by task branch and push.
"""
