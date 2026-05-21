"""File checkpoints for reversible agent edits."""

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from hyperagent.runtime.workspace import utc_now


@dataclass
class FileCheckpoint:
    checkpoint_id: str
    created_at: str
    reason: str
    project_root: str
    files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    manifest_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileCheckpoint":
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            created_at=str(data.get("created_at", "")),
            reason=str(data.get("reason", "")),
            project_root=str(data.get("project_root", "")),
            files=[str(v) for v in data.get("files", [])],
            missing_files=[str(v) for v in data.get("missing_files", [])],
            manifest_path=str(data.get("manifest_path", "")),
        )


class CheckpointStore:
    def __init__(self, project_root: Path, workspace_dir: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = Path(workspace_dir) / "checkpoints"

    def create(self, paths: Iterable[str], *, reason: str = "") -> FileCheckpoint:
        checkpoint_id = f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}"
        checkpoint_dir = self.root / checkpoint_id
        files_dir = checkpoint_dir / "files"
        files: List[str] = []
        missing: List[str] = []
        for raw in sorted({str(path) for path in paths if str(path).strip()}):
            try:
                relative = self._safe_relative(raw)
            except ValueError:
                missing.append(raw)
                continue
            target = self.project_root / relative
            if not target.exists() or not target.is_file():
                missing.append(str(relative))
                continue
            destination = files_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, destination)
            files.append(str(relative))
        checkpoint = FileCheckpoint(
            checkpoint_id=checkpoint_id,
            created_at=utc_now(),
            reason=reason,
            project_root=str(self.project_root),
            files=files,
            missing_files=missing,
            manifest_path=str(checkpoint_dir / "manifest.json"),
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "manifest.json").write_text(
            json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return checkpoint

    def list(self) -> List[FileCheckpoint]:
        if not self.root.exists():
            return []
        checkpoints: List[FileCheckpoint] = []
        for manifest in sorted(self.root.glob("*/manifest.json")):
            try:
                checkpoints.append(
                    FileCheckpoint.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return checkpoints

    def load(self, checkpoint_id: str) -> FileCheckpoint:
        manifest = self.root / checkpoint_id / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_id}")
        return FileCheckpoint.from_dict(json.loads(manifest.read_text(encoding="utf-8")))

    def restore(self, checkpoint_id: str) -> FileCheckpoint:
        checkpoint = self.load(checkpoint_id)
        files_dir = self.root / checkpoint_id / "files"
        for relative in checkpoint.files:
            source = files_dir / relative
            target = self.project_root / relative
            if not source.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return checkpoint

    def _safe_relative(self, path: str) -> Path:
        raw = Path(path)
        target = raw if raw.is_absolute() else self.project_root / raw
        resolved = target.resolve()
        try:
            return resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"path escapes project root: {path}") from exc


def paths_from_unified_diff(patch_text: str) -> List[str]:
    paths: List[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:].strip()
            if raw == "/dev/null":
                continue
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            if raw and raw not in paths:
                paths.append(raw)
    return paths
