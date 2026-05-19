"""Obsidian vault indexing and search."""

import re
from pathlib import Path
from typing import List

from hyperagent.core.io import read_json, write_json
from hyperagent.schemas import ObsidianNote


TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_\-/]+)")
LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


class ObsidianVaultIndex:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = workspace_dir / "obsidian_index.json"

    def index(self, vault: Path) -> List[ObsidianNote]:
        notes: List[ObsidianNote] = []
        for path in sorted(Path(vault).rglob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            title = self._title(path, text)
            preview = " ".join(text.split())[:280]
            notes.append(
                ObsidianNote(
                    path=str(path),
                    title=title,
                    tags=sorted(set(TAG_RE.findall(text))),
                    links=sorted(set(LINK_RE.findall(text))),
                    preview=preview,
                )
            )
        write_json(self.path, {"vault": str(Path(vault).resolve()), "notes": [n.to_dict() for n in notes]})
        return notes

    def search(self, query: str, limit: int = 10) -> List[ObsidianNote]:
        if not self.path.exists():
            return []
        data = read_json(self.path)
        terms = [term.lower() for term in query.split() if term.strip()]
        scored = []
        for item in data.get("notes", []):
            note = ObsidianNote.from_dict(item)
            haystack = " ".join([note.title, note.preview, " ".join(note.tags)]).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, note))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [note for _, note in scored[:limit]]

    def _title(self, path: Path, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return path.stem

