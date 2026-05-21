"""Lightweight local semantic-index placeholder using lexical scoring.

This keeps the public workflow in place without adding embedding dependencies.
Embedding-backed providers can later replace the scoring implementation behind
the same index file shape.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from hyperagent.runtime.repo_context import SKIP_DIRS, TEXT_SUFFIXES
from hyperagent.runtime.workspace import utc_now


@dataclass
class IndexDocument:
    path: str
    title: str = ""
    preview: str = ""
    tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexDocument":
        return cls(
            path=str(data.get("path", "")),
            title=str(data.get("title", "")),
            preview=str(data.get("preview", "")),
            tokens=[str(v) for v in data.get("tokens", [])],
        )


class SemanticIndexStore:
    def __init__(self, project_root: Path, workspace_dir: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = Path(workspace_dir) / "semantic_index"
        self.path = self.root / "index.json"

    def build(self, roots: Iterable[str], *, max_file_bytes: int = 200_000) -> Dict[str, Any]:
        docs: List[IndexDocument] = []
        for raw_root in roots:
            root = self._resolve(raw_root)
            candidates = [root] if root.is_file() else root.rglob("*")
            for path in candidates:
                if not path.is_file() or self._skip(path):
                    continue
                if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix.lower() not in {".md", ".json", ".yaml", ".yml"}:
                    continue
                try:
                    if path.stat().st_size > max_file_bytes:
                        continue
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                relative = str(path.relative_to(self.project_root))
                docs.append(
                    IndexDocument(
                        path=relative,
                        title=self._title(text, path),
                        preview=text.strip()[:500],
                        tokens=sorted(set(_tokens(text)))[:2000],
                    )
                )
        payload = {
            "generated_at": utc_now(),
            "project_root": str(self.project_root),
            "documents": [doc.to_dict() for doc in docs],
            "engine": "lexical-v1",
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def search(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        query_tokens = set(_tokens(query))
        scored: List[Dict[str, Any]] = []
        for item in payload.get("documents", []):
            doc = IndexDocument.from_dict(dict(item))
            overlap = query_tokens.intersection(doc.tokens)
            if not overlap:
                continue
            scored.append(
                {
                    "path": doc.path,
                    "title": doc.title,
                    "score": len(overlap),
                    "matched_terms": sorted(overlap),
                    "preview": doc.preview,
                }
            )
        return sorted(scored, key=lambda item: (-int(item["score"]), item["path"]))[:limit]

    def _resolve(self, raw: str) -> Path:
        path = Path(raw)
        target = path if path.is_absolute() else self.project_root / path
        target = target.resolve()
        target.relative_to(self.project_root)
        return target

    def _skip(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.project_root)
        except ValueError:
            return True
        return any(part in SKIP_DIRS or part.startswith(".git") for part in relative.parts)

    def _title(self, text: str, path: Path) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.strip("# ").strip()[:120]
        return path.name


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]{2,}", text)]
