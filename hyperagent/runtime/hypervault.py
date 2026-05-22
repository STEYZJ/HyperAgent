"""Thin HyperVault client used by HyperAgent research-experience tools.

The client talks to HyperVault through HTTP when the API is available and uses
a local vault filesystem fallback for offline tests and notes writing. It never
imports HyperVault Python modules, preserving the framework/vault boundary.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class HyperVaultHit:
    chunk_id: str
    file_path: str
    title: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    heading_path: List[str] = field(default_factory=list)
    is_memory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperVaultHit":
        heading = data.get("heading_path", [])
        return cls(
            chunk_id=str(data.get("chunk_id", "")),
            file_path=str(data.get("file_path", data.get("path", ""))),
            title=str(data.get("title", "")),
            text=str(data.get("text", data.get("content", ""))),
            score=float(data.get("score", 0.0) or 0.0),
            metadata=dict(data.get("metadata", {})),
            heading_path=[str(item) for item in heading] if isinstance(heading, list) else [],
            is_memory=bool(data.get("is_memory", False)),
        )


@dataclass
class HyperVaultStatus:
    api_url: str
    api_available: bool
    vault_path: str
    vault_exists: bool
    strategy_dir: str
    memory_dir: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HyperVaultClient:
    def __init__(
        self,
        api_url: Optional[str] = None,
        vault_path: Optional[Path] = None,
        timeout_sec: int = 20,
    ) -> None:
        root = Path(os.environ.get("HYPERVAULT_ROOT", "/data2/lzj/HyperVault"))
        configured_vault = os.environ.get("HYPERVAULT_VAULT_PATH")
        self.api_url = (api_url or os.environ.get("HYPERVAULT_URL") or "http://127.0.0.1:8088").rstrip("/")
        self.vault_path = (vault_path or Path(configured_vault) if configured_vault else vault_path or root / "knowledge-vault").resolve()
        self.timeout_sec = timeout_sec

    @property
    def strategy_dir(self) -> Path:
        return self.vault_path / "summaries" / "research-strategies"

    @property
    def research_memory_dir(self) -> Path:
        return self.vault_path / "memory" / "research-experience"

    def status(self) -> HyperVaultStatus:
        warnings: List[str] = []
        api_available = False
        try:
            self._request_json("GET", "/health")
            api_available = True
        except Exception as exc:
            warnings.append("HyperVault API unavailable: %s" % exc)
        return HyperVaultStatus(
            api_url=self.api_url,
            api_available=api_available,
            vault_path=str(self.vault_path),
            vault_exists=self.vault_path.exists(),
            strategy_dir=str(self.strategy_dir),
            memory_dir=str(self.research_memory_dir),
            warnings=warnings,
        )

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        memory: bool = False,
    ) -> List[HyperVaultHit]:
        endpoint = "/memory/search" if memory else "/search"
        payload = {"query": query, "top_k": top_k, "filters": filters or {}}
        try:
            data = self._request_json("POST", endpoint, payload)
            return [HyperVaultHit.from_dict(item) for item in data.get("hits", [])]
        except Exception:
            return self._local_search(query, top_k=top_k, filters=filters, memory=memory)

    def write_markdown(self, relative_path: str, content: str) -> Path:
        target = self._resolve_vault_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def read_markdown(self, path_or_id: str) -> Optional[tuple]:
        candidates: List[Path] = []
        raw = Path(path_or_id)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append((Path.cwd() / raw).resolve())
            candidates.append((self.vault_path / raw).resolve())
            candidates.extend(self.vault_path.rglob("%s.md" % path_or_id))
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.exists() and resolved.is_file() and resolved.suffix.lower() == ".md":
                try:
                    relative = str(resolved.relative_to(self.vault_path))
                except ValueError:
                    relative = str(resolved)
                return relative, resolved.read_text(encoding="utf-8", errors="replace")
        return None

    def list_strategy_cards(self) -> List[Path]:
        roots = [self.strategy_dir, self.research_memory_dir]
        paths: List[Path] = []
        for root in roots:
            if root.exists():
                paths.extend(sorted(root.rglob("*.md")))
        return paths

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.api_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError("HTTP %s: %s" % (exc.code, detail))
        except URLError as exc:
            raise RuntimeError(str(exc))
        if not isinstance(data, dict):
            raise RuntimeError("HyperVault response root is not an object")
        return data

    def _resolve_vault_path(self, relative_path: str) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute():
            raise ValueError("vault write path must be relative")
        resolved = (self.vault_path / raw).resolve()
        if self.vault_path not in resolved.parents and resolved != self.vault_path:
            raise ValueError("vault write path escapes knowledge-vault")
        if resolved.suffix.lower() != ".md":
            raise ValueError("vault write path must end with .md")
        return resolved

    def _local_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        memory: bool,
    ) -> List[HyperVaultHit]:
        if not self.vault_path.exists():
            return []
        terms = [term for term in re.split(r"\W+", query.lower()) if term]
        root = self.vault_path / "memory" if memory else self.vault_path
        scored: List[HyperVaultHit] = []
        for path in root.rglob("*.md"):
            if any(part.startswith(".") for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            metadata, body = parse_markdown_metadata(text)
            try:
                relative = str(path.relative_to(self.vault_path))
            except ValueError:
                relative = str(path)
            if not metadata_matches(relative, metadata, filters or {}):
                continue
            haystack = (relative + "\n" + body).lower()
            score = sum(haystack.count(term) for term in terms) if terms else 1
            if score <= 0:
                continue
            title = str(metadata.get("title") or infer_title(body, path))
            scored.append(
                HyperVaultHit(
                    chunk_id="local:%s" % relative,
                    file_path=relative,
                    title=title,
                    text=body[:5000],
                    score=float(score),
                    metadata=metadata,
                    heading_path=[],
                    is_memory="/memory/" in ("/" + relative),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[: max(top_k, 1)]


def parse_markdown_metadata(text: str) -> tuple:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    metadata: Dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if value.startswith("[") and value.endswith("]"):
            metadata[key] = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
        elif value.lower() in {"true", "false"}:
            metadata[key] = value.lower() == "true"
        else:
            metadata[key] = value
    return metadata, text[match.end():]


def infer_title(body: str, path: Path) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def metadata_matches(file_path: str, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    if filters.get("type") and metadata.get("type") != filters.get("type"):
        return False
    if filters.get("field") and metadata.get("field") != filters.get("field"):
        return False
    if filters.get("verified") is not None and bool(metadata.get("verified")) != bool(filters.get("verified")):
        return False
    if filters.get("is_memory") is not None:
        is_memory = "/memory/" in ("/" + file_path)
        if is_memory != bool(filters.get("is_memory")):
            return False
    dimension = filters.get("dimension") or filters.get("strategy_dimension")
    if dimension:
        dims = metadata.get("strategy_dimensions") or metadata.get("dimensions") or []
        if isinstance(dims, str):
            dims = [dims]
        if str(dimension) not in {str(item) for item in dims}:
            return False
    return True
