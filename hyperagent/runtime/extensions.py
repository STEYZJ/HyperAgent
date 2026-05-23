"""Lightweight registries for subagents, hooks, plugins, and rewind snapshots."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import yaml

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.workspace import utc_now


@dataclass
class PluginBundle:
    id: str
    name: str
    description: str = ""
    enabled: bool = True
    skills: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    mcp: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    source: str = ""
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, object], *, source: str = "") -> "PluginBundle":
        return cls(
            id=str(data.get("id") or Path(source).parent.name),
            name=str(data.get("name") or data.get("id") or Path(source).parent.name),
            description=str(data.get("description", "")),
            enabled=bool(data.get("enabled", True)),
            skills=_string_list(data.get("skills", [])),
            agents=_string_list(data.get("agents", [])),
            hooks=_string_list(data.get("hooks", [])),
            mcp=_string_list(data.get("mcp", [])),
            commands=_string_list(data.get("commands", [])),
            metadata=dict(data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}),
            source=source,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class PluginBundleStore:
    """Scans local plugin bundle manifests without executing them."""

    def __init__(self, workspace_dir: Path, project_root: Optional[Path] = None) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.project_root = Path(project_root) if project_root is not None else self.workspace_dir.parent

    def list(self) -> List[PluginBundle]:
        bundles: List[PluginBundle] = []
        for root in self._roots():
            if not root.exists():
                continue
            for path in sorted(root.glob("*/bundle.json")):
                try:
                    bundle = PluginBundle.from_dict(read_json(path), source=str(path))
                except Exception as exc:
                    bundle = PluginBundle(
                        id=path.parent.name,
                        name=path.parent.name,
                        enabled=False,
                        source=str(path),
                        warnings=[f"{type(exc).__name__}: {exc}"],
                    )
                bundle.warnings.extend(self.validate(bundle))
                bundles.append(bundle)
        return sorted(bundles, key=lambda item: (item.name.lower(), item.id))

    def inspect(self, id_or_name: str) -> Optional[PluginBundle]:
        needle = str(id_or_name).strip().lower()
        for bundle in self.list():
            if bundle.id.lower() == needle or bundle.name.lower() == needle:
                return bundle
        return None

    def validate(self, bundle: PluginBundle) -> List[str]:
        warnings: List[str] = []
        if not bundle.id:
            warnings.append("bundle id is missing")
        if not bundle.name:
            warnings.append("bundle name is missing")
        if not any([bundle.skills, bundle.agents, bundle.hooks, bundle.mcp, bundle.commands]):
            warnings.append("bundle has no skills, agents, hooks, mcp, or commands")
        return warnings

    def summary(self) -> Dict[str, object]:
        bundles = self.list()
        return {
            "total": len(bundles),
            "enabled": sum(1 for bundle in bundles if bundle.enabled),
            "disabled": sum(1 for bundle in bundles if not bundle.enabled),
            "with_warnings": sum(1 for bundle in bundles if bundle.warnings),
            "bundles": [bundle.to_dict() for bundle in bundles],
        }

    def _roots(self) -> List[Path]:
        return [
            self.workspace_dir / "plugins",
            self.project_root / "plugins",
        ]


class RuntimeExtensionStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.root = workspace_dir / "runtime_extensions"
        self.subagents_path = self.root / "subagents.json"
        self.hooks_path = self.root / "hooks.json"
        self.plugins_path = self.root / "plugins.json"
        self.rewind_dir = workspace_dir / "rewind"

    def list_subagents(self) -> List[Dict[str, object]]:
        items = self._list(self.subagents_path, "subagents")
        items.extend(self._markdown_subagents())
        return items

    def add_subagent(
        self,
        name: str,
        role: str,
        tools: Optional[List[str]] = None,
        model: str = "",
        profile: str = "",
        color: str = "",
    ) -> Dict[str, object]:
        item = {
            "id": self._new_id("agent"),
            "name": name,
            "role": role,
            "tools": tools or [],
            "model": model,
            "profile": profile,
            "color": color,
            "created_at": utc_now(),
        }
        items = self.list_subagents()
        items.append(item)
        self._write(self.subagents_path, "subagents", items)
        return item

    def list_hooks(self) -> List[Dict[str, object]]:
        return self._list(self.hooks_path, "hooks")

    def add_hook(
        self,
        name: str,
        event: str,
        command: str,
        enabled: bool = True,
    ) -> Dict[str, object]:
        item = {
            "id": self._new_id("hook"),
            "name": name,
            "event": event,
            "command": command,
            "enabled": enabled,
            "created_at": utc_now(),
        }
        items = self.list_hooks()
        items.append(item)
        self._write(self.hooks_path, "hooks", items)
        return item

    def list_plugins(self) -> List[Dict[str, object]]:
        return self._list(self.plugins_path, "plugins")

    def add_plugin(
        self,
        name: str,
        description: str = "",
        source: str = "",
        enabled: bool = True,
    ) -> Dict[str, object]:
        item = {
            "id": self._new_id("plugin"),
            "name": name,
            "description": description,
            "source": source,
            "enabled": enabled,
            "created_at": utc_now(),
        }
        items = self.list_plugins()
        items.append(item)
        self._write(self.plugins_path, "plugins", items)
        return item

    def create_rewind_snapshot(self, session_id: str, payload: Dict[str, object]) -> Path:
        self.rewind_dir.mkdir(parents=True, exist_ok=True)
        path = self.rewind_dir / f"{utc_now().replace(':', '').replace('-', '')}-{session_id}.json"
        write_json(path, payload)
        return path

    def list_rewind_snapshots(self) -> List[Path]:
        if not self.rewind_dir.exists():
            return []
        return sorted(self.rewind_dir.glob("*.json"))

    def _list(self, path: Path, key: str) -> List[Dict[str, object]]:
        if not path.exists():
            return []
        data = read_json(path)
        return [dict(item) for item in data.get(key, []) if isinstance(item, dict)]

    def _write(self, path: Path, key: str, items: List[Dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {key: items})

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid4().hex[:8]}"

    def _markdown_subagents(self) -> List[Dict[str, object]]:
        roots = [Path(__file__).resolve().parents[1] / "agent_definitions", self.root.parent / "agents"]
        plugins_root = self.root.parent / "plugins"
        if plugins_root.exists():
            roots.extend(path / "agents" for path in sorted(plugins_root.iterdir()))
        items: List[Dict[str, object]] = []
        for agents_dir in roots:
            if not agents_dir.exists():
                continue
            for path in sorted(agents_dir.glob("*.md")):
                text = path.read_text(encoding="utf-8")
                metadata: Dict[str, object] = {}
                body = text
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        raw = yaml.safe_load(parts[1]) or {}
                        metadata = raw if isinstance(raw, dict) else {}
                        body = parts[2].strip()
                tools = metadata.get("tools", [])
                if isinstance(tools, str):
                    tools = [item.strip() for item in tools.split(",") if item.strip()]
                item = {
                    "id": str(metadata.get("id") or f"md-agent-{path.stem}"),
                    "name": str(metadata.get("name") or path.stem),
                    "role": str(metadata.get("role") or metadata.get("description") or path.stem),
                    "description": str(metadata.get("description", "")),
                    "tools": tools if isinstance(tools, list) else [],
                    "model": str(metadata.get("model", "")),
                    "profile": str(metadata.get("profile", "")),
                    "color": str(metadata.get("color", "")),
                    "prompt": body,
                    "source": str(path),
                    "created_at": "",
                }
                items.append(item)
        return items


def _string_list(value: object) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []
