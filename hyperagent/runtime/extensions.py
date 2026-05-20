"""Lightweight registries for subagents, hooks, plugins, and rewind snapshots."""

from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.workspace import utc_now


class RuntimeExtensionStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.root = workspace_dir / "runtime_extensions"
        self.subagents_path = self.root / "subagents.json"
        self.hooks_path = self.root / "hooks.json"
        self.plugins_path = self.root / "plugins.json"
        self.rewind_dir = workspace_dir / "rewind"

    def list_subagents(self) -> List[Dict[str, object]]:
        return self._list(self.subagents_path, "subagents")

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
