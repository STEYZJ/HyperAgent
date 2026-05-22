"""Small persisted state stores for UI feature toggles."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.workspace import utc_now
from hyperagent.runtime.web_tools import search_provider_status


class IDEContextStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "ide_context.json"

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {
                "enabled": False,
                "open_files": [],
                "selected_text_summary": "",
                "updated_at": "",
            }
        return read_json(self.path)

    def save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(data)
        payload["updated_at"] = utc_now()
        write_json(self.path, payload)
        return payload

    def set_enabled(self, enabled: bool) -> Dict[str, Any]:
        data = self.load()
        data["enabled"] = bool(enabled)
        return self.save(data)

    def set_open_files(self, paths: Iterable[str]) -> Dict[str, Any]:
        data = self.load()
        data["open_files"] = [str(path) for path in paths if str(path).strip()]
        data["enabled"] = True
        return self.save(data)

    def clear(self) -> Dict[str, Any]:
        data = {
            "enabled": False,
            "open_files": [],
            "selected_text_summary": "",
            "updated_at": utc_now(),
        }
        return self.save(data)


class PlanModeStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "plan_mode.json"

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"enabled": False, "reason": "", "updated_at": ""}
        return read_json(self.path)

    def set_enabled(self, enabled: bool, reason: str = "") -> Dict[str, Any]:
        data = self.load()
        data["enabled"] = bool(enabled)
        if reason:
            data["reason"] = reason
        data["updated_at"] = utc_now()
        write_json(self.path, data)
        return data


class PersonalityStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "personality.json"

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"text": "", "updated_at": ""}
        return read_json(self.path)

    def set(self, text: str) -> Dict[str, Any]:
        data = {"text": str(text).strip(), "updated_at": utc_now()}
        write_json(self.path, data)
        return data

    def clear(self) -> Dict[str, Any]:
        data = {"text": "", "updated_at": utc_now()}
        write_json(self.path, data)
        return data


class FeedbackStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "feedback.jsonl"

    def add(self, text: str, source: str = "cli") -> Dict[str, Any]:
        item = {"created_at": utc_now(), "source": source, "text": str(text).strip()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        return item

    def list(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        items: List[Dict[str, Any]] = []
        for line in lines[-max(int(limit), 1) :]:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = {"created_at": "", "source": "decode_error", "text": line}
            if isinstance(item, dict):
                items.append(item)
        return items


def web_status() -> Dict[str, Any]:
    providers = search_provider_status()
    return {
        "providers": providers,
        "search_configured": any(providers.values()),
        "fetch_available": True,
        "policy": {
            "allowed_schemes": ["http", "https"],
            "blocked_schemes": ["file", "data", "javascript", "ftp", "ssh"],
            "blocks_private_hosts": True,
        },
    }


def image_status() -> Dict[str, Any]:
    provider = os.environ.get("HYPERAGENT_IMAGE_PROVIDER", "openai")
    required_env = "OPENAI_API_KEY" if provider == "openai" else "HYPERAGENT_IMAGE_API_KEY"
    return {
        "provider": provider,
        "required_env": required_env,
        "configured": bool(os.environ.get(required_env)),
        "output_root": ".hyperagent/image_runs",
    }


def worktree_status(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root)
    return {
        "branch": _git(root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "head": _git(root, ["rev-parse", "--short", "HEAD"]).strip(),
        "dirty_files": _git(root, ["status", "--short"]).splitlines(),
        "recent_commits": _git(root, ["log", "--oneline", "-5"]).splitlines(),
    }


def _git(root: Path, args: List[str]) -> str:
    try:
        completed = subprocess.run(
            ["git"] + args,
            cwd=str(root),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"
    return completed.stdout if completed.returncode == 0 else completed.stderr
