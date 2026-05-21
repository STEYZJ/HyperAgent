"""Lightweight hook engine for Claude-Code-like runtime events."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.workspace import utc_now


HOOK_EVENTS = {
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
}


@dataclass
class HookRule:
    id: str
    name: str
    event: str
    action: str = "warn"
    message: str = ""
    pattern: str = ""
    tool_name: str = ""
    command: str = ""
    enabled: bool = True
    source: str = "registry"

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "event": self.event,
            "action": self.action,
            "message": self.message,
            "pattern": self.pattern,
            "tool_name": self.tool_name,
            "command": self.command,
            "enabled": self.enabled,
            "source": self.source,
        }


@dataclass
class HookEventResult:
    event: str
    blocked: bool = False
    warnings: List[str] = field(default_factory=list)
    system_messages: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)


class HookEngine:
    """Evaluates registry and Markdown hooks without coupling to REPL/TUI."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.registry_path = workspace_dir / "runtime_extensions" / "hooks.json"
        self.hooks_dir = workspace_dir / "hooks"

    def list_rules(self) -> List[HookRule]:
        rules: List[HookRule] = []
        rules.extend(self._registry_rules())
        rules.extend(self._markdown_rules())
        return sorted(rules, key=lambda item: (item.event, item.name))

    def set_enabled(self, rule_id_or_name: str, enabled: bool) -> bool:
        if not self.registry_path.exists():
            return False
        data = read_json(self.registry_path)
        items = [dict(item) for item in data.get("hooks", []) if isinstance(item, dict)]
        matched = False
        for item in items:
            if item.get("id") == rule_id_or_name or item.get("name") == rule_id_or_name:
                item["enabled"] = bool(enabled)
                matched = True
        if matched:
            write_json(self.registry_path, {"hooks": items})
        return matched

    def run(self, event: str, payload: Optional[Dict[str, object]] = None) -> HookEventResult:
        normalized = self._normalize_event(event)
        result = HookEventResult(event=normalized)
        payload = payload or {}
        for rule in self.list_rules():
            if not rule.enabled or self._normalize_event(rule.event) != normalized:
                continue
            if not self._matches(rule, payload):
                continue
            result.matched_rules.append(rule.id)
            message = rule.message or rule.command or f"hook matched: {rule.name}"
            if rule.action == "block":
                result.blocked = True
                result.warnings.append(message)
            elif rule.action == "systemMessage":
                result.system_messages.append(message)
            elif rule.action == "runCommand":
                result.warnings.append(
                    f"runCommand hook matched but is not executed automatically: {rule.name}"
                )
            else:
                result.warnings.append(message)
        if result.matched_rules:
            self._record_event(normalized, payload, result)
        return result

    def _registry_rules(self) -> List[HookRule]:
        if not self.registry_path.exists():
            return []
        data = read_json(self.registry_path)
        rules: List[HookRule] = []
        for item in data.get("hooks", []):
            if not isinstance(item, dict):
                continue
            event = self._normalize_event(str(item.get("event", "")))
            rules.append(
                HookRule(
                    id=str(item.get("id", item.get("name", ""))),
                    name=str(item.get("name", item.get("id", ""))),
                    event=event,
                    action=str(item.get("action", "warn")),
                    message=str(item.get("message", item.get("command", ""))),
                    pattern=str(item.get("pattern", "")),
                    tool_name=str(item.get("tool_name", item.get("tool", ""))),
                    command=str(item.get("command", "")),
                    enabled=bool(item.get("enabled", True)),
                    source="registry",
                )
            )
        return rules

    def _markdown_rules(self) -> List[HookRule]:
        if not self.hooks_dir.exists():
            return []
        rules: List[HookRule] = []
        for path in sorted(self.hooks_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            metadata: Dict[str, object] = {}
            body = text
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    raw = yaml.safe_load(parts[1]) or {}
                    metadata = raw if isinstance(raw, dict) else {}
                    body = parts[2]
            rules.append(
                HookRule(
                    id=str(metadata.get("id") or path.stem),
                    name=str(metadata.get("name") or path.stem),
                    event=self._normalize_event(str(metadata.get("event", "UserPromptSubmit"))),
                    action=str(metadata.get("action", "warn")),
                    message=str(metadata.get("message") or body.strip()),
                    pattern=str(metadata.get("pattern", "")),
                    tool_name=str(metadata.get("tool_name", metadata.get("tool", ""))),
                    command=str(metadata.get("command", "")),
                    enabled=bool(metadata.get("enabled", True)),
                    source=str(path),
                )
            )
        return rules

    def _matches(self, rule: HookRule, payload: Dict[str, object]) -> bool:
        if rule.tool_name and rule.tool_name != str(payload.get("tool_name", "")):
            return False
        if rule.pattern:
            haystack = " ".join(str(value) for value in payload.values())
            return rule.pattern.lower() in haystack.lower()
        return True

    def _normalize_event(self, event: str) -> str:
        aliases = {
            "sessionstart": "SessionStart",
            "userpromptsubmit": "UserPromptSubmit",
            "pretooluse": "PreToolUse",
            "posttooluse": "PostToolUse",
            "stop": "Stop",
            "beforetool": "PreToolUse",
            "aftertool": "PostToolUse",
        }
        key = str(event).replace("-", "").replace("_", "").lower()
        if key in aliases:
            return aliases[key]
        return event if event in HOOK_EVENTS else event

    def _record_event(
        self,
        event: str,
        payload: Dict[str, object],
        result: HookEventResult,
    ) -> None:
        path = self.workspace_dir / "hook_runs" / f"{utc_now().replace(':', '').replace('-', '')}-{event}.json"
        write_json(
            path,
            {
                "event": event,
                "payload": payload,
                "result": {
                    "blocked": result.blocked,
                    "warnings": result.warnings,
                    "system_messages": result.system_messages,
                    "matched_rules": result.matched_rules,
                },
            },
        )
