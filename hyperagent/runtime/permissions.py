"""Remembered tool-permission rules for Claude-style local approvals."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hyperagent.core.io import read_json, write_json
from hyperagent.core.worklog import redact_secrets
from hyperagent.runtime.workspace import utc_now


_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|token|secret|password|credential|authorization)",
    re.IGNORECASE,
)


@dataclass
class RememberedPermissionRule:
    id: str
    key: str
    tool_name: str
    risk_level: str
    args_fingerprint: str
    args: Dict[str, Any]
    reason: str = ""
    scope: str = "project"
    created_at: str = ""
    last_used_at: str = ""
    uses: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RememberedPermissionRule":
        return cls(
            id=str(data.get("id", "")),
            key=str(data.get("key", "")),
            tool_name=str(data.get("tool_name", "")),
            risk_level=str(data.get("risk_level", "")),
            args_fingerprint=str(data.get("args_fingerprint", "")),
            args=dict(data.get("args", {}) if isinstance(data.get("args"), dict) else {}),
            reason=str(data.get("reason", "")),
            scope=str(data.get("scope", "project")),
            created_at=str(data.get("created_at", "")),
            last_used_at=str(data.get("last_used_at", "")),
            uses=int(data.get("uses") or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "tool_name": self.tool_name,
            "risk_level": self.risk_level,
            "args_fingerprint": self.args_fingerprint,
            "args": self.args,
            "reason": self.reason,
            "scope": self.scope,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "uses": self.uses,
        }


class RememberedPermissionStore:
    """Stores exact, local-only permission approvals under `.hyperagent`."""

    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "permissions" / "remembered.json"

    def list_rules(self) -> List[RememberedPermissionRule]:
        return sorted(self._load(), key=lambda rule: (rule.created_at, rule.id))

    def remember(self, request: Any, *, scope: str = "project") -> RememberedPermissionRule:
        tool_name = str(getattr(request, "tool_name", ""))
        risk_level = str(getattr(request, "risk_level", ""))
        args = dict(getattr(request, "args", {}) or {})
        key = self.key_for(tool_name, risk_level, args)
        now = utc_now()
        rules = self._load()
        for rule in rules:
            if rule.key == key:
                rule.reason = redact_secrets(str(getattr(request, "reason", "")))
                rule.last_used_at = now
                rule.scope = scope
                self._save(rules)
                return rule
        rule = RememberedPermissionRule(
            id=f"perm-{uuid4().hex[:8]}",
            key=key,
            tool_name=tool_name,
            risk_level=risk_level,
            args_fingerprint=self.args_fingerprint(tool_name, args),
            args=self.redact_args(args),
            reason=redact_secrets(str(getattr(request, "reason", ""))),
            scope=scope,
            created_at=now,
            last_used_at=now,
        )
        rules.append(rule)
        self._save(rules)
        return rule

    def forget(self, rule_id_or_key: str) -> bool:
        needle = str(rule_id_or_key)
        rules = self._load()
        kept = [
            rule
            for rule in rules
            if rule.id != needle
            and rule.key != needle
            and rule.args_fingerprint != needle
        ]
        if len(kept) == len(rules):
            return False
        self._save(kept)
        return True

    def clear(self) -> int:
        count = len(self._load())
        self._save([])
        return count

    def summary(self) -> Dict[str, Any]:
        rules = self.list_rules()
        return {
            "path": str(self.path),
            "count": len(rules),
            "rules": [rule.to_dict() for rule in rules],
        }

    def is_allowed(self, request: Any) -> bool:
        tool_name = str(getattr(request, "tool_name", ""))
        risk_level = str(getattr(request, "risk_level", ""))
        args = dict(getattr(request, "args", {}) or {})
        key = self.key_for(tool_name, risk_level, args)
        rules = self._load()
        matched: Optional[RememberedPermissionRule] = None
        for rule in rules:
            if rule.key == key:
                matched = rule
                break
        if matched is None:
            return False
        matched.uses += 1
        matched.last_used_at = utc_now()
        self._save(rules)
        return True

    @classmethod
    def key_for(cls, tool_name: str, risk_level: str, args: Dict[str, Any]) -> str:
        fingerprint = cls.args_fingerprint(tool_name, args)
        return f"{risk_level}:{tool_name}:{fingerprint}"

    @classmethod
    def args_fingerprint(cls, tool_name: str, args: Dict[str, Any]) -> str:
        normalized = cls._normalize_args(tool_name, args)
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def redact_args(cls, value: Any) -> Any:
        if isinstance(value, dict):
            redacted: Dict[str, Any] = {}
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                key_text = str(key)
                if _SENSITIVE_KEY_RE.search(key_text):
                    redacted[key_text] = "[REDACTED_SECRET]"
                else:
                    redacted[key_text] = cls.redact_args(item)
            return redacted
        if isinstance(value, list):
            return [cls.redact_args(item) for item in value]
        if isinstance(value, tuple):
            return [cls.redact_args(item) for item in value]
        if isinstance(value, str):
            return redact_secrets(value)
        return value

    @classmethod
    def _normalize_args(cls, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name == "run_command":
            return {
                "argv": [str(item) for item in args.get("argv", [])]
                if isinstance(args.get("argv", []), list)
                else [],
                "cwd": str(args.get("cwd", "") or ""),
                "timeout_sec": int(args.get("timeout_sec") or 0),
            }
        return cls._normalize_value(args)

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._normalize_value(value[key]) for key in sorted(value, key=str)}
        if isinstance(value, (list, tuple)):
            return [cls._normalize_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _load(self) -> List[RememberedPermissionRule]:
        if not self.path.exists():
            return []
        data = read_json(self.path)
        items = data.get("rules", [])
        if not isinstance(items, list):
            return []
        return [
            RememberedPermissionRule.from_dict(item)
            for item in items
            if isinstance(item, dict)
        ]

    def _save(self, rules: List[RememberedPermissionRule]) -> None:
        write_json(
            self.path,
            {
                "version": 1,
                "rules": [rule.to_dict() for rule in rules],
            },
        )
