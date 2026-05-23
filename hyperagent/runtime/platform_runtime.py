"""Hermes-style platform observability, session search, and skill telemetry."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.core.worklog import redact_secrets
from hyperagent.runtime.channels.config import ChannelConfigStore
from hyperagent.runtime.channels.delivery import ChannelDeliveryStore
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.events import RuntimeEventLog
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.skills import SkillStore
from hyperagent.runtime.workspace import HyperAgentWorkspace, utc_now


def default_skill_roots(project_root: Path, workspace_dir: Path) -> List[Path]:
    codex_home = os.environ.get("CODEX_HOME")
    package_root = Path(__file__).resolve().parents[1]
    return [
        package_root / "skills",
        Path(project_root) / "skills",
        Path(workspace_dir) / "skills",
        Path(codex_home) / "skills" if codex_home else Path.home() / ".codex" / "skills",
    ]


def skill_bundle_name(skill: object) -> str:
    metadata = getattr(skill, "metadata", {}) or {}
    source = str(getattr(skill, "source", "") or "")
    return str(
        metadata.get("bundle")
        or metadata.get("category")
        or (Path(source).name if source else "")
        or "local"
    )


class PlatformStatusReporter:
    """Aggregates platform health; live network checks are explicit opt-in."""

    def __init__(
        self,
        workspace: HyperAgentWorkspace,
        conversations: ConversationStore,
        providers: LLMProviderStore,
        *,
        channel_store: Optional[ChannelConfigStore] = None,
        skill_roots: Optional[Iterable[Path]] = None,
    ) -> None:
        self.workspace = workspace
        self.conversations = conversations
        self.providers = providers
        self.channel_store = channel_store or ChannelConfigStore(workspace.workspace_dir)
        self.skill_roots = list(
            skill_roots
            if skill_roots is not None
            else default_skill_roots(workspace.project_root, workspace.workspace_dir)
        )

    def report(self, *, live: bool = False, timeout_sec: float = 2.0) -> Dict[str, Any]:
        warnings: List[str] = []
        providers = self._provider_status(warnings, live=live, timeout_sec=timeout_sec)
        channels = self._channel_status(warnings, live=live, timeout_sec=timeout_sec)
        sessions = self._session_status()
        skills = self._skill_status(warnings)
        events = RuntimeEventLog(self.workspace.workspace_dir).summarize()
        skill_usage = SkillTelemetryStore(self.workspace.workspace_dir).summarize()
        channel_delivery = ChannelDeliveryStore(self.workspace.workspace_dir).summary()
        return {
            "status": "ok" if not warnings else "degraded",
            "workspace": str(self.workspace.workspace_dir),
            "live": bool(live),
            "providers": providers,
            "channels": channels,
            "sessions": sessions,
            "skills": skills,
            "events": events,
            "channel_delivery": channel_delivery,
            "skill_usage": {
                "total_events": skill_usage["total_events"],
                "by_action": skill_usage["by_action"],
            },
            "warnings": warnings,
        }

    def _provider_status(
        self,
        warnings: List[str],
        *,
        live: bool,
        timeout_sec: float,
    ) -> List[Dict[str, Any]]:
        rows = []
        for provider in self.providers.ensure_defaults():
            configured = bool(os.environ.get(provider.api_key_env, "").strip())
            health = "configured" if configured else "missing_api_key"
            if not configured:
                warnings.append(f"provider {provider.name} missing env {provider.api_key_env}")
            live_probe = self._probe_url(provider.base_url, timeout_sec) if live else {"checked": False}
            if live and configured and not live_probe.get("reachable"):
                health = "unreachable"
                warnings.append(f"provider {provider.name} base URL is not reachable")
            rows.append(
                {
                    "name": provider.name,
                    "kind": provider.kind,
                    "default_model": provider.default_model,
                    "api_key_env": provider.api_key_env,
                    "api_key_configured": configured,
                    "health": health,
                    "live": live_probe,
                }
            )
        return rows

    def _channel_status(
        self,
        warnings: List[str],
        *,
        live: bool,
        timeout_sec: float,
    ) -> List[Dict[str, Any]]:
        configs = self.channel_store.ensure_defaults()
        env_summary = self.channel_store.env_summary()
        env_configured = self.channel_store.env_configured_summary()
        rows = []
        for config in configs:
            provider_env = env_configured.get(config.provider, {})
            missing = [name for name, configured in provider_env.items() if not configured]
            if not config.enabled:
                health = "disabled"
            elif missing:
                health = "missing_env"
                warnings.append(f"channel {config.provider} missing env: {', '.join(missing)}")
            else:
                health = "ready"
            live_probe = self._probe_url(config.api_base_url, timeout_sec) if live else {"checked": False}
            if live and config.enabled and not live_probe.get("reachable"):
                health = "unreachable" if health == "ready" else health
                warnings.append(f"channel {config.provider} API base URL is not reachable")
            rows.append(
                {
                    "provider": config.provider,
                    "enabled": config.enabled,
                    "display_name": config.display_name,
                    "default_llm_provider": config.default_llm_provider,
                    "default_model": config.default_model,
                    "default_mode": config.default_mode,
                    "env_vars": env_summary.get(config.provider, []),
                    "env_configured": provider_env,
                    "chat_query_only": True,
                    "health": health,
                    "live": live_probe,
                }
            )
        return rows

    def _session_status(self) -> Dict[str, Any]:
        sessions = self.conversations.list(include_archived=True)
        by_status: Dict[str, int] = {}
        channel_sessions = 0
        message_count = 0
        summary_count = 0
        for session in sessions:
            by_status[session.status] = by_status.get(session.status, 0) + 1
            if session.metadata.get("channel_provider"):
                channel_sessions += 1
            message_count += len(session.messages)
            summary_count += len(session.summaries)
        return {
            "total": len(sessions),
            "by_status": dict(sorted(by_status.items())),
            "channel_sessions": channel_sessions,
            "message_count": message_count,
            "summary_count": summary_count,
        }

    def _skill_status(self, warnings: List[str]) -> Dict[str, Any]:
        skills = SkillStore(self.skill_roots).list()
        by_run_as: Dict[str, int] = {}
        for skill in skills:
            by_run_as[skill.run_as] = by_run_as.get(skill.run_as, 0) + 1
        bundle_summary = summarize_skill_bundles(skills)
        missing_bundle_metadata = bundle_summary["missing_bundle_metadata"]
        if missing_bundle_metadata:
            warnings.append(f"{missing_bundle_metadata} skills have no bundle/category metadata")
        return {
            "total": len(skills),
            "bundles": {
                name: int(item["count"])
                for name, item in sorted(bundle_summary["bundles"].items())
            },
            "bundle_metadata": bundle_summary["bundles"],
            "by_run_as": dict(sorted(by_run_as.items())),
            "missing_bundle_metadata": missing_bundle_metadata,
        }

    def _probe_url(self, url: str, timeout_sec: float) -> Dict[str, Any]:
        parsed = urlparse(str(url or ""))
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return {"checked": True, "reachable": False, "host": "", "error": "missing_host"}
        try:
            with socket.create_connection((host, int(port)), timeout=float(timeout_sec)):
                return {"checked": True, "reachable": True, "host": host, "port": int(port)}
        except OSError as exc:
            return {
                "checked": True,
                "reachable": False,
                "host": host,
                "port": int(port),
                "error": type(exc).__name__,
            }


def summarize_skill_bundles(skills: Iterable[object]) -> Dict[str, Any]:
    bundles: Dict[str, Dict[str, Any]] = {}
    missing_bundle_metadata = 0
    for skill in skills:
        metadata = getattr(skill, "metadata", {}) or {}
        bundle = skill_bundle_name(skill)
        source = str(getattr(skill, "source", "") or "")
        owner = str(metadata.get("owner") or metadata.get("maintainer") or "").strip()
        missing = "bundle" not in metadata and "category" not in metadata
        if missing:
            missing_bundle_metadata += 1
        row = bundles.setdefault(
            bundle,
            {
                "count": 0,
                "skills": [],
                "sources": [],
                "owners": [],
                "missing_metadata": 0,
            },
        )
        row["count"] += 1
        row["skills"].append(str(getattr(skill, "name", "")))
        if source and source not in row["sources"]:
            row["sources"].append(source)
        if owner and owner not in row["owners"]:
            row["owners"].append(owner)
        if missing:
            row["missing_metadata"] += 1
    for row in bundles.values():
        row["skills"] = sorted(row["skills"])
        row["sources"] = sorted(row["sources"])
        row["owners"] = sorted(row["owners"])
    return {
        "bundles": dict(sorted(bundles.items())),
        "missing_bundle_metadata": missing_bundle_metadata,
    }


@dataclass
class SessionSearchResult:
    session_id: str
    title: str
    status: str
    updated_at: str
    message_count: int
    summary_count: int
    score: int
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SessionSearchIndex:
    """Builds a lightweight local session index and returns short snippets."""

    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "sessions" / "session_index.json"

    def rebuild(
        self,
        conversations: ConversationStore,
        *,
        include_archived: bool = False,
    ) -> Dict[str, Any]:
        sessions = conversations.list(include_archived=include_archived)
        entries = [self._entry(session) for session in sessions]
        payload = {
            "version": 1,
            "updated_at": utc_now(),
            "include_archived": include_archived,
            "sessions": entries,
        }
        write_json(self.path, payload)
        return payload

    def search(
        self,
        conversations: ConversationStore,
        query: str,
        *,
        limit: int = 10,
        include_archived: bool = False,
    ) -> List[SessionSearchResult]:
        query = str(query or "").strip()
        payload = self.rebuild(conversations, include_archived=include_archived)
        terms = [term for term in query.lower().split() if term]
        if not terms:
            return []
        results: List[SessionSearchResult] = []
        for entry in payload["sessions"]:
            text = str(entry.get("search_text", ""))
            lowered = text.lower()
            score = sum(lowered.count(term) for term in terms)
            title = str(entry.get("title", ""))
            title_lower = title.lower()
            score += 3 * sum(title_lower.count(term) for term in terms)
            if score <= 0:
                continue
            results.append(
                SessionSearchResult(
                    session_id=str(entry["session_id"]),
                    title=title,
                    status=str(entry["status"]),
                    updated_at=str(entry["updated_at"]),
                    message_count=int(entry["message_count"]),
                    summary_count=int(entry["summary_count"]),
                    score=score,
                    snippet=self._snippet(text, terms),
                )
            )
        return sorted(results, key=lambda item: (-item.score, item.updated_at), reverse=False)[: max(limit, 0)]

    def _entry(self, session: object) -> Dict[str, Any]:
        metadata = getattr(session, "metadata", {}) or {}
        parts = [getattr(session, "title", "")]
        parts.extend(str(value) for value in metadata.values())
        for summary in getattr(session, "summaries", []):
            parts.append(str(summary.content))
        for message in getattr(session, "messages", []):
            parts.append(str(message.content))
        text = "\n".join(part for part in parts if part)
        return {
            "session_id": session.session_id,
            "title": session.title,
            "status": session.status,
            "updated_at": session.updated_at,
            "message_count": len(session.messages),
            "summary_count": len(session.summaries),
            "metadata_keys": sorted(str(key) for key in metadata),
            "search_text": text,
        }

    def _snippet(self, text: str, terms: List[str], limit: int = 240) -> str:
        clean = " ".join(str(text or "").split())
        if not clean:
            return ""
        lowered = clean.lower()
        positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
        if not positions:
            return clean[:limit]
        start = max(min(positions) - 60, 0)
        snippet = clean[start : start + limit]
        if start > 0:
            snippet = "..." + snippet
        if len(clean) > start + limit:
            snippet = snippet.rstrip() + "..."
        return snippet[:limit]


class SkillTelemetryStore:
    """Append-only skill usage ledger and small curator summary."""

    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "skills" / "usage.jsonl"

    def record(
        self,
        action: str,
        *,
        skill: str = "",
        bundle: str = "",
        source: str = "runtime",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = {
            "event_id": uuid4().hex,
            "timestamp": utc_now(),
            "action": str(action),
            "skill": str(skill or ""),
            "bundle": str(bundle or ""),
            "source": str(source or "runtime"),
            "metadata": self._redact_metadata(metadata or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def summarize(self, *, limit: int = 20) -> Dict[str, Any]:
        events = self.records()
        return {
            "path": str(self.path),
            "total_events": len(events),
            "by_skill": self._counts(events, "skill", skip_empty=True),
            "by_bundle": self._counts(events, "bundle", skip_empty=True),
            "by_action": self._counts(events, "action"),
            "recent": events[-max(limit, 0) :],
        }

    def curate(self, skills: Iterable[object]) -> Dict[str, Any]:
        skill_list = list(skills)
        summary = self.summarize(limit=10)
        by_skill = summary["by_skill"]
        known = {str(skill.name): skill for skill in skill_list}
        unused = sorted(name for name in known if name not in by_skill)
        high_frequency = [
            {"skill": name, "uses": count}
            for name, count in sorted(by_skill.items(), key=lambda item: (-item[1], item[0]))[:5]
        ]
        missing_bundle_metadata = sorted(
            str(skill.name)
            for skill in skill_list
            if "bundle" not in (skill.metadata or {}) and "category" not in (skill.metadata or {})
        )
        return {
            "unused_skills": unused,
            "high_frequency_skills": high_frequency,
            "missing_bundle_metadata": missing_bundle_metadata,
        }

    def records(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                events.append(data)
        return events

    def _counts(
        self,
        events: Iterable[Dict[str, Any]],
        key: str,
        *,
        skip_empty: bool = False,
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in events:
            value = str(event.get(key, "") or "")
            if skip_empty and not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    def _redact_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        redacted: Dict[str, Any] = {}
        for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
            if any(marker in str(key).lower() for marker in ("token", "key", "password", "secret")):
                redacted[str(key)] = "[REDACTED_SECRET]"
                continue
            if isinstance(value, (dict, list, tuple)):
                redacted[str(key)] = redact_secrets(
                    json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                )
            else:
                redacted[str(key)] = redact_secrets(value)
        return redacted
