"""Append-only runtime event log for replayable agent operations."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from hyperagent.runtime.workspace import utc_now


@dataclass
class RuntimeEvent:
    event_id: str
    timestamp: str
    event_type: str
    source: str = "runtime"
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    tool_name: Optional[str] = None
    status: str = ""
    message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeEvent":
        return cls(
            event_id=str(data.get("event_id", "")),
            timestamp=str(data.get("timestamp", "")),
            event_type=str(data.get("event_type", "")),
            source=str(data.get("source", "runtime")),
            session_id=(
                None if data.get("session_id") is None else str(data.get("session_id"))
            ),
            run_id=None if data.get("run_id") is None else str(data.get("run_id")),
            tool_name=(
                None if data.get("tool_name") is None else str(data.get("tool_name"))
            ),
            status=str(data.get("status", "")),
            message=str(data.get("message", "")),
            payload=dict(data.get("payload", {})),
        )


class RuntimeEventLog:
    """Small JSONL event sink/source used by CLI, REPL, TUI, and tools."""

    def __init__(self, workspace_dir: Path) -> None:
        self.root = Path(workspace_dir) / "events"
        self.path = self.root / "runtime_events.jsonl"

    def append(
        self,
        event_type: str,
        *,
        source: str = "runtime",
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        status: str = "",
        message: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            event_id=uuid4().hex,
            timestamp=utc_now(),
            event_type=event_type,
            source=source,
            session_id=session_id,
            run_id=run_id,
            tool_name=tool_name,
            status=status,
            message=message,
            payload=dict(payload or {}),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def records(
        self,
        *,
        limit: Optional[int] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[RuntimeEvent]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if limit is not None and limit > 0:
            lines = lines[-limit:]
        events: List[RuntimeEvent] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                event = RuntimeEvent.from_dict(json.loads(line))
            except json.JSONDecodeError:
                event = RuntimeEvent(
                    event_id=uuid4().hex,
                    timestamp="",
                    event_type="event.decode_error",
                    status="error",
                    message=line,
                )
            if event_type and event.event_type != event_type:
                continue
            if session_id and event.session_id != session_id:
                continue
            if run_id and event.run_id != run_id:
                continue
            events.append(event)
        return events

    def summarize(self, events: Optional[Iterable[RuntimeEvent]] = None) -> Dict[str, Any]:
        selected = list(events) if events is not None else self.records()
        by_type: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_tool: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        run_ids = set()
        for event in selected:
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
            status = event.status or "none"
            by_status[status] = by_status.get(status, 0) + 1
            source = event.source or "runtime"
            by_source[source] = by_source.get(source, 0) + 1
            if event.tool_name:
                by_tool[event.tool_name] = by_tool.get(event.tool_name, 0) + 1
            if event.run_id:
                run_ids.add(event.run_id)
        return {
            "event_count": len(selected),
            "by_type": dict(sorted(by_type.items())),
            "by_status": dict(sorted(by_status.items())),
            "by_tool": dict(sorted(by_tool.items())),
            "by_source": dict(sorted(by_source.items())),
            "run_count": len(run_ids),
            "path": str(self.path),
        }
