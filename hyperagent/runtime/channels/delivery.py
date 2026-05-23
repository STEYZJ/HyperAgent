"""Append-only channel delivery telemetry and retry support."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from hyperagent.runtime.channels.config import ChannelConfigStore
from hyperagent.runtime.channels.registry import ChannelPlatformRegistry, register_builtin_channel_platforms
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import ChannelEventResult, ChannelOutboundMessage


_FAILURE_MARKERS = (
    "send failed",
    "missing",
    "token request failed",
    "HTTPError",
    "URLError",
    "TimeoutError",
)


@dataclass
class ChannelDeliveryRecord:
    id: str
    provider: str
    status: str
    outbound: Dict[str, Any]
    response_payload: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    warnings: List[str] = field(default_factory=list)
    error: str = ""
    attempts: int = 1
    created_at: str = ""
    updated_at: str = ""
    source: str = "channel_router"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelDeliveryRecord":
        return cls(
            id=str(data.get("id", "")),
            provider=str(data.get("provider", "")),
            status=str(data.get("status", "")),
            outbound=dict(data.get("outbound", {}) if isinstance(data.get("outbound"), dict) else {}),
            response_payload=dict(
                data.get("response_payload", {})
                if isinstance(data.get("response_payload"), dict)
                else {}
            ),
            session_id=str(data.get("session_id", "")),
            warnings=[str(item) for item in data.get("warnings", [])],
            error=str(data.get("error", "")),
            attempts=int(data.get("attempts") or 0),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            source=str(data.get("source", "channel_router")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChannelDeliveryStore:
    """Stores channel outbound attempts under `.hyperagent/channels`."""

    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "channels" / "delivery.jsonl"

    def record(
        self,
        result: ChannelEventResult,
        *,
        source: str = "channel_router",
        delivery_id: str = "",
        attempts: Optional[int] = None,
    ) -> Optional[ChannelDeliveryRecord]:
        if result.outbound is None:
            return None
        now = utc_now()
        prior = self._latest().get(delivery_id) if delivery_id else None
        record = ChannelDeliveryRecord(
            id=delivery_id or f"delivery-{uuid4().hex[:10]}",
            provider=result.provider,
            status=self._delivery_status(result),
            outbound=result.outbound.to_dict(),
            response_payload=dict(result.response_payload),
            session_id=str(result.session_id or result.outbound.metadata.get("session_id", "")),
            warnings=list(result.warnings),
            error=result.error,
            attempts=attempts if attempts is not None else ((prior.attempts + 1) if prior else 1),
            created_at=prior.created_at if prior else now,
            updated_at=now,
            source=source,
        )
        self._append(record)
        return record

    def list_pending(
        self,
        *,
        provider: str = "",
        limit: Optional[int] = None,
    ) -> List[ChannelDeliveryRecord]:
        provider = provider.strip().lower()
        rows = [
            record
            for record in self._latest().values()
            if record.status == "pending"
            and (not provider or record.provider.lower() == provider)
        ]
        rows.sort(key=lambda item: (item.updated_at, item.id))
        return rows[: max(limit, 0)] if limit is not None else rows

    def retry_pending(
        self,
        *,
        config_store: ChannelConfigStore,
        platform_registry: Optional[ChannelPlatformRegistry] = None,
        provider: str = "",
        limit: Optional[int] = None,
    ) -> List[ChannelDeliveryRecord]:
        registry = platform_registry or register_builtin_channel_platforms()
        retried: List[ChannelDeliveryRecord] = []
        for record in self.list_pending(provider=provider, limit=limit):
            outbound = ChannelOutboundMessage.from_dict(record.outbound)
            config = config_store.get(record.provider)
            adapter = registry.create_adapter(config)
            outbound.metadata["retry_delivery_id"] = record.id
            result = adapter.send_message(outbound)
            result.session_id = record.session_id or result.session_id
            retried_record = self.record(
                result,
                source="channel_retry",
                delivery_id=record.id,
                attempts=record.attempts + 1,
            )
            if retried_record is not None:
                retried.append(retried_record)
        return retried

    def summary(self) -> Dict[str, Any]:
        latest = list(self._latest().values())
        by_status: Dict[str, int] = {}
        by_provider: Dict[str, int] = {}
        for record in latest:
            by_status[record.status] = by_status.get(record.status, 0) + 1
            by_provider[record.provider] = by_provider.get(record.provider, 0) + 1
        return {
            "path": str(self.path),
            "total": len(latest),
            "pending": by_status.get("pending", 0),
            "by_status": dict(sorted(by_status.items())),
            "by_provider": dict(sorted(by_provider.items())),
        }

    def list_records(self) -> List[ChannelDeliveryRecord]:
        return list(self._read())

    def _latest(self) -> Dict[str, ChannelDeliveryRecord]:
        latest: Dict[str, ChannelDeliveryRecord] = {}
        for record in self._read():
            latest[record.id] = record
        return latest

    def _read(self) -> Iterable[ChannelDeliveryRecord]:
        if not self.path.exists():
            return []
        records: List[ChannelDeliveryRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if isinstance(data, dict):
                    records.append(ChannelDeliveryRecord.from_dict(data))
        return records

    def _append(self, record: ChannelDeliveryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    def _delivery_status(self, result: ChannelEventResult) -> str:
        if result.status != "replied":
            return "pending"
        if self._has_failure_warning(result.warnings):
            return "pending"
        if result.response_payload.get("dry_run") is True:
            return "dry_run"
        return "delivered"

    def _has_failure_warning(self, warnings: Iterable[str]) -> bool:
        text = "\n".join(str(item) for item in warnings)
        return any(marker.lower() in text.lower() for marker in _FAILURE_MARKERS)
