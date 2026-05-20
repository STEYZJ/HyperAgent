"""LLM usage and event ledger for cost/cache-aware agent runs."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from hyperagent.core.io import read_json
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import LLMProviderSpec, LLMResponse


class LLMUsageLedger:
    """Append-only JSONL ledger for model calls.

    The ledger is intentionally runtime-only: provider clients emit raw usage,
    and this class turns it into auditable records without importing CLI,
    training, or agent orchestration code.
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.root = workspace_dir / "usage"
        self.path = self.root / "llm_usage.jsonl"

    def record_response(
        self,
        response: LLMResponse,
        *,
        spec: LLMProviderSpec,
        session_id: Optional[str] = None,
        event_type: str = "llm.response",
        context_chars: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        usage = dict(response.usage or {})
        record = {
            "event_id": uuid4().hex,
            "timestamp": utc_now(),
            "event_type": event_type,
            "provider": response.provider,
            "model": response.model,
            "session_id": session_id,
            "context_chars": context_chars,
            "usage": usage,
            "prompt_tokens": _int_usage(usage, "prompt_tokens", "input_tokens"),
            "completion_tokens": _int_usage(
                usage,
                "completion_tokens",
                "output_tokens",
            ),
            "total_tokens": _int_usage(usage, "total_tokens"),
            "prompt_cache_hit_tokens": _int_usage(usage, "prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": _int_usage(usage, "prompt_cache_miss_tokens"),
            "warnings": list(response.warnings),
            "metadata": dict(metadata or {}),
        }
        record["cache_hit_ratio"] = _cache_hit_ratio(record)
        record["cost_estimate_usd"] = estimate_cost_usd(spec, usage)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if limit is not None and limit > 0:
            lines = lines[-limit:]
        records = []
        for line in lines:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"event_type": "ledger.decode_error", "raw": line})
        return records

    def summarize(self, limit: Optional[int] = None) -> Dict[str, Any]:
        records = self.records(limit=limit)
        totals = {
            "request_count": len(records),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "cost_estimate_usd": 0.0,
            "by_provider": {},
            "ledger_path": str(self.path),
        }
        for record in records:
            provider = str(record.get("provider", "unknown"))
            provider_totals = totals["by_provider"].setdefault(
                provider,
                {
                    "request_count": 0,
                    "total_tokens": 0,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 0,
                    "cost_estimate_usd": 0.0,
                },
            )
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ):
                value = int(record.get(key) or 0)
                totals[key] += value
                if key in provider_totals:
                    provider_totals[key] += value
            cost = float(record.get("cost_estimate_usd") or 0.0)
            totals["cost_estimate_usd"] += cost
            provider_totals["request_count"] += 1
            provider_totals["cost_estimate_usd"] += cost
        totals["cache_hit_ratio"] = _cache_hit_ratio(totals)
        return totals


def estimate_cost_usd(spec: LLMProviderSpec, usage: Dict[str, Any]) -> Optional[float]:
    """Estimate cost from provider metadata when explicit pricing is configured."""

    pricing = spec.metadata.get("pricing_per_million_tokens")
    if not isinstance(pricing, dict):
        return None
    input_price = _float_or_none(pricing.get("input"))
    output_price = _float_or_none(pricing.get("output"))
    cache_hit_price = _float_or_none(pricing.get("cache_hit_input"))
    cache_miss_price = _float_or_none(pricing.get("cache_miss_input"))
    prompt_tokens = _int_usage(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _int_usage(usage, "completion_tokens", "output_tokens")
    cache_hit_tokens = _int_usage(usage, "prompt_cache_hit_tokens")
    cache_miss_tokens = _int_usage(usage, "prompt_cache_miss_tokens")

    cost = 0.0
    if cache_hit_tokens or cache_miss_tokens:
        if cache_hit_price is not None:
            cost += cache_hit_tokens * cache_hit_price / 1_000_000
        if cache_miss_price is not None:
            cost += cache_miss_tokens * cache_miss_price / 1_000_000
        uncategorized = max(prompt_tokens - cache_hit_tokens - cache_miss_tokens, 0)
        if input_price is not None:
            cost += uncategorized * input_price / 1_000_000
    elif input_price is not None:
        cost += prompt_tokens * input_price / 1_000_000
    if output_price is not None:
        cost += completion_tokens * output_price / 1_000_000
    return round(cost, 8)


def load_usage_json(path: Path) -> Dict[str, Any]:
    """Small helper for tests and downstream reports."""

    data = read_json(path)
    return dict(data) if isinstance(data, dict) else {}


def _int_usage(usage: Dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cache_hit_ratio(record: Dict[str, Any]) -> Optional[float]:
    hit = int(record.get("prompt_cache_hit_tokens") or 0)
    miss = int(record.get("prompt_cache_miss_tokens") or 0)
    total = hit + miss
    if total <= 0:
        return None
    return round(hit / total, 4)
