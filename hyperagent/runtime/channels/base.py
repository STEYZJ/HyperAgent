"""Shared channel adapter helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hyperagent.schemas import (
    ChannelBotConfig,
    ChannelEventResult,
    ChannelInboundMessage,
    ChannelOutboundMessage,
)


class ChannelAdapter:
    provider = ""

    def __init__(self, config: ChannelBotConfig) -> None:
        self.config = config

    def handle_webhook(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        body: bytes,
    ) -> ChannelEventResult:
        raise NotImplementedError

    def send_message(self, outbound: ChannelOutboundMessage) -> ChannelEventResult:
        raise NotImplementedError


def env_value(name: str) -> str:
    return os.environ.get(name, "").strip() if name else ""


def lower_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def json_dumps(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def nested_get(data: Dict[str, Any], paths: Iterable[Iterable[str]]) -> str:
    for path in paths:
        current: Any = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current not in {None, ""}:
            return str(current)
    return ""


def verify_token(
    config: ChannelBotConfig,
    candidates: Iterable[Optional[str]],
) -> Optional[str]:
    expected = env_value(config.verification_token_env)
    if not expected:
        return None
    for candidate in candidates:
        if candidate and hmac.compare_digest(str(candidate), expected):
            return None
    return "verification token mismatch"


def verify_hmac_sha256(
    secret: str,
    body: bytes,
    signature: str,
    prefixes: Iterable[bytes] = (),
) -> bool:
    if not secret or not signature:
        return False
    candidates = []
    for prefix in prefixes:
        digest = hmac.new(secret.encode("utf-8"), prefix + body, hashlib.sha256).hexdigest()
        candidates.extend([digest, "sha256=" + digest])
    if not prefixes:
        digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        candidates.extend([digest, "sha256=" + digest])
    return any(hmac.compare_digest(signature, candidate) for candidate in candidates)


def post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout_sec: int = 20,
) -> Dict[str, Any]:
    body = json_dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **dict(headers or {})},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {"raw": text}
    return data if isinstance(data, dict) else {"raw": data}


def ignored(provider: str, reason: str) -> ChannelEventResult:
    return ChannelEventResult(
        provider=provider,
        status="ignored",
        warnings=[reason],
    )


def error(provider: str, message: str, warnings: Optional[Iterable[str]] = None) -> ChannelEventResult:
    return ChannelEventResult(
        provider=provider,
        status="error",
        error=message,
        warnings=[str(v) for v in warnings or []],
    )


def replied(
    provider: str,
    outbound: ChannelOutboundMessage,
    payload: Dict[str, Any],
    warnings: Optional[Iterable[str]] = None,
) -> ChannelEventResult:
    return ChannelEventResult(
        provider=provider,
        status="replied",
        outbound=outbound,
        response_payload=payload,
        warnings=[str(v) for v in warnings or []],
    )
