"""QQ official bot channel adapter."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from hyperagent.runtime.channels.base import (
    ChannelAdapter,
    env_value,
    error,
    ignored,
    lower_headers,
    nested_get,
    post_json,
    replied,
    verify_hmac_sha256,
    verify_token,
)
from hyperagent.schemas import (
    ChannelEventResult,
    ChannelInboundMessage,
    ChannelOutboundMessage,
)


class QQAdapter(ChannelAdapter):
    provider = "qq"

    def handle_webhook(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        body: bytes,
    ) -> ChannelEventResult:
        token_error = verify_token(
            self.config,
            [
                payload.get("token"),
                payload.get("verification_token"),
                nested_get(payload, [["d", "token"]]),
            ],
        )
        if token_error:
            return error(self.provider, token_error)

        if payload.get("challenge"):
            return ChannelEventResult(
                provider=self.provider,
                status="verified",
                response_payload={"challenge": str(payload["challenge"])},
            )

        signature_error = self._verify_signature(headers, body)
        if signature_error:
            return error(self.provider, signature_error)

        inbound = self._parse_text_message(payload)
        if inbound is None:
            event_type = str(payload.get("t") or payload.get("type") or "")
            return ignored(
                self.provider,
                f"QQ event did not contain a supported text message: {event_type}",
            )

        bot_id = env_value(self.config.bot_user_id_env)
        if bot_id and inbound.channel_user_id == bot_id:
            return ignored(self.provider, "ignored self message")

        return ChannelEventResult(
            provider=self.provider,
            status="received",
            inbound=inbound,
            response_payload={"ok": True},
        )

    def send_message(self, outbound: ChannelOutboundMessage) -> ChannelEventResult:
        payload = {
            "content": outbound.text,
            "msg_id": outbound.reply_to_message_id or "",
        }
        endpoint = self._send_endpoint(outbound)
        outbound.raw_payload = {
            "url": endpoint,
            "json": payload,
        }
        if self.config.dry_run or not self.config.send_enabled:
            return replied(
                self.provider,
                outbound,
                {"dry_run": True, "payload": outbound.raw_payload},
            )
        token = env_value(self.config.access_token_env)
        if not token:
            return replied(
                self.provider,
                outbound,
                {"dry_run": True, "payload": outbound.raw_payload},
                warnings=[
                    "QQ bot token is missing; set QQ_BOT_TOKEN or the configured "
                    "access_token_env to send replies."
                ],
            )
        auth_prefix = str(self.config.metadata.get("authorization_prefix", "QQBot"))
        try:
            response = post_json(
                endpoint,
                payload,
                headers={"Authorization": f"{auth_prefix} {token}"},
            )
        except RuntimeError as exc:
            return replied(
                self.provider,
                outbound,
                {"dry_run": False, "payload": outbound.raw_payload},
                warnings=[f"QQ send failed: {exc}"],
            )
        return replied(self.provider, outbound, response)

    def _verify_signature(self, headers: Dict[str, str], body: bytes) -> Optional[str]:
        secret = env_value(self.config.signing_secret_env)
        if not secret:
            return None
        normalized = lower_headers(headers)
        signature = (
            normalized.get("x-hyperagent-signature")
            or normalized.get("x-tencent-signature")
            or normalized.get("x-qq-signature")
            or ""
        )
        if not signature:
            return "QQ signature header missing"
        timestamp = normalized.get("x-tencent-timestamp", "")
        prefixes = [timestamp.encode("utf-8")] if timestamp else ()
        if verify_hmac_sha256(secret, body, signature, prefixes=prefixes):
            return None
        return "QQ signature verification failed"

    def _parse_text_message(self, payload: Dict[str, Any]) -> Optional[ChannelInboundMessage]:
        event_type = str(payload.get("t") or payload.get("type") or "")
        data = payload.get("d", payload)
        if not isinstance(data, dict):
            return None
        content = str(
            data.get("content")
            or data.get("text")
            or nested_get(data, [["message", "content"]])
            or ""
        ).strip()
        content = self._strip_bot_mention(content)
        if not content:
            return None
        message_id = str(data.get("id") or data.get("msg_id") or data.get("message_id") or "")
        user_id = str(
            nested_get(data, [["author", "id"], ["author", "user_openid"]])
            or data.get("user_openid")
            or data.get("openid")
            or data.get("user_id")
            or "unknown"
        )
        chat_id = str(
            data.get("group_openid")
            or data.get("guild_id")
            or data.get("channel_id")
            or data.get("chat_id")
            or data.get("user_openid")
            or data.get("openid")
            or user_id
        )
        if not chat_id:
            return None
        chat_type = self._chat_type(event_type, data)
        return ChannelInboundMessage(
            provider=self.provider,
            channel_user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            text=content,
            timestamp=str(data.get("timestamp") or data.get("create_time") or ""),
            chat_type=chat_type,
            raw_event=payload,
            metadata={
                "event_type": event_type,
                "seq": str(payload.get("s", "")),
            },
        )

    def _send_endpoint(self, outbound: ChannelOutboundMessage) -> str:
        base = self.config.api_base_url.rstrip("/")
        if outbound.chat_type == "group":
            return f"{base}/v2/groups/{outbound.chat_id}/messages"
        if outbound.chat_type == "guild":
            return f"{base}/channels/{outbound.chat_id}/messages"
        return f"{base}/v2/users/{outbound.chat_id}/messages"

    def _chat_type(self, event_type: str, data: Dict[str, Any]) -> str:
        upper = event_type.upper()
        if "GROUP" in upper or data.get("group_openid"):
            return "group"
        if "GUILD" in upper or data.get("guild_id") or data.get("channel_id"):
            return "guild"
        return "direct"

    def _strip_bot_mention(self, text: str) -> str:
        return re.sub(r"<@!?[A-Za-z0-9_\-]+>", "", text).strip()
