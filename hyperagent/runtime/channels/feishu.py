"""Feishu bot channel adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hyperagent.runtime.channels.base import (
    ChannelAdapter,
    env_value,
    error,
    ignored,
    json_dumps,
    lower_headers,
    nested_get,
    parse_json_object,
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


class FeishuAdapter(ChannelAdapter):
    provider = "feishu"

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
                nested_get(payload, [["header", "token"]]),
            ],
        )
        if token_error:
            return error(self.provider, token_error)

        if payload.get("type") == "url_verification" and payload.get("challenge"):
            return ChannelEventResult(
                provider=self.provider,
                status="verified",
                response_payload={"challenge": str(payload["challenge"])},
            )

        signature_error = self._verify_signature(headers, body)
        if signature_error:
            return error(self.provider, signature_error)

        event_type = nested_get(payload, [["header", "event_type"]])
        if event_type and event_type != "im.message.receive_v1":
            return ignored(self.provider, f"unsupported Feishu event_type: {event_type}")

        inbound = self._parse_text_message(payload)
        if inbound is None:
            return ignored(self.provider, "Feishu event did not contain a supported text message")

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
            "receive_id": outbound.chat_id,
            "msg_type": "text",
            "content": json_dumps({"text": outbound.text}),
        }
        outbound.raw_payload = payload
        if self.config.dry_run or not self.config.send_enabled:
            return replied(self.provider, outbound, {"dry_run": True, "payload": payload})

        token = env_value(self.config.access_token_env)
        if not token:
            app_id = env_value(self.config.app_id_env)
            app_secret = env_value(self.config.app_secret_env)
            if app_id and app_secret:
                try:
                    token = self._tenant_access_token(app_id, app_secret)
                except RuntimeError as exc:
                    return replied(
                        self.provider,
                        outbound,
                        {"dry_run": True, "payload": payload},
                        warnings=[f"Feishu token request failed: {exc}"],
                    )
        if not token:
            return replied(
                self.provider,
                outbound,
                {"dry_run": True, "payload": payload},
                warnings=[
                    "Feishu access token is missing; set FEISHU_APP_ID/FEISHU_APP_SECRET "
                    "or a configured access_token_env to send replies."
                ],
            )

        url = (
            self.config.api_base_url.rstrip("/")
            + "/open-apis/im/v1/messages?receive_id_type=chat_id"
        )
        try:
            response = post_json(
                url,
                payload,
                headers={"Authorization": f"Bearer {token}"},
            )
        except RuntimeError as exc:
            return replied(
                self.provider,
                outbound,
                {"dry_run": False, "payload": payload},
                warnings=[f"Feishu send failed: {exc}"],
            )
        return replied(self.provider, outbound, response)

    def _verify_signature(self, headers: Dict[str, str], body: bytes) -> Optional[str]:
        secret = env_value(self.config.signing_secret_env)
        if not secret:
            return None
        normalized = lower_headers(headers)
        signature = normalized.get("x-lark-signature") or normalized.get(
            "x-feishu-signature", ""
        )
        if not signature:
            return "Feishu signature header missing"
        timestamp = normalized.get("x-lark-request-timestamp", "")
        nonce = normalized.get("x-lark-request-nonce", "")
        prefixes = [
            (timestamp + nonce).encode("utf-8"),
            (timestamp + nonce + secret).encode("utf-8"),
        ]
        if verify_hmac_sha256(secret, body, signature, prefixes=prefixes):
            return None
        return "Feishu signature verification failed"

    def _parse_text_message(self, payload: Dict[str, Any]) -> Optional[ChannelInboundMessage]:
        event = payload.get("event", {})
        if not isinstance(event, dict):
            return None
        message = event.get("message", {})
        if not isinstance(message, dict):
            return None
        message_type = str(message.get("message_type", message.get("msg_type", "")))
        if message_type != "text":
            return None
        content = parse_json_object(str(message.get("content", "")))
        text = str(content.get("text") or message.get("text") or "").strip()
        if not text:
            return None
        sender = event.get("sender", {})
        sender_id = {}
        if isinstance(sender, dict):
            raw_sender_id = sender.get("sender_id", {})
            if isinstance(raw_sender_id, dict):
                sender_id = raw_sender_id
        channel_user_id = str(
            sender_id.get("open_id")
            or sender_id.get("user_id")
            or sender_id.get("union_id")
            or nested_get(event, [["operator_id", "open_id"]])
            or "unknown"
        )
        chat_id = str(message.get("chat_id") or message.get("open_chat_id") or "")
        if not chat_id:
            return None
        return ChannelInboundMessage(
            provider=self.provider,
            channel_user_id=channel_user_id,
            chat_id=chat_id,
            message_id=str(message.get("message_id", "")),
            text=text,
            timestamp=str(message.get("create_time", "")),
            chat_type=str(message.get("chat_type", "")),
            raw_event=payload,
            metadata={
                "event_type": nested_get(payload, [["header", "event_type"]]),
                "tenant_key": nested_get(payload, [["header", "tenant_key"]]),
            },
        )

    def _tenant_access_token(self, app_id: str, app_secret: str) -> str:
        url = (
            self.config.api_base_url.rstrip("/")
            + "/open-apis/auth/v3/tenant_access_token/internal"
        )
        response = post_json(url, {"app_id": app_id, "app_secret": app_secret})
        token = str(response.get("tenant_access_token") or "")
        if not token:
            raise RuntimeError(response.get("msg") or "tenant_access_token missing")
        return token
