"""Schemas for external chat channel integrations."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


SUPPORTED_CHANNEL_PROVIDERS = {"feishu", "qq"}


@dataclass
class ChannelBotConfig:
    provider: str
    enabled: bool = True
    display_name: str = ""
    app_id_env: str = ""
    app_secret_env: str = ""
    access_token_env: str = ""
    verification_token_env: str = ""
    signing_secret_env: str = ""
    bot_user_id_env: str = ""
    api_base_url: str = ""
    default_llm_provider: str = "deepseek"
    default_model: Optional[str] = None
    default_mode: str = "research"
    max_context_chars: int = 12000
    send_enabled: bool = True
    dry_run: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelBotConfig":
        return cls(
            provider=str(data["provider"]),
            enabled=bool(data.get("enabled", True)),
            display_name=str(data.get("display_name", "")),
            app_id_env=str(data.get("app_id_env", "")),
            app_secret_env=str(data.get("app_secret_env", "")),
            access_token_env=str(data.get("access_token_env", "")),
            verification_token_env=str(data.get("verification_token_env", "")),
            signing_secret_env=str(data.get("signing_secret_env", "")),
            bot_user_id_env=str(data.get("bot_user_id_env", "")),
            api_base_url=str(data.get("api_base_url", "")),
            default_llm_provider=str(data.get("default_llm_provider", "deepseek")),
            default_model=(
                None
                if data.get("default_model") in {None, ""}
                else str(data.get("default_model"))
            ),
            default_mode=str(data.get("default_mode", "research")),
            max_context_chars=int(data.get("max_context_chars", 12000)),
            send_enabled=bool(data.get("send_enabled", True)),
            dry_run=bool(data.get("dry_run", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ChannelInboundMessage:
    provider: str
    channel_user_id: str
    chat_id: str
    message_id: str
    text: str
    timestamp: str = ""
    thread_id: Optional[str] = None
    chat_type: str = ""
    raw_event: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelInboundMessage":
        return cls(
            provider=str(data["provider"]),
            channel_user_id=str(data["channel_user_id"]),
            chat_id=str(data["chat_id"]),
            message_id=str(data.get("message_id", "")),
            text=str(data.get("text", "")),
            timestamp=str(data.get("timestamp", "")),
            thread_id=(
                None if data.get("thread_id") is None else str(data.get("thread_id"))
            ),
            chat_type=str(data.get("chat_type", "")),
            raw_event=dict(data.get("raw_event", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ChannelOutboundMessage:
    provider: str
    chat_id: str
    text: str
    reply_to_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    chat_type: str = ""
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelOutboundMessage":
        return cls(
            provider=str(data["provider"]),
            chat_id=str(data["chat_id"]),
            text=str(data.get("text", "")),
            reply_to_message_id=(
                None
                if data.get("reply_to_message_id") is None
                else str(data.get("reply_to_message_id"))
            ),
            thread_id=(
                None if data.get("thread_id") is None else str(data.get("thread_id"))
            ),
            chat_type=str(data.get("chat_type", "")),
            raw_payload=dict(data.get("raw_payload", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ChannelEventResult:
    provider: str
    status: str
    session_id: Optional[str] = None
    inbound: Optional[ChannelInboundMessage] = None
    outbound: Optional[ChannelOutboundMessage] = None
    response_payload: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.inbound is not None:
            payload["inbound"] = self.inbound.to_dict()
        if self.outbound is not None:
            payload["outbound"] = self.outbound.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelEventResult":
        inbound = data.get("inbound")
        outbound = data.get("outbound")
        return cls(
            provider=str(data["provider"]),
            status=str(data["status"]),
            session_id=(
                None if data.get("session_id") is None else str(data.get("session_id"))
            ),
            inbound=(
                ChannelInboundMessage.from_dict(inbound)
                if isinstance(inbound, dict)
                else None
            ),
            outbound=(
                ChannelOutboundMessage.from_dict(outbound)
                if isinstance(outbound, dict)
                else None
            ),
            response_payload=dict(data.get("response_payload", {})),
            warnings=[str(v) for v in data.get("warnings", [])],
            error=str(data.get("error", "")),
        )
