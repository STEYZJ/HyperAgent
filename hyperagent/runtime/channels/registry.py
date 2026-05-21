"""Pluggable channel platform registry."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

from hyperagent.runtime.channels.base import ChannelAdapter
from hyperagent.schemas import ChannelBotConfig


@dataclass(frozen=True)
class ChannelPlatformEntry:
    provider: str
    display_name: str
    adapter_cls: Type[ChannelAdapter]
    required_env: List[str] = field(default_factory=list)
    chat_query_only: bool = True
    supports_send: bool = True
    supports_webhook: bool = True
    validator: Optional[Callable[[ChannelBotConfig], bool]] = None

    def configured(self, config: ChannelBotConfig) -> bool:
        if self.validator is not None:
            return bool(self.validator(config))
        return bool(config.enabled)

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "display_name": self.display_name,
            "required_env": list(self.required_env),
            "chat_query_only": self.chat_query_only,
            "supports_send": self.supports_send,
            "supports_webhook": self.supports_webhook,
        }


class ChannelPlatformRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, ChannelPlatformEntry] = {}

    def register(self, entry: ChannelPlatformEntry) -> None:
        self._entries[entry.provider] = entry

    def get(self, provider: str) -> ChannelPlatformEntry:
        key = provider.lower().strip()
        if key not in self._entries:
            raise KeyError(f"Unknown channel provider: {provider}")
        return self._entries[key]

    def create_adapter(self, config: ChannelBotConfig) -> ChannelAdapter:
        entry = self.get(config.provider)
        return entry.adapter_cls(config)

    def list(self) -> List[ChannelPlatformEntry]:
        return [self._entries[key] for key in sorted(self._entries)]


platform_registry = ChannelPlatformRegistry()


def register_builtin_channel_platforms() -> ChannelPlatformRegistry:
    from hyperagent.runtime.channels.feishu import FeishuAdapter
    from hyperagent.runtime.channels.qq import QQAdapter

    platform_registry.register(
        ChannelPlatformEntry(
            provider="feishu",
            display_name="Feishu Bot",
            adapter_cls=FeishuAdapter,
            required_env=[
                "FEISHU_APP_ID",
                "FEISHU_APP_SECRET",
                "FEISHU_VERIFICATION_TOKEN",
                "FEISHU_SIGNING_SECRET",
            ],
        )
    )
    platform_registry.register(
        ChannelPlatformEntry(
            provider="qq",
            display_name="QQ Official Bot",
            adapter_cls=QQAdapter,
            required_env=[
                "QQ_BOT_APP_ID",
                "QQ_BOT_SECRET",
                "QQ_BOT_TOKEN",
                "QQ_BOT_VERIFICATION_TOKEN",
                "QQ_BOT_SIGNING_SECRET",
            ],
        )
    )
    return platform_registry
