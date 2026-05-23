"""External bot channel adapters for HyperAgent."""

from hyperagent.runtime.channels.config import ChannelConfigStore
from hyperagent.runtime.channels.delivery import ChannelDeliveryRecord, ChannelDeliveryStore
from hyperagent.runtime.channels.registry import (
    ChannelPlatformEntry,
    ChannelPlatformRegistry,
    platform_registry,
    register_builtin_channel_platforms,
)
from hyperagent.runtime.channels.router import ChannelRouter

__all__ = [
    "ChannelConfigStore",
    "ChannelDeliveryRecord",
    "ChannelDeliveryStore",
    "ChannelPlatformEntry",
    "ChannelPlatformRegistry",
    "ChannelRouter",
    "platform_registry",
    "register_builtin_channel_platforms",
]
