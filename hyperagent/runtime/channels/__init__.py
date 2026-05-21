"""External bot channel adapters for HyperAgent."""

from hyperagent.runtime.channels.config import ChannelConfigStore
from hyperagent.runtime.channels.registry import (
    ChannelPlatformEntry,
    ChannelPlatformRegistry,
    platform_registry,
    register_builtin_channel_platforms,
)
from hyperagent.runtime.channels.router import ChannelRouter

__all__ = [
    "ChannelConfigStore",
    "ChannelPlatformEntry",
    "ChannelPlatformRegistry",
    "ChannelRouter",
    "platform_registry",
    "register_builtin_channel_platforms",
]
