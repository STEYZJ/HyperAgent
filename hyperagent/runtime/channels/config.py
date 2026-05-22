"""Configuration store for external bot channels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from hyperagent.core.io import read_yaml, write_yaml
from hyperagent.schemas import ChannelBotConfig
from hyperagent.schemas.channels import SUPPORTED_CHANNEL_PROVIDERS


DEFAULT_CHANNEL_CONFIGS = {
    "feishu": ChannelBotConfig(
        provider="feishu",
        display_name="Feishu Bot",
        app_id_env="FEISHU_APP_ID",
        app_secret_env="FEISHU_APP_SECRET",
        access_token_env="",
        verification_token_env="FEISHU_VERIFICATION_TOKEN",
        signing_secret_env="FEISHU_SIGNING_SECRET",
        bot_user_id_env="FEISHU_BOT_OPEN_ID",
        api_base_url="https://open.feishu.cn",
        metadata={
            "supported_events": ["im.message.receive_v1"],
            "notes": "Official Feishu event subscription and IM send-message path.",
        },
    ),
    "qq": ChannelBotConfig(
        provider="qq",
        display_name="QQ Official Bot",
        app_id_env="QQ_BOT_APP_ID",
        app_secret_env="QQ_BOT_SECRET",
        access_token_env="QQ_BOT_TOKEN",
        verification_token_env="QQ_BOT_VERIFICATION_TOKEN",
        signing_secret_env="QQ_BOT_SIGNING_SECRET",
        bot_user_id_env="QQ_BOT_ID",
        api_base_url="https://api.sgroup.qq.com",
        metadata={
            "supported_events": [
                "C2C_MESSAGE_CREATE",
                "GROUP_AT_MESSAGE_CREATE",
                "DIRECT_MESSAGE_CREATE",
            ],
            "authorization_prefix": "QQBot",
            "notes": "Official QQ bot/OpenAPI only; personal-account automation is out of scope.",
        },
    ),
}


class ChannelConfigStore:
    """Stores channel configs without persisting secret values."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.path = self.workspace_dir / "channels.yaml"

    def ensure_defaults(self) -> List[ChannelBotConfig]:
        if not self.path.exists():
            configs = list(DEFAULT_CHANNEL_CONFIGS.values())
            self.save_all(configs)
            return configs
        configs = {item.provider: item for item in self.list()}
        changed = False
        for provider, default in DEFAULT_CHANNEL_CONFIGS.items():
            if provider not in configs:
                configs[provider] = default
                changed = True
        if changed:
            self.save_all([configs[key] for key in sorted(configs)])
        return self.list()

    def init_provider(self, provider: str) -> ChannelBotConfig:
        provider = provider.lower().strip()
        if provider not in SUPPORTED_CHANNEL_PROVIDERS:
            raise ValueError(f"Unsupported channel provider: {provider}")
        configs = {item.provider: item for item in self.ensure_defaults()}
        config = configs.get(provider, DEFAULT_CHANNEL_CONFIGS[provider])
        config.enabled = True
        configs[provider] = config
        self.save_all([configs[key] for key in sorted(configs)])
        return config

    def list(self) -> List[ChannelBotConfig]:
        if not self.path.exists():
            return []
        data = read_yaml(self.path)
        raw_items = data.get("channels", [])
        if not isinstance(raw_items, list):
            raise ValueError(f"channels must be a list: {self.path}")
        return [ChannelBotConfig.from_dict(dict(item)) for item in raw_items]

    def get(self, provider: str) -> ChannelBotConfig:
        provider = provider.lower().strip()
        for config in self.ensure_defaults():
            if config.provider == provider:
                return config
        raise KeyError(f"Unknown channel provider: {provider}")

    def save_all(self, configs: List[ChannelBotConfig]) -> Path:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return write_yaml(
            self.path,
            {
                "version": 1,
                "channels": [item.to_dict() for item in configs],
            },
        )

    def env_summary(self) -> Dict[str, List[str]]:
        summary: Dict[str, List[str]] = {}
        for config in self.ensure_defaults():
            names = [
                config.app_id_env,
                config.app_secret_env,
                config.access_token_env,
                config.verification_token_env,
                config.signing_secret_env,
                config.bot_user_id_env,
            ]
            summary[config.provider] = sorted({name for name in names if name})
        return summary

    def env_configured_summary(self) -> Dict[str, Dict[str, bool]]:
        return {
            provider: {name: bool(os.environ.get(name, "").strip()) for name in names}
            for provider, names in self.env_summary().items()
        }
