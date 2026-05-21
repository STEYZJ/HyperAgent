"""Channel event routing into HyperAgent conversations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.agent_loop import AgentLoop
from hyperagent.runtime.channels.config import ChannelConfigStore
from hyperagent.runtime.channels.registry import (
    ChannelPlatformRegistry,
    register_builtin_channel_platforms,
)
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import (
    AgentTurnResult,
    ChannelBotConfig,
    ChannelEventResult,
    ChannelInboundMessage,
    ChannelOutboundMessage,
)


AgentResponder = Callable[[ChannelInboundMessage, str, ChannelBotConfig], str]


class ChannelSessionStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "channel_sessions.json"

    def get_or_create(
        self,
        inbound: ChannelInboundMessage,
        conversations: ConversationStore,
    ) -> str:
        data = self._load()
        key = self.key_for(inbound)
        if key in data:
            return str(data[key])
        session = conversations.new(
            f"{inbound.provider}:{inbound.chat_id}:{inbound.channel_user_id}"
        )
        session.metadata.update(
            {
                "channel_provider": inbound.provider,
                "channel_chat_id": inbound.chat_id,
                "channel_user_id": inbound.channel_user_id,
                "channel_key": key,
            }
        )
        conversations.save(session)
        data[key] = session.session_id
        write_json(self.path, {"sessions": data})
        return session.session_id

    def key_for(self, inbound: ChannelInboundMessage) -> str:
        return f"channel:{inbound.provider}:{inbound.chat_id}:{inbound.channel_user_id}"

    def _load(self) -> Dict[str, str]:
        if not self.path.exists():
            return {}
        data = read_json(self.path)
        raw = data.get("sessions", {})
        if not isinstance(raw, dict):
            return {}
        return {str(key): str(value) for key, value in raw.items()}


class ChannelRouter:
    """Routes verified channel events into chat-only AgentLoop turns."""

    def __init__(
        self,
        workspace: HyperAgentWorkspace,
        conversations: ConversationStore,
        providers: LLMProviderStore,
        prompt_library: Optional[PromptLibrary] = None,
        config_store: Optional[ChannelConfigStore] = None,
        platform_registry: Optional[ChannelPlatformRegistry] = None,
        responder: Optional[AgentResponder] = None,
    ) -> None:
        self.workspace = workspace
        self.conversations = conversations
        self.providers = providers
        self.prompt_library = prompt_library
        self.config_store = config_store or ChannelConfigStore(workspace.workspace_dir)
        self.platform_registry = platform_registry or register_builtin_channel_platforms()
        self.session_store = ChannelSessionStore(workspace.workspace_dir)
        self.responder = responder

    def handle_webhook(
        self,
        provider: str,
        payload: Dict[str, object],
        headers: Dict[str, str],
        body: bytes,
        *,
        dry_run_agent: bool = False,
    ) -> ChannelEventResult:
        try:
            config = self.config_store.get(provider)
            adapter = self._adapter(config)
        except (KeyError, ValueError) as exc:
            return ChannelEventResult(
                provider=provider,
                status="error",
                error=str(exc),
            )
        if not config.enabled:
            return ChannelEventResult(
                provider=provider,
                status="error",
                error=f"channel provider is disabled: {provider}",
            )
        parsed = adapter.handle_webhook(dict(payload), headers, body)
        if parsed.status != "received" or parsed.inbound is None:
            return parsed
        return self.handle_message(parsed.inbound, config, dry_run_agent=dry_run_agent)

    def handle_message(
        self,
        inbound: ChannelInboundMessage,
        config: Optional[ChannelBotConfig] = None,
        *,
        dry_run_agent: bool = False,
    ) -> ChannelEventResult:
        config = config or self.config_store.get(inbound.provider)
        session_id = self.session_store.get_or_create(inbound, self.conversations)
        warnings = []
        if dry_run_agent:
            answer = f"dry-run: {inbound.text}"
        elif self.responder is not None:
            answer = self.responder(inbound, session_id, config)
        else:
            answer, turn = self._run_agent(inbound, session_id, config)
            warnings.extend(turn.warnings)

        outbound = ChannelOutboundMessage(
            provider=inbound.provider,
            chat_id=inbound.chat_id,
            text=answer,
            reply_to_message_id=inbound.message_id or None,
            thread_id=inbound.thread_id,
            chat_type=inbound.chat_type,
            metadata={"session_id": session_id},
        )
        send_result = self._adapter(config).send_message(outbound)
        send_result.session_id = session_id
        send_result.inbound = inbound
        send_result.warnings = warnings + send_result.warnings
        return send_result

    def _run_agent(
        self,
        inbound: ChannelInboundMessage,
        session_id: str,
        config: ChannelBotConfig,
    ) -> tuple[str, AgentTurnResult]:
        self.providers.ensure_defaults()
        result = AgentLoop(
            self.conversations,
            self.providers,
            self.workspace,
            prompt_library=self.prompt_library,
        ).run(
            session_id=session_id,
            provider=config.default_llm_provider,
            model=config.default_model,
            user_message=inbound.text,
            mode=config.default_mode,
            max_context_chars=config.max_context_chars,
        )
        return result.response.content, result

    def _adapter(self, config: ChannelBotConfig):
        return self.platform_registry.create_adapter(config)
