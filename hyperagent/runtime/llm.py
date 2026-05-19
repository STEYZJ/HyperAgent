"""LLM provider configuration and request payload builders."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from hyperagent.core.io import read_json, write_json
from hyperagent.schemas import LLMMessage, LLMProviderSpec, LLMRequest


DEFAULT_PROVIDERS = [
    LLMProviderSpec(
        name="openai",
        kind="openai_compatible",
        base_url="https://api.openai.com/v1/chat/completions",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4.1",
    ),
    LLMProviderSpec(
        name="anthropic",
        kind="anthropic_messages",
        base_url="https://api.anthropic.com/v1/messages",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-3-5-sonnet-latest",
    ),
    LLMProviderSpec(
        name="deepseek",
        kind="openai_compatible",
        base_url="https://api.deepseek.com/chat/completions",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    LLMProviderSpec(
        name="openrouter",
        kind="openai_compatible",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="OPENROUTER_API_KEY",
        default_model="openai/gpt-4.1",
    ),
    LLMProviderSpec(
        name="qwen",
        kind="openai_compatible",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen-plus",
    ),
    LLMProviderSpec(
        name="gemini",
        kind="google_generate_content",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GOOGLE_API_KEY",
        default_model="gemini-1.5-pro",
    ),
]


class LLMProviderStore:
    """Stores provider specs without importing vendor SDKs."""

    def __init__(self, workspace_dir: Path) -> None:
        self.path = workspace_dir / "llm_providers.json"

    def ensure_defaults(self) -> List[LLMProviderSpec]:
        if self.path.exists():
            return self.list()
        write_json(self.path, {"providers": [item.to_dict() for item in DEFAULT_PROVIDERS]})
        return list(DEFAULT_PROVIDERS)

    def list(self) -> List[LLMProviderSpec]:
        if not self.path.exists():
            return []
        data = read_json(self.path)
        return [LLMProviderSpec.from_dict(item) for item in data.get("providers", [])]

    def get(self, name: str) -> LLMProviderSpec:
        for provider in self.list():
            if provider.name == name:
                return provider
        raise KeyError(f"Unknown LLM provider: {name}")

    def upsert(self, provider: LLMProviderSpec) -> None:
        providers: Dict[str, LLMProviderSpec] = {item.name: item for item in self.list()}
        providers[provider.name] = provider
        write_json(self.path, {"providers": [item.to_dict() for item in providers.values()]})


class LLMRequestBuilder:
    """Builds vendor-specific request payloads for dry-run or future clients."""

    def build(
        self,
        spec: LLMProviderSpec,
        messages: Iterable[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        request = LLMRequest(
            provider=spec.name,
            model=model or spec.default_model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if spec.kind == "openai_compatible":
            return {
                "url": spec.base_url,
                "api_key_env": spec.api_key_env,
                "headers": {"Authorization": "Bearer ${%s}" % spec.api_key_env},
                "json": {
                    "model": request.model,
                    "messages": [message.to_dict() for message in request.messages],
                    "temperature": request.temperature,
                    **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                },
            }
        if spec.kind == "anthropic_messages":
            system_messages = [m.content for m in request.messages if m.role == "system"]
            chat_messages = [
                m.to_dict() for m in request.messages if m.role in ("user", "assistant")
            ]
            return {
                "url": spec.base_url,
                "api_key_env": spec.api_key_env,
                "headers": {
                    "x-api-key": "${%s}" % spec.api_key_env,
                    "anthropic-version": "2023-06-01",
                },
                "json": {
                    "model": request.model,
                    "messages": chat_messages,
                    "system": "\n\n".join(system_messages),
                    "temperature": request.temperature,
                    "max_tokens": max_tokens or 1024,
                },
            }
        if spec.kind == "google_generate_content":
            contents = [
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": [{"text": message.content}],
                }
                for message in request.messages
                if message.role != "system"
            ]
            system_text = "\n\n".join(
                message.content for message in request.messages if message.role == "system"
            )
            return {
                "url": f"{spec.base_url}/models/{request.model}:generateContent?key=${{{spec.api_key_env}}}",
                "api_key_env": spec.api_key_env,
                "headers": {},
                "json": {
                    "contents": contents,
                    **(
                        {"systemInstruction": {"parts": [{"text": system_text}]}}
                        if system_text
                        else {}
                    ),
                    "generationConfig": {
                        "temperature": request.temperature,
                        **(
                            {"maxOutputTokens": max_tokens}
                            if max_tokens is not None
                            else {}
                        ),
                    },
                },
            }
        raise ValueError(f"Unsupported provider kind: {spec.kind}")

