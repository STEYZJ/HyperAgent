"""LLM provider configuration, request payload builders, and HTTP client."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hyperagent.core.io import read_json, write_json
from hyperagent.schemas import LLMMessage, LLMProviderSpec, LLMRequest, LLMResponse


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
        default_model="deepseek-v4-flash",
        metadata={
            "recommended_models": ["deepseek-v4-flash", "deepseek-v4-pro"],
            "supports_thinking": True,
            "thinking_types": ["enabled", "disabled"],
            "reasoning_effort": ["high", "max"],
            "supports_response_format": True,
            "supports_tools": True,
            "supports_prompt_cache_usage": True,
            "prompt_cache_usage_fields": [
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ],
            "reasonix_profiles": [
                "reasonix-cheap",
                "reasonix-balanced",
                "reasonix-deep",
            ],
            "cache_first_guidance": (
                "Keep stable project memory, dataset cards, spectral rules, and "
                "tool schemas before volatile user/tool output to improve prefix "
                "cache reuse."
            ),
        },
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
            providers = self._merge_defaults(self.list())
            write_json(self.path, {"providers": [item.to_dict() for item in providers]})
            return providers
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

    def _merge_defaults(self, providers: List[LLMProviderSpec]) -> List[LLMProviderSpec]:
        by_name: Dict[str, LLMProviderSpec] = {item.name: item for item in providers}
        changed = False
        for default in DEFAULT_PROVIDERS:
            current = by_name.get(default.name)
            if current is None:
                by_name[default.name] = default
                changed = True
                continue
            for key, value in default.metadata.items():
                if key not in current.metadata:
                    current.metadata[key] = value
                    changed = True
            if (
                current.name == "deepseek"
                and current.default_model == "deepseek-chat"
                and default.default_model != current.default_model
            ):
                current.default_model = default.default_model
                changed = True
        if changed:
            return [by_name[item.name] for item in DEFAULT_PROVIDERS if item.name in by_name] + [
                item
                for name, item in by_name.items()
                if name not in {default.name for default in DEFAULT_PROVIDERS}
            ]
        return providers


class LLMRequestBuilder:
    """Builds vendor-specific request payloads for dry-run or future clients."""

    def build(
        self,
        spec: LLMProviderSpec,
        messages: Iterable[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        user: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        request = LLMRequest(
            provider=spec.name,
            model=model or spec.default_model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            user=user,
            extra_body=dict(extra_body or {}),
        )
        if spec.kind == "openai_compatible":
            body: Dict[str, Any] = {
                "model": request.model,
                "messages": [message.to_dict() for message in request.messages],
            }
            if not self._skip_sampling_for_thinking(spec, request.thinking):
                body["temperature"] = request.temperature
                if request.top_p is not None:
                    body["top_p"] = request.top_p
            if request.max_tokens is not None:
                body["max_tokens"] = request.max_tokens
            if request.response_format is not None:
                body["response_format"] = request.response_format
            if request.thinking is not None:
                body["thinking"] = request.thinking
            if request.reasoning_effort:
                body["reasoning_effort"] = request.reasoning_effort
            if request.user:
                body["user"] = request.user
            body.update(request.extra_body)
            return {
                "url": spec.base_url,
                "api_key_env": spec.api_key_env,
                "headers": {
                    "Authorization": "Bearer ${%s}" % spec.api_key_env,
                    "Content-Type": "application/json",
                },
                "json": body,
            }
        if spec.kind == "anthropic_messages":
            system_messages = [m.content for m in request.messages if m.role == "system"]
            chat_messages = [
                m.to_dict() for m in request.messages if m.role in ("user", "assistant")
            ]
            body = {
                "model": request.model,
                "messages": chat_messages,
                "system": "\n\n".join(system_messages),
                "temperature": request.temperature,
                "max_tokens": max_tokens or 1024,
            }
            if request.top_p is not None:
                body["top_p"] = request.top_p
            body.update(request.extra_body)
            return {
                "url": spec.base_url,
                "api_key_env": spec.api_key_env,
                "headers": {
                    "x-api-key": "${%s}" % spec.api_key_env,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                "json": body,
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
            generation_config: Dict[str, Any] = {
                "temperature": request.temperature,
                **(
                    {"maxOutputTokens": max_tokens}
                    if max_tokens is not None
                    else {}
                ),
            }
            if request.top_p is not None:
                generation_config["topP"] = request.top_p
            body = {
                "contents": contents,
                **(
                    {"systemInstruction": {"parts": [{"text": system_text}]}}
                    if system_text
                    else {}
                ),
                "generationConfig": generation_config,
            }
            body.update(request.extra_body)
            return {
                "url": f"{spec.base_url}/models/{request.model}:generateContent?key=${{{spec.api_key_env}}}",
                "api_key_env": spec.api_key_env,
                "headers": {},
                "json": body,
            }
        raise ValueError(f"Unsupported provider kind: {spec.kind}")

    def _skip_sampling_for_thinking(
        self,
        spec: LLMProviderSpec,
        thinking: Optional[Dict[str, Any]],
    ) -> bool:
        if spec.name != "deepseek" or not thinking:
            return False
        return str(thinking.get("type", "")).lower() == "enabled"


class LLMClient:
    """Minimal stdlib HTTP client for configured LLM providers."""

    def __init__(self, timeout_sec: int = 60) -> None:
        self.timeout_sec = timeout_sec
        self.builder = LLMRequestBuilder()

    def send(
        self,
        spec: LLMProviderSpec,
        messages: Iterable[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        user: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        payload = self.builder.build(
            spec,
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            user=user,
            extra_body=extra_body,
        )
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            return LLMResponse(
                provider=spec.name,
                model=model or spec.default_model,
                content="",
                warnings=[f"Missing required environment variable: {spec.api_key_env}"],
            )

        url = str(payload["url"]).replace("${" + spec.api_key_env + "}", api_key)
        headers = {
            str(key): str(value).replace("${" + spec.api_key_env + "}", api_key)
            for key, value in dict(payload["headers"]).items()
        }
        body = json.dumps(payload["json"]).encode("utf-8")
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return LLMResponse(
                provider=spec.name,
                model=model or spec.default_model,
                content="",
                warnings=[f"HTTPError {exc.code}: {detail[:500]}"],
            )
        except URLError as exc:
            return LLMResponse(
                provider=spec.name,
                model=model or spec.default_model,
                content="",
                warnings=[f"URLError: {exc}"],
            )
        except TimeoutError as exc:
            return LLMResponse(
                provider=spec.name,
                model=model or spec.default_model,
                content="",
                warnings=[f"TimeoutError: {exc}"],
            )

        return LLMResponse(
            provider=spec.name,
            model=model or spec.default_model,
            content=self._extract_content(spec, raw),
            reasoning_content=self._extract_reasoning_content(spec, raw),
            tool_calls=self._extract_tool_calls(spec, raw),
            usage=self._extract_usage(raw),
            message=self._extract_message(spec, raw),
            raw=raw,
        )

    def _extract_message(
        self,
        spec: LLMProviderSpec,
        raw: Dict[str, object],
    ) -> Dict[str, Any]:
        if spec.kind != "openai_compatible":
            return {}
        choices = raw.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return {}
        first = choices[0]
        if not isinstance(first, dict):
            return {}
        message = first.get("message", {})
        return dict(message) if isinstance(message, dict) else {}

    def _extract_content(self, spec: LLMProviderSpec, raw: Dict[str, object]) -> str:
        if spec.kind == "openai_compatible":
            message = self._extract_message(spec, raw)
            content = message.get("content", "")
            return "" if content is None else str(content)
        if spec.kind == "anthropic_messages":
            blocks = raw.get("content", [])
            if isinstance(blocks, list):
                texts = []
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(str(block.get("text", "")))
                return "\n".join(texts)
        if spec.kind == "google_generate_content":
            candidates = raw.get("candidates", [])
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    content = first.get("content", {})
                    if isinstance(content, dict):
                        parts = content.get("parts", [])
                        if isinstance(parts, list):
                            return "".join(
                                str(part.get("text", ""))
                                for part in parts
                                if isinstance(part, dict)
                            )
        return ""

    def _extract_reasoning_content(
        self,
        spec: LLMProviderSpec,
        raw: Dict[str, object],
    ) -> str:
        if spec.kind != "openai_compatible":
            return ""
        message = self._extract_message(spec, raw)
        reasoning = message.get("reasoning_content", "")
        return "" if reasoning is None else str(reasoning)

    def _extract_tool_calls(
        self,
        spec: LLMProviderSpec,
        raw: Dict[str, object],
    ) -> List[Dict[str, Any]]:
        if spec.kind != "openai_compatible":
            return []
        message = self._extract_message(spec, raw)
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return []
        return [
            self._normalize_tool_call(item)
            for item in tool_calls
            if isinstance(item, dict)
        ]

    def _extract_usage(self, raw: Dict[str, object]) -> Dict[str, Any]:
        usage = raw.get("usage", {})
        return dict(usage) if isinstance(usage, dict) else {}

    def _normalize_tool_call(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OpenAI-compatible tool calls across provider quirks."""

        normalized = dict(item)
        function = normalized.get("function")
        if isinstance(function, dict):
            function = dict(function)
            arguments = function.get("arguments")
            if isinstance(arguments, dict):
                function["arguments"] = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            elif arguments is None:
                function["arguments"] = "{}"
            else:
                function["arguments"] = str(arguments)
            if function.get("name") is not None:
                function["name"] = str(function.get("name"))
            normalized["function"] = function
        return normalized
