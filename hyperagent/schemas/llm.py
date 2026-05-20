"""LLM provider and agent-turn schemas."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMProviderSpec:
    name: str
    kind: str
    base_url: str
    api_key_env: str
    default_model: str
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMProviderSpec":
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            base_url=str(data["base_url"]),
            api_key_env=str(data["api_key_env"]),
            default_model=str(data["default_model"]),
            headers={str(k): str(v) for k, v in data.get("headers", {}).items()},
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class LLMMessage:
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.reasoning_content is not None:
            payload["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        payload.update(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMessage":
        known = {
            "role",
            "content",
            "name",
            "tool_call_id",
            "reasoning_content",
            "tool_calls",
            "metadata",
        }
        content = data.get("content", "")
        metadata = {str(key): value for key, value in data.get("metadata", {}).items()}
        metadata.update({str(key): value for key, value in data.items() if key not in known})
        raw_tool_calls = data.get("tool_calls", [])
        return cls(
            role=str(data["role"]),
            content="" if content is None else str(content),
            name=str(data["name"]) if data.get("name") else None,
            tool_call_id=(
                str(data["tool_call_id"]) if data.get("tool_call_id") else None
            ),
            reasoning_content=(
                str(data["reasoning_content"])
                if data.get("reasoning_content") is not None
                else None
            ),
            tool_calls=[
                dict(item)
                for item in raw_tool_calls
                if isinstance(item, dict)
            ]
            if isinstance(raw_tool_calls, list)
            else [],
            metadata=metadata,
        )


@dataclass
class LLMRequest:
    provider: str
    model: Optional[str]
    messages: List[LLMMessage]
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    response_format: Optional[Dict[str, Any]] = None
    thinking: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None
    user: Optional[str] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    message: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTurnResult:
    session_id: str
    provider: str
    model: str
    mode: str
    task_id: Optional[str]
    response: LLMResponse
    context_message_count: int
    context_chars: int
    output_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
