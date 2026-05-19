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

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMessage":
        return cls(role=str(data["role"]), content=str(data["content"]))


@dataclass
class LLMRequest:
    provider: str
    model: Optional[str]
    messages: List[LLMMessage]
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
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
