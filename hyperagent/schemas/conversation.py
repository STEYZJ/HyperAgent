"""Conversation persistence schemas."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ConversationMessage:
    role: str
    content: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        return cls(
            role=str(data["role"]),
            content=str(data["content"]),
            created_at=str(data["created_at"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ConversationSummary:
    summary_id: str
    created_at: str
    message_count: int
    content: str
    method: str = "extractive"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        return cls(
            summary_id=str(data["summary_id"]),
            created_at=str(data["created_at"]),
            message_count=int(data["message_count"]),
            content=str(data["content"]),
            method=str(data.get("method", "extractive")),
        )


@dataclass
class ConversationSession:
    session_id: str
    title: str
    status: str = "active"
    messages: List[ConversationMessage] = field(default_factory=list)
    summaries: List[ConversationSummary] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        return cls(
            session_id=str(data["session_id"]),
            title=str(data["title"]),
            status=str(data.get("status", "active")),
            messages=[
                ConversationMessage.from_dict(item)
                for item in data.get("messages", [])
            ],
            summaries=[
                ConversationSummary.from_dict(item)
                for item in data.get("summaries", [])
            ],
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            metadata=dict(data.get("metadata", {})),
        )

