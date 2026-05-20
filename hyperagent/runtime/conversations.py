"""Conversation persistence, archive, delete, and compression."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import ConversationMessage, ConversationSession, ConversationSummary


@dataclass
class ContextCompressionStatus:
    session_id: str
    message_count: int
    summary_count: int
    current_chars: int
    max_chars: int
    trigger_chars: int
    should_compress: bool
    keep_last: int


class ConversationStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.root = workspace_dir / "sessions"
        self.active_dir = self.root / "active"
        self.archive_dir = self.root / "archive"
        self.deleted_dir = self.root / "deleted"

    def new(self, title: str) -> ConversationSession:
        now = utc_now()
        session = ConversationSession(
            session_id=f"{now.replace(':', '').replace('-', '')}-{uuid4().hex[:6]}",
            title=title,
            status="active",
            created_at=now,
            updated_at=now,
        )
        self.save(session)
        return session

    def save(self, session: ConversationSession) -> Path:
        session.updated_at = utc_now()
        target_dir = self._dir_for_status(session.status)
        target_dir.mkdir(parents=True, exist_ok=True)
        return write_json(target_dir / f"{session.session_id}.json", session)

    def add_message(self, session_id: str, role: str, content: str) -> ConversationSession:
        session = self.load(session_id)
        session.messages.append(
            ConversationMessage(role=role, content=content, created_at=utc_now())
        )
        self.save(session)
        return session

    def load(self, session_id: str) -> ConversationSession:
        for directory in (self.active_dir, self.archive_dir, self.deleted_dir):
            path = directory / f"{session_id}.json"
            if path.exists():
                return ConversationSession.from_dict(read_json(path))
        raise FileNotFoundError(f"Conversation session not found: {session_id}")

    def list(self, include_archived: bool = False) -> List[ConversationSession]:
        directories = [self.active_dir]
        if include_archived:
            directories.append(self.archive_dir)
        sessions: List[ConversationSession] = []
        for directory in directories:
            if directory.exists():
                sessions.extend(
                    ConversationSession.from_dict(read_json(path))
                    for path in sorted(directory.glob("*.json"))
                )
        return sessions

    def archive(self, session_id: str) -> ConversationSession:
        session = self.load(session_id)
        self._remove_existing(session_id)
        session.status = "archived"
        self.save(session)
        return session

    def delete(self, session_id: str, hard: bool = False) -> None:
        if hard:
            self._remove_existing(session_id)
            return
        session = self.load(session_id)
        self._remove_existing(session_id)
        session.status = "deleted"
        self.save(session)

    def compress(
        self,
        session_id: str,
        keep_last: int = 4,
        max_chars: Optional[int] = None,
    ) -> ConversationSession:
        session = self.load(session_id)
        if max_chars is not None and self._message_chars(session) <= max_chars:
            return session
        if len(session.messages) <= keep_last:
            return session
        old_messages = session.messages[:-keep_last]
        kept_messages = session.messages[-keep_last:]
        summary_text = self._extractive_summary(old_messages)
        session.summaries.append(
            ConversationSummary(
                summary_id=f"summary-{uuid4().hex[:8]}",
                created_at=utc_now(),
                message_count=len(old_messages),
                content=summary_text,
                method="extractive",
            )
        )
        session.messages = kept_messages
        self.save(session)
        return session

    def clear(self, session_id: str) -> ConversationSession:
        session = self.load(session_id)
        session.messages = []
        session.summaries = []
        self.save(session)
        return session

    def auto_compress(
        self,
        session_id: str,
        max_chars: int = 12000,
        keep_last: int = 6,
        trigger_ratio: float = 1.0,
        min_messages: int = 8,
    ) -> ConversationSession:
        status = self.context_status(
            session_id,
            max_chars=max_chars,
            keep_last=keep_last,
            trigger_ratio=trigger_ratio,
            min_messages=min_messages,
        )
        if not status.should_compress:
            return self.load(session_id)
        return self.compress(session_id, keep_last=keep_last)

    def context_status(
        self,
        session_id: str,
        max_chars: int = 12000,
        keep_last: int = 6,
        trigger_ratio: float = 1.0,
        min_messages: int = 8,
    ) -> ContextCompressionStatus:
        session = self.load(session_id)
        current_chars = self._message_chars(session)
        trigger_chars = max(int(max_chars * trigger_ratio), 1)
        should_compress = (
            current_chars > trigger_chars
            and len(session.messages) > keep_last
            and len(session.messages) >= min_messages
        )
        return ContextCompressionStatus(
            session_id=session.session_id,
            message_count=len(session.messages),
            summary_count=len(session.summaries),
            current_chars=current_chars,
            max_chars=max_chars,
            trigger_chars=trigger_chars,
            should_compress=should_compress,
            keep_last=keep_last,
        )

    def _dir_for_status(self, status: str) -> Path:
        if status == "archived":
            return self.archive_dir
        if status == "deleted":
            return self.deleted_dir
        return self.active_dir

    def _remove_existing(self, session_id: str) -> None:
        for directory in (self.active_dir, self.archive_dir, self.deleted_dir):
            path = directory / f"{session_id}.json"
            if path.exists():
                path.unlink()

    def _message_chars(self, session: ConversationSession) -> int:
        return sum(len(message.content) for message in session.messages)

    def _extractive_summary(self, messages: List[ConversationMessage]) -> str:
        lines = []
        for message in messages:
            content = " ".join(message.content.split())
            if len(content) > 220:
                content = content[:217] + "..."
            lines.append(f"- {message.role}: {content}")
        return "\n".join(lines)
