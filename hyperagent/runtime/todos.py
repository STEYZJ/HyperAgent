"""Session-scoped TodoWrite state for agent workflows."""

from pathlib import Path
from typing import Iterable, List, Optional
from uuid import uuid4

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import TodoItem, TodoList


VALID_TODO_STATUSES = {"pending", "in_progress", "completed", "blocked"}
VALID_TODO_PRIORITIES = {"low", "normal", "high"}


class TodoStore:
    """Persists lightweight todo lists under `.hyperagent/todos`."""

    def __init__(self, workspace_dir: Path) -> None:
        self.root = workspace_dir / "todos"

    def path_for(self, owner: str) -> Path:
        safe_owner = self._safe_owner(owner)
        return self.root / f"{safe_owner}.json"

    def load(self, owner: str = "project") -> TodoList:
        path = self.path_for(owner)
        if not path.exists():
            return TodoList(owner=self._safe_owner(owner), updated_at=utc_now())
        return TodoList.from_dict(read_json(path))

    def save(self, todo_list: TodoList) -> Path:
        todo_list.owner = self._safe_owner(todo_list.owner)
        todo_list.updated_at = utc_now()
        return write_json(self.path_for(todo_list.owner), todo_list)

    def replace(
        self,
        owner: str,
        items: Iterable[dict],
    ) -> TodoList:
        safe_owner = self._safe_owner(owner)
        now = utc_now()
        normalized: List[TodoItem] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            status = str(item.get("status", "pending")).strip() or "pending"
            priority = str(item.get("priority", "normal")).strip() or "normal"
            normalized.append(
                TodoItem(
                    id=str(item.get("id") or f"todo-{index}-{uuid4().hex[:6]}"),
                    content=content,
                    status=status if status in VALID_TODO_STATUSES else "pending",
                    priority=priority if priority in VALID_TODO_PRIORITIES else "normal",
                    owner=safe_owner,
                    created_at=str(item.get("created_at") or now),
                    updated_at=now,
                    metadata=dict(item.get("metadata", {})),
                )
            )
        todo_list = TodoList(owner=safe_owner, updated_at=now, items=normalized)
        self.save(todo_list)
        return todo_list

    def clear(self, owner: str = "project") -> TodoList:
        todo_list = TodoList(owner=self._safe_owner(owner), updated_at=utc_now(), items=[])
        self.save(todo_list)
        return todo_list

    def export_markdown(self, owner: str = "project", output: Optional[Path] = None) -> Path:
        todo_list = self.load(owner)
        target = output or (self.root / f"{todo_list.owner}.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# HyperAgent Todos - {todo_list.owner}", ""]
        if not todo_list.items:
            lines.append("_No todos._")
        for item in todo_list.items:
            marker = "x" if item.status == "completed" else " "
            lines.append(
                f"- [{marker}] ({item.status}, {item.priority}) {item.content}"
            )
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def _safe_owner(self, owner: str) -> str:
        value = str(owner or "project").strip()
        return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value) or "project"
