"""Schemas for Claude-Code-like local agent runs."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RepoFileContext:
    path: str
    size_bytes: int
    language: str
    preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepoFileContext":
        return cls(
            path=str(data["path"]),
            size_bytes=int(data.get("size_bytes", 0)),
            language=str(data.get("language", "")),
            preview=str(data.get("preview", "")),
        )


@dataclass
class RepoSnapshot:
    project_root: str
    generated_at: str
    query: str = ""
    is_git_repo: bool = False
    branch: str = ""
    commit: str = ""
    dirty_files: List[str] = field(default_factory=list)
    file_count: int = 0
    selected_files: List[RepoFileContext] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepoSnapshot":
        return cls(
            project_root=str(data["project_root"]),
            generated_at=str(data["generated_at"]),
            query=str(data.get("query", "")),
            is_git_repo=bool(data.get("is_git_repo", False)),
            branch=str(data.get("branch", "")),
            commit=str(data.get("commit", "")),
            dirty_files=[str(v) for v in data.get("dirty_files", [])],
            file_count=int(data.get("file_count", 0)),
            selected_files=[
                RepoFileContext.from_dict(item)
                for item in data.get("selected_files", [])
            ],
            warnings=[str(v) for v in data.get("warnings", [])],
        )


@dataclass
class CodingAgentRun:
    run_id: str
    session_id: str
    provider: str
    model: str
    mode: str
    instruction: str
    created_at: str
    run_dir: str
    task_id: Optional[str] = None
    repo_context_path: Optional[str] = None
    repo_context_markdown_path: Optional[str] = None
    response_path: Optional[str] = None
    plan_path: Optional[str] = None
    status: str = "planned"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingAgentRun":
        return cls(
            run_id=str(data["run_id"]),
            session_id=str(data["session_id"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            mode=str(data["mode"]),
            instruction=str(data["instruction"]),
            created_at=str(data["created_at"]),
            run_dir=str(data["run_dir"]),
            task_id=(
                None if data.get("task_id") is None else str(data.get("task_id"))
            ),
            repo_context_path=data.get("repo_context_path"),
            repo_context_markdown_path=data.get("repo_context_markdown_path"),
            response_path=data.get("response_path"),
            plan_path=data.get("plan_path"),
            status=str(data.get("status", "planned")),
            warnings=[str(v) for v in data.get("warnings", [])],
        )


@dataclass
class AgentToolCall:
    call_id: str
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentToolCall":
        return cls(
            call_id=str(data["call_id"]),
            tool_name=str(data["tool_name"]),
            args=dict(data.get("args", {})),
            created_at=str(data.get("created_at", "")),
            run_id=None if data.get("run_id") is None else str(data.get("run_id")),
        )


@dataclass
class AgentToolResult:
    call_id: str
    tool_name: str
    status: str
    created_at: str
    content: str = ""
    exit_code: Optional[int] = None
    artifact_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentToolResult":
        return cls(
            call_id=str(data["call_id"]),
            tool_name=str(data["tool_name"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            content=str(data.get("content", "")),
            exit_code=(
                None if data.get("exit_code") is None else int(data.get("exit_code"))
            ),
            artifact_path=data.get("artifact_path"),
            warnings=[str(v) for v in data.get("warnings", [])],
        )


@dataclass
class AgentActionStep:
    step_index: int
    response_content: str
    action: str
    status: str
    tool_name: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[AgentToolResult] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentActionStep":
        raw_result = data.get("tool_result")
        return cls(
            step_index=int(data["step_index"]),
            response_content=str(data.get("response_content", "")),
            action=str(data.get("action", "")),
            status=str(data.get("status", "")),
            tool_name=None if data.get("tool_name") is None else str(data.get("tool_name")),
            args=dict(data.get("args", {})),
            tool_result=(
                AgentToolResult.from_dict(dict(raw_result))
                if isinstance(raw_result, dict)
                else None
            ),
            warnings=[str(v) for v in data.get("warnings", [])],
        )


@dataclass
class AgentActionRun:
    run_id: str
    session_id: str
    provider: str
    model: str
    instruction: str
    created_at: str
    run_dir: str
    status: str = "running"
    task_id: Optional[str] = None
    final_response: str = ""
    steps: List[AgentActionStep] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentActionRun":
        return cls(
            run_id=str(data["run_id"]),
            session_id=str(data["session_id"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            instruction=str(data["instruction"]),
            created_at=str(data["created_at"]),
            run_dir=str(data["run_dir"]),
            status=str(data.get("status", "running")),
            task_id=None if data.get("task_id") is None else str(data.get("task_id")),
            final_response=str(data.get("final_response", "")),
            steps=[
                AgentActionStep.from_dict(dict(item))
                for item in data.get("steps", [])
            ],
            warnings=[str(v) for v in data.get("warnings", [])],
        )
