"""Runtime workspace schemas for the HyperAgent CLI workbench."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProjectConfig:
    dataset_root: str
    output_root: str = "experiments"
    reports_root: str = "reports"
    literature_root: str = "literature/papers"
    default_provider: str = "arxiv"
    default_year_from: Optional[int] = 2024
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        return cls(
            dataset_root=str(data["dataset_root"]),
            output_root=str(data.get("output_root", "experiments")),
            reports_root=str(data.get("reports_root", "reports")),
            literature_root=str(data.get("literature_root", "literature/papers")),
            default_provider=str(data.get("default_provider", "arxiv")),
            default_year_from=(
                None
                if data.get("default_year_from") is None
                else int(data.get("default_year_from"))
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ResearchTask:
    task_id: str
    goal: str
    dataset: str
    objective: str
    status: str = "created"
    keywords: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchTask":
        return cls(
            task_id=str(data["task_id"]),
            goal=str(data["goal"]),
            dataset=str(data["dataset"]),
            objective=str(data["objective"]),
            status=str(data.get("status", "created")),
            keywords=[str(v) for v in data.get("keywords", [])],
            artifacts={str(k): str(v) for k, v in data.get("artifacts", {}).items()},
            notes=[str(v) for v in data.get("notes", [])],
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


@dataclass
class WorkspaceStatus:
    initialized: bool
    workspace_dir: str
    config_path: Optional[str]
    dataset_root: Optional[str]
    task_count: int
    tasks_by_status: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

