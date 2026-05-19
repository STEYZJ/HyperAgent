"""Project workspace state for the HyperAgent CLI workbench."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from hyperagent.core.io import read_json, read_yaml, write_json, write_yaml
from hyperagent.schemas import ProjectConfig, ResearchTask, WorkspaceStatus


WORKSPACE_DIRNAME = ".hyperagent"


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class HyperAgentWorkspace:
    """Manages local project state under `.hyperagent/`."""

    def __init__(self, project_root: Path = Path(".")) -> None:
        self.project_root = project_root.resolve()
        self.workspace_dir = self.project_root / WORKSPACE_DIRNAME
        self.tasks_dir = self.workspace_dir / "tasks"
        self.artifacts_dir = self.workspace_dir / "artifacts"
        self.config_path = self.workspace_dir / "config.yaml"

    def init(
        self,
        dataset_root: Path,
        output_root: str = "experiments",
        reports_root: str = "reports",
        literature_root: str = "literature/papers",
        default_provider: str = "arxiv",
        default_year_from: Optional[int] = 2024,
    ) -> ProjectConfig:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        config = ProjectConfig(
            dataset_root=str(Path(dataset_root).resolve()),
            output_root=output_root,
            reports_root=reports_root,
            literature_root=literature_root,
            default_provider=default_provider,
            default_year_from=default_year_from,
        )
        write_yaml(self.config_path, config)
        return config

    def is_initialized(self) -> bool:
        return self.config_path.exists()

    def load_config(self) -> ProjectConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"HyperAgent workspace is not initialized: {self.workspace_dir}"
            )
        return ProjectConfig.from_dict(read_yaml(self.config_path))

    def status(self) -> WorkspaceStatus:
        if not self.is_initialized():
            return WorkspaceStatus(
                initialized=False,
                workspace_dir=str(self.workspace_dir),
                config_path=None,
                dataset_root=None,
                task_count=0,
                tasks_by_status={},
            )
        config = self.load_config()
        tasks = list(self.list_tasks())
        by_status: Dict[str, int] = {}
        for task in tasks:
            by_status[task.status] = by_status.get(task.status, 0) + 1
        return WorkspaceStatus(
            initialized=True,
            workspace_dir=str(self.workspace_dir),
            config_path=str(self.config_path),
            dataset_root=config.dataset_root,
            task_count=len(tasks),
            tasks_by_status=by_status,
        )

    def create_task(
        self,
        goal: str,
        dataset: str,
        objective: str,
        keywords: Optional[Iterable[str]] = None,
    ) -> ResearchTask:
        self.load_config()
        now = utc_now()
        task_id = self._new_task_id(dataset)
        task = ResearchTask(
            task_id=task_id,
            goal=goal,
            dataset=dataset,
            objective=objective,
            keywords=[str(v).strip() for v in keywords or [] if str(v).strip()],
            created_at=now,
            updated_at=now,
        )
        self.save_task(task)
        return task

    def save_task(self, task: ResearchTask) -> Path:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        task.updated_at = utc_now()
        return write_json(self.task_path(task.task_id), task)

    def load_task(self, task_id: str) -> ResearchTask:
        return ResearchTask.from_dict(read_json(self.task_path(task_id)))

    def list_tasks(self) -> List[ResearchTask]:
        if not self.tasks_dir.exists():
            return []
        tasks = [
            ResearchTask.from_dict(read_json(path))
            for path in sorted(self.tasks_dir.glob("*.json"))
        ]
        return sorted(tasks, key=lambda task: task.created_at)

    def task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def task_artifact_dir(self, task_id: str) -> Path:
        path = self.artifacts_dir / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_dataset_path(self, dataset: str) -> Path:
        dataset_path = Path(dataset)
        if dataset_path.is_absolute():
            return dataset_path
        config = self.load_config()
        return Path(config.dataset_root) / dataset

    def _new_task_id(self, dataset: str) -> str:
        safe_dataset = "".join(
            char.lower() if char.isalnum() else "-" for char in dataset
        ).strip("-")
        safe_dataset = safe_dataset[:24] or "task"
        return f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{safe_dataset}-{uuid4().hex[:6]}"

