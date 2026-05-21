"""Small persistent background job ledger for REPL/TUI orchestration."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.workspace import utc_now


@dataclass
class BackgroundJob:
    job_id: str
    kind: str
    instruction: str
    status: str = "queued"
    created_at: str = ""
    updated_at: str = ""
    session_id: str = ""
    run_path: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BackgroundJob":
        return cls(
            job_id=str(data.get("job_id", "")),
            kind=str(data.get("kind", "")),
            instruction=str(data.get("instruction", "")),
            status=str(data.get("status", "queued")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            session_id=str(data.get("session_id", "")),
            run_path=str(data.get("run_path", "")),
            warnings=[str(v) for v in data.get("warnings", [])],
        )


class BackgroundJobStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = Path(workspace_dir) / "jobs" / "jobs.json"

    def create(
        self,
        *,
        kind: str,
        instruction: str,
        session_id: str = "",
        status: str = "queued",
    ) -> BackgroundJob:
        job = BackgroundJob(
            job_id=f"job-{uuid4().hex[:8]}",
            kind=kind,
            instruction=instruction,
            status=status,
            created_at=utc_now(),
            updated_at=utc_now(),
            session_id=session_id,
        )
        jobs = self.list()
        jobs.append(job)
        self.save_all(jobs)
        return job

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        run_path: Optional[str] = None,
        warning: Optional[str] = None,
    ) -> Optional[BackgroundJob]:
        jobs = self.list()
        matched = None
        for job in jobs:
            if job.job_id != job_id:
                continue
            if status:
                job.status = status
            if run_path is not None:
                job.run_path = run_path
            if warning:
                job.warnings.append(warning)
            job.updated_at = utc_now()
            matched = job
            break
        self.save_all(jobs)
        return matched

    def list(self) -> List[BackgroundJob]:
        if not self.path.exists():
            return []
        data = read_json(self.path)
        raw = data.get("jobs", [])
        if not isinstance(raw, list):
            return []
        return [BackgroundJob.from_dict(item) for item in raw if isinstance(item, dict)]

    def save_all(self, jobs: List[BackgroundJob]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return write_json(self.path, {"jobs": [job.to_dict() for job in jobs]})
