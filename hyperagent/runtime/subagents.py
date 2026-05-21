"""Runtime registry for active and recently completed subagents."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

from hyperagent.core.io import read_json, write_json
from hyperagent.runtime.workspace import utc_now


@dataclass
class SubagentState:
    subagent_id: str
    agent_name: str
    role: str
    status: str
    instruction: str = ""
    parent_id: str = ""
    run_id: str = ""
    depth: int = 0
    delegation_role: str = "leaf"
    model: str = ""
    action_run_path: str = ""
    started_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SubagentState":
        return cls(
            subagent_id=str(data.get("subagent_id", "")),
            agent_name=str(data.get("agent_name", "")),
            role=str(data.get("role", "")),
            status=str(data.get("status", "running")),
            instruction=str(data.get("instruction", "")),
            parent_id=str(data.get("parent_id", "")),
            run_id=str(data.get("run_id", "")),
            depth=int(data.get("depth", 0)),
            delegation_role=str(data.get("delegation_role", "leaf")),
            model=str(data.get("model", "")),
            action_run_path=str(data.get("action_run_path", "")),
            started_at=str(data.get("started_at", "")),
            updated_at=str(data.get("updated_at", "")),
            completed_at=str(data.get("completed_at", "")),
            warnings=[str(v) for v in data.get("warnings", [])],
        )


class SubagentRuntimeRegistry:
    """Persisted subagent control plane used by REPL, TUI, and task tools."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.root = self.workspace_dir / "agent_runs"
        self.path = self.root / "active_subagents.json"
        self.control_path = self.root / "subagent_control.json"
        self._lock = RLock()

    def register(
        self,
        *,
        subagent_id: str,
        agent_name: str,
        role: str,
        instruction: str,
        run_id: str,
        parent_id: str = "",
        depth: int = 0,
        delegation_role: str = "leaf",
        model: str = "",
    ) -> SubagentState:
        state = SubagentState(
            subagent_id=subagent_id,
            agent_name=agent_name,
            role=role,
            status="running",
            instruction=instruction,
            parent_id=parent_id,
            run_id=run_id,
            depth=depth,
            delegation_role=delegation_role,
            model=model,
            started_at=utc_now(),
            updated_at=utc_now(),
        )
        with self._lock:
            data = self._load()
            data[state.subagent_id] = state
            self._save(data)
        return state

    def update(
        self,
        subagent_id: str,
        *,
        status: Optional[str] = None,
        action_run_path: Optional[str] = None,
        warning: Optional[str] = None,
    ) -> Optional[SubagentState]:
        with self._lock:
            data = self._load()
            state = data.get(subagent_id)
            if state is None:
                return None
            if status:
                state.status = status
                if status in {"completed", "failed", "blocked", "stopped"}:
                    state.completed_at = utc_now()
            if action_run_path is not None:
                state.action_run_path = action_run_path
            if warning:
                state.warnings.append(warning)
            state.updated_at = utc_now()
            data[subagent_id] = state
            self._save(data)
            return state

    def complete(
        self,
        subagent_id: str,
        *,
        status: str,
        action_run_path: str = "",
        warnings: Optional[List[str]] = None,
    ) -> Optional[SubagentState]:
        state = self.update(subagent_id, status=status, action_run_path=action_run_path)
        if state is None:
            return None
        for warning in warnings or []:
            state = self.update(subagent_id, warning=str(warning)) or state
        return state

    def list(self, include_completed: bool = True) -> List[SubagentState]:
        with self._lock:
            states = list(self._load().values())
        if not include_completed:
            states = [
                state
                for state in states
                if state.status not in {"completed", "failed", "blocked", "stopped"}
            ]
        return sorted(states, key=lambda item: (item.started_at, item.subagent_id))

    def pause(self, reason: str = "") -> Dict[str, object]:
        return self._write_control(paused=True, stop_ids=[], reason=reason)

    def resume(self) -> Dict[str, object]:
        current = self.control()
        return self._write_control(
            paused=False,
            stop_ids=[str(v) for v in current.get("stop_ids", [])],
            reason="",
        )

    def stop(self, subagent_id: str) -> Dict[str, object]:
        current = self.control()
        stop_ids = {str(v) for v in current.get("stop_ids", [])}
        stop_ids.add(subagent_id)
        self.update(subagent_id, status="stopped", warning="stop requested")
        return self._write_control(
            paused=bool(current.get("paused", False)),
            stop_ids=sorted(stop_ids),
            reason=str(current.get("reason", "")),
        )

    def control(self) -> Dict[str, object]:
        if not self.control_path.exists():
            return {
                "paused": False,
                "stop_ids": [],
                "reason": "",
                "updated_at": "",
            }
        data = read_json(self.control_path)
        return {
            "paused": bool(data.get("paused", False)),
            "stop_ids": [str(v) for v in data.get("stop_ids", [])],
            "reason": str(data.get("reason", "")),
            "updated_at": str(data.get("updated_at", "")),
        }

    def is_paused(self) -> bool:
        return bool(self.control().get("paused", False))

    def should_stop(self, subagent_id: str) -> bool:
        return subagent_id in set(self.control().get("stop_ids", []))

    def _write_control(
        self,
        *,
        paused: bool,
        stop_ids: List[str],
        reason: str,
    ) -> Dict[str, object]:
        payload = {
            "paused": bool(paused),
            "stop_ids": list(stop_ids),
            "reason": reason,
            "updated_at": utc_now(),
        }
        self.control_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.control_path, payload)
        return payload

    def _load(self) -> Dict[str, SubagentState]:
        if not self.path.exists():
            return {}
        data = read_json(self.path)
        raw = data.get("subagents", [])
        if not isinstance(raw, list):
            return {}
        states = [SubagentState.from_dict(item) for item in raw if isinstance(item, dict)]
        return {state.subagent_id: state for state in states if state.subagent_id}

    def _save(self, states: Dict[str, SubagentState]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            self.path,
            {"subagents": [state.to_dict() for state in states.values()]},
        )
