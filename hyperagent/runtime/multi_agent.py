"""Executable multi-agent task runtime."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.general_agent import GeneralAgentRunner
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.subagents import SubagentRuntimeRegistry
from hyperagent.runtime.workspace import HyperAgentWorkspace, utc_now
from hyperagent.schemas import MultiAgentRoleRun, MultiAgentTaskRun


class MultiAgentTaskRunner:
    """Runs selected subagents sequentially or in parallel and aggregates traces."""

    def __init__(
        self,
        workspace: HyperAgentWorkspace,
        conversations: ConversationStore,
        providers: LLMProviderStore,
        *,
        llm_client: Optional[LLMClient] = None,
        permission_policy: str = "session-ask",
        permission_callback: Optional[object] = None,
        session_permission_cache: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.workspace = workspace
        self.conversations = conversations
        self.providers = providers
        self.llm_client = llm_client
        self.permission_policy = permission_policy
        self.permission_callback = permission_callback
        self.session_permission_cache = session_permission_cache
        self.extensions = RuntimeExtensionStore(workspace.workspace_dir)
        self.registry = SubagentRuntimeRegistry(workspace.workspace_dir)

    def run(
        self,
        *,
        session_id: str,
        provider: str,
        instruction: str,
        agents: Sequence[str],
        model: Optional[str] = None,
        profile: str = "",
        mode: str = "sequential",
        task_id: Optional[str] = None,
        max_steps: int = 3,
        max_files: int = 12,
        max_preview_chars: int = 1000,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        loop_mode: str = "standard",
        token_budget: Optional[int] = None,
        max_depth: int = 1,
        max_concurrent: int = 4,
        parent_subagent_id: str = "",
        depth: int = 0,
        delegation_role: str = "leaf",
        llm_kwargs: Optional[Dict[str, object]] = None,
    ) -> MultiAgentTaskRun:
        self.providers.ensure_defaults()
        spec = self.providers.get(provider)
        run_id = f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}"
        run_dir = self.workspace.workspace_dir / "agent_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run = MultiAgentTaskRun(
            run_id=run_id,
            session_id=session_id,
            provider=provider,
            model=model or spec.default_model,
            instruction=instruction,
            mode=mode,
            created_at=utc_now(),
            run_dir=str(run_dir),
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            active_registry_path=str(self.registry.path),
        )
        if self.registry.is_paused():
            run.status = "blocked"
            run.paused = True
            run.warnings.append("Subagent spawning is paused.")
            self._persist(run)
            return run
        if depth > max_depth:
            run.status = "blocked"
            run.warnings.append(
                f"Subagent depth {depth} exceeds max_depth={max_depth}."
            )
            self._persist(run)
            return run
        agent_names = [str(agent).strip() for agent in agents if str(agent).strip()]
        if not agent_names:
            run.status = "blocked"
            run.warnings.append("No subagents were selected for the task run.")
            self._persist(run)
            return run
        if mode in {"parallel", "tree"}:
            self._run_parallel(
                run,
                agent_names,
                provider,
                model,
                profile,
                task_id,
                max_steps,
                max_files,
                max_preview_chars,
                temperature,
                max_tokens,
                loop_mode,
                token_budget,
                max_depth,
                max_concurrent,
                parent_subagent_id,
                depth,
                delegation_role,
                llm_kwargs or {},
            )
        else:
            self._run_sequential(
                run,
                agent_names,
                provider,
                model,
                profile,
                task_id,
                max_steps,
                max_files,
                max_preview_chars,
                temperature,
                max_tokens,
                loop_mode,
                token_budget,
                max_depth,
                max_concurrent,
                parent_subagent_id,
                depth,
                delegation_role,
                llm_kwargs or {},
            )
        if not run.role_runs:
            run.status = "failed"
        elif any(item.status == "completed" for item in run.role_runs):
            run.status = "completed"
        else:
            run.status = "failed"
        run.aggregate_response = self._aggregate(run)
        self._persist(run)
        return run

    def _run_sequential(self, run: MultiAgentTaskRun, agent_names: List[str], provider: str, model: Optional[str], profile: str, task_id: Optional[str], max_steps: int, max_files: int, max_preview_chars: int, temperature: float, max_tokens: Optional[int], loop_mode: str, token_budget: Optional[int], max_depth: int, max_concurrent: int, parent_subagent_id: str, depth: int, delegation_role: str, llm_kwargs: Dict[str, object]) -> None:
        del max_concurrent
        for agent_name in agent_names:
            run.role_runs.append(
                self._run_one(run.run_id, agent_name, run.instruction, provider, model, profile, task_id, max_steps, max_files, max_preview_chars, temperature, max_tokens, loop_mode, token_budget, max_depth, parent_subagent_id, depth, delegation_role, llm_kwargs)
            )
            self._persist(run)

    def _run_parallel(self, run: MultiAgentTaskRun, agent_names: List[str], provider: str, model: Optional[str], profile: str, task_id: Optional[str], max_steps: int, max_files: int, max_preview_chars: int, temperature: float, max_tokens: Optional[int], loop_mode: str, token_budget: Optional[int], max_depth: int, max_concurrent: int, parent_subagent_id: str, depth: int, delegation_role: str, llm_kwargs: Dict[str, object]) -> None:
        with ThreadPoolExecutor(max_workers=min(len(agent_names), max(1, max_concurrent))) as pool:
            futures = {
                pool.submit(
                    self._run_one,
                    run.run_id,
                    agent_name,
                    run.instruction,
                    provider,
                    model,
                    profile,
                    task_id,
                    max_steps,
                    max_files,
                    max_preview_chars,
                    temperature,
                    max_tokens,
                    loop_mode,
                    token_budget,
                    max_depth,
                    parent_subagent_id,
                    depth,
                    delegation_role,
                    llm_kwargs,
                ): agent_name
                for agent_name in agent_names
            }
            for future in as_completed(futures):
                run.role_runs.append(future.result())
                self._persist(run)

    def _run_one(
        self,
        run_id: str,
        agent_name: str,
        instruction: str,
        provider: str,
        model: Optional[str],
        profile: str,
        task_id: Optional[str],
        max_steps: int,
        max_files: int,
        max_preview_chars: int,
        temperature: float,
        max_tokens: Optional[int],
        loop_mode: str,
        token_budget: Optional[int],
        max_depth: int,
        parent_subagent_id: str,
        depth: int,
        delegation_role: str,
        llm_kwargs: Dict[str, object],
    ) -> MultiAgentRoleRun:
        subagent_id = f"sa-{uuid4().hex[:8]}"
        started_at = utc_now()
        if self.registry.is_paused():
            return MultiAgentRoleRun(
                agent_id="",
                agent_name=agent_name,
                role="unknown",
                instruction=instruction,
                status="blocked",
                subagent_id=subagent_id,
                parent_id=parent_subagent_id,
                depth=depth + 1,
                delegation_role=delegation_role,
                started_at=started_at,
                completed_at=utc_now(),
                warnings=["Subagent spawning is paused."],
            )
        if depth + 1 > max_depth:
            return MultiAgentRoleRun(
                agent_id="",
                agent_name=agent_name,
                role="unknown",
                instruction=instruction,
                status="blocked",
                subagent_id=subagent_id,
                parent_id=parent_subagent_id,
                depth=depth + 1,
                delegation_role=delegation_role,
                started_at=started_at,
                completed_at=utc_now(),
                warnings=[f"Subagent depth exceeds max_depth={max_depth}."],
            )
        try:
            agent_meta = self._agent_metadata(agent_name)
            self.registry.register(
                subagent_id=subagent_id,
                agent_name=str(agent_meta.get("name", agent_name)),
                role=str(agent_meta.get("role", "")),
                instruction=instruction,
                run_id=run_id,
                parent_id=parent_subagent_id,
                depth=depth + 1,
                delegation_role=delegation_role,
                model=model or "",
            )
            if self.registry.should_stop(subagent_id):
                self.registry.complete(subagent_id, status="stopped")
                return MultiAgentRoleRun(
                    agent_id=str(agent_meta.get("id", "")),
                    agent_name=str(agent_meta.get("name", agent_name)),
                    role=str(agent_meta.get("role", "")),
                    instruction=instruction,
                    status="stopped",
                    subagent_id=subagent_id,
                    parent_id=parent_subagent_id,
                    depth=depth + 1,
                    delegation_role=delegation_role,
                    started_at=started_at,
                    completed_at=utc_now(),
                    warnings=["Stop requested before subagent start."],
                )
            agent_run = GeneralAgentRunner(
                self.workspace,
                self.conversations,
                self.providers,
                permission_policy=self.permission_policy,
                permission_callback=self.permission_callback,
                session_permission_cache=self.session_permission_cache,
                llm_client=self.llm_client,
            ).run(
                agent_name,
                instruction,
                session_id=None,
                provider=provider,
                model=model,
                profile=profile,
                task_id=task_id,
                max_steps=max_steps,
                max_files=max_files,
                max_preview_chars=max_preview_chars,
                temperature=temperature,
                max_tokens=max_tokens,
                loop_mode=loop_mode,
                token_budget=token_budget,
                llm_kwargs=llm_kwargs,
            )
            final_response = ""
            if agent_run.action_run_path:
                try:
                    import json

                    data = json.loads(Path(agent_run.action_run_path).read_text(encoding="utf-8"))
                    final_response = str(data.get("final_response", ""))
                except Exception:
                    final_response = ""
            completed = utc_now()
            self.registry.complete(
                subagent_id,
                status=agent_run.status,
                action_run_path=agent_run.action_run_path or "",
                warnings=agent_run.warnings,
            )
            return MultiAgentRoleRun(
                agent_id=agent_run.agent_id,
                agent_name=agent_run.agent_name,
                role=agent_run.role,
                instruction=instruction,
                status=agent_run.status,
                subagent_id=subagent_id,
                parent_id=parent_subagent_id,
                depth=depth + 1,
                delegation_role=delegation_role,
                started_at=started_at,
                completed_at=completed,
                action_run_path=agent_run.action_run_path,
                final_response=final_response,
                warnings=agent_run.warnings,
            )
        except Exception as exc:
            self.registry.complete(
                subagent_id,
                status="failed",
                warnings=[f"{type(exc).__name__}: {exc}"],
            )
            return MultiAgentRoleRun(
                agent_id="",
                agent_name=agent_name,
                role="unknown",
                instruction=instruction,
                status="failed",
                subagent_id=subagent_id,
                parent_id=parent_subagent_id,
                depth=depth + 1,
                delegation_role=delegation_role,
                started_at=started_at,
                completed_at=utc_now(),
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

    def _aggregate(self, run: MultiAgentTaskRun) -> str:
        lines = [f"Multi-agent task run {run.run_id} ({run.mode})"]
        for item in run.role_runs:
            lines.append(f"- {item.agent_name} [{item.role}] status={item.status}")
            if item.final_response:
                lines.append(f"  {item.final_response[:500]}")
            for warning in item.warnings:
                lines.append(f"  warning: {warning}")
        return "\n".join(lines)

    def _persist(self, run: MultiAgentTaskRun) -> Path:
        return write_json(Path(run.run_dir) / "multi_agent_run.json", run)

    def _agent_metadata(self, agent_name: str) -> Dict[str, object]:
        for agent in self.extensions.list_subagents():
            if agent_name in {str(agent.get("id", "")), str(agent.get("name", ""))}:
                return agent
        return {"id": "", "name": agent_name, "role": "unknown"}
