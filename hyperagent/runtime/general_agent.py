"""Executable project subagent runtime."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.deepseek_reasonix import get_reasonix_profile
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.hooks import HookEngine
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.workspace import HyperAgentWorkspace, utc_now
from hyperagent.schemas import GeneralAgentRun


class GeneralAgentRunner:
    """Runs one registered subagent through the controlled action loop."""

    def __init__(
        self,
        workspace: HyperAgentWorkspace,
        conversations: ConversationStore,
        providers: LLMProviderStore,
        *,
        permission_policy: str = "session-ask",
        permission_callback: Optional[Any] = None,
        session_permission_cache: Optional[Dict[str, bool]] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.workspace = workspace
        self.conversations = conversations
        self.providers = providers
        self.permission_policy = permission_policy
        self.permission_callback = permission_callback
        self.session_permission_cache = (
            session_permission_cache if session_permission_cache is not None else {}
        )
        self.llm_client = llm_client
        self.extensions = RuntimeExtensionStore(workspace.workspace_dir)

    def run(
        self,
        agent_ref: str,
        instruction: str,
        *,
        session_id: Optional[str] = None,
        provider: str = "deepseek",
        model: Optional[str] = None,
        profile: Optional[str] = None,
        task_id: Optional[str] = None,
        max_steps: int = 3,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        max_files: int = 12,
        max_preview_chars: int = 1000,
        loop_mode: str = "standard",
        token_budget: Optional[int] = None,
        llm_kwargs: Optional[Dict[str, object]] = None,
    ) -> GeneralAgentRun:
        agent = self._find_agent(agent_ref)
        self.providers.ensure_defaults()
        spec = self.providers.get(provider)
        profile_name = profile or str(agent.get("profile", ""))
        profile_spec = get_reasonix_profile(profile_name)
        selected_model = (
            model
            or str(agent.get("model", ""))
            or (profile_spec.model if profile_spec else "")
            or spec.default_model
        )
        run_id = f"general-{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}"
        run_dir = self.workspace.workspace_dir / "agent_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if session_id:
            self.conversations.load(session_id)
        else:
            session_id = self.conversations.new(
                f"{agent.get('name', 'subagent')} run"
            ).session_id

        run = GeneralAgentRun(
            run_id=run_id,
            agent_id=str(agent.get("id", "")),
            agent_name=str(agent.get("name", "")),
            role=str(agent.get("role", "")),
            instruction=instruction,
            created_at=utc_now(),
            run_dir=str(run_dir),
            provider=provider,
            model=selected_model,
            profile=profile_name,
            session_id=session_id,
            permission_policy=self.permission_policy,
        )
        write_json(run_dir / "agent_run.json", run)

        executor = SafeAgentToolExecutor(
            self.workspace.project_root,
            self.workspace.workspace_dir,
            permission_policy=self.permission_policy,
            permission_callback=self.permission_callback,
            session_permission_cache=self.session_permission_cache,
            allow_arbitrary_commands=True,
            hook_engine=HookEngine(self.workspace.workspace_dir),
        )
        action_loop = AgentActionLoop(
            self.conversations,
            self.providers,
            self.workspace,
            llm_client=self.llm_client,
            tool_executor=executor,
        )
        role_instruction = self._role_instruction(agent, instruction)
        runtime_kwargs = dict(llm_kwargs or {})
        if runtime_kwargs.get("thinking") is None:
            runtime_kwargs["thinking"] = (
                {"type": profile_spec.thinking}
                if profile_spec and profile_spec.thinking
                else None
            )
        if runtime_kwargs.get("reasoning_effort") is None:
            runtime_kwargs["reasoning_effort"] = (
                profile_spec.reasoning_effort if profile_spec else None
            )
        action_run = action_loop.run(
            session_id=session_id,
            provider=provider,
            instruction=role_instruction,
            model=selected_model,
            task_id=task_id,
            max_steps=max_steps,
            max_files=max_files,
            max_preview_chars=max_preview_chars,
            temperature=temperature,
            max_tokens=max_tokens,
            loop_mode=loop_mode,
            token_budget=token_budget,
            **runtime_kwargs,
        )
        action_path = Path(action_run.run_dir) / "action_run.json"
        run.action_run_path = str(action_path)
        run.status = action_run.status
        run.tool_artifacts = self._tool_artifacts(action_run)
        run.warnings = list(action_run.warnings)
        write_json(run_dir / "agent_run.json", run)
        return run

    def _find_agent(self, agent_ref: str) -> Dict[str, object]:
        for agent in self.extensions.list_subagents():
            if agent_ref in {str(agent.get("id", "")), str(agent.get("name", ""))}:
                return agent
        raise KeyError(f"Unknown subagent: {agent_ref}")

    def _role_instruction(self, agent: Dict[str, object], instruction: str) -> str:
        tools = ", ".join(str(v) for v in agent.get("tools", [])) or "unspecified"
        prompt = str(agent.get("prompt", "")).strip()
        prompt_block = f"\nAgent prompt:\n{prompt}\n" if prompt else ""
        return (
            f"You are HyperAgent subagent `{agent.get('name')}`.\n"
            f"Role: {agent.get('role')}.\n"
            f"Declared tools: {tools}.\n"
            f"{prompt_block}"
            "Act only on the user's instruction, use tools when useful, and stop "
            "with a concise final answer when the task is handled.\n\n"
            f"User instruction:\n{instruction}"
        )

    def _tool_artifacts(self, action_run) -> List[str]:
        artifacts: List[str] = []
        for step in action_run.steps:
            if step.tool_result and step.tool_result.artifact_path:
                artifacts.append(step.tool_result.artifact_path)
        return artifacts
