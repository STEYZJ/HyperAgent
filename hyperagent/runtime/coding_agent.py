"""Claude-Code-like planning workflow over repository context."""

from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.runtime.agent_loop import AgentLoop
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.repo_context import RepoContextBuilder
from hyperagent.runtime.workspace import HyperAgentWorkspace, utc_now
from hyperagent.schemas import CodingAgentRun


class CodingAgent:
    """Creates saved coding/algorithm planning runs from repo + chat context."""

    def __init__(
        self,
        workspace: HyperAgentWorkspace,
        conversation_store: ConversationStore,
        provider_store: LLMProviderStore,
        prompt_library: Optional[PromptLibrary] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.workspace = workspace
        self.conversations = conversation_store
        self.providers = provider_store
        self.prompt_library = prompt_library
        self.llm_client = llm_client
        self.context_builder = RepoContextBuilder(workspace.project_root)

    def plan(
        self,
        session_id: str,
        provider: str,
        instruction: str,
        model: Optional[str] = None,
        mode: str = "code",
        task_id: Optional[str] = None,
        max_files: int = 20,
        max_preview_chars: int = 1200,
        max_context_chars: int = 18000,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        user: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> CodingAgentRun:
        run_id = self._new_run_id()
        run_dir = self.workspace.workspace_dir / "agent_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        snapshot = self.context_builder.build(
            query=instruction,
            max_files=max_files,
            max_preview_chars=max_preview_chars,
        )
        repo_context_path = run_dir / "repo_context.json"
        repo_context_markdown_path = run_dir / "repo_context.md"
        write_json(repo_context_path, snapshot)
        repo_context_markdown_path.write_text(
            self.context_builder.to_markdown(snapshot),
            encoding="utf-8",
        )

        response_path = run_dir / "agent_turn.json"
        plan_path = run_dir / "plan.md"
        message = self._build_instruction(
            instruction=instruction,
            repo_context_markdown=repo_context_markdown_path.read_text(
                encoding="utf-8"
            ),
        )
        result = AgentLoop(
            self.conversations,
            self.providers,
            self.workspace,
            prompt_library=self.prompt_library,
            llm_client=self.llm_client,
        ).run(
            session_id=session_id,
            provider=provider,
            user_message=message,
            model=model,
            mode=mode,
            task_id=task_id,
            max_context_chars=max_context_chars,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            user=user,
            extra_body=extra_body,
            output_path=response_path,
        )
        plan_text = result.response.content or "\n".join(result.warnings)
        plan_path.write_text(plan_text + "\n", encoding="utf-8")

        run = CodingAgentRun(
            run_id=run_id,
            session_id=session_id,
            provider=result.provider,
            model=result.model,
            mode=mode,
            instruction=instruction,
            task_id=task_id,
            created_at=utc_now(),
            run_dir=str(run_dir),
            repo_context_path=str(repo_context_path),
            repo_context_markdown_path=str(repo_context_markdown_path),
            response_path=str(response_path),
            plan_path=str(plan_path),
            status="planned" if not result.warnings else "warning",
            warnings=list(result.warnings) + list(snapshot.warnings),
        )
        write_json(run_dir / "run.json", run)
        return run

    def _build_instruction(self, instruction: str, repo_context_markdown: str) -> str:
        return (
            "User instruction:\n"
            f"{instruction.strip()}\n\n"
            "Repository context:\n"
            f"{repo_context_markdown}\n"
            "Respond as a local coding agent. Produce a concrete plan with these "
            "sections: Understanding, Relevant Files, Proposed Changes, "
            "Experiment/Test Commands, Risks, and Next Patch Step. The available "
            "controlled local tools are agent-tool read-file, search-code, "
            "run-command, check-patch, and apply-patch. If the request needs paper "
            "content that is not present, ask for a link or downloaded file instead "
            "of inventing details."
        )

    def _new_run_id(self) -> str:
        return f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}"
