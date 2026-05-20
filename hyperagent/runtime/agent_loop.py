"""Conversation-backed LLM agent loop for HyperAgent."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hyperagent.core.io import write_json
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.llm_usage import LLMUsageLedger
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import (
    AgentTurnResult,
    ConversationMessage,
    ConversationSession,
    LLMMessage,
    ResearchTask,
)


MODE_PROMPTS = {
    "research": "agent_research_loop",
    "code": "code_writer",
    "algorithm": "algorithm_designer",
}

FALLBACK_PROMPTS = {
    "research": (
        "You are HyperAgent, a research agent for hyperspectral image "
        "classification. Use prior conversation, task artifacts, dataset "
        "evidence, and reproducibility constraints before recommending next "
        "actions."
    ),
    "code": (
        "You are HyperAgent in code-writing mode. Propose or implement changes "
        "with clear file boundaries, tests, and rollback-safe reasoning."
    ),
    "algorithm": (
        "You are HyperAgent in algorithm-design mode. Design HSI classification "
        "modules from evidence, state assumptions, define ablations, and explain "
        "how to verify the idea."
    ),
}


class AgentLoop:
    """Runs one LLM-backed turn while preserving conversation state."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        provider_store: LLMProviderStore,
        workspace: HyperAgentWorkspace,
        prompt_library: Optional[PromptLibrary] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.conversations = conversation_store
        self.providers = provider_store
        self.workspace = workspace
        self.prompt_library = prompt_library
        self.llm_client = llm_client or LLMClient()

    def run(
        self,
        session_id: str,
        provider: str,
        user_message: str,
        model: Optional[str] = None,
        mode: str = "research",
        task_id: Optional[str] = None,
        auto_compress: bool = True,
        max_context_chars: int = 12000,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        user: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
    ) -> AgentTurnResult:
        if auto_compress:
            self.conversations.auto_compress(
                session_id,
                max_chars=max_context_chars,
                keep_last=6,
            )
        session = self.conversations.add_message(session_id, "user", user_message)
        messages = self.build_messages(
            session,
            mode=mode,
            task_id=task_id,
            max_context_chars=max_context_chars,
        )
        self.providers.ensure_defaults()
        spec = self.providers.get(provider)
        response = self.llm_client.send(
            spec,
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            user=user,
            extra_body=extra_body,
        )
        LLMUsageLedger(self.workspace.workspace_dir).record_response(
            response,
            spec=spec,
            session_id=session_id,
            event_type="agent_loop.response",
            context_chars=sum(len(message.content) for message in messages),
            metadata={"mode": mode, "task_id": task_id},
        )
        assistant_content = response.content or "\n".join(response.warnings)
        self.conversations.add_message(session_id, "assistant", assistant_content)
        result = AgentTurnResult(
            session_id=session_id,
            provider=spec.name,
            model=model or spec.default_model,
            mode=mode,
            task_id=task_id,
            response=response,
            context_message_count=len(messages),
            context_chars=sum(len(message.content) for message in messages),
            output_path=str(output_path) if output_path else None,
            warnings=list(response.warnings),
        )
        if output_path:
            write_json(output_path, result)
        return result

    def build_messages(
        self,
        session: ConversationSession,
        mode: str = "research",
        task_id: Optional[str] = None,
        max_context_chars: int = 12000,
    ) -> List[LLMMessage]:
        task = self._load_task(task_id)
        system = self._build_system_prompt(mode, task)
        task_context = self._task_context(task) if task else ""
        summary_context = self._summary_context(session)

        fixed_messages = [LLMMessage(role="system", content=system)]
        if summary_context:
            fixed_messages.append(
                LLMMessage(
                    role="system",
                    content="Conversation summaries:\n" + summary_context,
                )
            )
        if task_context:
            fixed_messages.append(
                LLMMessage(
                    role="system",
                    content="Task and artifact context:\n" + task_context,
                )
            )

        used_chars = sum(len(message.content) for message in fixed_messages)
        remaining = max(max_context_chars - used_chars, 1000)
        history = self._trim_history(
            self._conversation_to_llm(session.messages),
            max_chars=remaining,
        )
        return fixed_messages + history

    def _build_system_prompt(self, mode: str, task: Optional[ResearchTask]) -> str:
        prompt_name = MODE_PROMPTS.get(mode, MODE_PROMPTS["research"])
        variables = {
            "task_id": task.task_id if task else "",
            "dataset": task.dataset if task else "",
            "objective": task.objective if task else "",
            "goal": task.goal if task else "",
        }
        if self.prompt_library is not None:
            try:
                rendered = self.prompt_library.render(prompt_name, variables)
            except KeyError:
                rendered = FALLBACK_PROMPTS.get(mode, FALLBACK_PROMPTS["research"])
        else:
            rendered = FALLBACK_PROMPTS.get(mode, FALLBACK_PROMPTS["research"])
        guardrails = (
            "\n\nOperating rules:\n"
            "- Read the prior conversation before answering.\n"
            "- Ground experiment or module suggestions in available artifacts.\n"
            "- When proposing code changes, name the files and tests that should change.\n"
            "- When proposing experiments, include the objective, controlled variable, "
            "dataset split, metrics, and expected evidence.\n"
            "- Do not invent unavailable paper contents. Ask for a link or downloaded file "
            "when a paper is needed but not present."
        )
        return rendered.strip() + guardrails

    def _summary_context(self, session: ConversationSession) -> str:
        lines = []
        for summary in session.summaries:
            lines.append(
                f"[{summary.summary_id}] {summary.message_count} messages via "
                f"{summary.method}:\n{summary.content}"
            )
        return "\n\n".join(lines)

    def _load_task(self, task_id: Optional[str]) -> Optional[ResearchTask]:
        if not task_id:
            return None
        return self.workspace.load_task(task_id)

    def _task_context(self, task: ResearchTask) -> str:
        lines = [
            f"Task id: {task.task_id}",
            f"Goal: {task.goal}",
            f"Dataset: {task.dataset}",
            f"Objective: {task.objective}",
            f"Status: {task.status}",
            "Keywords: " + (", ".join(task.keywords) if task.keywords else "none"),
        ]
        if task.notes:
            lines.append("Notes:\n" + "\n".join(f"- {note}" for note in task.notes))
        if task.artifacts:
            lines.append("Artifacts:")
            for name, raw_path in sorted(task.artifacts.items()):
                path = self._resolve_artifact_path(raw_path)
                preview = self._artifact_preview(path)
                lines.append(f"- {name}: {path}\n{preview}")
        return "\n".join(lines)

    def _resolve_artifact_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.workspace.project_root / path

    def _artifact_preview(self, path: Path, max_chars: int = 1800) -> str:
        if not path.exists():
            return "  preview: missing file"
        if path.is_dir():
            children = sorted(child.name for child in path.iterdir())[:20]
            return "  preview: directory entries: " + ", ".join(children)
        if path.stat().st_size > 5_000_000:
            return f"  preview: skipped large artifact ({path.stat().st_size} bytes)"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"  preview: unreadable artifact: {exc}"
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...<truncated>"
        return "  preview:\n" + self._indent(text, "  ")

    def _conversation_to_llm(
        self,
        messages: Iterable[ConversationMessage],
    ) -> List[LLMMessage]:
        converted: List[LLMMessage] = []
        for message in messages:
            role = (
                message.role
                if message.role in {"system", "user", "assistant"}
                else "user"
            )
            content = message.content
            if message.role not in {"system", "user", "assistant"}:
                content = f"[{message.role}] {content}"
            converted.append(LLMMessage(role=role, content=content))
        return converted

    def _trim_history(
        self,
        messages: List[LLMMessage],
        max_chars: int,
    ) -> List[LLMMessage]:
        selected: List[LLMMessage] = []
        total = 0
        for message in reversed(messages):
            size = len(message.content)
            if selected and total + size > max_chars:
                break
            selected.append(message)
            total += size
            if total >= max_chars:
                break
        return list(reversed(selected))

    def _indent(self, text: str, prefix: str) -> str:
        return "\n".join(prefix + line for line in text.splitlines())
