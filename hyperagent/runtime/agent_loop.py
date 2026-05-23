"""Conversation-backed LLM agent loop for HyperAgent."""

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hyperagent.core.io import write_json
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.events import RuntimeEventLog
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.llm_usage import LLMUsageLedger
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.workspace import HyperAgentWorkspace, utc_now
from hyperagent.schemas import (
    AgentTurnTiming,
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
        self.event_log = RuntimeEventLog(workspace.workspace_dir)

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
        thinking_displayed: Optional[bool] = None,
        reasoning_content_expanded: Optional[bool] = None,
        fallback_providers: Optional[Iterable[str]] = None,
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
        turn_started_at = utc_now()
        started = time.monotonic()
        self.event_log.append(
            "agent_loop.start",
            source="agent_loop",
            session_id=session_id,
            status="running",
            message=user_message[:500],
            payload={"provider": provider, "model": model, "mode": mode},
        )
        spec, response, attempted_providers = self._send_with_fallback(
            provider,
            fallback_providers or [],
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
        elapsed_sec = round(time.monotonic() - started, 4)
        turn_completed_at = utc_now()
        timing = AgentTurnTiming(
            turn_started_at=turn_started_at,
            turn_completed_at=turn_completed_at,
            model_wait_elapsed_sec=elapsed_sec,
        )
        context_chars = sum(len(message.content) for message in messages)
        context_message_count = len(messages)
        turn_warnings = self._attempt_warnings(attempted_providers, response.warnings)
        LLMUsageLedger(self.workspace.workspace_dir).record_response(
            response,
            spec=spec,
            session_id=session_id,
            event_type="agent_loop.response",
            context_chars=context_chars,
            metadata={
                "mode": mode,
                "task_id": task_id,
                "model_wait_elapsed_sec": elapsed_sec,
                "attempted_providers": attempted_providers,
            },
        )
        self.event_log.append(
            "agent_loop.response",
            source="agent_loop",
            session_id=session_id,
            status="ok" if not turn_warnings else "warning",
            message=(response.content or "\n".join(turn_warnings))[:500],
            payload={
                "provider": spec.name,
                "model": response.model,
                "mode": mode,
                "context_chars": context_chars,
                "context_message_count": context_message_count,
                "model_wait_elapsed_sec": elapsed_sec,
                "usage": response.usage,
                "warnings": turn_warnings,
                "attempted_providers": attempted_providers,
            },
        )
        assistant_content = response.content or "\n".join(turn_warnings)
        expanded = bool(
            reasoning_content_expanded
            if reasoning_content_expanded is not None
            else thinking_displayed
        )
        self.conversations.add_message(
            session_id,
            "assistant",
            assistant_content,
            metadata={
                "turn_started_at": turn_started_at,
                "turn_completed_at": turn_completed_at,
                "model_wait_elapsed_sec": elapsed_sec,
                "context_chars": context_chars,
                "context_message_count": context_message_count,
                "provider": spec.name,
                "model": response.model,
                "mode": mode,
                "reasoning_content_chars": len(response.reasoning_content or ""),
                "thinking_displayed": expanded,
                "reasoning_content_expanded": expanded,
            },
        )
        result = AgentTurnResult(
            session_id=session_id,
            provider=spec.name,
            model=response.model,
            mode=mode,
            task_id=task_id,
            response=response,
            context_message_count=context_message_count,
            context_chars=context_chars,
            output_path=str(output_path) if output_path else None,
            warnings=turn_warnings,
            timing=timing,
        )
        if output_path:
            write_json(output_path, result)
        return result

    def _send_with_fallback(
        self,
        provider: str,
        fallback_providers: Iterable[str],
        messages: Iterable[LLMMessage],
        **kwargs: Any,
    ):
        candidates: List[str] = []
        for name in [provider, *list(fallback_providers)]:
            normalized = str(name or "").strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        if not candidates:
            candidates = [provider]
        attempted: List[Dict[str, Any]] = []
        final_spec = self.providers.get(candidates[0])
        final_response = None
        for index, name in enumerate(candidates):
            spec = self.providers.get(name)
            send_kwargs = dict(kwargs)
            if name != provider:
                send_kwargs["model"] = None
            response = self.llm_client.send(spec, messages, **send_kwargs)
            attempted.append(
                {
                    "provider": spec.name,
                    "model": response.model,
                    "warnings": list(response.warnings),
                    "content_chars": len(response.content or ""),
                }
            )
            final_spec = spec
            final_response = response
            if index >= len(candidates) - 1 or not self._should_try_fallback(response):
                break
        return final_spec, final_response, attempted

    def _should_try_fallback(self, response) -> bool:
        if str(response.content or "").strip():
            return False
        warnings = "\n".join(str(item) for item in response.warnings)
        markers = (
            "Missing required environment variable",
            "HTTPError",
            "URLError",
            "TimeoutError",
        )
        return any(marker in warnings for marker in markers)

    def _attempt_warnings(
        self,
        attempted_providers: List[Dict[str, Any]],
        final_warnings: Iterable[str],
    ) -> List[str]:
        warnings = [str(item) for item in final_warnings]
        if len(attempted_providers) <= 1:
            return warnings
        route = " -> ".join(str(item.get("provider", "")) for item in attempted_providers)
        combined = [f"LLM provider fallback attempted: {route}"]
        for item in attempted_providers[:-1]:
            for warning in item.get("warnings", []):
                combined.append(f"{item.get('provider')}: {warning}")
        combined.extend(warnings)
        return combined

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
