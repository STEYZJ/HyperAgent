"""LLM-driven controlled tool-call loop."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.repo_context import RepoContextBuilder
from hyperagent.runtime.workspace import HyperAgentWorkspace, utc_now
from hyperagent.schemas import (
    AgentActionRun,
    AgentActionStep,
    AgentToolResult,
    LLMMessage,
)


ACTION_SYSTEM_PROMPT = """You are HyperAgent's controlled action loop.
You help with hyperspectral image classification research by selecting one safe local tool at a time.

Allowed tools:
- read_file: {"path": "relative/path", "start_line": 1, "max_lines": 120}
- search_code: {"query": "text", "path": ".", "max_results": 30}
- run_command: {"argv": ["python", "-m", "unittest", "discover", "-s", "tests"], "timeout_sec": 60}
- check_patch: {"patch_text": "unified diff"}
- apply_patch: {"patch_text": "unified diff"}

Return exactly one JSON object and no prose:
{"thought": "brief reason", "action": "tool", "tool_name": "search_code", "args": {"query": "ExperimentSuiteRunner"}}
or:
{"thought": "brief reason", "action": "final", "final": "answer or next decision"}

Do not request unsafe shell commands. Ground experiment decisions in saved artifacts when they are available.
"""


class AgentActionLoop:
    """Runs short LLM -> tool -> LLM loops with auditable local tools."""

    def __init__(
        self,
        session_store: ConversationStore,
        llm_store: LLMProviderStore,
        workspace: HyperAgentWorkspace,
        *,
        llm_client: Optional[LLMClient] = None,
        tool_executor: Optional[SafeAgentToolExecutor] = None,
    ) -> None:
        self.session_store = session_store
        self.llm_store = llm_store
        self.workspace = workspace
        self.llm_client = llm_client or LLMClient()
        self.tool_executor = tool_executor or SafeAgentToolExecutor(
            workspace.project_root,
            workspace.workspace_dir,
        )

    def run(
        self,
        session_id: str,
        provider: str,
        instruction: str,
        *,
        model: Optional[str] = None,
        task_id: Optional[str] = None,
        max_steps: int = 3,
        max_files: int = 12,
        max_preview_chars: int = 1000,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        user: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AgentActionRun:
        self.llm_store.ensure_defaults()
        spec = self.llm_store.get(provider)
        run_id = f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}"
        run_dir = self.workspace.workspace_dir / "agent_action_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run = AgentActionRun(
            run_id=run_id,
            session_id=session_id,
            provider=provider,
            model=model or spec.default_model,
            instruction=instruction,
            created_at=utc_now(),
            run_dir=str(run_dir),
            task_id=task_id,
        )

        self.session_store.add_message(session_id, "user", instruction)
        messages = self._build_messages(
            session_id,
            instruction,
            task_id=task_id,
            max_files=max_files,
            max_preview_chars=max_preview_chars,
        )

        for step_index in range(1, max(max_steps, 1) + 1):
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
            if response.warnings:
                run.status = "failed"
                run.warnings.extend(response.warnings)
                self._persist(run)
                return run

            parsed, parse_warning = self._parse_action(response.content)
            warnings = [parse_warning] if parse_warning else []
            action = str(parsed.get("action", "final"))
            if action == "final":
                final = str(parsed.get("final", response.content))
                run.final_response = final
                run.status = "completed"
                run.steps.append(
                    AgentActionStep(
                        step_index=step_index,
                        response_content=response.content,
                        action="final",
                        status="completed",
                        warnings=warnings,
                    )
                )
                self.session_store.add_message(session_id, "assistant", final)
                self._persist(run)
                return run

            if action != "tool":
                run.status = "failed"
                run.warnings.append(f"Unsupported action: {action}")
                run.steps.append(
                    AgentActionStep(
                        step_index=step_index,
                        response_content=response.content,
                        action=action,
                        status="failed",
                        warnings=warnings + [f"Unsupported action: {action}"],
                    )
                )
                self._persist(run)
                return run

            tool_name = str(parsed.get("tool_name", ""))
            args = dict(parsed.get("args", {}))
            result = self._execute_tool(tool_name, args, run_id)
            run.steps.append(
                AgentActionStep(
                    step_index=step_index,
                    response_content=response.content,
                    action="tool",
                    status=result.status,
                    tool_name=tool_name,
                    args=args,
                    tool_result=result,
                    warnings=warnings + result.warnings,
                )
            )
            self.session_store.add_message(session_id, "assistant", response.content)
            self.session_store.add_message(
                session_id,
                "tool",
                f"{tool_name} status={result.status}\n{result.content}",
            )
            messages.append(LLMMessage(role="assistant", content=response.content))
            messages.append(
                LLMMessage(
                    role="user",
                    content=(
                        f"Tool result for {tool_name}: status={result.status}\n"
                        f"{result.content}\nReturn the next JSON action."
                    ),
                )
            )
            self._persist(run)

        run.status = "max_steps_reached"
        run.warnings.append("Maximum action-loop steps reached before final response.")
        self._persist(run)
        return run

    def _build_messages(
        self,
        session_id: str,
        instruction: str,
        *,
        task_id: Optional[str],
        max_files: int,
        max_preview_chars: int,
    ) -> List[LLMMessage]:
        session = self.session_store.load(session_id)
        repo_context = RepoContextBuilder(self.workspace.project_root).to_markdown(
            RepoContextBuilder(self.workspace.project_root).build(
                query=instruction,
                max_files=max_files,
                max_preview_chars=max_preview_chars,
            )
        )
        context_parts = ["Repository context:\n" + repo_context]
        task_context = self._task_context(task_id)
        if task_context:
            context_parts.append("Task/artifact context:\n" + task_context)
        messages = [
            LLMMessage(role="system", content=ACTION_SYSTEM_PROMPT),
            LLMMessage(role="user", content="\n\n".join(context_parts)),
        ]
        for summary in session.summaries:
            messages.append(
                LLMMessage(
                    role="user",
                    content=f"Conversation summary {summary.summary_id}:\n{summary.content}",
                )
            )
        for message in session.messages[-8:]:
            role = message.role if message.role in {"user", "assistant"} else "user"
            messages.append(LLMMessage(role=role, content=message.content))
        return messages

    def _task_context(self, task_id: Optional[str]) -> str:
        if not task_id:
            return ""
        try:
            task = self.workspace.load_task(task_id)
        except Exception as exc:
            return f"Task {task_id} could not be loaded: {exc}"
        lines = [
            f"Task: {task.task_id}",
            f"Dataset: {task.dataset}",
            f"Objective: {task.objective}",
            f"Goal: {task.goal}",
        ]
        for name, raw_path in sorted(task.artifacts.items()):
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.workspace.project_root / path
            preview = ""
            if path.exists() and path.is_file() and path.stat().st_size < 200000:
                preview = path.read_text(encoding="utf-8", errors="replace")[:1600]
            lines.append(f"Artifact {name}: {path}\n{preview}")
        return "\n".join(lines)

    def _parse_action(self, content: str) -> tuple:
        text = content.strip()
        if text.startswith("```"):
            text = self._strip_fence(text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return (
                {"action": "final", "final": content},
                f"Response was not valid JSON and was treated as final: {exc}",
            )
        if not isinstance(parsed, dict):
            return (
                {"action": "final", "final": content},
                "Response JSON root was not an object and was treated as final.",
            )
        return parsed, ""

    def _strip_fence(self, text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        run_id: str,
    ) -> AgentToolResult:
        if tool_name == "read_file":
            return self.tool_executor.read_file(
                str(args.get("path", "")),
                start_line=int(args.get("start_line", 1)),
                max_lines=int(args.get("max_lines", 120)),
                run_id=run_id,
            )
        if tool_name == "search_code":
            return self.tool_executor.search_code(
                str(args.get("query", "")),
                path=str(args.get("path", ".")),
                max_results=int(args.get("max_results", 30)),
                run_id=run_id,
            )
        if tool_name == "run_command":
            argv = args.get("argv", [])
            return self.tool_executor.run_command(
                argv if isinstance(argv, list) else [],
                timeout_sec=int(args.get("timeout_sec", 60)),
                run_id=run_id,
            )
        if tool_name == "check_patch":
            return self.tool_executor.check_patch(
                str(args.get("patch_text", "")),
                run_id=run_id,
            )
        if tool_name == "apply_patch":
            return self.tool_executor.apply_patch(
                str(args.get("patch_text", "")),
                run_id=run_id,
            )
        return AgentToolResult(
            call_id=f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}",
            tool_name=tool_name,
            status="blocked",
            created_at=utc_now(),
            content=f"Unsupported tool requested: {tool_name}",
            warnings=["Unsupported tool requested by LLM action loop"],
        )

    def _persist(self, run: AgentActionRun) -> Path:
        return write_json(Path(run.run_dir) / "action_run.json", run)
