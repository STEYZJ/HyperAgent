"""LLM-driven controlled tool-call loop."""

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.runtime.action_repair import ActionRepairPipeline, ToolCallStormBreaker
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor, tool_metadata
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.events import RuntimeEventLog
from hyperagent.runtime.hooks import HookEngine
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.llm_usage import LLMUsageLedger
from hyperagent.runtime.repo_context import RepoContextBuilder
from hyperagent.runtime.deepseek_reasonix import reasonix_cache_guidance
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
- run_experiment: {"plan_path": "experiments/demo/experiment.yaml", "seeds": [42, 43], "output_dir": "experiments/demo_suite"}
- task: {"agents": ["reviewer", "experiment-analyst"], "instruction": "review this result", "mode": "parallel", "max_steps": 2, "max_depth": 1, "max_concurrent": 3, "role": "leaf"}
- run_skill: {"name": "review-experiment", "instruction": "review reports/result.json", "max_steps": 2}
- todo_write: {"owner": "project", "items": [{"content": "inspect tests", "status": "in_progress", "priority": "high"}]}
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
        permission_policy: str = "auto",
        permission_callback: Optional[Any] = None,
    ) -> None:
        self.session_store = session_store
        self.llm_store = llm_store
        self.workspace = workspace
        self.llm_client = llm_client or LLMClient()
        self.permission_policy = permission_policy
        self.permission_callback = permission_callback
        self._active_provider = ""
        self._active_model: Optional[str] = None
        self.event_log = RuntimeEventLog(workspace.workspace_dir)
        self.repair_pipeline = ActionRepairPipeline()
        self.tool_executor = tool_executor or SafeAgentToolExecutor(
            workspace.project_root,
            workspace.workspace_dir,
            permission_policy=permission_policy,
            permission_callback=permission_callback,
            hook_engine=HookEngine(workspace.workspace_dir),
            event_log=self.event_log,
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
        loop_mode: str = "standard",
        token_budget: Optional[int] = None,
        storm_max_repeats: int = 2,
    ) -> AgentActionRun:
        self.llm_store.ensure_defaults()
        spec = self.llm_store.get(provider)
        self._active_provider = provider
        self._active_model = model
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
            loop_mode=loop_mode,
            token_budget=token_budget,
            event_log_path=str(self.event_log.path),
        )

        self.session_store.add_message(session_id, "user", instruction)
        messages = self._build_messages(
            session_id,
            instruction,
            task_id=task_id,
            max_files=max_files,
            max_preview_chars=max_preview_chars,
            loop_mode=loop_mode,
        )
        run.stable_prefix_hash = self._stable_prefix_hash(messages, loop_mode)
        storm_breaker = ToolCallStormBreaker(max_repeats=storm_max_repeats)
        self.event_log.append(
            "action_loop.start",
            source="action_loop",
            session_id=session_id,
            run_id=run_id,
            status="running",
            payload={
                "provider": provider,
                "model": run.model,
                "loop_mode": loop_mode,
                "stable_prefix_hash": run.stable_prefix_hash,
                "token_budget": token_budget,
            },
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
            usage_record = LLMUsageLedger(self.workspace.workspace_dir).record_response(
                response,
                spec=spec,
                session_id=session_id,
                event_type="action_loop.response",
                context_chars=sum(len(message.content) for message in messages),
                metadata={
                    "run_id": run_id,
                    "step_index": step_index,
                    "loop_mode": loop_mode,
                    "stable_prefix_hash": run.stable_prefix_hash,
                    "parse_capable": True,
                },
            )
            if token_budget is not None and int(usage_record.get("total_tokens") or 0) > token_budget:
                run.status = "paused"
                run.budget_exhausted = True
                run.warnings.append(
                    f"Token budget exhausted: used {usage_record.get('total_tokens')} > budget {token_budget}."
                )
                self.event_log.append(
                    "action_loop.paused",
                    source="action_loop",
                    session_id=session_id,
                    run_id=run_id,
                    status="paused",
                    message=run.warnings[-1],
                )
                self._persist(run)
                return run
            if response.warnings:
                run.status = "failed"
                run.warnings.extend(response.warnings)
                self._persist(run)
                return run

            parsed_results = self.repair_pipeline.parse_many(response)
            parsed_result = parsed_results[0]
            parsed = parsed_result.action
            warnings = list(parsed_result.warnings)
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
                        parse_source=parsed_result.source,
                    )
                )
                self.session_store.add_message(session_id, "assistant", final)
                self.event_log.append(
                    "action_loop.completed",
                    source="action_loop",
                    session_id=session_id,
                    run_id=run_id,
                    status="completed",
                    message=final[:500],
                    payload={"steps": len(run.steps)},
                )
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
                        parse_source=parsed_result.source,
                    )
                )
                self._persist(run)
                return run

            tool_actions = []
            for item in parsed_results:
                parsed_action = item.action
                item_action = str(parsed_action.get("action", "final"))
                if item_action != "tool":
                    continue
                item_args = dict(parsed_action.get("args", {}))
                tool_actions.append(
                    (
                        item,
                        str(parsed_action.get("tool_name", "")),
                        item_args,
                        list(item.warnings),
                    )
                )
            parallel = len(tool_actions) > 1 and all(
                tool_metadata(tool_name).parallel_safe
                for _, tool_name, _, _ in tool_actions
            )
            tool_results = self._execute_tool_actions(
                tool_actions,
                storm_breaker,
                run_id,
                parallel=parallel,
            )
            tool_summary_lines = []
            for item, tool_name, args, item_warnings, result in tool_results:
                run.steps.append(
                    AgentActionStep(
                        step_index=step_index,
                        response_content=response.content,
                        action="tool",
                        status=result.status,
                        tool_name=tool_name,
                        args=args,
                        tool_result=result,
                        warnings=item_warnings + result.warnings,
                        parse_source=item.source,
                    )
                )
                self.event_log.append(
                    "action_loop.step",
                    source="action_loop",
                    session_id=session_id,
                    run_id=run_id,
                    tool_name=tool_name,
                    status=result.status,
                    message=result.content[:500],
                    payload={
                        "step_index": step_index,
                        "parse_source": item.source,
                        "args": args,
                        "parallel_dispatch": parallel,
                        "warnings": item_warnings + result.warnings,
                    },
                )
                tool_summary_lines.append(
                    f"Tool result for {tool_name}: status={result.status}\n{result.content}"
                )
            self.session_store.add_message(session_id, "assistant", response.content)
            self.session_store.add_message(
                session_id,
                "tool",
                "\n\n".join(tool_summary_lines),
            )
            messages.append(LLMMessage(role="assistant", content=response.content))
            messages.append(
                LLMMessage(
                    role="user",
                    content=(
                        "\n\n".join(tool_summary_lines)
                        + "\nReturn the next JSON action."
                    ),
                )
            )
            self._persist(run)

        run.status = "max_steps_reached"
        run.warnings.append("Maximum action-loop steps reached before final response.")
        self.event_log.append(
            "action_loop.max_steps",
            source="action_loop",
            session_id=session_id,
            run_id=run_id,
            status=run.status,
            message=run.warnings[-1],
        )
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
        loop_mode: str = "standard",
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
        if loop_mode == "cache-first":
            guidance = reasonix_cache_guidance()
            context_parts.insert(
                0,
                (
                    "Reasonix cache-first partition:\n"
                    f"- stable_prefix: {', '.join(guidance['stable_prefix'])}\n"
                    f"- semi_stable_context: {', '.join(guidance['semi_stable_context'])}\n"
                    f"- volatile_suffix: {', '.join(guidance['volatile_suffix'])}\n"
                    f"- rule: {guidance['rule']}"
                ),
            )
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

    def _stable_prefix_hash(self, messages: List[LLMMessage], loop_mode: str) -> str:
        if loop_mode != "cache-first":
            return ""
        stable = "\n\n".join(message.content for message in messages[:2])
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]

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

    def _execute_tool_actions(
        self,
        tool_actions: List[tuple],
        storm_breaker: ToolCallStormBreaker,
        run_id: str,
        *,
        parallel: bool,
    ) -> List[tuple]:
        if not parallel:
            return [
                (
                    item,
                    tool_name,
                    args,
                    warnings,
                    self._execute_tool_with_storm(
                        tool_name,
                        args,
                        run_id,
                        storm_breaker,
                    ),
                )
                for item, tool_name, args, warnings in tool_actions
            ]
        results: List[tuple] = []
        with ThreadPoolExecutor(max_workers=min(len(tool_actions), 4)) as pool:
            futures = {
                pool.submit(
                    self._execute_tool_with_storm,
                    tool_name,
                    args,
                    run_id,
                    storm_breaker,
                ): (item, tool_name, args, warnings)
                for item, tool_name, args, warnings in tool_actions
            }
            for future in as_completed(futures):
                item, tool_name, args, warnings = futures[future]
                results.append((item, tool_name, args, warnings, future.result()))
        return results

    def _execute_tool_with_storm(
        self,
        tool_name: str,
        args: Dict[str, Any],
        run_id: str,
        storm_breaker: ToolCallStormBreaker,
    ) -> AgentToolResult:
        storm_warning = storm_breaker.check(tool_name, args)
        if storm_warning:
            return AgentToolResult(
                call_id=f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}",
                tool_name=tool_name,
                status="blocked",
                created_at=utc_now(),
                content=storm_warning,
                warnings=["tool-call storm breaker"],
            )
        return self._execute_tool(tool_name, args, run_id)

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
        if tool_name == "run_experiment":
            seeds = args.get("seeds", [])
            normalized_seeds = (
                [int(seed) for seed in seeds]
                if isinstance(seeds, list) and seeds
                else None
            )
            return self.tool_executor.run_experiment(
                str(args.get("plan_path", "")),
                seeds=normalized_seeds,
                output_dir=(
                    None
                    if args.get("output_dir") in {None, ""}
                    else str(args.get("output_dir"))
                ),
                suite_name=(
                    None
                    if args.get("suite_name") in {None, ""}
                    else str(args.get("suite_name"))
                ),
                run_id=run_id,
            )
        if tool_name == "task":
            from hyperagent.runtime.multi_agent import MultiAgentTaskRunner

            agents = args.get("agents", [])
            if isinstance(agents, str):
                normalized_agents = [agents]
            elif isinstance(agents, list):
                normalized_agents = [str(agent) for agent in agents]
            else:
                normalized_agents = []
            runner = MultiAgentTaskRunner(
                self.workspace,
                self.session_store,
                self.llm_store,
                llm_client=self.llm_client,
                permission_policy=self.permission_policy,
                permission_callback=self.permission_callback,
            )
            task_run = runner.run(
                session_id=self.session_store.new("subagent task").session_id,
                provider=str(args.get("provider", "")) or self._active_provider or "deepseek",
                instruction=str(args.get("instruction", "")),
                agents=normalized_agents,
                model=(
                    self._active_model
                    if args.get("model") in {None, ""}
                    else str(args.get("model"))
                ),
                profile=str(args.get("profile", "")),
                mode=str(args.get("mode", "sequential")),
                max_steps=int(args.get("max_steps", 2)),
                max_depth=int(args.get("max_depth", 1)),
                max_concurrent=int(args.get("max_concurrent", 4)),
                delegation_role=str(args.get("role", "leaf")),
                llm_kwargs={},
            )
            status = "ok" if task_run.status == "completed" else "error"
            return AgentToolResult(
                call_id=f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}",
                tool_name=tool_name,
                status=status,
                created_at=utc_now(),
                content=task_run.aggregate_response,
                artifact_path=str(Path(task_run.run_dir) / "multi_agent_run.json"),
                warnings=task_run.warnings,
            )
        if tool_name == "run_skill":
            from hyperagent.runtime.multi_agent import MultiAgentTaskRunner
            from hyperagent.runtime.skills import SkillStore

            roots = [
                Path(__file__).resolve().parents[1] / "skills",
                self.workspace.project_root / "skills",
                self.workspace.workspace_dir / "skills",
            ]
            skill_name = str(args.get("name", args.get("skill", "")))
            instruction = str(args.get("instruction", args.get("arguments", "")))
            try:
                skill = SkillStore(roots).render(skill_name, instruction)
            except KeyError as exc:
                return AgentToolResult(
                    call_id=f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}",
                    tool_name=tool_name,
                    status="error",
                    created_at=utc_now(),
                    content=str(exc),
                    warnings=["skill not found"],
                )
            if skill.run_as.lower() == "subagent":
                agent_ref = str(args.get("agent", "") or skill.metadata.get("agent", "") or "")
                if not agent_ref:
                    agent_ref = {
                        "explore": "code-explorer",
                        "research-literature": "experiment-analyst",
                        "review-experiment": "experiment-analyst",
                        "spectral-critic": "spectral-critic",
                        "paper-method-extractor": "code-architect",
                    }.get(skill.name, "code-explorer")
                task_run = MultiAgentTaskRunner(
                    self.workspace,
                    self.session_store,
                    self.llm_store,
                    llm_client=self.llm_client,
                    permission_policy=self.permission_policy,
                    permission_callback=self.permission_callback,
                ).run(
                    session_id=self.session_store.new(f"skill:{skill.name}").session_id,
                    provider=str(args.get("provider", "")) or self._active_provider or "deepseek",
                    instruction=skill.body,
                    agents=[agent_ref],
                    model=skill.model or self._active_model,
                    profile=skill.profile,
                    mode="sequential",
                    max_steps=int(args.get("max_steps", 2)),
                    llm_kwargs={},
                )
                return AgentToolResult(
                    call_id=f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}",
                    tool_name=tool_name,
                    status="ok" if task_run.status == "completed" else "error",
                    created_at=utc_now(),
                    content=task_run.aggregate_response,
                    artifact_path=str(Path(task_run.run_dir) / "multi_agent_run.json"),
                    warnings=task_run.warnings,
                )
            return AgentToolResult(
                call_id=f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}",
                tool_name=tool_name,
                status="ok",
                created_at=utc_now(),
                content=skill.body,
                artifact_path=skill.path,
            )
        if tool_name == "todo_write":
            items = args.get("items", [])
            if not isinstance(items, list):
                items = []
            return self.tool_executor.todo_write(
                [dict(item) for item in items if isinstance(item, dict)],
                owner=str(args.get("owner", "project")),
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
