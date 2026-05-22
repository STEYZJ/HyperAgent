"""LLM-driven controlled tool-call loop."""

import json
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.runtime.action_repair import ActionRepairPipeline, ToolCallStormBreaker
from hyperagent.runtime.agent_tools import TOOL_METADATA, SafeAgentToolExecutor, tool_metadata
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
- run_command: {"argv": ["python", "-m", "unittest", "discover", "-s", "tests"], "timeout_sec": 60, "cwd": "."}
- run_experiment: {"plan_path": "experiments/demo/experiment.yaml", "seeds": [42, 43], "output_dir": "experiments/demo_suite"}
- task: {"agents": ["reviewer", "experiment-analyst"], "instruction": "review this result", "mode": "parallel", "max_steps": 2, "max_depth": 1, "max_concurrent": 3, "role": "leaf"}
- run_skill: {"name": "review-experiment", "instruction": "review reports/result.json", "max_steps": 2}
- install_skill: {"repo": "owner/repo", "skill_path": "skills/foo", "dry_run": true}
- install_skill: {"url": "https://github.com/owner/repo", "all_skills": true, "dry_run": true}
- framework_command: {"command": "status|usage|web status|mcp status|skills list|worktree|todos|sessions|stats", "args": []}
- todo_write: {"owner": "project", "items": [{"content": "inspect tests", "status": "in_progress", "priority": "high"}]}
- check_patch: {"patch_text": "unified diff"}
- apply_patch: {"patch_text": "unified diff"}
- web_search: {"query": "latest hyperspectral image classification mamba", "provider": "auto", "max_results": 5}
- web_fetch: {"url": "https://example.org/paper", "max_chars": 12000}
- web_extract: {"url": "https://example.org/paper", "max_chars": 12000}
- web_cite: {"citation_id": "web:...", "limit": 10}
- research_pattern_search: {"query": "how authors frame the gap", "field": "", "top_k": 5}
- experiment_strategy_search: {"query": "baseline selection logic", "field": "", "top_k": 5}
- storytelling_search: {"query": "reviewer persuasion and claim scaffolding", "field": "", "top_k": 5}
- research_taste_search: {"query": "what problems are worth doing", "field": "", "top_k": 5}
- extract_research_pattern: {"paper": "paper_id_or_path", "provider": "deepseek", "write": true}
- extract_experiment_strategy: {"paper": "paper_id_or_path", "provider": "deepseek", "write": true}
- extract_storytelling: {"paper": "paper_id_or_path", "provider": "deepseek", "write": true}
- paper_strategy_compare: {"papers": ["paper_a", "paper_b"], "field": ""}
- research_experience_consolidate: {"topic": "baseline selection", "papers": ["paper_a", "paper_b"], "field": ""}

Return exactly one JSON object and no prose:
{"thought": "brief reason", "action": "tool", "tool_name": "search_code", "args": {"query": "ExperimentSuiteRunner"}}
or:
{"thought": "brief reason", "action": "final", "final": "answer or next decision"}

Do not request unsafe shell commands. Ground experiment decisions in saved artifacts when they are available.
Use framework_command before answering questions about HyperAgent status, usage/cost, configured providers, web availability, image availability, MCP, skills, commands, sessions, todos, plan mode, IDE context, worktree, hooks, agents, research strategy status, or available framework capabilities.
When the user asks to list, inspect, or choose skills, call framework_command with "skills list" or "skills search". When the user asks to install a third-party skill from a local path, GitHub repo/path, or GitHub URL, call install_skill with dry_run=true first; install_skill can accept a GitHub repo-root URL and will auto-detect a SKILL.md when possible. Only install with dry_run=false after human authorization. If the user only gives a skill name, first check whether it is already installed; otherwise ask for a repo/path/URL. Do not execute external skill-installer helper scripts with run_command for HyperAgent skill installation.
When a GitHub repository contains Claude plugin files such as .claude/CLAUDE.md or .claude-plugin/plugin.json but no SKILL.md, explain that it is not directly installable as a HyperAgent/Codex SKILL.md skill and offer to convert it into a local SKILL.md.
When the user asks to use a known skill, call run_skill with the skill name and the user's instruction instead of explaining the skill manually.
When a SKILL.md mentions relative helper scripts such as scripts/list-skills.py, resolve them from the provided Skill directory and call run_command with cwd set to that Skill directory.
Use web tools only when current external information is necessary. Cite source URLs/citation ids from web_search/web_fetch results. Never request non-http(s), localhost, private IP, file, data, or javascript URLs.
Sensitive tools such as run_command, run_experiment, apply_patch, todo_write, web_search, web_fetch, web_extract, image_generate, and image_edit may require human authorization. If a tool is blocked for permission, report that approval is needed instead of pretending the capability is unavailable.
You do have controlled access to project files, approved shell commands, experiments, skills, and web tools through the tool list above. Do not claim you have no filesystem, shell, or network capability; instead, choose the appropriate tool or explain which provider/API key or user permission is missing.
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
        permission_policy: str = "session-ask",
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
            parsed, normalization_warning = self._normalize_action(parsed_result.action)
            warnings = list(parsed_result.warnings)
            if normalization_warning:
                warnings.append(normalization_warning)
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
                parsed_action, normalization_warning = self._normalize_action(item.action)
                item_action = str(parsed_action.get("action", "final"))
                if item_action != "tool":
                    continue
                item_args = dict(parsed_action.get("args", {}))
                item_warnings = list(item.warnings)
                if normalization_warning:
                    item_warnings.append(normalization_warning)
                tool_actions.append(
                    (
                        item,
                        str(parsed_action.get("tool_name", "")),
                        item_args,
                        item_warnings,
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

    def _normalize_action(self, parsed: Dict[str, Any]) -> tuple:
        normalized = dict(parsed or {})
        action = str(normalized.get("action", "")).strip()
        if action not in TOOL_METADATA:
            return normalized, ""
        args = normalized.get("args", {})
        if not isinstance(args, dict):
            args = {}
        for key, value in normalized.items():
            if key in {"action", "thought", "tool_name", "args"}:
                continue
            args.setdefault(key, value)
        normalized["action"] = "tool"
        normalized["tool_name"] = action
        normalized["args"] = args
        return normalized, f"Normalized direct tool action: {action}"

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
                cwd=(None if args.get("cwd") in {None, ""} else str(args.get("cwd"))),
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
            codex_home = os.environ.get("CODEX_HOME")
            roots.append(Path(codex_home) / "skills" if codex_home else Path.home() / ".codex" / "skills")
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
                    instruction=self._skill_runtime_context(skill, instruction),
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
                content=self._skill_runtime_context(skill, instruction),
                artifact_path=skill.path,
            )

        if tool_name == "install_skill":
            return self.tool_executor.install_skill(
                path=str(args.get("path", "")),
                repo=str(args.get("repo", "")),
                skill_path=str(args.get("skill_path", args.get("skill-path", ""))),
                url=str(args.get("url", "")),
                ref=str(args.get("ref", "main")),
                name=str(args.get("name", "")),
                force=bool(args.get("force", False)),
                dry_run=bool(args.get("dry_run", args.get("dry-run", True))),
                all_skills=bool(args.get("all_skills", args.get("all", False))),
                run_id=run_id,
            )

        if tool_name in {"research_pattern_search", "experiment_strategy_search", "storytelling_search", "research_taste_search"}:
            method = getattr(self.tool_executor, tool_name)
            return method(
                str(args.get("query", "")),
                field=str(args.get("field", "")),
                top_k=int(args.get("top_k", 8)),
                run_id=run_id,
            )
        if tool_name in {"extract_research_pattern", "extract_experiment_strategy", "extract_storytelling", "extract_research_taste"}:
            method = getattr(self.tool_executor, tool_name)
            return method(
                str(args.get("paper", "")),
                provider=str(args.get("provider", "")),
                model=(None if args.get("model") in {None, ""} else str(args.get("model"))),
                field=str(args.get("field", "")),
                write=bool(args.get("write", True)),
                run_id=run_id,
            )
        if tool_name in {"paper_strategy_compare", "compare_paper_strategies"}:
            raw_papers = args.get("papers", [])
            papers = [raw_papers] if isinstance(raw_papers, str) else raw_papers
            return self.tool_executor.paper_strategy_compare(
                [str(item) for item in papers] if isinstance(papers, list) else [],
                provider=str(args.get("provider", "")),
                model=(None if args.get("model") in {None, ""} else str(args.get("model"))),
                field=str(args.get("field", "")),
                run_id=run_id,
            )
        if tool_name in {"research_experience_consolidate", "consolidate_research_experience"}:
            raw_papers = args.get("papers", [])
            papers = [raw_papers] if isinstance(raw_papers, str) else raw_papers
            return self.tool_executor.research_experience_consolidate(
                str(args.get("topic", "")),
                papers=[str(item) for item in papers] if isinstance(papers, list) else [],
                provider=str(args.get("provider", "")),
                model=(None if args.get("model") in {None, ""} else str(args.get("model"))),
                field=str(args.get("field", "")),
                run_id=run_id,
            )
        if tool_name == "framework_command":
            raw_args = args.get("args", [])
            normalized_args = (
                [str(item) for item in raw_args]
                if isinstance(raw_args, list)
                else []
            )
            return self.tool_executor.framework_command(
                str(args.get("command", "")),
                args=normalized_args,
                run_id=run_id,
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
        if tool_name == "web_search":
            return self.tool_executor.web_search(
                str(args.get("query", "")),
                provider=str(args.get("provider", "auto")),
                max_results=int(args.get("max_results", 5)),
                timeout_sec=int(args.get("timeout_sec", 20)),
                run_id=run_id,
            )
        if tool_name == "web_fetch":
            return self.tool_executor.web_fetch(
                str(args.get("url", "")),
                max_chars=int(args.get("max_chars", 12000)),
                timeout_sec=int(args.get("timeout_sec", 20)),
                run_id=run_id,
            )
        if tool_name == "web_extract":
            return self.tool_executor.web_extract(
                str(args.get("url", "")),
                selector=str(args.get("selector", "")),
                max_chars=int(args.get("max_chars", 12000)),
                timeout_sec=int(args.get("timeout_sec", 20)),
                run_id=run_id,
            )
        if tool_name == "web_cite":
            return self.tool_executor.web_cite(
                citation_id=str(args.get("citation_id", "")),
                limit=int(args.get("limit", 20)),
                run_id=run_id,
            )
        if tool_name == "image_generate":
            return self.tool_executor.image_generate(
                str(args.get("prompt", "")),
                run_id=run_id,
            )
        if tool_name == "image_edit":
            return self.tool_executor.image_edit(
                str(args.get("image_path", "")),
                str(args.get("instruction", "")),
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

    def _skill_runtime_context(self, skill: Any, instruction: str = "") -> str:
        skill_path = Path(str(skill.path))
        return (
            f"Skill: {skill.name}\n"
            f"Skill path: {skill.path}\n"
            f"Skill directory: {skill_path.parent}\n"
            f"User instruction: {instruction}\n\n"
            "Execution note: resolve relative files mentioned by this SKILL.md "
            "from Skill directory. For helper scripts, call run_command with "
            "cwd set to Skill directory.\n\n"
            f"{skill.body}"
        )

    def _persist(self, run: AgentActionRun) -> Path:
        return write_json(Path(run.run_dir) / "action_run.json", run)
