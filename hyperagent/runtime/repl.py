"""Interactive HyperAgent REPL."""

from pathlib import Path
from typing import Callable, Dict, List, Optional

from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.agent_loop import AgentLoop
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor, ToolPermissionRequest
from hyperagent.runtime.background_jobs import BackgroundJobStore
from hyperagent.runtime.checkpoints import CheckpointStore
from hyperagent.runtime.coding_agent import CodingAgent
from hyperagent.runtime.commands import SlashCommandStore
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.events import RuntimeEventLog
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.feature_state import (
    FeedbackStore,
    IDEContextStore,
    PersonalityStore,
    PlanModeStore,
    image_status,
    web_status,
    worktree_status,
)
from hyperagent.runtime.general_agent import GeneralAgentRunner
from hyperagent.runtime.hooks import HookEngine
from hyperagent.runtime.i18n import Translator
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.llm_usage import LLMUsageLedger
from hyperagent.runtime.memory import MemoryStore
from hyperagent.runtime.mcp import MCPServerStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.slash_registry import grouped_help
from hyperagent.runtime.deepseek_reasonix import (
    get_reasonix_profile,
    list_reasonix_profiles,
    reasonix_cache_guidance,
)
from hyperagent.runtime.skills import SkillStore
from hyperagent.runtime.subagents import SubagentRuntimeRegistry
from hyperagent.runtime.todos import TodoStore
from hyperagent.runtime.tool_panel import (
    render_action_run,
    render_tool_catalog,
    render_tool_result,
)
from hyperagent.runtime.wait_indicator import (
    ConsoleWaitIndicator,
    WaitIndicator,
    run_with_wait_indicator,
)
from hyperagent.runtime.workspace import HyperAgentWorkspace, utc_now
from hyperagent.schemas import LLMMessage


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]
WaitIndicatorFactory = Callable[[], WaitIndicator]


class HyperAgentRepl:
    """Small stdlib REPL that keeps session state and shows tool panels."""

    def __init__(
        self,
        *,
        workspace: HyperAgentWorkspace,
        conversations: ConversationStore,
        providers: LLMProviderStore,
        prompt_library: PromptLibrary,
        provider: str = "deepseek",
        model: Optional[str] = None,
        mode: str = "research",
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        new_title: Optional[str] = None,
        permission_policy: str = "ask",
        max_context_chars: int = 12000,
        keep_last: int = 6,
        llm_kwargs: Optional[Dict[str, object]] = None,
        translator: Optional[Translator] = None,
        input_func: InputFunc = input,
        output_func: OutputFunc = print,
        wait_indicator_factory: Optional[WaitIndicatorFactory] = None,
    ) -> None:
        self.workspace = workspace
        self.conversations = conversations
        self.providers = providers
        self.prompt_library = prompt_library
        self.provider = provider
        self.model = model
        self.mode = mode
        self.task_id = task_id
        self.permission_policy = permission_policy
        self.max_context_chars = max_context_chars
        self.keep_last = keep_last
        self.llm_kwargs = dict(llm_kwargs or {})
        self.translator = translator
        self.input = input_func
        self.output = output_func
        self.wait_indicator_factory = wait_indicator_factory
        self.memory = MemoryStore(workspace.project_root, workspace.workspace_dir)
        self.extensions = RuntimeExtensionStore(workspace.workspace_dir)
        self.command_store = SlashCommandStore(workspace.project_root, workspace.workspace_dir)
        self.hooks = HookEngine(workspace.workspace_dir)
        self.todos = TodoStore(workspace.workspace_dir)
        self.subagents = SubagentRuntimeRegistry(workspace.workspace_dir)
        self.jobs = BackgroundJobStore(workspace.workspace_dir)
        self.ide_context = IDEContextStore(workspace.workspace_dir)
        self.plan_mode = PlanModeStore(workspace.workspace_dir)
        self.personality = PersonalityStore(workspace.workspace_dir)
        self.feedback = FeedbackStore(workspace.workspace_dir)
        self.usage = LLMUsageLedger(workspace.workspace_dir)
        self.permission_cache: Dict[str, bool] = {}
        self.session_id = self._ensure_session(session_id, new_title)
        self.expand_reasoning_content = self._default_expand_reasoning_content()
        self.last_user_message = ""
        self.action_loop_mode = "standard"
        self.action_token_budget: Optional[int] = None

    def run(self) -> int:
        self.output(self._banner())
        self._emit_hook_messages("SessionStart", {"session_id": self.session_id})
        while True:
            try:
                raw = self.input("HyperAgent> ")
            except EOFError:
                self.output("")
                return 0
            line = raw.strip()
            if not line:
                continue
            if not self.handle_line(line):
                return 0

    def handle_line(self, line: str) -> bool:
        if line in {"/exit", "/quit", "exit", "quit"}:
            self._emit_hook_messages("Stop", {"session_id": self.session_id})
            self.output(self._t("repl.bye", "bye"))
            return False
        try:
            hook_result = self.hooks.run(
                "UserPromptSubmit",
                {"line": line, "session_id": self.session_id},
            )
            for warning in hook_result.warnings:
                self.output(f"hook warning: {warning}")
            if hook_result.blocked:
                self.output("blocked by hook")
                return True
            if line.startswith("/"):
                return self._handle_command(line)
            self._chat(line)
        except Exception as exc:
            self.output(self._t("repl.error", "error: {error}", error=exc))
        return True

    def _ensure_session(
        self,
        session_id: Optional[str],
        new_title: Optional[str],
    ) -> str:
        if session_id:
            self.conversations.load(session_id)
            return session_id
        title = new_title or "HyperAgent REPL"
        return self.conversations.new(title).session_id

    def _banner(self) -> str:
        return self._t(
            "repl.banner",
            (
                "HyperAgent interactive mode\n"
                "session: {session_id}\n"
                "provider: {provider}\n"
                "type /help for commands, /exit to quit"
            ),
            session_id=self.session_id,
            provider=self.provider,
        )

    def _handle_command(self, line: str) -> bool:
        parts = line.split()
        command = parts[0].lower()
        args = parts[1:]
        if command in {"/help", "/commands"}:
            self.output(self._help())
        elif command == "/status":
            self._status()
        elif command == "/session":
            self.output(f"session: {self.session_id}")
        elif command in {"/background", "/bg"}:
            self._background(args)
        elif command == "/btw":
            self._btw(args)
        elif command == "/sessions":
            self._sessions()
        elif command == "/new":
            title = " ".join(args) or "HyperAgent REPL"
            self.session_id = self.conversations.new(title).session_id
            self.output(f"new session: {self.session_id}")
        elif command in {"/undo", "/branch", "/fork", "/queue", "/steer", "/goal", "/subgoal"}:
            self._session_flow_command(command, args)
        elif command == "/resume":
            self._resume(args)
        elif command in {"/compact", "/compress"}:
            self._compact(args)
        elif command == "/clear":
            self._clear()
        elif command == "/context":
            self._context()
        elif command in {"/usage", "/cost"}:
            self._usage(args)
        elif command == "/stats":
            self._stats()
        elif command == "/retry":
            self._retry()
        elif command == "/stop":
            self.output("no active model request")
        elif command == "/copy":
            self._export(args)
        elif command == "/init":
            self._init_memory()
        elif command == "/memory":
            self._memory(args)
        elif command == "/agents":
            self._agents(args)
        elif command == "/hooks":
            self._hooks(args)
        elif command in {"/commands", "/command"}:
            self._commands(args)
        elif command == "/todos":
            self._todos(args)
        elif command == "/permissions":
            self._permissions()
        elif command == "/export":
            self._export(args)
        elif command == "/doctor":
            self._doctor()
        elif command in {"/plugin", "/plugins"}:
            self._plugins(args)
        elif command == "/rewind":
            self._rewind(args)
        elif command in {"/reasonix", "/deepseek"}:
            self._reasonix(args)
        elif command == "/preset":
            self._preset(args)
        elif command == "/pro":
            self._pro()
        elif command == "/budget":
            self._budget(args)
        elif command == "/thinking":
            self._thinking(args)
        elif command == "/simplify":
            self._simplify()
        elif command == "/model":
            self._models()
        elif command == "/mcp":
            self._mcp(args)
        elif command == "/ide-context":
            self._ide_context(args)
        elif command == "/plan-mode":
            self._plan_mode(args)
        elif command == "/personality":
            self._personality(args)
        elif command == "/feedback":
            self._feedback(args)
        elif command == "/web":
            self._web(args)
        elif command == "/image":
            self._image(args)
        elif command == "/worktree":
            self._worktree()
        elif command in {"/skills", "/skill"}:
            self._skills(args)
        elif command == "/checkpoint":
            self._checkpoint(args)
        elif command == "/restore":
            self._restore(args)
        elif command == "/jobs":
            self._jobs(args)
        elif command == "/logs":
            self._logs(args)
        elif command == "/tools":
            self.output(
                render_tool_catalog(
                    self._tool_names(),
                    title=self._t("tool.catalog.title", "Available local tools"),
                )
            )
        elif command == "/tool":
            self._manual_tool(args)
        elif command == "/act":
            self._act(" ".join(args))
        elif command == "/plan":
            self._plan(" ".join(args))
        elif command in {"/exit", "/quit"}:
            return False
        else:
            if not self._custom_command(command, args):
                self.output(
                    self._t(
                        "repl.unknown_command",
                        "unknown command: {command}",
                        command=command,
                    )
                )
        return True

    def _chat(self, message: str) -> None:
        self.last_user_message = message

        def turn():
            return AgentLoop(
                self.conversations,
                self.providers,
                self.workspace,
                prompt_library=self.prompt_library,
            ).run(
                session_id=self.session_id,
                provider=self.provider,
                user_message=message,
                model=self.model,
                mode=self.mode,
                task_id=self.task_id,
                max_context_chars=self.max_context_chars,
                thinking_displayed=self.expand_reasoning_content,
                reasoning_content_expanded=self.expand_reasoning_content,
                **self.llm_kwargs,
            )

        indicator = (
            self.wait_indicator_factory()
            if self.wait_indicator_factory is not None
            else ConsoleWaitIndicator(self.output)
        )
        result = run_with_wait_indicator(turn, indicator).value
        for warning in result.warnings:
            self.output(f"warning: {warning}")
        if result.response.reasoning_content:
            if self.expand_reasoning_content:
                self.output("【模型思考内容】")
                self.output(result.response.reasoning_content)
            else:
                self.output("【模型思考内容已折叠，可用 /thinking on 展开】")
        if result.response.content:
            self.output(result.response.content)

    def _act(self, instruction: str) -> None:
        if not instruction:
            self.output("usage: /act <instruction>")
            return
        if self.plan_mode.load().get("enabled"):
            self.output("plan-mode is on; tool execution is paused. Use /plan or /plan-mode off.")
            return
        run = AgentActionLoop(
            self.conversations,
            self.providers,
            self.workspace,
            tool_executor=SafeAgentToolExecutor(
                self.workspace.project_root,
                self.workspace.workspace_dir,
                permission_policy=self.permission_policy,
                permission_callback=self._confirm_permission,
                session_permission_cache=self.permission_cache,
                hook_engine=self.hooks,
            ),
        ).run(
            session_id=self.session_id,
            provider=self.provider,
            instruction=instruction,
            model=self.model,
            task_id=self.task_id,
            loop_mode=self.action_loop_mode,
            token_budget=self.action_token_budget,
            **self.llm_kwargs,
        )
        self.output(render_action_run(run))

    def _plan(self, instruction: str) -> None:
        if not instruction:
            self.output("usage: /plan <instruction>")
            return
        run = CodingAgent(
            self.workspace,
            self.conversations,
            self.providers,
            prompt_library=self.prompt_library,
        ).plan(
            session_id=self.session_id,
            provider=self.provider,
            instruction=instruction,
            model=self.model,
            mode="code",
            task_id=self.task_id,
            **self.llm_kwargs,
        )
        self.output(f"run_id: {run.run_id}")
        self.output(f"plan: {run.plan_path}")
        for warning in run.warnings:
            self.output(f"warning: {warning}")

    def _status(self) -> None:
        status = self.workspace.status()
        self.output(f"initialized: {status.initialized}")
        self.output(f"workspace: {status.workspace_dir}")
        self.output(f"dataset_root: {status.dataset_root}")
        self.output(f"tasks: {status.task_count}")
        self.output(f"tasks_by_status: {status.tasks_by_status}")

    def _sessions(self) -> None:
        sessions = self.conversations.list(include_archived=True)
        if not sessions:
            self.output("no sessions")
            return
        for session in sessions:
            marker = "*" if session.session_id == self.session_id else " "
            self.output(
                f"{marker} {session.session_id}\t{session.status}\t"
                f"{len(session.messages)}\t{session.title}"
            )

    def _resume(self, args: List[str]) -> None:
        if not args:
            self.output("usage: /resume <session_id>")
            return
        session = self.conversations.load(args[0])
        self.session_id = session.session_id
        self.output(f"resumed: {self.session_id}")

    def _btw(self, args: List[str]) -> None:
        text = " ".join(args).strip()
        if not text:
            self.output("usage: /btw <temporary question>")
            return
        self.output("[btw] temporary isolated answer")
        messages = AgentLoop(
            self.conversations,
            self.providers,
            self.workspace,
            prompt_library=self.prompt_library,
        ).build_messages(
            self.conversations.load(self.session_id),
            mode=self.mode,
            task_id=self.task_id,
            max_context_chars=min(self.max_context_chars, 4000),
        )
        messages.append(
            LLMMessage(
                role="user",
                content="[temporary /btw question, do not treat as project state]\n" + text,
            )
        )
        spec = self.providers.get(self.provider)
        response = AgentLoop(
            self.conversations,
            self.providers,
            self.workspace,
            prompt_library=self.prompt_library,
        ).llm_client.send(
            spec,
            messages,
            model=self.model,
            **self.llm_kwargs,
        )
        for warning in response.warnings:
            self.output(f"warning: {warning}")
        if response.content:
            self.output(response.content)

    def _background(self, args: List[str]) -> None:
        text = " ".join(args).strip()
        if not text:
            self.output("usage: /background <prompt>")
            return
        job = self.jobs.create(
            kind="prompt",
            instruction=text,
            session_id=self.session_id,
            status="queued",
        )
        self.output(f"background job queued: {job.job_id}")
        self.output("run /jobs to inspect queued work")

    def _session_flow_command(self, command: str, args: List[str]) -> None:
        session = self.conversations.load(self.session_id)
        canonical = command.lstrip("/")
        if canonical == "fork":
            canonical = "branch"
        if canonical == "undo":
            if not session.messages:
                self.output("nothing to undo")
                return
            removed = 0
            while session.messages and removed < 2:
                session.messages.pop()
                removed += 1
                if removed == 1 and session.messages and session.messages[-1].role != "assistant":
                    break
            self.conversations.save(session)
            self.output(f"undone messages: {removed}")
            return
        if canonical == "branch":
            title = " ".join(args).strip() or f"{session.title} branch"
            branch = self.conversations.new(title)
            branch.messages = list(session.messages)
            branch.summaries = list(session.summaries)
            branch.metadata.update(session.metadata)
            branch.metadata["branched_from"] = self.session_id
            self.conversations.save(branch)
            self.session_id = branch.session_id
            self.output(f"branched: {branch.session_id}")
            return
        if canonical in {"queue", "steer"}:
            text = " ".join(args).strip()
            if not text:
                self.output(f"usage: /{canonical} <prompt>")
                return
            key = "queued_prompts" if canonical == "queue" else "steering_notes"
            values = list(session.metadata.get(key, []))
            values.append({"created_at": utc_now(), "text": text})
            session.metadata[key] = values
            self.conversations.save(session)
            self.output(f"{canonical} saved")
            return
        if canonical in {"goal", "subgoal"}:
            self._goal_command(canonical, args, session)
            return

    def _goal_command(self, canonical: str, args: List[str], session) -> None:
        if canonical == "goal":
            action = args[0] if args else "status"
            if action == "status":
                self.output(str(session.metadata.get("goal", "no active goal")))
                return
            if action == "clear":
                session.metadata.pop("goal", None)
                self.conversations.save(session)
                self.output("goal cleared")
                return
            if action in {"pause", "resume"}:
                goal = dict(session.metadata.get("goal", {}))
                goal["status"] = "paused" if action == "pause" else "active"
                session.metadata["goal"] = goal
                self.conversations.save(session)
                self.output(f"goal {goal['status']}")
                return
            text = " ".join(args).strip()
            session.metadata["goal"] = {"status": "active", "text": text}
            self.conversations.save(session)
            self.output("goal set")
            return
        subgoals = list(session.metadata.get("subgoals", []))
        if args and args[0] == "clear":
            session.metadata["subgoals"] = []
            self.conversations.save(session)
            self.output("subgoals cleared")
            return
        if args and args[0] == "remove" and len(args) > 1:
            index = max(int(args[1]) - 1, 0)
            if index < len(subgoals):
                subgoals.pop(index)
            session.metadata["subgoals"] = subgoals
            self.conversations.save(session)
            self.output("subgoal removed")
            return
        text = " ".join(args).strip()
        if not text:
            self.output("\n".join(f"{i + 1}. {v}" for i, v in enumerate(subgoals)) or "no subgoals")
            return
        subgoals.append(text)
        session.metadata["subgoals"] = subgoals
        self.conversations.save(session)
        self.output("subgoal added")

    def _compact(self, args: List[str]) -> None:
        keep_last = int(args[0]) if args else self.keep_last
        session = self.conversations.compress(self.session_id, keep_last=keep_last)
        self.output(
            f"compacted: messages={len(session.messages)} summaries={len(session.summaries)}"
        )

    def _clear(self) -> None:
        self.extensions.create_rewind_snapshot(
            self.session_id,
            self.conversations.load(self.session_id).to_dict(),
        )
        session = self.conversations.clear(self.session_id)
        self.output(f"cleared: messages={len(session.messages)} summaries={len(session.summaries)}")

    def _context(self) -> None:
        status = self.conversations.context_status(
            self.session_id,
            max_chars=self.max_context_chars,
            keep_last=self.keep_last,
            trigger_ratio=0.85,
        )
        self.output(f"messages: {status.message_count}")
        self.output(f"summaries: {status.summary_count}")
        self.output(f"chars: {status.current_chars}/{status.max_chars}")
        self.output(f"trigger_chars: {status.trigger_chars}")
        self.output(f"should_compress: {status.should_compress}")

    def _usage(self, args: List[str]) -> None:
        limit = int(args[0]) if args else None
        summary = self.usage.summarize(limit=limit)
        self.output(
            "llm usage:\n"
            f"- requests: {summary['request_count']}\n"
            f"- total_tokens: {summary['total_tokens']}\n"
            f"- prompt_tokens: {summary['prompt_tokens']}\n"
            f"- completion_tokens: {summary['completion_tokens']}\n"
            f"- prompt_cache_hit_tokens: {summary['prompt_cache_hit_tokens']}\n"
            f"- prompt_cache_miss_tokens: {summary['prompt_cache_miss_tokens']}\n"
            f"- cache_hit_ratio: {summary['cache_hit_ratio']}\n"
            f"- ledger: {summary['ledger_path']}"
        )

    def _stats(self) -> None:
        events = RuntimeEventLog(self.workspace.workspace_dir).summarize()
        usage = self.usage.summarize()
        self.output(
            "runtime stats:\n"
            f"- events: {events['event_count']}\n"
            f"- event_log: {events['path']}\n"
            f"- llm_requests: {usage['request_count']}\n"
            f"- llm_total_tokens: {usage['total_tokens']}\n"
            f"- cache_hit_ratio: {usage['cache_hit_ratio']}"
        )

    def _retry(self) -> None:
        if not self.last_user_message:
            self.output("no previous user message to retry")
            return
        self.output("retrying previous message")
        self._chat(self.last_user_message)

    def _init_memory(self) -> None:
        path = self.memory.ensure_project_memory()
        self.output(f"initialized project memory: {path}")

    def _memory(self, args: List[str]) -> None:
        if not args or args[0] == "list":
            self.output("\n".join(self.memory.list()))
            return
        action = args[0]
        scope = args[1] if len(args) > 1 else "project"
        if action == "show":
            self.output(self.memory.read(scope) or f"{scope} memory is empty")
            return
        if action == "add":
            text = " ".join(args[2:]).strip()
            if not text:
                self.output("usage: /memory add <project|user|auto> <text>")
                return
            path = self.memory.append(scope, text)
            self.output(f"memory updated: {path}")
            return
        self.output("usage: /memory [list|show <scope>|add <scope> <text>]")

    def _agents(self, args: List[str]) -> None:
        if args and args[0] == "status":
            control = self.subagents.control()
            self.output(
                f"subagent_control: paused={control.get('paused')} "
                f"stop_ids={','.join(control.get('stop_ids', []))}"
            )
            states = self.subagents.list(include_completed=True)
            if not states:
                self.output("no subagent runs")
                return
            for state in states[-20:]:
                self.output(
                    f"{state.subagent_id}\tdepth={state.depth}\t{state.status}\t"
                    f"{state.agent_name}\tparent={state.parent_id or '-'}"
                )
            return
        if args and args[0] == "pause":
            control = self.subagents.pause(" ".join(args[1:]))
            self.output(f"subagent spawning paused: {control.get('reason', '')}")
            return
        if args and args[0] == "resume":
            self.subagents.resume()
            self.output("subagent spawning resumed")
            return
        if args and args[0] == "stop":
            if len(args) < 2:
                self.output("usage: /agents stop <subagent_id>")
                return
            self.subagents.stop(args[1])
            self.output(f"stop requested: {args[1]}")
            return
        if not args or args[0] == "list":
            agents = self.extensions.list_subagents()
            if not agents:
                self.output("no subagents")
                return
            for agent in agents:
                self.output(
                    f"{agent.get('id')}\t{agent.get('name')}\t"
                    f"{agent.get('role')}\ttools={','.join(agent.get('tools', []))}"
                )
            return
        if args[0] == "add":
            if len(args) < 3:
                self.output("usage: /agents add <name> <role> [tool1,tool2]")
                return
            tools = args[3].split(",") if len(args) > 3 and args[3] else []
            item = self.extensions.add_subagent(args[1], args[2], tools=tools)
            self.output(f"subagent added: {item['id']}")
            return
        if args[0] == "run":
            if len(args) < 3:
                self.output("usage: /agents run <name|id> <instruction>")
                return
            run = GeneralAgentRunner(
                self.workspace,
                self.conversations,
                self.providers,
                permission_policy=self.permission_policy,
                permission_callback=self._confirm_permission,
                session_permission_cache=self.permission_cache,
            ).run(
                args[1],
                " ".join(args[2:]),
                session_id=self.session_id,
                provider=self.provider,
                model=self.model,
                profile=str(self.llm_kwargs.get("reasonix_profile", "")),
                task_id=self.task_id,
            )
            self.output(f"agent_run: {Path(run.run_dir) / 'agent_run.json'}")
            self.output(f"status: {run.status}")
            if run.action_run_path:
                self.output(f"action_run: {run.action_run_path}")
            for warning in run.warnings:
                self.output(f"warning: {warning}")
            return
        self.output("usage: /agents [list|add|run|status|pause|resume|stop]")

    def _hooks(self, args: List[str]) -> None:
        if not args or args[0] == "list":
            hooks = [rule.to_dict() for rule in self.hooks.list_rules()]
            if not hooks:
                self.output("no hooks")
                return
            for hook in hooks:
                self.output(
                    f"{hook.get('id')}\t{hook.get('event')}\t"
                    f"{hook.get('enabled')}\t{hook.get('name')}\t{hook.get('action')}"
                )
            return
        if args[0] == "add":
            if len(args) < 4:
                self.output("usage: /hooks add <name> <event> <command>")
                return
            item = self.extensions.add_hook(args[1], args[2], " ".join(args[3:]))
            self.output(f"hook added: {item['id']}")
            return
        if args[0] in {"enable", "disable"}:
            if len(args) < 2:
                self.output("usage: /hooks enable|disable <id|name>")
                return
            matched = self.hooks.set_enabled(args[1], args[0] == "enable")
            self.output("hook updated" if matched else "hook not found")
            return
        if args[0] == "test":
            event = args[1] if len(args) > 1 else "UserPromptSubmit"
            text = " ".join(args[2:])
            result = self.hooks.run(event, {"line": text, "tool_name": text})
            self.output(
                f"blocked={result.blocked} matched={','.join(result.matched_rules)}"
            )
            for warning in result.warnings:
                self.output(f"warning: {warning}")
            return
        self.output("usage: /hooks [list|add <name> <event> <command>|enable <id>|disable <id>|test <event> <text>]")

    def _commands(self, args: List[str]) -> None:
        if args and args[0] == "render":
            if len(args) < 2:
                self.output("usage: /commands render <name> [arguments]")
                return
            rendered = self.command_store.render(args[1], " ".join(args[2:]))
            self.output(rendered.prompt)
            return
        commands = self.command_store.discover()
        if not commands:
            self.output("no slash commands")
            return
        for command in commands:
            tools = ",".join(command.allowed_tools)
            hint = f" {command.argument_hint}" if command.argument_hint else ""
            self.output(f"/{command.name}{hint}\t{command.source}\t{tools}\t{command.description}")

    def _custom_command(self, command: str, args: List[str]) -> bool:
        name = command.lstrip("/")
        spec = self.command_store.get(name)
        if spec is None:
            return False
        executor = SafeAgentToolExecutor(
            self.workspace.project_root,
            self.workspace.workspace_dir,
            permission_policy=self.permission_policy,
            permission_callback=self._confirm_permission,
            session_permission_cache=self.permission_cache,
            hook_engine=self.hooks,
        )
        rendered = self.command_store.render(
            name,
            " ".join(args),
            expand_shell=True,
            executor=executor,
        )
        for warning in rendered.warnings:
            self.output(f"warning: {warning}")
        if spec.allowed_tools:
            self._act(rendered.prompt)
        else:
            self._chat(rendered.prompt)
        return True

    def _todos(self, args: List[str]) -> None:
        owner = self.session_id
        if args and args[0] == "clear":
            todo_list = self.todos.clear(owner)
            self.output(f"todos cleared: {todo_list.owner}")
            return
        if args and args[0] == "export":
            output = Path(args[1]) if len(args) > 1 else None
            path = self.todos.export_markdown(owner, output)
            self.output(f"todos exported: {path}")
            return
        if args and args[0] == "set":
            content = " ".join(args[1:]).strip()
            if not content:
                self.output("usage: /todos set <content>")
                return
            todo_list = self.todos.replace(owner, [{"content": content, "status": "pending"}])
            self.output(f"todos updated: {len(todo_list.items)}")
            return
        todo_list = self.todos.load(owner)
        if not todo_list.items:
            self.output("no todos")
            return
        for item in todo_list.items:
            self.output(f"{item.id}\t{item.status}\t{item.priority}\t{item.content}")

    def _permissions(self) -> None:
        self.output(f"permission_policy: {self.permission_policy}")
        self.output("session grants:")
        if not self.permission_cache:
            self.output("  none")
            return
        for key, allowed in sorted(self.permission_cache.items()):
            self.output(f"  {key}: {allowed}")

    def _export(self, args: List[str]) -> None:
        session = self.conversations.load(self.session_id)
        output = (
            Path(args[0])
            if args
            else self.workspace.workspace_dir / "exports" / f"{self.session_id}.md"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# HyperAgent Session Export - {session.session_id}", ""]
        for message in session.messages:
            lines.append(f"## {message.role}")
            lines.append(message.content)
            lines.append("")
        output.write_text("\n".join(lines), encoding="utf-8")
        self.output(f"exported: {output}")

    def _doctor(self) -> None:
        status = self.workspace.status()
        lines = [
            "HyperAgent doctor:",
            f"- initialized: {status.initialized}",
            f"- workspace: {status.workspace_dir}",
            f"- permission_policy: {self.permission_policy}",
            f"- providers: {len(self.providers.ensure_defaults())}",
            f"- commands: {len(self.command_store.discover())}",
            f"- subagents: {len(self.extensions.list_subagents())}",
            f"- hooks: {len(self.hooks.list_rules())}",
        ]
        self.output("\n".join(lines))

    def _plugins(self, args: List[str]) -> None:
        if not args or args[0] == "list":
            plugins = self.extensions.list_plugins()
            if not plugins:
                self.output("no plugins")
                return
            for plugin in plugins:
                self.output(
                    f"{plugin.get('id')}\t{plugin.get('name')}\t"
                    f"{plugin.get('enabled')}\t{plugin.get('description')}"
                )
            return
        if args[0] == "add":
            if len(args) < 2:
                self.output("usage: /plugin add <name> [description]")
                return
            item = self.extensions.add_plugin(args[1], description=" ".join(args[2:]))
            self.output(f"plugin added: {item['id']}")
            return
        self.output("usage: /plugin [list|add <name> [description]]")

    def _rewind(self, args: List[str]) -> None:
        if args and args[0] == "save":
            path = self.extensions.create_rewind_snapshot(
                self.session_id,
                self.conversations.load(self.session_id).to_dict(),
            )
            self.output(f"rewind snapshot: {path}")
            return
        snapshots = self.extensions.list_rewind_snapshots()
        if not snapshots:
            self.output("no rewind snapshots")
            return
        for path in snapshots[-20:]:
            self.output(str(path))

    def _reasonix(self, args: List[str]) -> None:
        if args:
            profile = get_reasonix_profile(args[0])
            if profile is None:
                self.output("usage: /reasonix [profile]")
                return
            self.output(
                f"{profile.name}\n"
                f"- model: {profile.model}\n"
                f"- thinking: {profile.thinking}\n"
                f"- reasoning_effort: {profile.reasoning_effort}\n"
                f"- intent: {profile.intent}\n"
                f"- use_cases: {', '.join(profile.use_cases)}"
            )
            return
        lines = ["reasonix profiles:"]
        for profile in list_reasonix_profiles():
            lines.append(
                f"- {profile.name}: {profile.model}, thinking={profile.thinking}, "
                f"effort={profile.reasoning_effort}"
            )
        guidance = reasonix_cache_guidance()
        lines.append("cache rule: " + str(guidance["rule"]))
        self.output("\n".join(lines))

    def _preset(self, args: List[str]) -> None:
        if not args or args[0] == "list":
            self._reasonix([])
            return
        profile = get_reasonix_profile(args[0])
        if profile is None:
            self.output("usage: /preset [list|reasonix-cheap|reasonix-balanced|reasonix-deep]")
            return
        self.llm_kwargs["reasonix_profile"] = profile.name
        if profile.thinking:
            self.llm_kwargs["thinking"] = {"type": profile.thinking}
        if profile.reasoning_effort:
            self.llm_kwargs["reasoning_effort"] = profile.reasoning_effort
        self.model = profile.model
        self.output(f"preset: {profile.name} model={profile.model}")

    def _pro(self) -> None:
        self._preset(["reasonix-deep"])
        self.action_loop_mode = "cache-first"
        self.output("pro mode armed: reasonix-deep + cache-first action loop")

    def _budget(self, args: List[str]) -> None:
        if not args:
            self.output(f"token_budget: {self.action_token_budget}")
            return
        if args[0].lower() in {"off", "none", "clear"}:
            self.action_token_budget = None
        else:
            self.action_token_budget = int(args[0])
        self.output(f"token_budget: {self.action_token_budget}")

    def _thinking(self, args: List[str]) -> None:
        if not args or args[0] == "status":
            self.output(self._thinking_status())
            return
        action = args[0].lower()
        if action == "on":
            self.expand_reasoning_content = True
        elif action == "off":
            self.expand_reasoning_content = False
        elif action == "toggle":
            self.expand_reasoning_content = not self.expand_reasoning_content
        else:
            self.output("usage: /thinking [on|off|toggle|status]")
            return
        self.output(self._thinking_status())

    def _default_expand_reasoning_content(self) -> bool:
        return self._model_thinking_mode() == "enabled"

    def _model_thinking_mode(self) -> str:
        thinking = self.llm_kwargs.get("thinking")
        if isinstance(thinking, dict):
            value = str(thinking.get("type", "")).lower()
            if value in {"enabled", "disabled"}:
                return value
        return "unknown"

    def _reasoning_display_mode(self) -> str:
        return "expanded" if self.expand_reasoning_content else "collapsed"

    def _thinking_status(self) -> str:
        return (
            f"model thinking: {self._model_thinking_mode()}\n"
            f"reasoning display: {self._reasoning_display_mode()}"
        )

    def _simplify(self) -> None:
        self.output(
            "simplify council:\n"
            "- code_quality: inspect duplication, naming, tests, and module boundaries\n"
            "- runtime_efficiency: inspect slow paths, repeated IO, and avoidable work\n"
            "- reuse: inspect abstractions, registry use, and extension points\n"
            "Run `/plan simplify current changes from code quality, efficiency, and reuse perspectives` to generate a saved plan."
        )

    def _models(self) -> None:
        for provider in self.providers.ensure_defaults():
            self.output(
                f"{provider.name}\t{provider.kind}\t{provider.default_model}\t"
                f"{provider.api_key_env}"
            )

    def _mcp(self, args: Optional[List[str]] = None) -> None:
        args = args or []
        servers = MCPServerStore(self.workspace.workspace_dir).list()
        if args and args[0] in {"status", "health", "tools"}:
            self.output(f"mcp servers: {len(servers)}")
            for server in servers:
                self.output(f"{server.name}\t{server.enabled}\tregistered\t{server.command}")
            if not servers:
                self.output("no MCP servers")
            return
        if args and args[0] == "inspect":
            name = args[1] if len(args) > 1 else ""
            for server in servers:
                if server.name == name:
                    self.output(
                        f"name: {server.name}\n"
                        f"enabled: {server.enabled}\n"
                        f"command: {server.command} {' '.join(server.args)}\n"
                        "runtime_client: not_connected"
                    )
                    return
            self.output("MCP server not found")
            return
        if not servers:
            self.output("no MCP servers")
            return
        for server in servers:
            self.output(f"{server.name}\t{server.command}\t{' '.join(server.args)}")

    def _ide_context(self, args: List[str]) -> None:
        action = args[0] if args else "status"
        if action == "on":
            payload = self.ide_context.set_enabled(True)
        elif action == "off":
            payload = self.ide_context.set_enabled(False)
        elif action == "set-open-files":
            payload = self.ide_context.set_open_files(args[1:])
        elif action == "clear":
            payload = self.ide_context.clear()
        elif action == "status":
            payload = self.ide_context.load()
        else:
            self.output("usage: /ide-context status|on|off|set-open-files|clear")
            return
        self.output(
            f"ide_context: enabled={payload.get('enabled')} "
            f"open_files={','.join(payload.get('open_files', [])) or 'none'}"
        )

    def _plan_mode(self, args: List[str]) -> None:
        action = args[0] if args else "status"
        if action == "on":
            payload = self.plan_mode.set_enabled(True, " ".join(args[1:]))
        elif action == "off":
            payload = self.plan_mode.set_enabled(False, " ".join(args[1:]))
        elif action == "status":
            payload = self.plan_mode.load()
        else:
            self.output("usage: /plan-mode status|on|off")
            return
        self.output(f"plan_mode: enabled={payload.get('enabled')} reason={payload.get('reason', '')}")

    def _personality(self, args: List[str]) -> None:
        action = args[0] if args else "status"
        if action == "set":
            payload = self.personality.set(" ".join(args[1:]))
        elif action == "clear":
            payload = self.personality.clear()
        elif action == "status":
            payload = self.personality.load()
        else:
            self.output("usage: /personality status|set|clear")
            return
        self.output(payload.get("text") or "no personality note")

    def _feedback(self, args: List[str]) -> None:
        action = args[0] if args else "list"
        if action == "add":
            if len(args) < 2:
                self.output("usage: /feedback add <text>")
                return
            self.feedback.add(" ".join(args[1:]), source="repl")
            self.output("feedback recorded")
            return
        if action == "list":
            items = self.feedback.list()
            if not items:
                self.output("no feedback")
                return
            for item in items:
                self.output(f"{item.get('created_at', '')}\t{item.get('text', '')}")
            return
        self.output("usage: /feedback add|list")

    def _web(self, args: List[str]) -> None:
        action = args[0] if args else "status"
        if action == "status":
            payload = web_status()
            self.output(
                "web:\n"
                f"- search_configured: {payload['search_configured']}\n"
                f"- providers: {payload['providers']}\n"
                "- fetch_available: true"
            )
            return
        executor = SafeAgentToolExecutor(
            self.workspace.project_root,
            self.workspace.workspace_dir,
            permission_policy=self.permission_policy,
            permission_callback=self._confirm_permission,
            session_permission_cache=self.permission_cache,
            hook_engine=self.hooks,
        )
        if action == "search":
            query = " ".join(args[1:]).strip()
            if not query:
                self.output("usage: /web search <query>")
                return
            result = executor.web_search(query)
        elif action in {"fetch", "extract"}:
            if len(args) < 2:
                self.output(f"usage: /web {action} <url>")
                return
            result = (
                executor.web_fetch(args[1])
                if action == "fetch"
                else executor.web_extract(args[1])
            )
        elif action == "cite":
            result = executor.web_cite(args[1] if len(args) > 1 else "")
        else:
            self.output("usage: /web status|search <query>|fetch <url>|extract <url>|cite [id]")
            return
        self.output(render_tool_result(result))

    def _image(self, args: List[str]) -> None:
        action = args[0] if args else "status"
        if action == "status":
            payload = image_status()
            self.output(
                f"image: provider={payload['provider']} configured={payload['configured']} "
                f"required_env={payload['required_env']}"
            )
            return
        executor = SafeAgentToolExecutor(
            self.workspace.project_root,
            self.workspace.workspace_dir,
            permission_policy=self.permission_policy,
            permission_callback=self._confirm_permission,
            session_permission_cache=self.permission_cache,
            hook_engine=self.hooks,
        )
        if action == "generate":
            prompt = " ".join(args[1:]).strip()
            if not prompt:
                self.output("usage: /image generate <prompt>")
                return
            result = executor.image_generate(prompt)
        elif action == "edit":
            if len(args) < 3:
                self.output("usage: /image edit <path> <instruction>")
                return
            result = executor.image_edit(args[1], " ".join(args[2:]))
        else:
            self.output("usage: /image status|generate <prompt>|edit <path> <instruction>")
            return
        self.output(render_tool_result(result))

    def _worktree(self) -> None:
        payload = worktree_status(self.workspace.project_root)
        self.output(
            f"branch: {payload['branch']}\n"
            f"head: {payload['head']}\n"
            "dirty_files:\n"
            + "\n".join(f"- {item}" for item in (payload["dirty_files"] or ["none"]))
        )

    def _skills(self, args: Optional[List[str]] = None) -> None:
        args = args or []
        roots = [
            Path(__file__).resolve().parents[1] / "skills",
            Path("skills"),
            self.workspace.workspace_dir / "skills",
        ]
        store = SkillStore(roots)
        if args and args[0] in {"show", "render"} and len(args) >= 2:
            try:
                skill = store.render(args[1], " ".join(args[2:]))
            except KeyError as exc:
                self.output(str(exc))
                return
            self.output(skill.body)
            return
        if args and args[0] == "run" and len(args) >= 2:
            self._act(
                "Run skill `{}` with arguments:\n{}".format(
                    args[1],
                    " ".join(args[2:]),
                )
            )
            return
        skills = store.list()
        if not skills:
            self.output("no skills")
            return
        for skill in skills:
            self.output(f"{skill.name}\t{skill.run_as}\t{skill.path}\t{skill.description}")

    def _checkpoint(self, args: List[str]) -> None:
        store = CheckpointStore(self.workspace.project_root, self.workspace.workspace_dir)
        if not args or args[0] == "list":
            checkpoints = store.list()
            if not checkpoints:
                self.output("no checkpoints")
                return
            for item in checkpoints[-20:]:
                self.output(f"{item.checkpoint_id}\tfiles={len(item.files)}\t{item.reason}")
            return
        checkpoint = store.create(args, reason="manual REPL checkpoint")
        self.output(f"checkpoint: {checkpoint.checkpoint_id}")

    def _restore(self, args: List[str]) -> None:
        if not args:
            self.output("usage: /restore <checkpoint_id>")
            return
        checkpoint = CheckpointStore(
            self.workspace.project_root,
            self.workspace.workspace_dir,
        ).restore(args[0])
        self.output(f"restored: {checkpoint.checkpoint_id} files={len(checkpoint.files)}")

    def _logs(self, args: List[str]) -> None:
        limit = int(args[0]) if args else 20
        events = RuntimeEventLog(self.workspace.workspace_dir).records(limit=limit)
        if not events:
            self.output("no runtime events")
            return
        for event in events:
            self.output(
                f"{event.timestamp}\t{event.event_type}\t{event.status}\t"
                f"{event.message[:160]}"
            )

    def _jobs(self, args: List[str]) -> None:
        jobs = self.jobs.list()
        if args and args[0] == "clear":
            self.jobs.save_all([])
            self.output("jobs cleared")
            return
        if not jobs:
            self.output("no background jobs")
            return
        for job in jobs[-20:]:
            self.output(
                f"{job.job_id}\t{job.status}\t{job.kind}\t"
                f"session={job.session_id or '-'}\t{job.instruction[:120]}"
            )

    def _manual_tool(self, args: List[str]) -> None:
        if not args:
            self.output(self._tool_usage())
            return
        if self.plan_mode.load().get("enabled"):
            self.output("plan-mode is on; manual tools are paused. Use /plan-mode off.")
            return
        executor = SafeAgentToolExecutor(
            self.workspace.project_root,
            self.workspace.workspace_dir,
            permission_policy=self.permission_policy,
            permission_callback=self._confirm_permission,
            session_permission_cache=self.permission_cache,
            hook_engine=self.hooks,
        )
        tool = args[0]
        rest = args[1:]
        if tool == "read":
            path = rest[0] if rest else ""
            start_line = int(rest[1]) if len(rest) > 1 else 1
            max_lines = int(rest[2]) if len(rest) > 2 else 120
            result = executor.read_file(path, start_line=start_line, max_lines=max_lines)
        elif tool == "search":
            query = rest[0] if rest else ""
            path = rest[1] if len(rest) > 1 else "."
            result = executor.search_code(query, path=path)
        elif tool == "run":
            result = executor.run_command(rest)
        elif tool in {"run-experiment", "experiment"}:
            path = rest[0] if rest else ""
            seeds = (
                [int(seed) for seed in rest[1].split(",") if seed]
                if len(rest) > 1 and rest[1]
                else None
            )
            result = executor.run_experiment(path, seeds=seeds)
        elif tool == "check-patch":
            result = executor.check_patch(self._read_text_arg(rest))
        elif tool == "apply-patch":
            result = executor.apply_patch(self._read_text_arg(rest))
        elif tool in {"todo", "todo-write", "todo_write"}:
            content = " ".join(rest).strip()
            items = [{"content": content, "status": "pending"}] if content else []
            result = executor.todo_write(items, owner=self.session_id)
        elif tool == "web-search":
            result = executor.web_search(" ".join(rest))
        elif tool == "web-fetch":
            result = executor.web_fetch(rest[0] if rest else "")
        elif tool == "web-cite":
            result = executor.web_cite(rest[0] if rest else "")
        elif tool == "image-generate":
            result = executor.image_generate(" ".join(rest))
        else:
            self.output(f"unknown tool: {tool}")
            return
        self.output(render_tool_result(result))

    def _read_text_arg(self, args: List[str]) -> str:
        if not args:
            return ""
        path = Path(args[0])
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        return " ".join(args)

    def _confirm_permission(self, request: ToolPermissionRequest) -> bool:
        if self.permission_policy not in {"ask", "session-ask"}:
            return True
        self.output(
            self._t(
                "repl.permission_requested",
                "permission requested: {tool_name} risk={risk_level} reason={reason}",
                tool_name=request.tool_name,
                risk_level=request.risk_level,
                reason=request.reason,
            )
        )
        answer = self.input(self._t("repl.allow_prompt", "allow? [y/N] ")).strip().lower()
        return answer in {"y", "yes"}

    def _tool_names(self) -> List[str]:
        return [
            "read",
            "search",
            "run",
            "run-experiment",
            "check-patch",
            "apply-patch",
            "todo-write",
            "web-search",
            "web-fetch",
            "web-cite",
            "image-generate",
        ]

    def _tool_usage(self) -> str:
        return (
            "usage:\n"
            "  /tool read <path> [start_line] [max_lines]\n"
            "  /tool search <query> [path]\n"
            "  /tool run <argv...>\n"
            "  /tool run-experiment <experiment_yaml> [seed1,seed2]\n"
            "  /tool check-patch <patch_file_or_text>\n"
            "  /tool apply-patch <patch_file_or_text>\n"
            "  /tool todo-write <todo content>\n"
            "  /tool web-search <query>\n"
            "  /tool web-fetch <url>\n"
            "  /tool web-cite [citation_id]\n"
            "  /tool image-generate <prompt>"
        )

    def _help(self) -> str:
        help_text = self._t(
            "repl.help",
            (
            "HyperAgent REPL commands:\n"
            "/help                 show commands\n"
            "/status               show workspace status\n"
            "/session              show current session\n"
            "/btw <question>       ask an isolated temporary question\n"
            "/sessions             list sessions\n"
            "/new [title]          create a new session\n"
            "/resume <session_id>  switch session\n"
            "/context              show context compression status\n"
            "/usage [limit]        summarize LLM usage and cache-hit ledger\n"
            "/cost [limit]         alias for usage/cost ledger\n"
            "/stats                summarize events and LLM usage\n"
            "/retry                retry the previous plain-text user message\n"
            "/stop                 report active request status\n"
            "/copy [path]          export current session for terminal copy\n"
            "/compact [keep_last]  compress current session\n"
            "/clear                clear current context after saving a rewind snapshot\n"
            "/init                 create project HyperAgent.md memory\n"
            "/memory ...           list/show/add memory entries\n"
            "/agents ...           list/add/run project subagents\n"
            "/commands ...         list/render Markdown slash commands\n"
            "/todos ...            list/clear/export TodoWrite state\n"
            "/hooks ...            list/add/enable/disable/test project hooks\n"
            "/permissions          show current permission policy and session grants\n"
            "/export [path]        export current session as Markdown\n"
            "/doctor               run a local workspace self-check\n"
            "/plugin ...           list/add project plugins\n"
            "/rewind [save]        list or save rewind snapshots\n"
            "/reasonix [profile]   show DeepSeek Reasonix-inspired profiles\n"
            "/preset ...           list or set Reasonix-style runtime preset\n"
            "/pro                  arm reasonix-deep + cache-first mode\n"
            "/budget [tokens|off]  show or set action-loop token budget\n"
            "/thinking ...         expand/collapse model reasoning_content display\n"
            "/simplify             show the three-agent simplification council\n"
            "/model                list LLM providers\n"
            "/ide-context ...      manage manually supplied IDE context\n"
            "/mcp ...              list/check MCP servers\n"
            "/personality ...      show/update local interaction personality note\n"
            "/feedback ...         record/list local feedback notes\n"
            "/plan-mode ...        toggle plan-only mode\n"
            "/web ...              search/fetch public web through controlled tools\n"
            "/image ...            create image request artifacts\n"
            "/worktree             show git worktree status\n"
            "/skill ...            list/show/run skills\n"
            "/checkpoint ...       list or create file checkpoints\n"
            "/restore <id>         restore a checkpoint\n"
            "/jobs                 show background job status\n"
            "/logs [limit]         show recent runtime events\n"
            "/tools                list local tools\n"
            "/tool ...             run a local tool with permission policy\n"
            "/plan <instruction>   generate a coding/algorithm plan\n"
            "/act <instruction>    run controlled LLM tool loop\n"
            "/exit                 quit\n"
            "Plain text sends a persistent agent-chat turn."
            ),
        )
        heading = self._t(
            "repl.central_command_registry",
            "Central command registry",
        )
        return help_text + f"\n\n{heading}:\n" + grouped_help(translator=self.translator)

    def _t(self, key: str, default: str, **kwargs) -> str:
        if self.translator is None:
            try:
                return default.format(**kwargs)
            except (KeyError, ValueError):
                return default
        return self.translator.t(key, default=default, **kwargs)

    def _emit_hook_messages(self, event: str, payload: Dict[str, object]) -> None:
        result = self.hooks.run(event, payload)
        for message in result.system_messages:
            self.output(f"hook: {message}")
        for warning in result.warnings:
            self.output(f"hook warning: {warning}")
