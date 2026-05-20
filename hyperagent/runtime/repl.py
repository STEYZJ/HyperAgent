"""Interactive HyperAgent REPL."""

from pathlib import Path
from typing import Callable, Dict, List, Optional

from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.agent_loop import AgentLoop
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor, ToolPermissionRequest
from hyperagent.runtime.coding_agent import CodingAgent
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.general_agent import GeneralAgentRunner
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.llm_usage import LLMUsageLedger
from hyperagent.runtime.memory import MemoryStore
from hyperagent.runtime.mcp import MCPServerStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.deepseek_reasonix import (
    get_reasonix_profile,
    list_reasonix_profiles,
    reasonix_cache_guidance,
)
from hyperagent.runtime.skills import SkillStore
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
from hyperagent.runtime.workspace import HyperAgentWorkspace
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
        self.input = input_func
        self.output = output_func
        self.wait_indicator_factory = wait_indicator_factory
        self.memory = MemoryStore(workspace.project_root, workspace.workspace_dir)
        self.extensions = RuntimeExtensionStore(workspace.workspace_dir)
        self.usage = LLMUsageLedger(workspace.workspace_dir)
        self.permission_cache: Dict[str, bool] = {}
        self.session_id = self._ensure_session(session_id, new_title)
        self.show_thinking = self._default_show_thinking()

    def run(self) -> int:
        self.output(self._banner())
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
            self.output("bye")
            return False
        try:
            if line.startswith("/"):
                return self._handle_command(line)
            self._chat(line)
        except Exception as exc:
            self.output(f"error: {exc}")
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
        return (
            "HyperAgent interactive mode\n"
            f"session: {self.session_id}\n"
            f"provider: {self.provider}\n"
            "type /help for commands, /exit to quit"
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
        elif command == "/btw":
            self._btw(args)
        elif command == "/sessions":
            self._sessions()
        elif command == "/new":
            title = " ".join(args) or "HyperAgent REPL"
            self.session_id = self.conversations.new(title).session_id
            self.output(f"new session: {self.session_id}")
        elif command == "/resume":
            self._resume(args)
        elif command in {"/compact", "/compress"}:
            self._compact(args)
        elif command == "/clear":
            self._clear()
        elif command == "/context":
            self._context()
        elif command == "/usage":
            self._usage(args)
        elif command == "/init":
            self._init_memory()
        elif command == "/memory":
            self._memory(args)
        elif command == "/agents":
            self._agents(args)
        elif command == "/hooks":
            self._hooks(args)
        elif command in {"/plugin", "/plugins"}:
            self._plugins(args)
        elif command == "/rewind":
            self._rewind(args)
        elif command in {"/reasonix", "/deepseek"}:
            self._reasonix(args)
        elif command == "/thinking":
            self._thinking(args)
        elif command == "/simplify":
            self._simplify()
        elif command == "/model":
            self._models()
        elif command == "/mcp":
            self._mcp()
        elif command in {"/skills", "/skill"}:
            self._skills()
        elif command == "/tools":
            self.output(render_tool_catalog(self._tool_names()))
        elif command == "/tool":
            self._manual_tool(args)
        elif command == "/act":
            self._act(" ".join(args))
        elif command == "/plan":
            self._plan(" ".join(args))
        elif command in {"/exit", "/quit"}:
            return False
        else:
            self.output(f"unknown command: {command}")
        return True

    def _chat(self, message: str) -> None:
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
                thinking_displayed=self.show_thinking,
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
            if self.show_thinking:
                self.output("【思考内容】")
                self.output(result.response.reasoning_content)
            else:
                self.output("【思考内容已隐藏，可用 /thinking on 查看】")
        if result.response.content:
            self.output(result.response.content)

    def _act(self, instruction: str) -> None:
        if not instruction:
            self.output("usage: /act <instruction>")
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
            ),
        ).run(
            session_id=self.session_id,
            provider=self.provider,
            instruction=instruction,
            model=self.model,
            task_id=self.task_id,
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
        self.output("usage: /agents [list|add <name> <role> [tools]|run <name|id> <instruction>]")

    def _hooks(self, args: List[str]) -> None:
        if not args or args[0] == "list":
            hooks = self.extensions.list_hooks()
            if not hooks:
                self.output("no hooks")
                return
            for hook in hooks:
                self.output(
                    f"{hook.get('id')}\t{hook.get('event')}\t"
                    f"{hook.get('name')}\t{hook.get('command')}"
                )
            return
        if args[0] == "add":
            if len(args) < 4:
                self.output("usage: /hooks add <name> <event> <command>")
                return
            item = self.extensions.add_hook(args[1], args[2], " ".join(args[3:]))
            self.output(f"hook added: {item['id']}")
            return
        self.output("usage: /hooks [list|add <name> <event> <command>]")

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

    def _thinking(self, args: List[str]) -> None:
        if not args or args[0] == "status":
            self.output(f"thinking: {'on' if self.show_thinking else 'off'}")
            return
        action = args[0].lower()
        if action == "on":
            self.show_thinking = True
        elif action == "off":
            self.show_thinking = False
        elif action == "toggle":
            self.show_thinking = not self.show_thinking
        else:
            self.output("usage: /thinking [on|off|toggle|status]")
            return
        self.output(f"thinking: {'on' if self.show_thinking else 'off'}")

    def _default_show_thinking(self) -> bool:
        thinking = self.llm_kwargs.get("thinking")
        if isinstance(thinking, dict):
            return str(thinking.get("type", "")).lower() == "enabled"
        return False

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

    def _mcp(self) -> None:
        servers = MCPServerStore(self.workspace.workspace_dir).list()
        if not servers:
            self.output("no MCP servers")
            return
        for server in servers:
            self.output(f"{server.name}\t{server.command}\t{' '.join(server.args)}")

    def _skills(self) -> None:
        roots = [Path("skills"), self.workspace.workspace_dir / "skills"]
        skills = SkillStore(roots).list()
        if not skills:
            self.output("no skills")
            return
        for skill in skills:
            self.output(f"{skill.name}\t{skill.path}\t{skill.description}")

    def _manual_tool(self, args: List[str]) -> None:
        if not args:
            self.output(self._tool_usage())
            return
        executor = SafeAgentToolExecutor(
            self.workspace.project_root,
            self.workspace.workspace_dir,
            permission_policy=self.permission_policy,
            permission_callback=self._confirm_permission,
            session_permission_cache=self.permission_cache,
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
            f"permission requested: {request.tool_name} "
            f"risk={request.risk_level} reason={request.reason}"
        )
        answer = self.input("allow? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def _tool_names(self) -> List[str]:
        return ["read", "search", "run", "run-experiment", "check-patch", "apply-patch"]

    def _tool_usage(self) -> str:
        return (
            "usage:\n"
            "  /tool read <path> [start_line] [max_lines]\n"
            "  /tool search <query> [path]\n"
            "  /tool run <argv...>\n"
            "  /tool run-experiment <experiment_yaml> [seed1,seed2]\n"
            "  /tool check-patch <patch_file_or_text>\n"
            "  /tool apply-patch <patch_file_or_text>"
        )

    def _help(self) -> str:
        return (
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
            "/compact [keep_last]  compress current session\n"
            "/clear                clear current context after saving a rewind snapshot\n"
            "/init                 create project HyperAgent.md memory\n"
            "/memory ...           list/show/add memory entries\n"
            "/agents ...           list/add/run project subagents\n"
            "/hooks ...            list/add project hooks\n"
            "/plugin ...           list/add project plugins\n"
            "/rewind [save]        list or save rewind snapshots\n"
            "/reasonix [profile]   show DeepSeek Reasonix-inspired profiles\n"
            "/thinking ...         show/hide reasoning_content blocks\n"
            "/simplify             show the three-agent simplification council\n"
            "/model                list LLM providers\n"
            "/mcp                  list MCP servers\n"
            "/skills               list skills\n"
            "/tools                list local tools\n"
            "/tool ...             run a local tool with permission policy\n"
            "/plan <instruction>   generate a coding/algorithm plan\n"
            "/act <instruction>    run controlled LLM tool loop\n"
            "/exit                 quit\n"
            "Plain text sends a persistent agent-chat turn."
        )
