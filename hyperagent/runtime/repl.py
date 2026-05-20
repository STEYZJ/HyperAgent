"""Interactive HyperAgent REPL."""

from pathlib import Path
from typing import Callable, Dict, List, Optional

from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.agent_loop import AgentLoop
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor, ToolPermissionRequest
from hyperagent.runtime.coding_agent import CodingAgent
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.mcp import MCPServerStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.skills import SkillStore
from hyperagent.runtime.tool_panel import (
    render_action_run,
    render_tool_catalog,
    render_tool_result,
)
from hyperagent.runtime.workspace import HyperAgentWorkspace


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


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
        self.session_id = self._ensure_session(session_id, new_title)

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
            if line in {"/exit", "/quit", "exit", "quit"}:
                self.output("bye")
                return 0
            try:
                if line.startswith("/"):
                    if self._handle_command(line):
                        continue
                    return 0
                self._chat(line)
            except Exception as exc:
                self.output(f"error: {exc}")

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
        elif command == "/context":
            self._context()
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
        result = AgentLoop(
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
            **self.llm_kwargs,
        )
        for warning in result.warnings:
            self.output(f"warning: {warning}")
        if result.response.reasoning_content:
            self.output("[reasoning]")
            self.output(result.response.reasoning_content)
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
            permission_policy=self.permission_policy,
            permission_callback=self._confirm_permission,
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

    def _compact(self, args: List[str]) -> None:
        keep_last = int(args[0]) if args else self.keep_last
        session = self.conversations.compress(self.session_id, keep_last=keep_last)
        self.output(
            f"compacted: messages={len(session.messages)} summaries={len(session.summaries)}"
        )

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
        if self.permission_policy != "ask":
            return True
        self.output(
            f"permission requested: {request.tool_name} "
            f"risk={request.risk_level} reason={request.reason}"
        )
        answer = self.input("allow? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def _tool_names(self) -> List[str]:
        return ["read", "search", "run", "check-patch", "apply-patch"]

    def _tool_usage(self) -> str:
        return (
            "usage:\n"
            "  /tool read <path> [start_line] [max_lines]\n"
            "  /tool search <query> [path]\n"
            "  /tool run <argv...>\n"
            "  /tool check-patch <patch_file_or_text>\n"
            "  /tool apply-patch <patch_file_or_text>"
        )

    def _help(self) -> str:
        return (
            "HyperAgent REPL commands:\n"
            "/help                 show commands\n"
            "/status               show workspace status\n"
            "/session              show current session\n"
            "/sessions             list sessions\n"
            "/new [title]          create a new session\n"
            "/resume <session_id>  switch session\n"
            "/context              show context compression status\n"
            "/compact [keep_last]  compress current session\n"
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
