"""Markdown slash command discovery and rendering."""

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.schemas import SlashCommandSpec


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class RenderedSlashCommand:
    spec: SlashCommandSpec
    arguments: str
    prompt: str
    warnings: List[str]


class SlashCommandStore:
    """Loads Claude-Code-style slash command Markdown files."""

    def __init__(
        self,
        project_root: Path,
        workspace_dir: Path,
        *,
        builtin_root: Optional[Path] = None,
    ) -> None:
        self.project_root = project_root
        self.workspace_dir = workspace_dir
        self.builtin_root = builtin_root or Path(__file__).resolve().parents[1] / "commands"

    def discover(self, include_hidden: bool = False) -> List[SlashCommandSpec]:
        specs: Dict[str, SlashCommandSpec] = {}
        for root, source, namespace in self._roots():
            if not root.exists():
                continue
            for path in sorted(root.glob("*.md")):
                spec = self._load_file(path, source=source, namespace=namespace)
                if spec.hidden and not include_hidden:
                    continue
                specs[spec.name] = spec
        return sorted(specs.values(), key=lambda item: item.name)

    def get(self, name: str) -> Optional[SlashCommandSpec]:
        normalized = self._normalize_name(name)
        for spec in self.discover(include_hidden=True):
            if spec.name == normalized:
                return spec
        return None

    def render(
        self,
        name: str,
        arguments: str = "",
        *,
        expand_shell: bool = False,
        executor: Optional[SafeAgentToolExecutor] = None,
    ) -> RenderedSlashCommand:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"slash command not found: {name}")
        warnings: List[str] = []
        prompt = spec.body.replace("$ARGUMENTS", arguments)
        if expand_shell:
            prompt, shell_warnings = self._expand_shell_context(prompt, spec, executor)
            warnings.extend(shell_warnings)
        return RenderedSlashCommand(spec=spec, arguments=arguments, prompt=prompt, warnings=warnings)

    def _roots(self) -> Iterable[tuple]:
        yield self.builtin_root, "builtin", ""
        yield self.workspace_dir / "commands", "workspace", ""
        plugins_root = self.workspace_dir / "plugins"
        if plugins_root.exists():
            for plugin_dir in sorted(plugins_root.iterdir()):
                commands_dir = plugin_dir / "commands"
                if commands_dir.exists():
                    yield commands_dir, "plugin", plugin_dir.name

    def _load_file(self, path: Path, *, source: str, namespace: str) -> SlashCommandSpec:
        text = path.read_text(encoding="utf-8")
        metadata: Dict[str, object] = {}
        body = text
        match = FRONTMATTER_RE.match(text)
        if match:
            raw = yaml.safe_load(match.group(1)) or {}
            if isinstance(raw, dict):
                metadata = raw
            body = text[match.end() :]
        base_name = self._normalize_name(str(metadata.get("name") or path.stem))
        name = f"{namespace}:{base_name}" if namespace and ":" not in base_name else base_name
        allowed_tools = metadata.get("allowed-tools", metadata.get("allowed_tools", []))
        if isinstance(allowed_tools, str):
            allowed_tools = [item.strip() for item in allowed_tools.split(",") if item.strip()]
        return SlashCommandSpec(
            name=name,
            path=str(path),
            body=body.strip(),
            description=str(metadata.get("description", "")).strip(),
            argument_hint=str(metadata.get("argument-hint", metadata.get("argument_hint", ""))).strip(),
            allowed_tools=[str(item) for item in allowed_tools] if isinstance(allowed_tools, list) else [],
            model=str(metadata.get("model", "")).strip(),
            profile=str(metadata.get("profile", "")).strip(),
            hidden=bool(metadata.get("hidden", False)),
            source=source,
            namespace=namespace,
            metadata=dict(metadata),
        )

    def _normalize_name(self, name: str) -> str:
        return str(name).strip().lstrip("/").lower()

    def _expand_shell_context(
        self,
        prompt: str,
        spec: SlashCommandSpec,
        executor: Optional[SafeAgentToolExecutor],
    ) -> tuple:
        warnings: List[str] = []
        rendered_lines: List[str] = []
        allowed = set(spec.allowed_tools)
        for line in prompt.splitlines():
            stripped = line.strip()
            if not stripped.startswith("!"):
                rendered_lines.append(line)
                continue
            command_text = stripped[1:].strip()
            if "run_command" not in allowed:
                warnings.append(f"shell context skipped because run_command is not allowed: {command_text}")
                rendered_lines.append(f"[shell context skipped: {command_text}]")
                continue
            if executor is None:
                warnings.append(f"shell context skipped because no executor is configured: {command_text}")
                rendered_lines.append(f"[shell context unavailable: {command_text}]")
                continue
            try:
                argv = shlex.split(command_text)
            except ValueError as exc:
                warnings.append(f"shell context parse failed: {exc}")
                rendered_lines.append(f"[shell context parse failed: {command_text}]")
                continue
            result = executor.run_command(argv, timeout_sec=20)
            rendered_lines.append(
                f"[shell context: {command_text}]\nstatus={result.status}\n{result.content}".rstrip()
            )
            warnings.extend(result.warnings)
        return "\n".join(rendered_lines), warnings
