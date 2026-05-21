"""Skill discovery and rendering compatible with SKILL.md style directories."""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from hyperagent.schemas import SkillSpec


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillStore:
    def __init__(self, roots: Iterable[Path]) -> None:
        self.roots = [Path(root) for root in roots]

    def list(self) -> List[SkillSpec]:
        skills: List[SkillSpec] = []
        for root in self.roots:
            if not root.exists():
                continue
            for skill_file in root.rglob("SKILL.md"):
                skills.append(self._load(skill_file, source=str(root)))
        return sorted(skills, key=lambda item: item.name)

    def get(self, name: str) -> Optional[SkillSpec]:
        normalized = str(name).strip().lower()
        for skill in self.list():
            if skill.name.lower() == normalized:
                return skill
        return None

    def render(self, name: str, arguments: str = "") -> SkillSpec:
        skill = self.get(name)
        if skill is None:
            raise KeyError(f"skill not found: {name}")
        return SkillSpec(
            name=skill.name,
            path=skill.path,
            description=skill.description,
            source=skill.source,
            body=skill.body.replace("$ARGUMENTS", arguments),
            run_as=skill.run_as,
            allowed_tools=list(skill.allowed_tools),
            model=skill.model,
            profile=skill.profile,
            metadata=dict(skill.metadata),
        )

    def _load(self, skill_file: Path, *, source: str) -> SkillSpec:
        content = skill_file.read_text(encoding="utf-8", errors="ignore")
        metadata: Dict[str, object] = {}
        body = content
        match = FRONTMATTER_RE.match(content)
        if match:
            raw = yaml.safe_load(match.group(1)) or {}
            if isinstance(raw, dict):
                metadata = raw
            body = content[match.end() :]
        allowed_tools = metadata.get("allowed-tools", metadata.get("allowed_tools", []))
        if isinstance(allowed_tools, str):
            allowed_tools = [item.strip() for item in allowed_tools.split(",") if item.strip()]
        run_as = str(metadata.get("runAs", metadata.get("run_as", "inline"))).strip() or "inline"
        return SkillSpec(
            name=str(metadata.get("name") or skill_file.parent.name).strip(),
            path=str(skill_file),
            description=str(metadata.get("description") or self._first_description(body)).strip(),
            source=source,
            body=body.strip(),
            run_as=run_as,
            allowed_tools=[str(item) for item in allowed_tools] if isinstance(allowed_tools, list) else [],
            model=str(metadata.get("model", "")).strip(),
            profile=str(metadata.get("profile", "")).strip(),
            metadata=dict(metadata),
        )

    def _first_description(self, content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip("# ").strip()
            if stripped and not stripped.lower().startswith("name:"):
                return stripped[:240]
        return ""
