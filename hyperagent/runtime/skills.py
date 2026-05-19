"""Skill discovery compatible with SKILL.md style directories."""

from pathlib import Path
from typing import Iterable, List

from hyperagent.schemas import SkillSpec


class SkillStore:
    def __init__(self, roots: Iterable[Path]) -> None:
        self.roots = [Path(root) for root in roots]

    def list(self) -> List[SkillSpec]:
        skills: List[SkillSpec] = []
        for root in self.roots:
            if not root.exists():
                continue
            for skill_file in root.rglob("SKILL.md"):
                content = skill_file.read_text(encoding="utf-8", errors="ignore")
                description = self._first_description(content)
                skills.append(
                    SkillSpec(
                        name=skill_file.parent.name,
                        path=str(skill_file),
                        description=description,
                        source=str(root),
                    )
                )
        return sorted(skills, key=lambda item: item.name)

    def _first_description(self, content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip("# ").strip()
            if stripped and not stripped.lower().startswith("name:"):
                return stripped[:240]
        return ""

