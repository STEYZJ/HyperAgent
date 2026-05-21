"""Skill discovery and rendering compatible with SKILL.md style directories."""

import re
import shutil
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

    def search(self, query: str) -> List[SkillSpec]:
        needle = str(query).strip().lower()
        if not needle:
            return self.list()
        matches = []
        for skill in self.list():
            haystack = "\n".join(
                [
                    skill.name,
                    skill.description,
                    skill.body,
                    " ".join(skill.allowed_tools),
                    skill.run_as,
                ]
            ).lower()
            if needle in haystack:
                matches.append(skill)
        return matches

    def bundles(self) -> Dict[str, List[SkillSpec]]:
        grouped: Dict[str, List[SkillSpec]] = {}
        for skill in self.list():
            bundle = str(
                skill.metadata.get("bundle")
                or skill.metadata.get("category")
                or Path(skill.source).name
                or "local"
            )
            grouped.setdefault(bundle, []).append(skill)
        return grouped

    def install(self, source: Path, install_root: Path, *, name: str = "") -> SkillSpec:
        source = Path(source)
        install_root = Path(install_root)
        skill_file = source if source.name == "SKILL.md" else source / "SKILL.md"
        if not skill_file.exists():
            raise FileNotFoundError(f"SKILL.md not found: {source}")
        skill = self._load(skill_file, source=str(source.parent))
        target_name = name.strip() or skill.name or skill_file.parent.name
        target_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", target_name).strip("-")
        if not target_name:
            raise ValueError("skill name is empty")
        target_dir = install_root / target_name
        install_root.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target_dir, dirs_exist_ok=True)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill_file, target_dir / "SKILL.md")
        return self._load(target_dir / "SKILL.md", source=str(install_root))

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
