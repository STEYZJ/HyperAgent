"""Prompt template library."""

import re
from pathlib import Path
from typing import Dict, List

from hyperagent.schemas import PromptTemplate


VAR_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


class PromptLibrary:
    def __init__(self, roots: List[Path]) -> None:
        self.roots = roots

    def list(self) -> List[PromptTemplate]:
        templates: List[PromptTemplate] = []
        for root in self.roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.md")):
                text = path.read_text(encoding="utf-8", errors="ignore")
                description = self._description(text)
                variables = sorted(set(VAR_RE.findall(text)))
                templates.append(
                    PromptTemplate(
                        name=path.stem,
                        path=str(path),
                        description=description,
                        variables=variables,
                    )
                )
        return templates

    def render(self, name: str, values: Dict[str, str]) -> str:
        template = self._find(name)
        text = Path(template.path).read_text(encoding="utf-8")
        for variable in template.variables:
            text = text.replace("{{" + variable + "}}", values.get(variable, ""))
        return text

    def _find(self, name: str) -> PromptTemplate:
        for template in self.list():
            if template.name == name:
                return template
        raise KeyError(f"Prompt template not found: {name}")

    def _description(self, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip("# ").strip()
            if stripped:
                return stripped[:240]
        return ""

