"""Integration schemas for skills, MCP, Obsidian, and prompts."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillSpec:
    name: str
    path: str
    description: str = ""
    source: str = "local"
    body: str = ""
    run_as: str = "inline"
    allowed_tools: List[str] = field(default_factory=list)
    model: str = ""
    profile: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillSpec":
        return cls(
            name=str(data["name"]),
            path=str(data.get("path", "")),
            description=str(data.get("description", "")),
            source=str(data.get("source", "local")),
            body=str(data.get("body", "")),
            run_as=str(data.get("run_as", data.get("runAs", "inline"))),
            allowed_tools=[str(v) for v in data.get("allowed_tools", data.get("allowed-tools", []))],
            model=str(data.get("model", "")),
            profile=str(data.get("profile", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MCPServerSpec:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerSpec":
        return cls(
            name=str(data["name"]),
            command=str(data["command"]),
            args=[str(v) for v in data.get("args", [])],
            env={str(k): str(v) for k, v in data.get("env", {}).items()},
            enabled=bool(data.get("enabled", True)),
            description=str(data.get("description", "")),
        )


@dataclass
class ObsidianNote:
    path: str
    title: str
    tags: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObsidianNote":
        return cls(
            path=str(data["path"]),
            title=str(data["title"]),
            tags=[str(v) for v in data.get("tags", [])],
            links=[str(v) for v in data.get("links", [])],
            preview=str(data.get("preview", "")),
        )


@dataclass
class PromptTemplate:
    name: str
    path: str
    description: str = ""
    variables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaterializationResult:
    proposal_name: str
    model_name: str
    model_file: str
    ablation_dir: Optional[str] = None
    generated_configs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
