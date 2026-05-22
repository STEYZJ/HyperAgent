"""Schemas for paper-level research experience extraction.

These dataclasses intentionally describe transferable research strategy instead
of paper content. They are used by HyperAgent tools, CLI commands, and MCP
adapters, while HyperVault remains the decoupled storage/RAG backend.
"""

from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional


RESEARCH_EXPERIENCE_DIMENSIONS = {
    "research_pattern": [
        "novelty_construction",
        "problem_framing",
        "gap_definition",
        "contribution_packaging",
        "reviewer_expectation",
    ],
    "experiment_strategy": [
        "baseline_selection",
        "ablation_logic",
        "control_variable",
        "robustness_validation",
        "visualization_strategy",
    ],
    "scientific_storytelling": [
        "narrative_pacing",
        "figure_order",
        "motivation_progression",
        "claim_scaffolding",
        "failure_hiding",
        "reviewer_persuasion",
    ],
    "research_taste": [
        "trend_sensing",
        "sota_evolution",
        "field_hotspot",
        "benchmark_lifecycle",
        "innovation_density",
    ],
}


STRATEGY_WORDS = {
    "author",
    "authors",
    "strategy",
    "frame",
    "framing",
    "package",
    "packaging",
    "claim",
    "reviewer",
    "persuasion",
    "baseline",
    "ablation",
    "credibility",
    "limitation",
    "narrative",
    "evidence",
    "transfer",
    "作者",
    "策略",
    "包装",
    "构造",
    "叙事",
    "说服",
    "审稿",
    "可信度",
    "弱点",
    "规避",
}


SUMMARY_PATTERNS = {
    "we propose",
    "this paper proposes",
    "the paper proposes",
    "we present",
    "this work presents",
    "本文提出",
    "本文设计",
    "本文介绍",
    "论文提出",
    "该方法",
}


def looks_like_content_summary(text: str) -> bool:
    """Return True when a lesson sounds like method-summary rather than strategy."""

    lowered = str(text or "").lower()
    has_summary_marker = any(pattern in lowered for pattern in SUMMARY_PATTERNS)
    has_strategy_marker = any(word in lowered for word in STRATEGY_WORDS)
    return has_summary_marker and not has_strategy_marker


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class EvidenceSpan:
    source_path: str
    quote: str
    chunk_id: str = ""
    heading: str = ""
    page: Optional[int] = None
    role: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceSpan":
        return cls(
            source_path=str(data.get("source_path", "")),
            quote=str(data.get("quote", "")),
            chunk_id=str(data.get("chunk_id", "")),
            heading=str(data.get("heading", "")),
            page=(None if data.get("page") in {None, ""} else int(data.get("page"))),
            role=str(data.get("role", "")),
            confidence=float(data.get("confidence", 0.5)),
        )


@dataclass
class StrategyLesson:
    dimension: str
    strategy_claim: str
    why_it_works: str
    evidence_span: EvidenceSpan
    transferable_template: str
    risk_or_limit: str
    confidence: float = 0.5
    tags: List[str] = dataclass_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyLesson":
        evidence = data.get("evidence_span", {})
        if not isinstance(evidence, dict):
            evidence = {}
        return cls(
            dimension=str(data.get("dimension", "")),
            strategy_claim=str(data.get("strategy_claim", "")),
            why_it_works=str(data.get("why_it_works", "")),
            evidence_span=EvidenceSpan.from_dict(evidence),
            transferable_template=str(data.get("transferable_template", "")),
            risk_or_limit=str(data.get("risk_or_limit", "")),
            confidence=float(data.get("confidence", 0.5)),
            tags=[str(item) for item in data.get("tags", [])],
        )

    def validation_warnings(self) -> List[str]:
        warnings: List[str] = []
        if not self.evidence_span.quote.strip() or not self.evidence_span.source_path.strip():
            warnings.append("strategy lesson is missing bound evidence")
        if looks_like_content_summary(self.strategy_claim):
            warnings.append("strategy_claim looks like paper-content summary")
        if looks_like_content_summary(self.transferable_template):
            warnings.append("transferable_template looks like paper-content summary")
        if self.dimension not in all_research_experience_dimensions():
            warnings.append("unknown research experience dimension: %s" % self.dimension)
        return warnings


def _lessons_from_dicts(items: Any) -> List[StrategyLesson]:
    return [StrategyLesson.from_dict(dict(item)) for item in _as_list(items) if isinstance(item, dict)]


@dataclass
class ResearchPattern:
    novelty_construction: List[StrategyLesson] = dataclass_field(default_factory=list)
    problem_framing: List[StrategyLesson] = dataclass_field(default_factory=list)
    gap_definition: List[StrategyLesson] = dataclass_field(default_factory=list)
    contribution_packaging: List[StrategyLesson] = dataclass_field(default_factory=list)
    reviewer_expectation: List[StrategyLesson] = dataclass_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchPattern":
        return cls(**{name: _lessons_from_dicts(data.get(name, [])) for name in RESEARCH_EXPERIENCE_DIMENSIONS["research_pattern"]})

    def lessons(self) -> List[StrategyLesson]:
        return sum((getattr(self, name) for name in RESEARCH_EXPERIENCE_DIMENSIONS["research_pattern"]), [])


@dataclass
class ExperimentStrategy:
    baseline_selection: List[StrategyLesson] = dataclass_field(default_factory=list)
    ablation_logic: List[StrategyLesson] = dataclass_field(default_factory=list)
    control_variable: List[StrategyLesson] = dataclass_field(default_factory=list)
    robustness_validation: List[StrategyLesson] = dataclass_field(default_factory=list)
    visualization_strategy: List[StrategyLesson] = dataclass_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentStrategy":
        return cls(**{name: _lessons_from_dicts(data.get(name, [])) for name in RESEARCH_EXPERIENCE_DIMENSIONS["experiment_strategy"]})

    def lessons(self) -> List[StrategyLesson]:
        return sum((getattr(self, name) for name in RESEARCH_EXPERIENCE_DIMENSIONS["experiment_strategy"]), [])


@dataclass
class ScientificStorytelling:
    narrative_pacing: List[StrategyLesson] = dataclass_field(default_factory=list)
    figure_order: List[StrategyLesson] = dataclass_field(default_factory=list)
    motivation_progression: List[StrategyLesson] = dataclass_field(default_factory=list)
    claim_scaffolding: List[StrategyLesson] = dataclass_field(default_factory=list)
    failure_hiding: List[StrategyLesson] = dataclass_field(default_factory=list)
    reviewer_persuasion: List[StrategyLesson] = dataclass_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScientificStorytelling":
        return cls(**{name: _lessons_from_dicts(data.get(name, [])) for name in RESEARCH_EXPERIENCE_DIMENSIONS["scientific_storytelling"]})

    def lessons(self) -> List[StrategyLesson]:
        return sum((getattr(self, name) for name in RESEARCH_EXPERIENCE_DIMENSIONS["scientific_storytelling"]), [])


@dataclass
class ResearchTaste:
    trend_sensing: List[StrategyLesson] = dataclass_field(default_factory=list)
    sota_evolution: List[StrategyLesson] = dataclass_field(default_factory=list)
    field_hotspot: List[StrategyLesson] = dataclass_field(default_factory=list)
    benchmark_lifecycle: List[StrategyLesson] = dataclass_field(default_factory=list)
    innovation_density: List[StrategyLesson] = dataclass_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchTaste":
        return cls(**{name: _lessons_from_dicts(data.get(name, [])) for name in RESEARCH_EXPERIENCE_DIMENSIONS["research_taste"]})

    def lessons(self) -> List[StrategyLesson]:
        return sum((getattr(self, name) for name in RESEARCH_EXPERIENCE_DIMENSIONS["research_taste"]), [])


@dataclass
class PaperStrategyCard:
    paper_id: str
    title: str
    research_pattern: ResearchPattern = dataclass_field(default_factory=ResearchPattern)
    experiment_strategy: ExperimentStrategy = dataclass_field(default_factory=ExperimentStrategy)
    scientific_storytelling: ScientificStorytelling = dataclass_field(default_factory=ScientificStorytelling)
    research_taste: ResearchTaste = dataclass_field(default_factory=ResearchTaste)
    venue: str = ""
    year: Optional[int] = None
    field: str = ""
    verified: bool = False
    evidence_level: str = "chunk"
    source_paths: List[str] = dataclass_field(default_factory=list)
    warnings: List[str] = dataclass_field(default_factory=list)
    extraction_mode: str = "heuristic"
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperStrategyCard":
        return cls(
            paper_id=str(data.get("paper_id", "")),
            title=str(data.get("title", "")),
            research_pattern=ResearchPattern.from_dict(dict(data.get("research_pattern", {}))),
            experiment_strategy=ExperimentStrategy.from_dict(dict(data.get("experiment_strategy", {}))),
            scientific_storytelling=ScientificStorytelling.from_dict(dict(data.get("scientific_storytelling", {}))),
            research_taste=ResearchTaste.from_dict(dict(data.get("research_taste", {}))),
            venue=str(data.get("venue", "")),
            year=(None if data.get("year") in {None, ""} else int(data.get("year"))),
            field=str(data.get("field", "")),
            verified=bool(data.get("verified", False)),
            evidence_level=str(data.get("evidence_level", "chunk")),
            source_paths=[str(item) for item in data.get("source_paths", [])],
            warnings=[str(item) for item in data.get("warnings", [])],
            extraction_mode=str(data.get("extraction_mode", "heuristic")),
            created_at=str(data.get("created_at", "")),
        )

    def all_lessons(self) -> List[StrategyLesson]:
        return (
            self.research_pattern.lessons()
            + self.experiment_strategy.lessons()
            + self.scientific_storytelling.lessons()
            + self.research_taste.lessons()
        )

    def validation_warnings(self) -> List[str]:
        warnings = list(self.warnings)
        for lesson in self.all_lessons():
            warnings.extend(lesson.validation_warnings())
        return warnings


def all_research_experience_dimensions() -> List[str]:
    dimensions: List[str] = []
    for values in RESEARCH_EXPERIENCE_DIMENSIONS.values():
        dimensions.extend(values)
    return dimensions
