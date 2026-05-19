"""Literature search schemas."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LiteraturePaper:
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    url: str
    abstract: str = ""
    source: str = "unknown"
    published: Optional[str] = None
    updated: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiteraturePaper":
        return cls(
            title=str(data["title"]),
            authors=[str(v) for v in data.get("authors", [])],
            year=None if data.get("year") is None else int(data["year"]),
            venue=data.get("venue"),
            url=str(data["url"]),
            abstract=str(data.get("abstract", "")),
            source=str(data.get("source", "unknown")),
            published=data.get("published"),
            updated=data.get("updated"),
            keywords=[str(v) for v in data.get("keywords", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class LiteratureSearchResult:
    query: str
    source: str
    max_results: int
    papers: List[LiteraturePaper]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiteratureSearchResult":
        return cls(
            query=str(data["query"]),
            source=str(data["source"]),
            max_results=int(data.get("max_results", len(data.get("papers", [])))),
            papers=[
                LiteraturePaper.from_dict(item) for item in data.get("papers", [])
            ],
            warnings=[str(v) for v in data.get("warnings", [])],
        )

