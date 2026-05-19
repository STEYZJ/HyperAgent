"""Semantic Scholar literature provider."""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hyperagent.core.registries import literature_provider_registry
from hyperagent.schemas import LiteraturePaper, LiteratureSearchResult


class SemanticScholarProvider:
    name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(
        self,
        query: str,
        max_results: int = 10,
        *,
        year_from: Optional[int] = None,
        sort_by: str = "latest",
    ) -> LiteratureSearchResult:
        fields = "title,authors,year,venue,url,abstract,publicationDate,citationCount"
        params: Dict[str, Any] = {
            "query": query,
            "limit": max_results,
            "fields": fields,
        }
        if year_from is not None:
            params["year"] = f"{int(year_from)}-"
        if sort_by == "latest":
            params["sort"] = "publicationDate:desc"
        request = Request(
            f"{self.endpoint}?{urlencode(params)}",
            headers={"User-Agent": "HyperAgent/0.1"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return LiteratureSearchResult(
                query=query,
                source=self.name,
                max_results=max_results,
                papers=[],
                warnings=[
                    f"Semantic Scholar request failed: {type(exc).__name__}: {exc}"
                ],
            )
        papers: List[LiteraturePaper] = []
        for item in payload.get("data", []):
            authors = [author.get("name", "") for author in item.get("authors", [])]
            papers.append(
                LiteraturePaper(
                    title=str(item.get("title", "")),
                    authors=[name for name in authors if name],
                    year=item.get("year"),
                    venue=item.get("venue"),
                    url=str(item.get("url", "")),
                    abstract=str(item.get("abstract") or ""),
                    source=self.name,
                    published=item.get("publicationDate"),
                    keywords=[query],
                    metadata={"citation_count": item.get("citationCount")},
                )
            )
        return LiteratureSearchResult(
            query=query,
            source=self.name,
            max_results=max_results,
            papers=papers,
        )


literature_provider_registry.register(
    SemanticScholarProvider.name, SemanticScholarProvider(), replace=True
)
