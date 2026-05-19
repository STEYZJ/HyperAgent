"""arXiv literature provider."""

from datetime import datetime
from typing import Any, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from hyperagent.core.registries import literature_provider_registry
from hyperagent.schemas import LiteraturePaper, LiteratureSearchResult


class ArxivProvider:
    name = "arxiv"
    endpoint = "https://export.arxiv.org/api/query"

    def search(
        self,
        query: str,
        max_results: int = 10,
        *,
        year_from: Optional[int] = None,
        sort_by: str = "latest",
    ) -> LiteratureSearchResult:
        search_query = self._build_query(query, year_from)
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate" if sort_by == "latest" else "relevance",
            "sortOrder": "descending",
        }
        url = f"{self.endpoint}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "HyperAgent/0.1"})
        try:
            with urlopen(request, timeout=20) as response:
                payload = response.read()
            return self._parse(query, max_results, payload)
        except Exception as exc:
            return LiteratureSearchResult(
                query=query,
                source=self.name,
                max_results=max_results,
                papers=[],
                warnings=[f"arXiv request failed: {type(exc).__name__}: {exc}"],
            )

    def _build_query(self, query: str, year_from: Optional[int]) -> str:
        escaped = query.replace('"', "")
        base = f'all:"{escaped}"'
        if year_from is None:
            return base
        start = f"{int(year_from)}01010000"
        end = datetime.utcnow().strftime("%Y%m%d%H%M")
        return f"{base} AND submittedDate:[{start} TO {end}]"

    def _parse(self, query: str, max_results: int, payload: bytes) -> LiteratureSearchResult:
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(payload)
        papers: List[LiteraturePaper] = []
        for entry in root.findall("atom:entry", namespace):
            title = self._text(entry, "atom:title", namespace)
            abstract = self._text(entry, "atom:summary", namespace)
            published = self._text(entry, "atom:published", namespace)
            updated = self._text(entry, "atom:updated", namespace)
            url = self._text(entry, "atom:id", namespace)
            authors = [
                self._text(author, "atom:name", namespace)
                for author in entry.findall("atom:author", namespace)
            ]
            year = None
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    year = None
            papers.append(
                LiteraturePaper(
                    title=" ".join(title.split()),
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    url=url,
                    abstract=" ".join(abstract.split()),
                    source=self.name,
                    published=published,
                    updated=updated,
                    keywords=[query],
                )
            )
        return LiteratureSearchResult(
            query=query,
            source=self.name,
            max_results=max_results,
            papers=papers,
        )

    def _text(self, node: ET.Element, path: str, namespace: Any) -> str:
        found = node.find(path, namespace)
        return "" if found is None or found.text is None else found.text.strip()


literature_provider_registry.register(ArxivProvider.name, ArxivProvider(), replace=True)
