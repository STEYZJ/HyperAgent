"""Tool facade for literature search."""

from pathlib import Path
from typing import Optional

from hyperagent.core.io import write_json
from hyperagent.core.registries import literature_provider_registry
from hyperagent.schemas import LiteratureSearchResult


class LiteratureSearcher:
    """Search literature through a registered provider."""

    def search(
        self,
        query: str,
        output_path: Optional[Path] = None,
        provider_name: str = "arxiv",
        max_results: int = 10,
        year_from: Optional[int] = None,
        sort_by: str = "latest",
    ) -> LiteratureSearchResult:
        provider = literature_provider_registry.get(provider_name)
        result = provider.search(
            query,
            max_results=max_results,
            year_from=year_from,
            sort_by=sort_by,
        )
        if output_path is not None:
            write_json(output_path, result)
        return result

