"""Controlled web access tools for HyperAgent action loops."""

import hashlib
import html
import ipaddress
import json
import os
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import parse, request
from urllib.error import HTTPError, URLError
from uuid import uuid4

from hyperagent.core.io import write_json
from hyperagent.runtime.workspace import utc_now


BLOCKED_SCHEMES = {"file", "data", "javascript", "ftp", "ssh"}
ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_USER_AGENT = "HyperAgent/0.1 controlled-web-fetch"


@dataclass
class WebSearchItem:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WebToolPayload:
    kind: str
    status: str
    created_at: str
    provider: str = ""
    query: str = ""
    url: str = ""
    final_url: str = ""
    title: str = ""
    text: str = ""
    citation_id: str = ""
    results: List[WebSearchItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["results"] = [item.to_dict() for item in self.results]
        return data


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = True
        if lowered in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if lowered in {"p", "br", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = False
        if lowered in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if lowered in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = html.unescape(data)
        if self._in_title:
            self.title += text
            return
        self.parts.append(text)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [
            re.sub(r"[ \t\r\f\v]+", " ", line).strip()
            for line in raw.splitlines()
        ]
        return "\n".join(line for line in lines if line)


def validate_public_url(url: str) -> str:
    """Validate that URL is a public http(s) URL and return normalized URL."""

    raw = str(url or "").strip()
    if not raw:
        raise ValueError("URL is required")
    parsed = parse.urlparse(raw)
    scheme = parsed.scheme.lower()
    if scheme in BLOCKED_SCHEMES or scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme is not allowed: {parsed.scheme or '(missing)'}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("URL host is required")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("localhost/private hosts are not allowed")
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        ip = None
    if ip is not None and (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError("private or non-public IP addresses are not allowed")
    return raw


def configured_search_provider(provider: str = "auto") -> str:
    requested = str(provider or "auto").lower()
    if requested != "auto":
        return requested
    if os.environ.get("BRAVE_SEARCH_API_KEY"):
        return "brave"
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    if os.environ.get("SERPAPI_API_KEY"):
        return "serpapi"
    if os.environ.get("SEARXNG_BASE_URL"):
        return "searxng"
    return ""


def search_provider_status() -> Dict[str, bool]:
    return {
        "brave": bool(os.environ.get("BRAVE_SEARCH_API_KEY")),
        "tavily": bool(os.environ.get("TAVILY_API_KEY")),
        "serpapi": bool(os.environ.get("SERPAPI_API_KEY")),
        "searxng": bool(os.environ.get("SEARXNG_BASE_URL")),
    }


def web_fetch(
    url: str,
    *,
    max_chars: int = 12000,
    timeout_sec: int = 20,
) -> WebToolPayload:
    normalized = validate_public_url(url)
    req = request.Request(
        normalized,
        headers={
            "User-Agent": os.environ.get("HYPERAGENT_WEB_USER_AGENT", DEFAULT_USER_AGENT),
            "Accept": "text/html, text/plain, application/xhtml+xml;q=0.9,*/*;q=0.1",
        },
    )
    try:
        with request.urlopen(req, timeout=max(int(timeout_sec), 1)) as response:
            raw = response.read(max(int(max_chars) * 4, 4096))
            content_type = response.headers.get("content-type", "")
            final_url = response.geturl()
            status_code = getattr(response, "status", None) or response.getcode()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"network error: {exc.reason}") from exc
    encoding = _encoding_from_content_type(content_type) or "utf-8"
    decoded = raw.decode(encoding, errors="replace")
    if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
        extractor = _HTMLTextExtractor()
        extractor.feed(decoded)
        title = _clean_space(extractor.title)
        text = extractor.text()
    else:
        title = ""
        text = decoded
    text = text[: max(int(max_chars), 1)]
    return WebToolPayload(
        kind="web_fetch",
        status="ok",
        created_at=utc_now(),
        url=normalized,
        final_url=final_url,
        title=title,
        text=text,
        citation_id=citation_id(final_url or normalized),
        metadata={
            "content_type": content_type,
            "status_code": status_code,
            "max_chars": max_chars,
            "text_chars": len(text),
        },
    )


def web_search(
    query: str,
    *,
    provider: str = "auto",
    max_results: int = 5,
    timeout_sec: int = 20,
) -> WebToolPayload:
    text = str(query or "").strip()
    if not text:
        raise ValueError("query is required")
    selected = configured_search_provider(provider)
    if not selected:
        raise RuntimeError(
            "No web search provider configured. Set one of "
            "BRAVE_SEARCH_API_KEY, TAVILY_API_KEY, SERPAPI_API_KEY, or SEARXNG_BASE_URL."
        )
    max_results = max(1, min(int(max_results), 20))
    if selected == "brave":
        results = _search_brave(text, max_results, timeout_sec)
    elif selected == "tavily":
        results = _search_tavily(text, max_results, timeout_sec)
    elif selected == "serpapi":
        results = _search_serpapi(text, max_results, timeout_sec)
    elif selected == "searxng":
        results = _search_searxng(text, max_results, timeout_sec)
    else:
        raise RuntimeError(f"Unsupported web search provider: {selected}")
    return WebToolPayload(
        kind="web_search",
        status="ok",
        created_at=utc_now(),
        provider=selected,
        query=text,
        results=results,
        metadata={"max_results": max_results},
    )


def write_web_artifact(
    workspace_dir: Path,
    payload: WebToolPayload,
    *,
    run_id: Optional[str] = None,
) -> Path:
    run = run_id or _new_run_id()
    root = Path(workspace_dir) / "web_runs" / run
    root.mkdir(parents=True, exist_ok=True)
    safe_name = payload.kind
    if payload.query:
        safe_name += "-" + hashlib.sha1(payload.query.encode("utf-8")).hexdigest()[:8]
    elif payload.url:
        safe_name += "-" + hashlib.sha1(payload.url.encode("utf-8")).hexdigest()[:8]
    return write_json(root / f"{safe_name}.json", payload)


def recent_citations(workspace_dir: Path, citation_id_filter: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    root = Path(workspace_dir) / "web_runs"
    if not root.exists():
        return []
    items: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        citations = []
        if data.get("citation_id"):
            citations.append(
                {
                    "citation_id": data.get("citation_id"),
                    "title": data.get("title", ""),
                    "url": data.get("final_url") or data.get("url", ""),
                    "artifact_path": str(path),
                }
            )
        for result in data.get("results", []) or []:
            if isinstance(result, dict) and result.get("url"):
                citations.append(
                    {
                        "citation_id": citation_id(result["url"]),
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "artifact_path": str(path),
                    }
                )
        for item in citations:
            if citation_id_filter and item["citation_id"] != citation_id_filter:
                continue
            items.append(item)
            if len(items) >= limit:
                return items
    return items


def citation_id(url: str) -> str:
    return "web:" + hashlib.sha1(str(url).encode("utf-8")).hexdigest()[:10]


def _search_brave(query: str, max_results: int, timeout_sec: int) -> List[WebSearchItem]:
    key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not configured")
    params = parse.urlencode({"q": query, "count": max_results})
    data = _fetch_json(
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        timeout_sec,
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
    )
    raw_results = ((data.get("web") or {}).get("results") or [])[:max_results]
    return [
        WebSearchItem(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("description", "")),
            source="brave",
            rank=index,
        )
        for index, item in enumerate(raw_results, start=1)
        if isinstance(item, dict) and item.get("url")
    ]


def _search_tavily(query: str, max_results: int, timeout_sec: int) -> List[WebSearchItem]:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    body = json.dumps(
        {"api_key": key, "query": query, "max_results": max_results},
        ensure_ascii=False,
    ).encode("utf-8")
    data = _fetch_json(
        "https://api.tavily.com/search",
        timeout_sec,
        method="POST",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    raw_results = data.get("results") or []
    return [
        WebSearchItem(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("content", item.get("snippet", ""))),
            source="tavily",
            rank=index,
        )
        for index, item in enumerate(raw_results[:max_results], start=1)
        if isinstance(item, dict) and item.get("url")
    ]


def _search_serpapi(query: str, max_results: int, timeout_sec: int) -> List[WebSearchItem]:
    key = os.environ.get("SERPAPI_API_KEY", "")
    if not key:
        raise RuntimeError("SERPAPI_API_KEY is not configured")
    params = parse.urlencode({"engine": "google", "q": query, "api_key": key, "num": max_results})
    data = _fetch_json(f"https://serpapi.com/search.json?{params}", timeout_sec)
    raw_results = data.get("organic_results") or []
    return [
        WebSearchItem(
            title=str(item.get("title", "")),
            url=str(item.get("link", "")),
            snippet=str(item.get("snippet", "")),
            source="serpapi",
            rank=index,
        )
        for index, item in enumerate(raw_results[:max_results], start=1)
        if isinstance(item, dict) and item.get("link")
    ]


def _search_searxng(query: str, max_results: int, timeout_sec: int) -> List[WebSearchItem]:
    base = os.environ.get("SEARXNG_BASE_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("SEARXNG_BASE_URL is not configured")
    validate_public_url(base)
    params = parse.urlencode({"q": query, "format": "json"})
    data = _fetch_json(f"{base}/search?{params}", timeout_sec)
    raw_results = data.get("results") or []
    return [
        WebSearchItem(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("content", "")),
            source="searxng",
            rank=index,
        )
        for index, item in enumerate(raw_results[:max_results], start=1)
        if isinstance(item, dict) and item.get("url")
    ]


def _fetch_json(
    url: str,
    timeout_sec: int,
    *,
    method: str = "GET",
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    validate_public_url(url)
    req = request.Request(
        url,
        data=body,
        method=method,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
            **dict(headers or {}),
        },
    )
    try:
        with request.urlopen(req, timeout=max(int(timeout_sec), 1)) as response:
            data = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"network error: {exc.reason}") from exc
    parsed = json.loads(data.decode("utf-8", errors="replace"))
    if not isinstance(parsed, dict):
        raise RuntimeError("search provider returned non-object JSON")
    return parsed


def _encoding_from_content_type(content_type: str) -> str:
    match = re.search(r"charset=([^;\s]+)", content_type or "", flags=re.IGNORECASE)
    return match.group(1).strip("\"'") if match else ""


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _new_run_id() -> str:
    return f"{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}"
