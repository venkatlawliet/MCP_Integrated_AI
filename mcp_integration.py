import os
import requests
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

DUCKDUCKGO_API_URL = "https://api.duckduckgo.com"
REQUEST_TIMEOUT = int(os.environ.get("DDG_TIMEOUT", "6"))

@dataclass
class DuckDuckGoQuery:
    q: str
    format: str = "json"
    no_html: int = 1
    skip_disambig: int = 1
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SearchResult:
    title: str
    url: str
    description: str

class WebSearchClient:
    def __init__(self, endpoint: str = DUCKDUCKGO_API_URL, timeout: int = REQUEST_TIMEOUT):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, count: int = 5) -> List[SearchResult]:
        """Perform a DuckDuckGo search and return structured results."""
        payload = DuckDuckGoQuery(q=query).to_dict()
        try:
            r = requests.get(self.endpoint, params=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            results: List[SearchResult] = []
            # Main abstract
            if data.get("Abstract"):
                results.append(SearchResult(
                    title=data.get("Heading", "") or query,
                    url=data.get("AbstractURL", "") or "",
                    description=data.get("Abstract", "") or ""
                ))

            # Regular results
            for item in data.get("Results", []) or []:
                text = item.get("Text") or ""
                url = item.get("FirstURL") or ""
                title = (text.split(" - ", 1)[0] if text else url) or query
                results.append(SearchResult(title=title, url=url, description=text))
                if len(results) >= count:
                    return results[:count]

            # Related topics (nested)
            def _flatten_related(topics):
                for t in topics or []:
                    if "Topics" in t:
                        yield from t.get("Topics", [])
                    else:
                        yield t

            for t in _flatten_related(data.get("RelatedTopics", [])):
                text = t.get("Text") or ""
                url = t.get("FirstURL") or ""
                title = (text.split(" - ", 1)[0] if text else url) or query
                results.append(SearchResult(title=title, url=url, description=text))
                if len(results) >= count:
                    break

            return results[:count]

        except requests.Timeout:
            print("DuckDuckGo search timed out.")
            return []
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            return []


def handle_tool_call_from_claude(tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for Claude's MCP server tool calls."""
    if tool_name == "fetch_web_content":
        query = (tool_params.get("query") or "").strip()
        if not query:
            return {"error": "No query provided."}
        results = WebSearchClient().search(query, count=5)
        return {"results": [asdict(r) for r in results]}
    
    return {"error": f"Unsupported tool: {tool_name}"}
