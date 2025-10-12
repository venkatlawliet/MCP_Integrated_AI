from __future__ import annotations
import os, time, requests
from typing import List, Dict, Any
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.getenv("S2_API_KEY")  
_last_call_ts = 0.0
def _respect_rate():
    global _last_call_ts
    now = time.time()
    gap = now - _last_call_ts
    if gap < 1.05:
        time.sleep(1.05 - gap)
    _last_call_ts = time.time()
def _headers() -> Dict[str, str]:
    key = os.getenv("S2_API_KEY")
    if not key:
        raise RuntimeError("S2_API_KEY is missing. Set it in your env.")
    return {"x-api-key": key, "User-Agent": "paper-search/1.0 (+https://semanticscholar.org)"}
def search_papers(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    _respect_rate()
    url = f"{S2_BASE}/paper/search"
    fields = ",".join([
        "title",
        "year",
        "venue",
        "paperId",
        "url",
        "authors",          
        "externalIds",     
        "isOpenAccess",
        "openAccessPdf"     
    ])
    params = {"query": query, "limit": max(1, min(limit, 20)), "fields": fields}
    r = requests.get(url, headers=_headers(), params=params, timeout=25)
    r.raise_for_status()
    return r.json().get("data", []) or []