from __future__ import annotations
import os, requests
from typing import Optional, Dict, Any
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "you@example.com")
def resolve_pdf_url_from_s2_item(s2_item: Dict[str, Any]) -> Optional[str]:
    oa = (s2_item.get("openAccessPdf") or {}).get("url")
    if oa:
        return oa
    ext = s2_item.get("externalIds") or {}
    arxiv = ext.get("ArXiv") or ext.get("ARXIV") or ext.get("arXiv")
    if arxiv:
        return f"https://arxiv.org/pdf/{arxiv}.pdf"
    doi = ext.get("DOI") or ext.get("doi")
    if doi:
        try:
            r = requests.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": UNPAYWALL_EMAIL},
                timeout=20
            )
            if r.ok:
                best = (r.json().get("best_oa_location") or {})
                pdf = best.get("url_for_pdf")
                if pdf:
                    return pdf
        except Exception:
            pass
    return None