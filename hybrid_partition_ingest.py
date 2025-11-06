from __future__ import annotations
import os, re, uuid, logging, json, tempfile, time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from landingai_ade import LandingAIADE
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = LandingAIADE(apikey=os.environ.get("VISION_AGENT_API_KEY"))
#text_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
@dataclass
class Part:
    text: str
    page: int
    type: str
    caption: Optional[str] = None
    extra: Optional[Dict] = None 
REF_HEADERS_RE = re.compile(
    r"^\s*(references?|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$", re.I,
)
INLINE_NUM_CIT_RE = re.compile(r"\[\s*(?:\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*\]")
def clean_text(text: str) -> str:
    """Removes citation markers, URLs, emails from extracted text."""
    text = INLINE_NUM_CIT_RE.sub("", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)
    return text.strip()
def extract_parts_from_url(pdf_url: str) -> List[Part]:
    try:
        retries = 3
        response = None
        for attempt in range(retries):
            try:
                response = client.parse(document_url=pdf_url, model="dpt-2-latest")
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Retrying ADE parse (attempt {attempt+1}/{retries}) due to error: {e}")
                time.sleep(5)
        ade_json = json.loads(response.model_dump_json())
        content_list = ade_json.get("chunks") or ade_json.get("content") or []
        if not content_list:
            logger.error("ADE returned no chunks. Raw ADE JSON:")
            logger.error(json.dumps(ade_json, indent=2))
            raise ValueError("ADE returned 0 chunks. The PDF might be scanned, empty, or extraction failed.")
        parts: List[Part] = []
        for item in content_list:
            text = item.get("markdown") or item.get("text", "")
            text = text.strip()
            if not text:
                continue
            parts.append(Part(
                text=clean_text(text),
                page=item.get("grounding", {}).get("page", 1),
                type=item.get("type", "text"),
                caption=None,
                extra=item.get("grounding", {})
            ))
        logger.info(f"ADE extracted {len(parts)} structured segments from the PDF (via URL).")
        return parts
    except Exception as e:
        logger.error(f"ADE extraction failed: {e}")
        raise
def create_ids(n: int) -> List[str]:
    return [str(uuid.uuid4()) for _ in range(n)]
def create_meta(parts: List[Part]) -> List[Dict[str, Any]]:
    out = []
    for p in parts:
        grounding_page = None
        if isinstance(p.extra, dict):
            grounding_page = p.extra.get("page")
        meta = {
            "page": p.page, 
            "type": p.type,
            "caption": p.caption if p.caption else "",
            "text": p.text,
            "grounding": grounding_page if grounding_page is not None else p.page
        }
        out.append(meta)
    return out
def to_pinecone_vectors(ids, vecs, metas):
    return [{"id": ids[i], "values": vecs[i], "metadata": metas[i]} for i in range(len(ids))]
def pinecone_index(index_name: str = "ade-unified-index", dimension: int = 768, metric: str = "dotproduct"):
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set.")
    pc = Pinecone(api_key=api_key)
    names = pc.list_indexes().names()
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)
ade_index = pinecone_index()
def sparse_fit_and_encode(text_chunks: List[Document]) -> tuple[BM25Encoder, List[Dict[str, List]]]:
    texts = [c.page_content for c in text_chunks]
    bm25 = BM25Encoder()
    bm25.fit(texts)
    svarr = bm25.encode_documents(texts)
    return bm25, svarr
def weight_by_alpha(sparse_embedding: Dict[str, List], dense_embedding: List[float], alpha: float):
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    hsparse = {
        "indices": sparse_embedding["indices"],
        "values": [v * (1.0 - alpha) for v in sparse_embedding["values"]],
    }
    hdense = [v * alpha for v in dense_embedding]
    return hsparse, hdense
def query_ade_index(query: str, bm25, dense_model, top_k=5, alpha=0.6):
    q_dense = dense_model.encode([query])[0].tolist()
    q_sparse = bm25.encode_queries(query)
    sq, dq = weight_by_alpha(q_sparse, q_dense, alpha)
    results = ade_index.query(
        vector=dq,
        sparse_vector=sq,
        top_k=top_k,
        include_metadata=True
    )
    return results
PRIMER = (
    "You are a Q&A bot. Answer ONLY from the text below. "
    'If the answer is not present, say "I don\'t know." '
    "Cite the page number (and table/figure/equation if mentioned).\n\n"
)
def build_llm_context(matches: Dict[str, Any]) -> str:
    lines = []
    for m in matches.get("matches", []):
        meta = m.get("metadata", {}) or {}
        page = meta.get("page")
        mtype = meta.get("type", "text")
        caption = meta.get("caption", "")
        snippet = (meta.get("text") or "").strip()
        score = m.get("score", 0.0)
        lines.append(
            f"[Page {page} | {mtype.upper()} | {caption}] {snippet}\n(Score: {score:.3f})"
        )
    return PRIMER + "\n".join(lines)