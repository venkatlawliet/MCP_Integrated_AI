from __future__ import annotations
import os, io, re, uuid, tempfile, logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import torch
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from transformers import TapasTokenizer, TapasModel, AutoProcessor, AutoModel
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
import fitz
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@dataclass
class Part:
    text: str
    page: int
    type: str
    caption: Optional[str] = None
    extra: Optional[Dict] = None
text_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
math_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
tapas_model = TapasModel.from_pretrained("google/tapas-base-finetuned-tabfact")
siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
REF_HEADERS_RE = re.compile(
    r"^\s*(references?|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$",
    re.I,
)
INLINE_NUM_CIT_RE = re.compile(r"\[\s*(?:\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*\]")
def crop_equation(pdf_path: str, page_num: int, bbox: Dict[str, float]) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    rect = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
    pix = page.get_pixmap(clip=rect)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img
def clean_text(text: str) -> str:
    text = INLINE_NUM_CIT_RE.sub("", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)
    return text.strip()
def extract_parts(pdf_path: str) -> List[Part]:
    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="basic",
        max_characters=800,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    parts: List[Part] = []
    in_refs = False
    for el in elements:
        text = getattr(el, "text", "") or ""
        category = str(el.category).lower()
        page = (el.metadata.page_number if el.metadata else 1) or 1
        if REF_HEADERS_RE.match(text.strip()):
            in_refs = True
            continue
        if in_refs:
            continue
        if category not in ["table", "image", "equation"]:
            if text.strip():
                parts.append(Part(text=clean_text(text), page=page, type="text"))
        elif category == "table":
            rows = []
            if hasattr(el, "metadata") and el.metadata and "text_as_html" in el.metadata:
                try:
                    df = pd.DataFrame(el.metadata.text_as_html)
                    rows = df.values.tolist()
                except Exception:
                    rows = []
            parts.append(Part(text=text, page=page, type="table", extra={"rows": rows}))
        elif category == "image":
            if el.metadata and el.metadata.get("image_base64"):
                parts.append(
                    Part(
                        text="(image)",
                        page=page,
                        type="image",
                        extra={"image_bytes": el.metadata["image_base64"]},
                    )
                )
        elif category == "equation":
            extra_data = {}
            if el.metadata:
                if "coordinates" in el.metadata:
                    extra_data["bbox"] = el.metadata["coordinates"]
                if "image_base64" in el.metadata:
                    extra_data["image_bytes"] = el.metadata["image_base64"]
            parts.append(Part(text=text, page=page, type="equation", extra=extra_data))
    return parts
def embed_text(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    vecs = text_model.encode(texts, batch_size=32, show_progress_bar=False)
    return [v.tolist() for v in vecs]
def embed_tables(table_parts: List[Part]) -> List[List[float]]:
    vecs = []
    for p in table_parts:
        rows = (p.extra or {}).get("rows") or []
        if rows:
            df = pd.DataFrame(rows)
            inputs = tapas_tokenizer(
                table=df, queries=[""], padding="max_length", return_tensors="pt"
            )
            with torch.no_grad():
                outputs = tapas_model(**inputs)
            pooled = outputs.pooler_output[0].numpy()
            vecs.append(pooled.tolist())
        elif p.text.strip():
            v = text_model.encode([p.text], show_progress_bar=False)[0]
            vecs.append(v.tolist())
    return vecs
def embed_images(image_parts: List[Part]) -> List[List[float]]:
    vecs = []
    for p in image_parts:
        img_bytes = p.extra.get("image_bytes") if p.extra else None
        if not img_bytes:
            continue
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = siglip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = siglip_model(**inputs)
        pooled = outputs.pooler_output[0].numpy()
        vecs.append(pooled.tolist())
    return vecs
def embed_equations(eq_parts: List[Part], pdf_path: Optional[str] = None) -> List[List[float]]:
    vecs = []
    for p in eq_parts:
        img = None
        if p.extra and "image_bytes" in p.extra:
            try:
                img = Image.open(io.BytesIO(p.extra["image_bytes"])).convert("RGB")
            except:
                img = None
        elif p.extra and "bbox" in p.extra and pdf_path:
            try:
                img = crop_equation(pdf_path, p.page, p.extra["bbox"])
            except:
                img = None
        if img is not None:
            inputs = siglip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = siglip_model(**inputs)
            pooled = outputs.pooler_output[0].numpy()
            vecs.append(pooled.tolist())
        elif p.text.strip():
            v = math_model.encode([p.text], batch_size=16, show_progress_bar=False)[0]
            vecs.append(v.tolist())
    return vecs
def embed_parts(parts: List[Part], pdf_path: Optional[str] = None):
    text_vecs, image_vecs = [], []
    text_parts, image_parts = [], []
    for p in parts:
        if p.type in ["text", "table", "equation"]:
            vecs = []
            if p.type == "text":
                vecs = embed_text([p.text])
            elif p.type == "table":
                vecs = embed_tables([p])
            elif p.type == "equation":
                vecs = embed_equations([p], pdf_path=pdf_path)
            text_vecs.extend(vecs)
            text_parts.extend([p] * len(vecs))
        elif p.type == "image":
            vecs = embed_images([p])
            image_vecs.extend(vecs)
            image_parts.extend([p] * len(vecs))
    return (text_parts, text_vecs), (image_parts, image_vecs)
def create_ids(n: int) -> List[str]:
    return [str(uuid.uuid4()) for _ in range(n)]
def create_meta(parts: List[Part]) -> List[Dict[str, Any]]:
    out = []
    for p in parts:
        meta = {
            "page": p.page,
            "type": p.type,
            "caption": p.caption if p.caption is not None else "", 
            "text": p.text
        }
        if p.extra:
            safe = dict(p.extra)
            for k in ("image_bytes", "pixmap", "raw", "image", "data"):
                safe.pop(k, None)
            meta.update(safe)
        out.append(meta)
    return out
def to_pinecone_vectors(ids, vecs, metas):
    return [{"id": ids[i], "values": vecs[i], "metadata": metas[i]} for i in range(len(ids))]
def pinecone_index(index_name: str, dimension: int = 768, metric: str = "dotproduct"):
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
text_index = pinecone_index("my-text-index", dimension=768)
image_index = pinecone_index("my-image-index", dimension=384)
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
def query_both(query: str, bm25, dense_model, top_k=5, alpha=0.6):
    q_dense = dense_model.encode([query])[0].tolist()
    q_sparse = bm25.encode_queries(query)
    sq, dq = weight_by_alpha(q_sparse, q_dense, alpha)
    text_results = text_index.query(
        vector=dq,
        sparse_vector=sq,
        top_k=top_k,
        include_metadata=True
    )
    inputs = siglip_processor(text=[query], return_tensors="pt")
    with torch.no_grad():
        q_img = siglip_model.get_text_features(**inputs)[0].numpy().tolist()
    image_results = image_index.query(vector=q_img, top_k=top_k, include_metadata=True)
    return text_results, image_results
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
        snippet = (meta.get("text") or "").strip()
        mtype = meta.get("type", "text")
        caption = meta.get("caption")
        score = m.get("score", 0.0)
        if mtype == "text":
            lines.append(f"(page {page}, score {score:.3f}) {snippet}")
        elif mtype == "table":
            lines.append(f"(page {page}, Table: {caption}, score {score:.3f}) {snippet}")
        elif mtype == "image":
            lines.append(f"(page {page}, Figure: {caption}, score {score:.3f})")
        elif mtype == "equation":
            lines.append(f"(page {page}, Equation: {caption or ''}, score {score:.3f}) {snippet}")
    return PRIMER + "\n".join(lines)
def build_combined_context(text_results, image_results):
    context = build_llm_context(text_results)
    if image_results.get("matches"):
        context += "\n\n[IMAGE RESULTS]\n"
        context += build_llm_context(image_results)
    return context