from __future__ import annotations
import os
import streamlit as st
import tempfile
from claude_mcp_client import ClaudeMCPClient 
from s2_client import search_papers
from langchain.schema import Document
from content_resolver import resolve_pdf_url_from_s2_item
from hybrid_partition_ingest import (
    extract_parts, embed_parts,
    create_ids, create_meta, to_pinecone_vectors,
    sparse_fit_and_encode, query_both, build_combined_context,
    text_index, image_index 
)
from utils import fetch_pdf_bytes
from groq import Groq
from sentence_transformers import SentenceTransformer
st.markdown(
    """
    <style>
        .stApp {
            background-color: black;
            background-size: cover;
            background-attachment: fixed;
        }
    </style>
    """,
    unsafe_allow_html=True
)
INSTRUCTIONS = (
    "Answer ONLY from the provided CONTEXT. "
    'If the answer is not there, say "I dont know." '
    "Include page/table references when possible."
)
text_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
def answer_with_llama(context_text: str, question: str,
                     model: str = "llama3-8b-8192",
                     max_tokens: int = 1024) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    client = Groq(api_key=api_key)
    message = (
        f"INSTRUCTIONS: {INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n"
    )
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": message}],
        model=model,
        max_tokens=max_tokens,
        temperature=0,
    )
    return chat_completion.choices[0].message.content
st.set_page_config(page_title="ResearchQuesta", page_icon="🪐", layout="wide")
st.title("I am an AI you can ask questions from research papers")
mode = st.radio("Choose mode:", ["General Q&A", "Research oriented Q&A"], horizontal=True)
if mode == "General Q&A":
    query = st.text_area("Enter your question:", placeholder="e.g., Latest news on AI advancements", height=100)
    if st.button("Ask Llama"):
        if not query.strip():
            st.warning("Please enter a question first!")
        else:
            try:
                st.subheader("Answer")
                final = answer_with_llama(query, query, model="llama3-8b-8192", max_tokens=1024)
                st.write(final)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    col1, col2 = st.columns(2)
    with col1:
        s2_query = st.text_area("Find your paper (title/author/venue/DOI):", height=140)
        do_search = st.button("Search on Semantic Scholar")
    st.divider()
    namespace  = "paper-1"
    alpha      = 0.3
    top_k      = 5
    if "s2_results" not in st.session_state:
        st.session_state.s2_results = []
    if "chosen_idx" not in st.session_state:
        st.session_state.chosen_idx = None
    if "pdf_url" not in st.session_state:
        st.session_state.pdf_url = None
    if "bm25" not in st.session_state:
        st.session_state.bm25 = None
    if do_search:
        if not s2_query.strip():
            st.warning("Please enter paper search text.")
        else:
            with st.status("Searching Semantic Scholar...", expanded=True) as s:
                try:
                    items = search_papers(s2_query, limit=3)
                    st.session_state.s2_results = items
                    st.session_state.chosen_idx = None
                    if not items:
                        s.update(label="No results. Try another query.", state="error")
                    else:
                        s.update(label="Found papers. Pick one below.", state="complete")
                except Exception as e:
                    s.update(label="S2 search failed.", state="error")
                    st.error(str(e))
    if st.session_state.s2_results:
        st.subheader("Select a paper below")
        for i, it in enumerate(st.session_state.s2_results[:3], start=1):
            title = it.get("title") or "(no title)"
            year  = it.get("year") or "Unknown"
            venue = it.get("venue") or "Unknown"
            authors = ", ".join([a.get("name") for a in (it.get("authors") or [])][:3]) or "N/A"
            ext = it.get("externalIds") or {}
            doi = ext.get("DOI") or "-"
            arxiv = ext.get("ArXiv") or ext.get("ARXIV") or ext.get("arXiv") or "-"
            oa = (it.get("openAccessPdf") or {}).get("url") or "No Open Access PDF"
            st.markdown(f"### {i}. {title}")
            st.write(f"Authors: {authors}")
            st.write(f"Year & Venue: {year} / {venue}")
            if st.button(f"Select a Paper {i}", key=f"pick_{i}"):
                st.session_state.chosen_idx = i - 1
                st.success(f"Selected: {title}")
        st.divider()
    do_ingest = st.button("Check out this paper")
    if do_ingest:
        if st.session_state.chosen_idx is None:
            st.warning("Please select a paper first.")
        else:
            chosen = st.session_state.s2_results[st.session_state.chosen_idx]
            title  = chosen.get("title") or "paper"
            with st.status("Resolving direct PDF URL...", expanded=True) as s:
                pdf_url = resolve_pdf_url_from_s2_item(chosen)
                if not pdf_url:
                    s.update(label="No OA PDF found. Please upload manually.", state="error")
                else:
                    st.session_state.pdf_url = pdf_url
                    s.update(label=f"Resolved PDF: {pdf_url}", state="complete")
            if st.session_state.pdf_url:
                with st.status("Fetching PDF content and building Pinecone indexes...", expanded=True) as s2:
                    try:
                        pdf_bytes = fetch_pdf_bytes(st.session_state.pdf_url)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(pdf_bytes)
                            tmp_path = tmp.name
                        parts = extract_parts(tmp_path)
                        (text_parts, text_vecs), (image_parts, image_vecs) = embed_parts(parts, pdf_path=tmp_path)
                        text_docs = [Document(page_content=p.text, metadata={"page": p.page}) for p in text_parts]
                        bm25, sparse_vecs = sparse_fit_and_encode(text_docs)
                        st.session_state.bm25 = bm25
                        if text_vecs:
                            ids = create_ids(len(text_vecs))
                            metas = create_meta(text_parts)
                            text_index.upsert(to_pinecone_vectors(ids, text_vecs, metas))
                        if image_vecs:
                            ids = create_ids(len(image_vecs))
                            metas = create_meta(image_parts)
                            image_index.upsert(to_pinecone_vectors(ids, image_vecs, metas))
                        s2.update(label=f"Ingested {len(parts)} parts into Pinecone.", state="complete")
                        st.success("All set. Now ask your question.")
                    except Exception as e:
                        s2.update(label="Ingest failed.", state="error")
                        st.error(str(e))
    if st.session_state.get("bm25"):
        st.markdown("Ask a question about this paper")
        paper_question = st.text_area(
            "Your question:",
            height=140,
            placeholder="e.g., What methodology is used in the experiments?"
        )
        if st.button("Answer from Paper"):
            if not paper_question.strip():
                st.warning("Please type your question about the paper.")
            else:
                with st.status("Retrieving from Pinecone and asking Llama...", expanded=True) as s3:
                    try:
                        text_results, image_results = query_both(
                            paper_question,
                            bm25=st.session_state.bm25,
                            dense_model=text_model,
                            top_k=top_k,
                            alpha=0.3
                        )
                        ctx = build_combined_context(text_results, image_results)
                        with st.expander("Show Retrieved Context", expanded=False):
                            st.markdown("**Retrieved context for language model:**")
                            st.write(ctx)
                        final = answer_with_llama(ctx, paper_question, model="llama-3.1-8b-instant", max_tokens=1024)
                        s3.update(label="Done", state="complete")
                        st.subheader("Answer")
                        st.write(final)
                    except Exception as e:
                        s3.update(label="Error during retrieval/LLM.", state="error")
                        st.error(str(e))
    else:
        st.info("")