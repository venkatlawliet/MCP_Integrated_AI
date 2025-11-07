from __future__ import annotations
import os
import json
import streamlit as st
from s2_client import search_papers
from langchain.schema import Document
from content_resolver import resolve_pdf_url_from_s2_item
from hybrid_partition_ingest import (
    extract_parts_from_url,
    create_ids, create_meta, to_pinecone_vectors,
    sparse_fit_and_encode, query_ade_index, build_llm_context, ade_index
)
from claude_mcp_client import ClaudeMCPClient
from groq import Groq
from sentence_transformers import SentenceTransformer
from claude_research_agent import ClaudeResearchAgent
from d2_utils import llm_generate_d2, render_d2_to_svg
st.set_page_config(page_title="ResearchMCP", page_icon="🪐", layout="wide")
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"], html, body {
            background-color: #000 !important;
            color: #fff !important;
        }
        [data-testid="stHeader"] {background: #000 !important;}
        h1,h2,h3,h4,h5,h6,label,p,span,div {color: #fff !important;}
        textarea, input[type="text"] {
            background: #111 !important;
            color: #fff !important;
            border: 1px solid #333 !important;
            border-radius: 8px !important;
        }
        .stButton>button {
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 0.6rem 1rem !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .stButton>button:hover {
            background-color: #1e40af !important;
        }
        .block-container {padding-top: 2rem !important;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🪐 ResearchMCP — Unified Chat Interface")
INSTRUCTIONS = (
    "Answer ONLY from the provided CONTEXT. "
    'If the answer is not there, say \"I don’t know.\" '
    "Include page/table references when possible."
)
text_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
def answer_with_llama(context_text: str, question: str,
                      model: str = "llama-3.1-8b-instant",
                      max_tokens: int = 1024) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")
    client = Groq(api_key=api_key)
    message = f"INSTRUCTIONS: {INSTRUCTIONS}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}\n"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": message}],
        model=model,
        max_tokens=max_tokens,
        temperature=0,
    )
    return chat_completion.choices[0].message.content
Main_claude = ClaudeMCPClient(request_timeout=60)
for key in ["bm25", "pdf_url", "s2_results", "chosen_idx",
            "tool_used", "tool_payload", "paper_title",
            "paper_question", "last_response"]:
    if key not in st.session_state:
        st.session_state[key] = None
query = st.text_area("💬 Ask anything (general, web, or research-based):", height=120)
ask_button = st.button("Ask Claude")
if ask_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🤖 Thinking..."):
            try:
                response = Main_claude.send_message(query)
                st.session_state.last_response = response  
                st.success("✅ Claude response received. Scroll down.")
            except Exception as e:
                st.error(f"Error: {e}")
if st.session_state.get("last_response"):
    response = st.session_state.last_response
    content_blocks = response.get("content", [])
    tool_used = response.get("__tool_name")
    tool_payload = response.get("__tool_payload")

    st.markdown("Claude Response Debug")
    st.code(json.dumps(response, indent=2), language="json")
    if not tool_used or tool_used == "direct_answer":
        st.subheader("Answer")
        final = Main_claude.get_final_answer(query)
        st.write(final)
    elif tool_used == "research_lookup":
        st.markdown("Research Paper Lookup Detected")
        st.info("Claude detected that your question relates to a research paper.")
        with st.form("research_lookup_form"):
            paper_title = st.text_input(
                "Enter the title or DOI of the research paper:",
                value=tool_payload.get("paper_title", "")
            )
            paper_question = st.text_area(
                "What is your question about this paper?",
                value=tool_payload.get("question", query),
                height=100
            )
            proceed = st.form_submit_button("Search & Analyze")

        if proceed:
            st.session_state.paper_title = paper_title
            st.session_state.paper_question = paper_question
if st.session_state.get("paper_title") and st.session_state.get("paper_question"):
    with st.status("Searching Semantic Scholar...", expanded=True) as s:
        try:
            items = search_papers(st.session_state.paper_title, limit=3)
            st.session_state.s2_results = items
            if not items:
                s.update(label="No papers found.", state="error")
            else:
                s.update(label="Papers found. Select one below.", state="complete")
        except Exception as e:
            s.update(label="Semantic Scholar search failed.", state="error")
            st.error(str(e))
    if st.session_state.s2_results:
        for i, it in enumerate(st.session_state.s2_results[:3], start=1):
            title = it.get("title", "(no title)")
            year = it.get("year", "Unknown")
            venue = it.get("venue", "Unknown")
            authors = ", ".join([a.get("name") for a in (it.get("authors") or [])][:3]) or "N/A"
            st.markdown(f"#### {i}. {title}")
            st.write(f"Authors: {authors}")
            st.write(f"Year & Venue: {year} / {venue}")
            if st.button(f"Select Paper {i}", key=f"pick_{i}"):
                st.session_state.chosen_idx = i - 1
                st.session_state.paper_ingested = False 
                st.success(f"Selected: {title}")

    if st.session_state.chosen_idx is not None:
        chosen = st.session_state.s2_results[st.session_state.chosen_idx]
        with st.status("Resolving and preparing the paper...", expanded=True) as s:
            pdf_url = resolve_pdf_url_from_s2_item(chosen)
            if not pdf_url:
                s.update(label="No Open Access PDF found.", state="error")
            else:
                st.session_state.pdf_url = pdf_url
                s.update(label=f"Resolved PDF: {pdf_url}", state="complete")
if st.session_state.pdf_url and not st.session_state.get("paper_ingested", False):
    with st.status("Extracting, embedding, and uploading to Pinecone...", expanded=True) as s2:
        try:
            parts = extract_parts_from_url(st.session_state.pdf_url)
            parts = [p for p in parts if p.text.strip()]
            texts = [p.text for p in parts]
            vecs = text_model.encode(texts, batch_size=32, show_progress_bar=True)
            ids = create_ids(len(vecs))
            metas = create_meta(parts)
            ade_index.upsert(to_pinecone_vectors(ids, vecs, metas))
            docs = [Document(page_content=p.text, metadata={"page": p.page}) for p in parts]
            bm25, _ = sparse_fit_and_encode(docs)
            st.session_state.bm25 = bm25
            st.session_state.paper_ingested = True
            s2.update(label="Paper ingested successfully!", state="complete")
            st.success("You can now ask detailed questions about this paper below.")
        except Exception as e:
            s2.update(label="Error during embedding.", state="error")
            st.error(str(e))
if st.session_state.get("paper_ingested", False):
    st.markdown("Ask about the ingested paper")
    paper_q = st.text_area("Ask about this paper:", height=120)
    if st.button("Get Answer from Paper"):
        st.session_state.original_query = paper_q
        research_agent = ClaudeResearchAgent()
        decision = research_agent.decide_and_rewrite(paper_q)
        with st.expander("Rewritten or parsed query", expanded=False):
                    st.text_area("query to be passed to Pinecone", value=decision["rephrased_query"], height=300)
        intent = decision["intent"]
        rewritten_q = decision["rephrased_query"]
        original_q = decision["original_query"]
        if intent == "generate":
            with st.status("Retrieving context for diagram...", expanded=True) as s:
                results = query_ade_index(
                    rewritten_q,
                    bm25=st.session_state.bm25,
                    dense_model=text_model,
                    top_k=5,
                    alpha=0.3,
                )
                ctx = build_llm_context(results)
                result = llm_generate_d2(ctx, st.session_state.original_query)
                raw_response = result["raw_response"]
                d2_code = result["d2_code"]
                svg_path = render_d2_to_svg(d2_code)
                with st.expander("View LLM Generated Response (Raw OutputD2 code)", expanded=False):
                    st.text_area("Raw LLM Output", value=raw_response, height=300)
                with st.expander("View Extracted D2 Code (Filtered for Rendering)", expanded=False):
                    st.code(d2_code, language="d2")
                s.update(label="Diagram ready!", state="complete")
                st.image(svg_path, caption="Generated diagram", use_column_width=False, width=800)
        else:
            with st.status("Retrieving and querying LLM...", expanded=True) as s:
                results = query_ade_index(
                    rewritten_q,
                    bm25=st.session_state.bm25,
                    dense_model=text_model,
                    top_k=5,
                    alpha=0.3,
                )
                ctx = build_llm_context(results)
                final = answer_with_llama(ctx, original_q)
                s.update(label="Done", state="complete")
                st.subheader("Answer")
                st.write(final)