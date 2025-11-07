import re, subprocess, tempfile, textwrap
from llm_bridge import answer_with_llama  

def extract_d2_block(response_text: str) -> str:
    match = re.search(r"```d2(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"```(.*?)```", response_text, re.DOTALL)
    return match2.group(1).strip() if match2 else response_text.strip()


def render_d2_to_svg(d2_code: str) -> str:
    """Renders a D2 code string into an SVG file using the D2 CLI."""
    with tempfile.NamedTemporaryFile(suffix=".d2", delete=False, mode="w") as f:
        f.write(d2_code)
        d2_path = f.name
    svg_path = d2_path.replace(".d2", ".svg")

    try:
        subprocess.run(["d2", d2_path, svg_path], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"D2 rendering failed: {e}")

    return svg_path


def llm_generate_d2(context_text: str, user_query: str) -> dict:
    """Uses an LLM (via Groq) to generate valid D2 diagram code from retrieved text and the user's query."""
    prompt = textwrap.dedent(f"""
        You are a specialized Large Language Model that acts as a **Diagram Context Interpreter and D2 Code Generator**
        within a research-oriented Retrieval-Augmented Generation (RAG) system.

        ### Background of the System
        The RAG system you are part of works as follows:
        1. A user asks a question about a research paper (for example: “Show me the architecture” or “Generate the data flow diagram”).
        2. The system retrieves a set of **text chunks (CONTEXT)** from a **vector database** such as Pinecone.
           These chunks are not the exact answer; they are *pieces of relevant text* extracted from the paper 
           that together contain information needed to construct the answer.
        3. Your role is to interpret these retrieved chunks, **infer what the final answer conceptually is**, 
           and then represent that understanding visually in **valid D2 diagram code**.
        4. The diagram you produce will then be rendered directly into an image by the D2 visualization engine.

        ---

        ### Your Objective
        Given:
        - **USER REQUEST** → The original user query (which may ask for an architecture, data flow, schematic, or visualization).
        - **CONTEXT** → The retrieved text chunks from the vector database (which together hold relevant information).

        You must:
        1. **Understand the intent** of the user's request — what kind of visualization they expect (e.g., architecture, data flow, relationships).
        2. **Read and analyze** the CONTEXT. Since these chunks are not the final answer, you must think like a researcher:
           - Mentally combine these pieces of information.
           - Infer the key components, their roles, and how they connect logically.
           - Essentially, imagine what the answer would look like if written out in words.
        3. Once you understand that conceptual answer, **translate it into valid D2 code** that visually represents the relationships, structure, or flow described.

        ---

        ### Important Guidelines
        - Identify **entities** (modules, layers, algorithms, datasets, processes, or components) and their **relationships** (data flow, dependencies, inputs/outputs, communications).
        - Represent entities as **nodes** and their interactions as **directed arrows (->)**.
        - Use clear, descriptive node labels (e.g., “Input Layer”, “Feature Extractor”, “Decoder”).
        - Add arrow labels when needed to show what flows between nodes (e.g., “embeddings”, “data stream”).
        - Maintain logical readability: keep left-to-right or top-down flow consistent.
        - Avoid inventing new technical details — stay faithful to the given CONTEXT.
        - Think carefully about how the *inferred answer* should look, and design the diagram to reflect that idea.
        - Output must be a well-structured D2 diagram — no prose, no explanations.

        ---

        ### Output Format
        Return **only** valid D2 code enclosed in triple backticks. (Do not include any boilerplate or disclaimers. 
Do not start with phrases like "As an AI model", "Sure", or "Here is".
Begin directly with the answer.)
        Do not add commentary or markdown outside the code block.

        Example output format:
        ```d2
        Input -> Encoder: extracts features
        Encoder -> Decoder: reconstructs data
        Decoder -> Output: produces result
        ```

        USER REQUEST:
        {user_query}

        CONTEXT:
        {context_text}
    """)
    raw_response = answer_with_llama("", prompt, model="llama-3.1-8b-instant")
    d2_code = extract_d2_block(raw_response)
    return { "raw_response": raw_response,
        "d2_code": d2_code
    }