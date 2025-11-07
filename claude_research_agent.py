import os, json
from groq import Groq

MODEL = "llama-3.1-8b-instant"  

class ClaudeResearchAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=api_key)

    def decide_and_rewrite(self, user_query: str) -> dict:
        """
        Use LLaMA via Groq to decide whether query is text-based or generation-based,
        and rephrase accordingly.  Returns a dict with intent, rephrased_query, and original_query.
        """
        system_prompt = """
        You are **Research llm**, a specialized sub-agent operating inside a research Q&A pipeline that uses a
        Retrieval-Augmented Generation (RAG) approach.

        Your purpose is to analyze the user's question and decide how the system should handle it.

        The system you are part of works like this:
        - If the question can be answered using text directly from the research paper content which is stored in a vector database, 
          it retrieves relevant chunks from a vector database and lets an LLM answer using those chunks.
        - However, if the question asks to *generate* something (like an image, diagram, flowchart, or architecture) from the research paper,
          it is not possible to retrieve the relevant chunks using that question directly since we can't retrieve the chunks based on that question.
          Instead, it first needs to retrieve descriptive text that explains the requested concept (e.g., "architecture", "data flow"),
          and then it sends both the retrieved context and the original user query to another LLM that generates D2 diagram code
          describing how the diagram should look since the user intially requested for a diagram generation. The D2 code is then used to create and display the requested diagram.

        Therefore, your role is to:
        1️. **Classify the intent** of the user's question as either:
            - "text" → The user wants a normal textual answer that can be retrieved and answered directly.
            - "generate" → The user is asking to create or visualize something (diagram, image, flow, architecture, etc.).
              In this case, RAG alone cannot directly answer the question.

        2️. **Rephrase the question** only if it’s a "generate" query.
            - Rewrite it into a form that can be used to retrieve relevant chunks about the requested concept  first since the another LLM needs to understand the answer to that question(how the architecture or data flow or whatever would look like) for generating D2 code.
            - Example:
                User query: "Generate an image of the data flow used in the paper."
                → intent = "generate"
                → rephrased_query = "Describe the data flow used in the paper."
            - The goal is to make the rephrased version suitable for semantic search and retrieval from the vector database.

        3️. **Preserve the original query** exactly as it is.
            - The original question is important for the next LLM, which uses it (along with retrieved text)
              to generate the actual D2 diagram code.

        4️. **Output Format**
        Return your response strictly as JSON, with no markdown or explanations:
        {
          "intent": "text" or "generate",
          "rephrased_query": "<rewritten query or same>",
          "original_query": "<exact user query>"
        }

        Do not include any text outside of the JSON object.
        Do not include explanations or commentary.
        """
        completion = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=256,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )

        text = completion.choices[0].message.content.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {
                "intent": "text",
                "rephrased_query": user_query,
                "original_query": user_query,
            }
        return parsed