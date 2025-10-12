from __future__ import annotations
from typing import Optional
from claude_mcp_client import ClaudeMCPClient
from groq import Groq
import os
INSTRUCTIONS = (
    "Answer ONLY from the provided CONTEXT. "
    'If the answer is not there, say "I dont know." '
    "Include page/table references when possible."
)
"""def answer_with_claude(context_text: str, question: str,
                       model: str = "claude-3-opus-20240229",
                       max_tokens: int = 1024) -> str:
    client = ClaudeMCPClient(
        model=model,
        enable_tools=False, 
        max_tokens=max_tokens,
        show_tool_output=False,
        tool_only=False,
        progress_callback=None
    )
    message = (
        f"INSTRUCTIONS: {INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n"
    )
    return client.get_final_answer(message)"""
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
        temperature=0.2,
    )

    return chat_completion.choices[0].message.content