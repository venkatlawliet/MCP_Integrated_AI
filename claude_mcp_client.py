import os
import json
import requests
from typing import Dict, List, Any, Optional
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:5050")
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229")
DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "4096"))
DEFAULT_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "20"))
class ClaudeMCPClient:
    def __init__(
        self,
        api_key: str = CLAUDE_API_KEY,
        model: str = DEFAULT_MODEL,
        enable_tools: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        request_timeout: int = DEFAULT_TIMEOUT,
        server_url: Optional[str] = None,
        progress_callback=None,
        **kwargs,
    ):
        if not api_key:
            raise ValueError("CLAUDE_API_KEY is not set")

        self.api_key = api_key
        self.model = model
        self.enable_tools = enable_tools
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.mcp_server_url = (server_url or MCP_SERVER_URL).rstrip("/")
        self.progress_callback = progress_callback
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        # TOOL DEFINITIONS
        self.tools = [
            {
                "name": "fetch_web_content",
                "description": (
                    "Retrieves info from websites or the web. "
                    "Use for latest news or live data — not research papers."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Topic or website to fetch content about."}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "research_lookup",
                "description": (
                    "Handles queries about research papers (titles, authors, or DOIs). "
                    "Use when user mentions 'research paper', 'study', 'DOI', or 'authors'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "paper_title": {"type": "string", "description": "Title or partial title of the paper."},
                        "question": {"type": "string", "description": "Question about the paper."}
                    },
                    "required": ["paper_title", "question"]
                }
            },
            {
                "name": "direct_answer",
                "description": (
                    "For direct factual questions that can be answered without using external tools."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "Direct answer to user query."}
                    },
                    "required": ["answer"]
                }
            }
        ]
    def _emit(self, msg: str):
        if self.progress_callback:
            try:
                self.progress_callback(msg)
            except Exception:
                pass
    def _call_claude(self, message: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send message to Claude and receive response."""
        self._emit("Query sent to Claude")
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": (
                "You are an orchestrator that can call tools ONLY by emitting structured tool_use JSON blocks.\n"
                "NEVER pretend to use a tool or describe tool behavior in text.\n\n"
                "You must definitely use tools based on the user's question.\n"
                "If the question is about a research paper (title, authors, DOI), you MUST use the research_lookup tool.\n"
                "Available tools:\n"
                "1️. direct_answer — for general knowledge.\n"
                "2️. fetch_web_content — for web queries and live info.\n"
                "3️. research_lookup — for research paper queries.\n\n"
                "RULES:\n"
                "- If user mentions 'research paper', 'paper:', 'DOI', or 'authors', "
                "you MUST call research_lookup using this JSON:\n"
                "{'paper_title': '<title>', 'question': '<user question>'}\n"
                "- Do not answer from your own knowledge before calling a tool.\n"
                "- Never say 'the tool returned...' unless you actually used one."
            ),
            "messages": (history or []) + [{"role": "user", "content": message}],
        }
        if self.enable_tools:
            payload["tools"] = self.tools
            payload["tool_choice"] = {"type": "auto"}

        resp = requests.post(
            CLAUDE_API_URL,
            headers=self.headers,
            json=payload,
            timeout=self.request_timeout,
        )

        if resp.status_code != 200:
            print("Claude API error:", resp.text)
        resp.raise_for_status()

        result = resp.json()
        self._emit("Response received from Claude")
        return result
    def send_message(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        history = conversation_history or []
        try:
            result = self._call_claude(message, history)

            print("\n=== RAW CLAUDE RESPONSE ===")
            print(json.dumps(result, indent=2))
            print("============================\n")
            for block in result.get("content", []):
                if block.get("type") == "tool_use":
                    tool_call = {
                        "name": block.get("name", ""),
                        "parameters": block.get("input", {}),
                    }
                    print(f"Tool call detected: {tool_call}")
                    result["__tool_name"] = tool_call["name"]
                    result["__tool_payload"] = tool_call["parameters"]
                    return result  
            self._emit("Claude answered directly (no tool).")
            return result

        except Exception as e:
            self._emit("Error while communicating with Claude")
            print("send_message error:", e)
            return {"error": str(e)}
    def get_final_answer(self, message: str) -> str:
        response = self.send_message(message)
        texts = [b.get("text", "") for b in response.get("content", []) if b.get("type") == "text"]
        return "\n".join(t for t in (s.strip() for s in texts) if t) or "No clear answer found."
ClaudeClient = ClaudeMCPClient