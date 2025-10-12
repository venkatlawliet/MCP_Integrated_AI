import os
import json
import time
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
        show_tool_output: bool = False,
        tool_only: bool = False,
        progress_callback=None,  
        **kwargs, 
    ):
        if not api_key:
            raise ValueError("CLAUDE_API_KEY is not set")
        self.progress_callback = progress_callback 
        self.api_key = api_key
        self.model = model
        self.enable_tools = enable_tools
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.mcp_server_url = (server_url or MCP_SERVER_URL).rstrip("/")
        self.show_tool_output = show_tool_output
        self.tool_only = tool_only
        self.last_tool_response = None
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self.tools = [{
            "name": "fetch_web_content",
            "description": "Retrieves info from websites based on user queries.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search topic or website to look up."
                    }
                },
                "required": ["query"]
            }
        }]
        if self.enable_tools and not self._check_mcp_server():
            print(f"MCP server not reachable at {self.mcp_server_url}/health — disabling tool usage")
            self.enable_tools = False
    def _check_mcp_server(self) -> bool:
        try:
            r = requests.get(f"{self.mcp_server_url}/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False
    def _emit(self, msg: str):
        if self.progress_callback:
            try:
                self.progress_callback(msg)
            except Exception:
                pass     
    def _call_claude(self, message: str, history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        self._emit("Query sent to Claude")  
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": (history or []) + [{"role": "user", "content": message}],
            "system": (
                "If the user asks for web-sourced, current, or citable information, "
                "call the tool 'fetch_web_content' first with a concise query, then answer. "
            ),
        }
        if self.enable_tools:
            payload["tools"] = self.tools
        self._emit(f"Using model: {self.model}")
        resp = requests.post(
            CLAUDE_API_URL,
            headers=self.headers,
            json=payload,
            timeout=self.request_timeout,
        )
        if resp.status_code != 200:
            print("Claude API error:", resp.text)
        self._emit("Response received from Claude")
        resp.raise_for_status()
        return resp.json()
    def _handle_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        self._emit("Claude wanted to use tool")      
        self._emit("Query sent to DuckDuckGo") 
        tool_name = tool_call.get("name")
        tool_params = tool_call.get("parameters", {})
        if not self._check_mcp_server():
            return {"error": "MCP server not reachable"}
        retries = 3
        for attempt in range(retries):
            try:
                r = requests.post(
                    f"{self.mcp_server_url}/tool_call",
                    json={"name": tool_name, "parameters": tool_params},
                    timeout=10,
                )
                r.raise_for_status()
                self._emit("Results received from DuckDuckGo")
                self.last_tool_response = r.json()
                return self.last_tool_response
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"tool_call retry in {wait}s (attempt {attempt + 1}/{retries - 1})")
                    time.sleep(wait)
                else:
                    body = ""
                    try:
                        body = f" body={getattr(e, 'response').text}"
                    except Exception:
                        pass
                    self._emit("Error during DuckDuckGo request")
                    return {"error": f"MCP server not responding: {e}{body}"}
    def _format_tool_results(self, tool_response: Dict[str, Any], k: int = 3) -> str:
        results = tool_response.get("results", []) or []
        if not isinstance(results, list) or not results:
            return "No results."
        lines = []
        for r in results[:k]:
            title = r.get("title", "") or "(no title)"
            url = r.get("url", "") or ""
            desc = r.get("description", "") or ""
            lines.append(f"- {title} — {url}\n  {desc}")
        return "\n".join(lines)
    def send_message(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        stage: str = "initial",
    ) -> Dict[str, Any]:
        history = conversation_history or []
        try:
            result = self._call_claude(message, history)
            if not self.enable_tools:
                if "content" not in result:
                    print(f"({stage}) Claude response had no 'content'.")
                return result
            used_tool = False
            for block in result.get("content", []):
                if block.get("type") == "tool_use":
                    used_tool = True
                    tool_call = {
                        "name": block.get("name", ""),
                        "parameters": block.get("input", {}),
                    }
                    print(f"({stage}) Tool call detected: {tool_call}")
                    tool_response = self._handle_tool_call(tool_call)
                    self._emit("Results sent back to Claude")
                    if self.show_tool_output:
                        print("\n====== RAW TOOL OUTPUT ======")
                        try:
                            print(json.dumps(tool_response, indent=2)[:8000])
                        except Exception:
                            print(tool_response)
                        print("====== END TOOL OUTPUT ======\n")
                    if "error" in tool_response:
                        print("Tool error; asking Claude to continue without tools:", tool_response["error"])
                        return self._call_claude(
                            "Please answer without using any tools.",
                            history + [{"role": "user", "content": message}],
                        )
                    formatted = self._format_tool_results(tool_response, k=3)
                    if self.tool_only:
                        return {"content": [{"type": "text", "text": "🛠️ Tool results:\n" + formatted}]}
                    next_history = history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": ("🛠️ Tool Output:\n" + formatted).strip()}
                        ]}
                    ]
                    return self.send_message(
                        "Please summarize the information above and do not send any more tool calls.",
                        next_history,
                        stage="summary",
                    )
            if not used_tool:
                print(f"({stage}) Claude did not request a tool.")
                self._emit("Claude answered directly (no tools)") 
            return result
        except Exception as e:
            self._emit("Error while talking to Claude")
            print("send_message error:", e)
            return {"error": str(e)}
    def get_final_answer(self, message: str) -> str:
        response = self.send_message(message)
        self._emit("Final answer received")
        texts = [b.get("text", "") for b in response.get("content", []) if b.get("type") == "text"]
        return "\n".join(t for t in (s.strip() for s in texts) if t) or "No clear answer found."
ClaudeClient = ClaudeMCPClient