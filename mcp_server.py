import os
from flask import Flask, request, jsonify
from mcp_integration import handle_tool_call_from_claude
PORT = int(os.environ.get("PORT", 5050))
app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
app.config["PROPAGATE_EXCEPTIONS"] = True
@app.get("/health")
def health_check():
    return jsonify({"status": "ok"}), 200
@app.get("/")
def index():
    return jsonify({
        "name": "Claude MCP Tool Server",
        "status": "running",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/tool_call", "method": "POST", "description": "Handle Claude tool calls"}
        ]
    }), 200
@app.post("/tool_call")
def tool_call():
    if not request.is_json:
        return jsonify({"error": "invalid request: expected JSON body"}), 400
    body = request.get_json(silent=True) or {}
    print("/tool_call incoming:", body)
    try:
        tool_name = body.get("name")
        params = body.get("parameters", {})
        if tool_name != "fetch_web_content":
            return jsonify({"error": f"unknown tool name '{tool_name}'"}), 400
        if not isinstance(params, dict):
            return jsonify({"error": "parameters must be an object"}), 400
        result = handle_tool_call_from_claude(tool_name, params)
        return jsonify(result), (200 if "error" not in result else 400)
    except Exception as e:
        import traceback, sys
        print("SERVER ERROR in /tool_call\n", traceback.format_exc(), file=sys.stderr)
        return jsonify({"error": "internal_error", "details": str(e)}), 500
if __name__ == "__main__":
    print(f"Starting MCP tool server on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT)