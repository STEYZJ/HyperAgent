"""Minimal stdio MCP adapter for HyperAgent research-experience tools."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.workspace import HyperAgentWorkspace


TOOL_NAMES = [
    "research_pattern_search",
    "experiment_strategy_search",
    "storytelling_search",
    "research_taste_search",
    "paper_strategy_compare",
    "research_experience_consolidate",
]


def run_research_mcp_server(
    project_root: Optional[Path] = None,
    input_stream: Optional[TextIO] = None,
    output_stream: Optional[TextIO] = None,
) -> int:
    workspace = HyperAgentWorkspace(project_root or Path.cwd())
    executor = SafeAgentToolExecutor(
        workspace.project_root,
        workspace.workspace_dir,
        permission_policy="auto",
    )
    stdin = input_stream or sys.stdin
    stdout = output_stream or sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            response = handle_mcp_message(message, executor)
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(exc)}}
        if response is None:
            continue
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()
    return 0


def handle_mcp_message(message: Dict[str, Any], executor: SafeAgentToolExecutor) -> Optional[Dict[str, Any]]:
    method = str(message.get("method", ""))
    request_id = message.get("id")
    if request_id is None and method.startswith("notifications/"):
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hyperagent-research-experience", "version": "0.1.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": mcp_tools()}}
    if method == "tools/call":
        params = dict(message.get("params", {}))
        name = str(params.get("name", ""))
        arguments = dict(params.get("arguments", {}))
        result = call_tool(name, arguments, executor)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": result.content}],
                "isError": result.status != "ok",
            },
        }
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "unknown method: %s" % method}}


def call_tool(name: str, arguments: Dict[str, Any], executor: SafeAgentToolExecutor):
    if name == "research_pattern_search":
        return executor.research_pattern_search(**arguments)
    if name == "experiment_strategy_search":
        return executor.experiment_strategy_search(**arguments)
    if name == "storytelling_search":
        return executor.storytelling_search(**arguments)
    if name == "research_taste_search":
        return executor.research_taste_search(**arguments)
    if name == "paper_strategy_compare":
        return executor.paper_strategy_compare(**arguments)
    if name == "research_experience_consolidate":
        return executor.research_experience_consolidate(**arguments)
    return executor.framework_command("research strategy status")


def mcp_tools() -> List[Dict[str, Any]]:
    query_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "field": {"type": "string"},
            "top_k": {"type": "integer", "default": 8},
        },
        "required": ["query"],
    }
    return [
        {"name": "research_pattern_search", "description": "Search novelty/problem/gap/contribution/reviewer strategy lessons.", "inputSchema": query_schema},
        {"name": "experiment_strategy_search", "description": "Search baseline, ablation, control-variable, robustness, and visualization strategy lessons.", "inputSchema": query_schema},
        {"name": "storytelling_search", "description": "Search scientific storytelling and reviewer-persuasion lessons.", "inputSchema": query_schema},
        {"name": "research_taste_search", "description": "Search cross-paper research taste lessons.", "inputSchema": query_schema},
        {
            "name": "paper_strategy_compare",
            "description": "Compare strategy patterns across papers.",
            "inputSchema": {
                "type": "object",
                "properties": {"papers": {"type": "array", "items": {"type": "string"}}, "field": {"type": "string"}},
                "required": ["papers"],
            },
        },
        {
            "name": "research_experience_consolidate",
            "description": "Consolidate paper strategy cards into long-term research-experience memory.",
            "inputSchema": {
                "type": "object",
                "properties": {"topic": {"type": "string"}, "papers": {"type": "array", "items": {"type": "string"}}, "field": {"type": "string"}},
                "required": ["topic"],
            },
        },
    ]
