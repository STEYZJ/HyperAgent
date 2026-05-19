"""MCP server configuration registry."""

from pathlib import Path
from typing import Dict, List

from hyperagent.core.io import read_json, write_json
from hyperagent.schemas import MCPServerSpec


class MCPServerStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = workspace_dir / "mcp_servers.json"

    def list(self) -> List[MCPServerSpec]:
        if not self.path.exists():
            return []
        data = read_json(self.path)
        return [MCPServerSpec.from_dict(item) for item in data.get("servers", [])]

    def upsert(self, server: MCPServerSpec) -> None:
        servers: Dict[str, MCPServerSpec] = {item.name: item for item in self.list()}
        servers[server.name] = server
        write_json(self.path, {"servers": [item.to_dict() for item in servers.values()]})

    def export_client_config(self) -> Dict[str, object]:
        return {
            "mcpServers": {
                server.name: {
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                    "disabled": not server.enabled,
                }
                for server in self.list()
            }
        }

