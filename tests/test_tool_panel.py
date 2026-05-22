import json
import unittest

from hyperagent.runtime.tool_panel import render_tool_result
from hyperagent.schemas import AgentToolResult


class ToolPanelTest(unittest.TestCase):
    def test_framework_command_skills_are_summarized(self):
        result = AgentToolResult(
            call_id="call",
            tool_name="framework_command",
            status="ok",
            created_at="now",
            content=json.dumps(
                {
                    "skills": [
                        {
                            "name": "open-design",
                            "description": "long description " * 40,
                            "path": "/tmp/SKILL.md",
                        },
                        {"name": "spectral-critic"},
                    ]
                },
                ensure_ascii=False,
            ),
        )

        rendered = render_tool_result(result)

        self.assertIn("- open-design", rendered)
        self.assertIn("- spectral-critic", rendered)
        self.assertNotIn("long description", rendered)

    def test_framework_command_web_status_is_summarized(self):
        result = AgentToolResult(
            call_id="call",
            tool_name="framework_command",
            status="ok",
            created_at="now",
            content=json.dumps(
                {
                    "providers": {
                        "brave": False,
                        "tavily": False,
                    },
                    "search_configured": False,
                    "fetch_available": True,
                    "policy": {"blocked_schemes": ["file", "data"]},
                },
                ensure_ascii=False,
            ),
        )

        rendered = render_tool_result(result)

        self.assertIn("search_configured: False", rendered)
        self.assertIn("providers: none", rendered)
        self.assertIn("fetch_available: True", rendered)
        self.assertNotIn("blocked_schemes", rendered)


if __name__ == "__main__":
    unittest.main()
