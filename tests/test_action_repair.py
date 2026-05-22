import unittest

from hyperagent.runtime.action_repair import ActionRepairPipeline
from hyperagent.schemas import LLMResponse


class ActionRepairPipelineTest(unittest.TestCase):
    def test_extracts_nested_tool_json_after_prose(self):
        response = LLMResponse(
            provider="deepseek",
            model="test",
            content=(
                "I'll check the available skills first.\n\n"
                '{"thought": "check skills", "action": "tool", '
                '"tool_name": "framework_command", '
                '"args": {"command": "skills list", "args": []}}'
            ),
        )

        parsed = ActionRepairPipeline().parse(response)

        self.assertEqual(parsed.action["action"], "tool")
        self.assertEqual(parsed.action["tool_name"], "framework_command")
        self.assertEqual(parsed.action["args"]["command"], "skills list")

    def test_extracts_direct_action_json_after_prose(self):
        response = LLMResponse(
            provider="deepseek",
            model="test",
            content=(
                "Let me inspect whether web tools are configured.\n"
                '{"thought": "inspect web", "action": "framework_command", '
                '"command": "web status"}'
            ),
        )

        parsed = ActionRepairPipeline().parse(response)

        self.assertEqual(parsed.action["action"], "framework_command")
        self.assertEqual(parsed.action["command"], "web status")


if __name__ == "__main__":
    unittest.main()
