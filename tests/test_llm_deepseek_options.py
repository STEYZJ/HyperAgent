import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hyperagent.cli import main
from hyperagent.runtime.llm import LLMClient, LLMProviderStore, LLMRequestBuilder
from hyperagent.schemas import LLMMessage, LLMProviderSpec


class _FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "final answer",
                            "reasoning_content": "reasoning trace",
                            "tool_calls": [{"id": "call_1", "type": "function"}],
                        }
                    }
                ],
                "usage": {"total_tokens": 17},
            }
        ).encode("utf-8")


def _deepseek_spec() -> LLMProviderSpec:
    return LLMProviderSpec(
        name="deepseek",
        kind="openai_compatible",
        base_url="https://api.deepseek.com/chat/completions",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-flash",
    )


class DeepSeekOptionsTest(unittest.TestCase):
    def test_builder_adds_thinking_json_and_tool_options(self):
        payload = LLMRequestBuilder().build(
            _deepseek_spec(),
            [LLMMessage(role="user", content="return json")],
            model="deepseek-v4-pro",
            temperature=0.7,
            top_p=0.8,
            max_tokens=256,
            response_format={"type": "json_object"},
            thinking={"type": "enabled"},
            reasoning_effort="max",
            extra_body={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "inspect_dataset",
                            "description": "Inspect one HSI dataset.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        body = payload["json"]
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["tool_choice"], "auto")
        self.assertIn("tools", body)
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)

    def test_builder_keeps_sampling_when_thinking_disabled(self):
        payload = LLMRequestBuilder().build(
            _deepseek_spec(),
            [LLMMessage(role="user", content="answer")],
            thinking={"type": "disabled"},
            temperature=0.3,
            top_p=0.9,
        )
        body = payload["json"]
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["top_p"], 0.9)

    def test_client_extracts_reasoning_tool_calls_and_usage(self):
        old_value = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "secret"
        try:
            with patch("hyperagent.runtime.llm.urlopen", return_value=_FakeHTTPResponse()):
                response = LLMClient().send(
                    _deepseek_spec(),
                    [LLMMessage(role="user", content="call a tool")],
                )
        finally:
            if old_value is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_value

        self.assertEqual(response.content, "final answer")
        self.assertEqual(response.reasoning_content, "reasoning trace")
        self.assertEqual(response.tool_calls[0]["id"], "call_1")
        self.assertEqual(response.usage["total_tokens"], 17)

    def test_cli_dry_run_accepts_deepseek_runtime_options(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(Path(tmp))
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "llm-dry-run",
                                "--provider",
                                "deepseek",
                                "--model",
                                "deepseek-v4-pro",
                                "--user",
                                "return json",
                                "--thinking",
                                "enabled",
                                "--reasoning-effort",
                                "max",
                                "--json-output",
                                "--extra-body-json",
                                '{"tool_choice":"auto"}',
                            ]
                        ),
                        0,
                    )
            finally:
                os.chdir(old_cwd)

        payload = json.loads(buffer.getvalue())
        body = payload["json"]
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["tool_choice"], "auto")

    def test_provider_store_migrates_legacy_deepseek_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LLMProviderStore(root)
            store.path.write_text(
                json.dumps(
                    {
                        "providers": [
                            {
                                "name": "deepseek",
                                "kind": "openai_compatible",
                                "base_url": "https://api.deepseek.com/chat/completions",
                                "api_key_env": "DEEPSEEK_API_KEY",
                                "default_model": "deepseek-chat",
                                "headers": {},
                                "metadata": {},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            providers = store.ensure_defaults()
            deepseek = next(provider for provider in providers if provider.name == "deepseek")

        self.assertEqual(deepseek.default_model, "deepseek-v4-flash")
        self.assertTrue(deepseek.metadata["supports_thinking"])
        self.assertIn("deepseek-v4-pro", deepseek.metadata["recommended_models"])


if __name__ == "__main__":
    unittest.main()
