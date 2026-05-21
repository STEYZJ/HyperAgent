import json
import os
import tempfile
import unittest
from pathlib import Path

from hyperagent.core.io import read_yaml
from hyperagent.runtime.channels import (
    ChannelConfigStore,
    ChannelRouter,
    register_builtin_channel_platforms,
)
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.llm import LLMProviderStore
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.schemas import ChannelInboundMessage

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - dependency installed in the project env.
    TestClient = None


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "hyperagent" / "prompts"


def _feishu_event(text="hello", token="token"):
    return {
        "schema": "2.0",
        "header": {
            "event_type": "im.message.receive_v1",
            "token": token,
            "tenant_key": "tenant",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "user-open-id"}},
            "message": {
                "message_id": "msg-feishu",
                "chat_id": "chat-feishu",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": text}),
            },
        },
    }


def _qq_event(text="hello", token="token"):
    return {
        "type": "GROUP_AT_MESSAGE_CREATE",
        "token": token,
        "d": {
            "id": "msg-qq",
            "content": text,
            "group_openid": "group-qq",
            "author": {"id": "user-qq"},
        },
    }


class ChannelGatewayTest(unittest.TestCase):
    def _workspace(self, root):
        workspace = HyperAgentWorkspace(root)
        workspace.init(root / "datasets")
        conversations = ConversationStore(workspace.workspace_dir)
        providers = LLMProviderStore(workspace.workspace_dir)
        prompts = PromptLibrary([PROMPT_ROOT])
        store = ChannelConfigStore(workspace.workspace_dir)
        return workspace, conversations, providers, prompts, store

    def _router(self, root, responder=None):
        workspace, conversations, providers, prompts, store = self._workspace(root)
        configs = store.ensure_defaults()
        for config in configs:
            config.dry_run = True
        store.save_all(configs)
        return (
            ChannelRouter(
                workspace,
                conversations,
                providers,
                prompt_library=prompts,
                config_store=store,
                responder=responder or (lambda inbound, session_id, config: "reply: " + inbound.text),
            ),
            store,
            conversations,
        )

    def test_feishu_challenge_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("FEISHU_VERIFICATION_TOKEN")
            os.environ["FEISHU_VERIFICATION_TOKEN"] = "token"
            try:
                router, _store, _conversations = self._router(Path(tmp))
                body = json.dumps({"type": "url_verification", "challenge": "abc", "token": "token"}).encode()
                result = router.handle_webhook(
                    "feishu",
                    json.loads(body.decode()),
                    headers={},
                    body=body,
                )
                self.assertEqual(result.status, "verified")
                self.assertEqual(result.response_payload["challenge"], "abc")
            finally:
                if old is None:
                    os.environ.pop("FEISHU_VERIFICATION_TOKEN", None)
                else:
                    os.environ["FEISHU_VERIFICATION_TOKEN"] = old

    def test_feishu_text_event_routes_to_session_and_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FEISHU_VERIFICATION_TOKEN"] = "token"
            router, _store, conversations = self._router(Path(tmp))
            payload = _feishu_event("高光谱下一步实验？")
            result = router.handle_webhook(
                "feishu",
                payload,
                headers={},
                body=json.dumps(payload).encode("utf-8"),
            )

            self.assertEqual(result.status, "replied")
            self.assertEqual(result.inbound.text, "高光谱下一步实验？")
            self.assertEqual(result.outbound.text, "reply: 高光谱下一步实验？")
            self.assertIsNotNone(result.session_id)
            saved = conversations.load(result.session_id)
            self.assertEqual(saved.metadata["channel_provider"], "feishu")

    def test_qq_text_event_routes_to_session_and_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["QQ_BOT_VERIFICATION_TOKEN"] = "token"
            router, _store, _conversations = self._router(Path(tmp))
            payload = _qq_event("<@123> 诊断实验")
            result = router.handle_webhook(
                "qq",
                payload,
                headers={},
                body=json.dumps(payload).encode("utf-8"),
            )

            self.assertEqual(result.status, "replied")
            self.assertEqual(result.inbound.provider, "qq")
            self.assertEqual(result.inbound.chat_type, "group")
            self.assertEqual(result.inbound.text, "诊断实验")
            self.assertIn("/v2/groups/group-qq/messages", result.outbound.raw_payload["url"])

    def test_non_text_events_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FEISHU_VERIFICATION_TOKEN"] = "token"
            router, _store, _conversations = self._router(Path(tmp))
            payload = _feishu_event("ignored")
            payload["event"]["message"]["message_type"] = "image"
            result = router.handle_webhook(
                "feishu",
                payload,
                headers={},
                body=json.dumps(payload).encode("utf-8"),
            )
            self.assertEqual(result.status, "ignored")

    def test_invalid_verification_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FEISHU_VERIFICATION_TOKEN"] = "expected"
            router, _store, _conversations = self._router(Path(tmp))
            payload = _feishu_event("hello", token="wrong")
            result = router.handle_webhook(
                "feishu",
                payload,
                headers={},
                body=json.dumps(payload).encode("utf-8"),
            )
            self.assertEqual(result.status, "error")
            self.assertIn("token", result.error)

    def test_disabled_provider_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            router, store, _conversations = self._router(Path(tmp))
            configs = store.ensure_defaults()
            for config in configs:
                if config.provider == "qq":
                    config.enabled = False
            store.save_all(configs)
            result = router.handle_webhook(
                "qq",
                _qq_event(),
                headers={},
                body=json.dumps(_qq_event()).encode("utf-8"),
            )
            self.assertEqual(result.status, "error")
            self.assertIn("disabled", result.error)

    def test_session_mapping_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            router, store, _conversations = self._router(Path(tmp))
            config = store.get("feishu")
            inbound = ChannelInboundMessage(
                provider="feishu",
                channel_user_id="same-user",
                chat_id="same-chat",
                message_id="m1",
                text="hello",
            )
            first = router.handle_message(inbound, config)
            second = router.handle_message(inbound, config)
            self.assertEqual(first.session_id, second.session_id)

    def test_config_stores_env_names_not_secret_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["QQ_BOT_TOKEN"] = "secret-token-value"
            store = ChannelConfigStore(Path(tmp) / ".hyperagent")
            store.init_provider("qq")
            text = store.path.read_text(encoding="utf-8")
            data = read_yaml(store.path)
            self.assertIn("QQ_BOT_TOKEN", text)
            self.assertNotIn("secret-token-value", text)
            self.assertEqual(data["channels"][1]["access_token_env"], "QQ_BOT_TOKEN")

    def test_channel_router_does_not_expose_local_tool_loops(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "hyperagent" / "runtime" / "channels" / "router.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("AgentActionLoop", text)
        self.assertNotIn("SafeAgentToolExecutor", text)
        self.assertNotIn("GeneralAgentRunner", text)

    def test_channel_platform_registry_creates_builtin_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChannelConfigStore(Path(tmp) / ".hyperagent")
            registry = register_builtin_channel_platforms()
            providers = [entry.provider for entry in registry.list()]

            self.assertEqual(providers, ["feishu", "qq"])
            config = store.init_provider("feishu")
            adapter = registry.create_adapter(config)
            self.assertEqual(adapter.provider, "feishu")

    def test_unknown_channel_provider_returns_structured_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            router, _store, _conversations = self._router(Path(tmp))
            result = router.handle_webhook(
                "unknown",
                {},
                headers={},
                body=b"{}",
            )

            self.assertEqual(result.status, "error")
            self.assertIn("Unknown channel provider", result.error)

    def test_fastapi_app_registers_expected_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            from hyperagent.runtime.channels.gateway import create_channel_app

            router, store, _conversations = self._router(Path(tmp))
            app = create_channel_app(router, store)
            paths = {route.path for route in app.routes}
            self.assertIn("/health", paths)
            self.assertIn("/channels", paths)
            self.assertIn("/webhooks/feishu", paths)
            self.assertIn("/webhooks/qq", paths)

    @unittest.skipIf(TestClient is None, "FastAPI is not installed")
    @unittest.skipUnless(
        os.environ.get("HYPERAGENT_RUN_FASTAPI_TESTCLIENT") == "1",
        "TestClient hangs in some restricted sandboxes; enable explicitly for local ASGI smoke tests",
    )
    def test_fastapi_health_channels_and_webhooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            from hyperagent.runtime.channels.gateway import create_channel_app

            os.environ["FEISHU_VERIFICATION_TOKEN"] = "token"
            router, store, _conversations = self._router(Path(tmp))
            client = TestClient(create_channel_app(router, store))

            self.assertEqual(client.get("/health").json()["status"], "ok")
            self.assertEqual(client.get("/channels").status_code, 200)
            response = client.post("/webhooks/feishu", json=_feishu_event("hello"))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "replied")
            bad = client.post("/webhooks/feishu", json=_feishu_event("hello", token="bad"))
            self.assertEqual(bad.status_code, 401)


if __name__ == "__main__":
    unittest.main()
