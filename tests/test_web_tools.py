import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperagent.cli import main
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.feature_state import IDEContextStore, PlanModeStore, web_status
from hyperagent.runtime.web_tools import validate_public_url, web_fetch, web_search
from hyperagent.runtime.workspace import HyperAgentWorkspace


class _FakeHTTPResponse:
    def __init__(self, body, url="https://example.com/page", content_type="text/html; charset=utf-8"):
        self._body = body.encode("utf-8")
        self._url = url
        self.headers = {"content-type": content_type}
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, max_bytes=-1):
        return self._body if max_bytes < 0 else self._body[:max_bytes]

    def geturl(self):
        return self._url

    def getcode(self):
        return self.status


class ControlledWebToolsTest(unittest.TestCase):
    def test_validate_public_url_blocks_private_and_non_http(self):
        self.assertEqual(validate_public_url("https://example.com/a"), "https://example.com/a")
        for url in [
            "file:///tmp/a",
            "data:text/plain,hi",
            "javascript:alert(1)",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://10.0.0.1",
        ]:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_public_url(url)

    def test_web_fetch_extracts_html_text_and_metadata(self):
        html = "<html><head><title>Demo</title><script>x</script></head><body><h1>Title</h1><p>Hello web.</p></body></html>"
        with patch(
            "hyperagent.runtime.web_tools.request.urlopen",
            return_value=_FakeHTTPResponse(html),
        ):
            payload = web_fetch("https://example.com/page", max_chars=200)

        self.assertEqual(payload.status, "ok")
        self.assertEqual(payload.title, "Demo")
        self.assertIn("Title", payload.text)
        self.assertIn("Hello web.", payload.text)
        self.assertTrue(payload.citation_id.startswith("web:"))

    def test_web_search_reports_missing_provider_without_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                web_search("hyperspectral image classification")

    def test_web_search_brave_normalizes_results(self):
        data = {
            "web": {
                "results": [
                    {
                        "title": "Paper",
                        "url": "https://example.com/paper",
                        "description": "snippet",
                    }
                ]
            }
        }
        fake = _FakeHTTPResponse(json.dumps(data), content_type="application/json")
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "key"}, clear=True), patch(
            "hyperagent.runtime.web_tools.request.urlopen",
            return_value=fake,
        ):
            payload = web_search("hsi", max_results=1)

        self.assertEqual(payload.provider, "brave")
        self.assertEqual(payload.results[0].url, "https://example.com/paper")

    def test_executor_web_fetch_uses_session_permission_and_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            approvals = []
            executor = SafeAgentToolExecutor(
                root,
                workspace.workspace_dir,
                permission_policy="session-ask",
                permission_callback=lambda request: approvals.append(request) or True,
                session_permission_cache={},
            )
            html = "<title>Demo</title><p>content</p>"
            with patch(
                "hyperagent.runtime.web_tools.request.urlopen",
                return_value=_FakeHTTPResponse(html),
            ):
                result = executor.web_fetch("https://example.com/page")

            self.assertEqual(result.status, "ok")
            self.assertEqual(len(approvals), 1)
            self.assertIn("citation:", "\n".join(result.warnings))
            self.assertTrue(Path(result.artifact_path).exists())
            self.assertTrue((workspace.workspace_dir / "web_runs").exists())

    def test_plan_mode_and_ide_context_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = HyperAgentWorkspace(root)
            workspace.init(root / "datasets")
            plan = PlanModeStore(workspace.workspace_dir).set_enabled(True, "inspect only")
            ide = IDEContextStore(workspace.workspace_dir).set_open_files(["hyperagent/cli.py"])

            self.assertTrue(plan["enabled"])
            self.assertEqual(ide["open_files"], ["hyperagent/cli.py"])

    def test_cli_web_status_and_plan_mode(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chdir(root)
            try:
                self.assertEqual(main(["init", "--dataset-root", str(root / "datasets")]), 0)
                self.assertEqual(main(["web", "status", "--json"]), 0)
                self.assertEqual(main(["plan-mode", "on", "test"]), 0)
                self.assertEqual(main(["ide-context", "set-open-files", "hyperagent/cli.py"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_web_status_has_fetch_available_without_search_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            payload = web_status()
        self.assertTrue(payload["fetch_available"])
        self.assertFalse(payload["search_configured"])


if __name__ == "__main__":
    unittest.main()
