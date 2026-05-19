import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperagent.cli import main


class AgentCompatCLITest(unittest.TestCase):
    def test_provider_session_mcp_obsidian_prompt_cli(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chdir(root)
            try:
                self.assertEqual(main(["init", "--dataset-root", str(root / "datasets")]), 0)

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "llm-dry-run",
                                "--provider",
                                "openai",
                                "--user",
                                "hello",
                            ]
                        ),
                        0,
                    )
                self.assertIn("OPENAI_API_KEY", buffer.getvalue())

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(main(["session-new", "--title", "demo"]), 0)
                session_id = buffer.getvalue().strip().splitlines()[-1]
                self.assertEqual(
                    main(["session-add", "--session-id", session_id, "--role", "user", "--content", "message one"]),
                    0,
                )
                self.assertEqual(
                    main(["session-add", "--session-id", session_id, "--role", "assistant", "--content", "message two"]),
                    0,
                )
                self.assertEqual(
                    main(["session-compress", "--session-id", session_id, "--keep-last", "1"]),
                    0,
                )
                self.assertEqual(main(["session-archive", "--session-id", session_id]), 0)
                self.assertEqual(main(["session-delete", "--session-id", session_id]), 0)

                self.assertEqual(
                    main(
                        [
                            "mcp-add",
                            "--name",
                            "demo",
                            "--command",
                            "python",
                            "--arg=-m",
                            "--arg",
                            "demo_server",
                            "--env",
                            "A=B",
                        ]
                    ),
                    0,
                )
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(main(["mcp-export"]), 0)
                self.assertIn("mcpServers", buffer.getvalue())

                vault = root / "vault"
                vault.mkdir()
                (vault / "Note.md").write_text(
                    "# HSI Idea\n\nUse [[Spectral Gate]] for #hsi classification.",
                    encoding="utf-8",
                )
                self.assertEqual(main(["obsidian-index", "--vault", str(vault)]), 0)
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(main(["obsidian-search", "--query", "spectral"]), 0)
                self.assertIn("HSI Idea", buffer.getvalue())

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    self.assertEqual(
                        main(
                            [
                                "prompt-render",
                                "--name",
                                "hsi_research_copilot",
                                "--var",
                                "dataset=Indian Pines",
                                "--var",
                                "objective=OA",
                            ]
                        ),
                        0,
                    )
                self.assertIn("Indian Pines", buffer.getvalue())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
