import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DecouplingTest(unittest.TestCase):
    def _read_tree(self, relative):
        base = ROOT / relative
        text = ""
        for path in sorted(base.rglob("*.py")):
            text += path.read_text(encoding="utf-8")
        return text

    def test_models_do_not_import_agents(self):
        self.assertNotIn("hyperagent.agents", self._read_tree("hyperagent/models"))

    def test_tools_do_not_import_hermes_plugin(self):
        self.assertNotIn("hermes_plugin", self._read_tree("hyperagent/tools"))

    def test_training_does_not_import_cli(self):
        self.assertNotIn("hyperagent.cli", self._read_tree("hyperagent/training"))

    def test_hermes_plugin_is_thin_adapter(self):
        plugin_text = self._read_tree("hermes_plugin")
        self.assertIn("CoordinatorAgent", plugin_text)
        self.assertNotIn("SVC(", plugin_text)
        self.assertNotIn("torch.optim", plugin_text)


if __name__ == "__main__":
    unittest.main()

