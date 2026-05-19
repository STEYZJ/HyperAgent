import tempfile
import unittest
from pathlib import Path

from hyperagent.agents import CoordinatorAgent
from hyperagent.core.registries import literature_provider_registry
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.schemas import LiteraturePaper, LiteratureSearchResult


class FakeLiteratureProvider:
    name = "fake"

    def search(self, query, max_results=10, *, year_from=None, sort_by="latest"):
        del year_from, sort_by
        return LiteratureSearchResult(
            query=query,
            source=self.name,
            max_results=max_results,
            papers=[
                LiteraturePaper(
                    title="Spectral gating for hyperspectral image classification",
                    authors=["A. Researcher"],
                    year=2025,
                    venue="arXiv",
                    url="https://example.org/paper",
                    abstract="A lightweight spectral gate reduces redundant bands.",
                    source=self.name,
                    keywords=[query],
                )
            ],
        )


class ResearchToolsTest(unittest.TestCase):
    def test_research_decision_flow(self):
        literature_provider_registry.register("fake", FakeLiteratureProvider(), replace=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            write_synthetic_mat(data_root, seed=11)
            agent = CoordinatorAgent()
            audit = agent.audit(data_root, root / "audit.json")
            spectral = agent.analyze(audit, root / "spectral.json")
            recommendation = agent.recommend(audit, spectral, root / "recommendation.json")
            plan = agent.plan(
                audit,
                spectral,
                recommendation,
                root / "plan.yaml",
                root / "run",
                seed=11,
            )
            self.assertIn("evidence", plan.metadata)

            result = agent.run(plan)
            literature = agent.search_literature(
                "hyperspectral image classification",
                root / "literature.json",
                provider_name="fake",
            )
            agenda = agent.design_auto_experiments(audit, spectral, recommendation)
            proposals = agent.propose_parameter_updates(plan, result, audit)
            module = agent.propose_module(audit, spectral, literature.papers)

            self.assertGreaterEqual(len(agenda.candidates), 1)
            self.assertGreaterEqual(len(proposals), 1)
            self.assertTrue(module.evidence)
            self.assertIn("ClassifierFactory", module.required_interfaces)


if __name__ == "__main__":
    unittest.main()

