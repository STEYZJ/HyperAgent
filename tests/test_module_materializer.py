import tempfile
import unittest
from pathlib import Path

from hyperagent.core.io import write_yaml
from hyperagent.schemas import (
    ModuleProposal,
    ExperimentPlan,
    ModelConfig,
    PreprocessingConfig,
    SplitConfig,
)
from hyperagent.tools.module_materializer import ModuleMaterializer


class ModuleMaterializerTest(unittest.TestCase):
    def test_materialize_module_and_ablation_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = ModuleProposal(
                name="EvidenceGuidedSpectralGate",
                module_type="lightweight_spectral_gate",
                insertion_point="models registry",
                design_summary="Use a spectral gate before classification.",
                expected_effect="Reduce redundant bands.",
                implementation_steps=["generate model"],
                required_interfaces=["ClassifierFactory"],
            )
            base_plan = ExperimentPlan(
                experiment_name="demo_svm_seed42",
                dataset_root="data",
                output_dir="out",
                seed=42,
                reader_name="mat",
                split=SplitConfig(),
                preprocessing=PreprocessingConfig(remove_bands=[1]),
                model=ModelConfig(name="svm", params={"C": 10.0}),
            )
            base_plan_path = root / "base.yaml"
            write_yaml(base_plan_path, base_plan)

            result = ModuleMaterializer().materialize(
                proposal,
                output_dir=root / "generated",
                base_plan_path=base_plan_path,
                ablation_output_dir=root / "ablations",
            )

            self.assertTrue(Path(result.model_file).exists())
            self.assertEqual(result.model_name, "evidence_guided_spectral_gate")
            self.assertEqual(len(result.generated_configs), 4)
            code = Path(result.model_file).read_text(encoding="utf-8")
            self.assertIn("model_registry.register", code)


if __name__ == "__main__":
    unittest.main()
