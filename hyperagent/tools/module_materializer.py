"""Materialize module proposals into generated model factories and ablation configs."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from hyperagent.core.io import read_yaml, write_json, write_yaml
from hyperagent.schemas import (
    AblationStudy,
    AblationVariant,
    ExperimentPlan,
    MaterializationResult,
    ModuleProposal,
)


class ModuleMaterializer:
    """Turns a ModuleProposal into a registry-backed classifier factory."""

    def materialize(
        self,
        proposal: ModuleProposal,
        output_dir: Path = Path("hyperagent/models/generated"),
        base_plan_path: Optional[Path] = None,
        ablation_output_dir: Optional[Path] = None,
        force: bool = False,
    ) -> MaterializationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = self._model_key(proposal.name)
        class_name = self._class_name(proposal.name)
        model_file = output_dir / f"{model_name}.py"
        if model_file.exists() and not force:
            raise FileExistsError(f"Generated model already exists: {model_file}")
        model_file.write_text(
            self._render_model_code(proposal, model_name, class_name),
            encoding="utf-8",
        )

        generated_configs: List[str] = []
        ablation_dir_text = None
        if base_plan_path is not None and ablation_output_dir is not None:
            generated_configs = self.generate_ablation_configs(
                base_plan_path,
                ablation_output_dir,
                proposal,
                model_name,
            )
            ablation_dir_text = str(ablation_output_dir)

        result = MaterializationResult(
            proposal_name=proposal.name,
            model_name=model_name,
            model_file=str(model_file),
            ablation_dir=ablation_dir_text,
            generated_configs=generated_configs,
            notes=[
                "Generated model uses the current pixel-spectral ClassifierFactory interface.",
                "Spatial context is represented as a lightweight adapter placeholder until patch-based runners are added.",
            ],
        )
        write_json(output_dir / f"{model_name}_materialization.json", result)
        return result

    def generate_ablation_configs(
        self,
        base_plan_path: Path,
        output_dir: Path,
        proposal: ModuleProposal,
        model_name: str,
    ) -> List[str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        base = read_yaml(base_plan_path)
        variants: List[AblationVariant] = []

        baseline = deepcopy(base)
        baseline["experiment_name"] = f"{base['experiment_name']}_baseline_control"
        baseline["output_dir"] = str(output_dir / "baseline_control")
        baseline_path = output_dir / "baseline_control.yaml"
        write_yaml(baseline_path, baseline)
        variants.append(
            AblationVariant(
                name="baseline_control",
                config_path=str(baseline_path),
                purpose="Original model and preprocessing protocol.",
            )
        )

        proposed = deepcopy(base)
        proposed["experiment_name"] = f"{base['experiment_name']}_{model_name}"
        proposed["output_dir"] = str(output_dir / model_name)
        proposed["model"] = {
            "name": model_name,
            "params": {
                "hidden_dim": 64,
                "epochs": 40,
                "lr": 0.001,
                "batch_size": 128,
                "gate_l1": 0.0005,
            },
        }
        proposed.setdefault("metadata", {})["module_proposal"] = proposal.to_dict()
        proposed_path = output_dir / f"{model_name}.yaml"
        write_yaml(proposed_path, proposed)
        variants.append(
            AblationVariant(
                name=model_name,
                config_path=str(proposed_path),
                purpose="Evaluate the materialized evidence-guided adapter.",
                changed_fields={"model.name": model_name},
            )
        )

        no_pruning = deepcopy(proposed)
        no_pruning["experiment_name"] = f"{base['experiment_name']}_{model_name}_no_band_pruning"
        no_pruning["output_dir"] = str(output_dir / f"{model_name}_no_band_pruning")
        no_pruning.setdefault("preprocessing", {})["remove_bands"] = []
        no_pruning_path = output_dir / f"{model_name}_no_band_pruning.yaml"
        write_yaml(no_pruning_path, no_pruning)
        variants.append(
            AblationVariant(
                name=f"{model_name}_no_band_pruning",
                config_path=str(no_pruning_path),
                purpose="Separate module contribution from spectral band pruning.",
                changed_fields={"preprocessing.remove_bands": []},
            )
        )

        study = AblationStudy(
            name=f"{model_name}_ablation",
            base_plan=str(base_plan_path),
            variants=variants,
            notes=[
                "Run variants with identical split seed before comparing OA/AA/Kappa.",
                "Add multi-seed repeats before claiming module improvement.",
            ],
        )
        study_path = output_dir / "ablation_study.json"
        write_json(study_path, study)
        return [str(item.config_path) for item in variants] + [str(study_path)]

    def _render_model_code(
        self,
        proposal: ModuleProposal,
        model_name: str,
        class_name: str,
    ) -> str:
        summary = repr(proposal.design_summary)
        return f'''"""Generated model factory for {proposal.name}.

This file is generated by HyperAgent's ModuleMaterializer.
Design summary: {proposal.design_summary}
"""

from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyperagent.core.registries import model_registry


DESIGN_SUMMARY = {summary}


class _SpectralGateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(input_dim))
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logits)
        return self.classifier(x * gate)

    def gate_l1(self) -> torch.Tensor:
        return torch.mean(torch.abs(torch.sigmoid(self.gate_logits)))


class {class_name}:
    """Evidence-guided spectral adapter using the current pixel classifier interface."""

    def __init__(self, params: Dict[str, Any], seed: int) -> None:
        self.params = params
        self.seed = seed
        self.labels: List[int] = []
        self.model = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        self.labels = sorted(int(v) for v in np.unique(y_train))
        label_to_index = {{label: idx for idx, label in enumerate(self.labels)}}
        y_index = np.asarray([label_to_index[int(v)] for v in y_train], dtype=np.int64)
        hidden_dim = int(self.params.get("hidden_dim", 64))
        epochs = int(self.params.get("epochs", 40))
        lr = float(self.params.get("lr", 0.001))
        batch_size = int(self.params.get("batch_size", 128))
        gate_l1 = float(self.params.get("gate_l1", 0.0005))

        self.model = _SpectralGateMLP(x_train.shape[1], hidden_dim, len(self.labels))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_index, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for _ in range(epochs):
            for features, target in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(features), target) + gate_l1 * self.model.gate_l1()
                loss.backward()
                optimizer.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("{class_name} must be fitted before predict")
        self.model.eval()
        outputs = []
        batch_size = int(self.params.get("batch_size", 128))
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32)
                pred = torch.argmax(self.model(batch), dim=1).cpu().numpy()
                outputs.extend(self.labels[int(idx)] for idx in pred)
        return np.asarray(outputs, dtype=np.int64)


def build_model(params: Dict[str, Any], seed: int) -> {class_name}:
    return {class_name}(params, seed)


model_registry.register("{model_name}", build_model, replace=True)
'''

    def _model_key(self, name: str) -> str:
        chars = []
        previous_was_lower_or_digit = False
        for char in name:
            if char.isalnum():
                if char.isupper() and previous_was_lower_or_digit:
                    chars.append("_")
                chars.append(char.lower())
                previous_was_lower_or_digit = char.islower() or char.isdigit()
            else:
                chars.append("_")
                previous_was_lower_or_digit = False
        key = "_".join(part for part in "".join(chars).split("_") if part)
        return key or "generated_model"

    def _class_name(self, name: str) -> str:
        words = [part for part in self._model_key(name).split("_") if part]
        return "".join(word.capitalize() for word in words) or "GeneratedModel"
