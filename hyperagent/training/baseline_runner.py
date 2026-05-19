"""Baseline experiment runner."""

import time
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hyperagent.core.io import write_json, write_yaml
from hyperagent.core.registries import (
    dataset_reader_registry,
    evaluator_registry,
    model_registry,
)
from hyperagent.data.preprocessing import flatten_labeled_pixels, normalize_cube, remove_bands
from hyperagent.data.splits import stratified_train_test_split
from hyperagent.schemas import ExperimentPlan, ExperimentResult


class BaselineRunner:
    name = "baseline"

    def run(self, plan: ExperimentPlan) -> ExperimentResult:
        start = time.time()
        output_dir = Path(plan.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(output_dir / "plan.yaml", plan)

        reader = dataset_reader_registry.get(plan.reader_name)
        cube, labels, metadata = reader.read(Path(plan.dataset_root))
        cube = remove_bands(cube, plan.preprocessing.remove_bands)
        cube = normalize_cube(cube, plan.preprocessing.normalization)
        x, y, labeled_mask = flatten_labeled_pixels(cube, labels)
        train_idx, test_idx = stratified_train_test_split(
            y,
            train_ratio=plan.split.train_ratio,
            seed=plan.seed,
        )

        model_factory = model_registry.get(plan.model.name)
        model = model_factory(plan.model.params, plan.seed)
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])

        evaluator = evaluator_registry.get("classification")
        evaluation = evaluator.evaluate(y[test_idx], y_pred)
        artifacts: List[str] = []
        map_paths = self._write_classification_map(output_dir, model, cube, labels)
        artifacts.extend(str(path) for path in map_paths)
        warnings = list(metadata.get("warnings", []))

        result = ExperimentResult(
            experiment_name=plan.experiment_name,
            experiment_dir=str(output_dir),
            model_name=plan.model.name,
            seed=plan.seed,
            train_samples=int(train_idx.shape[0]),
            test_samples=int(test_idx.shape[0]),
            evaluation=evaluation,
            artifacts=artifacts,
            duration_sec=float(time.time() - start),
            status="completed",
            warnings=warnings,
        )
        write_json(output_dir / "result.json", result)
        return result

    def _write_classification_map(self, output_dir: Path, model, cube: np.ndarray, labels: np.ndarray):
        flat = cube.reshape(-1, cube.shape[-1])
        predictions = model.predict(flat).reshape(labels.shape)
        predictions = np.where(labels > 0, predictions, 0)
        npy_path = output_dir / "classification_map.npy"
        png_path = output_dir / "classification_map.png"
        np.save(npy_path, predictions)
        plt.figure(figsize=(5, 5))
        plt.imshow(predictions, cmap="tab20")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(png_path, dpi=150)
        plt.close()
        return [npy_path, png_path]

