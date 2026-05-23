"""Reproducible benchmark protocol and fixed split fingerprints."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from hyperagent.agents.coordinator import CoordinatorAgent
from hyperagent.core.bootstrap import bootstrap_default_components
from hyperagent.core.io import read_json, read_yaml, write_json, write_yaml
from hyperagent.core.registries import dataset_reader_registry, model_registry
from hyperagent.data.preprocessing import flatten_labeled_pixels
from hyperagent.data.splits import stratified_train_test_split
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import ExperimentPlan


DEFAULT_STRONG_BASELINES = ["svm", "mlp", "random_forest", "knn"]


class FixedSplitStore:
    """Stores fixed split fingerprints without writing full sample indices."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def create_or_load(
        self,
        *,
        dataset_name: str,
        dataset_path: Path,
        reader_name: str,
        train_ratio: float,
        seed: int,
        method: str = "stratified_random",
    ) -> Dict[str, Any]:
        path = self.root / f"{_slug(dataset_name)}_seed{int(seed)}.json"
        if path.exists():
            return read_json(path)
        bootstrap_default_components()
        reader = dataset_reader_registry.get(reader_name)
        cube, labels, _metadata = reader.read(Path(dataset_path))
        _x, y, _mask = flatten_labeled_pixels(cube, labels)
        train_idx, test_idx = stratified_train_test_split(y, train_ratio=train_ratio, seed=int(seed))
        manifest = {
            "version": 1,
            "dataset": dataset_name,
            "dataset_path": str(dataset_path),
            "reader_name": reader_name,
            "method": method,
            "seed": int(seed),
            "train_ratio": float(train_ratio),
            "train_count": int(train_idx.shape[0]),
            "test_count": int(test_idx.shape[0]),
            "labeled_count": int(y.shape[0]),
            "fingerprint": self.fingerprint(train_idx, test_idx),
            "created_at": utc_now(),
        }
        write_json(path, manifest)
        return manifest

    @staticmethod
    def fingerprint(train_idx: np.ndarray, test_idx: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(train_idx, dtype=np.int64).tobytes())
        digest.update(b"|")
        digest.update(np.asarray(test_idx, dtype=np.int64).tobytes())
        return digest.hexdigest()[:24]


class BenchmarkProtocolStore:
    """Creates and loads dataset x seed x baseline benchmark protocols."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "benchmark_protocol.json"

    def create(
        self,
        *,
        catalog_path: Path,
        dataset_names: Optional[Iterable[str]] = None,
        seeds: Iterable[int] = (42, 43),
        baselines: Iterable[str] = DEFAULT_STRONG_BASELINES,
    ) -> Dict[str, Any]:
        bootstrap_default_components()
        self.root.mkdir(parents=True, exist_ok=True)
        catalog = read_yaml(catalog_path)
        datasets = catalog.get("datasets", {})
        if not isinstance(datasets, dict):
            raise ValueError(f"Catalog must contain a datasets mapping: {catalog_path}")
        selected_names = list(dataset_names or datasets.keys())
        normalized_seeds = [int(seed) for seed in seeds]
        normalized_baselines = self._available_baselines(baselines)
        coordinator = CoordinatorAgent()
        split_store = FixedSplitStore(self.root / "fixed_splits")
        rows: List[Dict[str, Any]] = []
        for name in selected_names:
            spec = datasets.get(name)
            if spec is None:
                rows.append(
                    {
                        "dataset": name,
                        "status": "missing_catalog_entry",
                        "warnings": [f"Dataset '{name}' is not in {catalog_path}."],
                    }
                )
                continue
            rows.append(
                self._dataset_protocol(
                    name,
                    spec,
                    coordinator,
                    split_store,
                    normalized_seeds,
                    normalized_baselines,
                )
            )
        protocol = {
            "version": 1,
            "created_at": utc_now(),
            "catalog_path": str(catalog_path),
            "root": str(self.root),
            "seeds": normalized_seeds,
            "baselines": normalized_baselines,
            "datasets": rows,
        }
        write_json(self.path, protocol)
        self._write_markdown(self.root / "benchmark_protocol.md", protocol)
        return protocol

    def load(self, path: Optional[Path] = None) -> Dict[str, Any]:
        return read_json(Path(path) if path is not None else self.path)

    def _dataset_protocol(
        self,
        name: str,
        spec: Dict[str, Any],
        coordinator: CoordinatorAgent,
        split_store: FixedSplitStore,
        seeds: List[int],
        baselines: List[str],
    ) -> Dict[str, Any]:
        dataset_path = Path(str(spec.get("local_example", "")))
        dataset_dir = self.root / _slug(name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if not dataset_path.exists():
            return {
                "dataset": name,
                "status": "missing_local_path",
                "dataset_path": str(dataset_path),
                "source_url": spec.get("source_url"),
                "baselines": baselines,
                "warnings": [f"Local path does not exist: {dataset_path}"],
            }
        audit_path = dataset_dir / "audit.json"
        spectral_path = dataset_dir / "spectral_report.json"
        recommendation_path = dataset_dir / "model_recommendation.json"
        plan_path = dataset_dir / "base_plan.yaml"
        audit = coordinator.audit(dataset_path, audit_path)
        spectral = coordinator.analyze(audit, spectral_path)
        recommendation = coordinator.recommend(audit, spectral, recommendation_path)
        plan = coordinator.plan(
            audit,
            spectral,
            recommendation,
            plan_path,
            self.root / "planned_runs" / _slug(name),
            seed=seeds[0] if seeds else 42,
        )
        splits = [
            {
                **split_store.create_or_load(
                    dataset_name=name,
                    dataset_path=dataset_path,
                    reader_name=audit.reader_name,
                    train_ratio=plan.split.train_ratio,
                    seed=seed,
                    method=plan.split.method,
                ),
                "manifest_path": str(split_store.root / f"{_slug(name)}_seed{int(seed)}.json"),
            }
            for seed in seeds
        ]
        return {
            "dataset": name,
            "status": "ready",
            "dataset_path": str(dataset_path),
            "source_url": spec.get("source_url"),
            "audit_path": str(audit_path),
            "spectral_report_path": str(spectral_path),
            "recommendation_path": str(recommendation_path),
            "plan_path": str(plan_path),
            "reader_name": audit.reader_name,
            "cube_shape": audit.cube_shape,
            "band_count": audit.band_count,
            "class_count": audit.class_count,
            "labeled_pixel_count": audit.labeled_pixel_count,
            "train_ratio": plan.split.train_ratio,
            "split_method": plan.split.method,
            "splits": splits,
            "baselines": baselines,
            "warnings": list(audit.warnings),
        }

    def _available_baselines(self, baselines: Iterable[str]) -> List[str]:
        requested = [str(item).strip() for item in baselines if str(item).strip()]
        available = set(model_registry.keys())
        selected = [name for name in requested if name in available]
        if not selected:
            selected = [name for name in DEFAULT_STRONG_BASELINES if name in available]
        return selected

    def _write_markdown(self, path: Path, protocol: Dict[str, Any]) -> Path:
        lines = [
            "# Benchmark Protocol",
            "",
            f"- Catalog: {protocol['catalog_path']}",
            f"- Seeds: {', '.join(str(seed) for seed in protocol['seeds'])}",
            f"- Baselines: {', '.join(protocol['baselines'])}",
            "",
            "| Dataset | Status | Reader | Classes | Labeled | Split Fingerprints | Baselines |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
        for row in protocol["datasets"]:
            fingerprints = ", ".join(
                str(split.get("fingerprint", "")) for split in row.get("splits", [])
            )
            lines.append(
                f"| {row.get('dataset', '')} | {row.get('status', '')} | "
                f"{row.get('reader_name', '')} | {row.get('class_count', '')} | "
                f"{row.get('labeled_pixel_count', '')} | {fingerprints} | "
                f"{', '.join(row.get('baselines', []))} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def build_baseline_plan(
    base_plan: ExperimentPlan,
    *,
    dataset_name: str,
    baseline: str,
    output_dir: Path,
    split_fingerprints: List[str],
) -> ExperimentPlan:
    data = deepcopy(base_plan.to_dict())
    data["experiment_name"] = f"{_slug(dataset_name)}_{baseline}_protocol"
    data["output_dir"] = str(output_dir)
    data["model"] = {
        "name": baseline,
        "params": default_baseline_params(baseline),
    }
    metadata = dict(data.get("metadata", {}))
    metadata["benchmark_protocol"] = {
        "dataset": dataset_name,
        "baseline": baseline,
        "split_fingerprints": list(split_fingerprints),
        "purpose": "Fixed-split multi-seed benchmark protocol.",
    }
    data["metadata"] = metadata
    return ExperimentPlan.from_dict(data)


def default_baseline_params(name: str) -> Dict[str, Any]:
    if name == "svm":
        return {"kernel": "rbf", "C": 10.0, "gamma": "scale"}
    if name == "mlp":
        return {"hidden_dim": 64, "epochs": 30, "lr": 0.001, "batch_size": 128}
    if name == "random_forest":
        return {"n_estimators": 100, "n_jobs": 1}
    if name == "knn":
        return {"n_neighbors": 5, "weights": "distance"}
    return {}


def save_protocol_plan(path: Path, plan: ExperimentPlan) -> Path:
    write_yaml(path, plan)
    return path


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
