"""MAT-file dataset reader for common HSI benchmark layouts."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.io import loadmat

from hyperagent.core.registries import dataset_reader_registry


class MatDatasetReader:
    name = "mat"

    def can_read(self, data_root: Path) -> bool:
        path = Path(data_root)
        if path.is_file():
            return path.suffix.lower() == ".mat"
        return any(path.rglob("*.mat")) if path.exists() else False

    def read(self, data_root: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        path = Path(data_root)
        mat_files = [path] if path.is_file() else sorted(path.rglob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found under {data_root}")

        arrays = self._load_arrays(mat_files)
        cube_record = self._select_cube(arrays)
        label_record = self._select_label(arrays, cube_record["array"])

        cube = np.asarray(cube_record["array"], dtype=np.float32)
        labels = np.asarray(label_record["array"])
        warnings: List[str] = []

        if cube.ndim != 3:
            raise ValueError(f"Selected cube is not 3D: {cube.shape}")
        if labels.ndim != 2:
            raise ValueError(f"Selected label map is not 2D: {labels.shape}")

        if cube.shape[:2] != labels.shape and cube.shape[1:] == labels.shape:
            cube = np.moveaxis(cube, 0, -1)
            warnings.append("Cube looked like band-first format and was moved to H,W,B.")

        if cube.shape[:2] != labels.shape:
            raise ValueError(
                f"Cube spatial shape {cube.shape[:2]} does not match labels {labels.shape}"
            )

        metadata = {
            "cube_key": cube_record["key"],
            "cube_path": str(cube_record["path"]),
            "label_key": label_record["key"],
            "label_path": str(label_record["path"]),
            "warnings": warnings,
        }
        return cube, labels.astype(np.int64), metadata

    def _load_arrays(self, mat_files: List[Path]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for mat_path in mat_files:
            content = loadmat(mat_path)
            for key, value in content.items():
                if key.startswith("__") or not isinstance(value, np.ndarray):
                    continue
                if value.size == 0:
                    continue
                records.append({"path": mat_path, "key": key, "array": value})
        if not records:
            raise ValueError(f"No ndarray payloads found in .mat files: {mat_files}")
        return records

    def _select_cube(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidates = [item for item in records if item["array"].ndim == 3]
        if not candidates:
            raise ValueError("No 3D HSI cube found in .mat files")
        return max(candidates, key=lambda item: int(np.prod(item["array"].shape)))

    def _select_label(
        self, records: List[Dict[str, Any]], cube: np.ndarray
    ) -> Dict[str, Any]:
        spatial_shapes = {tuple(cube.shape[:2]), tuple(cube.shape[1:])}
        candidates = [
            item
            for item in records
            if item["array"].ndim == 2 and tuple(item["array"].shape) in spatial_shapes
        ]
        if not candidates:
            raise ValueError("No 2D label map matching the cube spatial shape was found")

        def score(item: Dict[str, Any]) -> int:
            key = str(item["key"]).lower()
            label_hint = any(token in key for token in ("label", "gt", "truth", "class"))
            integer_like = np.allclose(item["array"], np.round(item["array"]))
            return int(label_hint) * 2 + int(integer_like)

        return max(candidates, key=score)


dataset_reader_registry.register(MatDatasetReader.name, MatDatasetReader(), replace=True)

