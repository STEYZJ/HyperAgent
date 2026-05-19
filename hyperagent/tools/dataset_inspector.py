"""Dataset audit tool."""

from pathlib import Path
from typing import Optional

import numpy as np

from hyperagent.core.registries import dataset_reader_registry
from hyperagent.schemas import DatasetAudit


class DatasetInspector:
    """Inspect an HSI dataset through a registered reader."""

    def inspect(self, data_root: Path, reader_name: Optional[str] = None) -> DatasetAudit:
        reader = self._select_reader(Path(data_root), reader_name)
        cube, labels, metadata = reader.read(Path(data_root))
        label_values, counts = np.unique(labels, return_counts=True)
        class_distribution = {
            str(int(label)): int(count)
            for label, count in zip(label_values, counts)
            if int(label) > 0
        }
        unlabeled_pixel_count = int(np.sum(labels <= 0))
        labeled_pixel_count = int(np.sum(labels > 0))
        warnings = list(metadata.get("warnings", []))
        if labeled_pixel_count == 0:
            warnings.append("No labeled pixels were found.")
        if cube.shape[:2] != labels.shape:
            warnings.append("Cube and label spatial shapes do not match.")

        return DatasetAudit(
            data_root=str(data_root),
            dataset_name=Path(data_root).stem if Path(data_root).is_file() else Path(data_root).name,
            cube_path=metadata.get("cube_path"),
            label_path=metadata.get("label_path"),
            cube_shape=[int(v) for v in cube.shape],
            label_shape=[int(v) for v in labels.shape],
            band_count=int(cube.shape[-1]),
            class_count=len(class_distribution),
            labeled_pixel_count=labeled_pixel_count,
            unlabeled_pixel_count=unlabeled_pixel_count,
            class_distribution=class_distribution,
            has_nan=bool(np.isnan(cube).any()),
            has_inf=bool(np.isinf(cube).any()),
            dtype=str(cube.dtype),
            reader_name=reader.name,
            warnings=warnings,
            metadata=metadata,
        )

    def _select_reader(self, data_root: Path, reader_name: Optional[str]):
        if reader_name:
            return dataset_reader_registry.get(reader_name)
        for key in dataset_reader_registry.keys():
            reader = dataset_reader_registry.get(key)
            if reader.can_read(data_root):
                return reader
        raise ValueError(f"No registered dataset reader can read {data_root}")

