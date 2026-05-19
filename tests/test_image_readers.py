import tempfile
import unittest
from pathlib import Path

import numpy as np

from hyperagent.core.bootstrap import bootstrap_default_components
from hyperagent.tools.dataset_inspector import DatasetInspector


try:
    import tifffile
except ImportError:
    tifffile = None


class ImageReaderTest(unittest.TestCase):
    @unittest.skipIf(tifffile is None, "tifffile is not installed")
    def test_tiff_reader_is_selected_by_dataset_inspector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube = np.arange(4 * 6 * 5, dtype=np.float32).reshape(4, 6, 5)
            labels = np.zeros((6, 5), dtype=np.int16)
            labels[:3, :] = 1
            labels[3:, :] = 2
            tifffile.imwrite(str(root / "scene_cube.tif"), cube, photometric="minisblack")
            tifffile.imwrite(str(root / "scene_gt.tif"), labels)

            bootstrap_default_components()
            audit = DatasetInspector().inspect(root)

            self.assertEqual(audit.reader_name, "tiff")
            self.assertEqual(audit.cube_shape, [6, 5, 4])
            self.assertEqual(audit.label_shape, [6, 5])
            self.assertEqual(audit.class_count, 2)
            self.assertEqual(audit.class_distribution, {"1": 15, "2": 15})

    def test_envi_reader_is_selected_by_dataset_inspector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
            labels = np.array(
                [
                    [1, 1, 2],
                    [1, 2, 2],
                    [3, 3, 2],
                    [0, 3, 3],
                ],
                dtype=np.int16,
            )
            self._write_envi(root / "cube", cube, np.float32, 4, "bip")
            self._write_envi(root / "labels_gt", labels, np.int16, 2, "bsq")

            bootstrap_default_components()
            audit = DatasetInspector().inspect(root)

            self.assertEqual(audit.reader_name, "envi")
            self.assertEqual(audit.cube_shape, [4, 3, 5])
            self.assertEqual(audit.label_shape, [4, 3])
            self.assertEqual(audit.class_count, 3)
            self.assertEqual(audit.class_distribution, {"1": 3, "2": 4, "3": 4})
            self.assertEqual(audit.metadata["interleave"], "bip")

    def test_envi_bsq_cube_becomes_last_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube = np.arange(3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4)
            labels = np.ones((3, 4), dtype=np.int16)
            self._write_envi(root / "cube", cube, np.float32, 4, "bsq", bands_first=True)
            self._write_envi(root / "labels_gt", labels, np.int16, 2, "bsq")

            bootstrap_default_components()
            audit = DatasetInspector().inspect(root)

            self.assertEqual(audit.reader_name, "envi")
            self.assertEqual(audit.cube_shape, [3, 4, 2])

    def _write_envi(
        self,
        stem: Path,
        array: np.ndarray,
        dtype,
        data_type: int,
        interleave: str,
        bands_first: bool = False,
    ) -> None:
        hdr_path = stem.with_suffix(".hdr")
        raw_path = stem.with_suffix(".raw")
        if array.ndim == 2:
            lines, samples = array.shape
            bands = 1
            payload = array.astype(dtype)
        elif bands_first:
            bands, lines, samples = array.shape
            payload = array.astype(dtype)
        else:
            lines, samples, bands = array.shape
            if interleave == "bip":
                payload = array.astype(dtype)
            elif interleave == "bsq":
                payload = np.moveaxis(array, -1, 0).astype(dtype)
            elif interleave == "bil":
                payload = np.transpose(array, (0, 2, 1)).astype(dtype)
            else:
                raise ValueError(interleave)
        payload.tofile(raw_path)
        hdr_path.write_text(
            "\n".join(
                [
                    "ENVI",
                    f"samples = {samples}",
                    f"lines = {lines}",
                    f"bands = {bands}",
                    "header offset = 0",
                    "file type = ENVI Standard",
                    f"data type = {data_type}",
                    f"interleave = {interleave}",
                    "byte order = 0",
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
