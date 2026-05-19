"""Raster dataset readers for TIFF and ENVI HSI layouts."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hyperagent.core.registries import dataset_reader_registry


LABEL_HINTS = ("label", "labels", "gt", "truth", "class", "mask", "ground")
TIFF_SUFFIXES = {".tif", ".tiff"}
ENVI_DATA_SUFFIXES = ("", ".raw", ".dat", ".img", ".bin", ".bsq", ".bil", ".bip")
ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


class TiffDatasetReader:
    """Read an HSI cube and label map from TIFF files."""

    name = "tiff"

    def can_read(self, data_root: Path) -> bool:
        path = Path(data_root)
        if path.is_file():
            return path.suffix.lower() in TIFF_SUFFIXES
        return any(_iter_files(path, TIFF_SUFFIXES)) if path.exists() else False

    def read(self, data_root: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        tifffile = _import_tifffile()
        path = Path(data_root)
        image_files = [path] if path.is_file() else list(_iter_files(path, TIFF_SUFFIXES))
        if not image_files:
            raise FileNotFoundError(f"No TIFF files found under {data_root}")

        records: List[Dict[str, Any]] = []
        for image_path in image_files:
            array = np.asarray(tifffile.imread(str(image_path)))
            if array.size == 0:
                continue
            records.append(
                {
                    "path": image_path,
                    "key": image_path.stem,
                    "array": array,
                    "format": "tiff",
                }
            )
        if not records:
            raise ValueError(f"No ndarray payloads found in TIFF files: {image_files}")

        cube_record = _select_cube(records, "TIFF")
        label_record = _select_label(records, cube_record)
        cube, labels, warnings = _normalize_cube_and_labels(
            cube_record["array"], label_record["array"]
        )

        metadata = _metadata_from_records(cube_record, label_record, warnings)
        return cube, labels, metadata


class EnviDatasetReader:
    """Read an HSI cube and label map from ENVI .hdr plus raw binary files."""

    name = "envi"

    def can_read(self, data_root: Path) -> bool:
        path = Path(data_root)
        if path.is_file():
            return path.suffix.lower() == ".hdr"
        return any(path.rglob("*.hdr")) if path.exists() else False

    def read(self, data_root: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        path = Path(data_root)
        hdr_files = [path] if path.is_file() else sorted(path.rglob("*.hdr"))
        if not hdr_files:
            raise FileNotFoundError(f"No ENVI .hdr files found under {data_root}")

        records: List[Dict[str, Any]] = []
        for hdr_path in hdr_files:
            header = _parse_envi_header(hdr_path)
            data_path = _find_envi_data_file(hdr_path, header)
            if data_path is None:
                continue
            array = _read_envi_array(data_path, header)
            records.append(
                {
                    "path": data_path,
                    "header_path": hdr_path,
                    "key": hdr_path.stem,
                    "array": array,
                    "header": header,
                    "format": "envi",
                }
            )
        if not records:
            raise ValueError(f"No readable ENVI data payloads found under {data_root}")

        cube_record = _select_cube(records, "ENVI")
        label_record = _select_label(records, cube_record)
        cube, labels, warnings = _normalize_cube_and_labels(
            cube_record["array"], label_record["array"]
        )
        metadata = _metadata_from_records(cube_record, label_record, warnings)
        metadata["cube_header_path"] = str(cube_record.get("header_path", ""))
        metadata["label_header_path"] = str(label_record.get("header_path", ""))
        metadata["interleave"] = cube_record.get("header", {}).get("interleave")
        return cube, labels, metadata


def _iter_files(path: Path, suffixes: Sequence[str]) -> List[Path]:
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in suffixes)


def _import_tifffile():
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "TIFF dataset reading requires the optional 'tifffile' package."
        ) from exc
    return tifffile


def _select_cube(records: List[Dict[str, Any]], format_name: str) -> Dict[str, Any]:
    candidates = [item for item in records if np.asarray(item["array"]).ndim == 3]
    if not candidates:
        raise ValueError(f"No 3D HSI cube found in {format_name} files")
    return max(candidates, key=lambda item: int(np.prod(np.asarray(item["array"]).shape)))


def _select_label(
    records: List[Dict[str, Any]], cube_record: Dict[str, Any]
) -> Dict[str, Any]:
    cube = np.asarray(cube_record["array"])
    spatial_shapes = {tuple(cube.shape[:2]), tuple(cube.shape[1:])}
    candidates: List[Dict[str, Any]] = []
    for item in records:
        label_array = _as_label_array(np.asarray(item["array"]))
        if label_array is None:
            continue
        if tuple(label_array.shape) not in spatial_shapes:
            continue
        candidate = dict(item)
        candidate["array"] = label_array
        candidates.append(candidate)
    if not candidates:
        raise ValueError("No 2D label map matching the cube spatial shape was found")
    return max(candidates, key=_label_score)


def _as_label_array(array: np.ndarray) -> Optional[np.ndarray]:
    if array.ndim == 2:
        return array
    if array.ndim == 3 and 1 in array.shape:
        squeezed = np.squeeze(array)
        if squeezed.ndim == 2:
            return squeezed
    return None


def _label_score(item: Dict[str, Any]) -> int:
    array = np.asarray(item["array"])
    text = " ".join(
        str(value).lower()
        for value in (item.get("key", ""), item.get("path", ""), item.get("header_path", ""))
    )
    label_hint = any(token in text for token in LABEL_HINTS)
    integer_like = _is_integer_like(array)
    compact_classes = False
    if integer_like:
        unique_count = int(np.unique(array).size)
        compact_classes = 1 < unique_count <= 1024
    return int(label_hint) * 4 + int(integer_like) * 2 + int(compact_classes)


def _is_integer_like(array: np.ndarray) -> bool:
    if not np.issubdtype(array.dtype, np.number):
        return False
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return False
    return bool(np.allclose(finite, np.round(finite)))


def _normalize_cube_and_labels(
    cube_array: np.ndarray, label_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cube = np.asarray(cube_array)
    labels = np.asarray(label_array)
    warnings: List[str] = []

    if labels.ndim == 3 and 1 in labels.shape:
        labels = np.squeeze(labels)
        warnings.append("Label map had a singleton band dimension and was squeezed to 2D.")
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

    return cube.astype(np.float32, copy=False), labels.astype(np.int64, copy=False), warnings


def _metadata_from_records(
    cube_record: Dict[str, Any], label_record: Dict[str, Any], warnings: List[str]
) -> Dict[str, Any]:
    return {
        "cube_key": cube_record["key"],
        "cube_path": str(cube_record["path"]),
        "label_key": label_record["key"],
        "label_path": str(label_record["path"]),
        "warnings": warnings,
    }


def _parse_envi_header(hdr_path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    pending_key: Optional[str] = None
    pending_parts: List[str] = []
    for raw_line in hdr_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.lower() == "envi":
            continue
        if pending_key:
            pending_parts.append(line)
            if "}" in line:
                entries[pending_key] = _clean_header_value(" ".join(pending_parts))
                pending_key = None
                pending_parts = []
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("{") and "}" not in value:
            pending_key = key
            pending_parts = [value]
            continue
        entries[key] = _clean_header_value(value)
    if pending_key:
        entries[pending_key] = _clean_header_value(" ".join(pending_parts))
    return entries


def _clean_header_value(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]
    return " ".join(cleaned.replace("\n", " ").split())


def _find_envi_data_file(hdr_path: Path, header: Dict[str, str]) -> Optional[Path]:
    data_file = header.get("data file")
    if data_file:
        candidate = Path(data_file)
        if not candidate.is_absolute():
            candidate = hdr_path.parent / candidate
        if candidate.exists() and candidate != hdr_path:
            return candidate

    for suffix in ENVI_DATA_SUFFIXES:
        candidate = hdr_path.with_suffix(suffix)
        if candidate.exists() and candidate != hdr_path:
            return candidate
    return None


def _read_envi_array(data_path: Path, header: Dict[str, str]) -> np.ndarray:
    samples = _header_int(header, "samples")
    lines = _header_int(header, "lines")
    bands = _header_int(header, "bands", default=1)
    dtype_code = _header_int(header, "data type")
    dtype = ENVI_DTYPES.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported ENVI data type {dtype_code} in {data_path}")
    byte_order = _header_int(header, "byte order", default=0)
    dtype = np.dtype(dtype).newbyteorder(">" if byte_order == 1 else "<")
    header_offset = _header_int(header, "header offset", default=0)
    interleave = header.get("interleave", "bsq").lower()

    expected_count = int(samples * lines * max(bands, 1))
    with data_path.open("rb") as handle:
        if header_offset:
            handle.seek(header_offset)
        flat = np.fromfile(handle, dtype=dtype, count=expected_count)
    if flat.size < expected_count:
        raise ValueError(
            f"ENVI data file {data_path} has {flat.size} values, expected {expected_count}"
        )

    if bands <= 1:
        return flat.reshape((lines, samples))
    if interleave == "bsq":
        return np.moveaxis(flat.reshape((bands, lines, samples)), 0, -1)
    if interleave == "bil":
        return np.transpose(flat.reshape((lines, bands, samples)), (0, 2, 1))
    if interleave == "bip":
        return flat.reshape((lines, samples, bands))
    raise ValueError(f"Unsupported ENVI interleave '{interleave}' in {data_path}")


def _header_int(header: Dict[str, str], key: str, default: Optional[int] = None) -> int:
    value = header.get(key)
    if value is None:
        if default is None:
            raise ValueError(f"ENVI header is missing required field '{key}'")
        return default
    token = value.replace(",", " ").split()[0]
    return int(float(token))


dataset_reader_registry.register(TiffDatasetReader.name, TiffDatasetReader(), replace=True)
dataset_reader_registry.register(EnviDatasetReader.name, EnviDatasetReader(), replace=True)
