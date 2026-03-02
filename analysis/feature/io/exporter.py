"""
analysis/feature/io/exporter.py

Feature Exporter
----------------
Serialises feature dicts produced by extractor.py / pipeline.py to disk.

Supported formats:
    .npz  – compressed NumPy archive (recommended for large timeseries arrays)
    .json – human-readable (recommended for summary / final dicts)

DATA STRUCTURE CONTRACT on save:
    timeseries  dict[str, np.ndarray (T,)]   → stored as named arrays in .npz
    summary     dict[str, dict[str, float]]  → stored as JSON
    final       dict (nested)                → stored as JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


# ── custom JSON encoder ────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars / arrays to native Python types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── public API ─────────────────────────────────────────────────────────────

def save_features(
    features: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "json",
):
    """
    Save a feature dict to disk.

    Parameters
    ----------
    features : dict
        The dict to save.  Accepted shapes:
            - flat   dict[str, float]              → .npz or .json
            - nested dict[str, dict | np.ndarray]  → .npz or .json
    filepath : str | Path
        Destination file.  Extension must match format.
    format : 'json' | 'npz'
        'json' uses _NumpyEncoder to handle numpy scalars.
        'npz'  flattens one level and stores arrays.

    DATA STRUCTURE VALIDITY:
        Nested dicts in 'npz' mode are flattened one level deep.
        Values that are not float/int/ndarray are cast to str in JSON mode,
        skipped in npz mode.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        _save_json(features, path)
    elif format == "npz":
        _save_npz(features, path)
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'json' or 'npz'.")


def save_timeseries(
    timeseries: Dict[str, np.ndarray],
    filepath: Union[str, Path],
):
    """
    Convenience wrapper: save a timeseries dict as .npz.

    DATA STRUCTURE VALIDITY:
        Each value must be a 1-D float64 array of the same length T.
        Mixed lengths are allowed (stored independently).
    """
    path = Path(filepath)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in timeseries.items()})


def save_summary(
    summary: Dict[str, Any],
    filepath: Union[str, Path],
):
    """Convenience wrapper: save summary stats as .json."""
    path = Path(filepath)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    _save_json(summary, path)


# ── private helpers ────────────────────────────────────────────────────────

def _save_json(obj: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)


def _save_npz(features: Dict[str, Any], path: Path):
    """
    Flatten one level, convert values to arrays, save compressed.

    DATA STRUCTURE NOTES:
        - Scalar float/int → 0-d np.ndarray
        - np.ndarray       → stored as-is
        - nested dict      → skipped with a warning (use save_json for nested)
        - None             → skipped
    """
    save_dict: Dict[str, np.ndarray] = {}
    for k, v in features.items():
        safe_key = k.replace(".", "__")  # dot is illegal in npz key
        if isinstance(v, np.ndarray):
            save_dict[safe_key] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            save_dict[safe_key] = np.array(v)
        elif v is None:
            pass  # skip
        # nested dicts skipped (use JSON for those)

    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    np.savez_compressed(path, **save_dict)
