"""
analysis/feature/io/loader.py

Feature Loader
--------------
Loads previously saved feature sets back into Python dicts.
Supports .npz (NumPy archive) and .json formats.

DATA STRUCTURE CONTRACT on load:
    .npz:  arrays stored under their flat dot-key names.
           Non-array scalars stored as 0-d arrays, unwrapped on load.
    .json: plain JSON produced by exporter.py.
           numpy arrays are stored as lists and converted back on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from ..multi import MultiRunResult


def load_features(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a feature set from disk.

    Parameters
    ----------
    filepath : str | Path
        Path to a .npz or .json file produced by exporter.save_features().

    Returns
    -------
    dict  –  feature dict (flat or nested depending on how it was saved).

    DATA STRUCTURE VALIDITY:
        .npz files with 0-d arrays are unwrapped to Python scalars.
        .json files restore lists as lists (not converted to np.ndarray).

    Raises
    ------
    FileNotFoundError  if file does not exist.
    ValueError         if file extension is not .npz or .json.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    if path.suffix == ".npz":
        return _load_npz(path)
    elif path.suffix == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported format '{path.suffix}'. Use .npz or .json.")


def _load_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for key in data.files:
        val = data[key]
        # Unwrap 0-d arrays to Python scalars
        if val.ndim == 0:
            out[key] = val.item()
        else:
            out[key] = val
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_multi_run_result(dirpath: Union[str, Path], run_label: str = "experiment") -> MultiRunResult:
    """Load multi-run artifacts from disk and reconstruct MultiRunResult."""
    in_dir = Path(dirpath)
    if not in_dir.exists():
        raise FileNotFoundError(f"Directory not found: {in_dir}")

    mean_summary = _load_json(in_dir / f"{run_label}_mean_summary.json")
    std_summary = _load_json(in_dir / f"{run_label}_std_summary.json")
    ci95_summary = _load_json(in_dir / f"{run_label}_ci95_summary.json")

    cv_path = in_dir / f"{run_label}_cv_summary.json"
    consensus_path = in_dir / f"{run_label}_consensus_summary.json"
    finals_path = in_dir / f"{run_label}_run_finals.json"
    metadata_path = in_dir / f"{run_label}_metadata.json"

    cv_summary = _load_json(cv_path) if cv_path.exists() else {}
    consensus_score = _load_json(consensus_path) if consensus_path.exists() else {}
    run_finals = _load_json(finals_path) if finals_path.exists() else []
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}

    n_runs = int(metadata.get("n_runs", 0))
    layer_idx = int(metadata.get("layer_idx", 0))

    return MultiRunResult(
        run_summaries=[],
        run_finals=run_finals,
        n_runs=n_runs,
        layer_idx=layer_idx,
        mean_summary=mean_summary,
        std_summary=std_summary,
        cv_summary=cv_summary,
        ci95_summary=ci95_summary,
        consensus_score=consensus_score,
    )
