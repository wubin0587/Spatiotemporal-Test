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
