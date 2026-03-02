"""
analysis/feature/composer.py

Feature Composer Module
------------------------
Aggregates per-step feature dicts (produced by extractor.py) across time to
generate longitudinal / time-series feature tables.

Also handles multi-layer aggregation: runs extractor across all L opinion
layers and combines results.

Data contract
-------------
Input:
    step_features : List[Dict]   – one dict per time step from extract_all_features()
    Each dict has keys: 'meta', 'opinion', 'spatial', 'topo', 'network_opinion', 'event'

Output:
    composed : Dict[str, np.ndarray]  key → 1-D time-series array of length T
    summary  : Dict[str, float]       aggregate statistics (mean / std / min / max)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """
    Recursively flatten a nested dict to dot-notation keys.

    Examples
    --------
    {'opinion': {'mean_opinion': 0.5}} → {'opinion.mean_opinion': 0.5}

    DATA STRUCTURE VALIDITY:
        Only scalar float / int values are kept.
        Lists, arrays, dicts are recursed or skipped.
    """
    out: Dict[str, float] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full_key))
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[full_key] = float(v)
        # lists / arrays / None are intentionally skipped
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Per-step flattening
# ═════════════════════════════════════════════════════════════════════════════

def flatten_step(step_feat: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten a single step's feature dict into a flat scalar dict.

    DATA STRUCTURE REQUIREMENTS:
        step_feat must be the output of extract_all_features():
            keys: 'meta', 'opinion', 'spatial', 'topo', 'network_opinion', 'event'
        Non-scalar values (lists, arrays) in 'meta' (e.g., 'data_issues') are skipped.

    Returns
    -------
    dict[str, float]  –  dot-separated keys, all float values
    """
    # Exclude 'meta' keys that are not scalar
    flat: Dict[str, float] = {}
    for section_key, section_val in step_feat.items():
        if section_key == "meta":
            # Only include numeric meta fields
            for mk, mv in section_val.items():
                if isinstance(mv, (int, float, np.floating, np.integer)) and mv is not None:
                    flat[f"meta.{mk}"] = float(mv)
        elif isinstance(section_val, dict):
            flat.update(_flatten(section_val, prefix=section_key))
    return flat


# ═════════════════════════════════════════════════════════════════════════════
# Time-series composition
# ═════════════════════════════════════════════════════════════════════════════

def compose_timeseries(
    step_features: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Stack per-step flat dicts into time-series arrays.

    DATA STRUCTURE REQUIREMENTS:
        step_features: non-empty list of dicts from extract_all_features()
        All steps should share the same set of keys (missing keys filled with NaN).

    Returns
    -------
    dict[str, np.ndarray]
        Each key maps to a 1-D float64 array of length T (num steps).

    DATA STRUCTURE VALIDITY:
        - Empty list → returns {}
        - Steps with missing keys get NaN for that step
        - Keys present in some steps but not others are kept with NaN padding
    """
    if not step_features:
        return {}

    T = len(step_features)
    flat_steps = [flatten_step(s) for s in step_features]

    # Union of all keys
    all_keys = sorted({k for fs in flat_steps for k in fs})

    timeseries: Dict[str, np.ndarray] = {}
    for key in all_keys:
        arr = np.array([fs.get(key, np.nan) for fs in flat_steps], dtype=np.float64)
        timeseries[key] = arr

    return timeseries


# ═════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═════════════════════════════════════════════════════════════════════════════

def summarize_timeseries(
    timeseries: Dict[str, np.ndarray],
    percentiles: Sequence[float] = (25.0, 50.0, 75.0),
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics over each time-series.

    Parameters
    ----------
    timeseries : output of compose_timeseries()
    percentiles : which percentiles to include

    Returns
    -------
    dict[str, dict]  – per-key stats:  mean, std, min, max, p25, p50, p75, ...

    DATA STRUCTURE VALIDITY:
        NaN values are ignored via np.nanmean / np.nanstd etc.
        All-NaN arrays produce NaN summary stats (not errors).
    """
    summary: Dict[str, Dict[str, float]] = {}
    for key, arr in timeseries.items():
        valid = arr[~np.isnan(arr)]
        stats: Dict[str, float] = {
            "mean":  float(np.nanmean(arr)),
            "std":   float(np.nanstd(arr)),
            "min":   float(np.nanmin(arr)),
            "max":   float(np.nanmax(arr)),
            "n_valid": float(len(valid)),
        }
        for p in percentiles:
            stats[f"p{int(p)}"] = float(np.nanpercentile(arr, p)) if len(valid) > 0 else float("nan")
        summary[key] = stats
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Multi-layer aggregation
# ═════════════════════════════════════════════════════════════════════════════

def compose_multilayer(
    opinion_matrix: np.ndarray,
    extract_fn,               # callable: (opinions, layer_idx) → Dict
    agg: str = "mean",
) -> Dict[str, float]:
    """
    Run an opinion-feature extractor across all L layers and aggregate.

    Parameters
    ----------
    opinion_matrix : (N, L) float array
    extract_fn     : callable accepting (opinion_matrix, layer_idx=int)
                     returning Dict[str, float]
    agg            : 'mean' | 'max' | 'min' | 'std'

    Returns
    -------
    dict[str, float]  – aggregated per-layer features

    DATA STRUCTURE VALIDITY:
        opinion_matrix must be 2-D with shape (N, L), L >= 1.
        If L == 0 returns {}.
    """
    if opinion_matrix.ndim != 2 or opinion_matrix.shape[1] == 0:
        return {}

    L = opinion_matrix.shape[1]
    layer_results: List[Dict[str, float]] = []

    for l in range(L):
        feats = extract_fn(opinion_matrix, layer_idx=l)
        flat = _flatten(feats)
        layer_results.append(flat)

    if not layer_results:
        return {}

    all_keys = sorted({k for lr in layer_results for k in lr})
    aggregated: Dict[str, float] = {}

    agg_fns = {
        "mean": np.nanmean,
        "max":  np.nanmax,
        "min":  np.nanmin,
        "std":  np.nanstd,
    }
    agg_fn = agg_fns.get(agg, np.nanmean)

    for key in all_keys:
        vals = np.array([lr.get(key, np.nan) for lr in layer_results])
        aggregated[f"{key}.layer_{agg}"] = float(agg_fn(vals))

    return aggregated
