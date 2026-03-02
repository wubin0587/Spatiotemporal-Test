"""
analysis/feature/composer.py

Feature Composer Module
------------------------
Aggregates per-step feature dicts (produced by extractor.py) across time to
generate longitudinal / time-series feature tables.

Also handles multi-layer aggregation and computes trend-based metrics for AI analysis.

Data contract
-------------
Input:
    step_features : List[Dict]   – one dict per time step from extract_all_features()

Output:
    composed : Dict[str, np.ndarray]  key → 1-D time-series array of length T
    summary  : Dict[str, float]       aggregate stats + trend metrics
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
    """
    out: Dict[str, float] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full_key))
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[full_key] = float(v)
    return out


def _calculate_trend_metrics(arr: np.ndarray, window_ratio: float = 0.1) -> Dict[str, float]:
    """
    Compute evolution metrics for a single time-series array.
    Used to give AI context about 'how' the value changed, not just the mean.
    
    Metrics:
    - trend_slope:      Slope of linear regression (direction of change).
    - evolution_delta:  Mean(End) - Mean(Start).
    - start_mean:       Average of first 10%.
    - end_mean:         Average of last 10%.
    - final_stability:  Std dev of last 10% (lower = converged).
    - volatility:       Std dev of step-to-step changes (roughness).
    """
    # Filter NaNs for trend calculation
    y = arr[~np.isnan(arr)]
    n = len(y)
    
    if n < 2:
        return {}

    # 1. Linear Trend (Slope)
    # We normalize X to [0, 1] so slope represents "total change if linear"
    # This makes the slope independent of the number of steps (T).
    x = np.linspace(0, 1, n)
    slope, _ = np.polyfit(x, y, 1)

    # 2. Early vs Late (Evolution)
    w_size = max(1, int(n * window_ratio))
    start_vals = y[:w_size]
    end_vals = y[-w_size:]
    
    start_mean = np.mean(start_vals)
    end_mean = np.mean(end_vals)
    
    # 3. Volatility (smoothness)
    # Std dev of the first difference: how much does it jump per step?
    diffs = np.diff(y)
    volatility = np.std(diffs) if len(diffs) > 0 else 0.0

    return {
        "trend_slope":     float(slope),
        "evolution_delta": float(end_mean - start_mean),
        "start_mean":      float(start_mean),
        "end_mean":        float(end_mean),
        "final_stability": float(np.std(end_vals)), # Low value = Converged
        "volatility":      float(volatility),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Per-step flattening
# ═════════════════════════════════════════════════════════════════════════════

def flatten_step(step_feat: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a single step's feature dict into a flat scalar dict."""
    flat: Dict[str, float] = {}
    for section_key, section_val in step_feat.items():
        if section_key == "meta":
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
    """Stack per-step flat dicts into time-series arrays."""
    if not step_features:
        return {}

    T = len(step_features)
    flat_steps = [flatten_step(s) for s in step_features]

    all_keys = sorted({k for fs in flat_steps for k in fs})

    timeseries: Dict[str, np.ndarray] = {}
    for key in all_keys:
        arr = np.array([fs.get(key, np.nan) for fs in flat_steps], dtype=np.float64)
        timeseries[key] = arr

    return timeseries


# ═════════════════════════════════════════════════════════════════════════════
# Summary statistics (Extended)
# ═════════════════════════════════════════════════════════════════════════════

def summarize_timeseries(
    timeseries: Dict[str, np.ndarray],
    percentiles: Sequence[float] = (25.0, 50.0, 75.0),
    include_trends: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics AND trend metrics over each time-series.

    Parameters
    ----------
    timeseries : output of compose_timeseries()
    percentiles : which percentiles to include
    include_trends : if True, calculates slope, volatility, convergence stats
                     (Recommended for AI analysis)

    Returns
    -------
    dict[str, dict]
        Per-key stats. Example keys:
        - 'mean', 'std', 'max'
        - 'trend_slope' (if positive, metric is growing)
        - 'final_stability' (if low, system converged)
    """
    summary: Dict[str, Dict[str, float]] = {}
    
    for key, arr in timeseries.items():
        # Remove NaNs for stat calculation
        valid = arr[~np.isnan(arr)]
        n_valid = len(valid)
        
        # 1. Basic Stats
        stats: Dict[str, float] = {
            "mean":    float(np.nanmean(arr)),
            "std":     float(np.nanstd(arr)),
            "min":     float(np.nanmin(arr)),
            "max":     float(np.nanmax(arr)),
            "n_valid": float(n_valid),
        }
        
        if n_valid > 0:
            for p in percentiles:
                stats[f"p{int(p)}"] = float(np.nanpercentile(valid, p))
        else:
            for p in percentiles:
                stats[f"p{int(p)}"] = float("nan")

        # 2. Trend Metrics (Feature Engineering for AI)
        if include_trends and n_valid >= 5: # Need a few points for trend
            trends = _calculate_trend_metrics(arr)
            stats.update(trends)
        
        summary[key] = stats

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Multi-layer aggregation
# ═════════════════════════════════════════════════════════════════════════════

def compose_multilayer(
    opinion_matrix: np.ndarray,
    extract_fn,
    agg: str = "mean",
) -> Dict[str, float]:
    """Run an opinion-feature extractor across all L layers and aggregate."""
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