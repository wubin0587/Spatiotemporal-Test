"""
analysis/feature/multi_run.py

Multi-run aggregation for repeated simulations under identical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _flatten_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a nested summary dict to dot-notation scalar keys."""
    flat: Dict[str, float] = {}

    def _walk(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for key, val in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                _walk(val, full_key)
        elif isinstance(obj, (int, float, np.integer, np.floating)):
            flat[prefix] = float(obj)

    _walk(summary)
    return flat


def _unflatten_summary(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dot-notation keys back to nested dict structure."""
    nested: Dict[str, Any] = {}
    for flat_key, val in flat.items():
        parts = flat_key.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return nested


@dataclass
class MultiRunResult:
    """Container for aggregated features across repeated runs."""

    run_summaries: List[Dict[str, Any]]
    run_finals: List[Dict[str, Any]]
    n_runs: int
    layer_idx: int = 0

    mean_summary: Dict[str, Any] = field(default_factory=dict)
    std_summary: Dict[str, Any] = field(default_factory=dict)
    cv_summary: Dict[str, Any] = field(default_factory=dict)
    ci95_summary: Dict[str, Any] = field(default_factory=dict)
    consensus_score: Dict[str, Any] = field(default_factory=dict)
    percentile_summary: Dict[str, Any] = field(default_factory=dict)



def aggregate_runs(
    run_results: List[Dict[str, Any]],
    percentiles: Sequence[float] = (5.0, 25.0, 50.0, 75.0, 95.0),
    layer_idx: int = 0,
) -> MultiRunResult:
    """Aggregate N `FeaturePipeline.run()` outputs at summary level."""
    if not run_results:
        return MultiRunResult(run_summaries=[], run_finals=[], n_runs=0, layer_idx=layer_idx)

    run_summaries = [res.get("summary", {}) for res in run_results]
    run_finals = [res.get("final", {}) for res in run_results]
    flat_summaries = [_flatten_summary(s) for s in run_summaries]

    all_keys = sorted({k for fs in flat_summaries for k in fs})

    mean_s: Dict[str, float] = {}
    std_s: Dict[str, float] = {}
    cv_s: Dict[str, float] = {}
    ci95_s: Dict[str, Tuple[float, float]] = {}
    consensus_s: Dict[str, float] = {}
    pctl_s: Dict[str, Dict[str, float]] = {}

    for key in all_keys:
        vals = np.array([fs.get(key, np.nan) for fs in flat_summaries], dtype=float)
        valid = vals[~np.isnan(vals)]
        n = len(valid)
        if n == 0:
            continue

        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid, ddof=1)) if n > 1 else 0.0
        se = std_val / np.sqrt(n) if n > 0 else 0.0

        mean_s[key] = mean_val
        std_s[key] = std_val
        ci95_s[key] = (mean_val - 1.96 * se, mean_val + 1.96 * se)

        cv_val = std_val / abs(mean_val) if abs(mean_val) > 1e-9 else 0.0
        cv_s[key] = float(cv_val)
        consensus_s[key] = float(max(0.0, 1.0 - cv_val))

        pctl_s[key] = {
            f"p{int(p)}": float(np.percentile(valid, p))
            for p in percentiles
        }

    return MultiRunResult(
        run_summaries=run_summaries,
        run_finals=run_finals,
        n_runs=len(run_results),
        layer_idx=layer_idx,
        mean_summary=_unflatten_summary(mean_s),
        std_summary=_unflatten_summary(std_s),
        cv_summary=_unflatten_summary(cv_s),
        ci95_summary=_unflatten_summary(ci95_s),
        consensus_score=_unflatten_summary(consensus_s),
        percentile_summary=_unflatten_summary(pctl_s),
    )
