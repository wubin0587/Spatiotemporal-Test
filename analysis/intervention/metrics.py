"""
analysis/intervention/metrics.py

Intervention Effect Metrics
----------------------------
Pure functions that quantify the causal effect of an intervention by comparing
a "treatment" branch (post-intervention) against a "control" baseline.

All functions operate on FeaturePipeline outputs (dicts with keys
'final', 'timeseries', 'summary') or on raw numpy arrays extracted from them.

Metric families
---------------
1. Opinion-space effects      – shift in mean, polarization, entropy
2. Spatial effects            – centroid displacement, clustering change
3. Temporal effects           – opinion velocity, convergence speed
4. Network-coupling effects   – homophily delta, disagreement delta
5. Aggregate effect score     – weighted composite for ranking policies

All metrics return plain Python floats / dicts for easy JSON serialisation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_ts(pipeline_output: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    """Retrieve a time-series array from pipeline output by dot-key."""
    return pipeline_output.get("timeseries", {}).get(key)


def _get_final(pipeline_output: Dict[str, Any], section: str, metric: str) -> Optional[float]:
    """Retrieve a scalar from the final snapshot."""
    val = pipeline_output.get("final", {}).get(section, {}).get(metric)
    return float(val) if val is not None else None


def _get_summary_mean(pipeline_output: Dict[str, Any], key: str) -> Optional[float]:
    """Retrieve summary mean for a dot-key metric."""
    entry = pipeline_output.get("summary", {}).get(key)
    if isinstance(entry, dict):
        v = entry.get("mean")
        return float(v) if v is not None else None
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. Opinion-space effect metrics
# ═════════════════════════════════════════════════════════════════════════════

def opinion_mean_shift(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Absolute shift in mean opinion: treatment_mean − control_mean.

    Positive → intervention pushed opinions higher (pro-social / pro-consensus).
    Negative → intervention polarised or dampened opinions.
    """
    c = _get_final(control,   "opinion", "mean_opinion") or 0.0
    t = _get_final(treatment, "opinion", "mean_opinion") or 0.0
    return float(t - c)


def polarization_reduction(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Reduction in polarization std: control_std − treatment_std.

    Positive → intervention reduced polarization (convergence effect).
    Negative → intervention amplified polarization.
    """
    c = _get_final(control,   "opinion", "polarization_std") or 0.0
    t = _get_final(treatment, "opinion", "polarization_std") or 0.0
    return float(c - t)


def bimodality_reduction(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Reduction in bimodality coefficient (BC): control_BC − treatment_BC.

    Positive → intervention suppressed two-camp structure.
    """
    c = _get_final(control,   "opinion", "bimodality_coefficient") or 0.0
    t = _get_final(treatment, "opinion", "bimodality_coefficient") or 0.0
    return float(c - t)


def entropy_change(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Change in opinion entropy: treatment_H − control_H.

    Positive → more diverse / mixed distribution.
    Negative → more concentrated / uniform distribution.
    """
    c = _get_final(control,   "opinion", "opinion_entropy") or 0.0
    t = _get_final(treatment, "opinion", "opinion_entropy") or 0.0
    return float(t - c)


def extreme_share_change(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Change in fraction of extreme-opinion holders: treatment − control.

    Negative → intervention moderated extreme agents.
    """
    c = _get_final(control,   "opinion", "extreme_share") or 0.0
    t = _get_final(treatment, "opinion", "extreme_share") or 0.0
    return float(t - c)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Spatial effect metrics
# ═════════════════════════════════════════════════════════════════════════════

def spatial_centroid_displacement(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Euclidean distance between the final population centroids of control and treatment.
    """
    cx_c = _get_final(control,   "spatial", "centroid_x") or 0.5
    cy_c = _get_final(control,   "spatial", "centroid_y") or 0.5
    cx_t = _get_final(treatment, "spatial", "centroid_x") or 0.5
    cy_t = _get_final(treatment, "spatial", "centroid_y") or 0.5
    return float(np.sqrt((cx_t - cx_c) ** 2 + (cy_t - cy_c) ** 2))


def spatial_clustering_change(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Change in nearest-neighbor index (NNI): treatment_NNI − control_NNI.

    Negative → agents became more clustered under treatment.
    Positive → agents became more dispersed.
    """
    c = _get_final(control,   "spatial", "nearest_neighbor_index") or 1.0
    t = _get_final(treatment, "spatial", "nearest_neighbor_index") or 1.0
    return float(t - c)


def moran_i_change(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Change in Moran's I spatial autocorrelation: treatment − control.

    Positive → intervention increased spatial echo-chamber strength.
    Negative → intervention broke up spatially clustered opinion groups.
    """
    c = _get_final(control,   "spatial", "moran_i") or 0.0
    t = _get_final(treatment, "spatial", "moran_i") or 0.0
    return float(t - c)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Temporal effect metrics
# ═════════════════════════════════════════════════════════════════════════════

def convergence_speed_ratio(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
    stability_threshold: float = 0.02,
) -> float:
    """
    Ratio of convergence speed: treatment / control.

    Convergence step is the first step where rolling-std of mean_opinion
    drops below ``stability_threshold``.  Returns > 1 if treatment converges
    faster, < 1 if slower, 0 if no convergence detected in either.

    Uses timeseries 'opinion.mean_opinion' arrays.
    """
    def _convergence_step(ts: np.ndarray, threshold: float) -> Optional[int]:
        if ts is None or len(ts) < 5:
            return None
        window = 5
        for i in range(window, len(ts)):
            if np.std(ts[i - window : i]) < threshold:
                return i
        return None

    ts_c = _get_ts(control,   "opinion.mean_opinion")
    ts_t = _get_ts(treatment, "opinion.mean_opinion")

    step_c = _convergence_step(ts_c, stability_threshold)
    step_t = _convergence_step(ts_t, stability_threshold)

    if step_c is None and step_t is None:
        return 0.0
    if step_c is None:
        return 2.0        # treatment converged but control did not
    if step_t is None:
        return 0.5        # control converged but treatment did not
    return float(step_c / max(step_t, 1))


def opinion_velocity(
    pipeline_output: Dict[str, Any],
    window_ratio: float = 0.1,
) -> float:
    """
    Mean absolute step-to-step change in population mean opinion.

    Captures how rapidly opinion is shifting post-intervention.
    Computed over the last ``window_ratio`` fraction of the time-series.
    """
    ts = _get_ts(pipeline_output, "opinion.mean_opinion")
    if ts is None or len(ts) < 3:
        return 0.0
    n = len(ts)
    w = max(2, int(n * window_ratio))
    tail = ts[-w:]
    return float(np.mean(np.abs(np.diff(tail))))


def opinion_velocity_delta(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Difference in opinion velocity: treatment − control.

    Positive → treatment caused faster late-stage opinion dynamics.
    Negative → treatment stabilised late-stage dynamics.
    """
    return opinion_velocity(treatment) - opinion_velocity(control)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Network-coupling effect metrics
# ═════════════════════════════════════════════════════════════════════════════

def homophily_change(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Change in edge homophily score: treatment − control.

    Positive → intervention created more opinion-similar connections.
    Negative → intervention bridged opinion gaps (cross-cutting ties).
    """
    c = _get_final(control,   "opinion", "edge_homophily_score") or 0.0
    t = _get_final(treatment, "opinion", "edge_homophily_score") or 0.0
    return float(t - c)


def edge_disagreement_change(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Change in mean cross-edge opinion disagreement: treatment − control.

    Positive → more cross-opinion edges (potentially bridging or fragmenting).
    Negative → edges became more homophilous.
    """
    c = _get_final(control,   "network_opinion", "edge_disagreement") or 0.0
    t = _get_final(treatment, "network_opinion", "edge_disagreement") or 0.0
    return float(t - c)


def modularity_change(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
) -> float:
    """
    Change in network modularity: treatment − control.

    Positive → intervention strengthened community structure (echo chambers).
    Negative → intervention weakened community structure (integration).
    """
    c = _get_final(control,   "topo", "modularity") or 0.0
    t = _get_final(treatment, "topo", "modularity") or 0.0
    return float(t - c)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Aggregate intervention effect score
# ═════════════════════════════════════════════════════════════════════════════

def compute_intervention_effect(
    control: Dict[str, Any],
    treatment: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of intervention effect metrics.

    Parameters
    ----------
    control : dict
        FeaturePipeline.run() output for the baseline (no intervention) branch.
    treatment : dict
        FeaturePipeline.run() output for the intervention branch.
    weights : dict, optional
        Relative weights for computing the composite ``effect_score``.
        Default weights are provided (see source).

    Returns
    -------
    dict[str, float]
        All individual delta metrics plus a composite ``effect_score``.
        Positive score = intervention had a net convergence / moderation effect.
        Negative score = intervention had a net polarization / destabilising effect.

    Notes
    -----
    ``effect_score`` is a weighted sum designed so that:
        - polarization_reduction    > 0 → positive contribution
        - extreme_share_change      < 0 → positive contribution (fewer extremists)
        - entropy_change            sign-neutral (diversity may or may not be desired)
        - bimodality_reduction      > 0 → positive contribution
        - convergence_speed_ratio   > 1 → positive contribution (faster convergence)
    """
    _default_weights: Dict[str, float] = {
        "polarization_reduction":   1.5,
        "bimodality_reduction":     1.2,
        "extreme_share_change":    -1.0,   # negative = fewer extremists = good
        "entropy_change":           0.5,
        "opinion_mean_shift":       0.3,
        "moran_i_change":          -0.8,   # negative = less echo-chamber = good
        "modularity_change":       -0.6,
        "convergence_speed_ratio":  0.4,
    }
    w = {**_default_weights, **(weights or {})}

    metrics: Dict[str, float] = {
        "opinion_mean_shift":        opinion_mean_shift(control, treatment),
        "polarization_reduction":    polarization_reduction(control, treatment),
        "bimodality_reduction":      bimodality_reduction(control, treatment),
        "entropy_change":            entropy_change(control, treatment),
        "extreme_share_change":      extreme_share_change(control, treatment),
        "spatial_centroid_disp":     spatial_centroid_displacement(control, treatment),
        "spatial_clustering_change": spatial_clustering_change(control, treatment),
        "moran_i_change":            moran_i_change(control, treatment),
        "homophily_change":          homophily_change(control, treatment),
        "edge_disagreement_change":  edge_disagreement_change(control, treatment),
        "modularity_change":         modularity_change(control, treatment),
        "convergence_speed_ratio":   convergence_speed_ratio(control, treatment),
        "opinion_velocity_delta":    opinion_velocity_delta(control, treatment),
    }

    # Composite score
    score = 0.0
    for key, weight in w.items():
        if key in metrics:
            score += weight * metrics[key]
    metrics["effect_score"] = float(score)

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# 6. Multi-intervention ranking
# ═════════════════════════════════════════════════════════════════════════════

def rank_interventions(
    control: Dict[str, Any],
    treatments: Dict[str, Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    sort_by: str = "effect_score",
) -> List[Dict[str, Any]]:
    """
    Compare multiple intervention branches against a shared control and rank them.

    Parameters
    ----------
    control : dict
        Baseline pipeline output.
    treatments : dict[str, dict]
        Mapping of intervention_label → pipeline output.
    weights : dict, optional
        Custom weights passed to ``compute_intervention_effect``.
    sort_by : str
        Metric key to sort by (default: 'effect_score', descending).

    Returns
    -------
    list[dict]
        Each entry: {'label': str, **metrics}. Sorted by ``sort_by`` descending.
    """
    results = []
    for label, treatment in treatments.items():
        m = compute_intervention_effect(control, treatment, weights=weights)
        m["label"] = label
        results.append(m)

    results.sort(key=lambda r: r.get(sort_by, 0.0), reverse=True)
    return results
