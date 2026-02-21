"""Opinion-space metrics.

All metrics are exposed as pure functions for external calls.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _to_1d_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Input opinions cannot be empty.")
    return arr


def mean_opinion(opinions: Sequence[float] | np.ndarray) -> float:
    """Return the mean opinion."""
    x = _to_1d_array(opinions)
    return float(np.mean(x))


def opinion_variance(opinions: Sequence[float] | np.ndarray, ddof: int = 0) -> float:
    """Return opinion variance."""
    x = _to_1d_array(opinions)
    return float(np.var(x, ddof=ddof))


def polarization_std(opinions: Sequence[float] | np.ndarray) -> float:
    """A simple polarization proxy: standard deviation of opinions."""
    x = _to_1d_array(opinions)
    return float(np.std(x))


def extreme_share(opinions: Sequence[float] | np.ndarray, threshold: float = 0.8) -> float:
    """Share of agents with |opinion| >= threshold."""
    x = _to_1d_array(opinions)
    return float(np.mean(np.abs(x) >= threshold))


def bimodality_coefficient(opinions: Sequence[float] | np.ndarray) -> float:
    """Compute the sample bimodality coefficient.

    BC = (gamma^2 + 1) / kappa
    where gamma is skewness and kappa is Pearson kurtosis.
    Values above ~0.555 are often treated as evidence of bimodality.
    """
    x = _to_1d_array(opinions)
    if x.size < 4:
        return 0.0

    centered = x - np.mean(x)
    m2 = np.mean(centered ** 2)
    if m2 <= 1e-12:
        return 0.0

    m3 = np.mean(centered ** 3)
    m4 = np.mean(centered ** 4)
    skewness = m3 / (m2 ** 1.5)
    kurtosis = m4 / (m2 ** 2)
    if kurtosis <= 1e-12:
        return 0.0

    return float((skewness ** 2 + 1.0) / kurtosis)


def opinion_entropy(opinions: Sequence[float] | np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of binned opinions."""
    x = _to_1d_array(opinions)
    hist, _ = np.histogram(x, bins=bins, density=False)
    probs = hist.astype(float) / np.sum(hist)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def edge_homophily_score(
    opinions: Sequence[float] | np.ndarray,
    edges: Iterable[tuple[int, int]],
    opinion_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Compute edge-wise homophily score in [0, 1].

    score_ij = 1 - |o_i - o_j| / range_width
    """
    x = _to_1d_array(opinions)
    width = float(opinion_range[1] - opinion_range[0])
    if width <= 0:
        raise ValueError("Invalid opinion_range: max must be greater than min.")

    edge_list = list(edges)
    if not edge_list:
        return 0.0

    scores = []
    for i, j in edge_list:
        if i < 0 or j < 0 or i >= x.size or j >= x.size:
            continue
        dist = abs(x[i] - x[j])
        score = max(0.0, 1.0 - dist / width)
        scores.append(score)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def homophilous_bimodality_index(
    opinions: Sequence[float] | np.ndarray,
    edges: Iterable[tuple[int, int]],
    alpha: float = 0.5,
    opinion_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Homophilous Bimodality Index, HBI.

    HBI = alpha * normalized_bimodality + (1 - alpha) * homophily

    - normalized_bimodality: min(BC / 0.555, 1)
    - homophily: average edge opinion similarity in [0, 1]
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    bc = bimodality_coefficient(opinions)
    bc_norm = min(max(bc / 0.555, 0.0), 1.0)
    hom = edge_homophily_score(opinions, edges, opinion_range=opinion_range)
    return float(alpha * bc_norm + (1.0 - alpha) * hom)
