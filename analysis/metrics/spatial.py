"""Spatial metrics based on 2D coordinates and optional attributes."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_xy(coords: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    if arr.shape[0] == 0:
        raise ValueError("coords cannot be empty")
    return arr


def centroid(coords: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    xy = _to_xy(coords)
    return np.mean(xy, axis=0)


def radius_of_gyration(coords: Sequence[Sequence[float]] | np.ndarray) -> float:
    xy = _to_xy(coords)
    c = np.mean(xy, axis=0)
    return float(np.sqrt(np.mean(np.sum((xy - c) ** 2, axis=1))))


def mean_pairwise_distance(coords: Sequence[Sequence[float]] | np.ndarray) -> float:
    xy = _to_xy(coords)
    n = xy.shape[0]
    if n < 2:
        return 0.0
    diffs = xy[:, None, :] - xy[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    return float(np.mean(dists[iu]))


def spatial_entropy(
    coords: Sequence[Sequence[float]] | np.ndarray,
    x_bins: int = 10,
    y_bins: int = 10,
) -> float:
    xy = _to_xy(coords)
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_bins, y_bins])
    probs = hist.reshape(-1)
    probs = probs / np.sum(probs)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def nearest_neighbor_index(
    coords: Sequence[Sequence[float]] | np.ndarray,
    area: float,
) -> float:
    """Clark-Evans nearest-neighbor index.

    NNI = observed_mean_nn / expected_mean_nn
    expected_mean_nn = 0.5 / sqrt(lambda), lambda = n/area
    """
    xy = _to_xy(coords)
    n = xy.shape[0]
    if n < 2 or area <= 0:
        return 0.0

    diffs = xy[:, None, :] - xy[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    observed = np.mean(np.min(dists, axis=1))

    lam = n / area
    expected = 0.5 / np.sqrt(lam)
    if expected <= 1e-12:
        return 0.0
    return float(observed / expected)


def moran_i(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Global Moran's I for spatial autocorrelation."""
    x = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] != x.size:
        raise ValueError("weights must be square matrix with same length as values")

    x_bar = np.mean(x)
    x_dev = x - x_bar
    denom = np.sum(x_dev**2)
    w_sum = np.sum(w)
    if abs(denom) <= 1e-12 or abs(w_sum) <= 1e-12:
        return 0.0

    num = np.sum(w * np.outer(x_dev, x_dev))
    n = x.size
    return float((n / w_sum) * (num / denom))
