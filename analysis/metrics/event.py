"""Event-stream metrics for temporal/spatial event analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def event_rate(event_times: Sequence[float] | np.ndarray, horizon: float | None = None) -> float:
    times = np.asarray(event_times, dtype=float).reshape(-1)
    if times.size == 0:
        return 0.0

    if horizon is None:
        duration = float(np.max(times) - np.min(times))
    else:
        duration = float(horizon)

    if duration <= 1e-12:
        return float(times.size)
    return float(times.size / duration)


def interevent_times(event_times: Sequence[float] | np.ndarray) -> np.ndarray:
    times = np.sort(np.asarray(event_times, dtype=float).reshape(-1))
    if times.size < 2:
        return np.array([], dtype=float)
    return np.diff(times)


def burstiness_index(event_times: Sequence[float] | np.ndarray) -> float:
    """Burstiness by Goh-Barabási: B = (sigma - mu)/(sigma + mu)."""
    dt = interevent_times(event_times)
    if dt.size == 0:
        return 0.0
    mu = np.mean(dt)
    sigma = np.std(dt)
    denom = sigma + mu
    if denom <= 1e-12:
        return 0.0
    return float((sigma - mu) / denom)


def temporal_gini(event_times: Sequence[float] | np.ndarray) -> float:
    dt = interevent_times(event_times)
    if dt.size == 0:
        return 0.0

    x = np.sort(dt)
    n = x.size
    mean_x = np.mean(x)
    if mean_x <= 1e-12:
        return 0.0

    index = np.arange(1, n + 1)
    gini = np.sum((2 * index - n - 1) * x) / (n * np.sum(x))
    return float(gini)


def event_intensity_stats(intensities: Sequence[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(intensities, dtype=float).reshape(-1)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
    }


def event_spatial_spread(event_coords: Sequence[Sequence[float]] | np.ndarray) -> float:
    xy = np.asarray(event_coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("event_coords must be (N,2)")
    if xy.shape[0] == 0:
        return 0.0
    c = np.mean(xy, axis=0)
    return float(np.mean(np.linalg.norm(xy - c, axis=1)))
