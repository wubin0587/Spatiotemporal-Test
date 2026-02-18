"""
models/events/archive/query.py

Vectorized Query Logic for Event Archive
----------------------------------------
This module provides pure functions to filter and search through event data.
It operates directly on the NumPy arrays provided by `EventVectorArchive.get_vectors()`.

Design Philosophy:
- Input: Raw NumPy arrays (Times, Locations, Intensities, etc.)
- Output: NumPy arrays of INDICES (integers) that point to the matching events.
- No Side Effects: These functions do not modify the input arrays.

Performance:
- Uses O(1) vector masking instead of loops.
- Suitable for filtering 10,000+ events per tick in real-time.
"""

import numpy as np
from typing import Optional, Tuple

# =========================================================================
# 1. Temporal Queries (Time-based filtering)
# =========================================================================

def filter_time_window(times: np.ndarray, 
                       current_time: float, 
                       lookback_window: float) -> np.ndarray:
    """
    Finds events that occurred within the recent time window.
    Logic: (current_time - lookback_window) <= event_time <= current_time

    Args:
        times (np.ndarray): The 'times' array from the archive. Shape (N,).
        current_time (float): The current simulation tick.
        lookback_window (float): How far back to check (e.g., max event lifecycle).

    Returns:
        np.ndarray: Array of indices satisfying the condition.
    """
    if times.size == 0:
        return np.array([], dtype=int)

    # Vectorized boolean mask
    mask = (times <= current_time) & (times >= (current_time - lookback_window))
    
    # Return indices where mask is True
    return np.where(mask)[0]


def filter_future_events(times: np.ndarray, current_time: float) -> np.ndarray:
    """
    Finds events that are scheduled for the future (if simulation allows pre-scheduling).
    """
    if times.size == 0:
        return np.array([], dtype=int)
    return np.where(times > current_time)[0]


# =========================================================================
# 2. Spatial Queries (Location-based filtering)
# =========================================================================

def filter_spatial_radius(locs: np.ndarray, 
                          center: np.ndarray, 
                          radius: float) -> np.ndarray:
    """
    Finds events within a circular radius of a target point.
    
    Args:
        locs (np.ndarray): The 'locs' array. Shape (N, 2).
        center (np.ndarray): Target coordinate [x, y].
        radius (float): Search radius.

    Returns:
        np.ndarray: Array of indices.
    """
    if locs.size == 0:
        return np.array([], dtype=int)

    # Calculate Euclidean distance for all points at once
    # (N, 2) - (2,) -> Broadcast subtraction
    deltas = locs - center
    dists_sq = np.sum(deltas ** 2, axis=1) # Avoid sqrt for speed if possible, but radius is linear
    
    mask = dists_sq <= (radius ** 2)
    return np.where(mask)[0]


def filter_spatial_box(locs: np.ndarray, 
                       bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Finds events within a rectangular bounding box.
    Much faster than radius search (no exponentiation).
    
    Args:
        locs (np.ndarray): The 'locs' array. Shape (N, 2).
        bounds (tuple): (min_x, max_x, min_y, max_y)

    Returns:
        np.ndarray: Array of indices.
    """
    if locs.size == 0:
        return np.array([], dtype=int)

    min_x, max_x, min_y, max_y = bounds
    
    # Logic: x >= min_x AND x <= max_x AND y >= min_y AND y <= max_y
    mask = (
        (locs[:, 0] >= min_x) & 
        (locs[:, 0] <= max_x) & 
        (locs[:, 1] >= min_y) & 
        (locs[:, 1] <= max_y)
    )
    return np.where(mask)[0]


def get_k_nearest(locs: np.ndarray, 
                  center: np.ndarray, 
                  k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the K nearest events to a point.
    
    Args:
        locs (np.ndarray): Shape (N, 2).
        center (np.ndarray): [x, y].
        k (int): Number of neighbors to find.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - indices: Indices of the k nearest events.
            - distances: Distances to those events.
    """
    if locs.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # 1. Calculate all distances
    dists = np.linalg.norm(locs - center, axis=1)
    
    # 2. Sort and slice
    # argpartition is faster than argsort because it doesn't sort the whole array,
    # it just guarantees the first k are the smallest.
    if k >= len(locs):
        k = len(locs) - 1
        
    k_indices = np.argpartition(dists, k)[:k]
    
    # 3. Sort the small result set for correctness (argpartition doesn't sort within k)
    sorted_k_indices = k_indices[np.argsort(dists[k_indices])]
    
    return sorted_k_indices, dists[sorted_k_indices]


# =========================================================================
# 3. Attribute Queries (Intensity/Polarity)
# =========================================================================

def filter_by_intensity(intensities: np.ndarray, min_val: float) -> np.ndarray:
    """
    Finds major events with intensity above a threshold.
    """
    if intensities.size == 0:
        return np.array([], dtype=int)
    return np.where(intensities >= min_val)[0]


def filter_by_polarity(polarities: np.ndarray, 
                       direction: str = 'positive') -> np.ndarray:
    """
    Finds polarizing events.
    
    Args:
        polarities (np.ndarray): Shape (N,).
        direction (str): 'positive' (>0), 'negative' (<0), or 'extreme' (abs > 0.8)
    """
    if polarities.size == 0:
        return np.array([], dtype=int)

    if direction == 'positive':
        return np.where(polarities > 0)[0]
    elif direction == 'negative':
        return np.where(polarities < 0)[0]
    elif direction == 'extreme':
        return np.where(np.abs(polarities) >= 0.8)[0]
    else:
        return np.arange(len(polarities))


# =========================================================================
# 4. Composite Queries (Intersection)
# =========================================================================

def intersect_indices(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """
    Helper to find the intersection of two query results.
    E.g., "Active events" AND "Nearby events".
    
    Uses numpy's intersect1d (assume sorted for speed if possible, but robust here).
    """
    return np.intersect1d(idx_a, idx_b, assume_unique=True)