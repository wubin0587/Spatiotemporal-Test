"""
models/events/utils/spatial.py

Spatial Dynamics Utilities
--------------------------
Pure functions to calculate the spatial weight based on location.
Input: Vectorized locations and shape parameters.
Output: A weight factor in [0, 1].
"""

import numpy as np

def decay_euclidean_gaussian(event_loc: np.ndarray, 
                             agent_locs: np.ndarray, 
                             params: dict) -> np.ndarray:
    """
    Standard Gaussian Decay over Euclidean distance.
    Formula: e^(-dist^2 / (2 * sigma^2))
    
    Args:
        event_loc: Shape (2,) - The event center.
        agent_locs: Shape (N, 2) - Agent locations.
        params: {'sigma': 0.1}
    """
    sigma = params.get('sigma', 0.1)
    
    # Vectorized distance calculation
    dists_sq = np.sum((agent_locs - event_loc)**2, axis=1)
    
    return np.exp(-dists_sq / (2 * sigma**2))

def decay_manhattan_linear(event_loc: np.ndarray, 
                           agent_locs: np.ndarray, 
                           params: dict) -> np.ndarray:
    """
    Linear decay over Manhattan distance (L1 norm).
    Suitable for: Grid-like cities.
    """
    radius = params.get('radius', 0.2)
    
    # L1 distance: |x1-x2| + |y1-y2|
    dists = np.sum(np.abs(agent_locs - event_loc), axis=1)
    
    # Linear drop: 1 at center, 0 at radius
    weight = 1.0 - (dists / radius)
    return np.maximum(weight, 0.0)

def decay_anisotropic(event_loc: np.ndarray, 
                      agent_locs: np.ndarray, 
                      params: dict) -> np.ndarray:
    """
    Anisotropic Gaussian Decay (Elliptical influence).
    The influence spreads further in one direction than the other.
    
    Args:
        params: {
            'sigma_x': 0.2, 
            'sigma_y': 0.05, 
            'theta': 0.0 (rotation angle in radians)
        }
    """
    sx = params.get('sigma_x', 0.1)
    sy = params.get('sigma_y', 0.1)
    theta = params.get('theta', 0.0)
    
    dx = agent_locs[:, 0] - event_loc[0]
    dy = agent_locs[:, 1] - event_loc[1]
    
    # Rotate coordinates
    # x' = x cos(theta) + y sin(theta)
    # y' = -x sin(theta) + y cos(theta)
    dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
    dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)
    
    # Elliptical Gaussian formula
    exponent = -0.5 * ((dx_rot**2 / sx**2) + (dy_rot**2 / sy**2))
    return np.exp(exponent)