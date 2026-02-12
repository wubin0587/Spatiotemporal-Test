"""
D:\Tiktok\models\events\generate\dist\spatial.py

Spatial Distribution Utilities
------------------------------
This module handles the spatial stochasticity of events. It provides standalone 
functions to:
1. Sample the location coordinates (L) for a new event.
2. Sample the spatial diffusion parameters (e.g., impact radius, sigma).

It allows the simulation to support various geographic scenarios, from 
uniform random events to clustered "hotspot" activities.
"""

import numpy as np
from typing import Dict, Any, List, Union

# =========================================================================
# 1. Location Logic: Determining "WHERE" an event happens
# =========================================================================

def sample_location(rng: np.random.Generator, dist_type: str, config: Dict[str, Any]) -> np.ndarray:
    """
    Generates a 2D coordinate [x, y] based on the specified distribution.
    Assumes a normalized continuous map space [0, 1] x [0, 1].

    Args:
        rng (np.random.Generator): Random number generator.
        dist_type (str): Logic for sampling location. Options:
                         - 'uniform': Random everywhere.
                         - 'center': Gaussian cluster around map center (0.5, 0.5).
                         - 'hotspots': Randomly picks from defined center points.
                         - 'ring': Events occur on the periphery.
        config (dict): Configuration parameters.

    Returns:
        np.ndarray: A numpy array of shape (2,) representing [x, y].
    """
    mode = dist_type.lower()
    
    # --- Case A: Uniform Random (No geography) ---
    if mode == 'uniform':
        # Simple random coordinates in [0, 1]
        return rng.random(size=2)
        
    # --- Case B: Single Center / City Center (Gaussian) ---
    elif mode == 'center' or mode == 'gaussian':
        # Config: {'center_x': 0.5, 'center_y': 0.5, 'scale': 0.15}
        cx = config.get('center_x', 0.5)
        cy = config.get('center_y', 0.5)
        scale = config.get('scale', 0.15) # Standard deviation
        
        loc = rng.normal(loc=[cx, cy], scale=scale, size=2)
        return np.clip(loc, 0.0, 1.0) # Ensure within map boundaries
        
    # --- Case C: Multiple Hotspots (e.g., Business District, University, Slums) ---
    elif mode == 'hotspots':
        # Config: {
        #   'centers': [[0.2, 0.2], [0.8, 0.8], [0.5, 0.5]],
        #   'weights': [0.5, 0.3, 0.2],  # Probability of picking each center
        #   'spread': 0.05
        # }
        centers = config.get('centers', [[0.5, 0.5]])
        weights = config.get('weights', None) # If None, uniform probability
        spread = config.get('spread', 0.05)
        
        # 1. Pick a hotspot index
        idx = rng.choice(len(centers), p=weights)
        chosen_center = np.array(centers[idx])
        
        # 2. Add local noise (gaussian) around that hotspot
        noise = rng.normal(scale=spread, size=2)
        
        loc = chosen_center + noise
        return np.clip(loc, 0.0, 1.0)
        
    # --- Case D: Ring / Periphery (e.g., Suburbs) ---
    elif mode == 'ring':
        # Rejection sampling or polar coordinates to put events away from center
        # Simple Polar method:
        # r ~ Uniform(0.3, 0.5), theta ~ Uniform(0, 2pi)
        r = rng.uniform(config.get('min_r', 0.3), config.get('max_r', 0.5))
        theta = rng.uniform(0, 2 * np.pi)
        
        cx, cy = 0.5, 0.5
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        
        return np.clip([x, y], 0.0, 1.0)
        
    else:
        # Fallback
        return rng.random(size=2)


# =========================================================================
# 2. Diffusion Logic: Determining "HOW BIG" the area of effect is
# =========================================================================

def sample_diffusion_params(rng: np.random.Generator, dist_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decides the Spatial Diffusion characteristics (Range/Sigma) of the event.
    These parameters are stored in the Event object and used by the SpatialEngine.

    Args:
        rng (np.random.Generator): Random number generator.
        dist_type (str): Logic for sampling parameters. Options:
                         - 'constant': Fixed size for all events.
                         - 'uniform': Random size within range.
                         - 'log_normal': Heavy-tailed sizes (some events cover the whole map).
        config (dict): Configuration parameters.

    Returns:
        Dict[str, Any]: A dictionary of params, e.g., {'sigma': 0.15}.
                        The keys must match what the SpatialEngine expects.
    """
    mode = dist_type.lower()
    
    # Note: 'sigma' usually represents the standard deviation of the Gaussian decay.
    #       Larger sigma = Wider area of influence.
    #       sigma=0.1 covers approx 10% of map width effectively.
    #       sigma=1.0 covers entire map.

    # --- Case A: Constant / Fixed Size ---
    if mode == 'constant':
        return {
            'sigma': config.get('static_sigma', 0.1)
        }

    # --- Case B: Uniform Random Range ---
    elif mode == 'uniform':
        low = config.get('min_sigma', 0.05)
        high = config.get('max_sigma', 0.2)
        return {
            'sigma': rng.uniform(low, high)
        }

    # --- Case C: Log-Normal / Heavy Tail (Realistic) ---
    # Most events are localized (neighborhood), a few are global (city-wide).
    elif mode == 'log_normal' or mode == 'heavy_tail':
        # Config: {'log_mean': -2.3, 'log_std': 0.5}
        # e^(-2.3) is approx 0.1.
        mean = config.get('log_mean', -2.3) 
        std = config.get('log_std', 0.5)
        
        val = rng.lognormal(mean, std)
        
        # Clamp to avoid computational absurdity (e.g. sigma > 10 is uselessly huge)
        val = min(val, 2.0) 
        
        return {
            'sigma': val
        }
    
    # --- Case D: Anisotropic (Advanced) ---
    # Placeholder for events that spread differently in X vs Y directions
    # (e.g., along a highway or river).
    elif mode == 'anisotropic':
        # Returns a covariance matrix diagonal or aspect ratio
        base_sigma = config.get('base_sigma', 0.1)
        ratio = rng.uniform(1.0, 5.0) # One axis is much longer
        
        # Randomly decide if it's horizontal or vertical stretch
        if rng.random() < 0.5:
            return {'sigma_x': base_sigma * ratio, 'sigma_y': base_sigma}
        else:
            return {'sigma_x': base_sigma, 'sigma_y': base_sigma * ratio}

    else:
        # Fallback
        return {'sigma': 0.1}