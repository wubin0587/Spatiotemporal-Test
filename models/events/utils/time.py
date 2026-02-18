"""
models/events/utils/time.py

Temporal Dynamics Utilities
---------------------------
Pure functions to calculate the temporal weight of an event at a given time.
Input: Time difference (delta_t) and shape parameters.
Output: A weight factor in [0, 1].

Paper Reference:
These functions implement the "Pulse & Decay" dynamics described in Section 3.
"""

import numpy as np

def decay_exponential(delta_t: np.ndarray, params: dict) -> np.ndarray:
    """
    Classic Exponential Decay.
    Suitable for: Breaking news, simple shocks.
    Formula: e^(-lambda * t)
    
    Args:
        delta_t: Time elapsed since event start (must be >= 0).
        params: {'lambda': 0.1}
    """
    lam = params.get('lambda', 0.1)
    # Avoid negative logic just in case
    dt = np.maximum(delta_t, 0)
    return np.exp(-lam * dt)

def decay_log_normal(delta_t: np.ndarray, params: dict) -> np.ndarray:
    """
    Log-Normal Lifecycle (The "Viral Curve").
    Suitable for: Social media topics, rumors.
    Shape: Rises quickly to a peak, then has a long tail.
    
    Args:
        delta_t: Time elapsed.
        params: {'mu': 0.0, 'sigma': 1.0} 
                (mu controls peak time, sigma controls width)
    """
    # Log-normal is undefined for t <= 0, handle safely
    # We add a tiny epsilon to avoid log(0)
    dt = np.maximum(delta_t, 1e-6)
    
    mu = params.get('mu', 0.0)
    sigma = params.get('sigma', 1.0)
    
    # Standard PDF formula (ignoring constant scaling for now, or normalizing to peak=1)
    # Here we implement the shape:
    numerator = np.exp(-(np.log(dt) - mu)**2 / (2 * sigma**2))
    denominator = dt * sigma * np.sqrt(2 * np.pi)
    
    val = numerator / denominator
    
    # Normalize so the peak is roughly 1.0 (optional, makes tuning easier)
    # For simulation, raw PDF is often too small, so we might scale it.
    scale = params.get('scale_factor', 1.0)
    return val * scale

def decay_step_function(delta_t: np.ndarray, params: dict) -> np.ndarray:
    """
    Hard Window / Step Function.
    Suitable for: Policy announcements (valid for N days).
    """
    duration = params.get('duration', 10.0)
    # Returns 1.0 if inside window, 0.0 otherwise
    return np.where((delta_t >= 0) & (delta_t <= duration), 1.0, 0.0)