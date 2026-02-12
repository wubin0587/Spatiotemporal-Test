"""
D:\Tiktok\models\events\generate\dist\time.py

Time Distribution Utilities
---------------------------
This module handles the temporal stochasticity of events. It provides standalone 
functions to:
1. Determine the probability of an event triggering at a specific time step.
2. Sample the lifecycle parameters (e.g., decay rate, duration) for a new event.

It is designed to be called by event generators (e.g., `exp.py`).
"""

import numpy as np
from typing import Dict, Any, Union

# =========================================================================
# 1. Trigger Logic: Determining "IF" an event happens now
# =========================================================================

def calculate_trigger_probability(t: float, dist_type: str, config: Dict[str, Any]) -> float:
    """
    Calculates the probability/intensity of an event triggering at current time t.
    Used to simulate different temporal patterns of external shocks.

    Args:
        t (float): Current simulation time step.
        dist_type (str): The distribution key. Options:
                         - 'poisson' (Constant rate)
                         - 'normal' (Peak event like an election)
                         - 'linear' (Rising/Falling tension)
                         - 'cyclic' (Day/Night cycles)
                         - 'burst' (Step function window)
        config (dict): Parameters specific to the chosen distribution.

    Returns:
        float: A probability value [0, 1] representing the chance of a trigger.
    """
    mode = dist_type.lower()
    
    # --- Case A: Poisson Process (Constant Rate) ---
    if mode == 'poisson' or mode == 'uniform':
        # Simple constant probability per tick.
        # Config: {'lambda_rate': 0.05}
        return config.get('lambda_rate', 0.05)
        
    # --- Case B: Normal / Gaussian (Event clustering around a specific date) ---
    elif mode == 'normal':
        # Config: {'mu': 50, 'sigma': 10, 'scale_factor': 0.5}
        # Formula: scale * exp( -0.5 * ((t - mu) / sigma)^2 )
        mu = config.get('mu', 50.0)
        sigma = config.get('sigma', 10.0)
        scale = config.get('scale_factor', 0.1) # Max probability at peak
        
        if sigma <= 0: return 0.0
        prob = scale * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        return float(prob)
        
    # --- Case C: Linear Trend (Rising or Falling Tension) ---
    elif mode == 'linear':
        # Config: {'slope': 0.001, 'intercept': 0.01}
        # Formula: y = mx + c
        slope = config.get('slope', 0.0)
        intercept = config.get('intercept', 0.01)
        prob = slope * t + intercept
        return max(0.0, min(1.0, prob))
    
    # --- Case D: Cyclic (Seasonality / Day-Night) ---
    elif mode == 'cyclic':
        # Config: {'period': 24, 'amplitude': 0.05, 'base_rate': 0.05, 'phase': 0}
        # Formula: base + amp * sin(2*pi*t/period + phase)
        period = config.get('period', 24.0)
        amp = config.get('amplitude', 0.02)
        base = config.get('base_rate', 0.05)
        phase = config.get('phase', 0.0)
        
        prob = base + amp * np.sin(2 * np.pi * t / period + phase)
        return max(0.0, min(1.0, prob))

    # --- Case E: Burst Window (Step Function) ---
    elif mode == 'burst':
        # Config: {'start': 40, 'end': 60, 'prob': 0.8}
        start = config.get('start', 0)
        end = config.get('end', 100)
        val = config.get('prob', 0.5)
        return val if start <= t <= end else 0.0
        
    else:
        # Safety fallback
        return 0.0


# =========================================================================
# 2. Lifecycle Logic: Determining "HOW LONG" an event lasts
# =========================================================================

def sample_lifecycle_params(rng: np.random.Generator, dist_type: str, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Generates the parameters that define the event's temporal decay curve.
    These parameters are stored in the Event object and used by the Engine.

    Args:
        rng (np.random.Generator): Random number generator for reproducibility.
        dist_type (str): Logic for sampling parameters. Options:
                         - 'constant': Fixed parameters for all events.
                         - 'uniform': Random duration within a range.
                         - 'bimodal': Mixture of fast (flash) and slow (lingering) events.
        config (dict): Configuration parameters.

    Returns:
        Dict[str, float]: A dictionary of params, e.g., {'sigma': 5.0, 'peak_delay': 2.0}.
                          The keys depend on the decay function used in the Engine.
                          Assuming 'Log-normal' or 'Gamma' style decay in Engine.
    """
    mode = dist_type.lower()
    
    # Note: 'sigma' usually controls the width/duration of the event curve.
    #       'mu' or 'peak_delay' controls how long it takes to reach peak intensity.

    # --- Case A: Constant (Homogeneous Events) ---
    if mode == 'constant':
        # All events have the exact same duration profile
        return {
            'sigma': config.get('static_sigma', 10.0),
            'mu': config.get('static_mu', 0.0) 
        }

    # --- Case B: Uniform Random (Heterogeneous Events) ---
    elif mode == 'uniform':
        # Random duration between min and max
        min_s = config.get('min_sigma', 1.0)
        max_s = config.get('max_sigma', 20.0)
        
        return {
            'sigma': rng.uniform(min_s, max_s),
            'mu': config.get('static_mu', 0.0)
        }

    # --- Case C: Bimodal / Mixture (Fast vs. Slow) ---
    # *Highly recommended for social media simulations*
    # Simulates "Flash Mobs" (short, sharp) vs "Social Movements" (long, sustained).
    elif mode == 'bimodal' or mode == 'fast_vs_slow':
        # Config: {'fast_prob': 0.8, 'fast_range': [1, 5], 'slow_range': [20, 50]}
        fast_prob = config.get('fast_prob', 0.8)
        
        if rng.random() < fast_prob:
            # Fast event
            f_range = config.get('fast_range', [1.0, 5.0])
            sigma = rng.uniform(f_range[0], f_range[1])
            mu = 0.0 # Fast events usually peak immediately
        else:
            # Slow event
            s_range = config.get('slow_range', [20.0, 50.0])
            sigma = rng.uniform(s_range[0], s_range[1])
            mu = rng.uniform(0, 5.0) # Slow events might take time to peak
            
        return {'sigma': sigma, 'mu': mu}

    # --- Case D: Correlated with Intensity (Configured externally usually) ---
    # Sometimes we want: Big Intensity = Long Duration.
    # Since this function doesn't take intensity as input, we simulate it via LogNormal
    # which has a heavy tail similar to Pareto intensity.
    elif mode == 'heavy_tail':
        # Lognormal distribution for duration
        mean = config.get('log_mean', 2.0) # e.g. e^2 approx 7.4
        std = config.get('log_std', 0.5)
        
        return {
            'sigma': rng.lognormal(mean, std),
            'mu': 0.0
        }

    else:
        # Default fallback
        return {'sigma': 5.0, 'mu': 0.0}