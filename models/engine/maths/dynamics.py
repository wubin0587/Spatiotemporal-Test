# -*- coding: utf-8 -*-
"""
@File    : dynamics.py
@Desc    : The numerical solver for Opinion Dynamics.
           Implements the Bounded Confidence Model with Parameter Modulation.
"""

import numpy as np

def calculate_opinion_change(X: np.ndarray, pairs: list, 
                             impact_vector: np.ndarray, params: dict) -> np.ndarray:
    """
    Computes the change in opinions (Delta X) for the current step.
    
    Mechanism:
        1. Modulate parameters based on Impact I(t):
           epsilon_eff = epsilon_base + alpha_mod * I(t)  (Trust openness)
           mu_eff      = mu_base + beta_mod * I(t)        (Learning speed)
        2. Apply Bounded Confidence Rule:
           If |x_i - x_j| < epsilon_eff:
               x_i += mu_eff * (x_j - x_i)

    ----------------------------------------------------------------
    Config Schema (dynamics_params):
    ----------------------------------------------------------------
    dynamics_params:
      epsilon_base: 0.2     # Standard trust threshold
      mu_base: 0.3          # Standard learning rate
      alpha_mod: 0.2        # How much impact expands trust (0.2 + 0.2*1.0 = 0.4 max)
      beta_mod: 0.1         # How much impact speeds up learning
      backfire: false       # (Optional) If true, repels when dist > epsilon & impact is high
    ----------------------------------------------------------------

    Args:
        X (np.ndarray): Opinion Matrix (N, L).
        pairs (list[tuple]): Interaction pairs [(i, j), ...].
        impact_vector (np.ndarray): Impact values (N,).
        params (dict): Dynamics configuration.

    Returns:
        np.ndarray: Delta X matrix (N, L), representing the shift in opinions.
    """
    N, L = X.shape
    delta_X = np.zeros_like(X)
    
    # Extract params
    eps_base = params.get('epsilon_base', 0.2)
    mu_base = params.get('mu_base', 0.3)
    alpha_mod = params.get('alpha_mod', 0.0)
    beta_mod = params.get('beta_mod', 0.0)
    enable_backfire = params.get('backfire', False)

    # Pre-calculate effective parameters for all agents
    # This vectorization avoids recalculating inside the loop
    # shape: (N,)
    epsilon_eff = eps_base + (alpha_mod * impact_vector)
    mu_eff = mu_base + (beta_mod * impact_vector)
    
    # Clamp mu to [0, 0.5] to ensure numerical stability
    mu_eff = np.clip(mu_eff, 0.0, 0.5)

    # Process interactions
    # Note: In a synchronous update, we calculate all deltas based on X(t)
    for i, j in pairs:
        xi = X[i]
        xj = X[j]
        
        # Calculate Euclidean distance in opinion space
        # shape: scalar
        op_dist = np.linalg.norm(xi - xj)
        
        # --- Standard Convergence (Bounded Confidence) ---
        # We use agent i's effective epsilon (subjective perception)
        if op_dist < epsilon_eff[i]:
            # Pull i towards j
            change = mu_eff[i] * (xj - xi)
            delta_X[i] += change
            
        # --- Backfire Effect (Polarization) ---
        # If enabled: High impact + Large distance => Repulsion
        elif enable_backfire:
            # Trigger condition: High impact (>0.5) AND Interaction failed (Outside confidence)
            if impact_vector[i] > 0.5:
                # Push i away from j
                # Small fixed repulsion rate (e.g., 0.05) or proportional
                repulsion_dir = (xi - xj) 
                norm_dir = np.linalg.norm(repulsion_dir)
                if norm_dir > 1e-9:
                    repulsion_dir /= norm_dir
                    
                delta_X[i] += 0.01 * repulsion_dir * impact_vector[i]

    return delta_X