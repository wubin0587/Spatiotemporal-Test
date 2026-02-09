# -*- coding: utf-8 -*-
"""
@File    : field.py
@Desc    : Calculates the Spatiotemporal Impact Field I(t).
           This represents the objective "pressure" of reality on agents.
"""

import numpy as np

def compute_impact_field(agent_pos: np.ndarray, active_events: list, params: dict) -> np.ndarray:
    """
    Computes the cumulative impact score for every agent based on active events.
    
    Mathematical Model:
        I_i(t) = Sum_{e} [ S_e * exp(-alpha * ||p_i - p_e||) * exp(-beta * (t - t_e)) ]

    ----------------------------------------------------------------
    Config Schema (field_params):
    ----------------------------------------------------------------
    field_params:
      alpha: 5.0    # Spatial decay constant (Higher = impact is more local)
      beta: 0.1     # Temporal decay constant (Higher = memory fades faster)
      current_time: <int/float> # Injected by the core engine at runtime
    ----------------------------------------------------------------

    Args:
        agent_pos (np.ndarray): Shape (N, 2). Agent coordinates [0, 1].
        active_events (list): List of dicts, e.g., [{'pos': (x,y), 'time': t, 'intensity': S}, ...]
        params (dict): Dictionary containing 'alpha', 'beta', and 'current_time'.

    Returns:
        np.ndarray: Shape (N,). A vector of float values representing impact I_i(t).
    """
    N = agent_pos.shape[0]
    impact_vector = np.zeros(N, dtype=np.float32)
    
    if not active_events:
        return impact_vector

    alpha = params.get('alpha', 5.0)
    beta = params.get('beta', 0.1)
    t_now = params.get('current_time', 0)

    # Vectorized calculation over agents, iterative over events (Events are sparse, Agents are dense)
    for event in active_events:
        e_pos = np.array(event['pos']) # Shape (2,)
        e_time = event['time']
        e_strength = event['intensity']
        
        # 1. Temporal Decay Component
        dt = t_now - e_time
        if dt < 0: continue # Event hasn't started yet
        
        time_factor = np.exp(-beta * dt)
        
        # Optimization: If event is essentially dead, skip spatial calc
        if time_factor < 1e-6:
            continue

        # 2. Spatial Decay Component
        # Calculate Euclidean distance between all agents and this event
        # agent_pos shape (N, 2), e_pos shape (2,) -> broadcasting works
        dists = np.linalg.norm(agent_pos - e_pos, axis=1)
        
        space_factor = np.exp(-alpha * dists)
        
        # 3. Accumulate Impact
        # I += S * Space * Time
        impact_vector += e_strength * space_factor * time_factor

    return impact_vector