# -*- coding: utf-8 -*-
"""
@File    : topo.py
@Desc    : Handles neighbor discovery and the "Algorithm vs. Reality" switching logic.
           Determines WHO interacts with WHOM in this time step.
"""

import numpy as np
import random
from scipy.spatial import cKDTree

def get_interaction_pairs(static_adj: list, kd_tree: cKDTree, agent_pos: np.ndarray, 
                          impact_vector: np.ndarray, params: dict) -> list:
    """
    Generates interaction pairs based on agent states (Passive/Active).
    
    Logic:
        1. IF I_i(t) < threshold (Passive):
           - Agent is in "Algorithmic Cocoon".
           - Picks a neighbor from `static_adj` (the social graph).
        2. IF I_i(t) >= threshold (Active):
           - Agent is in "Reality Mode".
           - Searches spatial neighbors using KD-Tree within a dynamic radius.
           - Radius R = R_base + k * I_i(t).

    ----------------------------------------------------------------
    Config Schema (topology_params):
    ----------------------------------------------------------------
    topology_params:
      threshold: 0.2        # Impact value to trigger Reality Mode
      radius_base: 0.05     # Minimum physical search radius
      radius_dynamic: 0.1   # Coefficient for radius expansion based on impact
    ----------------------------------------------------------------

    Args:
        static_adj (list[list]): Adjacency list of the static social network.
        kd_tree (cKDTree): Spatial index of agents.
        agent_pos (np.ndarray): Positions (N, 2).
        impact_vector (np.ndarray): Impact values (N,).
        params (dict): Thresholds and radius parameters.

    Returns:
        list[tuple]: A list of (i, j) integer tuples representing interaction pairs.
    """
    pairs = []
    N = len(static_adj)
    
    threshold = params.get('threshold', 0.2)
    r_base = params.get('radius_base', 0.05)
    r_dyn = params.get('radius_dynamic', 0.1)
    
    # Create a boolean mask for activated agents
    is_active = impact_vector >= threshold
    
    # Get indices for both groups
    active_indices = np.where(is_active)[0]
    passive_indices = np.where(~is_active)[0]

    # --- Case A: Passive Agents (Algorithmic Interactions) ---
    for i in passive_indices:
        neighbors = static_adj[i]
        if neighbors:
            # Randomly select one social connection to interact with
            # (Simulates the algorithm pushing a post from a friend)
            j = random.choice(neighbors)
            pairs.append((i, j))

    # --- Case B: Active Agents (Spatial Interactions) ---
    # They ignore the algorithm and look at "what is happening around them"
    if len(active_indices) > 0:
        # Optimization: We can query in batches, but radius varies per agent.
        # Loop is acceptable here as active_indices is usually a subset of N.
        
        for i in active_indices:
            # Dynamic Radius Calculation
            # Higher impact -> Wider search (panic/news spreads further)
            current_r = r_base + (impact_vector[i] * r_dyn)
            
            # Query KD-Tree
            # k=20 limits max neighbors to check to preserve performance
            dist, idxs = kd_tree.query(agent_pos[i], k=20, distance_upper_bound=current_r)
            
            # Filter valid indices (query returns N for out-of-bounds)
            valid_neighbors = [n for n in idxs if n < N and n != i]
            
            if valid_neighbors:
                # Pick one random spatial neighbor
                j = random.choice(valid_neighbors)
                pairs.append((i, j))
                
    return pairs