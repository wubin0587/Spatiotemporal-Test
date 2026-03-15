# -*- coding: utf-8 -*-
# Sznajd-EM: Social validation propagation with event modulation.
# Information flow is reversed: agreeing pair (i, j) persuades third parties,
# NOT i or j themselves.

import numpy as np
from collections import defaultdict
from typing import Optional


def calculate_opinion_change(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator] = None,
    agent_data: Optional[dict] = None,
) -> np.ndarray:
    """
    Sznajd model with impact modulation.

    Modulation:
        delta_agree_eff = delta_agree + alpha_mod * mean(I_i, I_j)
        mu_eff          = mu_persuade + beta_mod * mean(I_i, I_j)

    Update rule:
        if ||x_i - x_j|| < delta_agree_eff:
            joint = (x_i + x_j) / 2
            for k in neighbours(i) union neighbours(j), k != i, k != j:
                delta_x_k += mu_eff * (joint - x_k)

    Config keys: delta_agree, mu_persuade, alpha_mod, beta_mod.
    agent_data required key: 'adjacency' (list[list[int]]) — full social graph
        adjacency list used to find third-party neighbours. Falls back to
        inferring neighbours from pairs if not provided (sparse, less accurate).
    """
    N, L = X.shape
    delta_X = np.zeros_like(X)

    delta_base  = params.get('delta_agree', 0.1)
    alpha_mod   = params.get('alpha_mod', 0.15)
    mu_persuade = params.get('mu_persuade', 0.3)
    beta_mod    = params.get('beta_mod', 0.1)

    full_adj = (agent_data or {}).get('adjacency', None)

    # Build neighbour sets — prefer full adjacency list over pairs inference
    if full_adj is not None:
        neighbours_of = {i: set(full_adj[i]) for i in range(N)}
    else:
        nb: dict = defaultdict(set)
        for i, j in pairs:
            nb[i].add(j)
            nb[j].add(i)
        neighbours_of = nb

    processed: set = set()

    for i, j in pairs:
        key = (min(i, j), max(i, j))
        if key in processed:
            continue
        processed.add(key)

        avg_impact = 0.5 * (float(impact_vector[i]) + float(impact_vector[j]))
        delta_eff  = delta_base + alpha_mod * avg_impact
        mu_eff     = mu_persuade + beta_mod * avg_impact

        if np.linalg.norm(X[i] - X[j]) >= delta_eff:
            continue  # Pair not in agreement — Sznajd mechanism does not fire

        joint         = 0.5 * (X[i] + X[j])
        third_parties = (neighbours_of.get(i, set()) |
                         neighbours_of.get(j, set())) - {i, j}

        for k in third_parties:
            delta_X[k] += mu_eff * (joint - X[k])

    return delta_X