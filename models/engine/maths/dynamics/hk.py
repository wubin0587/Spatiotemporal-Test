# -*- coding: utf-8 -*-
# HK-EM: Hegselmann-Krause model with event modulation.
# Key difference from DW: agent i aggregates ALL in-confidence neighbours
# simultaneously rather than one random pair per step.

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
    HK bounded confidence with impact modulation.

    pairs are treated as a candidate neighbour set, not pairwise interactions.

    Modulation:
        epsilon_eff(i) = epsilon_base + alpha_mod * I_i
        mu_eff(i)      = clip(mu_base + beta_mod * I_i, 0, 0.5)

    Update rule:
        N_conf(i) = {j in neighbours(i) : ||x_i - x_j|| < epsilon_eff(i)}
        if N_conf non-empty:
            delta_x_i += mu_eff(i) * (mean(x_j for j in N_conf) - x_i)

    Config keys: epsilon_base, mu_base, alpha_mod, beta_mod.
    """
    N, L = X.shape
    delta_X = np.zeros_like(X)

    eps_base  = params.get('epsilon_base', 0.2)
    mu_base   = params.get('mu_base', 0.3)
    alpha_mod = params.get('alpha_mod', 0.0)
    beta_mod  = params.get('beta_mod', 0.0)

    epsilon_eff = eps_base + alpha_mod * impact_vector          # (N,)
    mu_eff      = np.clip(mu_base + beta_mod * impact_vector,
                          0.0, 0.5)                             # (N,)

    # Group candidates by agent i
    neighbours_of: dict = defaultdict(list)
    for i, j in pairs:
        neighbours_of[i].append(j)

    for i, nbrs in neighbours_of.items():
        in_conf = [j for j in nbrs
                   if np.linalg.norm(X[i] - X[j]) < epsilon_eff[i]]
        if not in_conf:
            continue
        # Include self in mean for numerical stability (standard HK variant)
        group = np.vstack([X[j] for j in in_conf] + [X[i]])
        delta_X[i] += mu_eff[i] * (group.mean(axis=0) - X[i])

    return delta_X