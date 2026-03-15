# -*- coding: utf-8 -*-
# DW-EM: Deffuant-Weisbuch bounded confidence model with event modulation.
# Refactored from the original monolithic dynamics.py — logic unchanged,
# signature extended to match the unified interface, repulsion_rate promoted
# from hardcoded 0.01 to a configurable param.

import numpy as np
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
    DW bounded confidence with impact modulation.

    Modulation:
        epsilon_eff(i) = epsilon_base + alpha_mod * I_i
        mu_eff(i)      = clip(mu_base + beta_mod * I_i, 0, 0.5)

    Convergence rule:
        if ||x_i - x_j|| < epsilon_eff(i):
            delta_x_i += mu_eff(i) * (x_j - x_i)

    Backfire (optional):
        if backfire=True and I_i > 0.5 and distance >= epsilon_eff(i):
            delta_x_i -= repulsion_rate * (x_j - x_i)/||...|| * I_i

    Config keys: epsilon_base, mu_base, alpha_mod, beta_mod,
                 backfire (bool), repulsion_rate.
    """
    N, L = X.shape
    delta_X = np.zeros_like(X)

    eps_base        = params.get('epsilon_base', 0.2)
    mu_base         = params.get('mu_base', 0.3)
    alpha_mod       = params.get('alpha_mod', 0.0)
    beta_mod        = params.get('beta_mod', 0.0)
    enable_backfire = params.get('backfire', False)
    repulsion_rate  = params.get('repulsion_rate', 0.01)

    epsilon_eff = eps_base + alpha_mod * impact_vector          # (N,)
    mu_eff      = np.clip(mu_base + beta_mod * impact_vector,
                          0.0, 0.5)                             # (N,)

    for i, j in pairs:
        diff = X[j] - X[i]
        dist = np.linalg.norm(diff)

        if dist < epsilon_eff[i]:
            delta_X[i] += mu_eff[i] * diff
        elif enable_backfire and impact_vector[i] > 0.5 and dist > 1e-9:
            delta_X[i] -= repulsion_rate * (diff / dist) * impact_vector[i]

    return delta_X