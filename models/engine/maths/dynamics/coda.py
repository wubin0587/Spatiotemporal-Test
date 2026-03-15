# -*- coding: utf-8 -*-
# CODA-EM: Continuous Opinions and Discrete Actions with event modulation.
# (Martins 2008) Opinion update fires only when i and j take different actions.
# High impact lowers the action threshold and speeds up Bayesian updates.

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
    CODA model with impact modulation.

    Action decision (per dimension l):
        tau_eff(i) = clip(tau_i - zeta_mod * I_i, 0.1, 0.9)
        action(i, l) = 1 if x_i,l > tau_eff(i) else 0

    Update fires only when action(i, l) != action(j, l):
        rate_eff = update_rate + delta_mod * I_i
        action(j, l) == 1:  delta_x_i,l += rate_eff * (1 - x_i,l)   [up]
        action(j, l) == 0:  delta_x_i,l -= rate_eff * x_i,l          [down]

    Config keys: tau_base, zeta_mod, update_rate, delta_mod.
    agent_data optional key: 'action_threshold' (N,) for per-agent tau.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, L = X.shape
    delta_X = np.zeros_like(X)

    tau_base    = params.get('tau_base', 0.5)
    zeta_mod    = params.get('zeta_mod', 0.1)
    update_rate = params.get('update_rate', 0.1)
    delta_mod   = params.get('delta_mod', 0.05)

    if agent_data is not None and 'action_threshold' in agent_data:
        tau_vec = agent_data['action_threshold'].copy()
    else:
        tau_vec = np.full(N, tau_base)

    for i, j in pairs:
        tau_i = float(np.clip(tau_vec[i] - zeta_mod * float(impact_vector[i]), 0.1, 0.9))
        tau_j = float(np.clip(tau_vec[j] - zeta_mod * float(impact_vector[j]), 0.1, 0.9))
        rate  = update_rate + delta_mod * float(impact_vector[i])

        for l in range(L):
            a_i = 1 if X[i, l] > tau_i else 0
            a_j = 1 if X[j, l] > tau_j else 0
            if a_i == a_j:
                continue
            if a_j == 1:
                delta_X[i, l] += rate * (1.0 - X[i, l])
            else:
                delta_X[i, l] -= rate * X[i, l]

    return delta_X