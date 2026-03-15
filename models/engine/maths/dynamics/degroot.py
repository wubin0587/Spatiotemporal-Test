# -*- coding: utf-8 -*-
# DeGroot-EM: Weighted social learning with event modulation.
# High impact erodes self-weight, making agents more receptive to peers.

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
    DeGroot weighted averaging with impact modulation.

    Modulation:
        sw_eff(i) = max(0.1, self_weight_base - gamma_mod * I_i)

    Update rule (uniform neighbour weights unless weight_matrix provided):
        w_nbr = (1 - sw_eff) / |neighbours(i)|
        target_i = sw_eff * x_i + sum(w_nbr * x_j for j in neighbours)
        delta_x_i += step_fraction * (target_i - x_i)

    Config keys: self_weight_base, gamma_mod, step_fraction.
    agent_data optional key: 'weight_matrix' (N, N) for non-uniform weights.
    """
    N, L = X.shape
    delta_X = np.zeros_like(X)

    sw_base        = params.get('self_weight_base', 0.5)
    gamma_mod      = params.get('gamma_mod', 0.2)
    step_fraction  = params.get('step_fraction', 1.0)
    weight_matrix  = (agent_data or {}).get('weight_matrix', None)

    neighbours_of: dict = defaultdict(list)
    for i, j in pairs:
        neighbours_of[i].append(j)

    for i, nbrs in neighbours_of.items():
        if not nbrs:
            continue

        sw = max(0.1, sw_base - gamma_mod * float(impact_vector[i]))

        if weight_matrix is not None:
            row = weight_matrix[i]
            raw_w = np.array([row[j] for j in nbrs], dtype=float)
            total = raw_w.sum()
            nbr_w = (raw_w / total * (1.0 - sw)) if total > 1e-9 else np.full(len(nbrs), (1.0 - sw) / len(nbrs))
        else:
            nbr_w = np.full(len(nbrs), (1.0 - sw) / len(nbrs))

        weighted_nbr = sum(nbr_w[k] * X[nbrs[k]] for k in range(len(nbrs)))
        target = sw * X[i] + weighted_nbr
        delta_X[i] += step_fraction * (target - X[i])

    return delta_X