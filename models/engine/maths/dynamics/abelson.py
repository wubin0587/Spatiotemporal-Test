# -*- coding: utf-8 -*-
# Abelson-EM: Abelson (1964) continuous-time social influence model,
# discretised via Euler integration with event modulation.
# Unlike DeGroot, there is no self-weight term — the system converges to
# full consensus on any connected graph, making it a useful reference baseline.

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
    Abelson model with impact modulation (Euler discretisation).

    Continuous-time original:
        dx_i/dt = sum_j a_ij * (x_j - x_i)

    Modulation:
        a_eff(i) = a_base + alpha_mod * I_i   [impact accelerates influence]
        noise    ~ N(0, sigma * (1 - I_i))    [low-impact idle drift]

    Discrete update:
        delta_x_i = a_eff * sum_{j in pairs(i)} (x_j - x_i) * dt
                  + noise

    Contrast with DeGroot: no self-weight means faster and complete convergence;
    useful for isolating the effect of self-weight in comparative experiments.

    Config keys: a_base, alpha_mod, sigma_noise, dt.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, L = X.shape
    delta_X = np.zeros_like(X)

    a_base    = params.get('a_base', 0.3)
    alpha_mod = params.get('alpha_mod', 0.2)
    sigma     = params.get('sigma_noise', 0.01)
    dt        = params.get('dt', 1.0)

    neighbours_of: dict = defaultdict(list)
    for i, j in pairs:
        neighbours_of[i].append(j)

    for i, nbrs in neighbours_of.items():
        if not nbrs:
            continue

        a_eff = a_base + alpha_mod * float(impact_vector[i])
        diff_sum = np.sum([X[j] - X[i] for j in nbrs], axis=0)
        delta_X[i] += a_eff * diff_sum * dt

        noise_scale = sigma * max(0.0, 1.0 - float(impact_vector[i]))
        if noise_scale > 1e-9:
            delta_X[i] += rng.normal(0.0, noise_scale, size=L)

    return delta_X