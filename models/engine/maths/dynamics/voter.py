# -*- coding: utf-8 -*-
# Voter-EM: Continuous-space voter model with event modulation.
# Core mechanism: probabilistic opinion copying.
# High impact -> stronger herding; low impact -> random drift.

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
    Voter model with impact modulation.

    Modulation:
        p_copy(i) = clip(p_base + gamma_mod * I_i, 0, 1)
        noise_scale = sigma_noise * (1 - I_i)

    Update rule:
        with prob p_copy:  delta_x_i += mu_copy * (x_j - x_i)   [copy]
        otherwise:         delta_x_i ~ N(0, noise_scale)          [drift]

    Config keys: p_base, gamma_mod, mu_copy, sigma_noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, L = X.shape
    delta_X = np.zeros_like(X)

    p_base      = params.get('p_base', 0.3)
    gamma_mod   = params.get('gamma_mod', 0.4)
    mu_copy     = params.get('mu_copy', 1.0)
    sigma_noise = params.get('sigma_noise', 0.02)

    for i, j in pairs:
        p_copy = float(np.clip(p_base + gamma_mod * impact_vector[i], 0.0, 1.0))
        if rng.random() < p_copy:
            delta_X[i] += mu_copy * (X[j] - X[i])
        else:
            scale = sigma_noise * max(0.0, 1.0 - float(impact_vector[i]))
            if scale > 1e-9:
                delta_X[i] += rng.normal(0.0, scale, size=L)

    return delta_X