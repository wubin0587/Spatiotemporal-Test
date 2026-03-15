# -*- coding: utf-8 -*-
# FJ-EM: Friedkin-Johnsen model with event modulation.
# Each agent has a fixed stubbornness s_i and an anchor opinion x0_i.
# High impact erodes stubbornness, making agents easier to persuade.

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
    Friedkin-Johnsen model with impact modulation.

    Modulation:
        s_eff(i) = clip(s_i * (1 - eta_mod * I_i), 0, 1)

    Update rule:
        nbr_mean(i) = mean(x_j for j in pairs(i))
        target_i    = s_eff * x0_i + (1 - s_eff) * nbr_mean
        delta_x_i  += mu_base * (target_i - x_i)

    Config keys: eta_mod, mu_base.
    agent_data required keys:
        'stubborn' : (N,) stubbornness values s_i in [0, 1].
        'anchor'   : (N, L) immutable anchor opinions x0_i.
    Falls back to s=0.5 and x0=current X if agent_data not provided.
    """
    N, L = X.shape
    delta_X = np.zeros_like(X)

    eta_mod = params.get('eta_mod', 0.3)
    mu_base = params.get('mu_base', 1.0)

    if agent_data is not None:
        stubborn = agent_data['stubborn']   # (N,)
        anchor   = agent_data['anchor']     # (N, L)
    else:
        stubborn = np.full(N, 0.5)
        anchor   = X.copy()

    neighbours_of: dict = defaultdict(list)
    for i, j in pairs:
        neighbours_of[i].append(j)

    for i, nbrs in neighbours_of.items():
        if not nbrs:
            continue

        s_eff = float(np.clip(
            stubborn[i] * (1.0 - eta_mod * float(impact_vector[i])),
            0.0, 1.0
        ))
        nbr_mean = np.mean([X[j] for j in nbrs], axis=0)
        target   = s_eff * anchor[i] + (1.0 - s_eff) * nbr_mean
        delta_X[i] += mu_base * (target - X[i])

    return delta_X