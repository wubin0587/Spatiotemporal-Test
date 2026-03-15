# -*- coding: utf-8 -*-
# Majority Rule-EM: Galam majority rule with event modulation.
# Agent conforms to local group mean when sufficiently far from it.
# High impact increases conformity pressure.

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
    Galam majority rule with impact modulation.

    Group definition: agent i plus all j paired with i in this step.

    Modulation:
        mu_eff(i) = mu_conform + beta_mod * I_i

    Update rule:
        group_mean = mean(x_j for j in group including i)
        dist = ||x_i - group_mean||
        if dist >= epsilon_majority:
            force = mu_eff * (group_mean - x_i)
            delta_x_i += force * max(0, 1 - resistance * dist)

    resistance > 0 models a stubborn-minority effect (larger deviation -> less yield).

    Config keys: epsilon_majority, mu_conform, beta_mod, resistance.
    """
    N, L = X.shape
    delta_X = np.zeros_like(X)

    eps_maj    = params.get('epsilon_majority', 0.15)
    mu_conform = params.get('mu_conform', 0.3)
    beta_mod   = params.get('beta_mod', 0.1)
    resistance = params.get('resistance', 0.0)

    group_of: dict = defaultdict(list)
    for i, j in pairs:
        group_of[i].append(j)

    for i, members in group_of.items():
        if not members:
            continue

        mu_eff = mu_conform + beta_mod * float(impact_vector[i])
        group_X = np.vstack([X[i]] + [X[m] for m in members])
        group_mean = group_X.mean(axis=0)
        dist = np.linalg.norm(X[i] - group_mean)

        if dist < eps_maj:
            continue

        force = mu_eff * (group_mean - X[i])
        if resistance > 1e-9:
            force *= max(0.0, 1.0 - resistance * dist)
        delta_X[i] += force

    return delta_X