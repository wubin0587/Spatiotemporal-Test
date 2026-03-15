# -*- coding: utf-8 -*-
# Ising-EM: Spin-system opinion dynamics with event modulation.
# Event impact acts as an external magnetic field biasing opinion direction,
# while also lowering effective temperature (reducing random flips).

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
    Ising model with impact modulation (continuous-space approximation).

    Opinion-to-spin mapping:
        s_i = 2 * x_i - 1   in [-1, 1]

    Local effective field:
        h_ext(i)   = h_base + kappa_mod * I_i      [external field from events]
        T_eff(i)   = max(0.05, T_base * (1 - tau_mod * I_i))
        h_local(i) = J * mean(s_j for j in neighbours) + h_ext(i)

    Metropolis flip (per dimension):
        delta_E  = -2 * s_i,l * h_local,l
        P_flip   = min(1, exp(-delta_E / T_eff))
        delta_s  = step_size * sign(h_local,l)  if rand < P_flip else 0
        delta_x_i,l = delta_s / 2

    Config keys: temperature, tau_mod, h_base, kappa_mod, coupling, step_size.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, L = X.shape
    delta_X = np.zeros_like(X)

    T_base   = params.get('temperature', 0.5)
    tau_mod  = params.get('tau_mod', 0.3)
    h_base   = params.get('h_base', 0.0)
    kappa    = params.get('kappa_mod', 0.2)
    J        = params.get('coupling', 1.0)
    step     = params.get('step_size', 0.1)

    neighbours_of: dict = defaultdict(list)
    for i, j in pairs:
        neighbours_of[i].append(j)

    for i, nbrs in neighbours_of.items():
        if not nbrs:
            continue

        T_eff  = max(0.05, T_base * (1.0 - tau_mod * float(impact_vector[i])))
        h_ext  = h_base + kappa * float(impact_vector[i])

        s_i    = 2.0 * X[i] - 1.0                                  # (L,)
        s_nbrs = np.array([2.0 * X[j] - 1.0 for j in nbrs])        # (M, L)
        h_local = J * s_nbrs.mean(axis=0) + h_ext                   # (L,)

        delta_E = -2.0 * s_i * h_local                              # (L,)
        accept  = np.where(delta_E <= 0,
                           1.0,
                           np.exp(np.clip(-delta_E / T_eff, -500, 0)))

        flip    = rng.random(L) < accept
        sign_h  = np.sign(h_local)
        sign_h[sign_h == 0] = 1.0

        delta_X[i] += np.where(flip, step * sign_h * 0.5, 0.0)

    return delta_X