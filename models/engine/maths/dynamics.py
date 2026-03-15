# -*- coding: utf-8 -*-
# Unified opinion-dynamics module.
# All kernels are defined inline; the active kernel is selected via
# params['kernel'] (default: 'dw').
#
# Public API (unchanged):
#   calculate_opinion_change(X, pairs, impact_vector, params,
#                            rng=None, agent_data=None) -> np.ndarray
#
# Supported kernels:
#   'dw'       Deffuant-Weisbuch bounded confidence
#   'hk'       Hegselmann-Krause bounded confidence
#   'voter'    Continuous voter model
#   'degroot'  DeGroot weighted averaging
#   'sznajd'   Sznajd social-validation propagation
#   'majority' Galam majority rule
#   'fj'       Friedkin-Johnsen stubbornness model
#   'coda'     Continuous Opinions & Discrete Actions (Martins 2008)
#   'ising'    Ising spin-system approximation
#   'abelson'  Abelson continuous social influence (Euler)

import numpy as np
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Internal type alias
# ---------------------------------------------------------------------------
_KernelFn = None  # forward-declaration placeholder (not used at runtime)


# ===========================================================================
# Kernel: DW — Deffuant-Weisbuch bounded confidence
# ===========================================================================

def _dw(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
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

    epsilon_eff = eps_base + alpha_mod * impact_vector   # (N,)
    mu_eff      = np.clip(mu_base + beta_mod * impact_vector, 0.0, 0.5)  # (N,)

    for i, j in pairs:
        diff = X[j] - X[i]
        dist = np.linalg.norm(diff)

        if dist < epsilon_eff[i]:
            delta_X[i] += mu_eff[i] * diff
        elif enable_backfire and impact_vector[i] > 0.5 and dist > 1e-9:
            delta_X[i] -= repulsion_rate * (diff / dist) * impact_vector[i]

    return delta_X


# ===========================================================================
# Kernel: HK — Hegselmann-Krause bounded confidence
# ===========================================================================

def _hk(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
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

    epsilon_eff = eps_base + alpha_mod * impact_vector   # (N,)
    mu_eff      = np.clip(mu_base + beta_mod * impact_vector, 0.0, 0.5)  # (N,)

    neighbours_of: dict = defaultdict(list)
    for i, j in pairs:
        neighbours_of[i].append(j)

    for i, nbrs in neighbours_of.items():
        in_conf = [j for j in nbrs
                   if np.linalg.norm(X[i] - X[j]) < epsilon_eff[i]]
        if not in_conf:
            continue
        # Include self in mean (standard HK variant for numerical stability)
        group = np.vstack([X[j] for j in in_conf] + [X[i]])
        delta_X[i] += mu_eff[i] * (group.mean(axis=0) - X[i])

    return delta_X


# ===========================================================================
# Kernel: Voter — continuous-space voter model
# ===========================================================================

def _voter(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
) -> np.ndarray:
    """
    Voter model with impact modulation.

    Modulation:
        p_copy(i)   = clip(p_base + gamma_mod * I_i, 0, 1)
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


# ===========================================================================
# Kernel: DeGroot — weighted social learning
# ===========================================================================

def _degroot(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
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

    sw_base       = params.get('self_weight_base', 0.5)
    gamma_mod     = params.get('gamma_mod', 0.2)
    step_fraction = params.get('step_fraction', 1.0)
    weight_matrix = (agent_data or {}).get('weight_matrix', None)

    neighbours_of: dict = defaultdict(list)
    for i, j in pairs:
        neighbours_of[i].append(j)

    for i, nbrs in neighbours_of.items():
        if not nbrs:
            continue

        sw = max(0.1, sw_base - gamma_mod * float(impact_vector[i]))

        if weight_matrix is not None:
            row   = weight_matrix[i]
            raw_w = np.array([row[j] for j in nbrs], dtype=float)
            total = raw_w.sum()
            nbr_w = (raw_w / total * (1.0 - sw)) if total > 1e-9 \
                    else np.full(len(nbrs), (1.0 - sw) / len(nbrs))
        else:
            nbr_w = np.full(len(nbrs), (1.0 - sw) / len(nbrs))

        weighted_nbr = sum(nbr_w[k] * X[nbrs[k]] for k in range(len(nbrs)))
        target = sw * X[i] + weighted_nbr
        delta_X[i] += step_fraction * (target - X[i])

    return delta_X


# ===========================================================================
# Kernel: Sznajd — social-validation propagation
# ===========================================================================

def _sznajd(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
) -> np.ndarray:
    """
    Sznajd model with impact modulation.

    Information flow is reversed: agreeing pair (i, j) persuades third parties.

    Modulation:
        delta_agree_eff = delta_agree + alpha_mod * mean(I_i, I_j)
        mu_eff          = mu_persuade + beta_mod * mean(I_i, I_j)

    Update rule:
        if ||x_i - x_j|| < delta_agree_eff:
            joint = (x_i + x_j) / 2
            for k in neighbours(i) ∪ neighbours(j), k ∉ {i, j}:
                delta_x_k += mu_eff * (joint - x_k)

    Config keys: delta_agree, mu_persuade, alpha_mod, beta_mod.
    agent_data optional key: 'adjacency' (list[list[int]]) — full graph adj list.
        Falls back to inferring neighbours from pairs if absent.
    """
    N, L = X.shape
    delta_X = np.zeros_like(X)

    delta_base  = params.get('delta_agree', 0.1)
    alpha_mod   = params.get('alpha_mod', 0.15)
    mu_persuade = params.get('mu_persuade', 0.3)
    beta_mod    = params.get('beta_mod', 0.1)

    full_adj = (agent_data or {}).get('adjacency', None)

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
            continue

        joint         = 0.5 * (X[i] + X[j])
        third_parties = (neighbours_of.get(i, set()) |
                         neighbours_of.get(j, set())) - {i, j}

        for k in third_parties:
            delta_X[k] += mu_eff * (joint - X[k])

    return delta_X


# ===========================================================================
# Kernel: Majority — Galam majority rule
# ===========================================================================

def _majority(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
) -> np.ndarray:
    """
    Galam majority rule with impact modulation.

    Group: agent i plus all j paired with i in this step.

    Modulation:
        mu_eff(i) = mu_conform + beta_mod * I_i

    Update rule:
        group_mean = mean(x_j for j in group including i)
        dist = ||x_i - group_mean||
        if dist >= epsilon_majority:
            force = mu_eff * (group_mean - x_i)
            delta_x_i += force * max(0, 1 - resistance * dist)

    resistance > 0 models stubborn-minority effect.

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

        mu_eff     = mu_conform + beta_mod * float(impact_vector[i])
        group_X    = np.vstack([X[i]] + [X[m] for m in members])
        group_mean = group_X.mean(axis=0)
        dist       = np.linalg.norm(X[i] - group_mean)

        if dist < eps_maj:
            continue

        force = mu_eff * (group_mean - X[i])
        if resistance > 1e-9:
            force *= max(0.0, 1.0 - resistance * dist)
        delta_X[i] += force

    return delta_X


# ===========================================================================
# Kernel: FJ — Friedkin-Johnsen stubbornness model
# ===========================================================================

def _fj(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
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
        'stubborn' : (N,)   stubbornness values s_i in [0, 1].
        'anchor'   : (N, L) immutable anchor opinions x0_i.
    Falls back to s=0.5 and x0=current X if agent_data absent.
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


# ===========================================================================
# Kernel: CODA — Continuous Opinions & Discrete Actions (Martins 2008)
# ===========================================================================

def _coda(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
) -> np.ndarray:
    """
    CODA model with impact modulation.

    Opinion update fires only when i and j take different actions.
    High impact lowers the action threshold and speeds up Bayesian updates.

    Action decision (per dimension l):
        tau_eff(i) = clip(tau_i - zeta_mod * I_i, 0.1, 0.9)
        action(i, l) = 1 if x_i,l > tau_eff(i) else 0

    Update fires only when action(i) != action(j):
        rate_eff = update_rate + delta_mod * I_i
        action(j) == 1:  delta_x_i,l += rate_eff * (1 - x_i,l)
        action(j) == 0:  delta_x_i,l -= rate_eff * x_i,l

    Config keys: tau_base, zeta_mod, update_rate, delta_mod.
    agent_data optional key: 'action_threshold' (N,) per-agent tau.
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


# ===========================================================================
# Kernel: Ising — spin-system opinion dynamics
# ===========================================================================

def _ising(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
) -> np.ndarray:
    """
    Ising model with impact modulation (continuous-space approximation).

    Opinion-to-spin mapping:
        s_i = 2 * x_i - 1   in [-1, 1]

    Local effective field:
        h_ext(i)   = h_base + kappa_mod * I_i
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

    T_base  = params.get('temperature', 0.5)
    tau_mod = params.get('tau_mod', 0.3)
    h_base  = params.get('h_base', 0.0)
    kappa   = params.get('kappa_mod', 0.2)
    J       = params.get('coupling', 1.0)
    step    = params.get('step_size', 0.1)

    neighbours_of: dict = defaultdict(list)
    for i, j in pairs:
        neighbours_of[i].append(j)

    for i, nbrs in neighbours_of.items():
        if not nbrs:
            continue

        T_eff  = max(0.05, T_base * (1.0 - tau_mod * float(impact_vector[i])))
        h_ext  = h_base + kappa * float(impact_vector[i])

        s_i    = 2.0 * X[i] - 1.0                                   # (L,)
        s_nbrs = np.array([2.0 * X[j] - 1.0 for j in nbrs])         # (M, L)
        h_local = J * s_nbrs.mean(axis=0) + h_ext                    # (L,)

        delta_E = -2.0 * s_i * h_local                               # (L,)
        accept  = np.where(delta_E <= 0,
                           1.0,
                           np.exp(np.clip(-delta_E / T_eff, -500, 0)))

        flip   = rng.random(L) < accept
        sign_h = np.sign(h_local)
        sign_h[sign_h == 0] = 1.0

        delta_X[i] += np.where(flip, step * sign_h * 0.5, 0.0)

    return delta_X


# ===========================================================================
# Kernel: Abelson — continuous social influence (Euler discretisation)
# ===========================================================================

def _abelson(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator],
    agent_data: Optional[dict],
) -> np.ndarray:
    """
    Abelson model with impact modulation (Euler discretisation).

    Continuous-time original:
        dx_i/dt = sum_j a_ij * (x_j - x_i)

    Unlike DeGroot, there is no self-weight term — the system converges to
    full consensus on any connected graph.

    Modulation:
        a_eff(i) = a_base + alpha_mod * I_i   [impact accelerates influence]
        noise    ~ N(0, sigma * (1 - I_i))    [low-impact idle drift]

    Discrete update:
        delta_x_i = a_eff * sum_{j in pairs(i)} (x_j - x_i) * dt + noise

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

        a_eff    = a_base + alpha_mod * float(impact_vector[i])
        diff_sum = np.sum([X[j] - X[i] for j in nbrs], axis=0)
        delta_X[i] += a_eff * diff_sum * dt

        noise_scale = sigma * max(0.0, 1.0 - float(impact_vector[i]))
        if noise_scale > 1e-9:
            delta_X[i] += rng.normal(0.0, noise_scale, size=L)

    return delta_X


# ===========================================================================
# Kernel registry
# ===========================================================================

_KERNELS: dict = {
    'dw':       _dw,
    'hk':       _hk,
    'voter':    _voter,
    'degroot':  _degroot,
    'sznajd':   _sznajd,
    'majority': _majority,
    'fj':       _fj,
    'coda':     _coda,
    'ising':    _ising,
    'abelson':  _abelson,
}


# ===========================================================================
# Public entry point
# ===========================================================================

def calculate_opinion_change(
    X: np.ndarray,
    pairs: list,
    impact_vector: np.ndarray,
    params: dict,
    rng: Optional[np.random.Generator] = None,
    agent_data: Optional[dict] = None,
) -> np.ndarray:
    """
    Unified entry point. Dispatches to the kernel specified by params['kernel'].
    Falls back to 'dw' if the key is absent.

    Args:
        X:             Opinion matrix (N, L).
        pairs:         Interaction pairs [(i, j), ...] from topology layer.
        impact_vector: Per-agent impact field values (N,).
        params:        Kernel config dict. Must contain 'kernel' key.
                       All remaining keys are forwarded to the kernel.
        rng:           Seeded random generator for reproducibility.
        agent_data:    Persistent agent-level state (stubborn, anchor, etc.).

    Returns:
        delta_X (N, L) — synchronous opinion update, applied by steps.py.

    Raises:
        ValueError: if params['kernel'] names an unregistered kernel.

    Example::

        params = {
            'kernel':       'hk',
            'epsilon_base': 0.25,
            'alpha_mod':    0.1,
            'mu_base':      0.4,
        }
        delta = calculate_opinion_change(X, pairs, impact_vector, params, rng=rng)
    """
    kernel_name = params.get('kernel', 'dw')
    fn = _KERNELS.get(kernel_name)
    if fn is None:
        raise ValueError(
            f"Unknown dynamics kernel '{kernel_name}'. "
            f"Available: {sorted(_KERNELS.keys())}"
        )
    return fn(X, pairs, impact_vector, params, rng=rng, agent_data=agent_data)