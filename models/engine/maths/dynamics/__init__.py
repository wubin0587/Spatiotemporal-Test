# -*- coding: utf-8 -*-
# Dispatcher for opinion dynamics kernels.
# All kernels share the same signature; steps.py calls this entry point only.

import numpy as np
from typing import Optional

from .dw       import calculate_opinion_change as _dw
from .hk       import calculate_opinion_change as _hk
from .voter    import calculate_opinion_change as _voter
from .degroot  import calculate_opinion_change as _degroot
from .sznajd   import calculate_opinion_change as _sznajd
from .majority import calculate_opinion_change as _majority
from .fj       import calculate_opinion_change as _fj
from .coda     import calculate_opinion_change as _coda
from .ising    import calculate_opinion_change as _ising
from .abelson  import calculate_opinion_change as _abelson

_KERNELS = {
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
    Falls back to 'dw' if not specified.

    Args:
        X:             Opinion matrix (N, L).
        pairs:         Interaction pairs [(i, j), ...] from topology layer.
        impact_vector: Per-agent impact field values (N,).
        params:        Kernel config dict. Must contain 'kernel' key.
        rng:           Seeded random generator for reproducibility.
        agent_data:    Persistent agent-level state (stubborn, anchor, etc.).

    Returns:
        delta_X (N, L) — synchronous opinion update, applied by steps.py.
    """
    kernel_name = params.get('kernel', 'dw')
    fn = _KERNELS.get(kernel_name)
    if fn is None:
        raise ValueError(
            f"Unknown dynamics kernel '{kernel_name}'. "
            f"Available: {sorted(_KERNELS.keys())}"
        )
    return fn(X, pairs, impact_vector, params, rng=rng, agent_data=agent_data)