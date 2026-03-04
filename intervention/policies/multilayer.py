# -*- coding: utf-8 -*-
"""
@File    : multilayer.py
@Desc    : Multilayer / Cross-layer Intervention Policies.

These policies operate across the opinion-layer dimension of the simulation,
modelling phenomena like:
  - Cross-topic opinion alignment (political sorting)
  - Layer-specific trust or learning-rate modulation
  - Opinion homogenisation / polarisation at the layer level

Concrete policies
-----------------
    LayerCouplingPolicy     -- Drag layer-k opinions towards layer-j opinions
    LayerWeightPolicy       -- Modify per-layer weights in the dynamics step
    LayerResetPolicy        -- Re-randomise one opinion layer for a subset of agents
    LayerPolarisePolicy     -- Push layer opinions towards the nearest extreme (0 or 1)

YAML config examples
--------------------
policy:
  type: layer_coupling
  source_layer: 0          # layer whose opinions act as the attractor
  target_layer: 1          # layer whose opinions are updated
  coupling_strength: 0.1   # fraction of distance to pull target towards source
  agents: null             # null => all agents

policy:
  type: layer_weight
  layer_weights: [1.0, 0.5, 2.0]   # per-layer multiplier on opinion updates
  # Stored in engine.dynamics_config['layer_weights']

policy:
  type: layer_reset
  layer: 2
  agents: null             # null => all agents
  init_type: uniform       # uniform | normal | polarized

policy:
  type: layer_polarise
  layer: 0                 # -1 => all layers
  strength: 0.3            # fraction of distance to push towards nearest extreme
  agents: null
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BasePolicy

logger = logging.getLogger(__name__)


# =============================================================================
# LayerCouplingPolicy
# =============================================================================

class LayerCouplingPolicy(BasePolicy):
    """
    Pull ``target_layer`` opinions towards ``source_layer`` opinions.

    Useful for modelling cross-topic attitude alignment (e.g. political
    sorting where economic and social opinions converge).

    The update for each targeted agent ``i`` is:
        X[i, target] += coupling_strength * (X[i, source] - X[i, target])

    Parameters (config keys)
    ------------------------
    source_layer : int
        The reference/attractor layer index.
    target_layer : int
        The layer whose opinions are updated.
    coupling_strength : float
        Fraction of the inter-layer gap to close per application.
        Range ``(0, 1]``.  Default ``0.1``.
    agents : list[int], optional
        Subset of agents.  ``None`` → all agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "LayerCouplingPolicy")
        self.source_layer: int = int(self.config.get("source_layer", 0))
        self.target_layer: int = int(self.config.get("target_layer", 1))
        self.coupling_strength: float = float(self.config.get("coupling_strength", 0.1))
        self.agents: Optional[List[int]] = self.config.get("agents", None)

        if self.source_layer == self.target_layer:
            raise ValueError(
                "LayerCouplingPolicy: source_layer and target_layer must differ."
            )
        if not (0.0 < self.coupling_strength <= 1.0):
            raise ValueError(
                "LayerCouplingPolicy: coupling_strength must be in (0, 1]."
            )

    def _apply(self, engine: Any) -> Optional[Dict]:
        X = engine.opinion_matrix  # (N, L)
        num_layers = X.shape[1]

        for layer in (self.source_layer, self.target_layer):
            if not (0 <= layer < num_layers):
                raise IndexError(
                    f"LayerCouplingPolicy: layer index {layer} out of range "
                    f"[0, {num_layers})."
                )

        rows = (
            np.array(self.agents, dtype=int)
            if self.agents is not None
            else np.arange(engine.num_agents)
        )

        gap = X[rows, self.source_layer] - X[rows, self.target_layer]
        X[rows, self.target_layer] += self.coupling_strength * gap

        # Clip to valid range
        engine.opinion_matrix = np.clip(X, 0.0, 1.0)

        mean_gap_before = float(np.mean(np.abs(gap)))
        mean_gap_after = float(
            np.mean(np.abs(X[rows, self.source_layer] - X[rows, self.target_layer]))
        )

        logger.info(
            f"[LayerCouplingPolicy] Coupled layer {self.target_layer} → "
            f"layer {self.source_layer} for {len(rows)} agents. "
            f"Mean gap: {mean_gap_before:.4f} → {mean_gap_after:.4f}."
        )
        return {
            "agents": len(rows),
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "mean_gap_before": mean_gap_before,
            "mean_gap_after": mean_gap_after,
        }

    def describe(self) -> str:
        return (
            f"LayerCouplingPolicy: couple layer {self.target_layer} → "
            f"layer {self.source_layer} with strength={self.coupling_strength}."
        )


# =============================================================================
# LayerWeightPolicy
# =============================================================================

class LayerWeightPolicy(BasePolicy):
    """
    Inject per-layer multipliers into the dynamics configuration.

    When the opinion-update kernel applies ``delta_X``, weights stored in
    ``engine.dynamics_config['layer_weights']`` can be used to scale the
    update magnitude differently per topic layer.

    This policy writes the weight vector into ``dynamics_config`` so that
    the dynamics module (if it respects the key) will apply the weights on
    subsequent steps.

    Parameters (config keys)
    ------------------------
    layer_weights : list[float]
        Per-layer multipliers.  Length must equal ``num_layers``.
        Values > 1 amplify opinion updates for that layer; < 1 dampen them.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "LayerWeightPolicy")
        self.layer_weights: List[float] = self.config.get("layer_weights", [])
        if not self.layer_weights:
            raise ValueError("LayerWeightPolicy: 'layer_weights' list is required.")

    def _apply(self, engine: Any) -> Optional[Dict]:
        num_layers = engine.opinion_matrix.shape[1]
        if len(self.layer_weights) != num_layers:
            raise ValueError(
                f"LayerWeightPolicy: layer_weights length {len(self.layer_weights)} "
                f"!= num_layers {num_layers}."
            )

        previous = engine.dynamics_config.get("layer_weights", None)
        engine.dynamics_config["layer_weights"] = list(self.layer_weights)

        logger.info(
            f"[LayerWeightPolicy] Set layer_weights={self.layer_weights}. "
            f"Previous: {previous}."
        )
        return {
            "layer_weights": self.layer_weights,
            "previous": previous,
        }

    def describe(self) -> str:
        return f"LayerWeightPolicy: set layer_weights={self.layer_weights}."


# =============================================================================
# LayerResetPolicy
# =============================================================================

class LayerResetPolicy(BasePolicy):
    """
    Re-initialise one opinion layer for a subset of agents.

    Models situations where a specific information channel is disrupted and
    agents lose their previously formed opinions on that topic (e.g. a
    platform shutdown, a media blackout, or a major scandal that resets
    trust in a topic area).

    Parameters (config keys)
    ------------------------
    layer : int
        Layer to reset.  Must be a valid index (no ``-1`` wildcard here).
    agents : list[int], optional
        Subset to reset.  ``None`` → all agents.
    init_type : str
        Reinitialisation distribution:
        - ``'uniform'``   : U(0, 1)
        - ``'normal'``    : N(0.5, 0.2) clipped to [0, 1]
        - ``'polarized'`` : 50/50 split at 0.2 and 0.8
    seed : int, optional
    """

    _VALID_TYPES = ("uniform", "normal", "polarized")

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "LayerResetPolicy")
        self.layer: int = int(self.config.get("layer", 0))
        self.agents: Optional[List[int]] = self.config.get("agents", None)
        self.init_type: str = self.config.get("init_type", "uniform").lower()
        self.seed = self.config.get("seed", None)

        if self.init_type not in self._VALID_TYPES:
            raise ValueError(
                f"LayerResetPolicy: init_type must be one of {self._VALID_TYPES}."
            )

    def _apply(self, engine: Any) -> Optional[Dict]:
        X = engine.opinion_matrix
        num_layers = X.shape[1]

        if not (0 <= self.layer < num_layers):
            raise IndexError(
                f"LayerResetPolicy: layer {self.layer} out of range [0, {num_layers})."
            )

        rows = (
            np.array(self.agents, dtype=int)
            if self.agents is not None
            else np.arange(engine.num_agents)
        )
        n = len(rows)

        rng = np.random.default_rng(
            self.seed if self.seed is not None else engine.time_step
        )

        if self.init_type == "uniform":
            new_opinions = rng.random(n).astype(np.float32)
        elif self.init_type == "normal":
            new_opinions = rng.normal(0.5, 0.2, n).astype(np.float32)
            new_opinions = np.clip(new_opinions, 0.0, 1.0)
        else:  # polarized
            half = n // 2
            left = rng.normal(0.2, 0.1, half)
            right = rng.normal(0.8, 0.1, n - half)
            new_opinions = np.concatenate([left, right]).astype(np.float32)
            rng.shuffle(new_opinions)
            new_opinions = np.clip(new_opinions, 0.0, 1.0)

        X[rows, self.layer] = new_opinions
        engine.opinion_matrix = X

        logger.info(
            f"[LayerResetPolicy] Reset layer {self.layer} for {n} agents "
            f"using init_type='{self.init_type}'."
        )
        return {
            "layer": self.layer,
            "reset_agents": n,
            "init_type": self.init_type,
            "new_mean": float(np.mean(new_opinions)),
        }

    def describe(self) -> str:
        return (
            f"LayerResetPolicy: re-initialise layer {self.layer} "
            f"({self.init_type} distribution)."
        )


# =============================================================================
# LayerPolarisePolicy
# =============================================================================

class LayerPolarisePolicy(BasePolicy):
    """
    Push agents' layer opinions towards the nearest extreme (0 or 1).

    Simulates algorithmic radicalisation, filter-bubble reinforcement, or
    any process that systematically amplifies existing opinions.

    For each agent ``i`` and each targeted layer ``l``:
        if X[i, l] >= 0.5:  X[i, l] += strength * (1.0 - X[i, l])
        else:               X[i, l] -= strength * X[i, l]

    Parameters (config keys)
    ------------------------
    layer : int
        Layer to polarise.  ``-1`` polarises all layers.  Default ``-1``.
    strength : float
        Fraction of remaining distance to the extreme to move per step.
        Range ``(0, 1]``.  Default ``0.3``.
    agents : list[int], optional
        Subset of agents.  ``None`` → all agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "LayerPolarisePolicy")
        self.layer: int = int(self.config.get("layer", -1))
        self.strength: float = float(self.config.get("strength", 0.3))
        self.agents: Optional[List[int]] = self.config.get("agents", None)

        if not (0.0 < self.strength <= 1.0):
            raise ValueError("LayerPolarisePolicy: strength must be in (0, 1].")

    def _apply(self, engine: Any) -> Optional[Dict]:
        X = engine.opinion_matrix  # (N, L)

        rows = (
            np.array(self.agents, dtype=int)
            if self.agents is not None
            else np.arange(engine.num_agents)
        )

        if self.layer == -1:
            cols = np.arange(X.shape[1])
        else:
            if not (0 <= self.layer < X.shape[1]):
                raise IndexError(
                    f"LayerPolarisePolicy: layer {self.layer} out of range."
                )
            cols = np.array([self.layer])

        std_before = float(np.std(X[np.ix_(rows, cols)]))

        for col in cols:
            view = X[rows, col]
            above = view >= 0.5
            below = ~above

            # Push towards 1
            view[above] += self.strength * (1.0 - view[above])
            # Push towards 0
            view[below] -= self.strength * view[below]

            X[rows, col] = view

        engine.opinion_matrix = np.clip(X, 0.0, 1.0)

        std_after = float(np.std(X[np.ix_(rows, cols)]))

        logger.info(
            f"[LayerPolarisePolicy] Polarised layer(s)={self.layer} for "
            f"{len(rows)} agents with strength={self.strength}. "
            f"Std: {std_before:.4f} → {std_after:.4f}."
        )
        return {
            "polarised_agents": len(rows),
            "layer": self.layer,
            "strength": self.strength,
            "std_before": std_before,
            "std_after": std_after,
        }

    def describe(self) -> str:
        layer_str = "all layers" if self.layer == -1 else f"layer {self.layer}"
        return (
            f"LayerPolarisePolicy: push {layer_str} towards extremes "
            f"with strength={self.strength}."
        )
