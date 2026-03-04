# -*- coding: utf-8 -*-
"""
@File    : time.py
@Desc    : Temporal / Dynamics-Parameter Intervention Policies.

Policies that modify the engine's runtime dynamics configuration, allowing
researchers to test how changes in trust thresholds, learning rates, and
interaction topology respond to interventions *mid-simulation*:

    DynamicsParamPolicy     -- Directly override epsilon/mu/alpha/beta values
    SimulationSpeedPolicy   -- Change the dt (time increment) per step
    OpinionClampPolicy      -- Hard-clamp agent opinions to a target range
    OpinionNudgePolicy      -- Softly shift selected agents' opinions

YAML config examples
--------------------
policy:
  type: dynamics_param
  overrides:
    epsilon_base: 0.35
    mu_base: 0.15
    alpha_mod: 0.1

policy:
  type: simulation_speed
  dt: 0.5                  # set time increment per step (default is 1.0)

policy:
  type: opinion_clamp
  layer: 0                 # opinion dimension to clamp (-1 = all)
  min_value: 0.1
  max_value: 0.9
  agents: null             # null => all agents

policy:
  type: opinion_nudge
  agents: null             # null => top_k_by_impact
  top_k_by_impact: 20
  layer: -1                # -1 => all layers
  delta: 0.05              # signed shift applied to opinions
  direction: positive      # 'positive' | 'negative' | 'center'
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BasePolicy

logger = logging.getLogger(__name__)


# =============================================================================
# DynamicsParamPolicy
# =============================================================================

class DynamicsParamPolicy(BasePolicy):
    """
    Override one or more keys in the engine's ``dynamics_config`` dict.

    Because ``StepExecutor.step()`` reads parameters from
    ``self.dynamics_config`` at every call, the change takes effect on the
    immediately following step.

    Parameters (config keys)
    ------------------------
    overrides : dict
        Key-value pairs to write into ``engine.dynamics_config``.  Any key
        present in the existing dynamics config can be overridden, e.g.:
        ``epsilon_base``, ``mu_base``, ``alpha_mod``, ``beta_mod``,
        ``backfire``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "DynamicsParamPolicy")
        self.overrides: Dict[str, Any] = self.config.get("overrides", {})
        if not self.overrides:
            logger.warning("[DynamicsParamPolicy] No 'overrides' provided — policy is a no-op.")

    def _apply(self, engine: Any) -> Optional[Dict]:
        previous = {k: engine.dynamics_config.get(k) for k in self.overrides}
        engine.dynamics_config.update(self.overrides)

        logger.info(
            f"[DynamicsParamPolicy] Updated dynamics_config with {self.overrides}. "
            f"Previous values: {previous}."
        )
        return {"overrides": self.overrides, "previous": previous}

    def describe(self) -> str:
        return f"DynamicsParamPolicy: override dynamics params {self.overrides}."


# =============================================================================
# SimulationSpeedPolicy
# =============================================================================

class SimulationSpeedPolicy(BasePolicy):
    """
    Change the simulated time increment (``dt``) per step.

    ``StepExecutor.step()`` uses a hard-coded ``dt = 1.0``; this policy
    patches the engine to use a different increment by injecting a
    ``_dt`` attribute and monkey-patching ``step``'s time-advancement.

    Because modifying the step function itself is fragile, this policy
    instead stores the new ``dt`` in ``engine._intervention_dt`` and
    relies on users to check that attribute, OR it directly patches the
    step outcome by adjusting ``engine.current_time`` by the delta
    (``new_dt - 1.0``) at the end of each future step via the
    ``InterventionManager`` hook.

    For simplicity this implementation stores the target dt on the engine
    and advances time accordingly.

    Parameters (config keys)
    ------------------------
    dt : float
        New time-per-step increment.  Values < 1.0 slow down simulated time;
        values > 1.0 speed it up.  Default ``1.0`` (no change).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "SimulationSpeedPolicy")
        self.dt = float(self.config.get("dt", 1.0))

    def _apply(self, engine: Any) -> Optional[Dict]:
        old_dt = getattr(engine, "_intervention_dt", 1.0)
        engine._intervention_dt = self.dt

        # Patch the time-advance so future steps use the new dt.
        # StepExecutor.step() does: self.current_time += 1.0
        # We override this by wrapping the step method once.
        if not getattr(engine, "_step_speed_patched", False):
            original_step = engine.step.__func__

            def patched_step(self_inner):
                result = original_step(self_inner)
                # Undo the default +1 and apply our dt instead
                delta = getattr(self_inner, "_intervention_dt", 1.0) - 1.0
                self_inner.current_time += delta
                return result

            import types
            engine.step = types.MethodType(patched_step, engine)
            engine._step_speed_patched = True

        logger.info(
            f"[SimulationSpeedPolicy] dt changed from {old_dt} to {self.dt}."
        )
        return {"old_dt": old_dt, "new_dt": self.dt}

    def describe(self) -> str:
        return f"SimulationSpeedPolicy: set time increment to dt={self.dt}."


# =============================================================================
# OpinionClampPolicy
# =============================================================================

class OpinionClampPolicy(BasePolicy):
    """
    Hard-clamp agent opinions within ``[min_value, max_value]``.

    Use this to model censorship (opinions forced away from extremes),
    regulatory floors/ceilings, or authoritarian opinion constraints.

    Parameters (config keys)
    ------------------------
    layer : int
        Which opinion dimension to clamp.  ``-1`` clamps all layers.
        Default ``-1``.
    min_value : float
        Lower bound.  Default ``0.0``.
    max_value : float
        Upper bound.  Default ``1.0``.
    agents : list[int], optional
        Subset of agents to clamp.  ``None`` → all agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "OpinionClampPolicy")
        self.layer: int = int(self.config.get("layer", -1))
        self.min_value: float = float(self.config.get("min_value", 0.0))
        self.max_value: float = float(self.config.get("max_value", 1.0))
        self.agents: Optional[List[int]] = self.config.get("agents", None)

        if self.min_value > self.max_value:
            raise ValueError(
                f"OpinionClampPolicy: min_value ({self.min_value}) > "
                f"max_value ({self.max_value})."
            )

    def _apply(self, engine: Any) -> Optional[Dict]:
        X = engine.opinion_matrix  # (N, L) float32

        # Resolve agent subset
        if self.agents is not None:
            rows = np.array(self.agents, dtype=int)
        else:
            rows = np.arange(engine.num_agents)

        # Resolve layer(s)
        if self.layer == -1:
            cols = slice(None)
        else:
            cols = self.layer

        before = float(np.mean(np.abs(X[np.ix_(rows, np.arange(X.shape[1]))]
                                      if self.layer == -1 else X[rows, self.layer])))

        X[rows, cols] = np.clip(X[rows, cols], self.min_value, self.max_value)

        after = float(np.mean(np.abs(X[np.ix_(rows, np.arange(X.shape[1]))]
                                     if self.layer == -1 else X[rows, self.layer])))

        logger.info(
            f"[OpinionClampPolicy] Clamped opinions for {len(rows)} agents "
            f"in layer={self.layer} to [{self.min_value}, {self.max_value}]."
        )
        return {
            "clamped_agents": len(rows),
            "layer": self.layer,
            "mean_before": before,
            "mean_after": after,
        }

    def describe(self) -> str:
        layer_str = "all layers" if self.layer == -1 else f"layer {self.layer}"
        return (
            f"OpinionClampPolicy: clamp {layer_str} to "
            f"[{self.min_value}, {self.max_value}]."
        )


# =============================================================================
# OpinionNudgePolicy
# =============================================================================

class OpinionNudgePolicy(BasePolicy):
    """
    Apply a soft signed shift to selected agents' opinions.

    Models propaganda, targeted information campaigns, or algorithmic
    recommendation bias that gradually pushes users in a particular direction.

    Parameters (config keys)
    ------------------------
    agents : list[int], optional
        Explicit agent indices.  If None, ``top_k_by_impact`` is used.
    top_k_by_impact : int, optional
        Automatically target the ``k`` agents with the highest impact value.
    layer : int
        Opinion dimension to nudge.  ``-1`` nudges all layers.  Default ``-1``.
    delta : float
        Magnitude of the shift.  Default ``0.05``.
    direction : str
        One of:
        - ``'positive'`` : add ``+delta``
        - ``'negative'`` : add ``-delta``
        - ``'center'``   : shift towards 0.5 by ``delta``
        Default ``'positive'``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "OpinionNudgePolicy")
        self.agents: Optional[List[int]] = self.config.get("agents", None)
        self.top_k: int = int(self.config.get("top_k_by_impact", 0))
        self.layer: int = int(self.config.get("layer", -1))
        self.delta: float = float(self.config.get("delta", 0.05))
        self.direction: str = self.config.get("direction", "positive").lower()

        if self.direction not in ("positive", "negative", "center"):
            raise ValueError(
                f"OpinionNudgePolicy: direction must be 'positive', 'negative', "
                f"or 'center', got '{self.direction}'."
            )

    def _apply(self, engine: Any) -> Optional[Dict]:
        targets = self._resolve_targets(engine)
        X = engine.opinion_matrix  # (N, L)

        # Resolve layer
        if self.layer == -1:
            cols = slice(None)
        else:
            cols = self.layer

        for idx in targets:
            if self.direction == "positive":
                X[idx, cols] += self.delta
            elif self.direction == "negative":
                X[idx, cols] -= self.delta
            else:  # center
                current = X[idx, cols]
                shift = self.delta * np.sign(0.5 - current)
                X[idx, cols] += shift

        # Keep opinions in [0, 1]
        engine.opinion_matrix = np.clip(X, 0.0, 1.0)

        logger.info(
            f"[OpinionNudgePolicy] Nudged {len(targets)} agents in layer={self.layer} "
            f"by delta={self.delta} direction={self.direction!r}."
        )
        return {
            "nudged_agents": len(targets),
            "layer": self.layer,
            "delta": self.delta,
            "direction": self.direction,
        }

    def _resolve_targets(self, engine: Any) -> List[int]:
        if self.agents is not None:
            return [int(a) for a in self.agents]
        if self.top_k > 0:
            sorted_idx = np.argsort(engine.impact_vector)[::-1]
            return sorted_idx[: self.top_k].tolist()
        # Default: all agents
        return list(range(engine.num_agents))

    def describe(self) -> str:
        layer_str = "all layers" if self.layer == -1 else f"layer {self.layer}"
        return (
            f"OpinionNudgePolicy: nudge {layer_str} by {self.delta:+.3f} "
            f"({self.direction}) for targeted agents."
        )
