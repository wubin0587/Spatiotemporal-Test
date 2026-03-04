# -*- coding: utf-8 -*-
"""
@File    : trigger.py
@Desc    : Intervention Trigger Conditions.

A trigger answers one question at each simulation step:
    "Should an intervention fire right now?"

Design principles
-----------------
- Stateless evaluation: triggers read engine state but do not modify it.
- Composable: CompositeTrigger combines multiple triggers with AND / OR logic.
- One-shot vs repeating: controlled via ``max_fires`` and ``cooldown`` parameters.
- All triggers receive the same two arguments that the facade hook exposes:
      engine : StepExecutor   -- live engine reference (read-only use only)
      stats  : Dict[str, Any] -- metrics dict returned by engine.step()

Concrete triggers
-----------------
    StepTrigger          -- fires at a fixed time-step number
    TimeTrigger          -- fires when simulation time crosses a threshold
    PolarizationTrigger  -- fires when opinion std-dev exceeds a threshold
    ImpactTrigger        -- fires when mean impact field exceeds a threshold
    CompositeTrigger     -- combines triggers with AND / OR logic

YAML config schema (used by InterventionManager.from_config)
------------------------------------------------------------
trigger:
  type: step          # step | time | polarization | impact | composite
  step: 100           # for StepTrigger
  # time: 50.0        # for TimeTrigger
  # threshold: 0.4    # for PolarizationTrigger / ImpactTrigger
  # logic: and        # for CompositeTrigger: 'and' | 'or'
  # triggers: [...]   # for CompositeTrigger: list of child trigger configs
  cooldown: 0         # steps to wait before firing again (0 = fire once and done)
  max_fires: 1        # maximum number of times this trigger may fire (0 = unlimited)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base
# =============================================================================

class InterventionTrigger(ABC):
    """
    Abstract base class for all intervention triggers.

    Subclasses implement ``_evaluate(engine, stats) -> bool`` which contains
    the pure condition logic.  The base class manages cooldown bookkeeping and
    fire-count limits so that concrete triggers stay simple.

    Parameters
    ----------
    cooldown : int
        Minimum number of steps between consecutive fires.
        ``0`` means "fire only once then stay silent" (equivalent to max_fires=1
        with no repeat).  Set both ``cooldown=N`` and ``max_fires=0`` for a
        repeating trigger with a cooldown gap.
    max_fires : int
        Hard cap on total number of fires.  ``0`` means unlimited.
    name : str, optional
        Human-readable label used in log messages.
    """

    def __init__(
        self,
        cooldown: int = 0,
        max_fires: int = 1,
        name: Optional[str] = None,
    ):
        self.cooldown = cooldown
        self.max_fires = max_fires
        self.name = name or self.__class__.__name__

        # Internal bookkeeping (mutated by evaluate())
        self._fire_count: int = 0
        self._last_fire_step: int = -1  # step index of the most recent fire

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, engine: Any, stats: Dict[str, Any]) -> bool:
        """
        Evaluate whether the trigger should fire this step.

        This is the method called by InterventionManager.  It wraps
        ``_evaluate`` with cooldown and max-fires guards.

        Returns
        -------
        bool
            True if the trigger fires (intervention should execute).
        """
        # Hard cap reached?
        if self.max_fires > 0 and self._fire_count >= self.max_fires:
            return False

        # Still in cooldown?
        current_step: int = stats.get('step', engine.time_step)
        steps_since_last = current_step - self._last_fire_step
        if self._fire_count > 0 and steps_since_last <= self.cooldown:
            return False

        # Delegate to the concrete condition
        result = self._evaluate(engine, stats)

        if result:
            self._fire_count += 1
            self._last_fire_step = current_step
            logger.info(
                f"[Trigger:{self.name}] FIRED at step={current_step} "
                f"(fire #{self._fire_count})"
            )

        return result

    def reset(self) -> None:
        """Reset fire-count and last-fire bookkeeping (used on simulation reset)."""
        self._fire_count = 0
        self._last_fire_step = -1

    @property
    def fire_count(self) -> int:
        """Total number of times this trigger has fired."""
        return self._fire_count

    @property
    def has_exhausted(self) -> bool:
        """True if the trigger can never fire again (max_fires reached)."""
        return self.max_fires > 0 and self._fire_count >= self.max_fires

    # ------------------------------------------------------------------
    # Abstract method for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _evaluate(self, engine: Any, stats: Dict[str, Any]) -> bool:
        """
        Pure condition logic.  Must NOT modify engine state.

        Parameters
        ----------
        engine : StepExecutor
            Live engine reference.
        stats : Dict[str, Any]
            Metrics dict from the most recent engine.step() call.

        Returns
        -------
        bool
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "InterventionTrigger":
        """
        Build a trigger from a YAML-derived configuration dict.

        Parameters
        ----------
        cfg : dict
            Must contain ``type`` key.  Other keys are trigger-specific.
        """
        trigger_type = cfg.get("type", "step").lower()
        cooldown = cfg.get("cooldown", 0)
        max_fires = cfg.get("max_fires", 1)
        name = cfg.get("name", None)

        if trigger_type == "step":
            return StepTrigger(
                step=cfg["step"],
                cooldown=cooldown,
                max_fires=max_fires,
                name=name,
            )
        elif trigger_type == "time":
            return TimeTrigger(
                time_threshold=cfg["time"],
                cooldown=cooldown,
                max_fires=max_fires,
                name=name,
            )
        elif trigger_type == "polarization":
            return PolarizationTrigger(
                threshold=cfg["threshold"],
                cooldown=cooldown,
                max_fires=max_fires,
                name=name,
            )
        elif trigger_type == "impact":
            return ImpactTrigger(
                threshold=cfg["threshold"],
                cooldown=cooldown,
                max_fires=max_fires,
                name=name,
            )
        elif trigger_type == "composite":
            children = [
                InterventionTrigger.from_config(child_cfg)
                for child_cfg in cfg.get("triggers", [])
            ]
            return CompositeTrigger(
                triggers=children,
                logic=cfg.get("logic", "and"),
                cooldown=cooldown,
                max_fires=max_fires,
                name=name,
            )
        else:
            raise ValueError(f"Unknown trigger type: '{trigger_type}'")

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"fires={self._fire_count}/{self.max_fires or '∞'} "
            f"cooldown={self.cooldown}>"
        )


# =============================================================================
# Concrete Triggers
# =============================================================================

class StepTrigger(InterventionTrigger):
    """
    Fire exactly when the simulation reaches a target step number.

    Parameters
    ----------
    step : int
        The step index at which to fire.

    YAML example
    ------------
    trigger:
      type: step
      step: 100
      max_fires: 1
    """

    def __init__(self, step: int, **kwargs):
        super().__init__(**kwargs)
        self.target_step = step

    def _evaluate(self, engine: Any, stats: Dict[str, Any]) -> bool:
        current_step = stats.get("step", engine.time_step)
        return current_step >= self.target_step

    def __repr__(self) -> str:
        return f"<StepTrigger target={self.target_step} fires={self._fire_count}>"


class TimeTrigger(InterventionTrigger):
    """
    Fire when simulation time crosses a continuous threshold.

    Parameters
    ----------
    time_threshold : float
        Simulation time at which to fire.

    YAML example
    ------------
    trigger:
      type: time
      time: 50.0
      max_fires: 1
    """

    def __init__(self, time_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.time_threshold = time_threshold

    def _evaluate(self, engine: Any, stats: Dict[str, Any]) -> bool:
        return engine.current_time >= self.time_threshold

    def __repr__(self) -> str:
        return (
            f"<TimeTrigger threshold={self.time_threshold} "
            f"fires={self._fire_count}>"
        )


class PolarizationTrigger(InterventionTrigger):
    """
    Fire when opinion polarization (std-dev of opinion matrix) exceeds a threshold.

    This trigger reads from the live engine's opinion matrix so it reflects
    the state *after* the current step's dynamics have been applied.

    Parameters
    ----------
    threshold : float
        Polarization level (std-dev) above which to fire.

    YAML example
    ------------
    trigger:
      type: polarization
      threshold: 0.35
      cooldown: 20
      max_fires: 0   # unlimited — fires every 20 steps while condition holds
    """

    def __init__(self, threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def _evaluate(self, engine: Any, stats: Dict[str, Any]) -> bool:
        # Prefer the pre-computed stats value to avoid redundant computation,
        # fall back to direct calculation if not present.
        polarization = stats.get("opinion_std")
        if polarization is None:
            polarization = float(np.std(engine.opinion_matrix))
        return polarization >= self.threshold

    def __repr__(self) -> str:
        return (
            f"<PolarizationTrigger threshold={self.threshold} "
            f"fires={self._fire_count}>"
        )


class ImpactTrigger(InterventionTrigger):
    """
    Fire when the mean impact field I(t) across all agents exceeds a threshold.

    Parameters
    ----------
    threshold : float
        Mean impact level above which to fire.

    YAML example
    ------------
    trigger:
      type: impact
      threshold: 0.5
      cooldown: 10
      max_fires: 0
    """

    def __init__(self, threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def _evaluate(self, engine: Any, stats: Dict[str, Any]) -> bool:
        mean_impact = stats.get("mean_impact")
        if mean_impact is None:
            mean_impact = float(np.mean(engine.impact_vector))
        return mean_impact >= self.threshold

    def __repr__(self) -> str:
        return (
            f"<ImpactTrigger threshold={self.threshold} "
            f"fires={self._fire_count}>"
        )


class CompositeTrigger(InterventionTrigger):
    """
    Combine multiple triggers with boolean AND / OR logic.

    When ``logic='and'``, ALL child triggers must evaluate True for this
    trigger to fire.  With ``logic='or'``, ANY child suffices.

    Note: child triggers' individual fire-count and cooldown bookkeeping is
    bypassed — only the composite's own bookkeeping applies.  Children are
    evaluated via their ``_evaluate`` method directly to avoid double-counting.

    Parameters
    ----------
    triggers : list[InterventionTrigger]
        Child triggers to combine.
    logic : str
        ``'and'`` (default) or ``'or'``.

    YAML example
    ------------
    trigger:
      type: composite
      logic: and
      max_fires: 1
      triggers:
        - type: step
          step: 50
        - type: polarization
          threshold: 0.4
    """

    def __init__(
        self,
        triggers: List[InterventionTrigger],
        logic: str = "and",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not triggers:
            raise ValueError("CompositeTrigger requires at least one child trigger.")
        self.triggers = triggers
        self.logic = logic.lower()
        if self.logic not in ("and", "or"):
            raise ValueError(f"CompositeTrigger logic must be 'and' or 'or', got '{logic}'")

    def _evaluate(self, engine: Any, stats: Dict[str, Any]) -> bool:
        # Use _evaluate on children to avoid mutating their fire counts.
        results = [t._evaluate(engine, stats) for t in self.triggers]
        if self.logic == "and":
            return all(results)
        return any(results)

    def reset(self) -> None:
        super().reset()
        for t in self.triggers:
            t.reset()

    def __repr__(self) -> str:
        return (
            f"<CompositeTrigger logic={self.logic!r} "
            f"children={len(self.triggers)} fires={self._fire_count}>"
        )
