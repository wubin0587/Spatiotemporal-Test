# -*- coding: utf-8 -*-
"""
@File    : base.py
@Desc    : Abstract Base Class for all Intervention Policies.

A policy encapsulates a single, well-defined mutation that is applied
to the simulation engine state when its associated trigger fires.

Design contract
---------------
- ``apply(engine)`` is the ONLY method that may mutate engine state.
- ``apply()`` must be idempotent where possible, or clearly document
  cumulative effects (e.g. a network rewiring applied twice).
- Policies must not start background threads or hold long-lived references
  to the engine beyond a single ``apply()`` call.
- ``undo()`` is optional and best-effort; not all policies can be reversed.
- ``describe()`` returns a human-readable summary for logging / reporting.

Application log
---------------
Each call to ``apply()`` is recorded in ``self.application_log``.  The log
entry is a dict:
    {
        'step'    : int,       # engine.time_step at application
        'time'    : float,     # engine.current_time at application
        'result'  : Any,       # optional return value from _apply()
    }

Subclassing
-----------
    class MyPolicy(BasePolicy):
        def _apply(self, engine):
            # ... mutate engine ...
            return {'changed': 42}   # optional metadata

        def describe(self):
            return "MyPolicy: does something specific"

Registry
--------
All concrete policy types are registered in ``_REGISTRY`` (at the bottom of
this module) as ``policy_type_string → (module_path, class_name)`` tuples.
The ``from_config`` factory uses this registry for lazy-import dispatch.

Available policy types
----------------------
Event policies (intervention.policies.event):
    event_suppress      -- Disable an event generator for N steps
    event_inject        -- Inject a synthetic event into the archive
    event_amplify       -- Scale existing event intensities by a factor
    event_filter        -- Remove events outside polarity/intensity bounds

Network policies (intervention.policies.network):
    network_rewire       -- Random edge rewiring
    network_remove_edges -- Sever cross-group connections
    network_add_edges    -- Add bridging edges between groups
    network_isolate      -- Remove all edges for target agents

Spatial policies (intervention.policies.spatial):
    agent_relocate       -- Teleport agents to a new position
    spatial_cluster      -- Pull agents towards a focal point
    spatial_dispersal    -- Push agents away from a focal point
    spatial_barrier      -- Reflect agents that cross a boundary

Temporal / dynamics policies (intervention.policies.time):
    dynamics_param       -- Override epsilon/mu and related parameters
    simulation_speed     -- Change dt per step
    opinion_clamp        -- Hard-clamp opinions to a range
    opinion_nudge        -- Soft signed shift on selected opinions

Multilayer policies (intervention.policies.multilayer):
    layer_coupling       -- Drag target-layer opinions towards source layer
    layer_weight         -- Set per-layer update multipliers
    layer_reset          -- Re-randomise one opinion layer
    layer_polarise       -- Push layer opinions towards nearest extreme
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Registry — maps YAML type strings to (module_path, class_name)
# =============================================================================

_REGISTRY: Dict[str, tuple] = {
    # --- Event policies ---
    "event_suppress":       ("intervention.policies.event",      "EventSuppressPolicy"),
    "event_inject":         ("intervention.policies.event",      "EventInjectPolicy"),
    "event_amplify":        ("intervention.policies.event",      "EventAmplifyPolicy"),
    "event_filter":         ("intervention.policies.event",      "EventFilterPolicy"),

    # --- Network policies ---
    "network_rewire":       ("intervention.policies.network",    "NetworkRewirePolicy"),
    "network_remove_edges": ("intervention.policies.network",    "NetworkRemoveEdgesPolicy"),
    "network_add_edges":    ("intervention.policies.network",    "NetworkAddEdgesPolicy"),
    "network_isolate":      ("intervention.policies.network",    "NetworkIsolatePolicy"),

    # --- Spatial policies ---
    "agent_relocate":       ("intervention.policies.spatial",    "AgentRelocationPolicy"),
    "spatial_cluster":      ("intervention.policies.spatial",    "SpatialClusterPolicy"),
    "spatial_dispersal":    ("intervention.policies.spatial",    "SpatialDispersalPolicy"),
    "spatial_barrier":      ("intervention.policies.spatial",    "SpatialBarrierPolicy"),

    # --- Temporal / dynamics policies ---
    "dynamics_param":       ("intervention.policies.time",       "DynamicsParamPolicy"),
    "simulation_speed":     ("intervention.policies.time",       "SimulationSpeedPolicy"),
    "opinion_clamp":        ("intervention.policies.time",       "OpinionClampPolicy"),
    "opinion_nudge":        ("intervention.policies.time",       "OpinionNudgePolicy"),

    # --- Multilayer policies ---
    "layer_coupling":       ("intervention.policies.multilayer", "LayerCouplingPolicy"),
    "layer_weight":         ("intervention.policies.multilayer", "LayerWeightPolicy"),
    "layer_reset":          ("intervention.policies.multilayer", "LayerResetPolicy"),
    "layer_polarise":       ("intervention.policies.multilayer", "LayerPolarisePolicy"),
}


# =============================================================================
# BasePolicy
# =============================================================================

class BasePolicy(ABC):
    """
    Abstract base class for intervention policies.

    Parameters
    ----------
    config : dict
        Policy-specific configuration dict (passed from YAML).
    name : str, optional
        Human-readable label.  Defaults to the class name.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        self.config: Dict[str, Any] = config or {}
        self.name: str = name or self.__class__.__name__
        self.application_log: List[Dict[str, Any]] = []
        self._apply_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self, engine: Any) -> Dict[str, Any]:
        """
        Apply the policy to the live engine.

        Wraps ``_apply()`` with bookkeeping and logging.

        Parameters
        ----------
        engine : StepExecutor
            Live engine reference.

        Returns
        -------
        dict
            Log entry dict including 'step', 'time', and any metadata
            returned by ``_apply()``.
        """
        self._apply_count += 1

        logger.info(
            f"[Policy:{self.name}] Applying at step={engine.time_step} "
            f"t={engine.current_time:.1f} (application #{self._apply_count})"
        )

        result = self._apply(engine)

        entry = {
            "step": engine.time_step,
            "time": engine.current_time,
            "result": result,
        }
        self.application_log.append(entry)

        logger.info(f"[Policy:{self.name}] Applied successfully. result={result}")
        return entry

    def undo(self, engine: Any) -> bool:
        """
        Attempt to reverse the last application of this policy.

        Base implementation is a no-op that returns False, indicating
        the policy is not reversible.  Subclasses may override.

        Returns
        -------
        bool
            True if undo was successful, False if not supported.
        """
        logger.warning(
            f"[Policy:{self.name}] undo() called but not implemented. "
            "Use BranchManager.restore() for full state rollback."
        )
        return False

    def reset(self) -> None:
        """Clear application log and count (used on simulation reset)."""
        self.application_log.clear()
        self._apply_count = 0

    @property
    def apply_count(self) -> int:
        """Number of times this policy has been applied."""
        return self._apply_count

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _apply(self, engine: Any) -> Optional[Any]:
        """
        Core mutation logic.  Implement in subclasses.

        Parameters
        ----------
        engine : StepExecutor

        Returns
        -------
        Any, optional
            Optional metadata dict describing what was changed.
            Stored in the application log.
        """

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable one-line description of this policy."""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BasePolicy":
        """
        Build a policy from a YAML-derived config dict.

        Dispatches to the correct concrete subclass based on ``cfg['type']``.
        Concrete policy modules are imported lazily to avoid circular imports.

        Parameters
        ----------
        cfg : dict
            Must contain ``type`` key matching a registered policy name.

        Raises
        ------
        ValueError
            If the policy type is not recognised.
        """
        policy_type = cfg.get("type", "").lower()

        if policy_type not in _REGISTRY:
            raise ValueError(
                f"Unknown policy type: '{policy_type}'. "
                f"Available: {sorted(_REGISTRY.keys())}"
            )

        module_path, class_name = _REGISTRY[policy_type]
        import importlib
        module = importlib.import_module(module_path)
        policy_cls = getattr(module, class_name)
        return policy_cls(config=cfg, name=cfg.get("name"))

    @classmethod
    def list_available(cls) -> List[str]:
        """Return a sorted list of all registered policy type strings."""
        return sorted(_REGISTRY.keys())

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"applied={self._apply_count}x>"
        )
