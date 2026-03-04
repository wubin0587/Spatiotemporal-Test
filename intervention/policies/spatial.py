# -*- coding: utf-8 -*-
"""
@File    : spatial.py
@Desc    : Spatial Intervention Policies.

Policies that modify agent positions (and therefore the KDTree spatial index
and impact-field calculations):

    AgentRelocationPolicy   -- Move a subset of agents to new positions
    SpatialClusterPolicy    -- Pull a group of agents towards a focal point
    SpatialDispersalPolicy  -- Push a group of agents away from a focal point
    SpatialBarrierPolicy    -- Teleport agents that cross a boundary line

All policies rebuild the KDTree after modifying positions so that the
topology module sees the change immediately.

YAML config examples
--------------------
policy:
  type: agent_relocate
  agents: [0, 1, 2]       # explicit indices, or use top_k_by_impact
  top_k_by_impact: 10     # relocate the 10 most-impacted agents
  destination: [0.5, 0.5] # [x, y]; null => random position

policy:
  type: spatial_cluster
  agents: null             # null => all agents in region
  region: [0.0, 0.5, 0.0, 0.5]  # [xmin, xmax, ymin, ymax]
  focal_point: [0.25, 0.25]
  strength: 0.3            # fraction of distance to move (0=no move, 1=teleport)

policy:
  type: spatial_dispersal
  focal_point: [0.5, 0.5]
  strength: 0.2
  agents: null             # null => all agents

policy:
  type: spatial_barrier
  axis: x                  # 'x' or 'y'
  value: 0.5               # boundary coordinate
  side: left               # 'left'/'right' or 'below'/'above' — agents on this side are reflected
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from .base import BasePolicy

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _rebuild_spatial_index(engine: Any) -> None:
    """Rebuild the KDTree from current agent positions."""
    engine.spatial_index = cKDTree(engine.agent_positions)
    logger.debug("Spatial index rebuilt after spatial intervention.")


# =============================================================================
# AgentRelocationPolicy
# =============================================================================

class AgentRelocationPolicy(BasePolicy):
    """
    Teleport a subset of agents to a new position.

    Useful for modelling evacuation events, forced migration, or
    targeted displacement of high-impact individuals.

    Parameters (config keys)
    ------------------------
    agents : list[int], optional
        Explicit agent indices.  If omitted, ``top_k_by_impact`` is used.
    top_k_by_impact : int, optional
        Select the ``k`` agents with the highest current impact values.
    destination : list[float], optional
        ``[x, y]`` target.  If ``None`` (default), agents are placed at
        random positions drawn uniformly from ``[0, 1]^2``.
    seed : int, optional
        RNG seed for random destination.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "AgentRelocationPolicy")
        self.agents: Optional[List[int]] = self.config.get("agents", None)
        self.top_k: int = int(self.config.get("top_k_by_impact", 0))
        self.destination: Optional[List[float]] = self.config.get("destination", None)
        self.seed = self.config.get("seed", None)

    def _apply(self, engine: Any) -> Optional[Dict]:
        targets = self._resolve_targets(engine)
        rng = np.random.default_rng(
            self.seed if self.seed is not None else engine.time_step
        )

        for idx in targets:
            if self.destination is not None:
                new_pos = np.array(self.destination, dtype=np.float32)
            else:
                new_pos = rng.random(2).astype(np.float32)
            engine.agent_positions[idx] = np.clip(new_pos, 0.0, 1.0)

        _rebuild_spatial_index(engine)

        logger.info(
            f"[AgentRelocationPolicy] Relocated {len(targets)} agents "
            f"to destination={self.destination}."
        )
        return {"relocated": len(targets), "destination": self.destination}

    def _resolve_targets(self, engine: Any) -> List[int]:
        if self.agents is not None:
            return [int(a) for a in self.agents]
        if self.top_k > 0:
            sorted_idx = np.argsort(engine.impact_vector)[::-1]
            return sorted_idx[: self.top_k].tolist()
        raise ValueError(
            "AgentRelocationPolicy requires 'agents' list or 'top_k_by_impact'."
        )

    def describe(self) -> str:
        dest_str = str(self.destination) if self.destination else "random"
        return f"AgentRelocationPolicy: move agents to {dest_str}."


# =============================================================================
# SpatialClusterPolicy
# =============================================================================

class SpatialClusterPolicy(BasePolicy):
    """
    Pull agents towards a focal point, simulating rally, protest, or gathering.

    Each targeted agent is moved a fraction ``strength`` of the way towards
    the ``focal_point``:
        new_pos = pos + strength * (focal - pos)

    Parameters (config keys)
    ------------------------
    focal_point : list[float]
        ``[x, y]`` centre of attraction.  Default ``[0.5, 0.5]``.
    strength : float
        Movement fraction in ``(0, 1]``.  ``1.0`` teleports directly.
        Default ``0.3``.
    agents : list[int], optional
        Agent indices.  If omitted, ``region`` is used to select agents.
    region : list[float], optional
        ``[xmin, xmax, ymin, ymax]`` bounding box.  Agents inside this box
        are targeted.  If both ``agents`` and ``region`` are None, all agents
        are targeted.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "SpatialClusterPolicy")
        self.focal_point = np.array(
            self.config.get("focal_point", [0.5, 0.5]), dtype=np.float32
        )
        self.strength = float(self.config.get("strength", 0.3))
        self.agents: Optional[List[int]] = self.config.get("agents", None)
        self.region: Optional[List[float]] = self.config.get("region", None)

    def _apply(self, engine: Any) -> Optional[Dict]:
        targets = self._resolve_targets(engine)

        for idx in targets:
            pos = engine.agent_positions[idx]
            direction = self.focal_point - pos
            engine.agent_positions[idx] = np.clip(
                pos + self.strength * direction, 0.0, 1.0
            )

        _rebuild_spatial_index(engine)

        logger.info(
            f"[SpatialClusterPolicy] Pulled {len(targets)} agents towards "
            f"focal_point={self.focal_point.tolist()} "
            f"with strength={self.strength}."
        )
        return {
            "clustered": len(targets),
            "focal_point": self.focal_point.tolist(),
            "strength": self.strength,
        }

    def _resolve_targets(self, engine: Any) -> List[int]:
        if self.agents is not None:
            return [int(a) for a in self.agents]

        if self.region is not None:
            xmin, xmax, ymin, ymax = self.region
            pos = engine.agent_positions
            mask = (
                (pos[:, 0] >= xmin)
                & (pos[:, 0] <= xmax)
                & (pos[:, 1] >= ymin)
                & (pos[:, 1] <= ymax)
            )
            return np.where(mask)[0].tolist()

        # Default: all agents
        return list(range(engine.num_agents))

    def describe(self) -> str:
        return (
            f"SpatialClusterPolicy: pull agents {self.strength*100:.0f}% "
            f"towards {self.focal_point.tolist()}."
        )


# =============================================================================
# SpatialDispersalPolicy
# =============================================================================

class SpatialDispersalPolicy(BasePolicy):
    """
    Push agents away from a focal point, simulating evacuation or dispersal.

    Each targeted agent is moved a fraction ``strength`` away from the
    ``focal_point``:
        direction = pos - focal   (normalised)
        new_pos   = pos + strength * |pos - focal| * direction

    Parameters (config keys)
    ------------------------
    focal_point : list[float]
        ``[x, y]`` centre of repulsion.  Default ``[0.5, 0.5]``.
    strength : float
        Movement fraction.  Default ``0.2``.
    agents : list[int], optional
        Explicit agent indices.  If None, all agents are pushed.
    min_radius : float, optional
        Only push agents within this radius of the focal_point.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "SpatialDispersalPolicy")
        self.focal_point = np.array(
            self.config.get("focal_point", [0.5, 0.5]), dtype=np.float32
        )
        self.strength = float(self.config.get("strength", 0.2))
        self.agents: Optional[List[int]] = self.config.get("agents", None)
        self.min_radius: Optional[float] = self.config.get("min_radius", None)

    def _apply(self, engine: Any) -> Optional[Dict]:
        targets = self._resolve_targets(engine)
        moved = 0

        for idx in targets:
            pos = engine.agent_positions[idx]
            diff = pos - self.focal_point
            dist = float(np.linalg.norm(diff))

            # Skip if agent is essentially at the focal point
            if dist < 1e-8:
                continue

            direction = diff / dist
            engine.agent_positions[idx] = np.clip(
                pos + self.strength * dist * direction, 0.0, 1.0
            )
            moved += 1

        _rebuild_spatial_index(engine)

        logger.info(
            f"[SpatialDispersalPolicy] Dispersed {moved} agents away from "
            f"focal_point={self.focal_point.tolist()} "
            f"with strength={self.strength}."
        )
        return {
            "dispersed": moved,
            "focal_point": self.focal_point.tolist(),
            "strength": self.strength,
        }

    def _resolve_targets(self, engine: Any) -> List[int]:
        if self.agents is not None:
            return [int(a) for a in self.agents]

        if self.min_radius is not None:
            dists = np.linalg.norm(
                engine.agent_positions - self.focal_point, axis=1
            )
            return np.where(dists <= self.min_radius)[0].tolist()

        return list(range(engine.num_agents))

    def describe(self) -> str:
        return (
            f"SpatialDispersalPolicy: push agents away from "
            f"{self.focal_point.tolist()} with strength={self.strength}."
        )


# =============================================================================
# SpatialBarrierPolicy
# =============================================================================

class SpatialBarrierPolicy(BasePolicy):
    """
    Enforce a hard spatial boundary by reflecting agents that have crossed it.

    Can model physical barriers (walls, rivers, borders) that prevent agents
    from occupying certain regions of the map.

    Parameters (config keys)
    ------------------------
    axis : str
        ``'x'`` or ``'y'``.  Defines the axis of the barrier line.
    value : float
        Coordinate of the barrier along ``axis``.  E.g. ``0.5`` divides the
        map in half.
    side : str
        Which side agents are *allowed* to remain on:
        - For axis ``'x'``: ``'left'`` (x < value) or ``'right'`` (x > value).
        - For axis ``'y'``: ``'below'`` (y < value) or ``'above'`` (y > value).
        Agents on the forbidden side are reflected back.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "SpatialBarrierPolicy")
        self.axis: str = self.config.get("axis", "x").lower()
        self.value: float = float(self.config.get("value", 0.5))
        self.side: str = self.config.get("side", "left").lower()

        if self.axis not in ("x", "y"):
            raise ValueError(f"SpatialBarrierPolicy: axis must be 'x' or 'y', got '{self.axis}'.")

    def _apply(self, engine: Any) -> Optional[Dict]:
        pos = engine.agent_positions
        axis_idx = 0 if self.axis == "x" else 1
        val = self.value

        # Determine which agents are on the forbidden side
        if self.side in ("left", "below"):
            # Agents are ALLOWED on left/below; push back those on the right/above
            mask = pos[:, axis_idx] > val
            # Reflect: 2*val - x flips across the barrier line
            pos[mask, axis_idx] = 2.0 * val - pos[mask, axis_idx]
        else:
            # Agents are ALLOWED on right/above; push back those on the left/below
            mask = pos[:, axis_idx] < val
            pos[mask, axis_idx] = 2.0 * val - pos[mask, axis_idx]

        # Clip to ensure positions remain in [0, 1]
        engine.agent_positions = np.clip(pos, 0.0, 1.0)

        reflected = int(np.sum(mask))
        _rebuild_spatial_index(engine)

        logger.info(
            f"[SpatialBarrierPolicy] Reflected {reflected} agents across "
            f"barrier {self.axis}={self.value} (allowed side: {self.side})."
        )
        return {
            "reflected": reflected,
            "axis": self.axis,
            "value": self.value,
            "side": self.side,
        }

    def describe(self) -> str:
        return (
            f"SpatialBarrierPolicy: enforce barrier at {self.axis}={self.value}, "
            f"agents kept on '{self.side}' side."
        )
