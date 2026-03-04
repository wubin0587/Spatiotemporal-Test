# -*- coding: utf-8 -*-
"""
@File    : checkpoint.py
@Desc    : Checkpoint (immutable state snapshot) and BranchManager.

These two classes together enable counterfactual / branching analysis:

    t=0 ──────── t=50 ──────── t=100   (baseline / control)
                   │
                   ├─[policy A]─────── t=100   (intervention branch A)
                   └─[policy B]─────── t=100   (intervention branch B)

Checkpoint
----------
An immutable data container that captures a complete serializable snapshot
of the engine at a given moment.  It stores:
  - opinion_matrix   (N, L) float32
  - agent_positions  (N, 2) float32
  - impact_vector    (N,)   float32
  - current_time     float
  - time_step        int
  - event_archive    serialized dict from EventVectorArchive.to_dict()

The event archive is the trickiest part: we deep-copy the archive's internal
lists so that future event generation does not corrupt the snapshot.

BranchManager
-------------
Attached to a live SimulationFacade (or directly to a StepExecutor).
Provides:
  create_checkpoint()  → branch_id (str)
  restore(branch_id)   → restores engine to that snapshot in-place
  list_checkpoints()   → metadata summary of all stored snapshots
  delete(branch_id)    → free memory
  compare(id_a, id_b)  → delta dict between two snapshots

Usage
-----
    mgr = BranchManager(sim)               # sim is a SimulationFacade
    baseline_id = mgr.create_checkpoint()  # save t=50 state

    # Branch A: apply intervention, run, collect results
    mgr.restore(baseline_id)
    policy.apply(sim._engine)
    results_A = sim.run(num_steps=50)

    # Branch B: control group
    mgr.restore(baseline_id)
    results_B = sim.run(num_steps=50)

    delta = mgr.compare(baseline_id, baseline_id)  # trivially zero
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint — immutable snapshot
# =============================================================================

@dataclass(frozen=True)
class Checkpoint:
    """
    Immutable snapshot of engine state at a specific simulation moment.

    All mutable numpy arrays are stored as copies so the checkpoint is
    independent of subsequent engine mutations.

    Attributes
    ----------
    branch_id : str
        UUID identifying this checkpoint.
    time_step : int
        Engine ``time_step`` counter at the moment of capture.
    current_time : float
        Engine ``current_time`` at the moment of capture.
    opinion_matrix : np.ndarray
        Shape (N, L).  Copy of opinions at capture time.
    agent_positions : np.ndarray
        Shape (N, 2).  Copy of positions at capture time.
    impact_vector : np.ndarray
        Shape (N,).  Copy of impact field at capture time.
    event_archive_dict : dict
        Serialised representation of EventVectorArchive (from ``.to_dict()``).
        Stored as a plain Python dict so it is JSON-serialisable and
        independent of the live archive object.
    label : str
        Optional human-readable description.
    meta : dict
        Arbitrary extra metadata (e.g. polarization at capture time).
    """

    branch_id: str
    time_step: int
    current_time: float
    opinion_matrix: np.ndarray
    agent_positions: np.ndarray
    impact_vector: np.ndarray
    event_archive_dict: dict
    label: str = ""
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_agents(self) -> int:
        return self.opinion_matrix.shape[0]

    @property
    def num_layers(self) -> int:
        return self.opinion_matrix.shape[1]

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight metadata dict (no large arrays)."""
        return {
            "branch_id": self.branch_id,
            "label": self.label,
            "time_step": self.time_step,
            "current_time": self.current_time,
            "num_agents": self.num_agents,
            "num_layers": self.num_layers,
            "num_events": self.event_archive_dict.get("count", 0),
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return (
            f"<Checkpoint id={self.branch_id[:8]}… "
            f"step={self.time_step} t={self.current_time:.1f} "
            f"label={self.label!r}>"
        )


# =============================================================================
# BranchManager — creates and restores checkpoints
# =============================================================================

class BranchManager:
    """
    Manages a registry of Checkpoint objects for a single simulation.

    Parameters
    ----------
    sim_or_engine : SimulationFacade | StepExecutor
        Accepts either a facade (preferred) or a raw engine for flexibility.
        The manager will detect which one it received.

    Notes
    -----
    The network graph (``engine.network_graph``) and spatial index
    (``engine.spatial_index``) are intentionally **not** included in the
    checkpoint.  Both are expensive to clone and are considered invariant
    across branches — policies that modify the graph (``network.py``) must
    rebuild the adjacency list themselves after restoration if needed.

    If a policy modifies ``agent_positions``, the spatial KDTree becomes
    stale.  The manager calls ``_rebuild_spatial_index()`` on the engine
    after restoring a checkpoint when positions have changed.
    """

    def __init__(self, sim_or_engine: Any):
        # Accept facade or raw engine
        if hasattr(sim_or_engine, "_engine"):
            # It is a SimulationFacade
            self._facade = sim_or_engine
            self._engine = sim_or_engine._engine
        else:
            # Assume it is a StepExecutor directly
            self._facade = None
            self._engine = sim_or_engine

        self._checkpoints: Dict[str, Checkpoint] = {}

        logger.info("BranchManager created.")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def create_checkpoint(self, label: str = "", meta: Optional[Dict] = None) -> str:
        """
        Capture the current engine state as an immutable Checkpoint.

        Parameters
        ----------
        label : str
            Optional descriptive name (e.g. ``'pre_intervention'``).
        meta : dict, optional
            Extra metadata to embed in the checkpoint (e.g. current metrics).

        Returns
        -------
        str
            The ``branch_id`` UUID string — use this to restore later.
        """
        engine = self._engine
        branch_id = str(uuid.uuid4())

        # Serialize the event archive as a plain dict.
        # deep-copy the lists so future appends to the live archive
        # do not affect the snapshot.
        archive_dict = copy.deepcopy(engine.event_manager.archive.to_dict())

        # Build snapshot meta
        snapshot_meta = meta or {}
        snapshot_meta.setdefault("opinion_std", float(np.std(engine.opinion_matrix)))
        snapshot_meta.setdefault("mean_impact", float(np.mean(engine.impact_vector)))

        cp = Checkpoint(
            branch_id=branch_id,
            time_step=engine.time_step,
            current_time=engine.current_time,
            opinion_matrix=engine.opinion_matrix.copy(),
            agent_positions=engine.agent_positions.copy(),
            impact_vector=engine.impact_vector.copy(),
            event_archive_dict=archive_dict,
            label=label,
            meta=snapshot_meta,
        )

        self._checkpoints[branch_id] = cp

        logger.info(
            f"Checkpoint created: id={branch_id[:8]}… "
            f"step={cp.time_step} t={cp.current_time:.1f} label={label!r}"
        )
        return branch_id

    def restore(self, branch_id: str) -> None:
        """
        Restore the engine to the state captured in the given checkpoint.

        This overwrites the engine's mutable state in-place so that the
        facade continues to point to the same engine object.

        The event archive is fully reconstructed from the serialised dict.

        Parameters
        ----------
        branch_id : str
            ID returned by ``create_checkpoint()``.

        Raises
        ------
        KeyError
            If ``branch_id`` is not found in the registry.
        """
        if branch_id not in self._checkpoints:
            raise KeyError(f"No checkpoint with id '{branch_id}'.")

        cp = self._checkpoints[branch_id]
        engine = self._engine

        logger.info(
            f"Restoring checkpoint id={branch_id[:8]}… "
            f"(step={cp.time_step}, t={cp.current_time:.1f})"
        )

        # Restore scalar state
        engine.time_step = cp.time_step
        engine.current_time = cp.current_time

        # Restore arrays (in-place replacement — keeps object identity)
        engine.opinion_matrix = cp.opinion_matrix.copy()
        engine.agent_positions = cp.agent_positions.copy()
        engine.impact_vector = cp.impact_vector.copy()

        # Rebuild spatial index if positions were part of the snapshot
        # (always safe to rebuild; cheap for typical N < 10k)
        self._rebuild_spatial_index(engine)

        # Restore event archive from serialised dict
        self._restore_event_archive(engine, cp.event_archive_dict)

        # Clear the engine's history buffer so recorded history starts fresh
        # from the restored point rather than mixing pre/post-restore data.
        engine.history = {
            "time": [],
            "opinions": [],
            "impact": [],
            "num_events": [],
        }

        logger.info("Restore complete.")

    def delete(self, branch_id: str) -> None:
        """
        Remove a checkpoint from the registry to free memory.

        Parameters
        ----------
        branch_id : str
            ID to delete.
        """
        if branch_id not in self._checkpoints:
            logger.warning(f"delete(): no checkpoint with id '{branch_id}'.")
            return
        del self._checkpoints[branch_id]
        logger.info(f"Checkpoint {branch_id[:8]}… deleted.")

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Return a list of lightweight summary dicts (no large arrays).

        Returns
        -------
        list[dict]
            Sorted by ``time_step`` ascending.
        """
        summaries = [cp.summary() for cp in self._checkpoints.values()]
        return sorted(summaries, key=lambda s: s["time_step"])

    def get_checkpoint(self, branch_id: str) -> Checkpoint:
        """
        Retrieve the Checkpoint object directly (read-only).

        Raises
        ------
        KeyError
        """
        if branch_id not in self._checkpoints:
            raise KeyError(f"No checkpoint with id '{branch_id}'.")
        return self._checkpoints[branch_id]

    def compare(self, id_a: str, id_b: str) -> Dict[str, Any]:
        """
        Compute the delta between two checkpoints.

        Returns a dict describing differences in:
          - step gap
          - time gap
          - opinion drift (mean absolute difference per agent-dimension)
          - position drift (mean Euclidean displacement)
          - impact drift (mean absolute difference)
          - event count difference

        Parameters
        ----------
        id_a, id_b : str
            Branch IDs to compare (order matters for signed differences).

        Returns
        -------
        dict
        """
        cp_a = self.get_checkpoint(id_a)
        cp_b = self.get_checkpoint(id_b)

        opinion_diff = np.mean(np.abs(cp_b.opinion_matrix - cp_a.opinion_matrix))
        position_diff = np.mean(
            np.linalg.norm(cp_b.agent_positions - cp_a.agent_positions, axis=1)
        )
        impact_diff = np.mean(np.abs(cp_b.impact_vector - cp_a.impact_vector))

        return {
            "step_gap": cp_b.time_step - cp_a.time_step,
            "time_gap": cp_b.current_time - cp_a.current_time,
            "opinion_mad": float(opinion_diff),
            "position_displacement": float(position_diff),
            "impact_mad": float(impact_diff),
            "event_count_a": cp_a.event_archive_dict.get("count", 0),
            "event_count_b": cp_b.event_archive_dict.get("count", 0),
            "event_count_delta": (
                cp_b.event_archive_dict.get("count", 0)
                - cp_a.event_archive_dict.get("count", 0)
            ),
        }

    @property
    def checkpoint_count(self) -> int:
        """Number of stored checkpoints."""
        return len(self._checkpoints)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_spatial_index(self, engine: Any) -> None:
        """Rebuild the KDTree spatial index from current agent positions."""
        from scipy.spatial import cKDTree
        engine.spatial_index = cKDTree(engine.agent_positions)
        logger.debug("Spatial index rebuilt after checkpoint restore.")

    def _restore_event_archive(self, engine: Any, archive_dict: dict) -> None:
        """
        Reconstruct the EventVectorArchive from its serialised dict representation.

        Rather than importing EventVectorArchive here (which would create a
        circular dependency path), we reconstruct by directly populating
        the archive's internal lists and marking the cache as dirty.
        """
        archive = engine.event_manager.archive

        data = archive_dict.get("data", {})

        # Wipe existing data
        archive._uids = list(data.get("uids", []))
        archive._times = list(data.get("times", []))
        archive._locs = list(data.get("locs", []))
        archive._intensities = list(data.get("intensities", []))
        archive._contents = list(data.get("contents", []))
        archive._polarities = list(data.get("polarities", []))
        archive._spatial_params = list(data.get("spatial_params", []))
        archive._temporal_params = list(data.get("temporal_params", []))
        archive._sources = list(data.get("sources", []))

        # Invalidate numpy cache so it is rebuilt on next get_vectors() call
        archive._dirty = True
        archive._cache_locs = None
        archive._cache_times = None
        archive._cache_intensities = None
        archive._cache_contents = None
        archive._cache_polarities = None

        logger.debug(
            f"Event archive restored: {archive.count} events."
        )

    def __repr__(self) -> str:
        return (
            f"<BranchManager checkpoints={self.checkpoint_count}>"
        )
