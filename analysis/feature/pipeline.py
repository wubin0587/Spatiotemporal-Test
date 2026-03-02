"""
analysis/feature/pipeline.py

Feature Pipeline
----------------
Orchestrates the full feature extraction workflow for a completed simulation run.

Typical usage
-------------
    from analysis.feature.pipeline import FeaturePipeline
    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_file('config.yaml')
    results = sim.run()

    pipe = FeaturePipeline(sim._engine)
    features = pipe.run()
    # features['timeseries']  – per-step time-series arrays
    # features['summary']     – aggregate stats
    # features['final']       – features of the last snapshot

Data flow
---------
    SimulationEngine  ──▶  snapshots  ──▶  extractor.extract_all_features()
                                       ──▶  composer.compose_timeseries()
                                       ──▶  composer.summarize_timeseries()

DATA STRUCTURE CONTRACT
-----------------------
    Simulation history (when record_history=True):
        history['time']     : List[float]           length T
        history['opinions'] : List[np.ndarray (N,L)] length T
        history['impact']   : List[np.ndarray (N,)]  length T

    All arrays validated by extractor before metric calls.
    graph node IDs guaranteed integer [0, N-1] by EngineInterface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

from .extractor import extract_all_features
from .composer import compose_timeseries, summarize_timeseries, compose_multilayer


class FeaturePipeline:
    """
    Wraps a completed StepExecutor (or any object exposing the same interface)
    and produces a comprehensive feature set.

    Parameters
    ----------
    engine : StepExecutor  (or duck-typed equivalent)
        Must expose:
            .opinion_matrix     np.ndarray (N, L)
            .agent_positions    np.ndarray (N, 2)
            .impact_vector      np.ndarray (N,)
            .network_graph      nx.Graph
            .event_manager      EventManager
            .history            dict  (populated when record_history=True)
            .current_time       float
            .time_step          int
    layer_idx : int
        Primary opinion layer for scalar summaries (default 0).
    """

    def __init__(self, engine: Any, layer_idx: int = 0):
        self.engine = engine
        self.layer_idx = layer_idx

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Returns
        -------
        dict with keys:
            'final'      – features of the current (final) engine state
            'timeseries' – per-key arrays over recorded history (empty if no history)
            'summary'    – aggregate statistics over timeseries
            'data_issues'– list of structural warnings found during extraction
        """
        # 1. Final snapshot features
        final_feats = self._extract_snapshot(
            opinions  = self.engine.opinion_matrix,
            positions = self.engine.agent_positions,
            impact    = self.engine.impact_vector,
            time      = self.engine.current_time,
            step      = self.engine.time_step,
        )

        # 2. History-based time-series (only if history was recorded)
        step_features: List[Dict[str, Any]] = []
        history = getattr(self.engine, "history", {})

        if history.get("time"):
            step_features = self._extract_history(history)

        timeseries = compose_timeseries(step_features) if step_features else {}
        summary    = summarize_timeseries(timeseries)   if timeseries else {}

        # 3. Collect all structural warnings
        all_issues: List[str] = list(final_feats["meta"].get("data_issues", []))

        return {
            "final":       final_feats,
            "timeseries":  timeseries,
            "summary":     summary,
            "data_issues": all_issues,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_event_vectors(self):
        """
        Pull event data from the engine's EventManager.

        Returns (event_times, event_intensities, event_locs) as np.ndarray or None.

        DATA STRUCTURE VALIDITY:
            EventVectorArchive.get_vectors() returns 5-tuple:
                (locs (M,2), times (M,), intensities (M,), contents (M,L), polarities (M,))
            We expose times, intensities, locs to the extractor.
            Empty archive returns empty arrays, not None.
        """
        try:
            locs, times, intensities, _, _ = (
                self.engine.event_manager.get_state_vectors()
            )
            if times is None or len(times) == 0:
                return None, None, None
            return (
                np.asarray(times,       dtype=float),
                np.asarray(intensities, dtype=float),
                np.asarray(locs,        dtype=float),   # shape (M, 2)
            )
        except Exception:
            return None, None, None

    def _extract_snapshot(
        self,
        opinions:  np.ndarray,
        positions: np.ndarray,
        impact:    np.ndarray,
        time:      float,
        step:      int,
    ) -> Dict[str, Any]:
        """Build the snapshot dict and call extract_all_features."""
        snapshot = {
            "opinions":  opinions,
            "positions": positions,
            "impact":    impact,
            "time":      time,
            "step":      step,
        }
        event_times, event_intensities, event_locs = self._get_event_vectors()

        return extract_all_features(
            snapshot          = snapshot,
            graph             = self.engine.network_graph,
            event_times       = event_times,
            event_intensities = event_intensities,
            event_locs        = event_locs,
            layer_idx         = self.layer_idx,
        )

    def _extract_history(
        self, history: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """
        Run extractor over every recorded time step.

        DATA STRUCTURE VALIDITY:
            history['opinions'][t] : np.ndarray (N, L)
            history['impact'][t]   : np.ndarray (N,)
            history['time'][t]     : float
            history['num_events'][t]: int  (not used here directly)

            All three lists must have the same length T.
            Mismatched lengths are guarded: we zip and stop at the shortest.
        """
        opinions_hist = history.get("opinions", [])
        impact_hist   = history.get("impact",   [])
        time_hist     = history.get("time",     [])

        # Guard: use shortest list length
        T = min(len(opinions_hist), len(impact_hist), len(time_hist))
        positions = self.engine.agent_positions   # static across steps

        step_feats: List[Dict[str, Any]] = []
        for t in range(T):
            feats = self._extract_snapshot(
                opinions  = opinions_hist[t],
                positions = positions,
                impact    = impact_hist[t],
                time      = time_hist[t],
                step      = t,
            )
            step_feats.append(feats)

        return step_feats


# ═════════════════════════════════════════════════════════════════════════════
# Convenience function
# ═════════════════════════════════════════════════════════════════════════════

def run_feature_pipeline(engine: Any, layer_idx: int = 0) -> Dict[str, Any]:
    """
    One-liner wrapper around FeaturePipeline.

    Example
    -------
        features = run_feature_pipeline(sim._engine)
        print(features['summary']['opinion.polarization_std'])
    """
    return FeaturePipeline(engine, layer_idx=layer_idx).run()
