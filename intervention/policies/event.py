# -*- coding: utf-8 -*-
"""
@File    : event.py
@Desc    : Event-related Intervention Policies.

Policies that directly manipulate the event subsystem:

    EventSuppressPolicy     -- Suppress new event generation for N steps
    EventInjectPolicy       -- Inject a synthetic event into the archive
    EventAmplifyPolicy      -- Multiply intensities of future events by a factor
    EventFilterPolicy       -- Remove events matching a source/polarity criterion

YAML config examples
--------------------
policy:
  type: event_suppress
  source: exogenous          # 'exogenous' | 'cascade' | 'endogenous_threshold' | 'all'
  duration: 20               # steps to suppress

policy:
  type: event_inject
  location: [0.5, 0.5]      # [x, y] in [0,1]^2
  intensity: 10.0
  polarity: 0.0
  content: [0.33, 0.33, 0.34]   # must match opinion_layers length
  source: intervention

policy:
  type: event_amplify
  factor: 2.0                # multiplier on future event intensities
  duration: 30               # 0 = permanent

policy:
  type: event_filter
  min_polarity: -0.5         # drop events with polarity < this
  max_polarity: 0.5          # drop events with polarity > this
  source: null               # null = all sources
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BasePolicy

logger = logging.getLogger(__name__)


# =============================================================================
# EventSuppressPolicy
# =============================================================================

class EventSuppressPolicy(BasePolicy):
    """
    Temporarily disable event generation for a specific source type.

    This policy works by monkey-patching the relevant generator's
    ``step`` method to return an empty list for ``duration`` steps.

    Parameters (config keys)
    ------------------------
    source : str
        One of ``'exogenous'``, ``'cascade'``, ``'endogenous_threshold'``,
        or ``'all'``.  Default ``'all'``.
    duration : int
        Number of steps to maintain suppression.  ``0`` means permanent
        (until the policy is undone or the generator is re-initialised).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "EventSuppressPolicy")
        self.source = self.config.get("source", "all").lower()
        self.duration = int(self.config.get("duration", 20))
        # Track patched generators so we can restore them
        self._patched: List[Any] = []
        self._restore_step = -1  # engine.time_step when we should un-patch

    # ------------------------------------------------------------------

    def _apply(self, engine: Any) -> Optional[Dict]:
        generators = self._select_generators(engine)

        if not generators:
            logger.warning(
                f"[EventSuppressPolicy] No generators matched source='{self.source}'."
            )
            return {"suppressed": 0}

        self._patched.clear()
        for gen in generators:
            # Save original method and patch with a no-op lambda
            gen._original_step = gen.step
            gen.step = lambda *a, **kw: []
            self._patched.append(gen)

        self._restore_step = (
            engine.time_step + self.duration if self.duration > 0 else -1
        )

        logger.info(
            f"[EventSuppressPolicy] Suppressed {len(generators)} generator(s) "
            f"(source={self.source!r}) for {self.duration} steps."
        )
        return {"suppressed": len(generators), "restore_at_step": self._restore_step}

    def _select_generators(self, engine: Any) -> List[Any]:
        """Return the generator objects that match ``self.source``."""
        from models.events.generate.exp import ExogenousShockGenerator
        from models.events.generate.cascade import CascadeGenerator
        from models.events.generate.imp import EndogenousThresholdGenerator

        _map = {
            "exogenous": ExogenousShockGenerator,
            "cascade": CascadeGenerator,
            "endogenous_threshold": EndogenousThresholdGenerator,
        }

        generators = engine.event_manager.generators

        if self.source == "all":
            return list(generators)

        cls = _map.get(self.source)
        if cls is None:
            logger.warning(f"Unknown source type: '{self.source}'")
            return []
        return [g for g in generators if isinstance(g, cls)]

    def undo(self, engine: Any) -> bool:
        """Restore all suppressed generators to their original step methods."""
        for gen in self._patched:
            if hasattr(gen, "_original_step"):
                gen.step = gen._original_step
                del gen._original_step
        self._patched.clear()
        logger.info("[EventSuppressPolicy] Suppression lifted (undo).")
        return True

    def describe(self) -> str:
        return (
            f"EventSuppressPolicy: suppress '{self.source}' generators "
            f"for {self.duration} steps."
        )


# =============================================================================
# EventInjectPolicy
# =============================================================================

class EventInjectPolicy(BasePolicy):
    """
    Inject a single synthetic event directly into the engine's event archive.

    This is useful for simulating targeted information campaigns or
    sudden exogenous shocks at a precise location.

    Parameters (config keys)
    ------------------------
    location : list[float]
        ``[x, y]`` coordinates in ``[0, 1]^2``.  Default ``[0.5, 0.5]``.
    intensity : float
        Event intensity.  Default ``10.0``.
    polarity : float
        Event polarity in ``[-1, 1]``.  Default ``0.0``.
    content : list[float], optional
        Topic vector.  If omitted, a uniform distribution is used.
    source : str
        Source label on the injected event.  Default ``'intervention'``.
    spatial_sigma : float
        Spatial diffusion parameter stored with the event.  Default ``0.1``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "EventInjectPolicy")
        self.location = self.config.get("location", [0.5, 0.5])
        self.intensity = float(self.config.get("intensity", 10.0))
        self.polarity = float(self.config.get("polarity", 0.0))
        self.content = self.config.get("content", None)
        self.source_label = self.config.get("source", "intervention")
        self.spatial_sigma = float(self.config.get("spatial_sigma", 0.1))

    def _apply(self, engine: Any) -> Optional[Dict]:
        from models.events.base import Event

        # Resolve content vector
        num_layers = engine.opinion_matrix.shape[1]
        if self.content is not None:
            content = np.array(self.content, dtype=np.float32)
            if len(content) != num_layers:
                logger.warning(
                    f"[EventInjectPolicy] content length {len(content)} != "
                    f"num_layers {num_layers}. Padding/truncating."
                )
                c = np.ones(num_layers, dtype=np.float32) / num_layers
                c[: len(content)] = content[: num_layers]
                content = c
        else:
            content = np.ones(num_layers, dtype=np.float32) / num_layers

        # Unique id: use archive count + large offset to avoid collisions
        uid = engine.event_manager.archive.count + 100_000 + self._apply_count

        event = Event(
            uid=uid,
            time=engine.current_time,
            loc=np.array(self.location, dtype=np.float32),
            intensity=self.intensity,
            content=content,
            polarity=self.polarity,
            spatial_params={"sigma": self.spatial_sigma},
            temporal_params={"sigma": 5.0, "mu": 0.0},
            source=self.source_label,
            meta={"injected": True, "step": engine.time_step},
        )

        engine.event_manager.archive.add_events([event])

        logger.info(
            f"[EventInjectPolicy] Injected event uid={uid} at "
            f"loc={self.location} intensity={self.intensity:.1f}."
        )
        return {
            "uid": uid,
            "location": self.location,
            "intensity": self.intensity,
        }

    def describe(self) -> str:
        return (
            f"EventInjectPolicy: inject event at {self.location} "
            f"intensity={self.intensity} polarity={self.polarity}."
        )


# =============================================================================
# EventAmplifyPolicy
# =============================================================================

class EventAmplifyPolicy(BasePolicy):
    """
    Scale the intensities of *all currently archived* events by a factor.

    Because ``EventVectorArchive`` stores intensities in a Python list
    (``_intensities``), this policy mutates those lists directly and marks
    the cache as dirty so the next ``get_vectors()`` call rebuilds arrays.

    Parameters (config keys)
    ------------------------
    factor : float
        Multiplier applied to all event intensities.  Default ``2.0``.
    target : str
        ``'all'`` (default) or a specific source name (``'exogenous'``, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "EventAmplifyPolicy")
        self.factor = float(self.config.get("factor", 2.0))
        self.target = self.config.get("target", "all").lower()

    def _apply(self, engine: Any) -> Optional[Dict]:
        archive = engine.event_manager.archive
        n = archive.count
        if n == 0:
            logger.info("[EventAmplifyPolicy] Archive empty — nothing to amplify.")
            return {"amplified": 0}

        amplified = 0
        for idx in range(n):
            if self.target != "all" and archive._sources[idx] != self.target:
                continue
            archive._intensities[idx] *= self.factor
            amplified += 1

        # Invalidate numpy cache
        archive._dirty = True
        archive._cache_intensities = None

        logger.info(
            f"[EventAmplifyPolicy] Amplified {amplified}/{n} events "
            f"by factor {self.factor:.2f} (target={self.target!r})."
        )
        return {"amplified": amplified, "factor": self.factor}

    def describe(self) -> str:
        return (
            f"EventAmplifyPolicy: multiply event intensities by "
            f"{self.factor}x (target={self.target!r})."
        )


# =============================================================================
# EventFilterPolicy
# =============================================================================

class EventFilterPolicy(BasePolicy):
    """
    Remove events from the archive whose attributes fall outside acceptable bounds.

    Removal is performed by rebuilding the archive's internal lists with only
    the events that pass the filter, then marking the cache dirty.

    Parameters (config keys)
    ------------------------
    min_polarity : float, optional
        Minimum acceptable polarity.  Events below this are removed.
    max_polarity : float, optional
        Maximum acceptable polarity.  Events above this are removed.
    min_intensity : float, optional
        Minimum acceptable intensity.
    source : str, optional
        If set, only events from this source are evaluated (others are kept).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "EventFilterPolicy")
        self.min_polarity = self.config.get("min_polarity", None)
        self.max_polarity = self.config.get("max_polarity", None)
        self.min_intensity = self.config.get("min_intensity", None)
        self.filter_source = self.config.get("source", None)

    def _apply(self, engine: Any) -> Optional[Dict]:
        archive = engine.event_manager.archive
        n_before = archive.count

        if n_before == 0:
            return {"removed": 0, "remaining": 0}

        keep_indices = []
        for idx in range(n_before):
            src = archive._sources[idx]
            pol = archive._polarities[idx]
            intensity = archive._intensities[idx]

            # If a source filter is specified, skip events from other sources
            if self.filter_source is not None and src != self.filter_source:
                keep_indices.append(idx)
                continue

            # Apply polarity bounds
            if self.min_polarity is not None and pol < self.min_polarity:
                continue
            if self.max_polarity is not None and pol > self.max_polarity:
                continue

            # Apply intensity bound
            if self.min_intensity is not None and intensity < self.min_intensity:
                continue

            keep_indices.append(idx)

        removed = n_before - len(keep_indices)

        if removed > 0:
            # Rebuild archive lists from surviving indices
            archive._uids = [archive._uids[i] for i in keep_indices]
            archive._times = [archive._times[i] for i in keep_indices]
            archive._locs = [archive._locs[i] for i in keep_indices]
            archive._intensities = [archive._intensities[i] for i in keep_indices]
            archive._contents = [archive._contents[i] for i in keep_indices]
            archive._polarities = [archive._polarities[i] for i in keep_indices]
            archive._spatial_params = [archive._spatial_params[i] for i in keep_indices]
            archive._temporal_params = [archive._temporal_params[i] for i in keep_indices]
            archive._sources = [archive._sources[i] for i in keep_indices]

            # Invalidate cache
            archive._dirty = True
            archive._cache_locs = None
            archive._cache_times = None
            archive._cache_intensities = None
            archive._cache_contents = None
            archive._cache_polarities = None

        logger.info(
            f"[EventFilterPolicy] Removed {removed} events "
            f"({n_before} → {len(keep_indices)} remaining)."
        )
        return {"removed": removed, "remaining": len(keep_indices)}

    def describe(self) -> str:
        parts = []
        if self.min_polarity is not None:
            parts.append(f"polarity>={self.min_polarity}")
        if self.max_polarity is not None:
            parts.append(f"polarity<={self.max_polarity}")
        if self.min_intensity is not None:
            parts.append(f"intensity>={self.min_intensity}")
        criteria = " AND ".join(parts) if parts else "no criteria"
        return f"EventFilterPolicy: keep events where {criteria}."
