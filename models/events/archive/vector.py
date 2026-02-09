"""
D:\Tiktok\models\events\archive\vector.py

Efficient Event Vector Archive (SoA - Structure of Arrays)
----------------------------------------------------------
This module implements a high-performance storage container for events.
Instead of storing a list of Event objects (which is slow for computation),
it creates parallel NumPy arrays for each attribute (Location, Time, Intensity, etc.).

Benefits:
1. Vectorization: Allows calculating the impact of ALL events on ALL agents simultaneously.
2. Serialization: Native support for dumping history to JSON for playback/analysis.
"""

import numpy as np
import json
import os
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
from ..base import Event

class EventVectorArchive:
    """
    A columnar storage system for Events.
    Maintains parallel lists/arrays for high-performance vectorized queries.
    """

    def __init__(self):
        # --- Columnar Storage (Lists for O(1) appends) ---
        # We use lists for storage and convert to numpy for calculation on demand.
        self._uids: List[int] = []
        self._times: List[float] = []
        self._locs: List[List[float]] = []      # Stored as list of [x, y]
        self._intensities: List[float] = []
        self._contents: List[List[float]] = []  # Stored as list of vectors
        self._polarities: List[float] = []
        
        # --- Parameter Storage (Complex objects) ---
        # Spatial/Temporal params are dictionaries, harder to vectorize perfectly,
        # but we store them aligned by index.
        self._spatial_params: List[Dict[str, Any]] = []
        self._temporal_params: List[Dict[str, Any]] = []
        self._sources: List[str] = []

        # --- Cache for Numpy Views ---
        # Used to avoid rebuilding arrays every time if no new events were added.
        self._dirty = False
        self._cache_locs: Optional[np.ndarray] = None
        self._cache_times: Optional[np.ndarray] = None
        self._cache_intensities: Optional[np.ndarray] = None
        self._cache_contents: Optional[np.ndarray] = None
        self._cache_polarities: Optional[np.ndarray] = None

    def add_events(self, events: List[Event]):
        """
        Batch insert new events into the archive.
        Deconstructs the Event objects into columns.
        """
        if not events:
            return

        for e in events:
            self._uids.append(e.uid)
            self._times.append(e.time)
            self._locs.append(e.loc.tolist() if isinstance(e.loc, np.ndarray) else e.loc)
            self._intensities.append(e.intensity)
            self._contents.append(e.content.tolist() if isinstance(e.content, np.ndarray) else e.content)
            self._polarities.append(e.polarity)
            self._spatial_params.append(e.spatial_params)
            self._temporal_params.append(e.temporal_params)
            self._sources.append(e.source)

        # Mark cache as invalid because data changed
        self._dirty = True

    @property
    def count(self) -> int:
        """Returns the total number of events stored."""
        return len(self._uids)

    # =========================================================================
    # Vectorized Accessors (The "High Performance" API)
    # =========================================================================

    def get_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns read-only numpy arrays for the core attributes.
        Use this for physics calculations (e.g., calculating influence fields).
        
        Returns:
            Tuple containing:
            - locations: (N, 2)
            - times: (N,)
            - intensities: (N,)
            - contents: (N, D)
            - polarities: (N,)
        """
        if self._dirty:
            self._rebuild_cache()
            
        return (
            self._cache_locs,
            self._cache_times,
            self._cache_intensities,
            self._cache_contents,
            self._cache_polarities
        )

    def _rebuild_cache(self):
        """Internal method to convert lists to numpy arrays."""
        if self.count == 0:
            # Handle empty state
            self._cache_locs = np.empty((0, 2))
            self._cache_times = np.array([])
            self._cache_intensities = np.array([])
            self._cache_contents = np.empty((0, 0))
            self._cache_polarities = np.array([])
        else:
            self._cache_locs = np.array(self._locs, dtype=np.float32)
            self._cache_times = np.array(self._times, dtype=np.float32)
            self._cache_intensities = np.array(self._intensities, dtype=np.float32)
            self._cache_contents = np.array(self._contents, dtype=np.float32)
            self._cache_polarities = np.array(self._polarities, dtype=np.float32)
        
        self._dirty = False

    # =========================================================================
    # Object Accessors (The "Human Friendly" API)
    # =========================================================================
    
    def get_event_by_index(self, idx: int) -> Event:
        """
        Reconstructs a single Event object from the columnar data.
        Useful for debugging or detailed inspection of a specific event.
        """
        if idx < 0 or idx >= self.count:
            raise IndexError("Event index out of bounds")

        return Event(
            uid=self._uids[idx],
            time=self._times[idx],
            loc=np.array(self._locs[idx]),
            intensity=self._intensities[idx],
            content=np.array(self._contents[idx]),
            polarity=self._polarities[idx],
            spatial_params=self._spatial_params[idx],
            temporal_params=self._temporal_params[idx],
            source=self._sources[idx]
        )

    def get_all_events(self) -> List[Event]:
        """Reconstructs ALL Event objects. Warning: Potentially slow."""
        return [self.get_event_by_index(i) for i in range(self.count)]

    # =========================================================================
    # Serialization (JSON Output)
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the entire archive into a dictionary suitable for JSON serialization.
        """
        return {
            "count": self.count,
            "data": {
                "uids": self._uids,
                "times": self._times,
                "locs": self._locs,
                "intensities": self._intensities,
                "contents": self._contents,
                "polarities": self._polarities,
                "spatial_params": self._convert_params_for_json(self._spatial_params),
                "temporal_params": self._convert_params_for_json(self._temporal_params),
                "sources": self._sources
            }
        }

    def save_to_json(self, filepath: str):
        """
        Dumps the archive to a JSON file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"[EventArchive] Saved {self.count} events to {filepath}")

    def _convert_params_for_json(self, params_list: List[Dict]) -> List[Dict]:
        """
        Helper to ensure all values in param dictionaries are JSON serializable.
        (e.g., convert numpy types to python float/int).
        """
        clean_list = []
        for p in params_list:
            clean_p = {}
            for k, v in p.items():
                if isinstance(v, (np.integer, int)):
                    clean_p[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_p[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_p[k] = v.tolist()
                else:
                    clean_p[k] = v
            clean_list.append(clean_p)
        return clean_list