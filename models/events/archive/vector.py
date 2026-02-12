import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from ..base import Event

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy data types by converting them
    to their standard Python equivalents.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

class EventVectorArchive:
    """
    A columnar storage system for Events.
    Maintains parallel lists/arrays for high-performance vectorized queries.
    """

    def __init__(self):
        # --- Columnar Storage (Lists for O(1) appends) ---
        self._uids: List[int] = []
        self._times: List[float] = []
        self._locs: List[List[float]] = []
        self._intensities: List[float] = []
        self._contents: List[List[float]] = []
        self._polarities: List[float] = []
        
        # --- Parameter Storage (Complex objects) ---
        self._spatial_params: List[Dict[str, Any]] = []
        self._temporal_params: List[Dict[str, Any]] = []
        self._sources: List[str] = []

        # --- Cache for Numpy Views ---
        self._dirty = True
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
            self._locs.append(e.loc.tolist() if isinstance(e.loc, np.ndarray) else list(e.loc))
            self._intensities.append(e.intensity)
            self._contents.append(e.content.tolist() if isinstance(e.content, np.ndarray) else list(e.content))
            self._polarities.append(e.polarity)
            self._spatial_params.append(e.spatial_params)
            self._temporal_params.append(e.temporal_params)
            self._sources.append(e.source)
        
        self._dirty = True

    @property
    def count(self) -> int:
        """Returns the total number of events stored."""
        return len(self._uids)

    def get_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns read-only numpy arrays for the core attributes.
        This is the high-performance API for physics calculations.
        """
        if self._dirty or self._cache_locs is None:
            self._rebuild_cache()
            
        return (
            self._cache_locs,
            self._cache_times,
            self._cache_intensities,
            self._cache_contents,
            self._cache_polarities
        )

    def _rebuild_cache(self):
        """Internal method to convert storage lists into cached numpy arrays."""
        if self.count == 0:
            content_dim = 0
            self._cache_locs = np.empty((0, 2), dtype=np.float32)
            self._cache_times = np.array([], dtype=np.float32)
            self._cache_intensities = np.array([], dtype=np.float32)
            self._cache_contents = np.empty((0, content_dim), dtype=np.float32)
            self._cache_polarities = np.array([], dtype=np.float32)
        else:
            self._cache_locs = np.array(self._locs, dtype=np.float32)
            self._cache_times = np.array(self._times, dtype=np.float32)
            self._cache_intensities = np.array(self._intensities, dtype=np.float32)
            self._cache_contents = np.array(self._contents, dtype=np.float32)
            self._cache_polarities = np.array(self._polarities, dtype=np.float32)
        
        self._dirty = False
    
    def get_event_by_index(self, idx: int) -> Event:
        """
        Reconstructs a single Event object from the columnar data.
        """
        if not (0 <= idx < self.count):
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
        """Reconstructs all Event objects from the archive."""
        return [self.get_event_by_index(i) for i in range(self.count)]

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
                "spatial_params": self._spatial_params,
                "temporal_params": self._temporal_params,
                "sources": self._sources
            }
        }

    def save_to_json(self, filepath: str):
        """
        Dumps the archive to a JSON file, handling NumPy data types.
        """
        if self.count == 0:
            return
            
        # Ensure the target directory exists
        dir_name = os.path.dirname(filepath)
        if dir_name:
             os.makedirs(dir_name, exist_ok=True)
        
        data_to_save = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
            
        print(f"[EventArchive] Saved {self.count} events to {filepath}")