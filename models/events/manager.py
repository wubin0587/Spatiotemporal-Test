"""

Event Subsystem Manager (The Facade)
------------------------------------
This module serves as the central controller for the entire Event System.
It acts as the bridge between the Simulation Engine and the various event components 
(Generators, Archive, Utilities).

Responsibilities:
1. Initialization: parsing config to setup specific generators (Exogenous, Cascade, etc.).
2. Execution: stepping through all generators each tick.
3. Storage: managing the EventVectorArchive.
4. Interface: providing high-performance data access for the Engine.

Usage in Engine:
    self.event_manager = EventManager(config)
    new_events = self.event_manager.step(current_time, agents_state)
    impact_vectors = self.event_manager.get_state_vectors()
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Import Data Structures
from .base import Event

# Import Storage
from .archive.vector import EventVectorArchive

# Import Generators
from .generator import EventGenerator
from .generate.exp import ExogenousShockGenerator
from .generate.imp import EndogenousThresholdGenerator
from .generate.cascade import CascadeGenerator

class EventManager:
    """
    The orchestrator class that manages event generation, storage, and retrieval.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Event Manager with the global configuration.

        Args:
            config (dict): The root configuration dictionary. 
                           Must contain an 'events' section.
        """
        self.config = config
        self.event_config = config.get('events', {})
        self.gen_config = self.event_config.get('generation', {})

        # 1. Initialize the Event Archive (High-performance storage)
        self.archive = EventVectorArchive()
        
        # 2. Initialize Generators based on Config
        self.generators: List[EventGenerator] = []
        self._setup_generators()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"EventManager initialized with {len(self.generators)} active generators.")

    def _setup_generators(self):
        """
        Internal method to instantiate generators if they are enabled in config.
        """
        # A. Exogenous Shocks (The "Black Swans")
        exo_conf = self.gen_config.get('exogenous', {})
        if exo_conf.get('enabled', False):
            self.generators.append(ExogenousShockGenerator(exo_conf))
            
        # B. Endogenous Thresholds (The "Grey Rhinos")
        # Note: Passes the full gen_config section or specific subsection
        imp_conf = self.gen_config.get('endogenous_threshold', {})
        if imp_conf.get('enabled', False):
            self.generators.append(EndogenousThresholdGenerator(imp_conf))
            
        # C. Cascading Events (The "Chain Reactions")
        cas_conf = self.gen_config.get('endogenous_cascade', {})
        if cas_conf.get('enabled', False):
            self.generators.append(CascadeGenerator(cas_conf))

    def step(self, 
             current_time: float, 
             agents_state: Any = None, 
             env_state: Any = None) -> List[Event]:
        """
        The main heartbeat method called by the Simulation Engine every tick.
        
        Args:
            current_time: Current simulation timestamp.
            agents_state: Object containing agent positions/emotions (for endogenous logic).
            env_state: Object containing environment map data.
            
        Returns:
            List[Event]: A list of ALL events generated in this tick (aggregated).
        """
        all_new_events = []

        for gen in self.generators:
            # Special handling for CascadeGenerator which needs history
            # We detect it by checking the class type or method signature, 
            # but simpler here is to pass extra kwargs that are ignored by others 
            # or handled explicitly.
            
            new_events = []
            
            if isinstance(gen, CascadeGenerator):
                # Cascade needs history. 
                # Optimization: In a real large-scale sim, pass a filtered view, 
                # but for now we pass the archive wrapper or list.
                # Assuming CascadeGenerator.step accepts `event_history` kwarg.
                # We fetch full object list for logic processing (slower but easier logic).
                history = self.archive.get_all_events() 
                new_events = gen.step(current_time, agents_state, env_state, event_history=history)
            else:
                # Standard Exogenous/Threshold generators
                new_events = gen.step(current_time, agents_state, env_state)
            
            if new_events:
                all_new_events.extend(new_events)

        # 3. Store new events in the Archive
        if all_new_events:
            self.archive.add_events(all_new_events)
            self.logger.debug(f"Tick {current_time}: Generated {len(all_new_events)} new events.")

        return all_new_events

    def get_state_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Exposes the raw NumPy arrays from the archive for vectorized physics calculations.
        
        Returns:
            Tuple: (locations, times, intensities, contents, polarities)
        """
        return self.archive.get_vectors()

    def get_active_event_indices(self, current_time: float, lookback: float = 100.0) -> np.ndarray:
        """
        Helper to get indices of events that are currently relevant (active).
        Wraps the query logic.
        """
        # This would use archive.query logic if implemented there, 
        # or raw numpy masking here.
        # Simple implementation:
        times = self.archive.get_vectors()[1]
        # Events that happened in the past [lookback] window
        mask = (times <= current_time) & (times >= current_time - lookback)
        return np.where(mask)[0]

    def save_log(self, filepath: str):
        """
        Persist the entire event history to disk.
        """
        self.archive.save_to_json(filepath)

    def reset(self):
        """
        Clears the event history for a new experiment run.
        """
        self.archive = EventVectorArchive()
        # Generators usually maintain their own RNG state, which we might want to reset 
        # or keep continuous depending on experiment design. 
        # Re-initializing generators is the safest bet for a hard reset.
        self.generators = []
        self._setup_generators()