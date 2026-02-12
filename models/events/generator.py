"""
D:\Tiktok\models\events\generator.py

Abstract Base Class for Event Generators.
-----------------------------------------
This module defines the interface that all specific event generators
(e.g., ExogenousShock, Cascade) must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

# Import the data structure from base
from .base import Event

class EventGenerator(ABC):
    """
    Abstract Base Class for all event generation mechanisms.
    
    Responsibilities:
    1. Initialize with a specific config section.
    2. Maintain a random number generator (RNG) for reproducibility.
    3. Implement `step()` to return new events based on simulation state.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the generator.

        Args:
            config (dict): Configuration dictionary specific to this generator module.
                           Must contain at least mechanism-specific params.
                           If 'seed' is present, it initializes the RNG.
        """
        self.config = config
        
        # Setup Reproducibility
        # Each generator gets its own RNG state to ensure that adding/removing
        # one generator type doesn't mess up the random sequence of another.
        seed = config.get('seed', 42)
        self.rng = np.random.default_rng(seed)
        
        # Internal counter for generating unique Event IDs within this source
        self.event_count = 0

    @abstractmethod
    def step(self, current_time: float, agents_state: Any = None, env_state: Any = None) -> List[Event]:
        """
        Execute one simulation step to potentially generate new events.

        Args:
            current_time (float): The current global simulation time.
            agents_state (Any, optional): Access to agent data (e.g., positions, emotions).
                                          Required for Endogenous/Cascade events.
            env_state (Any, optional): Access to map/topology data.

        Returns:
            List[Event]: A list of newly created Event objects. 
                         Returns an empty list [] if no event triggers.
        """
        pass

    def _get_next_id(self) -> int:
        """
        Helper to manage unique IDs within this generator instance.
        """
        self.event_count += 1
        return self.event_count