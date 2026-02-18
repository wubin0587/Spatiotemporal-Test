# -*- coding: utf-8 -*-
"""
@File    : core.py
@Desc    : Abstract Base Class for the Simulation Engine.
           Defines the interface and contract that concrete implementations must follow.
           Does NOT contain implementation logic - that belongs in steps.py.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import networkx as nx


class SimulationEngine(ABC):
    """
    Abstract Base Class for the Event-Modulated Spatiotemporal Opinion Dynamics Model.
    
    This class defines the contract for the simulation engine but does not implement
    the actual simulation logic. Concrete implementations should inherit from this
    class and implement all abstract methods.
    
    The engine orchestrates:
    1. Agent state management (opinions, positions)
    2. Event dynamics (via EventManager)
    3. Network topology (static social graph + dynamic spatial)
    4. Opinion update rules (bounded confidence with modulation)
    
    Design Philosophy:
    - Separation of Concerns: This base class only defines WHAT must exist, not HOW.
    - Dependency Injection: External managers (Events, Networks) are injected.
    - State Encapsulation: All simulation state is contained within the engine.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation engine with configuration.
        
        Args:
            config (Dict[str, Any]): The complete configuration dictionary loaded
                                     from YAML. Must contain sections for:
                                     - File-aligned schema:
                                       'engine', 'events', 'networks', 'spatial'
        """
        self.config = config
        
        # Simulation Control
        self.current_time: float = 0.0
        self.time_step: int = 0
        simulation_cfg = self.config.get('engine', {}).get('interface', {}).get('simulation', {})
        self.total_steps: int = simulation_cfg.get('total_steps', 1000)
        
        # State Containers (to be initialized by concrete implementations)
        self.num_agents: int = 0
        self.opinion_matrix: Optional[np.ndarray] = None  # Shape (N, L)
        self.agent_positions: Optional[np.ndarray] = None  # Shape (N, 2)
        self.impact_vector: Optional[np.ndarray] = None   # Shape (N,)
        
        # External Managers (to be injected via interface.py)
        self.event_manager = None
        self.network_graph: Optional[nx.Graph] = None
        self.spatial_index = None  # KDTree or similar
        
        # Cached structures for performance
        self.static_adjacency: Optional[list] = None  # List[List[int]]
        
    @abstractmethod
    def initialize(self):
        """
        Initialize all simulation components and state.
        
        This method should:
        1. Set up agent initial conditions (opinions, positions)
        2. Build network topology
        3. Initialize event generation system
        4. Prepare any cached data structures
        
        Must be called before step() can be executed.
        """
        pass
    
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        Execute one complete simulation time step.
        
        The standard workflow (to be implemented in steps.py) is:
        1. Generate new events (via EventManager)
        2. Calculate impact field I(t) for all agents
        3. Determine interaction pairs (algorithm vs. reality mode)
        4. Update opinions based on bounded confidence + modulation
        5. Increment time
        
        Returns:
            Dict[str, Any]: A dictionary containing step metrics and state snapshots.
                           Useful for logging, visualization, or analysis.
                           Example keys: 'time', 'num_events', 'polarization', 'consensus'
        """
        pass
    
    @abstractmethod
    def run(self, num_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the simulation for a specified number of steps.
        
        Args:
            num_steps (Optional[int]): Number of steps to run. 
                                       If None, runs until self.total_steps.
        
        Returns:
            Dict[str, Any]: Aggregated results and history from the full run.
                           This might include time series of metrics, final state, etc.
        """
        pass
    
    @abstractmethod
    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Returns a complete snapshot of the current simulation state.
        
        This is useful for:
        - Checkpointing (saving simulation progress)
        - Visualization (rendering current state)
        - Analysis (computing metrics at a specific time)
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'time': Current simulation time
                - 'opinions': Copy of opinion matrix (N, L)
                - 'positions': Copy of agent positions (N, 2)
                - 'impact': Copy of impact vector (N,)
                - 'events': Summary of active events
                - Any other relevant state information
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset the simulation to its initial state.
        
        This should:
        1. Reset time to 0
        2. Reinitialize agent opinions and positions
        3. Clear event history
        4. Reset any accumulated statistics
        
        Useful for running multiple experiments with the same configuration.
        """
        pass
    
    # =========================================================================
    # State Access Methods (Concrete implementations can override if needed)
    # =========================================================================
    
    def get_opinion_matrix(self) -> np.ndarray:
        """Returns a copy of the current opinion matrix."""
        if self.opinion_matrix is None:
            raise RuntimeError("Opinion matrix not initialized. Call initialize() first.")
        return self.opinion_matrix.copy()
    
    def get_agent_positions(self) -> np.ndarray:
        """Returns a copy of the current agent positions."""
        if self.agent_positions is None:
            raise RuntimeError("Agent positions not initialized. Call initialize() first.")
        return self.agent_positions.copy()
    
    def get_impact_vector(self) -> np.ndarray:
        """Returns a copy of the current impact field values."""
        if self.impact_vector is None:
            raise RuntimeError("Impact vector not initialized. Call initialize() first.")
        return self.impact_vector.copy()
    
    def get_current_time(self) -> float:
        """Returns the current simulation time."""
        return self.current_time
    
    def get_time_step(self) -> int:
        """Returns the current time step (iteration count)."""
        return self.time_step
    
    # =========================================================================
    # Configuration Access (Helper methods)
    # =========================================================================

    def get_dynamics_params(self) -> Dict[str, Any]:
        """Returns the engine.maths.dynamics configuration section."""
        return self.config.get('engine', {}).get('maths', {}).get('dynamics', {})

    def get_field_params(self) -> Dict[str, Any]:
        """Returns the engine.maths.field configuration section."""
        return self.config.get('engine', {}).get('maths', {}).get('field', {})

    def get_topology_params(self) -> Dict[str, Any]:
        """Returns the engine.maths.topo configuration section."""
        return self.config.get('engine', {}).get('maths', {}).get('topo', {})

    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def _validate_state(self):
        """
        Internal validation to ensure all required state is initialized.
        Should be called at the start of step() or run().
        """
        if self.opinion_matrix is None:
            raise RuntimeError("Opinion matrix is not initialized.")
        if self.agent_positions is None:
            raise RuntimeError("Agent positions are not initialized.")
        if self.event_manager is None:
            raise RuntimeError("Event manager is not initialized.")
        if self.network_graph is None:
            raise RuntimeError("Network graph is not initialized.")
        if self.spatial_index is None:
            raise RuntimeError("Spatial index is not initialized.")
    
    def _validate_config(self):
        """
        Validates strict file-aligned configuration naming.
        Should be called in initialize().
        """
        required_top_level = ['engine', 'events', 'networks', 'spatial']
        missing_top_level = [key for key in required_top_level if key not in self.config]
        if missing_top_level:
            raise ValueError(
                "Configuration missing required top-level sections: "
                f"{missing_top_level}"
            )

        engine_cfg = self.config.get('engine')
        if not isinstance(engine_cfg, dict):
            raise ValueError("Configuration section 'engine' must be a dictionary.")

        interface_cfg = engine_cfg.get('interface')
        if not isinstance(interface_cfg, dict):
            raise ValueError("Configuration section 'engine.interface' must be a dictionary.")

        maths_cfg = engine_cfg.get('maths')
        if not isinstance(maths_cfg, dict):
            raise ValueError("Configuration section 'engine.maths' must be a dictionary.")

        for key in ['agents', 'simulation']:
            if not isinstance(interface_cfg.get(key), dict):
                raise ValueError(f"Configuration section 'engine.interface.{key}' must be a dictionary.")

        for key in ['dynamics', 'field', 'topo']:
            if not isinstance(maths_cfg.get(key), dict):
                raise ValueError(f"Configuration section 'engine.maths.{key}' must be a dictionary.")

        networks_cfg = self.config.get('networks')
        if not isinstance(networks_cfg, dict) or not isinstance(networks_cfg.get('builder'), dict):
            raise ValueError("Configuration section 'networks.builder' must be a dictionary.")

        if not isinstance(self.config.get('events'), dict):
            raise ValueError("Configuration section 'events' must be a dictionary.")

        if not isinstance(self.config.get('spatial'), dict):
            raise ValueError("Configuration section 'spatial' must be a dictionary.")

    # =========================================================================
    # Utility Methods (Can be used by concrete implementations)
    # =========================================================================
    
    def _create_adjacency_list(self) -> list:
        """
        Converts NetworkX graph to adjacency list format for fast lookups.
        
        Returns:
            list: A list where adj_list[i] contains the list of neighbors of node i.
        """
        if self.network_graph is None:
            raise RuntimeError("Network graph not initialized.")
        
        adj_list = [[] for _ in range(self.num_agents)]
        for node in self.network_graph.nodes():
            neighbors = list(self.network_graph.neighbors(node))
            adj_list[node] = neighbors
        
        return adj_list
    
    def __repr__(self):
        """String representation of the engine state."""
        return (f"<SimulationEngine t={self.current_time:.1f} "
                f"step={self.time_step}/{self.total_steps} "
                f"agents={self.num_agents}>")
