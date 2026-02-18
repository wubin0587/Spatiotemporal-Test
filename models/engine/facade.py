# -*- coding: utf-8 -*-
"""
@File    : facade.py
@Desc    : Facade Pattern Implementation for the Simulation Engine.
           Provides a simplified, high-level API for users to interact with
           the complex simulation system without needing to understand internal details.
           
           This is the ONLY class that external code should instantiate and use.
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import networkx as nx

# Import the concrete implementation
from .steps import StepExecutor

logger = logging.getLogger(__name__)


class SimulationFacade:
    """
    Facade class that provides a simplified interface to the
    Event-Modulated Spatiotemporal Opinion Dynamics Model.
    
    This class hides the complexity of the internal engine, interface layer,
    and mathematical modules, exposing only essential operations:
    - Loading configuration
    - Running simulations
    - Accessing results
    - Saving/loading state
    
    Example Usage:
    --------------
    # Create simulation from config file
    sim = SimulationFacade.from_config_file('config.yaml')
    
    # Run simulation
    results = sim.run(num_steps=1000)
    
    # Access results
    final_opinions = sim.get_final_opinions()
    
    # Save results
    sim.save_results('output.npz')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation facade with configuration.
        
        Args:
            config (Dict[str, Any]): Complete configuration dictionary.
                                     Must contain all required sections.
        
        Note:
            It's recommended to use the factory methods (from_config_file,
            from_config_dict) rather than calling this constructor directly.
        """
        self.config = config
        self._validate_config_schema()
        self._engine: Optional[StepExecutor] = None
        self._initialized: bool = False
        self._results: Optional[Dict[str, Any]] = None
        
        logger.info("SimulationFacade created")

    def _validate_config_schema(self):
        """
        Validate strict file-aligned config naming.

        Required top-level keys:
            - engine
            - events
            - networks
            - spatial

        Required engine sub-keys:
            - engine.interface.agents
            - engine.interface.simulation
            - engine.maths.dynamics
            - engine.maths.field
            - engine.maths.topo
        """
        required_top_level = ['engine', 'events', 'networks', 'spatial']
        missing = [k for k in required_top_level if k not in self.config]
        if missing:
            raise ValueError(
                "Configuration missing top-level sections: " + ', '.join(missing)
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
            raise ValueError(
                "Configuration section 'networks.builder' must be provided and be a dictionary."
            )

    # =========================================================================
    # Factory Methods (Recommended Entry Points)
    # =========================================================================
    
    @classmethod
    def from_config_file(cls, filepath: Union[str, Path]) -> 'SimulationFacade':
        """
        Create a SimulationFacade from a YAML configuration file.
        
        Args:
            filepath: Path to the YAML configuration file.
        
        Returns:
            SimulationFacade: Initialized facade instance.
        
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        
        Example:
            >>> sim = SimulationFacade.from_config_file('experiments/config.yaml')
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        logger.info(f"Loading configuration from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Configuration file is empty: {filepath}")

        return cls(config)
    
    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationFacade':
        """
        Create a SimulationFacade from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary.
        
        Returns:
            SimulationFacade: Initialized facade instance.
        
        Example:
            >>> config = {'agents': {...}, 'network': {...}, ...}
            >>> sim = SimulationFacade.from_config_dict(config)
        """
        return cls(config_dict)
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def initialize(self):
        """
        Initialize the simulation engine and all subsystems.
        
        This must be called before run() or step().
        
        Workflow:
        1. Create the StepExecutor engine
        2. Initialize agents, networks, events, and spatial structures
        3. Validate all components
        
        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            logger.warning("Simulation already initialized. Skipping.")
            return
        
        logger.info("Initializing simulation...")
        
        try:
            # Create the engine
            self._engine = StepExecutor(self.config)
            
            # Run engine initialization
            self._engine.initialize()
            
            self._initialized = True
            logger.info("Simulation initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize simulation: {e}")
    
    def is_initialized(self) -> bool:
        """
        Check if the simulation has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise.
        """
        return self._initialized
    
    # =========================================================================
    # Simulation Execution
    # =========================================================================
    
    def run(self, num_steps: Optional[int] = None, 
            auto_initialize: bool = True) -> Dict[str, Any]:
        """
        Run the complete simulation.
        
        Args:
            num_steps: Number of time steps to execute.
                      If None, uses the value from configuration.
            auto_initialize: If True, automatically calls initialize()
                           if not already initialized.
        
        Returns:
            Dict[str, Any]: Simulation results including:
                - 'final_time': Final simulation time
                - 'total_steps': Number of steps executed
                - 'final_opinions': Final opinion matrix
                - 'final_positions': Final agent positions
                - 'history': Time series data (if enabled in config)
        
        Example:
            >>> sim = SimulationFacade.from_config_file('config.yaml')
            >>> results = sim.run(num_steps=1000)
            >>> print(f"Simulation completed at t={results['final_time']}")
        """
        # Auto-initialize if needed
        if not self._initialized:
            if auto_initialize:
                logger.info("Auto-initializing simulation...")
                self.initialize()
            else:
                raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        # Execute simulation
        logger.info("Starting simulation execution...")
        self._results = self._engine.run(num_steps=num_steps)
        
        logger.info(f"Simulation completed: {self._results['total_steps']} steps")
        
        return self._results
    
    def step(self, auto_initialize: bool = True) -> Dict[str, Any]:
        """
        Execute a single simulation time step.
        
        Useful for step-by-step execution, debugging, or interactive use.
        
        Args:
            auto_initialize: If True, automatically calls initialize()
                           if not already initialized.
        
        Returns:
            Dict[str, Any]: Step statistics.
        
        Example:
            >>> sim = SimulationFacade.from_config_file('config.yaml')
            >>> for _ in range(100):
            >>>     stats = sim.step()
            >>>     print(f"Step {stats['step']}: impact={stats['mean_impact']:.3f}")
        """
        # Auto-initialize if needed
        if not self._initialized:
            if auto_initialize:
                self.initialize()
            else:
                raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        # Execute one step
        return self._engine.step()
    
    def reset(self):
        """
        Reset the simulation to its initial state.
        
        This allows running multiple experiments with the same configuration
        without recreating the facade.
        
        Example:
            >>> sim = SimulationFacade.from_config_file('config.yaml')
            >>> results1 = sim.run(num_steps=500)
            >>> sim.reset()
            >>> results2 = sim.run(num_steps=500)  # Fresh run with same config
        """
        if not self._initialized:
            logger.warning("Cannot reset: simulation not initialized")
            return
        
        logger.info("Resetting simulation...")
        self._engine.reset()
        self._results = None
        logger.info("Reset complete")
    
    # =========================================================================
    # State Access (Read-only)
    # =========================================================================
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current simulation state.
        
        Returns:
            Dict[str, Any]: Current state including opinions, positions, impact, etc.
        
        Raises:
            RuntimeError: If simulation not initialized.
        """
        self._check_initialized()
        return self._engine.get_state_snapshot()
    
    def get_final_opinions(self) -> np.ndarray:
        """
        Get the final opinion matrix after simulation completes.
        
        Returns:
            np.ndarray: Opinion matrix of shape (N, L).
        
        Raises:
            RuntimeError: If simulation hasn't been run yet.
        """
        if self._results is None:
            raise RuntimeError("No results available. Run the simulation first.")
        return self._results['final_opinions'].copy()
    
    def get_final_positions(self) -> np.ndarray:
        """
        Get the final agent positions.
        
        Returns:
            np.ndarray: Position matrix of shape (N, 2).
        """
        if self._results is None:
            raise RuntimeError("No results available. Run the simulation first.")
        return self._results['final_positions'].copy()
    
    def get_history(self) -> Dict[str, list]:
        """
        Get the time series history (if recording was enabled).
        
        Returns:
            Dict[str, list]: History data with keys 'time', 'opinions', 'impact', etc.
        
        Raises:
            RuntimeError: If history recording was not enabled in config.
        """
        if self._results is None:
            raise RuntimeError("No results available. Run the simulation first.")
        
        if 'history' not in self._results:
            raise RuntimeError("History recording was not enabled in configuration. "
                             "Set 'simulation.record_history: true' to enable.")
        
        return self._results['history']
    
    def get_network(self) -> nx.Graph:
        """
        Get the network graph structure.
        
        Returns:
            nx.Graph: The social network graph.
        """
        self._check_initialized()
        return self._engine.network_graph
    
    def get_current_time(self) -> float:
        """
        Get the current simulation time.
        
        Returns:
            float: Current time.
        """
        self._check_initialized()
        return self._engine.get_current_time()
    
    def get_time_step(self) -> int:
        """
        Get the current time step (iteration count).
        
        Returns:
            int: Current step number.
        """
        self._check_initialized()
        return self._engine.get_time_step()
    
    # =========================================================================
    # Configuration Access
    # =========================================================================
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the simulation configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary (read-only copy).
        """
        return copy.deepcopy(self.config)
    
    def get_num_agents(self) -> int:
        """
        Get the number of agents in the simulation.
        
        Returns:
            int: Number of agents.
        """
        return self.config.get('agents', {}).get('num_agents', 0)
    
    def get_opinion_dimensions(self) -> int:
        """
        Get the number of opinion dimensions/layers.
        
        Returns:
            int: Number of opinion topics.
        """
        return self.config.get('agents', {}).get('opinion_layers', 0)
    
    # =========================================================================
    # Save/Load
    # =========================================================================
    
    def save_results(self, filepath: Union[str, Path], 
                    format: str = 'npz'):
        """
        Save simulation results to disk.
        
        Args:
            filepath: Output file path.
            format: Output format - 'npz' (NumPy) or 'json'.
        
        Raises:
            RuntimeError: If no results to save.
        
        Example:
            >>> sim.run(num_steps=1000)
            >>> sim.save_results('results/experiment_1.npz')
        """
        if self._results is None:
            raise RuntimeError("No results to save. Run the simulation first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {filepath} (format={format})")
        
        if format == 'npz':
            # Save as compressed NumPy archive
            save_dict = {
                'final_opinions': self._results['final_opinions'],
                'final_positions': self._results['final_positions'],
                'final_impact': self._results['final_impact'],
                'final_time': self._results['final_time'],
                'total_steps': self._results['total_steps']
            }
            
            # Add history if available
            if 'history' in self._results:
                history = self._results['history']
                save_dict['history_time'] = np.array(history['time'])
                save_dict['history_num_events'] = np.array(history['num_events'])
                # Note: opinions and impact arrays may be large
            
            np.savez_compressed(filepath, **save_dict)
            
        elif format == 'json':
            # Save as JSON (excluding large arrays)
            save_dict = {
                'final_time': float(self._results['final_time']),
                'total_steps': int(self._results['total_steps']),
                'config': self.config
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_dict, f, indent=2)
        
        else:
            raise ValueError(f"Unknown format: {format}. Use 'npz' or 'json'.")
        
        logger.info("Results saved successfully")
    
    def save_state(self, filepath: Union[str, Path]):
        """
        Save the current simulation state (for checkpointing).
        
        Args:
            filepath: Output file path (.npz format).
        
        Example:
            >>> sim.run(num_steps=500)
            >>> sim.save_state('checkpoint_t500.npz')
            >>> # Later: sim.load_state('checkpoint_t500.npz')
        """
        self._check_initialized()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self._engine.save_state(str(filepath))
    
    def load_state(self, filepath: Union[str, Path]):
        """
        Load simulation state from a checkpoint.
        
        Args:
            filepath: Path to the state file (.npz format).
        
        Example:
            >>> sim = SimulationFacade.from_config_file('config.yaml')
            >>> sim.initialize()
            >>> sim.load_state('checkpoint_t500.npz')
            >>> sim.run(num_steps=500)  # Continue from checkpoint
        """
        self._check_initialized()
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
        
        self._engine.load_state(str(filepath))
    
    def save_event_log(self, filepath: Union[str, Path]):
        """
        Save the complete event history to JSON.
        
        Args:
            filepath: Output file path (.json format).
        
        Example:
            >>> sim.run(num_steps=1000)
            >>> sim.save_event_log('events/event_history.json')
        """
        self._check_initialized()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self._engine.event_manager.save_log(str(filepath))
        logger.info(f"Event log saved to {filepath}")
    
    # =========================================================================
    # Analysis Utilities
    # =========================================================================
    
    def calculate_polarization(self, opinions: Optional[np.ndarray] = None) -> float:
        """
        Calculate opinion polarization (standard deviation across agents).
        
        Args:
            opinions: Opinion matrix. If None, uses current state.
        
        Returns:
            float: Polarization measure (higher = more polarized).
        """
        if opinions is None:
            self._check_initialized()
            opinions = self._engine.opinion_matrix
        
        return float(np.std(opinions))
    
    def calculate_consensus(self, opinions: Optional[np.ndarray] = None,
                           threshold: float = 0.1) -> float:
        """
        Calculate consensus level (proportion of agents with similar opinions).
        
        Args:
            opinions: Opinion matrix. If None, uses current state.
            threshold: Maximum distance to be considered in consensus.
        
        Returns:
            float: Consensus ratio [0, 1] (higher = more consensus).
        """
        if opinions is None:
            self._check_initialized()
            opinions = self._engine.opinion_matrix
        
        # Calculate pairwise distances
        mean_opinion = np.mean(opinions, axis=0)
        distances = np.linalg.norm(opinions - mean_opinion, axis=1)
        
        # Count agents within threshold
        in_consensus = np.sum(distances < threshold)
        
        return float(in_consensus / len(opinions))
    
    def get_opinion_clusters(self, opinions: Optional[np.ndarray] = None,
                            epsilon: float = 0.2) -> int:
        """
        Estimate the number of opinion clusters.
        
        Args:
            opinions: Opinion matrix. If None, uses current state.
            epsilon: Distance threshold for clustering.
        
        Returns:
            int: Estimated number of clusters.
        """
        if opinions is None:
            self._check_initialized()
            opinions = self._engine.opinion_matrix
        
        from sklearn.cluster import DBSCAN
        
        clustering = DBSCAN(eps=epsilon, min_samples=5).fit(opinions)
        num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        return num_clusters
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    def _check_initialized(self):
        """Raise error if simulation not initialized."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")
    
    def __repr__(self):
        """String representation."""
        status = "initialized" if self._initialized else "not initialized"
        if self._initialized:
            return (f"<SimulationFacade {status} "
                   f"[t={self._engine.current_time:.1f}, "
                   f"agents={self._engine.num_agents}]>")
        else:
            return f"<SimulationFacade {status}>"
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Could add cleanup logic here if needed
        pass


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_run(config_file: Union[str, Path], 
              num_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to quickly run a simulation from a config file.
    
    Args:
        config_file: Path to YAML configuration file.
        num_steps: Number of steps to run (uses config value if None).
    
    Returns:
        Dict[str, Any]: Simulation results.
    
    Example:
        >>> results = quick_run('experiments/baseline.yaml', num_steps=1000)
    """
    sim = SimulationFacade.from_config_file(config_file)
    return sim.run(num_steps=num_steps)
