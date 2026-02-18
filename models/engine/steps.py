# -*- coding: utf-8 -*-
"""
@File    : steps.py
@Desc    : Concrete implementation of the simulation step logic.
           Orchestrates the execution flow for each time step:
           1. Event generation
           2. Impact field calculation
           3. Neighbor selection (Algorithm vs. Reality mode)
           4. Opinion dynamics update
           
           This module contains the actual "physics" of the simulation.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Import the abstract base class
from .core import SimulationEngine
from .interface import EngineInterface

# Import math modules for the three-step pipeline
from .maths.field import compute_impact_field
from .maths.topo import get_interaction_pairs
from .maths.dynamics import calculate_opinion_change

logger = logging.getLogger(__name__)


class StepExecutor(SimulationEngine):
    """
    Concrete implementation of the SimulationEngine that executes the
    Event-Modulated Spatiotemporal Opinion Dynamics Model.
    
    This class implements the three-layer mathematical framework:
    - Layer 1: Field Calculation (Reality pressure on agents)
    - Layer 2: Topology Modulation (Who talks to whom)
    - Layer 3: Opinion Dynamics (How opinions change)
    
    The workflow follows the README.md description exactly.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the step executor with configuration.
        
        Args:
            config (Dict[str, Any]): Complete simulation configuration.
        """
        super().__init__(config)
        
        # Initialize the interface layer for external module integration
        self.interface = EngineInterface(config)
        
        # Extract configuration sections for easy access
        interface_cfg = config.get('engine', {}).get('interface', {})
        maths_cfg = config.get('engine', {}).get('maths', {})
        self.agent_config = interface_cfg.get('agents', {})
        self.sim_config = interface_cfg.get('simulation', {})
        self.dynamics_config = maths_cfg.get('dynamics', {})
        self.field_config = maths_cfg.get('field', {})
        self.topology_config = maths_cfg.get('topo', {})
        
        # History tracking (optional, for analysis)
        self.history = {
            'time': [],
            'opinions': [],
            'impact': [],
            'num_events': []
        }
        
        # Random seed for reproducibility
        seed = self.sim_config.get('seed', 42)
        np.random.seed(seed)
        
        logger.info(f"StepExecutor initialized with seed={seed}")
    
    def initialize(self):
        """
        Initialize all simulation components following the proper sequence.
        
        Workflow:
        1. Validate configuration
        2. Initialize agent count and dimensions
        3. Generate agent positions (spatial)
        4. Initialize opinion matrix
        5. Build network topology
        6. Create spatial index
        7. Initialize event manager
        8. Prepare cached structures
        """
        logger.info("=== Initializing Simulation ===")
        
        # Step 1: Validate configuration
        self._validate_config()
        
        # Step 2: Extract agent parameters
        self.num_agents = self.agent_config.get('num_agents', 1000)
        num_layers = self.agent_config.get('opinion_layers', 3)
        
        logger.info(f"Setting up {self.num_agents} agents with {num_layers} opinion dimensions")
        
        # Step 3: Generate spatial positions
        self.agent_positions = self.interface.generate_agent_positions(self.num_agents)
        self.interface.validate_spatial_data(self.agent_positions, self.num_agents)
        
        # Step 4: Initialize opinions
        opinion_init_config = self.agent_config.get('initial_opinions', {'type': 'uniform'})
        self.opinion_matrix = self.interface.initialize_opinion_matrix(
            self.num_agents, 
            num_layers, 
            opinion_init_config
        )
        self.interface.validate_opinion_matrix(self.opinion_matrix, self.num_agents, num_layers)
        
        # Step 5: Build network topology
        self.network_graph = self.interface.build_network_topology()
        self.interface.validate_network_structure(self.network_graph, self.num_agents)
        
        # Step 6: Create spatial index for efficient neighbor queries
        self.spatial_index = self.interface.build_spatial_index(self.agent_positions)
        
        # Step 7: Initialize event manager
        self.event_manager = self.interface.initialize_event_manager()
        
        # Step 8: Create adjacency list for fast neighbor lookups
        self.static_adjacency = self.interface.extract_adjacency_list(
            self.network_graph, 
            self.num_agents
        )
        
        # Step 9: Initialize impact vector (starts at zero - no events yet)
        self.impact_vector = np.zeros(self.num_agents, dtype=np.float32)
        
        # Step 10: Reset time
        self.current_time = 0.0
        self.time_step = 0
        
        logger.info("=== Initialization Complete ===")
        self._log_initialization_summary()
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one complete simulation time step.
        
        This implements the core pipeline described in README.md:
        
        A. State Evaluation & Mode Switching
           - Calculate impact field I(t) from active events
           - Determine which agents are in "algorithmic" vs "reality" mode
           
        B. Interaction & Parameter Modulation
           - Select interaction pairs based on mode
           - Modulate trust threshold and learning rate
           
        C. Opinion Update
           - Apply bounded confidence model with modulated parameters
           - Update opinion matrix
        
        Returns:
            Dict[str, Any]: Step statistics including:
                - 'time': Current time
                - 'step': Current step number
                - 'num_events': Number of active events
                - 'mean_impact': Average impact across agents
                - 'num_interactions': Number of agent interactions
        """
        # Validate that initialization was performed
        self._validate_state()
        
        logger.debug(f"--- Step {self.time_step} (t={self.current_time:.2f}) ---")
        
        # =====================================================================
        # Stage 1: Event Generation
        # =====================================================================
        agents_state = self.interface.prepare_agents_state_dict(
            self.agent_positions,
            self.opinion_matrix
        )
        
        new_events = self.interface.fetch_new_events(
            current_time=self.current_time,
            agents_state=agents_state
        )
        
        num_new_events = len(new_events)
        if num_new_events > 0:
            logger.debug(f"Generated {num_new_events} new events")
        
        # =====================================================================
        # Stage 2: Impact Field Calculation (Layer 1: Field Physics)
        # =====================================================================
        # Get all events from the archive (vectorized format)
        event_locs, event_times, event_intensities, event_contents, event_polarities = \
            self.interface.get_event_state_vectors()
        
        # EventManager might return None if no events exist yet
        has_events = (event_times is not None) and (len(event_times) > 0)

        if has_events:
            # Build list of active events for the field calculator
            active_events = self._build_active_events_list(
                event_locs, event_times, event_intensities,
                event_contents, event_polarities
            )
        else:
            active_events = []
        
        # Calculate the impact field I(t) for all agents
        field_params = self.field_config.copy()
        field_params['current_time'] = self.current_time
        
        self.impact_vector = compute_impact_field(
            agent_pos=self.agent_positions,
            active_events=active_events,
            params=field_params
        )
        
        mean_impact = np.mean(self.impact_vector)
        max_impact = np.max(self.impact_vector)
        
        if max_impact > 0:
            logger.debug(f"Impact field: mean={mean_impact:.3f}, max={max_impact:.3f}")
        
        # =====================================================================
        # Stage 3: Neighbor Selection (Layer 2: Topology Modulation)
        # =====================================================================
        # Determine interaction pairs based on impact levels
        # High impact -> spatial neighbors (reality mode)
        # Low impact -> social network neighbors (algorithm mode)
        
        interaction_pairs = get_interaction_pairs(
            static_adj=self.static_adjacency,
            kd_tree=self.spatial_index,
            agent_pos=self.agent_positions,
            impact_vector=self.impact_vector,
            params=self.topology_config
        )
        
        num_interactions = len(interaction_pairs)
        logger.debug(f"Generated {num_interactions} interaction pairs")
        
        # =====================================================================
        # Stage 4: Opinion Update (Layer 3: Bounded Confidence Dynamics)
        # =====================================================================
        # Calculate opinion changes based on interactions
        delta_opinions = calculate_opinion_change(
            X=self.opinion_matrix,
            pairs=interaction_pairs,
            impact_vector=self.impact_vector,
            params=self.dynamics_config
        )
        
        # Apply changes (synchronous update)
        self.opinion_matrix += delta_opinions
        
        # Ensure opinions stay in [0, 1] bounds
        self.opinion_matrix = np.clip(self.opinion_matrix, 0.0, 1.0)
        
        # =====================================================================
        # Stage 5: Time Advancement & History Recording
        # =====================================================================
        self.time_step += 1
        self.current_time += 1.0  # Assume dt=1.0, can be made configurable
        
        # Record history (optional, for analysis)
        if self.sim_config.get('record_history', False):
            self.history['time'].append(self.current_time)
            self.history['opinions'].append(self.opinion_matrix.copy())
            self.history['impact'].append(self.impact_vector.copy())
            self.history['num_events'].append(len(active_events))
        
        # =====================================================================
        # Return step statistics
        # =====================================================================
        return {
            'time': self.current_time,
            'step': self.time_step,
            'num_events': len(active_events),
            'num_new_events': num_new_events,
            'mean_impact': float(mean_impact),
            'max_impact': float(max_impact),
            'num_interactions': num_interactions,
            'opinion_std': float(np.std(self.opinion_matrix))
        }
    
    def run(self, num_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the simulation for multiple time steps.
        
        Args:
            num_steps (Optional[int]): Number of steps to run.
                                       If None, uses config value.
        
        Returns:
            Dict[str, Any]: Simulation results including:
                - 'final_time': End time
                - 'total_steps': Total steps executed
                - 'final_opinions': Final opinion matrix
                - 'history': Time series data (if recording enabled)
        """
        if num_steps is None:
            num_steps = self.total_steps
        
        logger.info(f"Running simulation for {num_steps} steps...")
        
        # Progress tracking
        report_interval = max(1, num_steps // 10)  # Report every 10%
        
        for step_num in range(num_steps):
            # Execute one step
            step_stats = self.step()
            
            # Progress reporting
            if (step_num + 1) % report_interval == 0 or step_num == 0:
                logger.info(f"Progress: {step_num + 1}/{num_steps} "
                          f"(t={self.current_time:.1f}, "
                          f"events={step_stats['num_events']}, "
                          f"impact={step_stats['mean_impact']:.3f})")
        
        logger.info("=== Simulation Complete ===")
        
        # Compile results
        results = {
            'final_time': self.current_time,
            'total_steps': self.time_step,
            'final_opinions': self.opinion_matrix.copy(),
            'final_positions': self.agent_positions.copy(),
            'final_impact': self.impact_vector.copy(),
            'config': self.config
        }
        
        # Include history if recorded
        if self.sim_config.get('record_history', False):
            results['history'] = self.history
        
        return results
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Capture a complete snapshot of the current simulation state.
        
        Returns:
            Dict[str, Any]: Complete state including all matrices and metadata.
        """
        return {
            'time': self.current_time,
            'step': self.time_step,
            'opinions': self.opinion_matrix.copy(),
            'positions': self.agent_positions.copy(),
            'impact': self.impact_vector.copy(),
            'num_agents': self.num_agents,
            'network_nodes': self.network_graph.number_of_nodes(),
            'network_edges': self.network_graph.number_of_edges()
        }
    
    def reset(self):
        """
        Reset the simulation to its initial state.
        
        This clears all state and re-runs initialization.
        Useful for running multiple experiments with the same configuration.
        """
        logger.info("Resetting simulation...")
        
        # Clear history
        self.history = {
            'time': [],
            'opinions': [],
            'impact': [],
            'num_events': []
        }
        
        # Reset event manager
        if self.event_manager is not None:
            self.event_manager.reset()
        
        # Re-run initialization
        self.initialize()
        
        logger.info("Reset complete.")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _build_active_events_list(self, locs: np.ndarray, times: np.ndarray,
                                   intensities: np.ndarray, contents: np.ndarray,
                                   polarities: np.ndarray) -> List[Dict[str, Any]]:
        """
        Converts vectorized event data into a list of dictionaries
        for the field calculator.
        
        Args:
            locs: Event locations (M, 2)
            times: Event times (M,)
            intensities: Event intensities (M,)
            contents: Event content vectors (M, L)
            polarities: Event polarities (M,)
        
        Returns:
            List[Dict]: List of event dictionaries with keys:
                        'pos', 'time', 'intensity', 'content', 'polarity'
        """
        # Filter events based on temporal window
        temporal_window = self.field_config.get('temporal_window', 100.0)
        active_mask = (times <= self.current_time) & \
                     (times >= self.current_time - temporal_window)
        
        active_indices = np.where(active_mask)[0]
        
        # Build list of event dictionaries
        events = []
        for idx in active_indices:
            events.append({
                'pos': locs[idx],
                'time': times[idx],
                'intensity': intensities[idx],
                'content': contents[idx] if len(contents) > 0 else None,
                'polarity': polarities[idx] if len(polarities) > 0 else 0.0
            })
        
        return events
    
    def _log_initialization_summary(self):
        """Log a summary of the initialized simulation."""
        logger.info("--- Initialization Summary ---")
        logger.info(f"Agents: {self.num_agents}")
        logger.info(f"Opinion dimensions: {self.opinion_matrix.shape[1]}")
        logger.info(f"Network edges: {self.network_graph.number_of_edges()}")
        logger.info(f"Event generators: {self.interface.get_subsystem_status()}")
        logger.info(f"Total steps planned: {self.total_steps}")
        logger.info("------------------------------")
    
    def save_state(self, filepath: str):
        """
        Save the current simulation state to disk.
        
        Args:
            filepath (str): Path to save the state file (NPZ format).
        """
        logger.info(f"Saving simulation state to {filepath}")
        
        np.savez_compressed(
            filepath,
            time=self.current_time,
            step=self.time_step,
            opinions=self.opinion_matrix,
            positions=self.agent_positions,
            impact=self.impact_vector,
            config=self.config
        )
        
        logger.info("State saved successfully.")
    
    def load_state(self, filepath: str):
        """
        Load simulation state from disk.
        
        Args:
            filepath (str): Path to the state file (NPZ format).
        """
        logger.info(f"Loading simulation state from {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        self.current_time = float(data['time'])
        self.time_step = int(data['step'])
        self.opinion_matrix = data['opinions']
        self.agent_positions = data['positions']
        self.impact_vector = data['impact']
        
        logger.info("State loaded successfully.")