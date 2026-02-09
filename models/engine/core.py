# -*- coding: utf-8 -*-
"""
@File    : core.py
@Time    : 2026/02/09
@Author  : Research Team
@Desc    : The Core Engine (Controller Layer). 
           It maintains the global state (Agents, Network, Time) and orchestrates 
           the simulation loop by calling stateless math modules.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, List

# Import internal stateless simulation modules
from .simulation import field
from .simulation import topology
from .simulation import dynamics

# Import helper modules for initialization
from ..events.generator import EventManager
from ..networks import builder as net_builder
from ..spatial import builder as spatial_builder
from ..utils import tools

class SimulationCore:
    """
    The central controller for the Event-Modulated Opinion Dynamics Model.
    
    Attributes:
        config (Dict): The full configuration dictionary.
        num_agents (int): N.
        dim_opinions (int): L (dimension of opinion vector).
        
        X (np.ndarray): Opinion Matrix [N, L], range [0, 1].
        P (np.ndarray): Position Matrix [N, 2], normalized to [0, 1]x[0, 1].
        
        static_adj (List[List[int]]): Adjacency list for the "Algorithmic" (static) network.
        event_manager (EventManager): Handles generation and lifecycle of events.
        
        current_step (int): Global time step counter.
        history (Dict): Storage for tracking simulation metrics over time.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation core.

        Args:
            config: A dictionary containing nested configurations:
                - 'system': {N, L, T_max, ...}
                - 'network': {type, k, p, ...} for static topology.
                - 'spatial': {type, distribution, ...} for agent positions.
                - 'events': {prob, intensity, ...} for event generation.
                - 'dynamics': {epsilon, mu, ...} for opinion update.
        """
        self.config = config
        self.system_cfg = config['system']
        
        self.num_agents = self.system_cfg['n_agents']
        self.dim_opinions = self.system_cfg['n_topics']
        self.current_step = 0

        # --- 1. State Initialization ---
        print(f"[Core] Initializing {self.num_agents} agents with {self.dim_opinions} opinion dimensions...")
        self._init_population()
        self._init_network()
        self._init_events()

        # --- 2. Runtime Metrics ---
        # Stores macroscopic indicators (e.g., polarization index)
        self.history = {
            'time': [],
            'polarization': [],
            'avg_impact': []
        }

    def _init_population(self):
        """
        Initialize Agent Opinions (X) and Positions (P).
        Uses 'spatial' config for P and 'system' config for initial X distribution.
        """
        # 1. Initialize Opinions X ~ Uniform(0, 1) or Gaussian
        # Config example: {'distribution': 'uniform', 'low': 0.0, 'high': 1.0}
        self.X = tools.init_opinions(
            self.num_agents, 
            self.dim_opinions, 
            self.config.get('opinion_init', 'uniform')
        )

        # 2. Initialize Spatial Embedding P
        # Config example: {'type': 'random', 'cluster_count': 3}
        self.P = spatial_builder.generate_positions(
            self.num_agents, 
            self.config['spatial']
        )
        
        # Pre-build Spatial Index (KD-Tree) if agents are static to speed up search
        # If agents move, this needs to be rebuilt in step()
        self.kd_tree = topology.build_spatial_index(self.P)

    def _init_network(self):
        """
        Initialize the static 'Algorithmic' social network (e.g., Small-World).
        This represents the connections maintained by the recommendation system.
        """
        # Config example: {'type': 'watts_strogatz', 'k': 10, 'p': 0.1}
        self.static_adj = net_builder.build_adjacency_list(
            self.num_agents, 
            self.config['network']
        )

    def _init_events(self):
        """
        Initialize the Event Manager.
        """
        # The EventManager handles probability of new events and decay of old ones.
        self.event_manager = EventManager(self.config['events'])

    def step(self):
        """
        Execute one discrete time step of the simulation.
        
        Flow:
            1. Update Environment (Events).
            2. Compute Field (Impact I).
            3. Dynamic Topology Switching (Algorithm vs. Reality).
            4. Opinion Dynamics (Deffuant with modulation).
        """
        # --- Phase 1: Event Generation & Environment Update ---
        # The manager may spawn new events based on probability or scripts
        active_events = self.event_manager.update(self.current_step)
        
        # --- Phase 2: Compute Spatiotemporal Impact Field ---
        # Mathematical Model: I_i(t) = Sum( S_e * Decay_space * Decay_time )
        # Returns: impact_vector (N,) array of floats [0, 1]
        impact_vector = field.compute_impact_field(
            agent_positions=self.P,
            active_events=active_events,
            params=self.config['field_params'] # Contains alpha, beta
        )

        # --- Phase 3: Topology & Neighbor Discovery ---
        # Mathematical Model: 
        #   If I_i < threshold: Neighbor ~ Static Graph (Homophily)
        #   If I_i > threshold: Neighbor ~ Spatial Proximity (KD-Tree search)
        
        # This function returns a list of pairs (i, j) to interact
        interaction_pairs, active_mask = topology.mix_neighbors(
            static_adj=self.static_adj,
            kd_tree=self.kd_tree,
            agent_positions=self.P,
            impact_vector=impact_vector,
            params=self.config['topology_params'] # Contains R_base, k_expansion, threshold
        )

        # --- Phase 4: Dynamics & State Update ---
        # Mathematical Model: 
        #   Effective Epsilon = Base + delta * Impact
        #   Effective Mu = Base + gamma * Impact
        #   X(t+1) = X(t) + Mu_eff * (X_j - X_i)
        
        # Calculate the change in opinions (delta_X)
        delta_X = dynamics.calculate_opinion_change(
            X=self.X,
            interaction_pairs=interaction_pairs,
            impact_vector=impact_vector,
            params=self.config['dynamics_params'] # Contains epsilon, mu, backfire_flag
        )
        
        # Apply update
        self.X += delta_X
        
        # Clip opinions to ensure they remain in [0, 1]
        np.clip(self.X, 0.0, 1.0, out=self.X)

        # --- Phase 5: Recording ---
        self._record_metrics(impact_vector)
        self.current_step += 1

    def _record_metrics(self, impact_vector):
        """
        Calculate and store macroscopic metrics for the current step.
        """
        # Example: Average polarization (standard deviation of opinions)
        pol = np.std(self.X)
        avg_imp = np.mean(impact_vector)
        
        self.history['time'].append(self.current_step)
        self.history['polarization'].append(pol)
        self.history['avg_impact'].append(avg_imp)

    def get_snapshot(self):
        """
        Export the current state for the Interface / Visualization.
        
        Returns:
            dict: {
                'X': copy of opinions,
                'P': copy of positions,
                'events': list of active event dicts,
                'step': current step index
            }
        """
        return {
            'X': self.X.copy(),
            'P': self.P, # Assuming static P, no copy needed strictly unless mobile
            'events': self.event_manager.get_active_events_data(),
            'step': self.current_step
        }

    def get_history(self):
        """Return the recorded time-series data."""
        return self.history