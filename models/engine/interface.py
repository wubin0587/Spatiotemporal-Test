# -*- coding: utf-8 -*-
"""
@File    : interface.py
@Desc    : Interface Layer for the Simulation Engine.
           Handles integration with external subsystems (Events, Networks, Spatial).
           Validates and transforms data before passing it to the core engine.
           
           This module acts as an adapter/facade between the engine and the modular
           components, ensuring data contracts are met and providing clean interfaces.
"""

import logging
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from typing import Dict, Any, List, Tuple, Optional

# Import external subsystems
from ..events.manager import EventManager
from ..events.base import Event
from ..networks.multilayer import build_multilayer_network
from ..spatial.distributions import create_spatial_distribution

logger = logging.getLogger(__name__)


class EngineInterface:
    """
    Interface layer that connects the simulation engine with external modules.
    
    Responsibilities:
    1. Initialize EventManager with validated configuration
    2. Build network topology via the networks module
    3. Generate spatial distributions via the spatial module
    4. Validate all data before passing to the engine
    5. Provide utility methods for data transformation
    
    This class does NOT implement simulation logic - it only handles I/O and validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the interface with simulation configuration.
        
        Args:
            config (Dict[str, Any]): The complete configuration dictionary.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Subsystem references (to be initialized)
        self.event_manager: Optional[EventManager] = None
        self.network_graph: Optional[nx.Graph] = None
        self.spatial_index: Optional[cKDTree] = None
        
    # =========================================================================
    # 1. Event Subsystem Interface
    # =========================================================================
    
    def initialize_event_manager(self) -> EventManager:
        """
        Creates and initializes the EventManager with validated configuration.
        
        Returns:
            EventManager: The initialized event management system.
            
        Raises:
            ValueError: If event configuration is invalid.
        """
        self.logger.info("Initializing Event Manager...")
        
        # Validate events section exists
        if 'events' not in self.config:
            raise ValueError("Configuration missing 'events' section.")
        
        event_config = self.config['events']
        
        # Validate that at least one generator is enabled
        gen_config = event_config.get('generation', {})
        enabled_generators = [
            gen_config.get('exogenous', {}).get('enabled', False),
            gen_config.get('endogenous_threshold', {}).get('enabled', False),
            gen_config.get('endogenous_cascade', {}).get('enabled', False)
        ]
        
        if not any(enabled_generators):
            self.logger.warning("No event generators enabled. Simulation will run without events.")
        
        # Create EventManager
        self.event_manager = EventManager(self.config)
        
        self.logger.info("Event Manager initialized successfully.")
        return self.event_manager
    
    def fetch_new_events(self, current_time: float, 
                        agents_state: Optional[Dict[str, Any]] = None) -> List[Event]:
        """
        Retrieves newly generated events for the current time step.
        
        Args:
            current_time (float): Current simulation time.
            agents_state (Optional[Dict]): Dictionary containing agent data.
                                          Required keys: 'positions', 'opinions'
                                          
        Returns:
            List[Event]: List of new events generated this step.
            
        Raises:
            RuntimeError: If EventManager is not initialized.
        """
        if self.event_manager is None:
            raise RuntimeError("EventManager not initialized. Call initialize_event_manager() first.")
        
        # Call EventManager's step method
        new_events = self.event_manager.step(
            current_time=current_time,
            agents_state=agents_state,
            env_state=None  # Can be extended in the future
        )
        
        return new_events
    
    def get_event_state_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                np.ndarray, np.ndarray]:
        """
        Retrieves vectorized event data for efficient computation.
        
        Returns:
            Tuple containing:
            - locations: (M, 2) array of event positions
            - times: (M,) array of event times
            - intensities: (M,) array of event intensities
            - contents: (M, L) array of event topic vectors
            - polarities: (M,) array of event polarity values
            
        Raises:
            RuntimeError: If EventManager is not initialized.
        """
        if self.event_manager is None:
            raise RuntimeError("EventManager not initialized.")
        
        return self.event_manager.get_state_vectors()
    
    def validate_event_data(self, events: List[Event]) -> bool:
        """
        Validates that event objects conform to expected schema.
        
        Args:
            events (List[Event]): List of event objects to validate.
            
        Returns:
            bool: True if all events are valid.
            
        Raises:
            ValueError: If any event is invalid.
        """
        for i, event in enumerate(events):
            # Check required attributes
            if not isinstance(event.loc, np.ndarray) or event.loc.shape != (2,):
                raise ValueError(f"Event {i} has invalid location: {event.loc}")
            
            if not isinstance(event.time, (int, float)):
                raise ValueError(f"Event {i} has invalid time: {event.time}")
            
            if not isinstance(event.intensity, (int, float)) or event.intensity < 0:
                raise ValueError(f"Event {i} has invalid intensity: {event.intensity}")
            
            if not isinstance(event.content, np.ndarray):
                raise ValueError(f"Event {i} has invalid content vector.")
            
            if not isinstance(event.polarity, (int, float)):
                raise ValueError(f"Event {i} has invalid polarity: {event.polarity}")
            
            # Validate location is in [0,1]x[0,1]
            if not (0 <= event.loc[0] <= 1 and 0 <= event.loc[1] <= 1):
                self.logger.warning(f"Event {i} location {event.loc} outside [0,1] bounds.")
        
        return True
    
    # =========================================================================
    # 2. Network Subsystem Interface
    # =========================================================================
    
    def build_network_topology(self) -> nx.Graph:
        """
        Constructs the network topology using the networks module.
        
        Returns:
            nx.Graph: The constructed network graph.
            
        Raises:
            ValueError: If network configuration is invalid.
        """
        self.logger.info("Building network topology...")
        
        # Validate network section exists
        if 'network' not in self.config:
            raise ValueError("Configuration missing 'network' section.")
        
        network_config = self.config['network']
        
        # Use the multilayer builder from networks module
        self.network_graph = build_multilayer_network(network_config)

        # The execution engine assumes integer node ids [0, N).
        # Multilayer builders may return tuple labels (e.g. (layer, node_id)),
        # so normalize labels here to keep the engine/network contract consistent.
        if self.network_graph.number_of_nodes() > 0:
            sample_node = next(iter(self.network_graph.nodes()))
            if not isinstance(sample_node, int):
                self.logger.info(
                    "Normalizing non-integer network node labels to integer indices."
                )
                self.network_graph = nx.convert_node_labels_to_integers(
                    self.network_graph,
                    label_attribute='supra_id'
                )
        
        # Validate the result
        if self.network_graph.number_of_nodes() == 0:
            raise ValueError("Network builder produced an empty graph.")
        
        self.logger.info(f"Network built: {self.network_graph.number_of_nodes()} nodes, "
                        f"{self.network_graph.number_of_edges()} edges.")
        
        return self.network_graph
    
    def extract_adjacency_list(self, graph: nx.Graph, num_agents: int) -> List[List[int]]:
        """
        Converts NetworkX graph to adjacency list format.
        
        Args:
            graph (nx.Graph): The network graph.
            num_agents (int): Expected number of nodes (for validation).
            
        Returns:
            List[List[int]]: Adjacency list where adj[i] = list of neighbors of node i.
            
        Raises:
            ValueError: If graph structure is inconsistent.
        """
        if graph.number_of_nodes() != num_agents:
            raise ValueError(f"Graph has {graph.number_of_nodes()} nodes but "
                           f"expected {num_agents} agents.")
        
        # Create adjacency list
        adj_list = [[] for _ in range(num_agents)]
        
        for node in graph.nodes():
            if not isinstance(node, int) or node < 0 or node >= num_agents:
                raise ValueError(f"Invalid node ID: {node}. Expected integer in [0, {num_agents}).")
            
            neighbors = list(graph.neighbors(node))
            adj_list[node] = neighbors
        
        return adj_list
    
    def validate_network_structure(self, graph: nx.Graph, num_agents: int) -> bool:
        """
        Validates network graph structure and properties.
        
        Args:
            graph (nx.Graph): The network to validate.
            num_agents (int): Expected number of agents.
            
        Returns:
            bool: True if valid.
            
        Raises:
            ValueError: If validation fails.
        """
        # Check node count
        if graph.number_of_nodes() != num_agents:
            raise ValueError(f"Network has {graph.number_of_nodes()} nodes, "
                           f"expected {num_agents}.")
        
        # Check node IDs are sequential integers
        expected_nodes = set(range(num_agents))
        actual_nodes = set(graph.nodes())
        if expected_nodes != actual_nodes:
            raise ValueError(f"Network nodes are not sequential integers [0, {num_agents}).")
        
        # Check for self-loops (usually not desired in social networks)
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            self.logger.warning(f"Network contains {len(self_loops)} self-loops.")
        
        # Check connectivity (warn if disconnected)
        if not nx.is_connected(graph):
            num_components = nx.number_connected_components(graph)
            self.logger.warning(f"Network is disconnected: {num_components} components.")
        
        return True
    
    # =========================================================================
    # 3. Spatial Subsystem Interface
    # =========================================================================
    
    def generate_agent_positions(self, num_agents: int) -> np.ndarray:
        """
        Generates spatial positions for agents using the spatial module.
        
        Args:
            num_agents (int): Number of agent positions to generate.
            
        Returns:
            np.ndarray: Array of shape (num_agents, 2) with coordinates in [0,1]x[0,1].
            
        Raises:
            ValueError: If spatial configuration is invalid.
        """
        self.logger.info(f"Generating spatial positions for {num_agents} agents...")
        
        # Get spatial configuration
        spatial_config = self.config.get('spatial', {})
        dist_config = spatial_config.get('distribution', {'type': 'uniform'})
        
        # Generate positions using spatial module
        positions = create_spatial_distribution(num_agents, dist_config)
        
        # Validate output
        if positions.shape != (num_agents, 2):
            raise ValueError(f"Expected positions shape ({num_agents}, 2), "
                           f"got {positions.shape}.")
        
        # Validate bounds
        if not (np.all(positions >= 0) and np.all(positions <= 1)):
            raise ValueError("Some positions are outside [0,1]x[0,1] bounds.")
        
        self.logger.info("Agent positions generated successfully.")
        return positions
    
    def build_spatial_index(self, positions: np.ndarray) -> cKDTree:
        """
        Builds a KD-Tree spatial index for efficient neighbor queries.
        
        Args:
            positions (np.ndarray): Agent positions array of shape (N, 2).
            
        Returns:
            cKDTree: The spatial index structure.
            
        Raises:
            ValueError: If positions array is invalid.
        """
        self.logger.info("Building spatial index (KD-Tree)...")
        
        # Validate input
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"Positions must be (N, 2) array, got {positions.shape}.")
        
        # Build KD-Tree
        self.spatial_index = cKDTree(positions)
        
        self.logger.info(f"Spatial index built for {positions.shape[0]} agents.")
        return self.spatial_index
    
    def validate_spatial_data(self, positions: np.ndarray, num_agents: int) -> bool:
        """
        Validates spatial position data.
        
        Args:
            positions (np.ndarray): Position array to validate.
            num_agents (int): Expected number of agents.
            
        Returns:
            bool: True if valid.
            
        Raises:
            ValueError: If validation fails.
        """
        if positions.shape != (num_agents, 2):
            raise ValueError(f"Position array shape {positions.shape} doesn't match "
                           f"expected ({num_agents}, 2).")
        
        # Check for NaN or Inf
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            raise ValueError("Position array contains NaN or Inf values.")
        
        # Check bounds
        if not (np.all(positions >= 0) and np.all(positions <= 1)):
            raise ValueError("Some positions are outside [0,1]x[0,1] bounds.")
        
        return True
    
    # =========================================================================
    # 4. Agent State Initialization
    # =========================================================================
    
    def initialize_opinion_matrix(self, num_agents: int, num_layers: int, 
                                   init_config: Dict[str, Any]) -> np.ndarray:
        """
        Initializes the opinion matrix for all agents.
        
        Args:
            num_agents (int): Number of agents.
            num_layers (int): Number of opinion dimensions/topics.
            init_config (Dict): Configuration for initialization.
                               Expected keys: 'type', 'params'
                               
        Returns:
            np.ndarray: Opinion matrix of shape (num_agents, num_layers).
        """
        self.logger.info(f"Initializing opinion matrix ({num_agents}, {num_layers})...")
        
        init_type = init_config.get('type', 'uniform').lower()
        params = init_config.get('params', {})
        
        if init_type == 'uniform':
            # Uniform random in [0, 1]
            opinions = np.random.uniform(0, 1, size=(num_agents, num_layers))
            
        elif init_type == 'normal':
            # Normal distribution clipped to [0, 1]
            mean = params.get('mean', 0.5)
            std = params.get('std', 0.2)
            opinions = np.random.normal(mean, std, size=(num_agents, num_layers))
            opinions = np.clip(opinions, 0, 1)
            
        elif init_type == 'polarized':
            # Create two clusters at opposite ends
            split = params.get('split', 0.5)  # Proportion in first cluster
            num_left = int(num_agents * split)
            
            left = np.random.normal(0.2, 0.1, size=(num_left, num_layers))
            right = np.random.normal(0.8, 0.1, size=(num_agents - num_left, num_layers))
            
            opinions = np.vstack([left, right])
            np.random.shuffle(opinions)
            opinions = np.clip(opinions, 0, 1)
            
        elif init_type == 'clustered':
            # Multiple opinion clusters
            num_clusters = params.get('num_clusters', 3)
            cluster_std = params.get('cluster_std', 0.1)
            
            # Assign agents to clusters
            cluster_ids = np.random.randint(0, num_clusters, size=num_agents)
            
            # Create cluster centers
            cluster_centers = np.random.uniform(0, 1, size=(num_clusters, num_layers))
            
            # Initialize opinions around cluster centers
            opinions = np.zeros((num_agents, num_layers))
            for i in range(num_agents):
                center = cluster_centers[cluster_ids[i]]
                noise = np.random.normal(0, cluster_std, size=num_layers)
                opinions[i] = center + noise
            
            opinions = np.clip(opinions, 0, 1)
            
        else:
            raise ValueError(f"Unknown opinion initialization type: '{init_type}'")
        
        self.logger.info("Opinion matrix initialized successfully.")
        return opinions.astype(np.float32)
    
    def validate_opinion_matrix(self, opinions: np.ndarray, 
                               num_agents: int, num_layers: int) -> bool:
        """
        Validates the opinion matrix structure and values.
        
        Args:
            opinions (np.ndarray): Opinion matrix to validate.
            num_agents (int): Expected number of agents.
            num_layers (int): Expected number of opinion dimensions.
            
        Returns:
            bool: True if valid.
            
        Raises:
            ValueError: If validation fails.
        """
        if opinions.shape != (num_agents, num_layers):
            raise ValueError(f"Opinion matrix shape {opinions.shape} doesn't match "
                           f"expected ({num_agents}, {num_layers}).")
        
        # Check for NaN or Inf
        if np.any(np.isnan(opinions)) or np.any(np.isinf(opinions)):
            raise ValueError("Opinion matrix contains NaN or Inf values.")
        
        # Check bounds [0, 1]
        if not (np.all(opinions >= 0) and np.all(opinions <= 1)):
            raise ValueError("Some opinions are outside [0, 1] bounds.")
        
        return True
    
    # =========================================================================
    # 5. Data Transformation Utilities
    # =========================================================================
    
    def prepare_agents_state_dict(self, positions: np.ndarray, 
                                  opinions: np.ndarray) -> Dict[str, Any]:
        """
        Packages agent data into a dictionary for passing to EventManager.
        
        Args:
            positions (np.ndarray): Agent positions (N, 2).
            opinions (np.ndarray): Agent opinions (N, L).
            
        Returns:
            Dict[str, Any]: Dictionary with 'positions' and 'opinions' keys.
        """
        return {
            'positions': positions,
            'opinions': opinions
        }
    
    def extract_layer_from_multilayer(self, graph: nx.Graph, 
                                      layer_name: str) -> nx.Graph:
        """
        Extracts a single layer from a multilayer network.
        
        Args:
            graph (nx.Graph): The multilayer network.
            layer_name (str): Name of the layer to extract.
            
        Returns:
            nx.Graph: Subgraph containing only nodes from the specified layer.
        """
        # Filter nodes by layer name (assumes nodes are tuples like (layer_name, node_id))
        layer_nodes = [n for n in graph.nodes() if isinstance(n, tuple) and n[0] == layer_name]
        
        if not layer_nodes:
            self.logger.warning(f"No nodes found for layer '{layer_name}'.")
            return nx.Graph()
        
        return graph.subgraph(layer_nodes).copy()
    
    # =========================================================================
    # 6. Summary and Diagnostics
    # =========================================================================
    
    def get_subsystem_status(self) -> Dict[str, bool]:
        """
        Returns the initialization status of all subsystems.
        
        Returns:
            Dict[str, bool]: Status dictionary with keys:
                - 'event_manager': EventManager initialized
                - 'network_graph': Network built
                - 'spatial_index': Spatial index built
        """
        return {
            'event_manager': self.event_manager is not None,
            'network_graph': self.network_graph is not None,
            'spatial_index': self.spatial_index is not None
        }
    
    def log_configuration_summary(self):
        """Logs a summary of the configuration for debugging."""
        self.logger.info("=== Configuration Summary ===")
        self.logger.info(f"Agents: {self.config.get('agents', {})}")
        self.logger.info(f"Network: {self.config.get('network', {}).get('type', 'N/A')}")
        self.logger.info(f"Events: {list(self.config.get('events', {}).get('generation', {}).keys())}")
        self.logger.info(f"Dynamics: {self.config.get('dynamics', {})}")
        self.logger.info("=============================")
