# D:\Tiktok\models\spatial\layer.py

import logging
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from typing import Dict, Any, Optional

# Import internal modules
from .distribution import create_spatial_distribution
from .ops import (
    build_spatial_index, 
    query_k_nearest_neighbors, 
    query_radius_neighbors, 
    calculate_gravity_weights,
    compute_pairwise_distances
)

logger = logging.getLogger(__name__)

class SpatialLayer:
    """
    Manages the creation and state of a Spatially Embedded Network Layer.
    
    This class acts as a facade that integrates:
    1. Node distribution generation (via distribution.py)
    2. Spatial indexing and query operations (via ops.py)
    3. Graph topology construction (NetworkX)

    It produces a NetworkX graph where nodes have a 'pos' attribute 
    containing their [x, y] coordinates in the normalized [0,1] space.
    """

    def __init__(self, layer_config: Dict[str, Any]):
        """
        Initialize the Spatial Layer manager.

        Args:
            layer_config (dict): The configuration dictionary for this specific layer.
                                 It typically comes from the 'layers' list in the YAML.
        """
        self.config = layer_config
        self.name = layer_config.get('name', 'spatial_layer')
        self.num_nodes = layer_config.get('num_nodes', 100)
        
        # Spatial-specific configurations
        # Expects a structure like: {'distribution': { ... }, 'graph_construction': { ... }}
        self.spatial_params = layer_config.get('spatial', {})
        
        # State placeholders
        self.locations: Optional[np.ndarray] = None # Shape (N, 2)
        self.kd_tree = None
        self.graph: Optional[nx.Graph] = None

    def build(self) -> nx.Graph:
        """
        Executes the full pipeline to generate the spatial graph.

        Pipeline:
        1. Generate node coordinates (distribution).
        2. Build spatial index (KDTree).
        3. Construct edges based on spatial rules (Radius, k-NN, etc.).
        4. Assign attributes (coordinates, weights).

        Returns:
            nx.Graph: The constructed graph with integer node IDs.
        """
        logger.info(f"Building Spatial Layer: '{self.name}' with {self.num_nodes} nodes.")

        # 1. Generate Coordinates
        dist_config = self.spatial_params.get('distribution', {'type': 'uniform'})
        self.locations = create_spatial_distribution(self.num_nodes, dist_config)

        # 2. Build Spatial Index (Optimization for building edges)
        self.kd_tree = build_spatial_index(self.locations)

        # 3. Initialize Graph and add nodes with positions
        self.graph = nx.Graph(name=self.name)
        # Add nodes 0 to N-1. Store position as a node attribute 'pos'.
        # This is crucial for visualization and distance-based calculations later.
        for i in range(self.num_nodes):
            self.graph.add_node(i, pos=self.locations[i])

        # 4. Construct Edges based on rules
        construct_config = self.spatial_params.get('graph_construction', {'method': 'radius', 'radius': 0.1})
        self._construct_topology(construct_config)

        logger.info(f"Spatial Layer '{self.name}' built. "
                    f"Nodes: {self.graph.number_of_nodes()}, "
                    f"Edges: {self.graph.number_of_edges()}")
        
        return self.graph

    def _construct_topology(self, config: Dict[str, Any]):
        """
        Internal method to determine edge creation strategy.
        
        Supported methods:
        - 'radius': Connect nodes within a certain distance.
        - 'knn': Connect each node to its k-nearest neighbors.
        - 'delaunay': Connect nodes via Delaunay triangulation (planar graph).
        
        YAML Config Examples:
        
        graph_construction:
          method: radius
          radius: 0.05
          
        graph_construction:
          method: knn
          k: 5
          
        graph_construction:
          method: delaunay
        """
        method = config.get('method', 'radius').lower()
        weight_decay = config.get('weight_decay', 0.0) # If > 0, apply gravity model weights
        
        if method == 'radius':
            radius = config.get('radius', 0.1)
            self._build_radius_graph(radius)
            
        elif method == 'knn':
            k = config.get('k', 5)
            self._build_knn_graph(k)
            
        elif method == 'delaunay':
            self._build_delaunay_graph()
            
        else:
            raise ValueError(f"Unknown spatial graph construction method: {method}")

        # Optional: Apply weights based on distance if requested
        if weight_decay > 0 or config.get('weighted', False):
            self._apply_edge_weights(beta=weight_decay)

    def _build_radius_graph(self, radius: float):
        """Connects nodes if distance < radius."""
        # query_radius_neighbors returns list of arrays of indices
        indices, _ = query_radius_neighbors(self.kd_tree, self.locations, radius, sort_results=False)
        
        edges = []
        for i, neighbors in enumerate(indices):
            for n_idx in neighbors:
                if i < n_idx: # Avoid duplicates (u, v) and (v, u) and self-loops
                    edges.append((i, n_idx))
        
        self.graph.add_edges_from(edges)

    def _build_knn_graph(self, k: int):
        """Connects each node to its k nearest neighbors."""
        # query_k_nearest_neighbors returns (distances, indices)
        # include_self=False ensures we don't connect node to itself
        _, indices = query_k_nearest_neighbors(self.kd_tree, self.locations, k, include_self=False)
        
        edges = []
        for i, neighbors in enumerate(indices):
            for n_idx in neighbors:
                # k-NN is directed by definition, but we are building an undirected graph here.
                # Adding (u, v) creates an undirected edge. 
                # NetworkX handles duplicates silently.
                edges.append((i, n_idx))
        
        self.graph.add_edges_from(edges)

    def _build_delaunay_graph(self):
        """Creates a planar graph using Delaunay Triangulation."""
        tri = Delaunay(self.locations)
        edges = set()
        for simplex in tri.simplices:
            # Simplex is [A, B, C] indices of the triangle
            path = list(simplex)
            for i in range(len(path)):
                u = path[i]
                v = path[(i + 1) % len(path)]
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        self.graph.add_edges_from(list(edges))

    def _apply_edge_weights(self, beta: float = 2.0):
        """
        Calculates and updates edge weights based on Euclidean distance.
        Uses the Gravity Model logic: Weight ~ 1 / distance^beta
        """
        for u, v in self.graph.edges():
            pos_u = self.locations[u]
            pos_v = self.locations[v]
            dist = np.linalg.norm(pos_u - pos_v)
            
            # Use ops function for consistency, though simple here
            # We treat alpha=1.0 for basic weight
            weight = calculate_gravity_weights(dist, alpha=1.0, beta=beta)
            
            # Update edge attribute
            self.graph[u][v]['weight'] = weight
            self.graph[u][v]['distance'] = dist

# Helper function to be called by builder.py or factory
def create_spatial_layer_graph(config: Dict[str, Any]) -> nx.Graph:
    """
    Factory function to instantiate the SpatialLayer and return its graph.
    Matches the signature expected by a generic builder dispatcher.
    """
    layer = SpatialLayer(config)
    return layer.build()