import networkx as nx
import random
import logging
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# =========================================================================
# 1. Core Coupling Strategies (Private Helper Functions)
# =========================================================================

def _couple_one_to_one(G, layer_a, layer_b, params):
    """Connects nodes with the same integer ID across two layers."""
    edges = []
    weight = params.get('weight', 1.0)
    
    # Node labels are tuples like ('layer_name', node_id)
    # We extract the integer node_id (n[1]) for matching
    ids_a = {n[1] for n in G.nodes if n[0] == layer_a}
    ids_b = {n[1] for n in G.nodes if n[0] == layer_b}
    
    common_ids = ids_a.intersection(ids_b)
    
    for nid in common_ids:
        u, v = (layer_a, nid), (layer_b, nid)
        edges.append((u, v, {'weight': weight}))
        
    return edges

def _couple_random(G, layer_a, layer_b, params):
    """Randomly connect nodes between layers with probability p."""
    edges = []
    p = params.get('p')
    if p is None:
        raise ValueError("Random coupling requires the 'p' (probability) parameter.")
    weight = params.get('weight', 1.0)

    nodes_a = [n for n in G.nodes if n[0] == layer_a]
    nodes_b = [n for n in G.nodes if n[0] == layer_b]

    # Note: This is a naive O(N*M) implementation. For very large layers,
    # a more optimized sampling approach would be needed.
    for u in nodes_a:
        for v in nodes_b:
            if random.random() < p:
                edges.append((u, v, {'weight': weight}))
    return edges

def _couple_spatial_proximity(G, layer_a, layer_b, params):
    """Connects nodes if their spatial distance is below a threshold."""
    edges = []
    threshold = params.get('threshold')
    if threshold is None:
        raise ValueError("Spatial coupling requires the 'threshold' parameter.")
    weight = params.get('weight', 1.0)

    # This relies on builder.py saving coordinates in 'original_id'
    nodes_a = [(n, G.nodes[n].get('original_id')) for n in G.nodes if n[0] == layer_a]
    nodes_b = [(n, G.nodes[n].get('original_id')) for n in G.nodes if n[0] == layer_b]
    
    # Filter out nodes that don't have spatial coordinates
    nodes_a = [item for item in nodes_a if isinstance(item[1], (tuple, list))]
    nodes_b = [item for item in nodes_b if isinstance(item[1], (tuple, list))]

    if not nodes_a or not nodes_b:
        logger.warning(f"No nodes with spatial coordinates found for coupling {layer_a}-{layer_b}.")
        return []

    # Unpack for Scipy
    node_labels_a, coords_a = zip(*nodes_a)
    node_labels_b, coords_b = zip(*nodes_b)

    # Calculate pairwise distances efficiently using Scipy
    distance_matrix = cdist(np.array(coords_a), np.array(coords_b))
    
    # Find indices where distance is below the threshold
    indices_a, indices_b = np.where(distance_matrix < threshold)
    
    for idx_a, idx_b in zip(indices_a, indices_b):
        u, v = node_labels_a[idx_a], node_labels_b[idx_b]
        edges.append((u, v, {'weight': weight}))
        
    return edges

# =========================================================================
# 2. Strategy Registry (Dispatcher)
# =========================================================================

COUPLING_REGISTRY = {
    'one_to_one': _couple_one_to_one,
    'multiplex': _couple_one_to_one, # Alias
    'random': _couple_random,
    'spatial': _couple_spatial_proximity
}

# =========================================================================
# 3. Validation and Public Interface
# =========================================================================

def _validate_coupling_rule(rule, existing_layers):
    """Performs pre-flight checks on a single coupling rule from YAML."""
    # Basic field check
    required_keys = ['from', 'to', 'method']
    if not all(key in rule for key in required_keys):
        raise ValueError(f"Coupling rule {rule} is missing one of required keys: {required_keys}")

    # Layer existence check
    src, dst = rule['from'], rule['to']
    if src not in existing_layers:
        raise ValueError(f"Source layer '{src}' in coupling rule not found in defined layers.")
    if dst not in existing_layers:
        raise ValueError(f"Destination layer '{dst}' in coupling rule not found in defined layers.")

    # Method existence check
    if rule['method'] not in COUPLING_REGISTRY:
        raise NotImplementedError(f"Coupling method '{rule['method']}' is not implemented.")
    
    # Node count warning for multiplex
    if rule['method'] in ['one_to_one', 'multiplex']:
        # This check is performed on the supra_graph G later, not here.
        # But we can note it.
        pass

def generate_interlayer_edges(G, coupling_rule, all_layer_names):
    """
    Generates interlayer edges based on a validated coupling rule.
    This is the main public function called by multilayer.py.

    Args:
        G (nx.Graph): The supra-graph containing nodes from all layers so far.
        coupling_rule (dict): A single dictionary from the 'interlayer' list in YAML.
        all_layer_names (set): A set of all layer names for validation.

    Returns:
        list: A list of edge tuples to be added to the graph.
              e.g., [ (u, v, {'weight': 1.0}), ... ]
    """
    # 1. Validate the rule first
    _validate_coupling_rule(coupling_rule, all_layer_names)
    
    method = coupling_rule['method']
    src_layer = coupling_rule['from']
    dst_layer = coupling_rule['to']
    params = coupling_rule.get('params', {})
    
    logger.info(f"Generating edges: {src_layer} <-> {dst_layer} using '{method}'")
    
    # 2. Perform node count check (if applicable)
    if method in ['one_to_one', 'multiplex']:
        count_a = sum(1 for n in G.nodes if n[0] == src_layer)
        count_b = sum(1 for n in G.nodes if n[0] == dst_layer)
        if count_a != count_b:
            logger.warning(
                f"Node count mismatch for 'one_to_one' coupling: "
                f"Layer '{src_layer}' has {count_a} nodes, while "
                f"Layer '{dst_layer}' has {count_b} nodes. "
                f"Only nodes with common IDs will be connected."
            )
            
    # 3. Dispatch to the correct strategy function
    coupling_func = COUPLING_REGISTRY[method]
    edges = coupling_func(G, src_layer, dst_layer, params)
    
    logger.info(f"Generated {len(edges)} edges between '{src_layer}' and '{dst_layer}'.")
    return edges