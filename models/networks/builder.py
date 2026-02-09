import networkx as nx
import logging

# Configure logger
logger = logging.getLogger(__name__)

def build_graph_topology(net_config):
    """
    Builds a single-layer network topology based on a configuration dictionary.
    
    This factory function supports a wide range of network models from NetworkX,
    categorized into deterministic, random, community-structured, and spatial types.
    It standardizes node labels to integers for compatibility with downstream tasks,
    while preserving original labels (like spatial coordinates) as a node attribute.

    Args:
        net_config (dict): Configuration for the network.

    Returns:
        nx.Graph: The constructed NetworkX graph object.
    """
    
    net_type = net_config.get('type', 'undefined').lower()
    params = net_config.get('params', {})
    convert_to_int = net_config.get('convert_to_int', True)

    logger.info(f"Building network topology: '{net_type}' with params: {params}")
    graph = None

    try:
        # =========================================================================
        # 1. Deterministic / Regular Networks
        # =========================================================================
        if net_type == 'path':
            # YAML: { type: "path", params: { n: 100 } }
            graph = nx.path_graph(**params)
            
        elif net_type == 'cycle':
            # YAML: { type: "cycle", params: { n: 100 } }
            graph = nx.cycle_graph(**params)

        elif net_type == 'complete':
            # YAML: { type: "complete", params: { n: 50 } }
            graph = nx.complete_graph(**params)
            
        elif net_type == 'star':
            # YAML: { type: "star", params: { n: 50 } } # n = total nodes, so n-1 leaves
            graph = nx.star_graph(**params)

        elif net_type == 'tree':
            # YAML: { type: "tree", params: { r: 2, h: 5 } } # r=branching, h=height
            graph = nx.balanced_tree(**params)

        # =========================================================================
        # 2. Random Network Models
        # =========================================================================
        elif net_type in ['random', 'erdos_renyi', 'er']:
            # YAML: { type: "random", params: { n: 1000, p: 0.01 } }
            graph = nx.erdos_renyi_graph(**params)

        elif net_type in ['small_world', 'watts_strogatz', 'ws']:
            # YAML: { type: "small_world", params: { n: 1000, k: 10, p: 0.1 } }
            graph = nx.watts_strogatz_graph(**params)
            
        elif net_type in ['scale_free', 'barabasi_albert', 'ba']:
            # YAML: { type: "scale_free", params: { n: 1000, m: 5 } }
            graph = nx.barabasi_albert_graph(**params)

        # =========================================================================
        # 3. Community-Structured Networks
        # =========================================================================
        elif net_type in ['stochastic_block_model', 'sbm']:
            # The standard for generating echo chambers / communities.
            # YAML:
            #   type: "sbm"
            #   params:
            #     sizes: [100, 100, 50]         # 3 communities of sizes 100, 100, 50
            #     probs: [[0.25, 0.05, 0.02],   # Connection prob matrix (p_in > p_out)
            #             [0.05, 0.35, 0.07],
            #             [0.02, 0.07, 0.40]]
            graph = nx.stochastic_block_model(**params)
            
        # =========================================================================
        # 4. Spatial & Geometric Networks
        # =========================================================================
        elif net_type == 'grid':
            # YAML: { type: "grid", params: { dim: [20, 20] } }
            graph = nx.grid_graph(**params)

        elif net_type == 'hexagonal_lattice':
            # YAML: { type: "hexagonal_lattice", params: { m: 10, n: 10 } }
            graph = nx.hexagonal_lattice_graph(**params)
            
        elif net_type in ['random_geometric', 'rgg']:
            # Nodes randomly placed in a unit square, connected if within radius.
            # YAML: { type: "random_geometric", params: { n: 500, radius: 0.1 } }
            graph = nx.random_geometric_graph(**params)

        elif net_type == 'waxman':
            # Classic model for communication networks.
            # YAML: { type: "waxman", params: { n: 500, beta: 0.4, alpha: 0.1 } }
            graph = nx.waxman_graph(**params)

        # =========================================================================
        # 5. Bipartite Networks
        # =========================================================================
        elif net_type == 'bipartite':
            # Models user-content, user-group, etc.
            # YAML: { type: "bipartite", params: { n: 1000, m: 50, p: 0.02 } } # n, m are partition sizes
            graph = nx.bipartite.random_graph(**params)

        # =========================================================================
        # 6. Classic Toy Networks (for testing/debugging)
        # =========================================================================
        elif net_type == 'karate_club':
            # YAML: { type: "karate_club", params: {} }
            graph = nx.karate_club_graph()

        # =========================================================================
        # 7. Load from File
        # =========================================================================
        elif net_type == 'custom_file':
            path = params.get('path')
            if not path:
                raise ValueError("Parameter 'path' is required for 'custom_file' type.")
                
            if path.endswith('.gexf'):
                graph = nx.read_gexf(path)
            elif path.endswith('.edgelist'):
                graph = nx.read_edgelist(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
        
        else:
            raise NotImplementedError(f"Network type '{net_type}' is not implemented in builder.")

        # Final check if a graph was created
        if graph is None:
             raise ValueError(f"Graph could not be constructed for type '{net_type}'.")

        # =========================================================================
        # Post-processing: Standardize Node IDs
        # =========================================================================
        if convert_to_int:
            # Converts node labels (e.g., grid tuples, bipartite tuples, strings)
            # into a standard integer sequence (0, 1, 2...).
            # The original label is saved in the 'original_id' attribute.
            graph = nx.convert_node_labels_to_integers(graph, label_attribute='original_id')
            logger.debug("Node labels converted to integers. Original IDs saved in 'original_id' attribute.")

        logger.info(f"Network built successfully. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        return graph

    except TypeError as e:
        logger.error(f"Parameter mismatch for network type '{net_type}'. "
                     f"Please check your YAML config against the documentation. Error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to build network '{net_type}': {e}")
        raise e