import networkx as nx
import logging

# Import the single-layer builder and the interlayer coupling logic
from .builder import build_graph_topology
from .coupling import generate_interlayer_edges

logger = logging.getLogger(__name__)

class MultilayerNetwork:
    """
    Constructs a multiplex network where nodes are shared across layers.
    
    Instead of creating separate nodes for each layer (e.g., ('social', 0)),
    this builder merges all layers onto a single set of integer-ID nodes.
    
    The distinction between layers is preserved in the Edge Attributes.
    For example, an edge might have data: {'layer': 'social', 'weight': 0.5}.
    
    This satisfies the engine's requirement that nodes must be sequential integers [0, N-1].
    """

    def __init__(self, config):
        """
        Args:
            config (dict): The 'network' section of the YAML configuration file.
        """
        self.config = config
        # We use a standard Graph. If multiple layers have an edge between the same
        # two nodes, we will merge the attributes.
        self.supra_graph = nx.Graph()

    def build(self):
        """
        Executes the full multilayer network construction pipeline.

        Returns:
            nx.Graph: The fully constructed graph with integer node IDs.
        """
        logger.info("Starting multilayer network construction (Multiplex Mode)...")

        # Step 1: Build individual layers and merge them into the main graph
        self._build_and_compose_layers()

        # Step 2: Add specific interlayer connections (if defined in config)
        self._add_interlayer_edges()

        logger.info(
            f"Multilayer construction complete. "
            f"Total Nodes: {self.supra_graph.number_of_nodes()}, "
            f"Total Edges: {self.supra_graph.number_of_edges()}"
        )
        return self.supra_graph

    def _build_and_compose_layers(self):
        """
        Iterates through the 'layers' config.
        Builds each layer independently, then merges the edges into the supra_graph.
        Ensures Node IDs remain Integers.
        """
        layers_config = self.config.get('layers', [])
        if not layers_config:
            raise ValueError("Configuration must contain a 'layers' list.")

        for layer_cfg in layers_config:
            layer_name = layer_cfg.get('name')
            if not layer_name:
                raise ValueError(f"Layer configuration {layer_cfg} is missing a 'name'.")
            
            logger.info(f"Constructing layer: '{layer_name}'")

            # 1. Force the builder to return integer node IDs
            layer_cfg['convert_to_int'] = True
            G_raw = build_graph_topology(layer_cfg)

            # 2. Merge nodes and edges into the main graph
            # We explicitly handle edge attributes to track which layer the edge belongs to.
            for u, v, data in G_raw.edges(data=True):
                
                # Prepare the new edge attributes
                # We copy the raw data (like weight) and add the layer name
                edge_attrs = data.copy()
                edge_attrs['layer'] = layer_name
                
                # If this edge already exists (e.g., from a previous layer), 
                # we can choose to update it or merge metadata.
                # Here, we keep a list of layers this edge belongs to.
                if self.supra_graph.has_edge(u, v):
                    # Edge exists: Update the 'layers' list attribute
                    existing_data = self.supra_graph[u][v]
                    
                    # Initialize 'layers' list if it doesn't exist
                    if 'layers' not in existing_data:
                        # Convert the old single 'layer' string to a list
                        existing_data['layers'] = [existing_data.get('layer', 'unknown')]
                    
                    # Add current layer if not present
                    if layer_name not in existing_data['layers']:
                        existing_data['layers'].append(layer_name)
                        
                    # Optional: Logic to combine weights (e.g., average or sum)
                    # existing_data['weight'] = (existing_data['weight'] + edge_attrs.get('weight', 1)) / 2
                    
                else:
                    # New Edge: Create it
                    edge_attrs['layers'] = [layer_name]
                    self.supra_graph.add_edge(u, v, **edge_attrs)

            # 3. Ensure isolated nodes are also added
            # This guarantees that if a node has no edges in this layer, it still exists in the graph.
            self.supra_graph.add_nodes_from(G_raw.nodes(data=True))

    def _add_interlayer_edges(self):
        """
        Adds edges explicitly defined as 'interlayer' in the config.
        
        In a Multiplex network (shared IDs), 'interlayer' edges usually mean
        connections between different agents based on cross-layer logic.
        """
        couplings = self.config.get('interlayer', [])
        if not couplings:
            return

        all_layer_names = {layer['name'] for layer in self.config.get('layers', [])}

        for rule in couplings:
            try:
                # We assume generate_interlayer_edges returns valid integer pairs (u, v)
                # If the generator creates edges, we simply add them to the graph.
                new_edges = generate_interlayer_edges(
                    self.supra_graph,
                    rule,
                    all_layer_names
                )

                if new_edges:
                    # Mark these edges as 'interlayer' type
                    for u, v, data in new_edges:
                        if not data: 
                            data = {}
                        data['type'] = 'interlayer'
                        self.supra_graph.add_edge(u, v, **data)

            except Exception as e:
                logger.error(f"Could not apply coupling rule {rule}. Reason: {e}")
                raise

def build_multilayer_network(network_config):
    """
    Facade function to simplify calling the multilayer builder.
    """
    builder = MultilayerNetwork(network_config)
    return builder.build()