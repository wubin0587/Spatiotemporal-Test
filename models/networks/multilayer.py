import networkx as nx
import logging

# Import the single-layer builder and the interlayer coupling logic
from .builder import build_graph_topology
from .coupling import generate_interlayer_edges

logger = logging.getLogger(__name__)

class MultilayerNetwork:
    """
    Constructs a multilayer network by assembling individual layers and
    connecting them according to specified coupling rules.

    This class orchestrates the build process:
    1. Sequentially builds each layer using `builder.py`.
    2. Relabels nodes to be unique across the entire network, e.g., (layer_name, node_id).
    3. Adds interlayer edges using the strategies defined in `coupling.py`.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): The 'network' section of the YAML configuration file.
        """
        self.config = config
        self.supra_graph = nx.Graph()

    def build(self):
        """
        Executes the full multilayer network construction pipeline.

        Returns:
            nx.Graph: The fully constructed supra-graph.
        """
        logger.info("Starting multilayer network construction...")

        # Step 1: Build and compose individual layers (intra-layer edges)
        self._build_and_compose_layers()

        # Step 2: Add connections between layers (inter-layer edges)
        self._add_interlayer_edges()

        logger.info(
            f"Multilayer construction complete. "
            f"Total Nodes: {self.supra_graph.number_of_nodes()}, "
            f"Total Edges: {self.supra_graph.number_of_edges()}"
        )
        return self.supra_graph

    def _build_and_compose_layers(self):
        """
        Iterates through the 'layers' config, builds each one, relabels its nodes,
        and adds it to the main supra-graph.
        """
        layers_config = self.config.get('layers', [])
        if not layers_config:
            raise ValueError("Configuration must contain a 'layers' list to build a multilayer network.")

        for layer_cfg in layers_config:
            layer_name = layer_cfg.get('name')
            if not layer_name:
                raise ValueError(f"Layer configuration {layer_cfg} is missing a 'name'.")
            
            logger.info(f"Constructing layer: '{layer_name}'")

            # 1. Build the single-layer graph using the builder
            # We pass the layer's config directly. builder.py will handle it.
            # We ensure integer conversion for consistent ID matching.
            layer_cfg['convert_to_int'] = True
            G_raw = build_graph_topology(layer_cfg)

            # 2. Relabel nodes to include the layer name, ensuring uniqueness
            # Mapping: e.g., node `0` becomes `('physical', 0)`
            mapping = {node_id: (layer_name, node_id) for node_id in G_raw.nodes()}
            G_relabeled = nx.relabel_nodes(G_raw, mapping, copy=True)
            
            # Preserve original node attributes (like 'original_id' for grids)
            for raw_node, relabeled_node in mapping.items():
                self.supra_graph.nodes[relabeled_node].update(G_raw.nodes[raw_node])

            # 3. Compose the relabeled graph into the main supra-graph
            self.supra_graph = nx.compose(self.supra_graph, G_relabeled)

    def _add_interlayer_edges(self):
        """
        Iterates through the 'interlayer' config and uses the coupling module
        to generate and add edges between the layers.
        """
        couplings = self.config.get('interlayer', [])
        if not couplings:
            logger.warning("No 'interlayer' rules found in config. The layers will be disconnected.")
            return

        # Get all defined layer names for validation purposes inside coupling.py
        all_layer_names = {layer['name'] for layer in self.config.get('layers', [])}

        for rule in couplings:
            try:
                # generate_interlayer_edges handles both validation and edge creation
                new_edges = generate_interlayer_edges(
                    self.supra_graph,
                    rule,
                    all_layer_names
                )

                # Add the generated edges with their attributes (e.g., weight)
                if new_edges:
                    self.supra_graph.add_edges_from(new_edges)

            except (ValueError, NotImplementedError) as e:
                logger.error(f"Could not apply coupling rule {rule}. Reason: {e}")
                # Re-raise the exception to halt the program, as a failed coupling
                # can lead to incorrect simulation results.
                raise

def build_multilayer_network(network_config):
    """
    Facade function to simplify calling the multilayer builder from external scripts.

    Args:
        network_config (dict): The 'network' section from the main YAML config.

    Returns:
        nx.Graph: The constructed multilayer network.
    """
    builder = MultilayerNetwork(network_config)
    return builder.build()