"""Topological metrics for graphs (NetworkX-based)."""

from __future__ import annotations

import networkx as nx
import numpy as np


def node_count(graph: nx.Graph) -> int:
    return int(graph.number_of_nodes())


def edge_count(graph: nx.Graph) -> int:
    return int(graph.number_of_edges())


def density(graph: nx.Graph) -> float:
    return float(nx.density(graph))


def average_degree(graph: nx.Graph) -> float:
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    return float(2.0 * graph.number_of_edges() / n)


def average_clustering(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(nx.average_clustering(graph))


def degree_gini(graph: nx.Graph) -> float:
    degrees = np.asarray([d for _, d in graph.degree()], dtype=float)
    if degrees.size == 0:
        return 0.0
    mean_deg = float(np.mean(degrees))
    if mean_deg <= 1e-12:
        return 0.0

    diff_sum = np.abs(degrees[:, None] - degrees[None, :]).sum()
    gini = diff_sum / (2.0 * degrees.size**2 * mean_deg)
    return float(gini)


def largest_component_ratio(graph: nx.Graph) -> float:
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(graph)), default=0)
    return float(largest / n)


def average_shortest_path_lcc(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0

    largest_nodes = max(nx.connected_components(graph), key=len, default=set())
    if len(largest_nodes) <= 1:
        return 0.0

    subgraph = graph.subgraph(largest_nodes)
    return float(nx.average_shortest_path_length(subgraph))


def degree_assortativity(graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    value = nx.degree_assortativity_coefficient(graph)
    if np.isnan(value):
        return 0.0
    return float(value)
