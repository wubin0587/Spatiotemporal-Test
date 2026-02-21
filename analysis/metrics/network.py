"""Network-state metrics for signed/opinion-coupled systems."""

from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx
import numpy as np


def edge_disagreement(
    opinions: Sequence[float] | np.ndarray,
    edges: Iterable[tuple[int, int]],
) -> float:
    """Average absolute opinion difference across edges."""
    x = np.asarray(opinions, dtype=float).reshape(-1)
    edge_list = list(edges)
    if x.size == 0 or not edge_list:
        return 0.0

    diffs = []
    for i, j in edge_list:
        if i < 0 or j < 0 or i >= x.size or j >= x.size:
            continue
        diffs.append(abs(x[i] - x[j]))

    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def signed_balance_ratio(graph: nx.Graph, sign_attr: str = "sign") -> float:
    """Share of balanced triads in a signed graph.

    A triad is balanced if product of edge signs is positive.
    Missing sign defaults to +1.
    """
    triads = 0
    balanced = 0

    for u in graph.nodes:
        nbrs = list(graph.neighbors(u))
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                v, w = nbrs[i], nbrs[j]
                if not graph.has_edge(v, w):
                    continue
                triads += 1
                s_uv = graph[u][v].get(sign_attr, 1)
                s_uw = graph[u][w].get(sign_attr, 1)
                s_vw = graph[v][w].get(sign_attr, 1)
                if s_uv * s_uw * s_vw > 0:
                    balanced += 1

    if triads == 0:
        return 0.0
    return float(balanced / triads)


def global_efficiency(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    return float(nx.global_efficiency(graph))


def modularity_from_partition(
    graph: nx.Graph,
    communities: Sequence[Sequence[int]],
) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    comms = [set(c) for c in communities if len(c) > 0]
    if not comms:
        return 0.0
    return float(nx.algorithms.community.modularity(graph, comms))


def overlap_ratio(edges_a: Iterable[tuple[int, int]], edges_b: Iterable[tuple[int, int]]) -> float:
    """Jaccard overlap for two edge sets (for multilayer coupling)."""

    def _norm(edge: tuple[int, int]) -> tuple[int, int]:
        i, j = edge
        return (i, j) if i <= j else (j, i)

    a = {_norm(e) for e in edges_a}
    b = {_norm(e) for e in edges_b}
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))
