"""
analysis/feature/extractor.py

Feature Extraction Module
--------------------------
Computes structured feature vectors from a simulation state snapshot.
Wraps all analysis/metrics functions into a single callable pipeline.

Input contract (snapshot dict from StepExecutor.get_state_snapshot / run()):
    {
        'opinions':   np.ndarray  (N, L)  float32  [0, 1]
        'positions':  np.ndarray  (N, 2)  float32  [0, 1]
        'impact':     np.ndarray  (N,)    float32  >= 0
        'time':       float
        'step':       int
    }

Optional extras passed separately:
    graph         nx.Graph         social network, integer nodes [0, N-1]
    event_times   np.ndarray (M,)
    event_locs    np.ndarray (M, 2)
    event_intensities np.ndarray (M,)
    event_sources list[str]

Returns:
    dict[str, float | dict]  – flat or nested feature dict

Data structure validity notes
------------------------------
LEGAL:
  opinions (N, L)       – every metric in opinion.py accepts 1-D slice (N,)
  positions (N, 2)      – every metric in spatial.py accepts (N, 2)
  impact (N,)           – edge_disagreement / edge_homophily take (N,) + edge list
  graph nx.Graph        – all topo/network metrics accept nx.Graph
  event_times (M,)      – all event metrics accept 1-D sequence

ILLEGAL / EDGE-CASES GUARDED:
  N == 0                – metrics raise; guarded with early return {}
  L == 0                – slicing opinions[:, 0] raises IndexError; guarded
  M < 2                 – interevent_times returns empty; burstiness returns 0.0
  moran_i needs (N,N)   – we build binary adjacency weight matrix from graph here
  modularity needs ≥1   – guarded via edge_count check
  opinion variance ddof  – using ddof=0 (population) throughout for consistency
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import networkx as nx

# ── opinion metrics ──────────────────────────────────────────────────────────
from analysis.metrics.opinion import (
    bimodality_coefficient,
    edge_homophily_score,
    extreme_share,
    homophilous_bimodality_index,
    mean_opinion,
    opinion_entropy,
    opinion_variance,
    polarization_std,
)

# ── spatial metrics ───────────────────────────────────────────────────────────
from analysis.metrics.spatial import (
    centroid,
    mean_pairwise_distance,
    moran_i,
    nearest_neighbor_index,
    radius_of_gyration,
    spatial_entropy,
)

# ── topological metrics ───────────────────────────────────────────────────────
from analysis.metrics.topo import (
    average_clustering,
    average_degree,
    average_shortest_path_lcc,
    degree_assortativity,
    degree_gini,
    density,
    edge_count,
    largest_component_ratio,
    node_count,
)

# ── network / opinion-coupling metrics ───────────────────────────────────────
from analysis.metrics.network import (
    edge_disagreement,
    global_efficiency,
    modularity_from_partition,
    overlap_ratio,
    signed_balance_ratio,
)

# ── event metrics ─────────────────────────────────────────────────────────────
from analysis.metrics.event import (
    burstiness_index,
    event_intensity_stats,
    event_rate,
    event_spatial_spread,
    interevent_times,
    temporal_gini,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _edge_list(graph: nx.Graph) -> List[tuple]:
    """Return list of (u, v) integer tuples from graph."""
    return list(graph.edges())


def _build_weight_matrix(graph: nx.Graph, n: int) -> np.ndarray:
    """
    Build a binary (0/1) adjacency weight matrix required by moran_i.

    moran_i expects:
        weights: np.ndarray shape (N, N), square, same length as values.

    DATA STRUCTURE VALIDITY:
        - Node IDs must be integers in [0, n-1].  builder.py guarantees this.
        - If graph has fewer nodes than n the missing rows/cols stay 0 (isolated).
        - Self-loops would add 1 to the diagonal; we zero the diagonal after.
    """
    W = np.zeros((n, n), dtype=float)
    for u, v in graph.edges():
        if 0 <= u < n and 0 <= v < n:
            W[u, v] = 1.0
            W[v, u] = 1.0
    np.fill_diagonal(W, 0.0)
    return W


def _community_partition(graph: nx.Graph) -> List[List[int]]:
    """
    Detect communities with greedy modularity (fast, deterministic-ish).
    Returns list of node-id lists.  Falls back to [{all nodes}] on failure.
    """
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        return [list(c) for c in communities]
    except Exception:
        return [list(graph.nodes())]


# ═════════════════════════════════════════════════════════════════════════════
# Core extraction functions (one per metric family)
# ═════════════════════════════════════════════════════════════════════════════

def extract_opinion_features(
    opinions: np.ndarray,
    graph: Optional[nx.Graph] = None,
    layer_idx: int = 0,
    opinion_range: tuple = (0.0, 1.0),
    extreme_threshold: float = 0.8,
) -> Dict[str, float]:
    """
    Compute all opinion-space metrics for a single opinion layer.

    DATA STRUCTURE REQUIREMENTS:
        opinions: (N, L) or (N,)   float, values in [0, 1]
        graph:    nx.Graph with integer node IDs in [0, N-1]  (optional)

    VALIDITY CHECKS PERFORMED:
        - N == 0  → return {}
        - L == 0  → return {}
        - layer_idx out of range → clamp to last valid layer
    """
    if opinions.ndim == 1:
        op = opinions.astype(float)
    else:
        n, L = opinions.shape
        if n == 0 or L == 0:
            return {}
        layer_idx = min(layer_idx, L - 1)
        op = opinions[:, layer_idx].astype(float)

    if op.size == 0:
        return {}

    feats: Dict[str, float] = {
        "mean_opinion":           mean_opinion(op),
        "opinion_variance":       opinion_variance(op, ddof=0),
        "polarization_std":       polarization_std(op),
        "extreme_share":          extreme_share(op, threshold=extreme_threshold),
        "bimodality_coefficient": bimodality_coefficient(op),
        "opinion_entropy":        opinion_entropy(op, bins=20),
    }

    if graph is not None and graph.number_of_edges() > 0:
        edges = _edge_list(graph)
        feats["edge_homophily_score"] = edge_homophily_score(
            op, edges, opinion_range=opinion_range
        )
        feats["homophilous_bimodality_index"] = homophilous_bimodality_index(
            op, edges, alpha=0.5, opinion_range=opinion_range
        )
        feats["edge_disagreement"] = edge_disagreement(op, edges)

    return feats


def extract_spatial_features(
    positions: np.ndarray,
    values: Optional[np.ndarray] = None,
    weight_matrix: Optional[np.ndarray] = None,
    area: float = 1.0,
) -> Dict[str, float]:
    """
    Compute spatial distribution metrics.

    DATA STRUCTURE REQUIREMENTS:
        positions:      (N, 2)  float, values in [0, 1]
        values:         (N,)    float (optional, used for Moran's I)
        weight_matrix:  (N, N)  float (optional, required for Moran's I)
                        If None but values provided, a unit matrix is attempted.
        area:           float > 0  (default 1.0 = unit square)

    VALIDITY CHECKS PERFORMED:
        - N < 2  → skip metrics that need ≥2 points
        - weight_matrix shape mismatch → skip moran_i
    """
    n = positions.shape[0]
    if n == 0:
        return {}

    feats: Dict[str, float] = {}

    cx, cy = centroid(positions)
    feats["centroid_x"] = float(cx)
    feats["centroid_y"] = float(cy)
    feats["radius_of_gyration"] = radius_of_gyration(positions)
    feats["spatial_entropy"] = spatial_entropy(positions, x_bins=10, y_bins=10)

    if n >= 2:
        feats["mean_pairwise_distance"] = mean_pairwise_distance(positions)
        feats["nearest_neighbor_index"] = nearest_neighbor_index(positions, area=area)

    # Moran's I: requires values (N,) and weight matrix (N, N)
    if values is not None and len(values) == n:
        W = weight_matrix
        if W is None:
            # Fallback: identity-like uniform weights (not meaningful, but type-safe)
            W = np.ones((n, n), dtype=float)
            np.fill_diagonal(W, 0.0)

        # DATA STRUCTURE VALIDITY: W must be (n, n)
        if isinstance(W, np.ndarray) and W.shape == (n, n):
            feats["moran_i"] = moran_i(values.astype(float), W)
        # else: skip — shape mismatch, not a hard error

    return feats


def extract_topo_features(graph: nx.Graph) -> Dict[str, float]:
    """
    Compute graph topology metrics.

    DATA STRUCTURE REQUIREMENTS:
        graph: nx.Graph, integer node IDs in [0, N-1]

    VALIDITY CHECKS PERFORMED:
        - Empty graph (0 nodes) → return {}
        - 0 edges → skip path-length and modularity
    """
    if graph.number_of_nodes() == 0:
        return {}

    feats: Dict[str, float] = {
        "node_count":              float(node_count(graph)),
        "edge_count":              float(edge_count(graph)),
        "density":                 density(graph),
        "average_degree":          average_degree(graph),
        "average_clustering":      average_clustering(graph),
        "degree_gini":             degree_gini(graph),
        "largest_component_ratio": largest_component_ratio(graph),
        "degree_assortativity":    degree_assortativity(graph),
    }

    if graph.number_of_edges() > 0:
        feats["average_shortest_path_lcc"] = average_shortest_path_lcc(graph)
        feats["global_efficiency"] = global_efficiency(graph)

        # Modularity from greedy community detection
        partition = _community_partition(graph)
        feats["modularity"] = modularity_from_partition(graph, partition)

    return feats


def extract_network_opinion_features(
    opinions: np.ndarray,
    graph: nx.Graph,
    layer_idx: int = 0,
) -> Dict[str, float]:
    """
    Compute network×opinion coupling metrics.

    DATA STRUCTURE REQUIREMENTS:
        opinions: (N, L) or (N,) float [0,1], N must equal graph.number_of_nodes()
        graph:    nx.Graph with integer nodes in [0, N-1]

    VALIDITY CHECKS PERFORMED:
        - N mismatch between opinions and graph → return {}
        - 0 edges → skip edge-based metrics
    """
    n_graph = graph.number_of_nodes()

    if opinions.ndim == 1:
        op = opinions.astype(float)
        n_op = len(op)
    else:
        if opinions.shape[1] == 0:
            return {}
        layer_idx = min(layer_idx, opinions.shape[1] - 1)
        op = opinions[:, layer_idx].astype(float)
        n_op = len(op)

    # DATA STRUCTURE VALIDITY: opinion vector length must match graph node count
    if n_op != n_graph:
        return {}

    feats: Dict[str, float] = {}

    if graph.number_of_edges() > 0:
        edges = _edge_list(graph)
        feats["edge_disagreement"] = edge_disagreement(op, edges)
        feats["edge_homophily_score"] = edge_homophily_score(
            op, edges, opinion_range=(0.0, 1.0)
        )
        feats["signed_balance_ratio"] = signed_balance_ratio(graph)

    return feats


def extract_event_features(
    event_times: np.ndarray,
    event_intensities: Optional[np.ndarray] = None,
    event_locs: Optional[np.ndarray] = None,
    horizon: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute event-stream statistics.

    DATA STRUCTURE REQUIREMENTS:
        event_times:       (M,)    float, monotonically increasing (sorted internally)
        event_intensities: (M,)    float, >= 0  (optional)
        event_locs:        (M, 2)  float        (optional, for spatial spread)
        horizon:           float   total time window (optional)

    VALIDITY CHECKS PERFORMED:
        - M == 0  → return {}
        - M == 1  → interevent_times returns empty → burstiness/gini return 0.0
        - event_locs shape must be (M, 2) for spatial_spread
    """
    if event_times is None or len(event_times) == 0:
        return {}

    feats: Dict[str, Any] = {
        "event_rate":     event_rate(event_times, horizon=horizon),
        "burstiness":     burstiness_index(event_times),
        "temporal_gini":  temporal_gini(event_times),
    }

    if event_intensities is not None and len(event_intensities) == len(event_times):
        stats = event_intensity_stats(event_intensities)
        feats["intensity_mean"] = stats["mean"]
        feats["intensity_std"]  = stats["std"]
        feats["intensity_max"]  = stats["max"]
        feats["intensity_min"]  = stats["min"]

    # DATA STRUCTURE VALIDITY: event_locs must be (M, 2)
    if (
        event_locs is not None
        and isinstance(event_locs, np.ndarray)
        and event_locs.ndim == 2
        and event_locs.shape[1] == 2
        and event_locs.shape[0] == len(event_times)
    ):
        feats["event_spatial_spread"] = event_spatial_spread(event_locs)

    return feats


# ═════════════════════════════════════════════════════════════════════════════
# Full-snapshot extractor
# ═════════════════════════════════════════════════════════════════════════════

def extract_all_features(
    snapshot: Dict[str, Any],
    graph: Optional[nx.Graph] = None,
    event_times: Optional[np.ndarray] = None,
    event_intensities: Optional[np.ndarray] = None,
    event_locs: Optional[np.ndarray] = None,
    layer_idx: int = 0,
) -> Dict[str, Any]:
    """
    Master extractor: runs all feature families on a simulation snapshot.

    Parameters
    ----------
    snapshot : dict  (from StepExecutor.get_state_snapshot() or run() results)
        Required keys: 'opinions', 'positions', 'impact'
        Optional keys: 'time', 'step'
    graph : nx.Graph   Social network.  Node IDs must be ints in [0, N-1].
    event_times, event_intensities, event_locs : from EventManager.get_state_vectors()
    layer_idx : which opinion layer to focus on for scalar metrics

    Returns
    -------
    dict with keys:
        'meta'    – time / step
        'opinion' – opinion-space features for layer_idx
        'spatial' – spatial distribution features
        'topo'    – graph topology features  (empty if graph is None)
        'network_opinion' – network×opinion coupling  (empty if graph is None)
        'event'   – event-stream features   (empty if event_times is None)

    DATA STRUCTURE VALIDITY SUMMARY
    --------------------------------
    opinions  (N, L): L must be >= 1; layer_idx is clamped to [0, L-1]
    positions (N, 2): must match N from opinions
    impact    (N,):   must match N from opinions
    graph nodes must equal N; guaranteed by EngineInterface.validate_network_structure
    event_times (M,): may be empty → event block returns {}
    moran_i weight matrix: built from graph adjacency → always (N,N) if graph given
    """
    opinions  = np.asarray(snapshot["opinions"],  dtype=float)
    positions = np.asarray(snapshot["positions"], dtype=float)
    impact    = np.asarray(snapshot["impact"],    dtype=float)

    # ── DATA STRUCTURE CROSS-CHECK ────────────────────────────────────────────
    n_op  = opinions.shape[0]
    n_pos = positions.shape[0]
    n_imp = impact.shape[0]

    issues = []
    if n_pos != n_op:
        issues.append(f"positions N={n_pos} != opinions N={n_op}")
    if n_imp != n_op:
        issues.append(f"impact N={n_imp} != opinions N={n_op}")
    if graph is not None and graph.number_of_nodes() != n_op:
        issues.append(
            f"graph nodes={graph.number_of_nodes()} != opinions N={n_op}"
        )
    if positions.ndim != 2 or positions.shape[1] != 2:
        issues.append(f"positions shape {positions.shape} is not (N, 2)")
    if impact.ndim != 1:
        issues.append(f"impact shape {impact.shape} is not (N,)")

    # ── BUILD WEIGHT MATRIX for Moran's I ────────────────────────────────────
    W = _build_weight_matrix(graph, n_op) if graph is not None else None

    # ── EXTRACT ───────────────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "meta": {
            "time":   snapshot.get("time", None),
            "step":   snapshot.get("step", None),
            "n_agents": n_op,
            "data_issues": issues,   # empty list == all valid
        },
        "opinion": extract_opinion_features(
            opinions, graph=graph, layer_idx=layer_idx
        ),
        "spatial": extract_spatial_features(
            positions,
            values=impact,        # use impact as spatial value for Moran's I
            weight_matrix=W,
            area=1.0,
        ),
        "topo": extract_topo_features(graph) if graph is not None else {},
        "network_opinion": (
            extract_network_opinion_features(opinions, graph, layer_idx=layer_idx)
            if graph is not None else {}
        ),
        "event": extract_event_features(
            event_times       if event_times is not None else np.array([]),
            event_intensities=event_intensities,
            event_locs=event_locs,
            horizon=snapshot.get("time", None),
        ),
    }

    return result
