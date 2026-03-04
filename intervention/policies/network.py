# -*- coding: utf-8 -*-
"""
@File    : network.py
@Desc    : Network-topology Intervention Policies.

Policies that modify the static social network graph and rebuild the
adjacency list cached on the engine:

    NetworkRewirePolicy      -- Random edge rewiring (small-world perturbation)
    NetworkRemoveEdgesPolicy -- Sever connections between two agent groups
    NetworkAddEdgesPolicy    -- Inject new connections (e.g., bridging ties)
    NetworkIsolatePolicy     -- Remove ALL edges for a subset of agents

Design notes
------------
All policies update ``engine.network_graph`` **and** rebuild
``engine.static_adjacency`` so that the topology module sees the change
on the very next step.  The graph is modified in-place so that any
external references (e.g. from a visualiser) also see the update.

YAML config examples
--------------------
policy:
  type: network_rewire
  fraction: 0.1        # fraction of edges to rewire
  seed: null           # optional int for reproducibility

policy:
  type: network_remove_edges
  group_a: [0, 1, 2]  # agent indices  (or use group_a_opinion_threshold)
  group_b: [3, 4, 5]

policy:
  type: network_add_edges
  pairs: [[0, 99], [1, 200]]   # explicit pairs, OR use random_k
  random_k: 10                 # add k random inter-group edges

policy:
  type: network_isolate
  agents: [0, 1, 2]   # agent indices to sever from the network
"""

from __future__ import annotations

import logging
import random as _random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from .base import BasePolicy

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _rebuild_adjacency(engine: Any) -> None:
    """Rebuild the engine's static_adjacency list from its network_graph."""
    N = engine.num_agents
    adj = [[] for _ in range(N)]
    for u, v in engine.network_graph.edges():
        if 0 <= u < N and 0 <= v < N:
            adj[u].append(v)
            adj[v].append(u)
    engine.static_adjacency = adj
    logger.debug("static_adjacency rebuilt after network intervention.")


# =============================================================================
# NetworkRewirePolicy
# =============================================================================

class NetworkRewirePolicy(BasePolicy):
    """
    Randomly rewire a fraction of edges in the social graph.

    For each selected edge ``(u, v)``, the policy removes it and adds a new
    edge ``(u, w)`` where ``w`` is chosen uniformly from non-neighbours of
    ``u``.  This is a targeted version of the Watts-Strogatz rewiring step.

    Parameters (config keys)
    ------------------------
    fraction : float
        Fraction of current edges to rewire.  Default ``0.1``.
    seed : int, optional
        Random seed for reproducibility.  If None, uses engine's time_step.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "NetworkRewirePolicy")
        self.fraction = float(self.config.get("fraction", 0.1))
        self.seed = self.config.get("seed", None)

    def _apply(self, engine: Any) -> Optional[Dict]:
        G = engine.network_graph
        seed = self.seed if self.seed is not None else engine.time_step
        rng = _random.Random(seed)

        edges = list(G.edges())
        n_rewire = max(1, int(len(edges) * self.fraction))
        selected = rng.sample(edges, min(n_rewire, len(edges)))

        rewired = 0
        N = engine.num_agents
        for u, v in selected:
            # Remove old edge
            G.remove_edge(u, v)

            # Find candidate new neighbour for u (avoid self and existing neighbours)
            neighbours_u = set(G.neighbors(u))
            candidates = [w for w in range(N) if w != u and w not in neighbours_u]
            if not candidates:
                G.add_edge(u, v)  # Restore if no candidate available
                continue

            w = rng.choice(candidates)
            G.add_edge(u, w)
            rewired += 1

        _rebuild_adjacency(engine)

        logger.info(
            f"[NetworkRewirePolicy] Rewired {rewired}/{len(selected)} edges "
            f"(fraction={self.fraction})."
        )
        return {"rewired": rewired, "total_edges": G.number_of_edges()}

    def describe(self) -> str:
        return f"NetworkRewirePolicy: rewire {self.fraction*100:.1f}% of edges."


# =============================================================================
# NetworkRemoveEdgesPolicy
# =============================================================================

class NetworkRemoveEdgesPolicy(BasePolicy):
    """
    Remove all edges that cross between two specified agent groups.

    Use this to model echo-chamber formation, social fragmentation, or
    deplatforming — situations where two communities lose the ability to
    communicate.

    Parameters (config keys)
    ------------------------
    group_a : list[int]
        Agent indices in the first group.
    group_b : list[int]
        Agent indices in the second group.  If omitted, removes all edges
        from group_a to any agent outside group_a.
    opinion_split : bool
        If True and groups are not explicitly provided, automatically split
        agents at the median opinion value (layer 0).  Default False.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "NetworkRemoveEdgesPolicy")
        self.group_a: Optional[List[int]] = self.config.get("group_a", None)
        self.group_b: Optional[List[int]] = self.config.get("group_b", None)
        self.opinion_split: bool = bool(self.config.get("opinion_split", False))

    def _apply(self, engine: Any) -> Optional[Dict]:
        G = engine.network_graph

        group_a, group_b = self._resolve_groups(engine)
        set_a = set(group_a)
        set_b = set(group_b)

        to_remove = [
            (u, v)
            for u, v in G.edges()
            if (u in set_a and v in set_b) or (u in set_b and v in set_a)
        ]

        G.remove_edges_from(to_remove)
        _rebuild_adjacency(engine)

        logger.info(
            f"[NetworkRemoveEdgesPolicy] Removed {len(to_remove)} cross-group edges."
        )
        return {
            "removed_edges": len(to_remove),
            "group_a_size": len(group_a),
            "group_b_size": len(group_b),
        }

    def _resolve_groups(
        self, engine: Any
    ) -> Tuple[List[int], List[int]]:
        """Return (group_a, group_b) agent index lists."""
        if self.group_a is not None:
            a = self.group_a
            b = (
                self.group_b
                if self.group_b is not None
                else [i for i in range(engine.num_agents) if i not in set(a)]
            )
            return a, b

        if self.opinion_split:
            opinions = engine.opinion_matrix[:, 0]
            median = float(np.median(opinions))
            a = [i for i in range(engine.num_agents) if opinions[i] <= median]
            b = [i for i in range(engine.num_agents) if opinions[i] > median]
            logger.info(
                f"[NetworkRemoveEdgesPolicy] Opinion-split at median={median:.3f}: "
                f"|A|={len(a)}, |B|={len(b)}."
            )
            return a, b

        raise ValueError(
            "NetworkRemoveEdgesPolicy requires either 'group_a'/'group_b' "
            "or 'opinion_split: true'."
        )

    def describe(self) -> str:
        return (
            "NetworkRemoveEdgesPolicy: remove all cross-group edges between "
            "group_a and group_b."
        )


# =============================================================================
# NetworkAddEdgesPolicy
# =============================================================================

class NetworkAddEdgesPolicy(BasePolicy):
    """
    Add new edges to the social graph — bridging ties between previously
    disconnected agents or groups.

    Parameters (config keys)
    ------------------------
    pairs : list[[int, int]], optional
        Explicit list of ``[u, v]`` pairs to connect.
    random_k : int, optional
        If set, adds ``random_k`` random edges between group_a and group_b.
    group_a : list[int], optional
        Source group for random edge generation.
    group_b : list[int], optional
        Target group for random edge generation.
    seed : int, optional
        RNG seed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "NetworkAddEdgesPolicy")
        self.pairs: List[List[int]] = self.config.get("pairs", [])
        self.random_k: int = int(self.config.get("random_k", 0))
        self.group_a: Optional[List[int]] = self.config.get("group_a", None)
        self.group_b: Optional[List[int]] = self.config.get("group_b", None)
        self.seed = self.config.get("seed", None)

    def _apply(self, engine: Any) -> Optional[Dict]:
        G = engine.network_graph
        seed = self.seed if self.seed is not None else engine.time_step
        rng = _random.Random(seed)

        new_edges: List[Tuple[int, int]] = []

        # Explicit pairs
        for pair in self.pairs:
            u, v = int(pair[0]), int(pair[1])
            if not G.has_edge(u, v):
                new_edges.append((u, v))

        # Random inter-group edges
        if self.random_k > 0:
            a = self.group_a or list(range(engine.num_agents // 2))
            b = self.group_b or list(range(engine.num_agents // 2, engine.num_agents))
            attempts = 0
            added_random = 0
            max_attempts = self.random_k * 10
            while added_random < self.random_k and attempts < max_attempts:
                u = rng.choice(a)
                v = rng.choice(b)
                if u != v and not G.has_edge(u, v):
                    new_edges.append((u, v))
                    added_random += 1
                attempts += 1

        G.add_edges_from(new_edges)
        _rebuild_adjacency(engine)

        logger.info(
            f"[NetworkAddEdgesPolicy] Added {len(new_edges)} new edges. "
            f"Total edges now: {G.number_of_edges()}."
        )
        return {"added_edges": len(new_edges), "total_edges": G.number_of_edges()}

    def describe(self) -> str:
        return (
            f"NetworkAddEdgesPolicy: add {len(self.pairs)} explicit + "
            f"{self.random_k} random inter-group edges."
        )


# =============================================================================
# NetworkIsolatePolicy
# =============================================================================

class NetworkIsolatePolicy(BasePolicy):
    """
    Sever all social connections for a specified set of agents.

    Useful for modelling deplatforming, quarantine, or targeted censorship
    where specific high-influence nodes are disconnected from the graph.

    Parameters (config keys)
    ------------------------
    agents : list[int]
        Agent indices to isolate.
    top_k_by_degree : int, optional
        Instead of specifying agents explicitly, automatically select the
        ``top_k_by_degree`` highest-degree nodes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(config, name or "NetworkIsolatePolicy")
        self.agents: Optional[List[int]] = self.config.get("agents", None)
        self.top_k: int = int(self.config.get("top_k_by_degree", 0))

    def _apply(self, engine: Any) -> Optional[Dict]:
        G = engine.network_graph

        targets = self._resolve_targets(G, engine.num_agents)

        removed_edges = 0
        for node in targets:
            neighbours = list(G.neighbors(node))
            for nb in neighbours:
                G.remove_edge(node, nb)
                removed_edges += 1

        _rebuild_adjacency(engine)

        logger.info(
            f"[NetworkIsolatePolicy] Isolated {len(targets)} agents, "
            f"removed {removed_edges} edges."
        )
        return {
            "isolated_agents": targets,
            "removed_edges": removed_edges,
        }

    def _resolve_targets(self, G: nx.Graph, num_agents: int) -> List[int]:
        if self.agents is not None:
            return [int(a) for a in self.agents]
        if self.top_k > 0:
            degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in degrees[: self.top_k]]
        raise ValueError(
            "NetworkIsolatePolicy requires 'agents' list or 'top_k_by_degree'."
        )

    def describe(self) -> str:
        if self.agents is not None:
            return f"NetworkIsolatePolicy: isolate agents {self.agents}."
        return f"NetworkIsolatePolicy: isolate top-{self.top_k} degree nodes."
