"""
Online Resonance Generator
--------------------------
This module detects network-level homophily resonance patterns and emits
endogenous online events when communities exhibit sustained convergence
or conflict trends.
"""

from collections import deque
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components

from ..base import Event
from ..generator import EventGenerator
from .dist import time as time_dist


class OnlineResonanceGenerator(EventGenerator):
    """
    Generate online resonance events from smoothed homophily dynamics.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.check_interval = max(1, int(config.get("check_interval", 1)))
        self.smoothing_window = max(2, int(config.get("smoothing_window", 5)))
        self.convergence_threshold = float(config.get("convergence_threshold", 0.01))
        self.conflict_threshold = float(config.get("conflict_threshold", 0.01))
        self.min_community_size = int(config.get("min_community_size", 3))

        self.layer_weights = np.asarray(config.get("layer_weights", []), dtype=float)
        self.attributes_config = config.get("attributes", {})

        self.logger = logging.getLogger(__name__)

        self._network_layers: Optional[List[csr_matrix]] = None
        self._homophily_history: deque = deque(maxlen=self.smoothing_window)
        self._step_counter = 0

    def set_network_layers(self, layers: List[Any]):
        """
        Inject network layers after engine topology initialization.
        """
        if not layers:
            self._network_layers = []
            return

        converted = []
        for layer in layers:
            if issparse(layer):
                converted.append(layer.tocsr())
            else:
                converted.append(csr_matrix(np.asarray(layer)))

        self._network_layers = converted

    def step(self,
             current_time: float,
             agents_state: Any = None,
             env_state: Any = None,
             event_history: List[Event] = None) -> List[Event]:
        """
        Evaluate resonance conditions and emit online events if triggered.
        """
        del env_state, event_history

        self._step_counter += 1
        if (self._step_counter % self.check_interval) != 0:
            return []

        if agents_state is None:
            return []

        opinions = agents_state.get("opinions")
        positions = agents_state.get("positions")
        if opinions is None or positions is None:
            return []

        homophily_vec = self._compute_homophily(opinions)
        self._homophily_history.append(homophily_vec)

        if len(self._homophily_history) < self.smoothing_window:
            return []

        smooth_delta = self._compute_smooth_delta()

        convergence_candidates = np.where(smooth_delta > self.convergence_threshold)[0]
        conflict_candidates = np.where(smooth_delta < -self.conflict_threshold)[0]

        new_events: List[Event] = []

        for event_type, candidates in (
            ("convergence", convergence_candidates),
            ("conflict", conflict_candidates),
        ):
            communities = self._find_communities(candidates)
            valid_communities = [c for c in communities if len(c) >= self.min_community_size]

            for community in valid_communities:
                new_events.append(
                    self._create_online_event(
                        community_agents=community,
                        event_type=event_type,
                        opinions=opinions,
                        positions=positions,
                        smooth_delta=smooth_delta,
                        current_time=current_time,
                    )
                )

        return new_events

    def _compute_homophily(self, opinions: np.ndarray) -> np.ndarray:
        """
        Compute per-agent weighted homophily over all network layers.
        """
        if self._network_layers is None or len(self._network_layers) == 0:
            return np.zeros(opinions.shape[0], dtype=float)

        n_layers = len(self._network_layers)
        n_agents, opinion_dim = opinions.shape

        if len(self.layer_weights) > 0 and len(self.layer_weights) != n_layers:
            self.logger.warning(
                "online_resonance.layer_weights length mismatch: got %d, expected %d; fallback to uniform weights.",
                len(self.layer_weights),
                n_layers,
            )

        if len(self.layer_weights) == n_layers and np.sum(self.layer_weights) > 0:
            weights = self.layer_weights
        else:
            weights = np.ones(n_layers, dtype=float)

        weighted_sum = np.zeros(n_agents, dtype=float)

        for idx, mat in enumerate(self._network_layers):
            degrees = np.asarray(mat.sum(axis=1)).ravel()
            degrees_safe = np.where(degrees > 0, degrees, 1.0)

            layer_homophily = np.zeros(n_agents, dtype=float)
            for dim_idx in range(opinion_dim):
                opinion_col = opinions[:, dim_idx]
                neighbor_mean = np.asarray(mat.dot(opinion_col)).ravel() / degrees_safe
                dim_homophily = 1.0 - np.abs(opinion_col - neighbor_mean)
                layer_homophily += np.clip(dim_homophily, 0.0, 1.0)

            layer_homophily /= opinion_dim
            weighted_sum += weights[idx] * layer_homophily

        return weighted_sum / np.sum(weights)

    def _compute_smooth_delta(self) -> np.ndarray:
        """
        Compute normalized slope from oldest to newest homophily snapshots.
        """
        oldest = self._homophily_history[0]
        newest = self._homophily_history[-1]
        norm = max(len(self._homophily_history), 1)
        return (newest - oldest) / norm

    def _find_communities(self, candidate_indices: np.ndarray) -> List[np.ndarray]:
        """
        Find connected candidate communities on the union network graph.
        """
        if candidate_indices is None or len(candidate_indices) == 0:
            return []
        if self._network_layers is None or len(self._network_layers) == 0:
            return []

        union_graph = self._network_layers[0].astype(bool)
        for mat in self._network_layers[1:]:
            union_graph = union_graph + mat.astype(bool)
        union_graph = union_graph.astype(np.int8)

        subgraph = union_graph[np.ix_(candidate_indices, candidate_indices)]
        n_components, labels = connected_components(subgraph, directed=False, return_labels=True)

        communities: List[np.ndarray] = []
        for comp in range(n_components):
            local_indices = np.where(labels == comp)[0]
            global_indices = candidate_indices[local_indices]
            communities.append(global_indices)

        return communities

    def _create_online_event(self,
                             community_agents: np.ndarray,
                             event_type: str,
                             opinions: np.ndarray,
                             positions: np.ndarray,
                             smooth_delta: np.ndarray,
                             current_time: float) -> Event:
        """
        Build one online resonance event from a detected community.
        """
        community_positions = positions[community_agents]
        community_opinions = opinions[community_agents]
        community_delta = smooth_delta[community_agents]

        # L: Event location is community centroid in physical space.
        loc = np.mean(community_positions, axis=0)

        # I: Intensity scales with size and trend strength.
        int_conf = self.attributes_config.get("intensity", {})
        base_value = float(int_conf.get("base_value", 4.0))
        size_scale = float(int_conf.get("size_scale", 8.0))
        mean_abs_delta = float(np.abs(np.mean(community_delta)))
        intensity = base_value + size_scale * len(community_agents) * mean_abs_delta
        intensity = max(intensity, 1.0)

        # C: Average topic vector, normalized.
        content = np.mean(community_opinions, axis=0)
        content = np.clip(content, 0.0, None)
        content_sum = np.sum(content)
        if content_sum <= 1e-12:
            content = np.ones(opinions.shape[1]) / opinions.shape[1]
        else:
            content = content / content_sum

        # P: Convergence gives lower polarity, conflict gives higher polarity.
        opinion_std = float(np.mean(np.std(community_opinions, axis=0)))
        polarity_magnitude = np.clip(opinion_std * 2.0, 0.0, 1.0)
        if event_type == "convergence":
            polarity = -polarity_magnitude
            source = "online_resonance_convergence"
        else:
            polarity = polarity_magnitude
            source = "online_resonance_conflict"

        # Spatial diffusion sigma derives from member dispersion.
        diff_conf = self.attributes_config.get("diffusion", {})
        distances = np.linalg.norm(community_positions - loc, axis=1)
        dispersion = float(np.std(distances)) if len(distances) > 0 else 0.0
        sigma_scale = float(diff_conf.get("dispersion_scale", 1.0))
        min_sigma = float(diff_conf.get("min_sigma", 0.03))
        max_sigma = float(diff_conf.get("max_sigma", 0.3))
        sigma = np.clip(dispersion * sigma_scale, min_sigma, max_sigma)
        spatial_params = {"sigma": float(sigma)}

        life_conf = self.attributes_config.get("lifecycle", {})
        temporal_params = time_dist.sample_lifecycle_params(
            self.rng,
            life_conf.get("type", "uniform"),
            life_conf,
        )

        mean_smooth_delta = float(np.mean(community_delta))

        return Event(
            uid=self._get_next_id(),
            time=current_time,
            loc=loc,
            intensity=float(intensity),
            content=content,
            polarity=float(polarity),
            spatial_params=spatial_params,
            temporal_params=temporal_params,
            source=source,
            meta={
                "community_size": int(len(community_agents)),
                "community_agents": community_agents.tolist(),
                "event_type": event_type,
                "mean_smooth_delta": mean_smooth_delta,
            },
        )
