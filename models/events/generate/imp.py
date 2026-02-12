# -*- coding: utf-8 -*-
"""
@File    : imp.py
@Desc    : Endogenous Threshold-Based Event Generator (The "Grey Rhinos")
           
           This module generates events that emerge from the internal state of the system.
           It monitors agent attributes (emotions, opinion extremism, spatial density) and
           triggers real-world events when critical thresholds are exceeded.
           
           Examples:
           - Online anger accumulates -> Offline protest
           - High opinion polarization in a region -> Political rally
           - Dense clustering of extreme opinions -> Flash mob
           
           This implements the "Online-to-Offline Feedback" mechanism described in README.md.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Import base classes
from ..base import Event
from ..generator import EventGenerator

# Import distribution utilities for generating event attributes
from .dist import spatial as spatial_dist
from .dist import time as time_dist


class EndogenousThresholdGenerator(EventGenerator):
    """
    Generates events when monitored agent attributes exceed critical thresholds
    in localized spatial regions.
    
    Mechanism:
    1. Divide the map into a spatial grid
    2. For each cell, calculate aggregate metrics (density, mean emotion, opinion variance)
    3. Check if any metric crosses its threshold
    4. If triggered, generate an event at that location with properties derived from
       the local agent population
    5. Apply cooldown to prevent continuous triggering
    
    This creates a feedback loop: Strong online sentiment -> Offline event -> More impact
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the endogenous threshold generator.
        
        Expected config structure:
        {
            'enabled': True,
            'seed': 2025,
            'monitor_attribute': 'opinion_extremism',  # or 'emotion_anger', 'density'
            'critical_threshold': 0.75,
            'grid_resolution': 20,
            'min_agents_in_cell': 5,
            'cooldown': 50,
            'attributes': {
                'intensity': {...},
                'content': {...},
                'polarity': {...},
                'diffusion': {...},
                'lifecycle': {...}
            }
        }
        """
        super().__init__(config)
        
        # Extract configuration parameters
        self.monitor_attr = config.get('monitor_attribute', 'opinion_extremism')
        self.threshold = config.get('critical_threshold', 0.75)
        self.grid_res = config.get('grid_resolution', 20)
        self.min_agents = config.get('min_agents_in_cell', 5)
        self.cooldown_period = config.get('cooldown', 50)
        
        # Attributes configuration for generated events
        self.attr_config = config.get('attributes', {})
        
        # Initialize grid structure
        self._init_grid()
        
        # Cooldown tracking: {(grid_x, grid_y): time_of_last_event}
        self.cell_cooldowns: Dict[Tuple[int, int], float] = {}
        
        self.logger = None
        try:
            import logging
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"EndogenousThresholdGenerator initialized: "
                           f"monitoring '{self.monitor_attr}', threshold={self.threshold}")
        except:
            pass
    
    def _init_grid(self):
        """
        Initialize the spatial grid for partitioning agents.
        
        Grid cells are indexed as (i, j) where i, j in [0, grid_res-1]
        Each cell covers a square region of size (1/grid_res) x (1/grid_res)
        """
        self.cell_size = 1.0 / self.grid_res
        
        # Pre-calculate cell boundaries for efficiency
        self.cell_boundaries = []
        for i in range(self.grid_res):
            for j in range(self.grid_res):
                x_min = i * self.cell_size
                x_max = (i + 1) * self.cell_size
                y_min = j * self.cell_size
                y_max = (j + 1) * self.cell_size
                self.cell_boundaries.append({
                    'idx': (i, j),
                    'bounds': (x_min, x_max, y_min, y_max),
                    'center': ((x_min + x_max) / 2, (y_min + y_max) / 2)
                })
    
    def step(self, current_time: float, 
             agents_state: Any = None, 
             env_state: Any = None,
             event_history: List[Event] = None) -> List[Event]:
        """
        Scan the agent population and trigger events in cells that exceed thresholds.
        
        Args:
            current_time: Current simulation time.
            agents_state: REQUIRED. Dictionary with keys:
                         - 'positions': np.ndarray of shape (N, 2)
                         - 'opinions': np.ndarray of shape (N, L)
                         Optional keys:
                         - 'emotions': np.ndarray of shape (N, E) for emotion monitoring
            env_state: Optional environment data.
            event_history: Not used by this generator (used by cascade generator).
        
        Returns:
            List[Event]: Newly triggered endogenous events.
        """
        if agents_state is None:
            # Cannot function without agent state
            return []
        
        # Extract agent data
        positions = agents_state.get('positions')
        opinions = agents_state.get('opinions')
        
        if positions is None or opinions is None:
            if self.logger:
                self.logger.warning("Missing agent positions or opinions. Cannot generate events.")
            return []
        
        # Partition agents into grid cells
        cell_agents = self._partition_agents_to_grid(positions, opinions, agents_state)
        
        # Check each cell for threshold violations
        triggered_events = []
        
        for cell_info in self.cell_boundaries:
            cell_idx = cell_info['idx']
            cell_center = cell_info['center']
            
            # Skip if in cooldown
            if self._is_in_cooldown(cell_idx, current_time):
                continue
            
            # Get agents in this cell
            agents_in_cell = cell_agents.get(cell_idx, [])
            
            # Skip if too few agents
            if len(agents_in_cell) < self.min_agents:
                continue
            
            # Calculate the monitored metric for this cell
            metric_value = self._calculate_cell_metric(agents_in_cell, opinions)
            
            # Check threshold
            if metric_value >= self.threshold:
                # TRIGGER EVENT!
                new_event = self._create_event(
                    t=current_time,
                    trigger_location=cell_center,
                    cell_data={
                        'agents': agents_in_cell,
                        'metric_value': metric_value,
                        'opinions': opinions[agents_in_cell] if len(agents_in_cell) > 0 else None
                    },
                    agent_positions=positions  # Pass full position matrix
                )
                
                triggered_events.append(new_event)
                
                # Set cooldown for this cell
                self.cell_cooldowns[cell_idx] = current_time
                
                if self.logger:
                    self.logger.debug(f"Endogenous event triggered at cell {cell_idx}: "
                                    f"metric={metric_value:.3f} > threshold={self.threshold}")
        
        return triggered_events
    
    def _partition_agents_to_grid(self, positions: np.ndarray, 
                                   opinions: np.ndarray,
                                   agents_state: Dict) -> Dict[Tuple[int, int], List[int]]:
        """
        Assigns each agent to a grid cell based on their position.
        
        Args:
            positions: Agent positions (N, 2)
            opinions: Agent opinions (N, L)
            agents_state: Full agent state dictionary
        
        Returns:
            Dict mapping (grid_x, grid_y) -> [list of agent indices]
        """
        cell_agents = defaultdict(list)
        
        N = len(positions)
        
        for agent_id in range(N):
            x, y = positions[agent_id]
            
            # Determine grid cell
            # Ensure positions are clipped to [0, 1] to avoid out-of-bounds
            x = np.clip(x, 0.0, 0.999999)
            y = np.clip(y, 0.0, 0.999999)
            
            grid_i = int(x * self.grid_res)
            grid_j = int(y * self.grid_res)
            
            cell_agents[(grid_i, grid_j)].append(agent_id)
        
        return cell_agents
    
    def _calculate_cell_metric(self, agent_indices: List[int], 
                               opinions: np.ndarray) -> float:
        """
        Calculate the monitored metric for a cell's agent population.
        
        Args:
            agent_indices: List of agent IDs in the cell
            opinions: Full opinion matrix (N, L)
        
        Returns:
            float: Metric value to compare against threshold
        """
        if len(agent_indices) == 0:
            return 0.0
        
        # Get opinions of agents in this cell
        cell_opinions = opinions[agent_indices]
        
        # Calculate metric based on configured attribute
        metric_type = self.monitor_attr.lower()
        
        if metric_type == 'opinion_extremism':
            # Measure how extreme opinions are (distance from center 0.5)
            # High extremism = opinions clustered near 0 or 1
            distances_from_center = np.abs(cell_opinions - 0.5)
            metric_value = np.mean(distances_from_center)
            
        elif metric_type == 'opinion_variance':
            # Measure polarization within the cell
            metric_value = np.std(cell_opinions)
            
        elif metric_type == 'density':
            # Simply the density (number of agents per unit area)
            cell_area = self.cell_size ** 2
            metric_value = len(agent_indices) / cell_area
            # Normalize to [0, 1] range (assume max density ~100 agents per cell)
            metric_value = min(metric_value / 100.0, 1.0)
            
        elif 'emotion' in metric_type:
            # For emotion-based monitoring
            # This requires 'emotions' key in agents_state
            # For now, we'll use a proxy: opinion extremism
            distances_from_center = np.abs(cell_opinions - 0.5)
            metric_value = np.mean(distances_from_center)
            
        else:
            # Default: use opinion variance
            metric_value = np.std(cell_opinions)
        
        return float(metric_value)
    
    def _is_in_cooldown(self, cell_idx: Tuple[int, int], current_time: float) -> bool:
        """
        Check if a cell is in cooldown period.
        
        Args:
            cell_idx: Grid cell coordinates (i, j)
            current_time: Current simulation time
        
        Returns:
            bool: True if in cooldown, False otherwise
        """
        if cell_idx not in self.cell_cooldowns:
            return False
        
        last_event_time = self.cell_cooldowns[cell_idx]
        time_since_event = current_time - last_event_time
        
        return time_since_event < self.cooldown_period
    
    def _create_event(self, t: float, 
                     trigger_location: Tuple[float, float],
                     cell_data: Dict,
                     agent_positions: np.ndarray) -> Event:
        """
        Create an Event object based on the triggering cell's state.
        
        Args:
            t: Current time
            trigger_location: (x, y) coordinates of the cell center (fallback)
            cell_data: Dictionary containing:
                      - 'agents': List of agent indices in the cell
                      - 'metric_value': The metric that exceeded threshold
                      - 'opinions': Opinion matrix subset for these agents
            agent_positions: Full position matrix (N, 2) to calculate true centroid
        
        Returns:
            Event: Newly created event
        """
        # --- L: Location ---
        # Calculate TRUE centroid of agents in cell (not mechanical grid center)
        # This preserves the "epicenter" micro-offset within the grid
        agents_in_cell = cell_data['agents']
        if len(agents_in_cell) > 0:
            # Physical center of mass of the crowd
            loc = np.mean(agent_positions[agents_in_cell], axis=0)
        else:
            # Fallback: use grid center if no agents (shouldn't happen)
            loc = np.array(trigger_location)
        
        # --- I: Intensity ---
        # Intensity is proportional to how much the threshold was exceeded
        int_conf = self.attr_config.get('intensity', {})
        base_value = int_conf.get('base_value', 5.0)
        scale_factor = int_conf.get('scale_factor', 5.0)
        
        metric_value = cell_data['metric_value']
        threshold_excess = metric_value - self.threshold
        
        intensity = base_value + (scale_factor * threshold_excess)
        intensity = max(intensity, 1.0)  # Ensure minimum intensity
        
        # --- C: Content ---
        # Derive content from the average opinion of agents in the cell
        cont_conf = self.attr_config.get('content', {})
        
        cell_opinions = cell_data.get('opinions')
        if cell_opinions is not None and len(cell_opinions) > 0:
            # Average opinion vector becomes event content
            content = np.mean(cell_opinions, axis=0)
            
            # Optional: Amplify the dominant dimension
            amplify = cont_conf.get('amplify_dominant', True)
            if amplify:
                # Make the dominant topic even more dominant
                max_idx = np.argmax(content)
                content[max_idx] = min(content[max_idx] * 1.5, 1.0)
                # Renormalize to sum to 1 (Dirichlet-like)
                content = content / np.sum(content)
        else:
            # Fallback: uniform content
            dim = cont_conf.get('topic_dim', 3)
            content = np.ones(dim) / dim
        
        # --- P: Polarity ---
        # High variance in cell -> High polarity (controversial event)
        pol_conf = self.attr_config.get('polarity', {})
        
        if cell_opinions is not None and len(cell_opinions) > 0:
            cell_variance = np.std(cell_opinions)
            # Map variance [0, 0.5] to polarity [-1, 1]
            polarity = (cell_variance / 0.5) * 2.0 - 1.0
            polarity = np.clip(polarity, -1.0, 1.0)
        else:
            # Fallback: neutral polarity
            polarity = 0.0
        
        # Override with config if specified
        if pol_conf.get('type') == 'constant':
            polarity = pol_conf.get('value', 0.0)
        
        # --- Spatial Dynamics (CRITICAL FIX) ---
        # DO NOT randomly sample - derive from actual crowd distribution!
        # This is the dialectic between online (sparse -> wide sigma) 
        # and offline (dense -> local sigma)
        diff_conf = self.attr_config.get('diffusion', {})
        
        # Calculate spatial spread based on agent distribution
        spatial_params = self._calculate_spatial_params_from_crowd(
            agents_in_cell=agents_in_cell,
            agent_positions=agent_positions,
            event_location=loc,
            config=diff_conf
        )
        
        # --- Temporal Dynamics ---
        life_conf = self.attr_config.get('lifecycle', {})
        temporal_params = time_dist.sample_lifecycle_params(
            self.rng,
            life_conf.get('type', 'uniform'),
            life_conf
        )
        
        # --- Assemble Event ---
        return Event(
            uid=self._get_next_id(),
            time=t,
            loc=loc,
            intensity=intensity,
            content=content,
            polarity=polarity,
            spatial_params=spatial_params,
            temporal_params=temporal_params,
            source="endogenous_threshold",
            meta={
                'trigger_metric': self.monitor_attr,
                'metric_value': metric_value,
                'threshold': self.threshold,
                'num_agents_in_cell': len(cell_data['agents'])
            }
        )
    
    def _calculate_spatial_params_from_crowd(self, 
                                            agents_in_cell: List[int],
                                            agent_positions: np.ndarray,
                                            event_location: np.ndarray,
                                            config: Dict) -> Dict[str, float]:
        """
        Calculate spatial diffusion parameters based on ACTUAL crowd distribution,
        not random sampling. This implements the dialectic:
        
        Dense offline crowd -> Small sigma (local event, e.g. protest)
        Sparse online crowd -> Large sigma (viral spread, e.g. hashtag)
        
        Physics:
        1. Calculate spatial variance of the crowd (how spread out they are)
        2. If variance is LOW (tight cluster) -> Event is OFFLINE (small sigma)
        3. If variance is HIGH (distributed) -> Event is ONLINE (large sigma)
        
        Args:
            agents_in_cell: List of agent indices involved
            agent_positions: Full position matrix (N, 2)
            event_location: Event epicenter (centroid)
            config: Configuration parameters for bounds
        
        Returns:
            Dict with 'sigma' key
        """
        if len(agents_in_cell) == 0:
            # Fallback: default parameters
            return {'sigma': config.get('default_sigma', 0.1)}
        
        # Extract positions of involved agents
        crowd_positions = agent_positions[agents_in_cell]
        
        # Calculate spatial variance (how spread out the crowd is)
        # This is the KEY INSIGHT: variance measures "online vs offline"
        distances_from_epicenter = np.linalg.norm(
            crowd_positions - event_location, 
            axis=1
        )
        spatial_variance = np.var(distances_from_epicenter)
        
        # Map variance to sigma using inverse relationship
        # High variance (spread out) -> Large sigma (online, wide impact)
        # Low variance (clustered) -> Small sigma (offline, local impact)
        
        # Get bounds from config
        min_sigma = config.get('min_sigma', 0.03)  # Offline: very local
        max_sigma = config.get('max_sigma', 0.3)   # Online: city-wide
        
        # Calibration: typical variance ranges
        # Tight cluster: variance ~ 0.0001 (radius ~0.01)
        # Spread crowd: variance ~ 0.01 (radius ~0.1)
        var_min = config.get('var_min', 0.0001)
        var_max = config.get('var_max', 0.01)
        
        # Linear mapping: variance -> sigma
        # Clamp to avoid extremes
        variance_ratio = (spatial_variance - var_min) / (var_max - var_min)
        variance_ratio = np.clip(variance_ratio, 0.0, 1.0)
        
        sigma = min_sigma + variance_ratio * (max_sigma - min_sigma)
        
        # Additional factor: crowd size
        # Larger crowds tend to have more "momentum" -> wider spread
        # But diminishing returns (log scale)
        size_factor = config.get('size_factor', 0.05)
        crowd_size = len(agents_in_cell)
        size_bonus = size_factor * np.log1p(crowd_size / 10.0)
        
        sigma = sigma + size_bonus
        sigma = np.clip(sigma, min_sigma, max_sigma)
        
        return {'sigma': float(sigma)}
    
    def reset_cooldowns(self):
        """
        Clear all cooldown timers. Useful when resetting the simulation.
        """
        self.cell_cooldowns.clear()
    
    def get_grid_status(self, current_time: float) -> Dict[str, Any]:
        """
        Get diagnostic information about the grid state.
        
        Args:
            current_time: Current simulation time
        
        Returns:
            Dict with grid statistics
        """
        active_cooldowns = 0
        for cell_idx, last_time in self.cell_cooldowns.items():
            if current_time - last_time < self.cooldown_period:
                active_cooldowns += 1
        
        return {
            'grid_resolution': self.grid_res,
            'total_cells': self.grid_res ** 2,
            'cells_in_cooldown': active_cooldowns,
            'cooldown_period': self.cooldown_period,
            'monitor_attribute': self.monitor_attr,
            'threshold': self.threshold
        }