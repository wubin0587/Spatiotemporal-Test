# -*- coding: utf-8 -*-
"""
@File    : cascade.py
@Desc    : Endogenous Cascading Event Generator (Hawkes Process)

This module simulates self-exciting event cascades, where the occurrence of one
event increases the probability of subsequent events occurring nearby in space
and time. It models phenomena like aftershocks, viral news cycles, or 
chain reactions.

Key Mechanism (Particle-Based Approach):
1. Each past event is a "particle" with decaying influence
2. Temporal decay: lambda(t) = mu * exp(-alpha * dt)
3. Spatial diffusion: New events spawn near parents with exponential kernel
4. Attribute inheritance: Content and polarity mutate slightly from parent
5. Avoids grid-based O(N^2) by filtering active parents first

This implements the "Chain Reaction" feedback described in README.md.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Import base classes
from ..base import Event
from ..generator import EventGenerator

# Import distribution utilities
from .dist import spatial as spatial_dist
from .dist import time as time_dist


class CascadeGenerator(EventGenerator):
    """
    Generates events based on a self-exciting Hawkes process, where past
    events trigger future ones in a spatiotemporal cascade.
    
    Unlike grid-based approaches, this uses a PARTICLE model:
    - Each event is a "particle" with decaying influence over time
    - New events spawn probabilistically near "hot" particles
    - Attributes (C, P) inherit from parent with mutation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cascade generator.
        
        Expected config structure:
        {
            'enabled': True,
            'seed': 2026,
            'time_decay_alpha': 0.5,      # Temporal decay rate
            'space_decay_beta': 10.0,     # Spatial decay rate (higher = more local)
            'mu_multiplier': 0.8,         # How much one event triggers others
            'background_lambda': 0.001,   # Base triggering rate (ambient)
            'temporal_window': 50.0,      # Only consider events within this window
            'max_spawn_distance': 0.2,    # Maximum spatial distance for children
            'content_mutation': 0.1,      # How much content drifts from parent
            'polarity_mutation': 0.2,     # How much polarity drifts
            'attributes': {
                'intensity': {...},       # For child event intensity
                'diffusion': {...},
                'lifecycle': {...}
            }
        }
        """
        super().__init__(config)
        
        # Extract Hawkes process parameters
        self.alpha = config.get('time_decay_alpha', 0.5)
        self.beta = config.get('space_decay_beta', 10.0)
        self.mu = config.get('mu_multiplier', 0.8)
        self.background = config.get('background_lambda', 0.001)
        self.temporal_window = config.get('temporal_window', 50.0)
        self.max_spawn_dist = config.get('max_spawn_distance', 0.2)
        
        # Mutation parameters for attribute inheritance
        self.content_mutation = config.get('content_mutation', 0.1)
        self.polarity_mutation = config.get('polarity_mutation', 0.2)
        
        # Event attribute config
        self.attr_config = config.get('attributes', {})
        
        self.logger = None
        try:
            import logging
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"CascadeGenerator initialized: "
                           f"alpha={self.alpha}, beta={self.beta}, mu={self.mu}")
        except:
            pass
    
    def step(self, current_time: float, 
             agents_state: Any = None, 
             env_state: Any = None, 
             event_history: List[Event] = None) -> List[Event]:
        """
        Calculate current cascade intensity from historical events and
        trigger new child events stochastically.
        
        Workflow:
        1. Filter event_history to get "active parents" (within temporal window)
        2. For each active parent, calculate its current trigger intensity
        3. Perform Poisson/Bernoulli trial to decide if it spawns a child
        4. If triggered, spawn child event near parent location
        5. Inherit/mutate attributes from parent
        
        Args:
            current_time: Current simulation time.
            agents_state: Optional. Can be used to modulate spawning.
            env_state: Optional.
            event_history: REQUIRED. List of all past events.
            
        Returns:
            List[Event]: Newly triggered cascade events.
        """
        if event_history is None or len(event_history) == 0:
            # No history -> no cascades (only background noise)
            # Could optionally generate background events here
            return []
        
        # Step 1: Temporal Filtering (avoid O(N) every step)
        active_parents = self._filter_active_parents(event_history, current_time)
        
        if len(active_parents) == 0:
            return []
        
        # Step 2: Stochastic Triggering (Particle-based Hawkes)
        triggered_events = []
        
        for parent in active_parents:
            # Calculate time since parent event
            dt = current_time - parent.time
            
            # Hawkes temporal kernel: lambda(t) = mu * exp(-alpha * dt)
            trigger_intensity = self.mu * np.exp(-self.alpha * dt)
            
            # Add background rate (ambient triggering)
            trigger_intensity += self.background
            
            # Perform Bernoulli trial (could also use Poisson for dt > 1)
            # Probability of triggering in this time step
            trigger_prob = min(trigger_intensity, 1.0)
            
            if self.rng.random() < trigger_prob:
                # SPAWN CHILD EVENT!
                child_event = self._spawn_child_event(
                    parent=parent,
                    current_time=current_time,
                    agents_state=agents_state
                )
                
                if child_event is not None:
                    triggered_events.append(child_event)
                    
                    if self.logger:
                        self.logger.debug(f"Cascade event spawned from parent {parent.uid} "
                                        f"at t={current_time:.1f}")
        
        return triggered_events
    
    def _filter_active_parents(self, event_history: List[Event], 
                               current_time: float) -> List[Event]:
        """
        Filter events that are still "active" (can trigger cascades).
        
        An event is active if: current_time - event.time <= temporal_window
        
        Args:
            event_history: Full event history
            current_time: Current time
        
        Returns:
            List[Event]: Events that can still trigger children
        """
        active = []
        
        for event in event_history:
            dt = current_time - event.time
            
            # Only consider recent events (within temporal window)
            if 0 < dt <= self.temporal_window:
                active.append(event)
        
        return active
    
    def _spawn_child_event(self, parent: Event, 
                          current_time: float,
                          agents_state: Optional[Dict] = None) -> Optional[Event]:
        """
        Create a child event near the parent location with inherited attributes.
        
        Spatial Diffusion:
        - Sample offset from parent location using exponential/Gaussian kernel
        - Controlled by beta parameter (higher = more local)
        
        Attribute Inheritance:
        - Content (C): Parent content + Gaussian noise (mutation)
        - Polarity (P): Parent polarity + noise
        - Intensity (I): Decayed from parent or sampled fresh
        
        Args:
            parent: The parent event that triggers this child
            current_time: Current time
            agents_state: Optional agent data for context
        
        Returns:
            Event: The newly spawned child event, or None if spawn failed
        """
        # --- L: Location (Spatial Diffusion) ---
        # Sample offset from parent location using spatial kernel
        # Use exponential decay: p(r) ~ exp(-beta * r)
        
        # Generate random direction
        angle = self.rng.uniform(0, 2 * np.pi)
        
        # Generate distance using inverse CDF of exponential
        # For exp(-beta*r): CDF^-1(u) = -ln(1-u)/beta
        u = self.rng.uniform(0, 1)
        distance = -np.log(1 - u) / self.beta
        
        # Clamp to max spawn distance (prevent events flying off map)
        distance = min(distance, self.max_spawn_dist)
        
        # Calculate child location
        offset_x = distance * np.cos(angle)
        offset_y = distance * np.sin(angle)
        
        child_loc = parent.loc + np.array([offset_x, offset_y])
        
        # Ensure within map bounds [0, 1]
        child_loc = np.clip(child_loc, 0.0, 1.0)
        
        # --- C: Content (Inheritance + Mutation) ---
        # Start with parent's content
        parent_content = parent.content
        
        # Add Gaussian noise (semantic drift)
        noise = self.rng.normal(0, self.content_mutation, size=len(parent_content))
        child_content = parent_content + noise
        
        # Renormalize to valid probability simplex (sum to 1, all positive)
        child_content = np.abs(child_content)
        child_content = child_content / np.sum(child_content)
        
        # --- P: Polarity (Inheritance + Mutation) ---
        parent_polarity = parent.polarity
        
        # Add noise with larger variance (polarization can drift faster)
        polarity_noise = self.rng.normal(0, self.polarity_mutation)
        child_polarity = parent_polarity + polarity_noise
        
        # Clamp to valid range [-1, 1]
        child_polarity = np.clip(child_polarity, -1.0, 1.0)
        
        # --- I: Intensity ---
        # Option 1: Decay from parent (cascades weaken)
        # Option 2: Sample fresh (cascades maintain strength)
        # Here we use Option 1 with config override
        
        int_conf = self.attr_config.get('intensity', {})
        decay_factor = int_conf.get('cascade_decay', 0.7)
        
        child_intensity = parent.intensity * decay_factor
        
        # Add some randomness
        intensity_noise = self.rng.normal(0, 1.0)
        child_intensity += intensity_noise
        child_intensity = max(child_intensity, 1.0)  # Minimum threshold
        
        # --- Spatial Parameters ---
        # Inherit parent's spatial params OR calculate from local context
        # For now, we inherit with slight mutation
        
        diff_conf = self.attr_config.get('diffusion', {})
        
        if diff_conf.get('inherit_from_parent', True):
            # Inherit parent's sigma with small perturbation
            parent_sigma = parent.spatial_params.get('sigma', 0.1)
            mutation_factor = diff_conf.get('spatial_mutation', 0.1)
            
            sigma_noise = self.rng.normal(0, mutation_factor)
            child_sigma = parent_sigma + sigma_noise
            child_sigma = np.clip(child_sigma, 0.03, 0.5)
            
            spatial_params = {'sigma': float(child_sigma)}
        else:
            # Sample fresh
            spatial_params = spatial_dist.sample_diffusion_params(
                self.rng,
                diff_conf.get('type', 'uniform'),
                diff_conf
            )
        
        # --- Temporal Parameters ---
        life_conf = self.attr_config.get('lifecycle', {})
        temporal_params = time_dist.sample_lifecycle_params(
            self.rng,
            life_conf.get('type', 'uniform'),
            life_conf
        )
        
        # --- Assemble Child Event ---
        child = Event(
            uid=self._get_next_id(),
            time=current_time,
            loc=child_loc,
            intensity=child_intensity,
            content=child_content,
            polarity=child_polarity,
            spatial_params=spatial_params,
            temporal_params=temporal_params,
            source="cascade",
            meta={
                'parent_uid': parent.uid,
                'parent_source': parent.source,
                'generation': parent.meta.get('generation', 0) + 1,
                'spawn_distance': float(distance)
            }
        )
        
        return child