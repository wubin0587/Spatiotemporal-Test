"""
D:\Tiktok\models\events\generate\exp.py

Exogenous Shock Generator (The "Black Swans")
-------------------------------------------------
This module implements the generator for external events that occur independently 
of the system's internal state (e.g., breaking news, natural disasters, announcements).

It orchestrates the calls to the distribution utilities in the `dist` package 
to determine the timing, location, and characteristics of new events.

Configuration Guide (config.yaml):
-------------------------------------------------------------------------
events:
  generation:
    # --- Configuration for this ExogenousShockGenerator ---
    exogenous:
      enabled: True
      seed: 2024 # For reproducibility

      # 1. TRIGGER TIMING: When do events happen? (Uses dist.time)
      # This section is passed to dist.time.calculate_trigger_probability
      time_trigger:
        type: poisson      # Options: poisson, normal, linear, cyclic, burst
        lambda_rate: 0.05  # For 'poisson'

      # 2. EVENT ATTRIBUTES: What do the events look like?
      # This section is used to sample L, I, C, P and their dynamics.
      attributes:
        # L - Location (Uses dist.spatial.sample_location)
        location:
          type: satellite  # Options: uniform, gaussian, hotspots, satellite, etc.
          # ... (parameters for the chosen location distribution) ...
          main_city: { center: [0.5, 0.5], proportion: 0.7, std_dev: 0.1 }
          satellites:
            - { center: [0.1, 0.8], proportion: 0.15, std_dev: 0.05 }
            - { center: [0.9, 0.2], proportion: 0.15, std_dev: 0.05 }

        # I - Intensity (Power-law distribution is standard for social phenomena)
        intensity:
          type: pareto     # Options: pareto, uniform
          shape: 2.5       # Pareto 'alpha'. Lower = heavier tail (more mega-events).
          min_val: 1.0     # Minimum possible event intensity.

        # C - Content (Dirichlet is standard for topic mixtures)
        content:
          topic_dim: 3     # Must match number of network layers.
          # Dirichlet concentration. [1,1,1]=uniform, [0.1,0.1,0.1]=sparse/specialized.
          concentration: [1.0, 1.0, 1.0] 

        # P - Polarity
        polarity:
          type: uniform    # Options: uniform, constant
          min: -1.0
          max: 1.0

        # Dynamics: Spatial Diffusion (Uses dist.spatial.sample_diffusion_params)
        diffusion:
          type: log_normal # Options: constant, uniform, log_normal
          log_mean: -2.3   # Corresponds to a median sigma of ~0.1
          log_std: 0.5

        # Dynamics: Temporal Lifecycle (Uses dist.time.sample_lifecycle_params)
        lifecycle:
          type: bimodal    # Options: constant, uniform, bimodal (fast_vs_slow)
          fast_prob: 0.8
          fast_range: [1.0, 5.0]
          slow_range: [20.0, 50.0]
-------------------------------------------------------------------------
"""

from typing import List, Dict, Any

# Import base classes and distribution functions
from ..base import Event
from ..generator import EventGenerator
from .dist import time as time_dist
from .dist import spatial as spatial_dist

class ExogenousShockGenerator(EventGenerator):
    """
    Generates events based on external, pre-configured stochastic processes,
    independent of agent states.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the generator by parsing the configuration.
        """
        super().__init__(config)
        
        # Isolate the relevant config sections for clarity
        self.time_trigger_config = self.config.get('time_trigger', {})
        self.attributes_config = self.config.get('attributes', {})

    def step(self, current_time: float, agents_state: Any = None, env_state: Any = None) -> List[Event]:
        """
        At each simulation step, check if an event should be triggered.
        If the trigger condition is met, a new Event object is created and returned.
        
        Args:
            current_time: The current global simulation time.
            agents_state: Ignored by this generator.
            env_state: Ignored by this generator.
            
        Returns:
            A list containing a single new event, or an empty list.
        """
        
        # 1. Calculate the probability of an event triggering AT THIS MOMENT
        # This logic is delegated to the time distribution module.
        trigger_prob = time_dist.calculate_trigger_probability(
            t=current_time,
            dist_type=self.time_trigger_config.get('type', 'poisson'),
            config=self.time_trigger_config
        )
        
        # 2. Perform a Bernoulli trial to decide if the event happens
        if self.rng.random() < trigger_prob:
            # If successful, create the full event object
            new_event = self._create_event(current_time)
            return [new_event]
        
        return []

    def _create_event(self, t: float) -> Event:
        """
        Internal helper to assemble a new Event object by sampling from
        all the configured attribute distributions.
        """
        attr_conf = self.attributes_config

        # --- Sample L (Location) and Spatial Dynamics ---
        loc_conf = attr_conf.get('location', {})
        loc = spatial_dist.sample_location(self.rng, loc_conf)

        diff_conf = attr_conf.get('diffusion', {})
        s_params = spatial_dist.sample_diffusion_params(self.rng, diff_conf)

        # --- Sample Temporal Dynamics ---
        life_conf = attr_conf.get('lifecycle', {})
        t_params = time_dist.sample_lifecycle_params(self.rng, life_conf)

        # --- Sample I (Intensity) ---
        int_conf = attr_conf.get('intensity', {})
        if int_conf.get('type', 'pareto') == 'pareto':
            shape = int_conf.get('shape', 2.5)
            min_val = int_conf.get('min_val', 1.0)
            # Standard formula to scale Pareto from numpy
            intensity = (self.rng.pareto(shape) + 1) * min_val
        else: # Fallback to uniform
            intensity = self.rng.uniform(int_conf.get('min', 1.0), int_conf.get('max', 10.0))

        # --- Sample C (Content) ---
        cont_conf = attr_conf.get('content', {})
        dim = cont_conf.get('topic_dim', 3)
        alpha = cont_conf.get('concentration', [1.0] * dim)
        content = self.rng.dirichlet(alpha)
        
        # --- Sample P (Polarity) ---
        pol_conf = attr_conf.get('polarity', {})
        if pol_conf.get('type', 'uniform') == 'uniform':
            polarity = self.rng.uniform(pol_conf.get('min', -1.0), pol_conf.get('max', 1.0))
        else: # Fallback to constant
            polarity = pol_conf.get('value', 0.0)

        # --- Assemble and return the Event object ---
        return Event(
            uid=self._get_next_id(),
            time=t,
            loc=loc,
            intensity=intensity,
            content=content,
            polarity=polarity,
            spatial_params=s_params,
            temporal_params=t_params,
            source="exogenous"
        )