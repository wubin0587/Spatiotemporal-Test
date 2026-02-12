# -*- coding: utf-8 -*-
"""
@File    : random.py
@Desc    : Randomness management utility.
           Ensures reproducibility by synchronizing seeds across different libraries.
           Provides local random number generators (RNG) for modular use.
"""

import random
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def setup_random_seed(seed: int):
    """
    Set global random seeds for Python and NumPy.
    
    Args:
        seed (int): The seed value to use.
    """
    if not isinstance(seed, int):
        logger.warning(f"Seed {seed} is not an integer. Converting to int.")
        seed = int(seed)

    # 1. Set Python standard library random seed
    random.seed(seed)

    # 2. Set NumPy global seed (for functions like np.random.randn)
    # Note: Modern NumPy recommends using Generator (RNG) instead of the global state.
    np.random.seed(seed)

    logger.info(f"Global random seed set to: {seed}")


def get_rng(config: Optional[Dict[str, Any]] = None, 
            seed_override: Optional[int] = None) -> np.random.Generator:
    """
    Creates and returns a modern NumPy Random Generator (RNG).
    
    This is the preferred way to generate random numbers in modern NumPy.
    Pass this RNG object to math modules or managers to ensure thread-safety
    and independence between different simulation components.
    
    Args:
        config (Dict): The configuration dictionary containing ['simulation']['seed'].
        seed_override (int): Directly provide a seed, bypassing config.
        
    Returns:
        np.random.Generator: A seeded NumPy RNG instance.
    """
    seed = 42  # Default fallback
    
    if seed_override is not None:
        seed = seed_override
    elif config:
        # Navigate through the nested config dict: config['simulation']['seed']
        seed = config.get('simulation', {}).get('seed', 42)

    # Create a new Generator instance
    rng = np.random.default_rng(seed)
    
    return rng


def set_seed_from_config(config: Dict[str, Any]):
    """
    High-level utility to initialize all seeds directly from the config dictionary.
    
    Args:
        config (Dict): The full simulation configuration.
    """
    seed = config.get('simulation', {}).get('seed', 42)
    setup_random_seed(seed)