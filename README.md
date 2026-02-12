# Quick Start: Using Endogenous Threshold Generator

## 30-Second Overview

The `EndogenousThresholdGenerator` creates events when agent populations in local regions become extreme, polarized, or dense. This enables the simulation to model **online sentiment → offline events** feedback loops.

## Basic Usage

### Step 1: Enable in Configuration

```yaml
events:
  generation:
    endogenous_threshold:
      enabled: true
      seed: 2025
      monitor_attribute: 'opinion_extremism'  # What to monitor
      critical_threshold: 0.7                 # When to trigger
      grid_resolution: 15                     # Map divided into 15x15 grid
      min_agents_in_cell: 5                   # Minimum population
      cooldown: 30                            # Steps before retrigger
      
      attributes:  # Event properties when triggered
        intensity:
          base_value: 10.0
          scale_factor: 15.0
        content:
          topic_dim: 3
          amplify_dominant: true
        polarity:
          type: 'dynamic'  # Derived from local variance
        diffusion:
          type: 'log_normal'
          log_mean: -2.3
          log_std: 0.5
        lifecycle:
          type: 'bimodal'
          fast_prob: 0.8
          fast_range: [3.0, 10.0]
          slow_range: [20.0, 60.0]
```

### Step 2: Run Simulation

```python
from models.engine.facade import SimulationFacade

# Load config and run
sim = SimulationFacade.from_config_file('config.yaml')
results = sim.run(num_steps=500)

# Check event log
sim.save_event_log('events.json')
```

### Step 3: Analyze Results

```python
# Get event history
import json
with open('events.json', 'r') as f:
    data = json.load(f)

# Count endogenous events
endogenous_events = [
    e for e in data['data']['sources'] 
    if e == 'endogenous_threshold'
]
print(f"Generated {len(endogenous_events)} endogenous events")

# Analyze final opinions
final_opinions = results['final_opinions']
polarization = np.std(final_opinions)
print(f"Final polarization: {polarization:.3f}")
```

## Common Scenarios

### Scenario A: Model Political Protests

**Goal**: Generate events when extreme opinions cluster spatially.

```yaml
monitor_attribute: 'opinion_extremism'
critical_threshold: 0.8    # Very extreme
grid_resolution: 20        # Fine-grained regions
min_agents_in_cell: 10     # Significant crowd
cooldown: 50               # Long cooldown (rare events)

attributes:
  intensity:
    base_value: 15.0       # High impact
    scale_factor: 20.0
  polarity:
    type: 'dynamic'        # Controversial if variance high
```

**Expected**: Events trigger in regions with highly extreme, concentrated opinions.

### Scenario B: Model Flash Mobs

**Goal**: Generate events when agent density spikes.

```yaml
monitor_attribute: 'density'
critical_threshold: 0.6    # High concentration
grid_resolution: 25        # Very fine grid
min_agents_in_cell: 8      # Moderate crowd
cooldown: 20               # Short cooldown (frequent gatherings)

attributes:
  intensity:
    base_value: 5.0        # Moderate impact
    scale_factor: 8.0
  lifecycle:
    type: 'bimodal'
    fast_prob: 0.9         # Usually short events
    fast_range: [1.0, 5.0]
```

**Expected**: Events trigger when agents physically cluster.

### Scenario C: Model Opinion Polarization

**Goal**: Generate events when local disagreement is high.

```yaml
monitor_attribute: 'opinion_variance'
critical_threshold: 0.5    # High variance
grid_resolution: 15
min_agents_in_cell: 6
cooldown: 40

attributes:
  intensity:
    base_value: 12.0
    scale_factor: 10.0
  polarity:
    type: 'dynamic'        # Always high for this scenario
  content:
    amplify_dominant: false  # Keep mixed content
```

**Expected**: Events trigger in polarized regions (mix of opposing views).

## Tuning Guide

### Problem: No Events Generated

**Symptoms**: Simulation runs but endogenous events never trigger.

**Solutions**:

1. **Lower threshold**:
   ```yaml
   critical_threshold: 0.4  # Instead of 0.8
   ```

2. **Check initial conditions**:
   ```yaml
   agents:
     initial_opinions:
       type: 'polarized'  # Creates extremism
       params:
         split: 0.5
   ```

3. **Reduce minimum agents**:
   ```yaml
   min_agents_in_cell: 3  # Instead of 10
   ```

4. **Verify spatial clustering**:
   ```yaml
   spatial:
     distribution:
       type: 'clustered'  # Groups agents
       n_clusters: 4
       cluster_std: 0.08
   ```

### Problem: Too Many Events

**Symptoms**: Events trigger constantly, overwhelming the system.

**Solutions**:

1. **Raise threshold**:
   ```yaml
   critical_threshold: 0.75  # Instead of 0.5
   ```

2. **Increase cooldown**:
   ```yaml
   cooldown: 60  # Instead of 20
   ```

3. **Require more agents**:
   ```yaml
   min_agents_in_cell: 12  # Instead of 5
   ```

### Problem: Events Only in One Location

**Symptoms**: Same grid cell triggers repeatedly.

**Solutions**:

1. **Check cooldown is working**:
   ```python
   # In custom code
   status = generator.get_grid_status(current_time)
   print(status['cells_in_cooldown'])
   ```

2. **Increase grid resolution**:
   ```yaml
   grid_resolution: 25  # More cells = more diversity
   ```

3. **Change spatial distribution**:
   ```yaml
   spatial:
     distribution:
       type: 'uniform'  # Spread agents out
   ```

## Monitor Attributes Explained

### `opinion_extremism`
- **Formula**: `mean(|opinions - 0.5|)`
- **Range**: [0, 0.5]
- **Interpretation**: How far from center (0.5)
- **High value means**: Strong beliefs (near 0 or 1)
- **Use for**: Detecting radicalization

### `opinion_variance`
- **Formula**: `std(opinions)`
- **Range**: [0, ~0.5]
- **Interpretation**: Opinion disagreement
- **High value means**: Polarization within cell
- **Use for**: Detecting conflicts

### `density`
- **Formula**: `num_agents / cell_area`
- **Range**: [0, 1] (normalized)
- **Interpretation**: Spatial concentration
- **High value means**: Many agents in small area
- **Use for**: Detecting gatherings

## Integration Verification

### Check 1: Is Generator Active?

```python
from models.events.manager import EventManager

config = {...}  # Your config
manager = EventManager(config)

print(f"Active generators: {len(manager.generators)}")
for gen in manager.generators:
    print(f"  - {gen.__class__.__name__}")

# Should show: EndogenousThresholdGenerator
```

### Check 2: Are Events Being Created?

```python
sim = SimulationFacade.from_config_file('config.yaml')
sim.initialize()

for i in range(20):
    stats = sim.step()
    if stats['num_events'] > 0:
        print(f"Step {i}: {stats['num_events']} events")

# Should see some steps with events > 0
```

### Check 3: Are Events Endogenous?

```python
# After simulation
import json
with open('events.json', 'r') as f:
    data = json.load(f)

sources = data['data']['sources']
endogenous_count = sources.count('endogenous_threshold')
print(f"Endogenous events: {endogenous_count}/{len(sources)}")

# Should show non-zero endogenous count
```

## Performance Tips

1. **Grid Resolution**: Start with 10-15, increase if needed
   - Low (5-10): Fast, coarse regions
   - Medium (15-20): Balanced
   - High (25-30): Slow, fine-grained

2. **Cooldown**: Balance frequency vs realism
   - Short (10-20): Frequent events
   - Medium (30-50): Balanced
   - Long (60-100): Rare events

3. **Monitor Attribute**: Choose based on research question
   - Extremism: Radicalization studies
   - Variance: Polarization studies
   - Density: Gathering/protest studies

## Complete Minimal Example

```python
#!/usr/bin/env python3
"""Minimal example of endogenous event generation."""

from models.engine.facade import SimulationFacade
import numpy as np

# Define config
config = {
    'agents': {
        'num_agents': 200,
        'opinion_layers': 3,
        'initial_opinions': {'type': 'polarized', 'params': {'split': 0.5}}
    },
    'network': {
        'layers': [{'name': 'social', 'type': 'small_world', 
                    'params': {'n': 200, 'k': 8, 'p': 0.1}}]
    },
    'spatial': {
        'distribution': {'type': 'clustered', 'n_clusters': 4, 'cluster_std': 0.1}
    },
    'events': {
        'generation': {
            'endogenous_threshold': {
                'enabled': True,
                'seed': 42,
                'monitor_attribute': 'opinion_extremism',
                'critical_threshold': 0.6,
                'grid_resolution': 12,
                'min_agents_in_cell': 5,
                'cooldown': 25,
                'attributes': {
                    'intensity': {'base_value': 8.0, 'scale_factor': 12.0},
                    'content': {'topic_dim': 3, 'amplify_dominant': True},
                    'polarity': {'type': 'dynamic'},
                    'diffusion': {'type': 'uniform', 'min_sigma': 0.06, 'max_sigma': 0.15},
                    'lifecycle': {'type': 'uniform', 'min_sigma': 8.0, 'max_sigma': 25.0}
                }
            }
        }
    },
    'dynamics': {'epsilon_base': 0.2, 'mu_base': 0.3, 'alpha_mod': 0.2, 'beta_mod': 0.1},
    'field': {'alpha': 5.0, 'beta': 0.1},
    'topology': {'threshold': 0.25, 'radius_base': 0.05, 'radius_dynamic': 0.12},
    'simulation': {'total_steps': 100, 'seed': 42}
}

# Run simulation
sim = SimulationFacade.from_config_dict(config)
results = sim.run()

# Analyze
print(f"Simulation completed: {results['total_steps']} steps")
print(f"Final polarization: {np.std(results['final_opinions']):.3f}")

# Save
sim.save_results('output.npz')
sim.save_event_log('events.json')

print("Done! Check events.json for endogenous events.")
```

Run with:
```bash
python minimal_example.py
```

## Further Reading

- `ENDOGENOUS_GUIDE.md`: Detailed technical documentation
- `INTEGRATION_SUMMARY.md`: System-wide integration overview
- `README.md`: Original model description (Chinese)
- `models/events/generate/imp.py`: Source code with inline comments

## Support

For issues or questions:
1. Check configuration against examples above
2. Run `test_endogenous_integration.py`
3. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
