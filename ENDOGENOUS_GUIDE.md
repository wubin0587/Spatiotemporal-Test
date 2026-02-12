# Endogenous Threshold-Based Event Generator

## Overview

The `EndogenousThresholdGenerator` implements the **"Grey Rhinos"** - events that emerge from the accumulation of internal system states rather than external shocks. This creates a crucial feedback loop: **online sentiment → offline events → increased impact**.

## Core Mechanism

### 1. Spatial Partitioning
```
┌─────┬─────┬─────┬─────┐
│ 0,0 │ 1,0 │ 2,0 │ 3,0 │
├─────┼─────┼─────┼─────┤
│ 0,1 │ 1,1 │ 2,1 │ 3,1 │  Grid Resolution: 4x4
├─────┼─────┼─────┼─────┤
│ 0,2 │ 1,2 │ 2,2 │ 3,2 │  Cell Size: 0.25 x 0.25
├─────┼─────┼─────┼─────┤
│ 0,3 │ 1,3 │ 2,3 │ 3,3 │
└─────┴─────┴─────┴─────┘
```

The map space [0,1] × [0,1] is divided into a regular grid. Each agent is assigned to a cell based on their position.

### 2. Metric Calculation

For each cell with sufficient agents (≥ `min_agents_in_cell`), calculate a monitored metric:

**Opinion Extremism** (default):
```python
metric = mean(|opinions - 0.5|)
```
Measures how far opinions are from the center. High values indicate strong beliefs.

**Opinion Variance**:
```python
metric = std(opinions)
```
Measures polarization within the cell. High values indicate disagreement.

**Density**:
```python
metric = num_agents / cell_area
```
Measures spatial concentration. High values indicate clustering.

### 3. Threshold Triggering

```
IF metric_value ≥ critical_threshold AND cell NOT in cooldown:
    TRIGGER EVENT at cell center
    SET cooldown for this cell
```

### 4. Event Attribute Derivation

Unlike exogenous events (which sample from distributions), endogenous events **inherit properties from the triggering population**:

| Attribute | Derivation |
|-----------|------------|
| **Location (L)** | Cell center coordinates |
| **Intensity (I)** | `base + scale × (metric - threshold)` |
| **Content (C)** | Mean opinion vector of agents in cell |
| **Polarity (P)** | Opinion variance in cell (high variance → controversial) |
| **Diffusion** | Sampled from configured distribution |
| **Lifecycle** | Sampled from configured distribution |

### 5. Cooldown Mechanism

To prevent continuous event generation in the same location:
- Once a cell triggers, it enters cooldown for `N` time steps
- During cooldown, threshold checks are skipped
- After cooldown expires, the cell can trigger again

## Integration with EventManager

### Step Signature
```python
def step(self, current_time: float, 
         agents_state: Dict = None, 
         env_state: Any = None,
         event_history: List[Event] = None) -> List[Event]
```

**Required**: `agents_state` must contain:
```python
{
    'positions': np.ndarray (N, 2),  # Agent locations
    'opinions': np.ndarray (N, L)    # Agent opinions
}
```

**Optional**: `emotions` key for emotion-based monitoring

### EventManager Workflow

```
EventManager.step(t, agents_state)
  │
  ├─ ExogenousShockGenerator.step(t) → [events]
  │
  ├─ EndogenousThresholdGenerator.step(t, agents_state) → [events]
  │     │
  │     ├─ Partition agents to grid
  │     ├─ For each cell:
  │     │   ├─ Check cooldown
  │     │   ├─ Calculate metric
  │     │   ├─ Check threshold
  │     │   └─ Create event if triggered
  │     │
  │     └─ Return new events
  │
  ├─ CascadeGenerator.step(t, event_history) → [events]
  │
  └─ Archive all events
```

## Configuration

### YAML Example

```yaml
events:
  generation:
    endogenous_threshold:
      enabled: true
      seed: 2025
      
      # --- Monitoring Settings ---
      monitor_attribute: 'opinion_extremism'  # or 'opinion_variance', 'density'
      critical_threshold: 0.75                # Metric must exceed this
      
      # --- Spatial Grid ---
      grid_resolution: 20      # 20x20 grid = 400 cells
      min_agents_in_cell: 5    # Minimum population to check
      
      # --- Temporal Control ---
      cooldown: 50             # Time steps before cell can retrigger
      
      # --- Event Attributes ---
      attributes:
        intensity:
          base_value: 10.0
          scale_factor: 5.0    # How much threshold excess amplifies intensity
        
        content:
          topic_dim: 3
          amplify_dominant: true  # Boost the dominant topic
        
        polarity:
          type: 'dynamic'      # Derived from variance
        
        diffusion:
          type: 'log_normal'
          log_mean: -2.3
          log_std: 0.5
        
        lifecycle:
          type: 'bimodal'
          fast_prob: 0.7
          fast_range: [5.0, 15.0]
          slow_range: [30.0, 80.0]
```

## Use Cases

### Scenario A: Political Protest
```yaml
monitor_attribute: 'opinion_extremism'
critical_threshold: 0.8
```
When agents in a region have very extreme opinions (close to 0 or 1), trigger a protest event.

### Scenario B: Flash Mob
```yaml
monitor_attribute: 'density'
critical_threshold: 0.7
```
When agent density in an area is very high, trigger a gathering event.

### Scenario C: Polarization Cascade
```yaml
monitor_attribute: 'opinion_variance'
critical_threshold: 0.6
```
When opinion disagreement within a region is high, trigger a contentious debate event.

## Integration with Simulation Engine

The engine's `EngineInterface` passes agent state to EventManager:

```python
# In steps.py
agents_state = self.interface.prepare_agents_state_dict(
    self.agent_positions,
    self.opinion_matrix
)

new_events = self.interface.fetch_new_events(
    current_time=self.current_time,
    agents_state=agents_state  # <-- This is passed to endogenous generator
)
```

## Expected Behavior

### Example Timeline

```
t=0:   No events (opinions not extreme yet)
t=10:  Algorithm pushes agents into echo chambers
t=20:  Cell (5,7) crosses threshold: extremism=0.82 → EVENT triggered
       → Event creates impact field
       → Agents in radius forced to interact
       → May intensify extremism or create convergence
t=70:  Cooldown expires for cell (5,7)
t=75:  If still extreme, cell can retrigger
```

### Feedback Loop

```
Strong Online Opinions
       ↓
Endogenous Event Generated
       ↓
Impact Field Created
       ↓
Algorithm Mode Disrupted
       ↓
Cross-Cluster Interactions
       ↓
Either: Consensus OR Further Polarization
       ↓
(Loop continues)
```

## Performance Considerations

- **Grid Resolution**: Higher resolution (e.g., 50×50) = more cells to check, but finer spatial granularity
- **Metric Calculation**: O(N) per cell with agents, vectorized using NumPy
- **Cooldown Storage**: Dictionary lookup is O(1)

Recommended: `grid_resolution ≤ 30` for N ≥ 1000 agents

## Diagnostic Tools

```python
# Get grid status
status = generator.get_grid_status(current_time)
# Returns:
# {
#     'grid_resolution': 20,
#     'total_cells': 400,
#     'cells_in_cooldown': 5,
#     'cooldown_period': 50,
#     'monitor_attribute': 'opinion_extremism',
#     'threshold': 0.75
# }

# Reset cooldowns (for new experiments)
generator.reset_cooldowns()
```

## Troubleshooting

### No Events Generated

**Problem**: Generator runs but never triggers events.

**Solutions**:
1. Lower `critical_threshold` (try 0.4 instead of 0.8)
2. Decrease `min_agents_in_cell` (try 3 instead of 10)
3. Check agent distribution - are they clustered or too spread out?
4. Verify `agents_state` contains required keys

### Too Many Events

**Problem**: Events trigger every step.

**Solutions**:
1. Increase `critical_threshold`
2. Increase `cooldown` period
3. Increase `min_agents_in_cell`
4. Check if opinions are being artificially polarized

### Events Only in One Location

**Problem**: Same cell triggers repeatedly.

**Solutions**:
1. Verify cooldown is working (check `cells_in_cooldown`)
2. Check agent position distribution
3. Try different `monitor_attribute`

## Comparison with Other Generators

| Generator | Trigger | Dependency | Example |
|-----------|---------|------------|---------|
| **Exogenous** | Random/Scheduled | Time only | Earthquake, Policy change |
| **Endogenous (Threshold)** | State-based | Agent attributes | Protest, Rally |
| **Cascade** | History-based | Past events | Aftershock, Rumor spread |

## References

See README.md sections:
- "外部事件生成 (Event Generation)" → Exogenous vs Endogenous distinction
- "一、 事件的生成机制 (Generation Mechanisms)" → Theoretical foundation
- "仿真主循环 (Simulation Loop)" → How events integrate with dynamics
