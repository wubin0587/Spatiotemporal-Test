# Event-Modulated Spatiotemporal Opinion Dynamics Model

**Author:** minun · SUFE · minunplus312@gmail.com

---

## Overview

This framework simulates **online sentiment → offline event feedback loops** using an agent-based opinion dynamics model. Agents hold multidimensional opinions, interact over social networks, move spatially, and both generate and respond to events. The system is designed for research into polarization, radicalization, protest dynamics, and information cascade phenomena.

**Core feedback loop:**
```
Agent opinions → Endogenous event generation → Event impact field → Opinion updates → (repeat)
```

---

## Architecture

The codebase is organized into five main subsystems:

### 1. Simulation Engine (`models/engine/`)

| File | Role |
|------|------|
| `core.py` | Abstract base class `SimulationEngine` — defines interface contract |
| `steps.py` | Concrete `StepExecutor` — implements all simulation logic |
| `facade.py` | `SimulationFacade` — the **only** public entry point for external code |

**Engine state per step:**
- `opinion_matrix` — shape `(N, L)`, values ∈ [0, 1], N agents × L opinion layers
- `agent_positions` — shape `(N, 2)`, spatial coordinates ∈ [0, 1]²
- `impact_vector` — shape `(N,)`, current event impact on each agent
- `network_graph` — NetworkX graph for social topology

**Step workflow:**
1. Generate new events (via EventManager)
2. Compute impact field I(t) for all agents
3. Determine interaction pairs (social graph + spatial proximity)
4. Update opinions via bounded confidence + event modulation
5. Increment time

### 2. Event System (`models/events/`)

Three event generator types:

| Generator | Config Key | Trigger Mechanism |
|-----------|------------|-------------------|
| Exogenous | `exogenous` | Poisson process (time-driven) |
| Endogenous Threshold | `endogenous_threshold` | Grid-cell attribute threshold |
| Endogenous Cascade | `endogenous_cascade` | Hawkes process (event-driven) |

**EndogenousThresholdGenerator** monitors one of three spatial attributes:

| Attribute | Formula | High value indicates |
|-----------|---------|----------------------|
| `opinion_extremism` | `mean(|opinions − 0.5|)` | Radicalization |
| `opinion_variance` | `std(opinions)` | Local polarization / conflict |
| `density` | `num_agents / cell_area` | Spatial gathering |

Each generated event carries: `location`, `intensity`, `content` (topic vector), `polarity`, `diffusion` (spatial spread σ), and `lifecycle` (duration).

### 3. Intervention System (`intervention/`)

Enables controlled experiments and counterfactual analysis.

**Trigger types** (`intervention/trigger.py`):

| Type | Config `type` | Fires when |
|------|--------------|-----------|
| `StepTrigger` | `step` | `time_step == step` |
| `TimeTrigger` | `time` | `current_time >= threshold` |
| `PolarizationTrigger` | `polarization` | `opinion_std > threshold` |
| `ImpactTrigger` | `impact` | `mean_impact > threshold` |
| `CompositeTrigger` | `composite` | AND / OR combination of above |

**Policy types** (via `BasePolicy.from_config()`):

| Policy | Config `type` | Effect |
|--------|--------------|--------|
| Opinion nudge | `opinion_nudge` | Shift opinions by delta |
| Opinion clamp | `opinion_clamp` | Hard-bound opinion range |
| Network rewire | `network_rewire` | Randomly rewire fraction of edges |
| Event suppress | `event_suppress` | Disable event source for duration |
| Dynamics param | `dynamics_param` | Override ε, μ at runtime |
| Simulation speed | `simulation_speed` | Change dt |

All triggers support `max_fires` (0 = unlimited) and `cooldown` parameters. Setting `auto_checkpoint: true` on a rule creates a `BranchManager` snapshot before each policy application, enabling counterfactual comparison.

### 4. Analysis Pipeline (`analysis/`)

Invoked via `run_analysis(engine, config)` → returns `AnalysisResult`.

**Pipeline stages:**

| Stage | Config key | Output |
|-------|-----------|--------|
| Feature extraction | `feature.enabled` | Timeseries + summary statistics |
| AI narrative parser | `parser.enabled` | Per-section LLM text (requires API key) |
| Report generation | `report.enabled` | `.md` / `.html` / `.tex` report |
| Visualization | `visual.enabled` | Static matplotlib figures (Agg) |

**Available figures:** `dashboard`, `opinion_distribution`, `spatial_opinions`, `opinion_timeseries`, `impact_heatmap`, `event_timeline`, `polarization_evolution`, `network_homophily`

**AI parser modes:** `chronicle`, `diagnostic`, `comparative`, `predictive`, `dramatic`

**AI parser themes:** `concert_crowd`, `political_rally`, and others (auto-detected if `theme: null`)

### 5. Configuration Schema

The config must follow strict file-aligned naming. Top-level required keys: `engine`, `events`, `networks`, `spatial`.

```yaml
engine:
  interface:
    agents:
      num_agents: 200
      opinion_layers: 3
      initial_opinions:
        type: polarized          # uniform | polarized | random
        params: {split: 0.5}
    simulation:
      total_steps: 500
      seed: 42
      record_history: true
  maths:
    dynamics:
      epsilon_base: 0.25         # bounded confidence threshold
      mu_base: 0.35              # opinion update rate
      alpha_mod: 0.25            # event amplification
      beta_mod: 0.15
      backfire: false
    field:
      alpha: 6.0                 # impact decay rate
      beta: 0.08
      temporal_window: 100.0
    topo:
      threshold: 0.3
      radius_base: 0.06
      radius_dynamic: 0.15

networks:
  builder:
    layers:
      - name: social
        type: small_world        # small_world | barabasi_albert | erdos_renyi
        params: {n: 200, k: 6, p: 0.1}

spatial:
  distribution:
    type: clustered              # clustered | uniform | gaussian
    n_clusters: 4
    cluster_std: 0.1

events:
  generation:
    exogenous:
      enabled: true
      seed: 43
      time_trigger: {type: poisson, lambda_rate: 0.25}
      attributes:
        location: {type: uniform}
        intensity: {type: pareto, shape: 2.5, min_val: 4.0}
        content: {topic_dim: 3, concentration: [1, 1, 1]}
        polarity: {type: uniform, min: -0.5, max: 0.5}
        diffusion: {type: log_normal, log_mean: -2.0, log_std: 0.5}
        lifecycle:
          type: bimodal
          fast_prob: 0.9
          fast_range: [2, 5]
          slow_range: [10, 20]

    endogenous_threshold:
      enabled: true
      seed: 44
      monitor_attribute: opinion_extremism   # or opinion_variance | density
      critical_threshold: 0.6
      grid_resolution: 15                    # NxN spatial grid
      min_agents_in_cell: 5
      cooldown: 30
      attributes:
        intensity: {base_value: 10.0, scale_factor: 15.0}
        content: {topic_dim: 3, amplify_dominant: true}
        polarity: {type: dynamic}
        diffusion: {type: log_normal, log_mean: -2.3, log_std: 0.5}
        lifecycle:
          type: bimodal
          fast_prob: 0.8
          fast_range: [3.0, 10.0]
          slow_range: [20.0, 60.0]

    endogenous_cascade:
      enabled: true
      seed: 45
      background_lambda: 0.0
      mu_multiplier: 0.6
      attributes:
        intensity: {cascade_decay: 0.5}
        diffusion: {inherit_from_parent: true, spatial_mutation: 0.04}
        lifecycle: {type: uniform, min_sigma: 2.0, max_sigma: 5.0}
```

---

## Quick Start

### Minimal Runnable Example

```python
from models.engine.facade import SimulationFacade
import numpy as np

config = { ... }  # see configuration schema above

sim = SimulationFacade.from_config_dict(config)
results = sim.run(num_steps=500)

print(f"Steps: {results['total_steps']}")
print(f"Final polarization: {np.std(results['final_opinions']):.4f}")

sim.save_results('output.npz')
sim.save_event_log('events.json')
```

### From YAML File

```python
sim = SimulationFacade.from_config_file('config.yaml')
results = sim.run()
```

### Step-by-Step Execution

```python
sim = SimulationFacade.from_config_dict(config)
sim.initialize()

for i in range(200):
    stats = sim.step()
    # stats keys: step, time, num_events, num_new_events, max_impact, mean_impact
    print(f"t={stats['time']:.2f}  events={stats['num_events']}  impact={stats['mean_impact']:.3f}")
```

---

## Running Experiments with Interventions

```python
from models.engine.facade import SimulationFacade
from intervention.manager import InterventionManager
from intervention.trigger import PolarizationTrigger
from intervention.policies.base import BasePolicy

sim = SimulationFacade.from_config_dict(config)
sim.initialize()

mgr = InterventionManager()
mgr.add_rule(
    trigger=PolarizationTrigger(threshold=0.35, cooldown=20, max_fires=3),
    policy=BasePolicy.from_config({'type': 'opinion_nudge', 'layer': -1,
                                   'delta': 0.05, 'direction': 'center'}),
    label="deradicalization",
    auto_checkpoint=True,      # saves counterfactual snapshot before each intervention
)

sim.set_intervention_manager(mgr)
results = sim.run(num_steps=500)

print(mgr.get_execution_log())             # when interventions fired
bm = mgr.branch_manager                    # access saved checkpoints
checkpoints = bm.list_checkpoints()
```

### From YAML Config

```python
mgr = InterventionManager.from_config({
    "interventions": [
        {
            "label": "rewire_at_100",
            "auto_checkpoint": True,
            "trigger": {"type": "step", "step": 100, "max_fires": 1},
            "policy": {"type": "network_rewire", "fraction": 0.1, "seed": 99}
        }
    ]
}, sim)
```

---

## Analysis

```python
from analysis.manager import run_analysis

analysis_config = {
    "output": {
        "dir": "output/run_001",
        "lang": "zh",           # zh | en
        "save_figures": True,
        "save_timeseries": True,
        "save_features_json": True,
    },
    "feature": {"enabled": True, "layer_idx": 0, "include_trends": True},
    "parser":  {
        "enabled": True,        # requires api_key
        "api_key": "sk-...",    # or set OPENAI_API_KEY env var
        "model": "gpt-4o",
        "lang": "zh",
        "narrative_mode": "diagnostic",    # chronicle | diagnostic | comparative | predictive | dramatic
        "theme": None,                     # auto-detect
        "sections": ["opinion", "spatial", "topo", "event"],
        "include_executive_summary": True,
    },
    "report":  {"enabled": True, "formats": ["md", "html"], "include_toc": True},
    "visual":  {
        "enabled": True,
        "dashboard": True,
        "opinion_distribution": True,
        "spatial_opinions": True,
        "opinion_timeseries": True,
        "event_timeline": True,
        "polarization_evolution": True,
        "dpi": 150,
    },
    "simulation_meta": {"n_agents": 200, "n_steps": 500}
}

result = run_analysis(sim._engine, analysis_config)

print(result.report_paths)     # {'md': '/output/run_001/report.md', ...}
print(result.figure_paths)     # {'dashboard': '...', 'spatial_opinions': '...', ...}
print(result.feature_paths)    # {'timeseries_npz': '...', 'summary_json': '...', ...}
print(result.errors)           # non-fatal errors
```

---

## Research Scenario Templates

### A — Radicalization Study

Monitor opinion extremism; events trigger when opinions cluster near 0 or 1.

```yaml
endogenous_threshold:
  monitor_attribute: opinion_extremism
  critical_threshold: 0.8
  grid_resolution: 20
  min_agents_in_cell: 10
  cooldown: 50
  attributes:
    intensity: {base_value: 15.0, scale_factor: 20.0}
    polarity: {type: dynamic}
```

### B — Polarization / Conflict Study

Monitor intra-cell opinion variance; events trigger in ideologically mixed regions.

```yaml
endogenous_threshold:
  monitor_attribute: opinion_variance
  critical_threshold: 0.5
  cooldown: 40
  attributes:
    intensity: {base_value: 12.0, scale_factor: 10.0}
    content: {amplify_dominant: false}
```

### C — Protest / Gathering Study

Monitor agent density; events trigger at spatial concentrations.

```yaml
endogenous_threshold:
  monitor_attribute: density
  critical_threshold: 0.6
  grid_resolution: 25
  cooldown: 20
  attributes:
    intensity: {base_value: 5.0, scale_factor: 8.0}
    lifecycle:
      type: bimodal
      fast_prob: 0.9
      fast_range: [1.0, 5.0]
```

---

## Diagnostics

### No events generating

```yaml
critical_threshold: 0.4          # lower from 0.8
min_agents_in_cell: 3            # lower minimum crowd
```
Also check that initial opinions create variation: use `type: polarized` not `type: uniform`.

### Too many events

```yaml
critical_threshold: 0.75
cooldown: 60
min_agents_in_cell: 12
```

### Events only in one location

Increase `grid_resolution` (more cells → more spatial diversity), or switch spatial distribution to `uniform`.

### Verify event composition after a run

```python
events = sim._engine.event_manager.archive.get_all_events()
from collections import Counter
print(Counter(e.source for e in events))
# Counter({'exogenous': 87, 'endogenous_threshold': 23, 'cascade': 11})
```

---

## Testing

Run the comprehensive test suite (18 test groups, ~60 individual assertions):

```bash
python test.py
```

Test output is written to `output/test_run/`. A JSON report is saved at `output/test_run/test_report.json`.

**Test coverage:**

| Range | Area |
|-------|------|
| T01–T02 | Config validation, engine initialization |
| T03–T04 | Step execution, full simulation run |
| T05 | Intervention hook path |
| T06 | All trigger types |
| T07–T09 | InterventionManager lifecycle, from_config, BranchManager |
| T10 | All policy types |
| T11–T13 | History recording, state save/load, event log |
| T14–T15 | Analysis manager (feature, report, visualization) |
| T16–T18 | Reset/re-run, edge cases, step consistency |

Enable debug logging for detailed output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Output Files

| File | Format | Contents |
|------|--------|----------|
| `output.npz` | NumPy archive | Final opinions, positions, impact, optional history |
| `events.json` | JSON | Full event log with sources, times, locations |
| `checkpoint_*.npz` | NumPy archive | Engine state snapshots for resume / counterfactuals |
| `features_summary.json` | JSON | Aggregated simulation statistics |
| `features_final.json` | JSON | Final-timestep feature snapshot |
| `timeseries.npz` | NumPy archive | All tracked metrics over time |
| `report.md` / `report.html` | Text / HTML | Analysis report (with or without AI narrative) |
| `figures/*.png` | PNG | Static visualization figures |

---

## Performance Notes

- **Grid resolution** 10–15 is fast; 25–30 is fine-grained but slower
- **record_history: true** increases memory proportionally to `total_steps × N × L`
- **Cascade generator** scales with number of active events; set `mu_multiplier < 1` to prevent supercritical branching
- For large runs (N > 1000, steps > 2000), consider disabling `network_homophily` and `impact_heatmap` visualizations

---

## File Reference

| Path | Description |
|------|-------------|
| `models/engine/facade.py` | `SimulationFacade` — public API |
| `models/engine/steps.py` | `StepExecutor` — simulation logic |
| `models/engine/core.py` | `SimulationEngine` — abstract interface |
| `models/events/generate/imp.py` | Endogenous threshold generator source |
| `intervention/manager.py` | `InterventionManager` |
| `intervention/trigger.py` | All trigger types + `from_config` factory |
| `intervention/policies/base.py` | `BasePolicy` + `from_config` factory |
| `intervention/branch/checkpoint.py` | `BranchManager` for checkpoints |
| `analysis/manager.py` | `run_analysis()` entry point |
| `analysis/feature/pipeline.py` | `FeaturePipeline` |
| `analysis/parser/client.py` | `ParserClient` (LLM narrative) |
| `analysis/report/builder.py` | `ReportBuilder` |
| `analysis/visual/static.py` | All matplotlib figure functions |
| `test.py` | Comprehensive integration test suite |
| `ENDOGENOUS_GUIDE.md` | Detailed endogenous generator documentation |
| `INTEGRATION_SUMMARY.md` | System-wide integration overview |

---

*minun · SUFE · minunplus312@gmail.com*
