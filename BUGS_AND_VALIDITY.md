# Data Structure & Bug Report

## Bugs Found in Existing Code

### 1. `models/engine/facade.py` — Missing `import copy`
**File:** `models/engine/facade.py`, method `get_config()`
```python
# Line ~320 — ILLEGAL: uses copy.deepcopy but never imports copy
def get_config(self) -> Dict[str, Any]:
    return copy.deepcopy(self.config)   # NameError at runtime
```
**Fix:**
```python
import copy  # add at top of file
```

---

### 2. `models/spatial/layer.py` — Wrong import name
**File:** `models/spatial/layer.py`, line 11
```python
from .distribution import create_spatial_distribution   # ILLEGAL: file does not exist
```
The actual file is `models/spatial/distributions.py` (plural).
**Fix:**
```python
from .distributions import create_spatial_distribution
```

---

### 3. `analysis/metrics/spatial.py` — `moran_i` weight matrix constraint undocumented
**File:** `analysis/metrics/spatial.py`, function `moran_i`

The function requires `weights` to be **square (N×N)** and the same length as `values`.
No caller in the existing codebase builds this matrix. Our `extractor.py`
handles this by calling `_build_weight_matrix(graph, n)`.

---

## Data Structure Validity Summary

| Structure | Shape | Dtype | Range | Status |
|-----------|-------|-------|-------|--------|
| `opinion_matrix` | `(N, L)` | float32 | [0, 1] | ✅ Legal — produced & clipped in `steps.py` |
| `agent_positions` | `(N, 2)` | float32 | [0, 1] | ✅ Legal — validated by `EngineInterface` |
| `impact_vector` | `(N,)` | float32 | ≥ 0 | ✅ Legal — Gaussian kernels always non-negative |
| `network_graph` nodes | integers `[0, N-1]` | — | — | ✅ Legal — `builder.py` calls `convert_node_labels_to_integers` |
| `event_times` | `(M,)` | float32 | — | ✅ Legal (M=0 possible; all event metrics guard) |
| `event_locs` | `(M, 2)` | float32 | — | ✅ Legal |
| `moran_i` weight matrix | `(N, N)` | float64 | — | ⚠️ Caller responsibility — built from graph in `extractor.py` |
| `modularity partition` | `List[List[int]]` | — | non-overlapping | ✅ Enforced by `greedy_modularity_communities` |
| `history['opinions']` | `List[ndarray (N,L)]` | float32 | [0,1] | ✅ Legal — appended as `.copy()` in `steps.py` |
| `history['impact']` | `List[ndarray (N,)]` | float32 | ≥ 0 | ✅ Legal |
| `EventVectorArchive` vectors | `(M,2),(M,),(M,),(M,L),(M,)` | float32 | — | ✅ Legal |

## Feature Module — Input/Output Contracts

### `extractor.extract_all_features(snapshot, graph, ...)`
```
IN:  snapshot['opinions']  (N, L)  float  [0,1]
     snapshot['positions'] (N, 2)  float  [0,1]
     snapshot['impact']    (N,)    float  >=0
     graph                 nx.Graph nodes in [0, N-1]
OUT: dict with keys: meta, opinion, spatial, topo, network_opinion, event
     meta['data_issues']: list of structural warnings (empty == all valid)
```

### `composer.compose_timeseries(step_features)`
```
IN:  List[Dict]  length T   (each dict from extract_all_features)
OUT: Dict[str, np.ndarray(T,)]  — NaN for missing keys at any step
```

### `pipeline.FeaturePipeline(engine).run()`
```
IN:  StepExecutor with .opinion_matrix, .agent_positions, .impact_vector,
     .network_graph, .event_manager, .history, .current_time, .time_step
OUT: { 'final': dict, 'timeseries': dict, 'summary': dict, 'data_issues': list }
```
