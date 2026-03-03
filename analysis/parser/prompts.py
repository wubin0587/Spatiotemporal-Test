"""
analysis/parser/prompts.py

Prompt Templates for AI-Powered Simulation Analysis
----------------------------------------------------
All prompts are written in English (instructions to the LLM).
The LLM is instructed to respond in the target language and format.

Supported output languages : "zh" (Chinese), "en" (English)
Supported output formats   : "md" (Markdown), "html" (HTML), "latex" (LaTeX)
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import json


# ═══════════════════════════════════════════
# Language / Format Directives
# ═══════════════════════════════════════════

_LANG_DIRECTIVE: Dict[str, str] = {
    "zh": "Please write your entire response in Simplified Chinese (简体中文).",
    "en": "Please write your entire response in English.",
}

_FORMAT_DIRECTIVE: Dict[str, str] = {
    "md": (
        "Format your response as clean Markdown. "
        "Use ## for section headers, **bold** for key terms, "
        "bullet lists for enumerations, and code blocks only for numeric tables."
    ),
    "html": (
        "Format your response as a self-contained HTML fragment (no <html>/<body> wrapper). "
        "Use <h2> for section headers, <strong> for key terms, <ul>/<li> for lists, "
        "and <table class='metrics-table'> for data tables. "
        "Do not include inline CSS or <style> tags — classes only."
    ),
    "latex": (
        "Format your response as a LaTeX fragment (no \\documentclass preamble). "
        "Use \\section{} for section headers, \\textbf{} for key terms, "
        "itemize environments for lists, and tabular environments for data tables. "
        "Escape special characters properly."
    ),
}

_CONCISENESS_DIRECTIVE = (
    "Be concise and analytical. Avoid filler phrases. "
    "Prioritize insight over description. "
    "Every sentence should deliver new information."
)


def _directives(lang: str, fmt: str) -> str:
    lang_dir = _LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE["en"])
    fmt_dir  = _FORMAT_DIRECTIVE.get(fmt,  _FORMAT_DIRECTIVE["md"])
    return f"{lang_dir}\n{fmt_dir}\n{_CONCISENESS_DIRECTIVE}"


# ═══════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert analyst specializing in agent-based opinion dynamics simulations.
Your role is to interpret quantitative simulation metrics and produce clear,
insightful, publication-quality analysis narratives.

You understand the following metric families:
- Opinion metrics: polarization, bimodality, entropy, homophily
- Spatial metrics: centroid drift, radius of gyration, Moran's I (spatial autocorrelation)
- Topological metrics: network density, clustering, modularity, degree distribution
- Event metrics: burstiness, temporal Gini coefficient, event intensity
- Trend metrics: slope (direction of change), evolution delta, final stability, volatility

When values are provided, always:
1. Interpret the number in its theoretical context (e.g., BC > 0.555 suggests bimodality).
2. Compare early vs late simulation state when trend data is available.
3. Identify the most important dynamics, not just describe all metrics.
4. Flag any anomalies or surprising patterns.
"""


# ═══════════════════════════════════════════
# Section-level Prompt Builders
# ═══════════════════════════════════════════

def opinion_analysis_prompt(
    opinion_summary: Dict[str, Any],
    lang: str = "en",
    fmt: str = "md",
) -> str:
    directives = _directives(lang, fmt)
    return f"""\
{directives}

## Task: Analyze Opinion Dynamics

Summary statistics of opinion metrics over the full simulation run.
Each metric dict contains: mean, std, min, max, trend_slope, evolution_delta,
start_mean, end_mean, final_stability, volatility.

### Opinion Metric Summary
```json
{_fmt_dict(opinion_summary)}
```

### Analysis Instructions
1. **Polarization**: Interpret polarization_std and bimodality_coefficient together.
   BC > 0.555 suggests bimodality. High std + high BC = genuine polarization.
2. **Convergence**: Use final_stability (low = converged) and trend_slope (near 0 = stable).
3. **Extremism**: Analyze extreme_share trend — growing or shrinking?
4. **Entropy**: High opinion_entropy = diverse opinions; low = consensus or polarization clusters.
5. **Overall Narrative**: Classify the opinion regime (convergence / polarization / fragmentation).

Write a structured analysis with a clear section for each point above.
"""


def spatial_analysis_prompt(
    spatial_summary: Dict[str, Any],
    lang: str = "en",
    fmt: str = "md",
) -> str:
    directives = _directives(lang, fmt)
    return f"""\
{directives}

## Task: Analyze Spatial Distribution Dynamics

### Spatial Metric Summary
```json
{_fmt_dict(spatial_summary)}
```

### Analysis Instructions
1. **Clustering**: nearest_neighbor_index < 1.0 = clustering; > 1.0 = dispersion.
2. **Centroid Drift**: centroid_x/y trend_slope reveals directional movement of the population.
3. **Spatial Autocorrelation**: moran_i > 0 means similar opinions cluster spatially (echo chambers).
   Magnitude: 0.0–0.2 weak, 0.2–0.5 moderate, >0.5 strong.
4. **Spread**: radius_of_gyration trend — concentrating or dispersing?
5. **Overall Narrative**: Characterize the spatial regime (random / clustered / dispersed).
"""


def topology_analysis_prompt(
    topo_summary: Dict[str, Any],
    lang: str = "en",
    fmt: str = "md",
) -> str:
    directives = _directives(lang, fmt)
    return f"""\
{directives}

## Task: Analyze Network Topology Dynamics

### Topology Metric Summary
```json
{_fmt_dict(topo_summary)}
```

### Analysis Instructions
1. **Connectivity**: largest_component_ratio near 1.0 = well-connected; low = fragmented.
2. **Community Structure**: modularity > 0.3 suggests meaningful community structure.
   Combine with average_clustering for local cohesion.
3. **Degree Distribution**: degree_gini high = hub-dominated; degree_assortativity > 0 = assortative.
4. **Efficiency**: global_efficiency and average_shortest_path_lcc measure information flow speed.
5. **Overall Narrative**: Classify the network regime (random / scale-free / community-structured).
"""


def event_analysis_prompt(
    event_summary: Dict[str, Any],
    lang: str = "en",
    fmt: str = "md",
) -> str:
    directives = _directives(lang, fmt)
    return f"""\
{directives}

## Task: Analyze Event Stream Dynamics

### Event Metric Summary
```json
{_fmt_dict(event_summary)}
```

### Analysis Instructions
1. **Burstiness**: burstiness_index: -1 (regular) to +1 (bursty). >0.3 = significantly bursty.
2. **Temporal Concentration**: temporal_gini near 1.0 = events concentrated in short bursts.
3. **Intensity Patterns**: intensity_mean/std/max — how variable is event magnitude?
4. **Spatial Spread**: event_spatial_spread indicates geographic concentration.
5. **Overall Narrative**: Classify event regime (regular / bursty / cascade-prone).
"""


def trend_summary_prompt(
    feature_summary: Dict[str, Any],
    lang: str = "en",
    fmt: str = "md",
    simulation_meta: Optional[Dict[str, Any]] = None,
) -> str:
    directives = _directives(lang, fmt)
    meta_str = _fmt_dict(simulation_meta) if simulation_meta else "Not provided."
    return f"""\
{directives}

## Task: Write Executive Summary of Simulation Results

### Simulation Metadata
```json
{meta_str}
```

### Full Feature Summary (all metric families)
```json
{_fmt_dict(feature_summary)}
```

### Instructions
Write a concise executive summary (3–5 paragraphs) covering:
1. **Overall Dynamics**: What opinion regime emerged? (consensus / polarization / chaos)
2. **Key Drivers**: Which forces drove the outcome? (events, network, spatial structure)
3. **Critical Transitions**: Were there notable phase shifts?
4. **Final State**: Converged, volatile, or fragmented?
5. **Notable Anomalies**: Surprising or counterintuitive patterns.

Tone: analytical, precise, objective. This will appear at the top of a scientific report.
"""


def comparative_prompt(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    label_a: str = "Run A",
    label_b: str = "Run B",
    lang: str = "en",
    fmt: str = "md",
) -> str:
    directives = _directives(lang, fmt)
    return f"""\
{directives}

## Task: Compare Two Simulation Runs

### {label_a} — Feature Summary
```json
{_fmt_dict(summary_a)}
```

### {label_b} — Feature Summary
```json
{_fmt_dict(summary_b)}
```

### Instructions
1. Identify the 3–5 most significant differences between the two runs.
2. For each difference, explain the likely mechanistic cause.
3. Identify metrics that are similar between runs (robust features).
4. Conclude which run produced stronger polarization / convergence / event coupling.

Use a comparison table (if format allows) followed by narrative analysis.
"""


def narrative_section_prompt(
    section_name: str,
    metrics: Dict[str, Any],
    context: str = "",
    lang: str = "en",
    fmt: str = "md",
) -> str:
    """Generic fallback prompt for custom/user-defined sections."""
    directives = _directives(lang, fmt)
    ctx = f"\n### Domain Context\n{context}\n" if context else ""
    return f"""\
{directives}

## Task: Analyze — {section_name}
{ctx}
### Metrics
```json
{_fmt_dict(metrics)}
```

Provide a concise, insightful analysis. Highlight the most important patterns,
trends, and implications.
"""


# ═══════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════

def _fmt_dict(d: Any, indent: int = 2) -> str:
    """JSON-serialize a dict for prompt embedding, handling numpy types."""
    class _SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                import numpy as np
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return round(float(obj), 6)
                if isinstance(obj, np.ndarray): return obj.tolist()
            except ImportError:
                pass
            if isinstance(obj, float): return round(obj, 6)
            return str(obj)
    return json.dumps(d, indent=indent, cls=_SafeEncoder, ensure_ascii=False)


# ═══════════════════════════════════════════
# Prompt Registry
# ═══════════════════════════════════════════

#: Maps section key -> prompt builder. Used by client.py for dispatch.
SECTION_PROMPT_REGISTRY: Dict[str, Any] = {
    "opinion":       opinion_analysis_prompt,
    "spatial":       spatial_analysis_prompt,
    "topo":          topology_analysis_prompt,
    "event":         event_analysis_prompt,
    "trend_summary": trend_summary_prompt,
    "comparative":   comparative_prompt,
}