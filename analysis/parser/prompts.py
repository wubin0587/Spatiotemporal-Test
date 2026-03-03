"""
analysis/parser/prompts.py

Prompt Templates for AI-Powered Simulation Analysis
----------------------------------------------------
Central prompt registry. Provides:

    1. Core analytical prompts (section-level metric interpretation)
    2. Theme-aware prompt builders (via themes.ThemeEngine)
    3. Narrative-mode prompt builders (via narrative.NarrativeBuilder)
    4. Composite builders that combine all three layers

Architecture of a complete prompt
----------------------------------
    ┌──────────────────────────────────────────────────┐
    │  SYSTEM_PROMPT  [+ get_theme_system_addendum()]  │  ← LLM system role
    ├──────────────────────────────────────────────────┤
    │  Theme context block  (themes.py)                │  ← real-world framing
    ├──────────────────────────────────────────────────┤
    │  Narrative mode block (narrative.py)             │  ← story structure
    ├──────────────────────────────────────────────────┤
    │  Section metric data  (this file)                │  ← numbers + analysis hints
    └──────────────────────────────────────────────────┘

Supported output languages : "zh" (Chinese), "en" (English)
Supported output formats   : "md" (Markdown), "html" (HTML), "latex" (LaTeX)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

# ── Shared utilities are re-exported so callers need only import prompts ────
from .themes import (
    Theme,
    ThemeEngine,
    THEME_REGISTRY,
    get_theme_prompt_override,
    get_theme_system_addendum,
)
from .narrative import (
    NarrativeBuilder,
    NarrativeMode,
    NARRATIVE_PROMPT_REGISTRY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Language / Format Directives
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert analyst specialising in agent-based opinion dynamics simulations.
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


def build_system_prompt(theme: Optional[Theme] = None, lang: str = "zh") -> str:
    """
    Build the full system prompt, optionally with a theme addendum.

    Parameters
    ----------
    theme : Theme, optional
        If provided, appends a theme-context reminder to the system prompt.
    lang : str
        Output language for the theme addendum.

    Returns
    -------
    str — complete system prompt.
    """
    if theme is not None:
        return SYSTEM_PROMPT + get_theme_system_addendum(theme, lang=lang)
    return SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# Section-Level Analytical Prompt Builders
# (These focus on metric interpretation, independent of narrative mode or theme.)
# ═══════════════════════════════════════════════════════════════════════════════

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




def multi_run_opinion_prompt(
    mean_summary: Dict[str, Any],
    std_summary: Dict[str, Any],
    ci95_summary: Dict[str, Any],
    n_runs: int,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    directives = _directives(lang, fmt)
    return f"""\
{directives}

## Task: Analyze Opinion Dynamics Across Replicate Runs

### Number of runs
{n_runs}

### Mean Summary
```json
{_fmt_dict(mean_summary)}
```

### Cross-run Std Summary
```json
{_fmt_dict(std_summary)}
```

### 95% Confidence Intervals
```json
{_fmt_dict(ci95_summary)}
```

### Instructions
1. Distinguish robust conclusions (CV < 0.1) from high-variance findings (CV > 0.3).
2. Add uncertainty qualifiers and confidence-interval references for key claims.
3. Explain which patterns are likely deterministic vs stochastic across replicate runs.
4. Conclude with reproducibility assessment and recommended follow-up experiments.
"""


def stability_analysis_prompt(
    cv_summary: Dict[str, Any],
    consensus_scores: Dict[str, Any],
    n_runs: int,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    directives = _directives(lang, fmt)
    return f"""\
{directives}

## Task: Cross-run Stability Analysis

### Number of runs
{n_runs}

### Coefficient of Variation Summary
```json
{_fmt_dict(cv_summary)}
```

### Consensus Scores (1-CV)
```json
{_fmt_dict(consensus_scores)}
```

### Instructions
1. Identify highly stable metrics and explain why they may be parameter-determined.
2. Identify unstable metrics and explain stochastic sensitivity.
3. Evaluate overall repeatability of this experiment setup.
4. Provide practical guidance for how many runs are needed for robust inference.
"""


def parameter_comparison_prompt(
    sweep_results: Dict[str, Dict],
    sweep_stds: Dict[str, Dict],
    focus_metrics: Optional[List[str]] = None,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    directives = _directives(lang, fmt)
    focus_block = _fmt_dict(focus_metrics) if focus_metrics else "Not specified."
    return f"""\
{directives}

## Task: Compare Parameter Sweeps (each entry includes replicate runs)

### Focus Metrics
```json
{focus_block}
```

### Mean Summaries by Configuration
```json
{_fmt_dict(sweep_results)}
```

### Std Summaries by Configuration
```json
{_fmt_dict(sweep_stds)}
```

### Instructions
1. Compare all provided parameter configurations and rank their outcomes on key dimensions.
2. Highlight differences that remain meaningful after considering uncertainty.
3. Separate robust cross-configuration effects from noisy effects.
4. Recommend a preferred parameter regime with evidence-backed reasoning.
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


# ═══════════════════════════════════════════════════════════════════════════════
# Theme-Aware Prompt Builders
# (Wrap the section prompts above with theme context injection.)
# ═══════════════════════════════════════════════════════════════════════════════

def themed_opinion_prompt(
    opinion_summary: Dict[str, Any],
    theme: Theme,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Opinion analysis prompt with real-world theme context injected.

    The theme block is prepended, instructing the LLM to interpret
    all metrics through the lens of the given scenario.
    """
    theme_ctx = get_theme_prompt_override(theme, lang=lang, fmt=fmt)
    base      = opinion_analysis_prompt(opinion_summary, lang=lang, fmt=fmt)
    return f"{theme_ctx}\n---\n{base}"


def themed_spatial_prompt(
    spatial_summary: Dict[str, Any],
    theme: Theme,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    theme_ctx = get_theme_prompt_override(theme, lang=lang, fmt=fmt)
    base      = spatial_analysis_prompt(spatial_summary, lang=lang, fmt=fmt)
    return f"{theme_ctx}\n---\n{base}"


def themed_topology_prompt(
    topo_summary: Dict[str, Any],
    theme: Theme,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    theme_ctx = get_theme_prompt_override(theme, lang=lang, fmt=fmt)
    base      = topology_analysis_prompt(topo_summary, lang=lang, fmt=fmt)
    return f"{theme_ctx}\n---\n{base}"


def themed_event_prompt(
    event_summary: Dict[str, Any],
    theme: Theme,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    theme_ctx = get_theme_prompt_override(theme, lang=lang, fmt=fmt)
    base      = event_analysis_prompt(event_summary, lang=lang, fmt=fmt)
    return f"{theme_ctx}\n---\n{base}"


def themed_executive_summary_prompt(
    feature_summary: Dict[str, Any],
    theme: Theme,
    lang: str = "zh",
    fmt: str = "md",
    simulation_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Executive summary prompt with theme context.
    Uses the NarrativeBuilder in CHRONICLE mode by default for coherence.
    """
    theme_ctx = get_theme_prompt_override(theme, lang=lang, fmt=fmt)
    builder   = NarrativeBuilder(NarrativeMode.CHRONICLE, lang=lang, fmt=fmt, word_count_hint=500)
    return builder.build_executive_narrative(feature_summary, theme_ctx, simulation_meta)


# ═══════════════════════════════════════════════════════════════════════════════
# Narrative-Mode Prompt Builders
# (Wrap section prompts with a specific narrative structure, without themes.)
# ═══════════════════════════════════════════════════════════════════════════════

def narrative_opinion_prompt(
    opinion_summary: Dict[str, Any],
    mode: str = NarrativeMode.CHRONICLE,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """Opinion analysis using a specific narrative mode."""
    return NarrativeBuilder(mode, lang, fmt).build("opinion", opinion_summary)


def narrative_spatial_prompt(
    spatial_summary: Dict[str, Any],
    mode: str = NarrativeMode.CHRONICLE,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    return NarrativeBuilder(mode, lang, fmt).build("spatial", spatial_summary)


def narrative_topo_prompt(
    topo_summary: Dict[str, Any],
    mode: str = NarrativeMode.CHRONICLE,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    return NarrativeBuilder(mode, lang, fmt).build("topo", topo_summary)


def narrative_event_prompt(
    event_summary: Dict[str, Any],
    mode: str = NarrativeMode.CHRONICLE,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    return NarrativeBuilder(mode, lang, fmt).build("event", event_summary)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Composite Prompt Builder
# (Theme + Narrative Mode + Analytical Hints — the richest prompt option.)
# ═══════════════════════════════════════════════════════════════════════════════

def composite_section_prompt(
    section: str,
    metrics: Dict[str, Any],
    theme: Optional[Theme] = None,
    narrative_mode: str = NarrativeMode.CHRONICLE,
    lang: str = "zh",
    fmt: str = "md",
    extra_context: str = "",
) -> str:
    """
    Build the richest possible prompt by combining:
        theme context + narrative mode + section-specific analytical hints.

    This is the recommended builder for production use in ParserClient.

    Parameters
    ----------
    section        : "opinion" | "spatial" | "topo" | "event" | custom
    metrics        : feature summary dict for this section
    theme          : Theme instance from themes.ThemeEngine (optional)
    narrative_mode : NarrativeMode constant (default: CHRONICLE)
    lang           : "zh" | "en"
    fmt            : "md" | "html" | "latex"
    extra_context  : additional free-text context

    Returns
    -------
    str — complete LLM user prompt.
    """
    theme_ctx = (
        get_theme_prompt_override(theme, lang=lang, fmt=fmt)
        if theme is not None
        else ""
    )

    # Append analytical hints from the section-specific analytical prompts
    # as a supplementary guidance block after the narrative instructions.
    analytical_hints = _get_analytical_hints(section, lang)

    builder = NarrativeBuilder(
        mode=narrative_mode,
        lang=lang,
        fmt=fmt,
        word_count_hint=350,
    )

    combined_context = "\n\n".join(
        filter(None, [theme_ctx, analytical_hints, extra_context])
    )

    return builder.build(
        section=section,
        metrics=metrics,
        theme_context=combined_context,
    )


def composite_executive_prompt(
    feature_summary: Dict[str, Any],
    theme: Optional[Theme] = None,
    narrative_mode: str = NarrativeMode.DRAMATIC,
    lang: str = "zh",
    fmt: str = "md",
    simulation_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Composite executive summary prompt.
    Defaults to DRAMATIC narrative mode for maximum impact.
    """
    theme_ctx = (
        get_theme_prompt_override(theme, lang=lang, fmt=fmt)
        if theme is not None
        else ""
    )
    builder = NarrativeBuilder(
        mode=narrative_mode,
        lang=lang,
        fmt=fmt,
        word_count_hint=600,
    )
    return builder.build_executive_narrative(feature_summary, theme_ctx, simulation_meta)


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-detect Theme + Build Composite Prompt
# ═══════════════════════════════════════════════════════════════════════════════

def auto_themed_composite_prompt(
    section: str,
    metrics: Dict[str, Any],
    full_summary: Dict[str, Any],
    narrative_mode: str = NarrativeMode.CHRONICLE,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Convenience: auto-detect theme from full_summary, then build composite prompt.

    Parameters
    ----------
    section      : section to analyse
    metrics      : summary for this section
    full_summary : complete pipeline summary (used for theme detection)
    """
    engine = ThemeEngine()
    theme  = engine.detect(full_summary)
    return composite_section_prompt(
        section=section,
        metrics=metrics,
        theme=theme,
        narrative_mode=narrative_mode,
        lang=lang,
        fmt=fmt,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_dict(d: Any, indent: int = 2) -> str:
    """JSON-serialise a dict for prompt embedding, handling numpy types."""
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


def _get_analytical_hints(section: str, lang: str) -> str:
    """
    Extract the analytical instructions block from each section prompt
    as a supplementary context string for composite prompts.
    Avoids duplicating the full prompt; provides just the interpretation keys.
    """
    _hints_zh: Dict[str, str] = {
        "opinion": (
            "**指标解读参考**\n"
            "- `polarization_std` + `bimodality_coefficient`（>0.555）→ 真实极化\n"
            "- `final_stability` 低 → 已收敛；`trend_slope` ≈0 → 趋于稳定\n"
            "- `extreme_share` 增长趋势 → 极端化加剧\n"
            "- `opinion_entropy` 低 → 共识或极化聚类"
        ),
        "spatial": (
            "**指标解读参考**\n"
            "- `nearest_neighbor_index` <1 → 聚集，>1 → 分散\n"
            "- `moran_i` >0 → 意见空间自相关（回声室）；0–0.2弱，0.2–0.5中，>0.5强\n"
            "- `radius_of_gyration` 下降趋势 → 空间收缩\n"
            "- `centroid` 漂移 → 人群定向移动"
        ),
        "topo": (
            "**指标解读参考**\n"
            "- `largest_component_ratio` ≈1 → 高连通；低值 → 碎片化\n"
            "- `modularity` >0.3 → 显著社区结构\n"
            "- `degree_gini` 高 → 枢纽节点主导\n"
            "- `degree_assortativity` >0 → 同配混合"
        ),
        "event": (
            "**指标解读参考**\n"
            "- `burstiness` -1（规律）→ +1（爆发）；>0.3 = 显著突发\n"
            "- `temporal_gini` ≈1 → 事件高度集中\n"
            "- `intensity_max` 提示最极端冲击强度\n"
            "- `event_rate` 反映持续压力水平"
        ),
    }
    _hints_en: Dict[str, str] = {
        "opinion": (
            "**Interpretation Reference**\n"
            "- `polarization_std` + `bimodality_coefficient` (>0.555) → genuine polarisation\n"
            "- `final_stability` low → converged; `trend_slope` ≈0 → stabilising\n"
            "- `extreme_share` rising trend → increasing radicalisation\n"
            "- `opinion_entropy` low → consensus or polarisation clusters"
        ),
        "spatial": (
            "**Interpretation Reference**\n"
            "- `nearest_neighbor_index` <1 → clustering; >1 → dispersion\n"
            "- `moran_i` >0 → spatial autocorrelation (echo chambers); 0–0.2 weak, 0.2–0.5 moderate, >0.5 strong\n"
            "- `radius_of_gyration` declining → spatial contraction\n"
            "- `centroid` drift → directional population movement"
        ),
        "topo": (
            "**Interpretation Reference**\n"
            "- `largest_component_ratio` ≈1 → highly connected; low → fragmented\n"
            "- `modularity` >0.3 → significant community structure\n"
            "- `degree_gini` high → hub-dominated network\n"
            "- `degree_assortativity` >0 → assortative mixing"
        ),
        "event": (
            "**Interpretation Reference**\n"
            "- `burstiness` -1 (regular) → +1 (bursty); >0.3 = significantly bursty\n"
            "- `temporal_gini` ≈1 → highly concentrated event timing\n"
            "- `intensity_max` indicates peak shock magnitude\n"
            "- `event_rate` reflects sustained pressure level"
        ),
    }
    hints = _hints_zh if lang == "zh" else _hints_en
    return hints.get(section, "")


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Registry  (backward-compatible with client.py)
# ═══════════════════════════════════════════════════════════════════════════════

#: Core section registry — maps section key → analytical prompt builder.
#: Used by ParserClient for basic (no-theme, no-narrative-mode) dispatch.
SECTION_PROMPT_REGISTRY: Dict[str, Any] = {
    "opinion":       opinion_analysis_prompt,
    "spatial":       spatial_analysis_prompt,
    "topo":          topology_analysis_prompt,
    "event":         event_analysis_prompt,
    "trend_summary": trend_summary_prompt,
    "comparative":   comparative_prompt,
}

#: Themed section registry — maps section key → theme-aware prompt builder.
THEMED_PROMPT_REGISTRY: Dict[str, Any] = {
    "opinion":  themed_opinion_prompt,
    "spatial":  themed_spatial_prompt,
    "topo":     themed_topology_prompt,
    "event":    themed_event_prompt,
}

#: Narrative section registry — maps (section, mode) strings → builders.
#: Key format: "section:mode", e.g. "opinion:dramatic"
MULTI_RUN_PROMPT_REGISTRY: Dict[str, Any] = {
    "multi_run_opinion": multi_run_opinion_prompt,
    "stability": stability_analysis_prompt,
    "parameter_comparison": parameter_comparison_prompt,
}

NARRATIVE_SECTION_REGISTRY: Dict[str, Any] = {
    f"{section}:{mode}": (
        lambda s, m: lambda metrics, lang="zh", fmt="md": (
            NarrativeBuilder(m, lang, fmt).build(s, metrics)
        )
    )(section, mode)
    for section in ["opinion", "spatial", "topo", "event"]
    for mode in NarrativeMode.ALL
}
