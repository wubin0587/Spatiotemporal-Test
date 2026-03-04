"""
analysis/intervention/prompts.py

AI Prompt Templates for Intervention Analysis
----------------------------------------------
Builds LLM prompts for interpreting intervention effect metrics and comparison
results.  Follows the same conventions as analysis/parser/prompts.py:
    - Composable sections
    - Language/format directives
    - JSON-embedded metric data

Output languages : "zh" | "en"
Output formats   : "md" | "html" | "latex"
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ═════════════════════════════════════════════════════════════════════════════
# Shared directives (mirrored from analysis/parser/prompts.py)
# ═════════════════════════════════════════════════════════════════════════════

_LANG: Dict[str, str] = {
    "zh": "Please write your entire response in Simplified Chinese (简体中文).",
    "en": "Please write your entire response in English.",
}

_FMT: Dict[str, str] = {
    "md": (
        "Format as clean Markdown. Use ## for section headers, "
        "**bold** for key terms, bullet lists for enumerations."
    ),
    "html": (
        "Format as a self-contained HTML fragment (no <html>/<body>). "
        "Use <h2>, <strong>, <ul>/<li>, <table class='metrics-table'>."
    ),
    "latex": (
        "Format as a LaTeX fragment (no \\documentclass). "
        "Use \\section{}, \\textbf{}, itemize, tabular."
    ),
}

_CONCISE = (
    "Be concise and analytical. Prioritise insight over description. "
    "Every sentence should deliver new information."
)

_SYSTEM_PROMPT = """\
You are an expert analyst specialising in agent-based opinion dynamics simulations
and causal inference for policy evaluation.

Your role is to interpret intervention effect metrics and produce clear,
publication-quality analysis narratives that:
1. Explain WHAT changed as a result of the intervention.
2. Explain WHY it likely changed (mechanism interpretation).
3. Quantify HOW MUCH it changed (magnitude + direction).
4. Assess whether the change is meaningful or noise.

Key metric interpretation guidelines:
- effect_score > 0:       net convergence / moderation effect
- effect_score < 0:       net polarization / destabilising effect
- polarization_reduction > 0:  intervention reduced opinion variance
- bimodality_reduction > 0:    two-camp structure weakened
- extreme_share_change < 0:    fewer radical agents (beneficial for consensus)
- convergence_speed_ratio > 1: treatment converged faster than control
- moran_i_change < 0:          spatial echo-chambers weakened
- modularity_change > 0:       community structure strengthened (echo-chamber risk)
- Cohen's d: < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large
"""


def _directives(lang: str, fmt: str) -> str:
    return f"{_LANG.get(lang, _LANG['en'])}\n{_FMT.get(fmt, _FMT['md'])}\n{_CONCISE}"


def _fmt_json(d: Any) -> str:
    class _E(json.JSONEncoder):
        def default(self, o):
            try:
                import numpy as np
                if isinstance(o, np.integer):  return int(o)
                if isinstance(o, np.floating): return round(float(o), 5)
                if isinstance(o, np.ndarray):  return o.tolist()
            except ImportError:
                pass
            return str(o)
    return json.dumps(d, indent=2, cls=_E, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Single intervention effect prompt
# ═════════════════════════════════════════════════════════════════════════════

def intervention_effect_prompt(
    effect_metrics: Dict[str, float],
    intervention_label: str = "",
    simulation_meta: Optional[Dict[str, Any]] = None,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Prompt for interpreting the effect metrics of a single intervention.
    """
    directives = _directives(lang, fmt)
    meta_str = _fmt_json(simulation_meta) if simulation_meta else "Not provided."
    label_str = f" ({intervention_label})" if intervention_label else ""

    return f"""\
{directives}

## Task: Analyse Intervention Effect{label_str}

### Simulation / Intervention Metadata
```json
{meta_str}
```

### Effect Metrics (treatment minus control)
```json
{_fmt_json(effect_metrics)}
```

### Analysis Instructions
1. **Overall assessment**: Interpret ``effect_score`` — was the intervention net-beneficial,
   harmful, or negligible?
2. **Opinion regime change**: Describe polarization_reduction, bimodality_reduction,
   and extreme_share_change together.  Did the intervention move agents toward consensus
   or deeper polarization?
3. **Spatial dynamics**: Comment on moran_i_change and spatial_clustering_change —
   did the intervention break or reinforce geographic echo-chambers?
4. **Network effects**: Interpret homophily_change, modularity_change,
   edge_disagreement_change.  Did the intervention create bridging ties or deepen silos?
5. **Convergence dynamics**: Use convergence_speed_ratio and opinion_velocity_delta
   to assess whether the intervention accelerated or delayed opinion stabilisation.
6. **Conclusion**: A 2–3 sentence verdict on the intervention's effectiveness.

Write a structured analysis using the above sections.
"""


# ═════════════════════════════════════════════════════════════════════════════
# 2. Multi-arm comparison prompt
# ═════════════════════════════════════════════════════════════════════════════

def multi_arm_comparison_prompt(
    ranking: List[Dict[str, Any]],
    metric_leaders: Dict[str, str],
    simulation_meta: Optional[Dict[str, Any]] = None,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Prompt for comparing multiple intervention policies against a shared control.
    """
    directives = _directives(lang, fmt)
    meta_str = _fmt_json(simulation_meta) if simulation_meta else "Not provided."

    return f"""\
{directives}

## Task: Compare Multiple Intervention Policies

### Simulation Metadata
```json
{meta_str}
```

### Policy Rankings (sorted by effect_score, descending)
```json
{_fmt_json(ranking)}
```

### Per-Metric Leaders
```json
{_fmt_json(metric_leaders)}
```

### Analysis Instructions
1. **Best policy**: Identify the top-ranked policy and explain why its mechanism
   was most effective.
2. **Worst policy**: Identify the worst-ranked policy and diagnose its failure.
3. **Key differentiators**: What metric differences best explain the ranking?
   (Highlight 2–3 decisive metrics.)
4. **Trade-offs**: Are there policies that excel in one dimension (e.g. fast convergence)
   but fail in another (e.g. increased extremism)?
5. **Recommendation**: Which policy should be preferred, under what conditions?
   Note any caveats or sensitivities.
"""


# ═════════════════════════════════════════════════════════════════════════════
# 3. Attribution report prompt
# ═════════════════════════════════════════════════════════════════════════════

def attribution_report_prompt(
    attributed_log: List[Dict[str, Any]],
    overall_effect: Dict[str, float],
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Prompt for a step-by-step attribution report across multiple intervention firings.
    """
    directives = _directives(lang, fmt)

    return f"""\
{directives}

## Task: Intervention Attribution Report

### Overall Intervention Effect (final state)
```json
{_fmt_json(overall_effect)}
```

### Per-Firing Attribution Log
```json
{_fmt_json(attributed_log)}
```

### Analysis Instructions
For each firing in the log:
1. **Timing**: When did this firing occur, and was the simulation in a stable or
   transitional state at that moment?
2. **Immediate effect**: What changed in the 'attribution' delta immediately
   following this firing?
3. **Cumulative contribution**: How much of the overall_effect can be traced to
   this specific firing?

Then provide a synthesis:
4. **Dominant firing**: Which individual firing had the largest measurable impact?
5. **Diminishing returns**: Did later firings have less effect than earlier ones?
6. **Overall attribution verdict**: Was the final outcome primarily caused by
   the timing, frequency, or magnitude of individual firings?
"""


# ═════════════════════════════════════════════════════════════════════════════
# 4. Counterfactual analysis prompt
# ═════════════════════════════════════════════════════════════════════════════

def counterfactual_prompt(
    causal_deltas: Dict[str, float],
    simple_deltas: Dict[str, float],
    intervention_label: str = "",
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Prompt for interpreting difference-in-differences (DiD) causal estimates.
    """
    directives = _directives(lang, fmt)
    label_str = f" — {intervention_label}" if intervention_label else ""

    return f"""\
{directives}

## Task: Counterfactual Causal Analysis{label_str}

### Raw (Pre → Post) Deltas
```json
{_fmt_json(simple_deltas)}
```

### Causal (Difference-in-Differences) Estimates
```json
{_fmt_json(causal_deltas)}
```

### Analysis Instructions
1. **Counterfactual baseline**: Compare the raw delta to the causal delta.
   How much of the observed change would have happened even WITHOUT the intervention?
2. **Net causal effect**: Which metrics show the largest positive causal effect?
   Which show negative causal effects (intervention made things worse)?
3. **Confound assessment**: Are any large raw deltas fully explained by natural drift
   (i.e., causal delta ≈ 0)?  Name these explicitly.
4. **Conclusion**: One paragraph summarising the true causal impact of the intervention,
   distinguishing it from natural system evolution.
"""


# ═════════════════════════════════════════════════════════════════════════════
# 5. Replicate-run robustness prompt
# ═════════════════════════════════════════════════════════════════════════════

def replicate_robustness_prompt(
    replicate_comparison: Dict[str, Any],
    n_control: int,
    n_treatment: int,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Prompt for assessing the robustness of intervention effects across replicate runs.
    """
    directives = _directives(lang, fmt)

    return f"""\
{directives}

## Task: Intervention Effect Robustness (N={n_control} control, N={n_treatment} treatment)

### Per-Metric Statistical Comparison
```json
{_fmt_json(replicate_comparison)}
```

### Analysis Instructions
1. **Robust effects**: Identify metrics where Cohen's d ≥ 0.5 AND the sign is
   consistent.  These are reliable intervention effects.
2. **Fragile effects**: Identify metrics where |Cohen's d| < 0.2 or where
   std_treatment >> std_control.  These results should not be over-interpreted.
3. **Reproducibility assessment**: Overall, how reproducible is this intervention's
   effect across random seeds?
4. **Recommended reporting**: Which 3 metrics provide the strongest evidence
   for the intervention's effect?  Provide specific numerical citations.
"""


# ═════════════════════════════════════════════════════════════════════════════
# 6. Executive summary for intervention report
# ═════════════════════════════════════════════════════════════════════════════

def intervention_executive_summary_prompt(
    effect_metrics: Dict[str, float],
    intervention_label: str = "",
    policy_type: str = "",
    execution_log_summary: Optional[Dict[str, Any]] = None,
    simulation_meta: Optional[Dict[str, Any]] = None,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Executive summary prompt for a complete intervention analysis report.
    """
    directives = _directives(lang, fmt)
    meta_str   = _fmt_json(simulation_meta)  if simulation_meta   else "Not provided."
    log_str    = _fmt_json(execution_log_summary) if execution_log_summary else "Not provided."
    label_str  = f"**Intervention**: {intervention_label}\n" if intervention_label else ""
    policy_str = f"**Policy Type**: {policy_type}\n"        if policy_type        else ""

    return f"""\
{directives}

## Task: Write Executive Summary — Intervention Analysis Report

{label_str}{policy_str}

### Simulation Metadata
```json
{meta_str}
```

### Intervention Execution Summary
```json
{log_str}
```

### Consolidated Effect Metrics
```json
{_fmt_json(effect_metrics)}
```

### Instructions
Write a 3–5 paragraph executive summary covering:
1. **What was done**: Describe the intervention type and when it was applied.
2. **What changed**: Summarise the most significant effects with numbers.
3. **Causal mechanism**: Explain WHY the intervention had the observed effect.
4. **Limitations**: Note any caveats (short simulation horizon, stochastic variance, etc.).
5. **Actionable conclusion**: What does this mean for policy design?

Tone: analytical, precise, suitable for a research report or policy brief.
"""


# ═════════════════════════════════════════════════════════════════════════════
# Registry
# ═════════════════════════════════════════════════════════════════════════════

INTERVENTION_PROMPT_REGISTRY: Dict[str, Any] = {
    "effect":              intervention_effect_prompt,
    "comparison":          multi_arm_comparison_prompt,
    "attribution":         attribution_report_prompt,
    "counterfactual":      counterfactual_prompt,
    "robustness":          replicate_robustness_prompt,
    "executive_summary":   intervention_executive_summary_prompt,
}

INTERVENTION_SYSTEM_PROMPT = _SYSTEM_PROMPT
