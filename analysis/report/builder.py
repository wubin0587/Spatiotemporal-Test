"""
analysis/report/builder.py

Report Builder — Top-Level Entry Point
---------------------------------------
Single function interface to assemble a complete simulation report.

Three usage modes:
    1. AI mode       — sections from ParserClient output (AI-generated narrative)
    2. Static mode   — sections auto-rendered from feature data (no LLM needed)
    3. Hybrid mode   — AI sections where available, static fallback for the rest

Quick start
-----------
    from analysis.report.builder import build_report

    # Mode 1: Full AI
    report = build_report(
        pipeline_output=features,
        parser_sections=client.parse_pipeline_result(features),
        fmt="md", lang="zh",
    )
    report.save("output/report.md")

    # Mode 2: Static only
    report = build_report(pipeline_output=features, fmt="html", lang="en")
    report.save("output/report.html")

    # Mode 3: Hybrid (AI sections + static fallback)
    report = build_report(
        pipeline_output=features,
        parser_sections=partial_sections,   # only some sections present
        fmt="md", lang="zh",
        fallback_to_static=True,
    )

Data Contract
-------------
pipeline_output : dict
    Output of FeaturePipeline.run() / run_feature_pipeline().
    Required keys: "final", "summary", "data_issues"

parser_sections : dict[str, str], optional
    Output of ParserClient.parse_pipeline_result() or any subset.
    Keys: "opinion" | "spatial" | "topo" | "event" | "executive_summary" | ...
    Values: rendered text strings (Markdown / HTML / LaTeX).

Returns
-------
ReportDocument — call .render() for string, .save(path) to write file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from analysis.constants import DEFAULT_LANGUAGE
from .language import get_label, SECTION_ORDER
from .renderer import ReportDocument, render_document


# ═════════════════════════════════════════════════════════════════════════════
# Primary Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def build_report(
    pipeline_output: Dict[str, Any],
    parser_sections: Optional[Dict[str, str]] = None,
    fmt: str = "md",
    lang: str = DEFAULT_LANGUAGE,
    title: Optional[str] = None,
    fallback_to_static: bool = True,
    include_metrics_snapshot: bool = True,
    simulation_meta: Optional[Dict[str, Any]] = None,
) -> ReportDocument:
    """
    Assemble a simulation analysis report.

    Parameters
    ----------
    pipeline_output : dict
        Feature pipeline output (from run_feature_pipeline or FeaturePipeline.run()).
        Must contain "final", "summary", "data_issues".
    parser_sections : dict[str, str], optional
        AI-generated section texts from ParserClient.
        If None and fallback_to_static=True, static rendering is used for all sections.
        If partial, static rendering fills missing sections (when fallback_to_static=True).
    fmt : "md" | "html" | "latex"
        Output format.
    lang : "zh" | "en"
        Output language (affects labels, headings, fallback text).
    title : str, optional
        Report title. Defaults to localised "Simulation Analysis Report".
    fallback_to_static : bool
        If True, sections absent from parser_sections are rendered statically.
        If False, only sections present in parser_sections are included.
    include_metrics_snapshot : bool
        If True, append a raw metrics JSON block to the report footer.
    simulation_meta : dict, optional
        Extra metadata shown in the report header (e.g. n_agents, n_steps, config).

    Returns
    -------
    ReportDocument
        Call .render() → str, or .save(filepath) → str (absolute path).
    """
    title = title or get_label("title", lang)

    # ── Resolve sections ──────────────────────────────────────────────────────
    ai_sections: Dict[str, str] = parser_sections or {}

    if fallback_to_static:
        static_sections = _build_static_sections(pipeline_output, fmt=fmt, lang=lang)
        # Merge: AI takes priority; static fills gaps
        merged: Dict[str, str] = {**static_sections, **ai_sections}
    else:
        merged = dict(ai_sections)

    return render_document(
        sections=merged,
        pipeline_output=pipeline_output,
        fmt=fmt,
        lang=lang,
        title=title,
        include_metrics_snapshot=include_metrics_snapshot,
        simulation_meta=simulation_meta,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Convenience wrappers
# ═════════════════════════════════════════════════════════════════════════════

def build_ai_report(
    pipeline_output: Dict[str, Any],
    parser_sections: Dict[str, str],
    fmt: str = "md",
    lang: str = DEFAULT_LANGUAGE,
    title: Optional[str] = None,
    simulation_meta: Optional[Dict[str, Any]] = None,
) -> ReportDocument:
    """
    Pure AI report — only sections from parser_sections are included.
    No static fallback. Useful when you want full control over content.
    """
    return build_report(
        pipeline_output=pipeline_output,
        parser_sections=parser_sections,
        fmt=fmt, lang=lang, title=title,
        fallback_to_static=False,
        simulation_meta=simulation_meta,
    )


def build_static_report(
    pipeline_output: Dict[str, Any],
    fmt: str = "md",
    lang: str = DEFAULT_LANGUAGE,
    title: Optional[str] = None,
    simulation_meta: Optional[Dict[str, Any]] = None,
) -> ReportDocument:
    """
    Pure static report — no AI, metric tables only.
    Useful for quick inspection or when no API key is available.
    """
    return build_report(
        pipeline_output=pipeline_output,
        parser_sections=None,
        fmt=fmt, lang=lang, title=title,
        fallback_to_static=True,
        simulation_meta=simulation_meta,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Internal: static section builder (thin adapter over parser.output)
# ═════════════════════════════════════════════════════════════════════════════

def _build_static_sections(
    pipeline_output: Dict[str, Any],
    fmt: str,
    lang: str,
) -> Dict[str, str]:
    """
    Build static (no-AI) section texts from pipeline_output.
    Delegates to analysis.parser.output for metric formatting.
    """
    from analysis.parser.output import build_static_sections
    return build_static_sections(pipeline_output, fmt=fmt, lang=lang)
