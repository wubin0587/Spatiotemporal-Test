"""
analysis/report/builder.py

Report Builder
--------------
Orchestrates the construction of a complete simulation analysis report.
Accepts input from two sources (either or both):

    1. AI-parsed sections:   dict[str, str]  from ParserClient
    2. Raw pipeline output:  dict            from FeaturePipeline.run()

If both are provided, AI sections are used where available;
static metric tables are used as fallback.

Usage
-----
    # AI mode (full)
    from analysis.parser.client import ParserClient
    from analysis.report.builder import ReportBuilder

    client = ParserClient(api_key="...", lang="zh", fmt="md")
    sections = client.parse_pipeline_result(pipeline_output)

    builder = ReportBuilder(lang="zh", fmt="md")
    report = builder.build(
        parsed_sections=sections,
        pipeline_output=pipeline_output,
        title="2024 Opinion Dynamics Report",
    )
    report.save("output/report.md")

    # Static mode (no AI)
    report = builder.build(pipeline_output=pipeline_output)
    report.save("output/report.md")

Data Contract
-------------
parsed_sections : dict[str, str]
    Keys: "executive_summary", "opinion", "spatial", "topo", "event",
          "stability", and any custom keys.
    Values: rendered text strings (Markdown / HTML / LaTeX).
    Produced by: ParserClient.parse_pipeline_result()
                 ParserClient.parse_multi_run_result()

pipeline_output : dict
    Keys: "final", "timeseries", "summary", "data_issues"
    Produced by: FeaturePipeline.run() / run_feature_pipeline()
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from analysis.constants import DEFAULT_LANGUAGE
from .outputs.markdown import render_markdown
from .outputs.html import render_html
from .outputs.latex import render_latex
from .printer import ReportPrinter


# ═════════════════════════════════════════════════════════════════════════════
# ReportDocument
# ═════════════════════════════════════════════════════════════════════════════

class ReportDocument:
    """
    Immutable container for a fully-assembled report.

    Attributes
    ----------
    content  : str    The fully rendered document string.
    fmt      : str    "md" | "html" | "latex"
    lang     : str    "zh" | "en"
    title    : str
    metadata : dict   Simulation metadata extracted from pipeline_output.
    sections : dict   Raw section texts before assembly.
    """

    def __init__(
        self,
        content: str,
        fmt: str,
        lang: str,
        title: str,
        metadata: Dict[str, Any],
        sections: Dict[str, str],
    ):
        self.content  = content
        self.fmt      = fmt
        self.lang     = lang
        self.title    = title
        self.metadata = metadata
        self.sections = sections

    def save(self, filepath: str) -> str:
        """
        Save the rendered content to a file.
        Extension is auto-appended if missing.
        Returns the absolute path of the saved file.
        """
        import os
        _ext = {"md": ".md", "html": ".html", "latex": ".tex"}
        ext = _ext.get(self.fmt, f".{self.fmt}")
        if not filepath.endswith(ext):
            filepath = filepath.rstrip(".") + ext

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.content)
        return os.path.abspath(filepath)

    def print_summary(self) -> None:
        """Print a human-readable summary of the report to stdout."""
        printer = ReportPrinter(lang=self.lang)
        printer.print_document(self)

    def __repr__(self) -> str:
        return (
            f"ReportDocument(fmt={self.fmt!r}, lang={self.lang!r}, "
            f"title={self.title!r}, sections={list(self.sections.keys())})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# ReportBuilder
# ═════════════════════════════════════════════════════════════════════════════

class ReportBuilder:
    """
    Builds a ReportDocument from parsed AI sections and/or raw pipeline output.

    Parameters
    ----------
    lang         : str   "zh" | "en"
    fmt          : str   "md" | "html" | "latex"
    include_toc  : bool  Include table of contents (Markdown/HTML only).
    include_meta : bool  Include simulation metadata block.
    include_snapshot : bool
        Include a raw metrics snapshot at the end of the document.
    """

    def __init__(
        self,
        lang: str = DEFAULT_LANGUAGE,
        fmt: str = "md",
        include_toc: bool = True,
        include_meta: bool = True,
        include_snapshot: bool = True,
    ):
        self.lang             = lang
        self.fmt              = fmt
        self.include_toc      = include_toc
        self.include_meta     = include_meta
        self.include_snapshot = include_snapshot

    # ── Primary Entry Point ───────────────────────────────────────────────────

    def build(
        self,
        parsed_sections: Optional[Dict[str, str]] = None,
        pipeline_output: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        simulation_meta: Optional[Dict[str, Any]] = None,
        extra_sections: Optional[Dict[str, str]] = None,
    ) -> ReportDocument:
        """
        Assemble a ReportDocument.

        Parameters
        ----------
        parsed_sections : dict[str, str], optional
            AI-generated section texts from ParserClient.
            If None, falls back to static metric tables from pipeline_output.
        pipeline_output : dict, optional
            Raw output from FeaturePipeline.run().
            Used for static fallback sections and metrics snapshot.
        title : str, optional
            Report title. Defaults to a language-appropriate generic title.
        simulation_meta : dict, optional
            Extra metadata to display (n_agents, n_steps, config, etc.).
            Merged with meta extracted from pipeline_output.
        extra_sections : dict[str, str], optional
            Additional custom sections appended after the standard ones.

        Returns
        -------
        ReportDocument

        Notes
        -----
        Section priority:
            1. parsed_sections[key]     (AI text, if provided)
            2. static fallback table    (from pipeline_output summary/final)
            3. section is omitted
        """
        pipeline_output  = pipeline_output or {}
        parsed_sections  = parsed_sections or {}
        extra_sections   = extra_sections or {}

        title = title or _default_title(self.lang)

        # Extract/merge metadata
        meta = _extract_meta(pipeline_output, simulation_meta)

        # Build section dict: AI takes priority over static fallback
        sections = self._resolve_sections(parsed_sections, pipeline_output)
        sections.update(extra_sections)

        # Render to target format
        render_kwargs = dict(
            title           = title,
            sections        = sections,
            lang            = self.lang,
            metadata        = meta,
            data_issues     = pipeline_output.get("data_issues", []),
            pipeline_output = pipeline_output,
            include_toc     = self.include_toc,
            include_meta    = self.include_meta,
            include_snapshot= self.include_snapshot,
            has_ai          = bool(parsed_sections),
        )

        if self.fmt == "md":
            content = render_markdown(**render_kwargs)
        elif self.fmt == "html":
            content = render_html(**render_kwargs)
        elif self.fmt == "latex":
            content = render_latex(**render_kwargs)
        else:
            raise ValueError(f"Unknown format '{self.fmt}'. Use 'md', 'html', or 'latex'.")

        return ReportDocument(
            content  = content,
            fmt      = self.fmt,
            lang     = self.lang,
            title    = title,
            metadata = meta,
            sections = sections,
        )

    # ── Section Resolution ────────────────────────────────────────────────────

    def _resolve_sections(
        self,
        parsed_sections: Dict[str, str],
        pipeline_output: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Merge AI sections with static fallback sections.
        AI sections take priority; static tables fill any gap.
        """
        from .outputs.static_tables import build_static_sections

        static = build_static_sections(pipeline_output, fmt=self.fmt, lang=self.lang)

        resolved: Dict[str, str] = {}

        # Standard section keys in canonical display order
        for key in _SECTION_ORDER:
            if key in parsed_sections and parsed_sections[key].strip():
                resolved[key] = parsed_sections[key]
            elif key in static and static[key].strip():
                resolved[key] = static[key]

        # Preserve any extra AI sections (stability, custom, etc.)
        for key, text in parsed_sections.items():
            if key not in resolved and text.strip():
                resolved[key] = text

        return resolved


# ═════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═════════════════════════════════════════════════════════════════════════════

def build_report(
    parsed_sections: Optional[Dict[str, str]] = None,
    pipeline_output: Optional[Dict[str, Any]] = None,
    lang: str = DEFAULT_LANGUAGE,
    fmt: str = "md",
    title: Optional[str] = None,
    simulation_meta: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ReportDocument:
    """
    One-liner convenience wrapper around ReportBuilder.build().

    Example
    -------
        doc = build_report(
            parsed_sections=ai_sections,
            pipeline_output=pipeline_output,
            lang="zh",
            fmt="md",
            title="My Simulation Report",
        )
        doc.save("output/report.md")
    """
    builder = ReportBuilder(lang=lang, fmt=fmt, **kwargs)
    return builder.build(
        parsed_sections=parsed_sections,
        pipeline_output=pipeline_output,
        title=title,
        simulation_meta=simulation_meta,
    )


def build_static_report(
    pipeline_output: Dict[str, Any],
    lang: str = DEFAULT_LANGUAGE,
    fmt: str = "md",
    title: Optional[str] = None,
) -> ReportDocument:
    """
    Build a no-AI static report directly from pipeline output.

    Example
    -------
        doc = build_static_report(pipeline_output, fmt="html", lang="zh")
        doc.save("output/report.html")
    """
    return build_report(
        pipeline_output=pipeline_output,
        lang=lang,
        fmt=fmt,
        title=title,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═════════════════════════════════════════════════════════════════════════════

#: Canonical section display order.
_SECTION_ORDER = [
    "executive_summary",
    "opinion",
    "spatial",
    "topo",
    "event",
    "stability",
    "network_opinion",
]


def _default_title(lang: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d")
    if lang == "zh":
        return f"仿真分析报告  {ts}"
    return f"Simulation Analysis Report  {ts}"


def _extract_meta(
    pipeline_output: Dict[str, Any],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract and merge metadata from pipeline output + user-supplied dict."""
    meta: Dict[str, Any] = {}

    # Pull from pipeline final snapshot meta block
    final_meta = (
        pipeline_output.get("final", {}).get("meta", {})
        if isinstance(pipeline_output.get("final"), dict)
        else {}
    )
    for k in ("time", "step", "n_agents"):
        if final_meta.get(k) is not None:
            meta[k] = final_meta[k]

    # Pull timeseries length as n_steps
    ts = pipeline_output.get("timeseries", {})
    if ts:
        first_arr = next(iter(ts.values()), None)
        if first_arr is not None:
            try:
                meta["n_steps"] = int(len(first_arr))
            except Exception:
                pass

    if extra:
        meta.update(extra)

    return meta
