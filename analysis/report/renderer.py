"""
analysis/report/renderer.py

Report Renderer
----------------
Core rendering engine. Converts assembled sections into a complete
formatted document string.

ReportDocument is the return type of build_report().
It exposes .render() and .save() as the primary interface.

Rendering pipeline
------------------
    sections dict
        + pipeline_output (for footer / data issues)
        + fmt / lang / title
        ↓
    template (from templates/)   ← for static sections only
        ↓
    ReportDocument.render()
        ↓
    str (Markdown / HTML / LaTeX)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .language import (
    get_label,
    get_extension,
    ordered_section_keys,
    section_heading,
)


# ═════════════════════════════════════════════════════════════════════════════
# ReportDocument
# ═════════════════════════════════════════════════════════════════════════════

class ReportDocument:
    """
    Container for an assembled report. Returned by build_report().

    Attributes
    ----------
    sections : dict[str, str]
        Section name → rendered content string.
    pipeline_output : dict
        Original feature pipeline output (used for footer).
    fmt : str
        Output format: "md" | "html" | "latex"
    lang : str
        Language: "zh" | "en"
    title : str
        Report title.
    include_metrics_snapshot : bool
        Whether to append raw metrics JSON to the footer.
    simulation_meta : dict, optional
        Extra metadata included in the header block.
    """

    def __init__(
        self,
        sections: Dict[str, str],
        pipeline_output: Dict[str, Any],
        fmt: str = "md",
        lang: str = "zh",
        title: str = "Simulation Analysis Report",
        include_metrics_snapshot: bool = True,
        simulation_meta: Optional[Dict[str, Any]] = None,
    ):
        self.sections               = sections
        self.pipeline_output        = pipeline_output
        self.fmt                    = fmt
        self.lang                   = lang
        self.title                  = title
        self.include_metrics_snapshot = include_metrics_snapshot
        self.simulation_meta        = simulation_meta or {}

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Render the full report as a string in the configured format."""
        if self.fmt == "md":
            return _render_md(self)
        elif self.fmt == "html":
            return _render_html(self)
        elif self.fmt == "latex":
            return _render_latex(self)
        else:
            raise ValueError(f"Unknown format '{self.fmt}'. Use 'md', 'html', or 'latex'.")

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> str:
        """
        Render and save to file. Appends correct extension if missing.

        Returns the absolute path of the saved file.
        """
        ext = get_extension(self.fmt)
        if not filepath.endswith(ext):
            filepath = filepath.rstrip(".") + ext

        dirpath = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(dirpath, exist_ok=True)

        content = self.render()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return os.path.abspath(filepath)

    # ── Convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ReportDocument(fmt={self.fmt!r}, lang={self.lang!r}, "
            f"sections={list(self.sections.keys())})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════

def render_document(
    sections: Dict[str, str],
    pipeline_output: Dict[str, Any],
    fmt: str = "md",
    lang: str = "zh",
    title: Optional[str] = None,
    include_metrics_snapshot: bool = True,
    simulation_meta: Optional[Dict[str, Any]] = None,
) -> ReportDocument:
    """Create and return a ReportDocument (does not render yet)."""
    title = title or get_label("title", lang)
    return ReportDocument(
        sections=sections,
        pipeline_output=pipeline_output,
        fmt=fmt,
        lang=lang,
        title=title,
        include_metrics_snapshot=include_metrics_snapshot,
        simulation_meta=simulation_meta,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Format Renderers
# ═════════════════════════════════════════════════════════════════════════════

def _render_md(doc: ReportDocument) -> str:
    from .templates.md import MdTemplate
    return MdTemplate(doc).render()


def _render_html(doc: ReportDocument) -> str:
    from .templates.html import HtmlTemplate
    return HtmlTemplate(doc).render()


def _render_latex(doc: ReportDocument) -> str:
    from .templates.latex import LatexTemplate
    return LatexTemplate(doc).render()


# ═════════════════════════════════════════════════════════════════════════════
# Shared rendering utilities (used by templates)
# ═════════════════════════════════════════════════════════════════════════════

def compact_json(obj: Any, max_chars: int = 6000) -> str:
    """JSON-serialise obj with numpy support, truncating if too long."""
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

    raw = json.dumps(obj, indent=2, cls=_E, ensure_ascii=False)
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "\n…"
    return raw


def format_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def tex_escape(text: str) -> str:
    """Escape LaTeX special characters."""
    replacements = [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text
