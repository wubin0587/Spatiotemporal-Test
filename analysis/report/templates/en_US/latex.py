"""
analysis/report/templates/latex.py

LaTeX Report Template
----------------------
Renders a ReportDocument to a LaTeX fragment (no \\documentclass preamble).

Structure
---------
    \\title{...}  \\date{...}  \\maketitle
    \\section*{Simulation Metadata}
    \\section*{Data Quality Warnings}
    \\section{Opinion Dynamics}
    ...
    \\section*{Final State Metrics}
    \\begin{verbatim}...\\end{verbatim}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..language import get_label, ordered_section_keys, section_heading
from ..renderer import compact_json, format_timestamp, tex_escape

if TYPE_CHECKING:
    from ..renderer import ReportDocument


class LatexTemplate:

    def __init__(self, doc: "ReportDocument"):
        self.doc = doc

    def render(self) -> str:
        doc  = self.doc
        lang = doc.lang
        lines: List[str] = []

        lines += [
            r"\title{" + tex_escape(doc.title) + "}",
            r"\date{" + format_timestamp() + "}",
            r"\maketitle",
            "",
        ]

        # Metadata
        if doc.simulation_meta:
            lines += [
                r"\section*{" + tex_escape(get_label("simulation_meta", lang)) + "}",
                r"\begin{verbatim}",
                compact_json(doc.simulation_meta, max_chars=2000),
                r"\end{verbatim}",
                "",
            ]

        # Data issues
        issues = doc.pipeline_output.get("data_issues", [])
        if issues:
            lines += [
                r"\section*{" + tex_escape(get_label("data_issues", lang)) + "}",
                r"\begin{itemize}",
            ]
            for issue in issues:
                lines.append(r"\item " + tex_escape(issue))
            lines += [r"\end{itemize}", ""]

        # Sections
        for key in ordered_section_keys(doc.sections):
            heading = section_heading(key, lang)
            content = doc.sections[key].strip()
            lines += [
                r"\section{" + tex_escape(heading) + "}",
                content,
                "",
            ]

        # Metrics snapshot
        if doc.include_metrics_snapshot:
            snapshot = doc.pipeline_output.get("final", {})
            if snapshot:
                lines += [
                    r"\section*{" + tex_escape(get_label("metrics_snapshot", lang)) + "}",
                    r"\begin{verbatim}",
                    compact_json(snapshot, max_chars=3000),
                    r"\end{verbatim}",
                    "",
                ]

        return "\n".join(lines)
