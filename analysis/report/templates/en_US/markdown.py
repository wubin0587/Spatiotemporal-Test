"""
analysis/report/templates/md.py

Markdown Report Template
-------------------------
Renders a ReportDocument to a Markdown string.

Structure
---------
    # {title}
    > Generated: {timestamp}

    ## ⚠ Data Quality Warnings   (if any)

    ## Simulation Metadata        (if provided)

    ## {Section Heading}
    {section content}
    ...

    ---
    ## Final State Metrics Snapshot
    ```json
    {metrics json}
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..language import get_label, ordered_section_keys, section_heading
from ..renderer import compact_json, format_timestamp

if TYPE_CHECKING:
    from ..renderer import ReportDocument


class MdTemplate:

    def __init__(self, doc: "ReportDocument"):
        self.doc = doc

    def render(self) -> str:
        doc  = self.doc
        lang = doc.lang
        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        lines += [
            f"# {doc.title}",
            f"> {get_label('generated', lang)}: {format_timestamp()}",
            "",
        ]

        # ── Simulation metadata ───────────────────────────────────────────────
        if doc.simulation_meta:
            lines += [
                f"## {get_label('simulation_meta', lang)}",
                "",
                "```json",
                compact_json(doc.simulation_meta),
                "```",
                "",
            ]

        # ── Data quality warnings ─────────────────────────────────────────────
        issues = doc.pipeline_output.get("data_issues", [])
        if issues:
            lines += [f"## ⚠️ {get_label('data_issues', lang)}", ""]
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")

        # ── Sections ──────────────────────────────────────────────────────────
        for key in ordered_section_keys(doc.sections):
            heading = section_heading(key, lang)
            content = doc.sections[key].strip()
            lines += [f"## {heading}", "", content, ""]

        # ── Metrics snapshot ──────────────────────────────────────────────────
        if doc.include_metrics_snapshot:
            snapshot = doc.pipeline_output.get("final", {})
            if snapshot:
                lines += [
                    "---",
                    f"## {get_label('metrics_snapshot', lang)}",
                    "",
                    "```json",
                    compact_json(snapshot),
                    "```",
                    "",
                ]

        return "\n".join(lines)
