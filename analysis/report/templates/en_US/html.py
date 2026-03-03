"""
analysis/report/templates/html.py

HTML Report Template
---------------------
Renders a ReportDocument to a self-contained HTML fragment.
No external CSS dependencies — minimal inline styles only.

Structure
---------
    <article class="sim-report">
      <h1>...</h1>
      <p class="meta">...</p>
      <section class="sim-meta">...</section>
      <section class="data-issues">...</section>
      <section class="report-section" id="section-{key}">
        <h2>...</h2>
        <div class="section-body">...</div>
      </section>
      ...
      <section class="metrics-snapshot">
        <pre>...</pre>
      </section>
    </article>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..language import get_label, ordered_section_keys, section_heading
from ..renderer import compact_json, format_timestamp

if TYPE_CHECKING:
    from ..renderer import ReportDocument


class HtmlTemplate:

    def __init__(self, doc: "ReportDocument"):
        self.doc = doc

    def render(self) -> str:
        doc  = self.doc
        lang = doc.lang
        parts: List[str] = []

        parts += [
            "<article class='sim-report'>",
            f"<h1 class='report-title'>{_esc(doc.title)}</h1>",
            f"<p class='report-meta'>{get_label('generated', lang)}: {format_timestamp()}</p>",
        ]

        # Metadata
        if doc.simulation_meta:
            parts += [
                "<section class='sim-meta'>",
                f"<h2>{_esc(get_label('simulation_meta', lang))}</h2>",
                f"<pre class='json-block'>{_esc(compact_json(doc.simulation_meta))}</pre>",
                "</section>",
            ]

        # Data issues
        issues = doc.pipeline_output.get("data_issues", [])
        if issues:
            items = "".join(f"<li>{_esc(i)}</li>" for i in issues)
            parts += [
                "<section class='data-issues'>",
                f"<h2>⚠ {_esc(get_label('data_issues', lang))}</h2>",
                f"<ul>{items}</ul>",
                "</section>",
            ]

        # Sections
        for key in ordered_section_keys(doc.sections):
            heading = section_heading(key, lang)
            content = doc.sections[key].strip()
            parts += [
                f"<section class='report-section' id='section-{key}'>",
                f"<h2>{_esc(heading)}</h2>",
                f"<div class='section-body'>{content}</div>",
                "</section>",
            ]

        # Metrics snapshot
        if doc.include_metrics_snapshot:
            snapshot = doc.pipeline_output.get("final", {})
            if snapshot:
                parts += [
                    "<section class='metrics-snapshot'>",
                    f"<h2>{_esc(get_label('metrics_snapshot', lang))}</h2>",
                    f"<pre class='json-block'>{_esc(compact_json(snapshot))}</pre>",
                    "</section>",
                ]

        parts.append("</article>")
        return "\n".join(parts)


def _esc(text: str) -> str:
    """Minimal HTML escaping for plain text inserted into tags."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )
