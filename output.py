"""
analysis/parser/output.py

Parser Output Module
--------------------
Assembles AI-generated section texts (from client.py) into complete,
formatted documents. Can also be used standalone (no AI) to render
a pure-metrics summary document.

Two rendering modes:
    1. AI mode   — sections contain AI-generated narrative text.
    2. Static mode — sections contain auto-formatted metric tables only.

Output formats:  "md" | "html" | "latex"
Output languages: "zh" | "en"

Integration with report/
    ParsedOutput.to_report_payload() produces a dict that can be passed
    directly to report.builder.build_report() (with or without AI sections).

Usage
-----
    from analysis.parser.client import ParserClient
    from analysis.parser.output import ParsedOutput

    client = ParserClient(api_key="...", lang="zh", fmt="md")
    sections = client.parse_pipeline_result(pipeline_output)

    doc = ParsedOutput(
        sections=sections,
        fmt="md",
        lang="zh",
        pipeline_output=pipeline_output,
    )
    doc.save("output/report.md")
    print(doc.render())
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from analysis.constants import DEFAULT_LANGUAGE


# ═════════════════════════════════════════════════════════════════════════════
# Localization Strings
# ═════════════════════════════════════════════════════════════════════════════

_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "title":             "Simulation Analysis Report",
        "generated":         "Generated",
        "executive_summary": "Executive Summary",
        "opinion":           "Opinion Dynamics",
        "spatial":           "Spatial Distribution",
        "topo":              "Network Topology",
        "event":             "Event Stream",
        "metrics_snapshot":  "Final State Metrics Snapshot",
        "data_issues":       "Data Quality Warnings",
        "no_ai":             "Metric Summary (No AI Analysis)",
        "section_unknown":   "Additional Analysis",
    },
    "zh": {
        "title":             "仿真分析报告",
        "generated":         "生成时间",
        "executive_summary": "执行摘要",
        "opinion":           "意见动态分析",
        "spatial":           "空间分布分析",
        "topo":              "网络拓扑分析",
        "event":             "事件流分析",
        "metrics_snapshot":  "终态指标快照",
        "data_issues":       "数据质量警告",
        "no_ai":             "指标摘要（无AI分析）",
        "section_unknown":   "补充分析",
    },
}

_SECTION_ORDER = [
    "executive_summary",
    "opinion",
    "spatial",
    "topo",
    "event",
]


def _label(key: str, lang: str) -> str:
    return _LABELS.get(lang, _LABELS["en"]).get(key, key)


# ═════════════════════════════════════════════════════════════════════════════
# ParsedOutput
# ═════════════════════════════════════════════════════════════════════════════

class ParsedOutput:
    """
    Container for assembled parser output.

    Parameters
    ----------
    sections : dict[str, str]
        Section name -> rendered text (AI or static).
        Produced by ParserClient.parse_pipeline_result()
        or build_static_sections().
    fmt : str
        Output format: "md" | "html" | "latex"
    lang : str
        Output language: "zh" | "en"
    pipeline_output : dict, optional
        Original pipeline output — used for the metrics snapshot footer
        and for data-issue warnings.
    title : str, optional
        Report title override.
    """

    def __init__(
        self,
        sections: Dict[str, str],
        fmt: str = "md",
        lang: str = DEFAULT_LANGUAGE,
        pipeline_output: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
    ):
        self.sections        = sections
        self.fmt             = fmt
        self.lang            = lang
        self.pipeline_output = pipeline_output or {}
        self.title           = title or _label("title", lang)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Render the full document as a string in the target format."""
        if self.fmt == "md":
            return self._render_md()
        elif self.fmt == "html":
            return self._render_html()
        elif self.fmt == "latex":
            return self._render_latex()
        else:
            raise ValueError(f"Unknown format '{self.fmt}'. Use 'md', 'html', or 'latex'.")

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> str:
        """
        Save the rendered document to a file.

        The file extension is inferred from `self.fmt` if not present.

        Returns
        -------
        str  –  Absolute path of the saved file.
        """
        _ext_map = {"md": ".md", "html": ".html", "latex": ".tex"}
        expected_ext = _ext_map.get(self.fmt, "." + self.fmt)

        if not filepath.endswith(expected_ext):
            filepath = filepath.rstrip(".") + expected_ext

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        content = self.render()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return os.path.abspath(filepath)

    # ── Report Payload ────────────────────────────────────────────────────────

    def to_report_payload(self) -> Dict[str, Any]:
        """
        Convert this output to a dict consumable by report.builder.build_report().

        The report builder accepts:
            {
                "title":    str,
                "lang":     str,
                "fmt":      str,
                "sections": dict[str, str],   # name -> rendered content
                "has_ai":   bool,
                "meta":     dict,             # optional metadata
            }
        """
        return {
            "title":    self.title,
            "lang":     self.lang,
            "fmt":      self.fmt,
            "sections": self.sections,
            "has_ai":   bool(self.sections),
            "meta":     self.pipeline_output.get("final", {}).get("meta", {}),
        }

    # ── Markdown Renderer ────────────────────────────────────────────────────

    def _render_md(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines: List[str] = [
            f"# {self.title}",
            f"> {_label('generated', self.lang)}: {ts}",
            "",
        ]

        # Data quality warnings
        issues = self.pipeline_output.get("data_issues", [])
        if issues:
            lines += [
                f"## ⚠️ {_label('data_issues', self.lang)}",
                "",
            ]
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")

        # Sections in canonical order, then any extras
        for key in _section_order(self.sections):
            heading = _label(key, self.lang)
            lines += [
                f"## {heading}",
                "",
                self.sections[key],
                "",
            ]

        # Metrics snapshot (final state)
        snapshot = self.pipeline_output.get("final", {})
        if snapshot:
            lines += [
                f"---",
                f"## {_label('metrics_snapshot', self.lang)}",
                "",
                "```json",
                _compact_json(snapshot),
                "```",
                "",
            ]

        return "\n".join(lines)

    # ── HTML Renderer ────────────────────────────────────────────────────────

    def _render_html(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        parts: List[str] = [
            f"<article class='simulation-report'>",
            f"<h1 class='report-title'>{self.title}</h1>",
            f"<p class='report-meta'>{_label('generated', self.lang)}: {ts}</p>",
        ]

        issues = self.pipeline_output.get("data_issues", [])
        if issues:
            parts.append(f"<section class='data-issues'>")
            parts.append(f"<h2>{_label('data_issues', self.lang)}</h2><ul>")
            for issue in issues:
                parts.append(f"<li>{issue}</li>")
            parts.append("</ul></section>")

        for key in _section_order(self.sections):
            heading = _label(key, self.lang)
            content = self.sections[key]
            parts += [
                f"<section class='report-section' id='section-{key}'>",
                f"<h2>{heading}</h2>",
                f"<div class='section-body'>{content}</div>",
                "</section>",
            ]

        snapshot = self.pipeline_output.get("final", {})
        if snapshot:
            parts += [
                f"<section class='metrics-snapshot'>",
                f"<h2>{_label('metrics_snapshot', self.lang)}</h2>",
                f"<pre class='metrics-json'>{_compact_json(snapshot)}</pre>",
                "</section>",
            ]

        parts.append("</article>")
        return "\n".join(parts)

    # ── LaTeX Renderer ────────────────────────────────────────────────────────

    def _render_latex(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines: List[str] = [
            r"\begin{document}",
            r"\title{" + _tex_escape(self.title) + "}",
            r"\date{" + ts + "}",
            r"\maketitle",
            "",
        ]

        issues = self.pipeline_output.get("data_issues", [])
        if issues:
            lines += [
                r"\section*{" + _tex_escape(_label("data_issues", self.lang)) + "}",
                r"\begin{itemize}",
            ]
            for issue in issues:
                lines.append(r"\item " + _tex_escape(issue))
            lines += [r"\end{itemize}", ""]

        for key in _section_order(self.sections):
            heading = _label(key, self.lang)
            content = self.sections[key]
            lines += [
                r"\section{" + _tex_escape(heading) + "}",
                content,
                "",
            ]

        snapshot = self.pipeline_output.get("final", {})
        if snapshot:
            lines += [
                r"\section*{" + _tex_escape(_label("metrics_snapshot", self.lang)) + "}",
                r"\begin{verbatim}",
                _compact_json(snapshot)[:3000],  # cap verbatim length
                r"\end{verbatim}",
            ]

        lines.append(r"\end{document}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Static (No-AI) Mode
# ═════════════════════════════════════════════════════════════════════════════

def build_static_sections(
    pipeline_output: Dict[str, Any],
    fmt: str = "md",
    lang: str = DEFAULT_LANGUAGE,
) -> Dict[str, str]:
    """
    Build section texts without AI — pure metric formatting.
    Produces a no-frills table/list of all computed metrics.

    Used when:
    - No API key is available.
    - A fast, deterministic report is needed.
    - As a fallback if AI parsing fails.

    Returns
    -------
    dict[str, str]  –  Same shape as ParserClient output, compatible with ParsedOutput.
    """
    summary  = pipeline_output.get("summary", {})
    final    = pipeline_output.get("final",   {})
    sections: Dict[str, str] = {}

    section_data = _group_by_prefix(summary)

    for section_key, metrics in section_data.items():
        sections[section_key] = _format_metrics_block(metrics, fmt=fmt)

    # If summary is empty, fall back to final snapshot values
    if not sections:
        for section_key in ["opinion", "spatial", "topo", "event"]:
            data = final.get(section_key, {})
            if data:
                sections[section_key] = _format_metrics_block(data, fmt=fmt)

    return sections


def build_static_output(
    pipeline_output: Dict[str, Any],
    fmt: str = "md",
    lang: str = DEFAULT_LANGUAGE,
    title: Optional[str] = None,
) -> "ParsedOutput":
    """
    Convenience: build a no-AI ParsedOutput directly.

    Example
    -------
        doc = build_static_output(pipeline_output, fmt="html", lang="zh")
        doc.save("reports/summary.html")
    """
    sections = build_static_sections(pipeline_output, fmt=fmt, lang=lang)
    return ParsedOutput(
        sections=sections,
        fmt=fmt,
        lang=lang,
        pipeline_output=pipeline_output,
        title=title,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Format Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _format_metrics_block(metrics: Dict[str, Any], fmt: str = "md") -> str:
    """
    Render a flat or nested dict of metric stats into a formatted block.
    Used for the static (no-AI) path.
    """
    rows: List[tuple] = []

    for key, val in sorted(metrics.items()):
        if isinstance(val, dict):
            # summary stats sub-dict: pick the most informative fields
            mean  = val.get("mean",  "—")
            std   = val.get("std",   "—")
            trend = val.get("trend_slope", None)
            fin   = val.get("end_mean", None)

            mean_str  = f"{mean:.4f}"  if isinstance(mean,  float) else str(mean)
            std_str   = f"{std:.4f}"   if isinstance(std,   float) else str(std)
            trend_str = f"{trend:+.4f}" if isinstance(trend, float) else "—"
            fin_str   = f"{fin:.4f}"   if isinstance(fin,   float) else "—"

            rows.append((key, mean_str, std_str, trend_str, fin_str))
        elif isinstance(val, float):
            rows.append((key, f"{val:.4f}", "—", "—", "—"))
        else:
            rows.append((key, str(val), "—", "—", "—"))

    if not rows:
        return "*(no data)*" if fmt == "md" else "<em>(no data)</em>"

    headers = ("Metric", "Mean / Value", "Std", "Trend Slope", "End Mean")

    if fmt == "md":
        return _rows_to_md_table(headers, rows)
    elif fmt == "html":
        return _rows_to_html_table(headers, rows)
    elif fmt == "latex":
        return _rows_to_latex_table(headers, rows)
    else:
        return _rows_to_md_table(headers, rows)


def _rows_to_md_table(headers: tuple, rows: List[tuple]) -> str:
    sep = " | ".join("---" for _ in headers)
    head = " | ".join(headers)
    lines = [f"| {head} |", f"| {sep} |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _rows_to_html_table(headers: tuple, rows: List[tuple]) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for row in rows:
        tds = "".join(f"<td>{c}</td>" for c in row)
        trs += f"<tr>{tds}</tr>"
    return f"<table class='metrics-table'><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"


def _rows_to_latex_table(headers: tuple, rows: List[tuple]) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    head = " & ".join(_tex_escape(h) for h in headers) + r" \\"
    body_lines = []
    for row in rows:
        body_lines.append(" & ".join(_tex_escape(str(c)) for c in row) + r" \\")
    body = "\n".join(body_lines)
    return (
        f"\\begin{{tabular}}{{{cols}}}\n"
        f"\\hline\n{head}\n\\hline\n{body}\n\\hline\n"
        f"\\end{{tabular}}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

def _section_order(sections: Dict[str, str]) -> List[str]:
    """Return section keys in canonical order, with unknown keys appended."""
    ordered = [k for k in _SECTION_ORDER if k in sections]
    extras  = [k for k in sections if k not in _SECTION_ORDER]
    return ordered + sorted(extras)


def _group_by_prefix(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group "section.metric" keys into nested section dicts."""
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in summary.items():
        if "." in key:
            section, metric = key.split(".", 1)
        else:
            section, metric = "other", key
        out.setdefault(section, {})[metric] = val
    return out


def _compact_json(obj: Any, max_chars: int = 8000) -> str:
    import json

    class _E(json.JSONEncoder):
        def default(self, o):
            try:
                import numpy as np
                if isinstance(o, np.integer): return int(o)
                if isinstance(o, np.floating): return round(float(o), 5)
                if isinstance(o, np.ndarray): return o.tolist()
            except ImportError:
                pass
            return str(o)

    raw = json.dumps(obj, indent=2, cls=_E, ensure_ascii=False)
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "\n... (truncated)"
    return raw


def _tex_escape(text: str) -> str:
    """Escape special LaTeX characters in a plain string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text