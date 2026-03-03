"""
analysis/report/outputs/static_tables.py

Static Metric Table Generator
------------------------------
Converts raw pipeline_output (feature dicts and summary stats) into
formatted metric tables. Used as a fallback when AI sections are unavailable,
and as the "metrics snapshot" footer in all report formats.

Output format adapts to the requested fmt ("md" | "html" | "latex").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from analysis.constants import DEFAULT_LANGUAGE


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def build_static_sections(
    pipeline_output: Dict[str, Any],
    fmt: str = "md",
    lang: str = DEFAULT_LANGUAGE,
) -> Dict[str, str]:
    """
    Build per-section metric tables from pipeline_output without AI.

    Priority order:
        1. pipeline_output["summary"]  — time-series stats (mean, std, trend_slope …)
        2. pipeline_output["final"]    — last-step snapshot values

    Returns
    -------
    dict[str, str]  — section_name → formatted table/text
    """
    summary  = pipeline_output.get("summary", {})
    final    = pipeline_output.get("final",   {})

    # Group summary by section prefix ("opinion.polarization_std" → "opinion")
    section_data = _group_by_prefix(summary)

    sections: Dict[str, str] = {}
    for key in _STANDARD_SECTIONS:
        data = section_data.get(key, {})
        if not data:
            # Fall back to final snapshot scalar values
            data = final.get(key, {})
        if data:
            sections[key] = _format_section(data, fmt=fmt, lang=lang)

    return sections


def format_metrics_snapshot(
    pipeline_output: Dict[str, Any],
    fmt: str = "md",
    max_chars: int = 6000,
) -> str:
    """
    Render the raw final-state metrics as a JSON code block or pre block.
    Used as the appendix / footer of reports.
    """
    final = pipeline_output.get("final", {})
    if not final:
        return ""
    raw = _compact_json(final, max_chars=max_chars)
    if fmt == "md":
        return f"```json\n{raw}\n```"
    elif fmt == "html":
        return f"<pre class='metrics-json'>{raw}</pre>"
    elif fmt == "latex":
        capped = raw[:3000]
        return f"\\begin{{verbatim}}\n{capped}\n\\end{{verbatim}}"
    return raw


# ═════════════════════════════════════════════════════════════════════════════
# Internal Section Formatters
# ═════════════════════════════════════════════════════════════════════════════

_STANDARD_SECTIONS = ["executive_summary", "opinion", "spatial", "topo", "event"]

# Columns shown in the summary table for time-series stats
_SUMMARY_COLS   = ("mean", "std", "trend_slope", "end_mean", "final_stability")
_COL_LABELS_EN  = ("Metric", "Mean", "Std", "Trend Slope", "End Mean", "Stability")
_COL_LABELS_ZH  = ("指标", "均值", "标准差", "趋势斜率", "终态均值", "稳定性")


def _format_section(
    data: Dict[str, Any],
    fmt: str = "md",
    lang: str = DEFAULT_LANGUAGE,
) -> str:
    """Format a single section's metric dict into a table."""
    rows: List[Tuple[str, ...]] = []
    headers = _COL_LABELS_ZH if lang == "zh" else _COL_LABELS_EN

    for metric_key, val in sorted(data.items()):
        if isinstance(val, dict):
            row = _extract_stat_row(metric_key, val)
        elif isinstance(val, (int, float)):
            row = (metric_key, f"{val:.5g}", "—", "—", "—", "—")
        elif val is None:
            continue
        else:
            row = (metric_key, str(val), "—", "—", "—", "—")
        rows.append(row)

    if not rows:
        return _no_data(fmt, lang)

    if fmt == "md":
        return _md_table(headers, rows)
    elif fmt == "html":
        return _html_table(headers, rows)
    elif fmt == "latex":
        return _latex_table(headers, rows)
    return _md_table(headers, rows)


def _extract_stat_row(key: str, stats: Dict[str, Any]) -> Tuple[str, ...]:
    """Pull the 5 display columns from a stats sub-dict."""
    def _fmt(v: Any, signed: bool = False) -> str:
        if v is None or (isinstance(v, float) and v != v):   # nan
            return "—"
        try:
            f = float(v)
            return f"{f:+.4f}" if signed else f"{f:.5g}"
        except (TypeError, ValueError):
            return str(v)

    return (
        key,
        _fmt(stats.get("mean")),
        _fmt(stats.get("std")),
        _fmt(stats.get("trend_slope"), signed=True),
        _fmt(stats.get("end_mean")),
        _fmt(stats.get("final_stability")),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Table Renderers
# ═════════════════════════════════════════════════════════════════════════════

def _md_table(headers: Tuple, rows: List[Tuple]) -> str:
    sep  = " | ".join("---" for _ in headers)
    head = " | ".join(str(h) for h in headers)
    lines = [f"| {head} |", f"| {sep} |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _html_table(headers: Tuple, rows: List[Tuple]) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return (
        "<table class='metrics-table'>"
        f"<thead><tr>{ths}</tr></thead>"
        f"<tbody>{trs}</tbody>"
        "</table>"
    )


def _latex_table(headers: Tuple, rows: List[Tuple]) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    head = " & ".join(_tex(str(h)) for h in headers) + r" \\"
    body = "\n".join(
        " & ".join(_tex(str(c)) for c in row) + r" \\"
        for row in rows
    )
    return (
        f"\\begin{{tabular}}{{{cols}}}\n"
        f"\\hline\n{head}\n\\hline\n{body}\n\\hline\n"
        f"\\end{{tabular}}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

def _no_data(fmt: str, lang: str) -> str:
    msg_zh = "（暂无数据）"
    msg_en = "*(no data)*"
    msg = msg_zh if lang == "zh" else msg_en
    if fmt == "html":
        return f"<em>{msg}</em>"
    if fmt == "latex":
        return f"\\textit{{{msg}}}"
    return msg


def _group_by_prefix(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group "section.metric" flat keys into nested section dicts."""
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in summary.items():
        if "." in key:
            section, metric = key.split(".", 1)
        else:
            section, metric = "other", key
        out.setdefault(section, {})[metric] = val
    return out


def _compact_json(obj: Any, max_chars: int = 6000) -> str:
    import json

    class _E(json.JSONEncoder):
        def default(self, o):
            try:
                import numpy as np
                if isinstance(o, np.integer): return int(o)
                if isinstance(o, np.floating): return round(float(o), 5)
                if isinstance(o, np.ndarray):  return o.tolist()
            except ImportError:
                pass
            return str(o)

    raw = json.dumps(obj, indent=2, cls=_E, ensure_ascii=False)
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "\n... (truncated)"
    return raw


def _tex(text: str) -> str:
    """Minimal LaTeX special-char escaping."""
    for old, new in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
        ("+", r"{+}"), ("-", r"{-}"),
    ]:
        text = text.replace(old, new)
    return text
