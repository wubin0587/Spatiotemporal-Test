"""
analysis/report/outputs/html.py

HTML Report Renderer
--------------------
Converts assembled section texts + metadata into a self-contained HTML document.
Includes embedded CSS (dark theme matching the visualization palette).
No external dependencies — single-file output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from analysis.constants import DEFAULT_LANGUAGE
from .static_tables import format_metrics_snapshot


_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "generated":         "Generated",
        "executive_summary": "Executive Summary",
        "opinion":           "Opinion Dynamics",
        "spatial":           "Spatial Distribution",
        "topo":              "Network Topology",
        "event":             "Event Stream",
        "stability":         "Cross-run Stability",
        "network_opinion":   "Network–Opinion Coupling",
        "metrics_snapshot":  "Metrics Snapshot (Final State)",
        "data_issues":       "Data Quality Warnings",
        "simulation_meta":   "Simulation Metadata",
        "ai_mode":           "AI-Assisted Analysis",
        "static_mode":       "Metric Summary",
        "toc":               "Contents",
        "section_unknown":   "Additional Analysis",
    },
    "zh": {
        "generated":         "生成时间",
        "executive_summary": "执行摘要",
        "opinion":           "意见动态分析",
        "spatial":           "空间分布分析",
        "topo":              "网络拓扑分析",
        "event":             "事件流分析",
        "stability":         "多次运行稳定性",
        "network_opinion":   "网络-意见耦合",
        "metrics_snapshot":  "终态指标快照",
        "data_issues":       "数据质量警告",
        "simulation_meta":   "仿真元数据",
        "ai_mode":           "AI 辅助分析",
        "static_mode":       "指标摘要",
        "toc":               "目录",
        "section_unknown":   "补充分析",
    },
}

_SECTION_ORDER = [
    "executive_summary", "opinion", "spatial", "topo",
    "event", "stability", "network_opinion",
]

_CSS = """
:root {
  --bg:       #0d0f14;
  --surface:  #141720;
  --border:   #1e2233;
  --text:     #e8ecf4;
  --dim:      #6b7280;
  --amber:    #f59e0b;
  --teal:     #14b8a6;
  --rose:     #f43f5e;
  --sky:      #38bdf8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.7;
  padding: 2rem;
}
.report-wrapper { max-width: 900px; margin: 0 auto; }
.report-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 1.8rem; }
.report-title  { font-size: 1.8rem; font-weight: 700; color: var(--amber); margin-bottom: .3rem; }
.report-meta   { color: var(--dim); font-size: .85rem; font-family: monospace; }
.mode-badge    { display: inline-block; padding: .15rem .6rem; border-radius: 3px;
                 background: var(--surface); border: 1px solid var(--border);
                 font-size: .75rem; color: var(--teal); margin-left: .5rem; }
.data-issues   { background: #2a1a1a; border-left: 3px solid var(--rose);
                 padding: .8rem 1.2rem; margin-bottom: 1.5rem; border-radius: 4px; }
.data-issues h2 { color: var(--rose); font-size: 1rem; margin-bottom: .5rem; }
.data-issues ul { padding-left: 1.2rem; color: #f9a8d4; font-size: .9rem; }
.sim-meta      { background: var(--surface); border: 1px solid var(--border);
                 padding: .8rem 1.2rem; margin-bottom: 1.5rem; border-radius: 6px; }
.sim-meta h2   { color: var(--sky); font-size: .95rem; margin-bottom: .5rem; }
.sim-meta ul   { list-style: none; font-family: monospace; font-size: .85rem; color: var(--dim); }
.sim-meta li   { padding: .1rem 0; }
.sim-meta li strong { color: var(--text); }
nav.toc        { background: var(--surface); border: 1px solid var(--border);
                 padding: .8rem 1.2rem; margin-bottom: 2rem; border-radius: 6px; }
nav.toc h2     { color: var(--dim); font-size: .85rem; text-transform: uppercase;
                 letter-spacing: .08em; margin-bottom: .5rem; }
nav.toc ol     { padding-left: 1.4rem; }
nav.toc li     { font-size: .9rem; }
nav.toc a      { color: var(--sky); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
.report-section { margin-bottom: 2.5rem; }
.report-section h2 { font-size: 1.25rem; color: var(--amber);
                     border-bottom: 1px solid var(--border);
                     padding-bottom: .4rem; margin-bottom: 1rem; }
.section-body  { line-height: 1.75; }
.section-body p { margin-bottom: .8rem; }
.section-body h3 { color: var(--sky); font-size: 1rem; margin: 1rem 0 .4rem; }
.section-body strong { color: var(--teal); }
.section-body code   { background: var(--surface); border: 1px solid var(--border);
                       padding: .1rem .3rem; border-radius: 3px;
                       font-family: monospace; font-size: .85em; color: var(--sky); }
table.metrics-table { width: 100%; border-collapse: collapse; font-size: .85rem;
                      font-family: monospace; margin-bottom: 1rem; }
table.metrics-table th { background: var(--surface); color: var(--dim);
                         padding: .4rem .7rem; text-align: left;
                         border-bottom: 2px solid var(--border); font-weight: 600; }
table.metrics-table td { padding: .35rem .7rem; border-bottom: 1px solid var(--border); }
table.metrics-table tr:hover td { background: var(--surface); }
.metrics-snapshot  { margin-top: 2rem; border-top: 1px solid var(--border); padding-top: 1.5rem; }
.metrics-snapshot h2 { color: var(--dim); font-size: .9rem; text-transform: uppercase;
                       letter-spacing: .07em; margin-bottom: .8rem; }
pre.metrics-json   { background: var(--surface); border: 1px solid var(--border);
                     padding: 1rem; border-radius: 6px; overflow-x: auto;
                     font-size: .78rem; color: var(--dim); line-height: 1.5; }
"""


def _lbl(key: str, lang: str) -> str:
    return _LABELS.get(lang, _LABELS["en"]).get(key, key)


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^\w-]", "", text.lower().replace(" ", "-"))


def render_html(
    title: str,
    sections: Dict[str, str],
    lang: str = DEFAULT_LANGUAGE,
    metadata: Optional[Dict[str, Any]] = None,
    data_issues: Optional[List[str]] = None,
    pipeline_output: Optional[Dict[str, Any]] = None,
    include_toc: bool = True,
    include_meta: bool = True,
    include_snapshot: bool = True,
    has_ai: bool = True,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lang_attr = "zh-CN" if lang == "zh" else "en"
    mode_label = _lbl("ai_mode" if has_ai else "static_mode", lang)

    parts: List[str] = [
        f'<!DOCTYPE html>',
        f'<html lang="{lang_attr}">',
        f'<head>',
        f'  <meta charset="UTF-8">',
        f'  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'  <title>{_esc(title)}</title>',
        f'  <style>{_CSS}</style>',
        f'</head>',
        f'<body>',
        f'<div class="report-wrapper">',

        # Header
        f'<header class="report-header">',
        f'  <div class="report-title">{_esc(title)}</div>',
        f'  <div class="report-meta">',
        f'    {_lbl("generated", lang)}: {ts}',
        f'    <span class="mode-badge">{_esc(mode_label)}</span>',
        f'  </div>',
        f'</header>',
    ]

    # Data issues
    if data_issues:
        parts.append(f'<div class="data-issues">')
        parts.append(f'  <h2>⚠ {_lbl("data_issues", lang)}</h2><ul>')
        for issue in data_issues:
            parts.append(f'  <li>{_esc(issue)}</li>')
        parts.append('  </ul></div>')

    # Simulation metadata
    if include_meta and metadata:
        parts.append(f'<div class="sim-meta">')
        parts.append(f'  <h2>{_lbl("simulation_meta", lang)}</h2><ul>')
        for k, v in metadata.items():
            parts.append(f'  <li><strong>{_esc(str(k))}</strong>: {_esc(str(v))}</li>')
        parts.append('  </ul></div>')

    # Table of contents
    ordered_keys = _ordered_keys(sections)
    if include_toc and ordered_keys:
        parts.append('<nav class="toc">')
        parts.append(f'  <h2>{_lbl("toc", lang)}</h2><ol>')
        for key in ordered_keys:
            label = _lbl(key, lang) if key in _LABELS.get(lang, {}) else _lbl("section_unknown", lang)
            sid = _slug(label)
            parts.append(f'  <li><a href="#{sid}">{_esc(label)}</a></li>')
        parts.append('  </ol></nav>')

    # Section bodies
    for key in ordered_keys:
        label = _lbl(key, lang) if key in _LABELS.get(lang, {}) else _lbl("section_unknown", lang)
        sid   = _slug(label)
        body  = sections[key].strip()
        parts += [
            f'<section class="report-section" id="{sid}">',
            f'  <h2>{_esc(label)}</h2>',
            f'  <div class="section-body">{body}</div>',
            f'</section>',
        ]

    # Metrics snapshot
    if include_snapshot and pipeline_output:
        snap = format_metrics_snapshot(pipeline_output, fmt="html")
        if snap:
            parts += [
                f'<div class="metrics-snapshot">',
                f'  <h2>{_lbl("metrics_snapshot", lang)}</h2>',
                snap,
                f'</div>',
            ]

    parts += ['</div>', '</body>', '</html>']
    return "\n".join(parts)


def _ordered_keys(sections: Dict[str, str]) -> List[str]:
    ordered = [k for k in _SECTION_ORDER if k in sections]
    extras  = sorted(k for k in sections if k not in _SECTION_ORDER)
    return ordered + extras


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )
