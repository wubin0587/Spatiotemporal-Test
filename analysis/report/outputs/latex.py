"""
analysis/report/outputs/latex.py

LaTeX Report Renderer
---------------------
Converts assembled section texts + metadata into a compilable LaTeX document.
Output is a complete .tex file (with \\documentclass preamble).
Suitable for academic paper appendices or standalone reports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from analysis.constants import DEFAULT_LANGUAGE
from .static_tables import format_metrics_snapshot


_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "executive_summary": "Executive Summary",
        "opinion":           "Opinion Dynamics",
        "spatial":           "Spatial Distribution",
        "topo":              "Network Topology",
        "event":             "Event Stream",
        "stability":         "Cross-run Stability",
        "network_opinion":   "Network--Opinion Coupling",
        "metrics_snapshot":  "Metrics Snapshot (Final State)",
        "data_issues":       "Data Quality Warnings",
        "simulation_meta":   "Simulation Metadata",
        "section_unknown":   "Additional Analysis",
    },
    "zh": {
        "executive_summary": "执行摘要",
        "opinion":           "意见动态分析",
        "spatial":           "空间分布分析",
        "topo":              "网络拓扑分析",
        "event":             "事件流分析",
        "stability":         "多次运行稳定性",
        "network_opinion":   "网络--意见耦合",
        "metrics_snapshot":  "终态指标快照",
        "data_issues":       "数据质量警告",
        "simulation_meta":   "仿真元数据",
        "section_unknown":   "补充分析",
    },
}

_SECTION_ORDER = [
    "executive_summary", "opinion", "spatial", "topo",
    "event", "stability", "network_opinion",
]


def _lbl(key: str, lang: str) -> str:
    return _LABELS.get(lang, _LABELS["en"]).get(key, key)


def render_latex(
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
    ts = datetime.now().strftime("%Y-%m-%d")
    use_cjk = lang == "zh"

    lines: List[str] = _preamble(use_cjk)

    lines += [
        r"\title{" + _tex(title) + "}",
        r"\date{" + ts + "}",
        r"\author{}",
        r"\begin{document}",
        r"\maketitle",
    ]

    if include_toc:
        lines += [r"\tableofcontents", r"\newpage"]

    # Data issues
    if data_issues:
        lines += [
            r"\section*{" + _tex(_lbl("data_issues", lang)) + "}",
            r"\begin{itemize}",
        ]
        for issue in data_issues:
            lines.append(r"\item " + _tex(issue))
        lines.append(r"\end{itemize}")

    # Simulation metadata
    if include_meta and metadata:
        lines += [r"\section*{" + _tex(_lbl("simulation_meta", lang)) + "}"]
        lines += [r"\begin{description}"]
        for k, v in metadata.items():
            lines.append(r"\item[" + _tex(str(k)) + r"] " + _tex(str(v)))
        lines += [r"\end{description}"]

    # Sections
    for key in _ordered_keys(sections):
        label = _lbl(key, lang) if key in _LABELS.get(lang, {}) else _lbl("section_unknown", lang)
        body  = sections[key].strip()
        lines += [
            r"\section{" + _tex(label) + "}",
            body,
            "",
        ]

    # Metrics snapshot
    if include_snapshot and pipeline_output:
        snap = format_metrics_snapshot(pipeline_output, fmt="latex")
        if snap:
            lines += [
                r"\section*{" + _tex(_lbl("metrics_snapshot", lang)) + "}",
                snap,
            ]

    lines.append(r"\end{document}")
    return "\n".join(lines)


def _preamble(use_cjk: bool) -> List[str]:
    lines = [
        r"\documentclass[12pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{geometry}",
        r"\geometry{margin=2.5cm}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=teal}",
        r"\usepackage{parskip}",
        r"\setlength{\parskip}{6pt}",
    ]
    if use_cjk:
        lines += [
            r"\usepackage{xeCJK}",
            r"\setCJKmainfont{Noto Serif CJK SC}",
        ]
    return lines


def _ordered_keys(sections: Dict[str, str]) -> List[str]:
    ordered = [k for k in _SECTION_ORDER if k in sections]
    extras  = sorted(k for k in sections if k not in _SECTION_ORDER)
    return ordered + extras


def _tex(text: str) -> str:
    for old, new in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(old, new)
    return text
