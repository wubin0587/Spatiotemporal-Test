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
from .header import lbl, ordered_section_keys, section_label
from .static_tables import format_metrics_snapshot


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

    if data_issues:
        lines += [
            r"\section*{" + _tex(lbl("data_issues", lang)) + "}",
            r"\begin{itemize}",
        ]
        for issue in data_issues:
            lines.append(r"\item " + _tex(issue))
        lines.append(r"\end{itemize}")

    if include_meta and metadata:
        lines += [r"\section*{" + _tex(lbl("simulation_meta", lang)) + "}"]
        lines += [r"\begin{description}"]
        for k, v in metadata.items():
            lines.append(r"\item[" + _tex(str(k)) + r"] " + _tex(str(v)))
        lines += [r"\end{description}"]

    for key in ordered_section_keys(sections):
        label = section_label(key, lang)
        body = sections[key].strip()
        lines += [
            r"\section{" + _tex(label) + "}",
            body,
            "",
        ]

    if include_snapshot and pipeline_output:
        snap = format_metrics_snapshot(pipeline_output, fmt="latex")
        if snap:
            lines += [
                r"\section*{" + _tex(lbl("metrics_snapshot", lang)) + "}",
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


def _tex(text: str) -> str:
    for old, new in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(old, new)
    return text
