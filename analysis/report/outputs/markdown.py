"""
analysis/report/outputs/markdown.py

Markdown Report Renderer
------------------------
Converts assembled section texts + metadata into a single Markdown document.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from analysis.constants import DEFAULT_LANGUAGE
from .header import lbl, ordered_section_keys, section_label
from .static_tables import format_metrics_snapshot


# ═════════════════════════════════════════════════════════════════════════════
# Renderer
# ═════════════════════════════════════════════════════════════════════════════

def render_markdown(
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
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = []

    # ── Title block ──────────────────────────────────────────────────────────
    mode_badge = f"[{lbl('ai_mode', lang)}]" if has_ai else f"[{lbl('static_mode', lang)}]"
    lines += [
        f"# {title}",
        f"> {lbl('generated', lang)}: {ts}  {mode_badge}",
        "",
    ]

    # ── Data quality warnings ────────────────────────────────────────────────
    if data_issues:
        lines += [f"## ⚠️ {lbl('data_issues', lang)}", ""]
        for issue in data_issues:
            lines.append(f"- {issue}")
        lines.append("")

    # ── Simulation metadata ───────────────────────────────────────────────────
    if include_meta and metadata:
        lines += [f"## {lbl('simulation_meta', lang)}", ""]
        for k, v in metadata.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "---", ""]

    # ── Table of contents ─────────────────────────────────────────────────────
    if include_toc and sections:
        ordered_keys = ordered_section_keys(sections)
        lines += [f"## {lbl('toc', lang)}", ""]
        for i, key in enumerate(ordered_keys, 1):
            label = section_label(key, lang)
            anchor = label.lower().replace(" ", "-").replace("/", "").replace("(", "").replace(")", "")
            lines.append(f"{i}. [{label}](#{anchor})")
        lines += ["", "---", ""]

    # ── Section bodies ────────────────────────────────────────────────────────
    for key in ordered_section_keys(sections):
        heading = section_label(key, lang)
        lines += [
            f"## {heading}",
            "",
            sections[key].strip(),
            "",
        ]

    # ── Metrics snapshot ──────────────────────────────────────────────────────
    if include_snapshot and pipeline_output:
        snap = format_metrics_snapshot(pipeline_output, fmt="md")
        if snap:
            lines += [
                "---",
                f"## {lbl('metrics_snapshot', lang)}",
                "",
                snap,
                "",
            ]

    return "\n".join(lines)
