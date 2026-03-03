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
from .static_tables import format_metrics_snapshot


# ── Localisation ─────────────────────────────────────────────────────────────

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
        "data_issues":       "⚠️ Data Quality Warnings",
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
        "data_issues":       "⚠️ 数据质量警告",
        "simulation_meta":   "仿真元数据",
        "ai_mode":           "AI 辅助分析",
        "static_mode":       "指标摘要",
        "toc":               "目录",
        "section_unknown":   "补充分析",
    },
}

_SECTION_ORDER = [
    "executive_summary",
    "opinion",
    "spatial",
    "topo",
    "event",
    "stability",
    "network_opinion",
]


def _lbl(key: str, lang: str) -> str:
    return _LABELS.get(lang, _LABELS["en"]).get(key, key)


def _section_label(key: str, lang: str) -> str:
    return _lbl(key, lang) if key in _LABELS.get(lang, {}) else _lbl("section_unknown", lang)


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
    mode_badge = f"[{_lbl('ai_mode', lang)}]" if has_ai else f"[{_lbl('static_mode', lang)}]"
    lines += [
        f"# {title}",
        f"> {_lbl('generated', lang)}: {ts}  {mode_badge}",
        "",
    ]

    # ── Data quality warnings ────────────────────────────────────────────────
    if data_issues:
        lines += [f"## {_lbl('data_issues', lang)}", ""]
        for issue in data_issues:
            lines.append(f"- {issue}")
        lines.append("")

    # ── Simulation metadata ───────────────────────────────────────────────────
    if include_meta and metadata:
        lines += [f"## {_lbl('simulation_meta', lang)}", ""]
        for k, v in metadata.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "---", ""]

    # ── Table of contents ─────────────────────────────────────────────────────
    if include_toc and sections:
        ordered_keys = _ordered_section_keys(sections)
        lines += [f"## {_lbl('toc', lang)}", ""]
        for i, key in enumerate(ordered_keys, 1):
            label = _section_label(key, lang)
            anchor = label.lower().replace(" ", "-").replace("/", "").replace("(", "").replace(")", "")
            lines.append(f"{i}. [{label}](#{anchor})")
        lines += ["", "---", ""]

    # ── Section bodies ────────────────────────────────────────────────────────
    for key in _ordered_section_keys(sections):
        heading = _section_label(key, lang)
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
                f"## {_lbl('metrics_snapshot', lang)}",
                "",
                snap,
                "",
            ]

    return "\n".join(lines)


def _ordered_section_keys(sections: Dict[str, str]) -> List[str]:
    ordered = [k for k in _SECTION_ORDER if k in sections]
    extras  = sorted(k for k in sections if k not in _SECTION_ORDER)
    return ordered + extras
