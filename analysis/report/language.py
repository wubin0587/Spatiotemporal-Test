"""
analysis/report/language.py

Localisation & Section Configuration
--------------------------------------
Centralises all human-readable labels, section ordering, and per-format
rendering helpers used across the report package.

Supported languages : "zh" (Simplified Chinese), "en" (English)
Supported formats   : "md" (Markdown), "html" (HTML), "latex" (LaTeX)
"""

from __future__ import annotations

from typing import Dict, List


# ═════════════════════════════════════════════════════════════════════════════
# Section order (canonical)
# ═════════════════════════════════════════════════════════════════════════════

#: Sections rendered in this order; unknown keys appended alphabetically.
SECTION_ORDER: List[str] = [
    "executive_summary",
    "stability",          # from multi-run analysis
    "opinion",
    "spatial",
    "topo",
    "event",
    "network_opinion",
]


# ═════════════════════════════════════════════════════════════════════════════
# Labels
# ═════════════════════════════════════════════════════════════════════════════

_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        # Document
        "title":              "Simulation Analysis Report",
        "generated":          "Generated",
        "simulation_meta":    "Simulation Metadata",
        "data_issues":        "Data Quality Warnings",
        "metrics_snapshot":   "Final State Metrics Snapshot",
        "static_note":        "Metric Summary (Auto-rendered)",
        "toc":                "Table of Contents",
        # Sections
        "executive_summary":  "Executive Summary",
        "stability":          "Cross-run Stability Analysis",
        "opinion":            "Opinion Dynamics",
        "spatial":            "Spatial Distribution",
        "topo":               "Network Topology",
        "event":              "Event Stream",
        "network_opinion":    "Network–Opinion Coupling",
        "other":              "Additional Analysis",
        # Misc
        "no_data":            "(no data)",
        "truncated":          "… (truncated)",
        "ai_section":         "AI Analysis",
        "static_section":     "Auto Summary",
    },
    "zh": {
        # Document
        "title":              "仿真分析报告",
        "generated":          "生成时间",
        "simulation_meta":    "仿真元数据",
        "data_issues":        "数据质量警告",
        "metrics_snapshot":   "终态指标快照",
        "static_note":        "指标摘要（自动渲染）",
        "toc":                "目录",
        # Sections
        "executive_summary":  "执行摘要",
        "stability":          "多轮稳定性分析",
        "opinion":            "意见动态分析",
        "spatial":            "空间分布分析",
        "topo":               "网络拓扑分析",
        "event":              "事件流分析",
        "network_opinion":    "网络—意见耦合分析",
        "other":              "补充分析",
        # Misc
        "no_data":            "（无数据）",
        "truncated":          "……（已截断）",
        "ai_section":         "AI分析",
        "static_section":     "自动摘要",
    },
}


def get_label(key: str, lang: str) -> str:
    """Return the localised label for `key` in `lang`. Falls back to `key`."""
    return _LABELS.get(lang, _LABELS["en"]).get(key, key)


def section_heading(section_key: str, lang: str) -> str:
    """Return a human-readable heading for a section key."""
    return get_label(section_key, lang)


def ordered_section_keys(sections: Dict[str, str]) -> List[str]:
    """
    Return section keys in canonical order, with unknown keys appended
    in alphabetical order at the end.
    """
    ordered = [k for k in SECTION_ORDER if k in sections]
    extras  = sorted(k for k in sections if k not in SECTION_ORDER)
    return ordered + extras


# ═════════════════════════════════════════════════════════════════════════════
# Format-specific helpers
# ═════════════════════════════════════════════════════════════════════════════

# File extension map
EXTENSION_MAP: Dict[str, str] = {
    "md":    ".md",
    "html":  ".html",
    "latex": ".tex",
}


def get_extension(fmt: str) -> str:
    return EXTENSION_MAP.get(fmt, f".{fmt}")
