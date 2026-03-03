"""Shared report output headers and section ordering helpers."""

from __future__ import annotations

from typing import Dict, List

COMMON_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "generated": "Generated",
        "executive_summary": "Executive Summary",
        "opinion": "Opinion Dynamics",
        "spatial": "Spatial Distribution",
        "topo": "Network Topology",
        "event": "Event Stream",
        "stability": "Cross-run Stability",
        "network_opinion": "Network–Opinion Coupling",
        "metrics_snapshot": "Metrics Snapshot (Final State)",
        "data_issues": "Data Quality Warnings",
        "simulation_meta": "Simulation Metadata",
        "ai_mode": "AI-Assisted Analysis",
        "static_mode": "Metric Summary",
        "toc": "Contents",
        "section_unknown": "Additional Analysis",
    },
    "zh": {
        "generated": "生成时间",
        "executive_summary": "执行摘要",
        "opinion": "意见动态分析",
        "spatial": "空间分布分析",
        "topo": "网络拓扑分析",
        "event": "事件流分析",
        "stability": "多次运行稳定性",
        "network_opinion": "网络-意见耦合",
        "metrics_snapshot": "终态指标快照",
        "data_issues": "数据质量警告",
        "simulation_meta": "仿真元数据",
        "ai_mode": "AI 辅助分析",
        "static_mode": "指标摘要",
        "toc": "目录",
        "section_unknown": "补充分析",
    },
}

SECTION_ORDER = [
    "executive_summary",
    "opinion",
    "spatial",
    "topo",
    "event",
    "stability",
    "network_opinion",
]


def lbl(key: str, lang: str) -> str:
    return COMMON_LABELS.get(lang, COMMON_LABELS["en"]).get(key, key)


def section_label(key: str, lang: str) -> str:
    language_labels = COMMON_LABELS.get(lang, {})
    return lbl(key, lang) if key in language_labels else lbl("section_unknown", lang)


def ordered_section_keys(sections: Dict[str, str]) -> List[str]:
    ordered = [k for k in SECTION_ORDER if k in sections]
    extras = sorted(k for k in sections if k not in SECTION_ORDER)
    return ordered + extras
