"""Reusable metric card components for dashboard tabs."""

from __future__ import annotations

import gradio as gr


_METRIC_LABELS = {
    "en": {
        "sigma": "Polarization σ",
        "mean": "Opinion Mean μ",
        "impact": "Mean Impact",
        "events": "Event Count",
        "consensus": "Consensus Ratio",
    },
    "zh": {
        "sigma": "极化度 σ",
        "mean": "意见均值 μ",
        "impact": "平均影响力",
        "events": "事件数量",
        "consensus": "共识比例",
    },
}


def build_metric_cards(lang: str = "en", interactive: bool = False) -> dict[str, gr.Number]:
    """Build a 5-card metric row used by monitor/analysis tabs."""
    labels = _METRIC_LABELS.get(lang, _METRIC_LABELS["en"])
    cards: dict[str, gr.Number] = {}

    with gr.Row(equal_height=True):
        cards["metric_sigma"] = gr.Number(
            label=labels["sigma"], value=0.0, precision=4, interactive=interactive, elem_classes=["metric-card"]
        )
        cards["metric_mean"] = gr.Number(
            label=labels["mean"], value=0.0, precision=4, interactive=interactive, elem_classes=["metric-card"]
        )
        cards["metric_impact"] = gr.Number(
            label=labels["impact"], value=0.0, precision=4, interactive=interactive, elem_classes=["metric-card"]
        )
        cards["metric_events"] = gr.Number(
            label=labels["events"], value=0, precision=0, interactive=interactive, elem_classes=["metric-card"]
        )
        cards["metric_consensus"] = gr.Number(
            label=labels["consensus"], value=0.0, precision=4, interactive=interactive, elem_classes=["metric-card"]
        )

    return cards
