"""Real-time monitor tab layout."""

from __future__ import annotations

import gradio as gr

from ui.components.metric_cards import build_metric_cards


_TEXT = {
    "en": {
        "status": "● Ready",
        "title": "Live Monitor",
        "pause": "⏸ Pause",
        "stop": "⏹ Stop",
        "charts": ["Polarization Timeline", "Spatial Snapshot", "Opinion Histogram", "Event Timeline"],
    },
    "zh": {
        "status": "● 就绪",
        "title": "实时监控",
        "pause": "⏸ 暂停",
        "stop": "⏹ 停止",
        "charts": ["极化演化", "空间快照", "意见直方图", "事件时间线"],
    },
}


def build_monitor_tab(lang: str = "en") -> dict[str, gr.Component]:
    """Build monitor tab components and return references by key."""
    t = _TEXT.get(lang, _TEXT["en"])
    c: dict[str, gr.Component] = {}

    c["status_md"] = gr.Markdown(f"**{t['status']}**")
    c["progress_bar"] = gr.Markdown("")
    c.update(build_metric_cards(lang=lang, interactive=False))

    with gr.Row():
        c["plot_timeseries"] = gr.Plot(label=t["charts"][0])
        c["plot_spatial"] = gr.Plot(label=t["charts"][1])
    with gr.Row():
        c["plot_histogram"] = gr.Plot(label=t["charts"][2])
        c["plot_events"] = gr.Plot(label=t["charts"][3])

    with gr.Row():
        c["pause_btn"] = gr.Button(t["pause"], interactive=False, elem_id="btn-pause")
        c["stop_btn"] = gr.Button(t["stop"], interactive=False, variant="stop", elem_id="btn-stop")

    return c


def get_output_list(components: dict[str, gr.Component]) -> list[gr.Component]:
    """Ordered output list aligned with SimulationRunner.run_stream() tuple contract."""
    return [
        components["status_md"],
        components["metric_sigma"],
        components["metric_mean"],
        components["metric_impact"],
        components["metric_events"],
        components["metric_consensus"],
        components["plot_timeseries"],
        components["plot_spatial"],
        components["plot_histogram"],
        components["plot_events"],
        components["pause_btn"],
        components["stop_btn"],
    ]
