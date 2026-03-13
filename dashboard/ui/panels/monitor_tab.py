"""
ui/panels/monitor_tab.py

Real-time Monitor tab for the Opinion Dynamics Simulation Dashboard.

Renders the live streaming view:
  • Step / status label
  • Five metric cards  (σ, μ, impact, events, consensus)
  • 2 × 2 chart grid   (timeseries | spatial / histogram | events)
  • Pause + Stop buttons

Public API
----------
MonitorTab
    Dataclass that holds all gr.components for the tab.

build_monitor_tab(lang) -> MonitorTab
    Render the monitor tab inside the current Gradio context.
    Must be called inside an active Tab context.

get_output_list(tab) -> list[gr.Component]
    Ordered list that matches the streaming yield tuple from
    SimulationRunner.run_stream().  Use as ``outputs=`` on the
    run button click.

    Yield index  Component
    ──────────── ─────────────────────────────────────
    0            status_md
    1            metric_sigma
    2            metric_mean
    3            metric_impact
    4            metric_events
    5            metric_consensus
    6            plot_timeseries
    7            plot_spatial
    8            plot_histogram
    9            plot_events
    10           pause_btn  (gr.update)
    11           stop_btn   (gr.update)

reset_outputs(tab) -> tuple
    Return a tuple of reset values that clears all monitor outputs,
    suitable for use as the return value of a reset handler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gradio as gr

from ui.components.metric_cards import MetricCards, build_metric_cards


# ─────────────────────────────────────────────────────────────────────────────
# MonitorTab dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonitorTab:
    """Holds all gr.Components for the Monitor tab."""

    # Status line
    status_md:       gr.Markdown

    # Metric cards (delegated to MetricCards)
    cards:           MetricCards

    # 2 × 2 charts
    plot_timeseries: gr.Plot
    plot_spatial:    gr.Plot
    plot_histogram:  gr.Plot
    plot_events:     gr.Plot

    # Controls
    pause_btn:       gr.Button
    stop_btn:        gr.Button

    # ── Convenience shortcuts ────────────────────────────────────────────

    @property
    def metric_sigma(self) -> gr.Number:
        return self.cards.sigma

    @property
    def metric_mean(self) -> gr.Number:
        return self.cards.mean

    @property
    def metric_impact(self) -> gr.Number:
        return self.cards.impact

    @property
    def metric_events(self) -> gr.Number:
        return self.cards.events

    @property
    def metric_consensus(self) -> gr.Number:
        return self.cards.consensus


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_monitor_tab(lang: str = "en") -> MonitorTab:
    """
    Render the Monitor tab contents inside the current Gradio context.

    Must be called inside an active ``gr.Tab()`` or ``gr.Blocks()`` context.

    Parameters
    ----------
    lang : {"en", "zh"}

    Returns
    -------
    MonitorTab
    """
    _t = lambda en, zh: zh if lang == "zh" else en

    # ── Status line ───────────────────────────────────────────────────────
    with gr.Row():
        status_md = gr.Markdown(
            "● Ready" if lang == "en" else "● 就绪",
            elem_id="step-label",
        )

    # ── Metric cards ──────────────────────────────────────────────────────
    with gr.Row():
        cards = build_metric_cards(lang=lang)

    # ── 2 × 2 chart grid ─────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(elem_classes="chart-tile"):
            plot_timeseries = gr.Plot(
                label=_t("Polarization Timeline", "极化时序"),
            )
        with gr.Column(elem_classes="chart-tile"):
            plot_spatial = gr.Plot(
                label=_t("Spatial Opinion Distribution", "空间意见分布"),
            )
    with gr.Row():
        with gr.Column(elem_classes="chart-tile"):
            plot_histogram = gr.Plot(
                label=_t("Opinion Histogram", "意见分布直方图"),
            )
        with gr.Column(elem_classes="chart-tile"):
            plot_events = gr.Plot(
                label=_t("Event Timeline", "事件时序"),
            )

    # ── Pause / Stop ──────────────────────────────────────────────────────
    with gr.Row():
        pause_btn = gr.Button(
            _t("⏸  Pause", "⏸  暂停"),
            size="sm",
            interactive=False,
            elem_id="btn-pause",
        )
        stop_btn = gr.Button(
            _t("⏹  Stop", "⏹  停止"),
            size="sm",
            interactive=False,
            variant="stop",
            elem_id="btn-stop",
        )

    return MonitorTab(
        status_md=status_md,
        cards=cards,
        plot_timeseries=plot_timeseries,
        plot_spatial=plot_spatial,
        plot_histogram=plot_histogram,
        plot_events=plot_events,
        pause_btn=pause_btn,
        stop_btn=stop_btn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output list (matches runner.py yield order)
# ─────────────────────────────────────────────────────────────────────────────

def get_output_list(tab: MonitorTab) -> list:
    """
    Return the ordered outputs list for ``run_btn.click(outputs=...)``.

    Index  Component
    ─────  ──────────────────────────
    0      status_md
    1      metric_sigma
    2      metric_mean
    3      metric_impact
    4      metric_events
    5      metric_consensus
    6      plot_timeseries
    7      plot_spatial
    8      plot_histogram
    9      plot_events
    10     pause_btn
    11     stop_btn
    """
    return [
        tab.status_md,
        tab.cards.sigma,
        tab.cards.mean,
        tab.cards.impact,
        tab.cards.events,
        tab.cards.consensus,
        tab.plot_timeseries,
        tab.plot_spatial,
        tab.plot_histogram,
        tab.plot_events,
        tab.pause_btn,
        tab.stop_btn,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Reset helper
# ─────────────────────────────────────────────────────────────────────────────

def reset_outputs(tab: MonitorTab, lang: str = "en") -> tuple:
    """
    Return a tuple of reset values that clears all monitor outputs.

    Suitable as the direct return value of a reset handler:

        def _reset(runner):
            runner.reset()
            return reset_outputs(monitor_tab)
    """
    _t = lambda en, zh: zh if lang == "zh" else en
    ready = _t("● Ready", "● 就绪")

    return (
        ready,                                              # 0  status_md
        0.0,                                               # 1  sigma
        0.0,                                               # 2  mean
        0.0,                                               # 3  impact
        0,                                                 # 4  events
        0.0,                                               # 5  consensus
        None,                                              # 6  timeseries
        None,                                              # 7  spatial
        None,                                              # 8  histogram
        None,                                              # 9  events
        gr.update(value=_t("⏸  Pause", "⏸  暂停"),
                  interactive=False),                      # 10 pause_btn
        gr.update(interactive=False),                      # 11 stop_btn
    )
