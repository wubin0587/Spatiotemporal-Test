"""
ui/panels/monitor_tab.py  ── v2 compatibility shim

实时监控面板内容已迁移到 page_experiment.py Phase B（phase_b_group）。

迁移指南
────────
旧::
    from ui.panels.monitor_tab import build_monitor_tab
    monitor = build_monitor_tab(lang)

新::
    from ui.panels.page_experiment import build_experiment_page
    experiment = build_experiment_page(lang, defaults)
    # Phase B 组件：phase_b_group, status_md, metric_*, plot_*,
    #               pause_btn, stop_btn

app.py 监控刷新 outputs 示例::
    outputs = [
        experiment.status_md,
        experiment.metric_step, experiment.metric_time,
        experiment.metric_events, experiment.metric_polar,
        experiment.metric_clusters,
        experiment.plot_polar, experiment.plot_spatial,
        experiment.plot_hist, experiment.plot_events,
    ]
"""

from __future__ import annotations
import warnings

from ui.panels.page_experiment import (
    ExperimentComponents,
    build_experiment_page,
    transition_to_monitor,
    transition_to_checklist,
    on_run_complete,
)


def build_monitor_tab(lang: str = "zh", defaults: dict | None = None):
    """Deprecated — use build_experiment_page() instead."""
    warnings.warn(
        "build_monitor_tab() is deprecated. Use build_experiment_page().",
        DeprecationWarning, stacklevel=2,
    )
    return build_experiment_page(lang=lang, defaults=defaults)


__all__ = [
    "ExperimentComponents", "build_experiment_page", "build_monitor_tab",
    "transition_to_monitor", "transition_to_checklist", "on_run_complete",
]
