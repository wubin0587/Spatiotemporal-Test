"""
ui/panels/page_dashboard_settings.py

P2 仪表盘设置 — 刷新频率、图表显示偏好

这些设置不影响仿真计算本身，只影响实时监控页的显示行为。
因此本页没有 validator 覆盖（所有字段均可选），侧边栏状态点不显示。

Layout
------
  Card 1: 实时监控设置
    - refresh_every: 每 N 步刷新一次图表
    - primary_layer: 主意见层索引（监控图的默认层）
    - record_history: 是否在内存中记录完整时序

  Card 2: 图表显示偏好
    - show_polarization:  显示极化时序图
    - show_spatial:       显示空间分布图
    - show_histogram:     显示意见直方图
    - show_events:        显示事件时序图

  [保存设置]  [恢复默认]

Public API
----------
DashboardSettingsComponents
    Dataclass with all gr.Components.

build_dashboard_settings_page(lang, defaults) -> DashboardSettingsComponents
    Must be called inside an active gr.Group() + gr.Blocks() context.

get_values(comps) -> dict
    Collect current component values into a flat dict.

apply_defaults(comps, defaults) -> list[gr.update]
    Push default values back to all components.
    Use as handler return for "恢复默认" button.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr


# ─── Default values ───────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "refresh_every":       10,
    "primary_layer":        0,
    "record_history":    True,
    "show_polarization": True,
    "show_spatial":      True,
    "show_histogram":    True,
    "show_events":       True,
}

# ─── Localised strings ────────────────────────────────────────────────────────

_STRINGS = {
    "zh": {
        "page_title":    "仪表盘设置",
        "page_subtitle": "控制实时监控页的刷新频率和显示偏好，不影响仿真计算。",
        "card1_title":   "实时监控设置",
        "refresh_label": "刷新间隔（步）",
        "refresh_info":  "每隔多少仿真步更新一次图表，设为 1 可查看每步变化（较慢）",
        "layer_label":   "主意见层",
        "layer_info":    "监控图和指标卡默认展示的意见层索引（从 0 开始）",
        "history_label": "记录完整时序",
        "history_info":  "启用后将全程记录每步指标，用于事后完整分析（占用更多内存）",
        "card2_title":   "图表显示",
        "polar_label":   "极化时序图",
        "spatial_label": "空间分布图",
        "hist_label":    "意见直方图",
        "events_label":  "事件时序图",
        "save_btn":      "保存设置",
        "reset_btn":     "恢复默认",
        "saved_msg":     "✓ 已保存",
    },
    "en": {
        "page_title":    "Dashboard Settings",
        "page_subtitle": "Controls refresh rate and display preferences for the live monitor. Does not affect simulation.",
        "card1_title":   "Live Monitor",
        "refresh_label": "Refresh every N steps",
        "refresh_info":  "How many simulation steps between chart updates. 1 = update every step (slow).",
        "layer_label":   "Primary layer",
        "layer_info":    "Default opinion layer index shown in monitor charts (0-indexed).",
        "history_label": "Record full history",
        "history_info":  "Stores per-step metrics in memory for full post-run analysis (uses more RAM).",
        "card2_title":   "Chart Visibility",
        "polar_label":   "Polarisation timeline",
        "spatial_label": "Spatial scatter",
        "hist_label":    "Opinion histogram",
        "events_label":  "Event timeline",
        "save_btn":      "Save",
        "reset_btn":     "Restore defaults",
        "saved_msg":     "✓ Saved",
    },
}


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class DashboardSettingsComponents:
    """
    All gr.Components created by build_dashboard_settings_page().

    Component keys match the _DEFAULTS dict — they are also the keys
    used in the all_param_components dict assembled in app.py.

    Attributes
    ----------
    refresh_every : gr.Number
    primary_layer : gr.Number
    record_history : gr.Checkbox
    show_polarization : gr.Checkbox
    show_spatial : gr.Checkbox
    show_histogram : gr.Checkbox
    show_events : gr.Checkbox
    save_btn : gr.Button
    reset_btn : gr.Button
    saved_md : gr.Markdown
    """
    refresh_every:     gr.Number
    primary_layer:     gr.Number
    record_history:    gr.Checkbox
    show_polarization: gr.Checkbox
    show_spatial:      gr.Checkbox
    show_histogram:    gr.Checkbox
    show_events:       gr.Checkbox
    save_btn:          gr.Button
    reset_btn:         gr.Button
    saved_md:          gr.Markdown

    @property
    def param_components(self) -> dict[str, gr.Component]:
        """Flat dict of {key: component} for sidebar status binding."""
        return {
            "refresh_every":     self.refresh_every,
            "primary_layer":     self.primary_layer,
            "record_history":    self.record_history,
            "show_polarization": self.show_polarization,
            "show_spatial":      self.show_spatial,
            "show_histogram":    self.show_histogram,
            "show_events":       self.show_events,
        }


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_dashboard_settings_page(
    lang:     str = "zh",
    defaults: dict[str, Any] | None = None,
) -> DashboardSettingsComponents:
    """
    Render the dashboard settings page inside the current Gradio context.

    Parameters
    ----------
    lang : {"zh", "en"}
    defaults : dict | None
        Override default values. Falls back to _DEFAULTS.

    Returns
    -------
    DashboardSettingsComponents
    """
    s  = _STRINGS[lang]
    dv = {**_DEFAULTS, **(defaults or {})}

    with gr.Column(elem_id="page-dashboard-settings", elem_classes="page-body"):

        # ── Page header ───────────────────────────────────────────────────
        gr.HTML(f"""
<div class="page-header" style="padding:20px 0 16px;">
  <div class="page-title">{s['page_title']}</div>
  <div class="page-subtitle">{s['page_subtitle']}</div>
</div>""")

        # ── Card 1: Live monitor settings ─────────────────────────────────
        with gr.Group(elem_classes="section-card"):
            gr.HTML(f'<div class="section-card-title">{s["card1_title"]}</div>')

            with gr.Row():
                refresh_every = gr.Number(
                    label       = s["refresh_label"],
                    info        = s["refresh_info"],
                    value       = dv["refresh_every"],
                    minimum     = 1,
                    maximum     = 500,
                    step        = 1,
                    precision   = 0,
                    interactive = True,
                    scale       = 1,
                )
                primary_layer = gr.Number(
                    label       = s["layer_label"],
                    info        = s["layer_info"],
                    value       = dv["primary_layer"],
                    minimum     = 0,
                    maximum     = 9,
                    step        = 1,
                    precision   = 0,
                    interactive = True,
                    scale       = 1,
                )

            record_history = gr.Checkbox(
                label       = s["history_label"],
                info        = s["history_info"],
                value       = dv["record_history"],
                interactive = True,
            )

        # ── Card 2: Chart visibility ──────────────────────────────────────
        with gr.Group(elem_classes="section-card"):
            gr.HTML(f'<div class="section-card-title">{s["card2_title"]}</div>')

            with gr.Row():
                show_polarization = gr.Checkbox(
                    label       = s["polar_label"],
                    value       = dv["show_polarization"],
                    interactive = True,
                    scale       = 1,
                )
                show_spatial = gr.Checkbox(
                    label       = s["spatial_label"],
                    value       = dv["show_spatial"],
                    interactive = True,
                    scale       = 1,
                )

            with gr.Row():
                show_histogram = gr.Checkbox(
                    label       = s["hist_label"],
                    value       = dv["show_histogram"],
                    interactive = True,
                    scale       = 1,
                )
                show_events = gr.Checkbox(
                    label       = s["events_label"],
                    value       = dv["show_events"],
                    interactive = True,
                    scale       = 1,
                )

        # ── Action row ────────────────────────────────────────────────────
        with gr.Row():
            save_btn = gr.Button(
                s["save_btn"],
                elem_id      = "btn-save-dashboard",
                elem_classes = "btn-primary-teal",
                size         = "sm",
            )
            reset_btn = gr.Button(
                s["reset_btn"],
                elem_id      = "btn-reset-dashboard",
                elem_classes = "btn-secondary",
                size         = "sm",
            )
            saved_md = gr.Markdown(
                value   = "",
                visible = False,
                elem_id = "dashboard-saved-msg",
            )

        # ── Save handler (inline — no navigation side-effects) ────────────
        def _on_save(*_args):
            return gr.update(value=s["saved_msg"], visible=True)

        save_btn.click(
            fn      = _on_save,
            inputs  = [],
            outputs = [saved_md],
        )

        # ── Reset handler ─────────────────────────────────────────────────
        def _on_reset():
            return (
                gr.update(value=_DEFAULTS["refresh_every"]),
                gr.update(value=_DEFAULTS["primary_layer"]),
                gr.update(value=_DEFAULTS["record_history"]),
                gr.update(value=_DEFAULTS["show_polarization"]),
                gr.update(value=_DEFAULTS["show_spatial"]),
                gr.update(value=_DEFAULTS["show_histogram"]),
                gr.update(value=_DEFAULTS["show_events"]),
                gr.update(value="", visible=False),
            )

        reset_btn.click(
            fn      = _on_reset,
            inputs  = [],
            outputs = [
                refresh_every, primary_layer, record_history,
                show_polarization, show_spatial, show_histogram,
                show_events, saved_md,
            ],
        )

    return DashboardSettingsComponents(
        refresh_every     = refresh_every,
        primary_layer     = primary_layer,
        record_history    = record_history,
        show_polarization = show_polarization,
        show_spatial      = show_spatial,
        show_histogram    = show_histogram,
        show_events       = show_events,
        save_btn          = save_btn,
        reset_btn         = reset_btn,
        saved_md          = saved_md,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_values(comps: DashboardSettingsComponents) -> dict[str, Any]:
    """
    Collect current gr.Component values into a flat dict.
    For use when assembling the full ui_values dict for validate_all().
    NOTE: In Gradio, you cannot read .value from components directly —
    this function returns the last-known Python-side values only.
    Use it only after a button-click handler has captured the inputs.
    """
    raise NotImplementedError(
        "Cannot read Gradio component values outside a handler. "
        "Capture inputs=[...] in a gr.Button.click() handler instead."
    )


def apply_defaults(
    comps: DashboardSettingsComponents,
    defaults: dict[str, Any] | None = None,
) -> list:
    """
    Return a list of gr.update values to restore default settings.
    Use as the return value of a handler:

        reset_btn.click(
            fn      = lambda: apply_defaults(dash_settings),
            outputs = [comps.refresh_every, ...],
        )
    """
    dv = {**_DEFAULTS, **(defaults or {})}
    return [
        gr.update(value=dv["refresh_every"]),
        gr.update(value=dv["primary_layer"]),
        gr.update(value=dv["record_history"]),
        gr.update(value=dv["show_polarization"]),
        gr.update(value=dv["show_spatial"]),
        gr.update(value=dv["show_histogram"]),
        gr.update(value=dv["show_events"]),
    ]
