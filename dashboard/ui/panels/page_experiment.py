"""
ui/panels/page_experiment.py

P6 实验运行 — 两阶段设计

阶段 A（参数确认）
─────────────────
  左栏：6 条 CheckItem 行 + 汇总 + 操作按钮
  右栏：关键参数快照（只读 key-value 表）

阶段 B（实时监控）
─────────────────
  状态行 + 5 个指标卡 + 2×2 图表网格

两阶段通过 gr.Group visible 切换，由 gr.State(phase) 控制。
所有重量级逻辑（runner 启动、图表刷新）保留在 app.py，
此文件只负责 UI 结构与纯展示逻辑。

Public API
──────────
ExperimentComponents
    Dataclass with all gr.Components.

build_experiment_page(lang, defaults) -> ExperimentComponents

render_checklist(items, lang) -> str
    Convert list[CheckItem] to HTML string for checklist_html.

render_snapshot(ui_values, lang) -> str
    Convert param dict to HTML key-value table for snapshot_html.

update_checklist(items, lang) -> tuple
    Return (checklist_html, summary_html, run_btn) gr.update tuple.

transition_to_monitor() -> tuple
transition_to_checklist() -> tuple
    Phase-switch gr.update tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr

# Lazy validator import — avoids hard dependency in isolated environments
try:
    from core.validator import STATUS_OK, STATUS_WARN, STATUS_ERROR, summarize
    _HAS_VALIDATOR = True
except ImportError:
    _HAS_VALIDATOR = False
    STATUS_OK    = "ok"
    STATUS_WARN  = "warn"
    STATUS_ERROR = "error"
    def summarize(items):  # noqa: E306
        return {"n_ok": 0, "n_warn": 0, "n_error": 0,
                "blocks_run": False, "progress": 0.0, "label_zh": ""}


# ─── Localised strings ────────────────────────────────────────────────────────

_S = {
    "zh": {
        "page_title":    "实验运行",
        "page_subtitle": "确认所有参数配置正确后，启动仿真。",
        "checklist_title": "配置检查",
        "snapshot_title":  "参数快照",
        "copy_btn":        "复制",
        "run_btn":         "▶ 确认并启动仿真",
        "export_btn":      "导出配置 YAML",
        "reconfigure_link":"← 重新配置",
        "ok_summary":      "全部 {n} 项配置通过，可以启动。",
        "warn_summary":    "{n_ok} 项通过，{n_warn} 项警告，建议检查后再启动。",
        "error_summary":   "存在 {n_error} 项错误，请修复后再启动。",
        "go_fix":          "去修改",
        "must_fix":        "必须修改",
        "status_idle":     "就绪",
        "status_running":  "运行中",
        "status_complete": "已完成",
        "status_stopped":  "已停止",
        "pause_btn":       "⏸ 暂停",
        "resume_btn":      "▶ 继续",
        "stop_btn":        "⏹ 停止",
        "view_results_btn":"查看结果 →",
        "m_step":     "步骤",
        "m_time":     "耗时(s)",
        "m_events":   "事件数",
        "m_polar":    "极化度",
        "m_clusters": "意见簇",
        "chart_polar":   "极化时序",
        "chart_spatial": "空间分布",
        "chart_hist":    "意见直方图",
        "chart_events":  "事件时序",
        "snap_keys": {
            "num_agents":    "智能体数",
            "total_steps":   "总步数",
            "opinion_layers":"意见层数",
            "seed":          "随机种子",
            "epsilon_base":  "ε 信任阈值",
            "mu_base":       "μ 更新速率",
            "alpha_mod":     "α 事件增益",
            "net_type":      "网络类型",
            "sw_k":          "SW 近邻数",
            "n_clusters":    "空间簇数",
            "output_dir":    "输出目录",
            "output_lang":   "报告语言",
        },
    },
    "en": {
        "page_title":    "Run Experiment",
        "page_subtitle": "Review all parameters then launch the simulation.",
        "checklist_title": "Config Checklist",
        "snapshot_title":  "Parameter Snapshot",
        "copy_btn":        "Copy",
        "run_btn":         "▶ Confirm & Run",
        "export_btn":      "Export Config YAML",
        "reconfigure_link":"← Reconfigure",
        "ok_summary":      "All {n} checks passed. Ready to run.",
        "warn_summary":    "{n_ok} passed, {n_warn} warning(s).",
        "error_summary":   "{n_error} error(s). Fix before running.",
        "go_fix":          "Fix",
        "must_fix":        "Must fix",
        "status_idle":     "Ready",
        "status_running":  "Running",
        "status_complete": "Complete",
        "status_stopped":  "Stopped",
        "pause_btn":       "⏸ Pause",
        "resume_btn":      "▶ Resume",
        "stop_btn":        "⏹ Stop",
        "view_results_btn":"View Results →",
        "m_step":     "Step",
        "m_time":     "Time(s)",
        "m_events":   "Events",
        "m_polar":    "Polarisation",
        "m_clusters": "Clusters",
        "chart_polar":   "Polarisation Series",
        "chart_spatial": "Spatial Distribution",
        "chart_hist":    "Opinion Histogram",
        "chart_events":  "Event Series",
        "snap_keys": {
            "num_agents":    "Agents",
            "total_steps":   "Steps",
            "opinion_layers":"Opinion layers",
            "seed":          "Seed",
            "epsilon_base":  "ε (trust threshold)",
            "mu_base":       "μ (update rate)",
            "alpha_mod":     "α (event gain)",
            "net_type":      "Network type",
            "sw_k":          "SW neighbours",
            "n_clusters":    "Spatial clusters",
            "output_dir":    "Output directory",
            "output_lang":   "Report language",
        },
    },
}

_SNAPSHOT_KEYS = [
    "num_agents", "total_steps", "opinion_layers", "seed",
    "epsilon_base", "mu_base", "alpha_mod",
    "net_type", "sw_k", "n_clusters",
    "output_dir", "output_lang",
]


# ─── HTML renderers ───────────────────────────────────────────────────────────

def render_checklist(items: list, lang: str = "zh") -> str:
    """
    Convert list[CheckItem] → HTML string for checklist_html.

    Each item maps to a .check-row div styled by status.
    Fix links carry data-page attr; JS in app.py forwards clicks
    to the matching sidebar nav button.
    """
    s = _S.get(lang, _S["zh"])
    _ICONS = {
        STATUS_OK:    ('✓', "ok"),
        STATUS_WARN:  ('!', "warn"),
        STATUS_ERROR: ('✕', "error"),
    }
    rows = ""
    for item in items:
        status = getattr(item, "status", STATUS_OK)
        title  = getattr(item, "title",  "—")
        detail = getattr(item, "detail", "")
        page   = getattr(item, "target_page", "model_config")

        icon_char, css_cls = _ICONS.get(status, _ICONS[STATUS_OK])
        icon_html = (
            f'<span style="font-size:12px;font-weight:600;">{icon_char}</span>'
        )

        if status == STATUS_WARN:
            fix_link = (
                f'<span class="check-fix-link warn" data-page="{page}">'
                f'{s["go_fix"]}</span>'
            )
        elif status == STATUS_ERROR:
            fix_link = (
                f'<span class="check-fix-link error" data-page="{page}">'
                f'{s["must_fix"]}</span>'
            )
        else:
            fix_link = ""

        rows += (
            f'<div class="check-row {css_cls}">'
            f'  <div class="check-icon {css_cls}">{icon_html}</div>'
            f'  <div class="check-row-body">'
            f'    <div class="check-row-title">{title}</div>'
            f'    <div class="check-row-detail">{detail}</div>'
            f'  </div>'
            f'  {fix_link}'
            f'</div>'
        )

    placeholder = (
        f'<div style="padding:20px 0;text-align:center;'
        f'font-size:12px;color:#94a3b8;">—</div>'
    ) if not rows else ""

    return f'<div id="checklist-area">{rows}{placeholder}</div>'


def render_snapshot(ui_values: dict, lang: str = "zh") -> str:
    """Render the parameter snapshot key-value table as HTML."""
    s    = _S.get(lang, _S["zh"])
    keys = s["snap_keys"]
    rows = ""
    for k in _SNAPSHOT_KEYS:
        label = keys.get(k, k)
        raw   = ui_values.get(k)
        if raw is None:
            val = "—"
        elif isinstance(raw, float):
            val = f"{raw:.3f}".rstrip("0").rstrip(".")
        else:
            val = str(raw)
        warn = val in ("", "—", "0", "None")
        val_style = "color:#d97706;" if warn else ""
        rows += (
            f'<div class="snapshot-row">'
            f'  <span class="snapshot-key">{label}</span>'
            f'  <span class="snapshot-value" style="{val_style}">{val}</span>'
            f'</div>'
        )
    header = (
        f'<div style="display:flex;align-items:center;'
        f'justify-content:space-between;margin-bottom:8px;">'
        f'  <span style="font-size:11px;font-weight:500;color:#64748b;">'
        f'    {s["snapshot_title"]}'
        f'    <span style="color:#94a3b8;">({len(_SNAPSHOT_KEYS)})</span>'
        f'  </span>'
        f'  <button class="snapshot-copy-btn" '
        f'          onclick="copySnapshot(this,\'snapshot-table\')">'
        f'    {s["copy_btn"]}'
        f'  </button>'
        f'</div>'
    )
    return (
        f'{header}'
        f'<div id="param-snapshot" id="snapshot-table">{rows}</div>'
    )


def _run_summary_html(items: list, lang: str) -> str:
    s  = _S.get(lang, _S["zh"])
    sm = summarize(items)
    if sm["n_error"]:
        cls = "error"
        txt = s["error_summary"].format(n_error=sm["n_error"])
    elif sm["n_warn"]:
        cls = "warn"
        txt = s["warn_summary"].format(n_ok=sm["n_ok"], n_warn=sm["n_warn"])
    elif sm["n_ok"]:
        cls = "ok"
        txt = s["ok_summary"].format(n=sm["n_ok"])
    else:
        return ""
    return f'<div class="run-summary-bar {cls}">{txt}</div>'


# ─── ExperimentComponents ─────────────────────────────────────────────────────

@dataclass
class ExperimentComponents:
    # Phase A
    checklist_html:   gr.HTML
    snapshot_html:    gr.HTML
    summary_html:     gr.HTML
    run_btn:          gr.Button
    export_btn:       gr.Button
    phase_a_group:    gr.Group
    # Phase B
    status_md:        gr.Markdown
    metric_step:      gr.Number
    metric_time:      gr.Number
    metric_events:    gr.Number
    metric_polar:     gr.Number
    metric_clusters:  gr.Number
    plot_polar:       gr.Plot
    plot_spatial:     gr.Plot
    plot_hist:        gr.Plot
    plot_events:      gr.Plot
    pause_btn:        gr.Button
    stop_btn:         gr.Button
    view_results_btn: gr.Button
    reconfigure_btn:  gr.Button
    reconfigure_html: gr.HTML
    phase_b_group:    gr.Group
    # Shared
    phase_state:      gr.State


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_experiment_page(
    lang: str = "zh",
    defaults: dict[str, Any] | None = None,
) -> ExperimentComponents:
    """
    Render P6 inside the current gr.Blocks context.
    Call inside a gr.Group(visible=...) wrapper in app.py.
    """
    s  = _S.get(lang, _S["zh"])
    dv = defaults or {}

    # Page header
    gr.HTML(
        f'<div class="page-header">'
        f'  <div class="page-title">{s["page_title"]}</div>'
        f'  <div class="page-subtitle">{s["page_subtitle"]}</div>'
        f'</div>'
    )

    # ══ Phase A ═══════════════════════════════════════════════════════════
    with gr.Group(visible=True, elem_id="phase-a-group") as phase_a_group:
        with gr.Row(equal_height=False, elem_classes="page-body"):

            # Left: checklist + run buttons
            with gr.Column(scale=3, min_width=300):
                gr.HTML(
                    f'<div style="font-size:11px;font-weight:500;color:#64748b;'
                    f'text-transform:uppercase;letter-spacing:.06em;'
                    f'margin-bottom:10px;">'
                    f'{s["checklist_title"]}</div>'
                )
                checklist_html = gr.HTML(
                    value   = render_checklist([], lang),
                    elem_id = "checklist-html",
                )
                summary_html = gr.HTML(
                    value   = "",
                    elem_id = "run-summary-html",
                )
                with gr.Row():
                    run_btn = gr.Button(
                        s["run_btn"],
                        elem_id      = "btn-run",
                        elem_classes = "btn-primary-teal",
                        size         = "lg",
                        interactive  = False,
                    )
                with gr.Row():
                    export_btn = gr.Button(
                        s["export_btn"],
                        elem_id      = "btn-export-yaml",
                        elem_classes = "btn-secondary",
                        size         = "sm",
                    )

            # Right: snapshot
            with gr.Column(scale=2, min_width=200):
                snapshot_html = gr.HTML(
                    value   = render_snapshot(dv, lang),
                    elem_id = "snapshot-html",
                )

    # ══ Phase B ═══════════════════════════════════════════════════════════
    with gr.Group(visible=False, elem_id="phase-b-group") as phase_b_group:

        # Status bar
        with gr.Row(elem_id="monitor-status-bar"):
            reconfigure_btn = gr.Button(
                s["reconfigure_link"],
                elem_id="btn-reconfigure",
                elem_classes="btn-secondary",
                size="sm",
                visible=True,
            )
            reconfigure_html = gr.HTML(
                value = (
                    f'<span id="reconfigure-link" '
                    f'style="font-size:12px;color:#64748b;'
                    f'text-decoration:underline;cursor:pointer;">'
                    f'{s["reconfigure_link"]}</span>'
                ),
            )
            status_md = gr.Markdown(
                value   = f"**{s['status_idle']}**",
                elem_id = "monitor-status-md",
            )
            gr.HTML('<div style="flex:1;"></div>')
            pause_btn = gr.Button(
                s["pause_btn"],
                elem_id      = "btn-pause",
                elem_classes = "btn-secondary",
                size         = "sm",
                interactive  = False,
            )
            stop_btn = gr.Button(
                s["stop_btn"],
                elem_id      = "btn-stop",
                size         = "sm",
                interactive  = False,
            )

        # Metric cards
        with gr.Row(elem_id="metric-card-row"):
            metric_step = gr.Number(
                label=s["m_step"], value=0,
                interactive=False, precision=0,
                elem_classes="metric-card",
            )
            metric_time = gr.Number(
                label=s["m_time"], value=0.0,
                interactive=False, precision=1,
                elem_classes="metric-card",
            )
            metric_events = gr.Number(
                label=s["m_events"], value=0,
                interactive=False, precision=0,
                elem_classes="metric-card",
            )
            metric_polar = gr.Number(
                label=s["m_polar"], value=0.0,
                interactive=False, precision=3,
                elem_classes="metric-card",
            )
            metric_clusters = gr.Number(
                label=s["m_clusters"], value=0,
                interactive=False, precision=0,
                elem_classes="metric-card",
            )

        # Chart grid row 1
        with gr.Row():
            with gr.Column(elem_classes="chart-tile", scale=1):
                gr.HTML(f'<div class="label">{s["chart_polar"]}</div>')
                plot_polar = gr.Plot(
                    label=s["chart_polar"], show_label=False,
                    elem_id="plot-polar",
                )
            with gr.Column(elem_classes="chart-tile", scale=1):
                gr.HTML(f'<div class="label">{s["chart_spatial"]}</div>')
                plot_spatial = gr.Plot(
                    label=s["chart_spatial"], show_label=False,
                    elem_id="plot-spatial",
                )

        # Chart grid row 2
        with gr.Row():
            with gr.Column(elem_classes="chart-tile", scale=1):
                gr.HTML(f'<div class="label">{s["chart_hist"]}</div>')
                plot_hist = gr.Plot(
                    label=s["chart_hist"], show_label=False,
                    elem_id="plot-hist",
                )
            with gr.Column(elem_classes="chart-tile", scale=1):
                gr.HTML(f'<div class="label">{s["chart_events"]}</div>')
                plot_events = gr.Plot(
                    label=s["chart_events"], show_label=False,
                    elem_id="plot-events",
                )

        # Post-run navigation
        with gr.Row():
            view_results_btn = gr.Button(
                s["view_results_btn"],
                elem_id      = "btn-view-results",
                elem_classes = "btn-primary-teal",
                size         = "lg",
                interactive  = False,
                visible      = False,
            )

    phase_state = gr.State(value="A")

    return ExperimentComponents(
        checklist_html   = checklist_html,
        snapshot_html    = snapshot_html,
        summary_html     = summary_html,
        run_btn          = run_btn,
        export_btn       = export_btn,
        phase_a_group    = phase_a_group,
        status_md        = status_md,
        metric_step      = metric_step,
        metric_time      = metric_time,
        metric_events    = metric_events,
        metric_polar     = metric_polar,
        metric_clusters  = metric_clusters,
        plot_polar       = plot_polar,
        plot_spatial     = plot_spatial,
        plot_hist        = plot_hist,
        plot_events      = plot_events,
        pause_btn        = pause_btn,
        stop_btn         = stop_btn,
        view_results_btn = view_results_btn,
        reconfigure_btn  = reconfigure_btn,
        reconfigure_html = reconfigure_html,
        phase_b_group    = phase_b_group,
        phase_state      = phase_state,
    )


# ─── Phase transition helpers ─────────────────────────────────────────────────

def transition_to_monitor() -> tuple:
    """
    gr.update tuple to switch from phase A to phase B.

    Outputs order: phase_a_group, phase_b_group,
                   run_btn, pause_btn, stop_btn
    """
    return (
        gr.update(visible=False),    # phase_a_group
        gr.update(visible=True),     # phase_b_group
        gr.update(interactive=False),   # run_btn
        gr.update(interactive=True),    # pause_btn
        gr.update(interactive=True),    # stop_btn
    )


def transition_to_checklist() -> tuple:
    """
    gr.update tuple to switch from phase B back to phase A.

    Outputs order: phase_a_group, phase_b_group,
                   pause_btn, stop_btn
    """
    return (
        gr.update(visible=True),     # phase_a_group
        gr.update(visible=False),    # phase_b_group
        gr.update(interactive=False),   # pause_btn
        gr.update(interactive=False),   # stop_btn
    )


def on_run_complete(lang: str = "zh") -> tuple:
    """
    Called when the simulation finishes normally.

    Outputs order: status_md, pause_btn, stop_btn, view_results_btn
    """
    s = _S.get(lang, _S["zh"])
    return (
        gr.update(value=f"**{s['status_complete']}**"),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True, visible=True),
    )


def update_checklist(items: list, lang: str = "zh") -> tuple:
    """
    Return (checklist_html, summary_html, run_btn) gr.update values.

    run_btn is enabled only when there are no ERROR items and
    at least one item was returned.
    """
    c_val = render_checklist(items, lang)
    s_val = _run_summary_html(items, lang)
    has_error = any(
        getattr(it, "status", "") == STATUS_ERROR for it in items
    )
    run_ok = (not has_error) and (len(items) > 0)
    return (
        gr.update(value=c_val),
        gr.update(value=s_val),
        gr.update(interactive=run_ok),
    )
