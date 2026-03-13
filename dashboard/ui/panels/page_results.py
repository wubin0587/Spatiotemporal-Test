"""
ui/panels/page_results.py

P7 实验结果 — 合并 analysis_tab + report_tab

四个子标签（pill 样式导航）：
  1. 动态分析   — 最终运行的 4 张图表（只读）
  2. 静态仪表盘 — render_dashboard() 合成图
  3. 特征摘要   — key-value 特征表
  4. AI 报告    — Markdown 预览 + 下载

Public API
──────────
ResultsComponents
    Dataclass with all gr.Components.

build_results_page(lang) -> ResultsComponents

render_features_table(features, lang) -> str

switch_subtab(target) -> tuple
    gr.update tuple for sub-tab group + pill button class updates.

populate_from_run(...) -> tuple
    Fills metrics + plots after a simulation completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr


# ─── Strings ──────────────────────────────────────────────────────────────────

_S = {
    "zh": {
        "page_title":    "实验结果",
        "page_subtitle": "查看本次仿真的完整分析结果。",
        "status_done":   "✓ 已完成",
        "status_none":   "尚未运行",
        "m_step":     "总步数",
        "m_time":     "耗时(s)",
        "m_events":   "事件总数",
        "m_polar":    "最终极化度",
        "m_clusters": "最终意见簇",
        "tab_dynamic":   "动态分析",
        "tab_static":    "静态仪表盘",
        "tab_features":  "特征摘要",
        "tab_report":    "AI 报告",
        "chart_polar":   "极化时序",
        "chart_spatial": "空间分布",
        "chart_hist":    "意见直方图",
        "chart_events":  "事件时序",
        "run_analysis_btn": "运行分析",
        "analysis_hint":    "首次进入此标签时将自动运行静态分析。",
        "features_count":   "共 {n} 项特征",
        "features_empty":   "暂无特征数据 — 请先运行分析。",
        "report_gen_btn":   "📄 生成 AI 报告",
        "report_no_ai":     "未启用 AI 解析",
        "report_no_ai_hint":"前往「分析配置」页开启 AI 解析并填写 API Key。",
        "report_goto_link": "前往分析配置 →",
        "report_empty":     "报告尚未生成 — 点击上方按钮开始。",
        "dl_figures_btn":   "⬇ 下载图表包",
        "dl_features_btn":  "⬇ 导出特征 JSON",
        "gen_report_btn":   "📄 生成报告",
    },
    "en": {
        "page_title":    "Results",
        "page_subtitle": "Review the complete analysis for this simulation run.",
        "status_done":   "✓ Complete",
        "status_none":   "Not yet run",
        "m_step":     "Steps",
        "m_time":     "Time(s)",
        "m_events":   "Events",
        "m_polar":    "Final Polar.",
        "m_clusters": "Final Clusters",
        "tab_dynamic":   "Dynamic Analysis",
        "tab_static":    "Static Dashboard",
        "tab_features":  "Feature Summary",
        "tab_report":    "AI Report",
        "chart_polar":   "Polarisation Series",
        "chart_spatial": "Spatial Distribution",
        "chart_hist":    "Opinion Histogram",
        "chart_events":  "Event Series",
        "run_analysis_btn": "Run Analysis",
        "analysis_hint":    "Static analysis runs automatically on first visit.",
        "features_count":   "{n} features",
        "features_empty":   "No feature data — run analysis first.",
        "report_gen_btn":   "📄 Generate AI Report",
        "report_no_ai":     "AI analysis not enabled",
        "report_no_ai_hint":"Go to Analysis Config to enable AI and add your API key.",
        "report_goto_link": "Go to Analysis Config →",
        "report_empty":     "Report not yet generated — click the button above.",
        "dl_figures_btn":   "⬇ Download Figures",
        "dl_features_btn":  "⬇ Export Features JSON",
        "gen_report_btn":   "📄 Generate Report",
    },
}

_SUBTABS = ["dynamic", "static", "features", "report"]


# ─── HTML renderers ───────────────────────────────────────────────────────────

def render_features_table(features: dict[str, Any], lang: str = "zh") -> str:
    """Convert flat {key: value} features dict → HTML table."""
    s = _S.get(lang, _S["zh"])
    if not features:
        return (
            f'<div style="padding:24px 0;text-align:center;'
            f'font-size:12px;color:#94a3b8;">'
            f'{s["features_empty"]}</div>'
        )
    rows = ""
    for k, v in features.items():
        vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows += (
            f'<tr>'
            f'  <td style="color:#64748b;padding:5px 12px;'
            f'      border-bottom:0.5px solid #f1f5f9;">{k}</td>'
            f'  <td style="color:#0f172a;padding:5px 12px;'
            f'      border-bottom:0.5px solid #f1f5f9;text-align:right;">'
            f'      {vstr}</td>'
            f'</tr>'
        )
    count_label = s["features_count"].format(n=len(features))
    return (
        f'<div style="font-size:11px;color:#94a3b8;margin-bottom:8px;">'
        f'{count_label}</div>'
        f'<table id="feature-table" style="width:100%;border-collapse:collapse;'
        f'font-family:IBM Plex Mono,monospace;font-size:12px;">'
        f'<thead><tr>'
        f'  <th style="background:#f8fafc;color:#64748b;font-weight:500;'
        f'      font-size:11px;text-transform:uppercase;letter-spacing:.04em;'
        f'      padding:7px 12px;text-align:left;border-bottom:0.5px solid #e2e8f0;">'
        f'      Metric</th>'
        f'  <th style="background:#f8fafc;color:#64748b;font-weight:500;'
        f'      font-size:11px;text-transform:uppercase;letter-spacing:.04em;'
        f'      padding:7px 12px;text-align:right;border-bottom:0.5px solid #e2e8f0;">'
        f'      Value</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _report_empty_html(lang: str) -> str:
    s = _S.get(lang, _S["zh"])
    return (
        f'<div class="report-empty-state">'
        f'  <span style="font-size:24px;opacity:.3;">📄</span>'
        f'  <span>{s["report_empty"]}</span>'
        f'</div>'
    )


def _report_no_ai_html(lang: str) -> str:
    s = _S.get(lang, _S["zh"])
    return (
        f'<div class="report-empty-state">'
        f'  <span style="font-size:22px;opacity:.3;">⚠</span>'
        f'  <span style="font-weight:500;">{s["report_no_ai"]}</span>'
        f'  <span style="font-size:12px;color:#94a3b8;">{s["report_no_ai_hint"]}</span>'
        f'  <span class="check-fix-link warn" data-page="analysis_config" '
        f'    style="cursor:pointer;">{s["report_goto_link"]}</span>'
        f'</div>'
    )


# ─── ResultsComponents ────────────────────────────────────────────────────────

@dataclass
class ResultsComponents:
    # Header
    status_html:      gr.HTML
    metric_step:      gr.Number
    metric_time:      gr.Number
    metric_events:    gr.Number
    metric_polar:     gr.Number
    metric_clusters:  gr.Number
    dl_figures_btn:   gr.Button
    dl_features_btn:  gr.Button
    gen_report_btn:   gr.Button
    # Sub-tab pills
    tab_btns:         list         # list[gr.Button], in _SUBTABS order
    # Dynamic
    dynamic_group:    gr.Group
    plot_polar:       gr.Plot
    plot_spatial:     gr.Plot
    plot_hist:        gr.Plot
    plot_events:      gr.Plot
    # Static
    static_group:     gr.Group
    dashboard_img:    gr.Image
    run_analysis_btn: gr.Button
    # Features
    features_group:   gr.Group
    features_html:    gr.HTML
    # Report
    report_group:         gr.Group
    report_html:          gr.HTML
    gen_report_btn_inner: gr.Button
    report_download_btn:  gr.Button
    # Shared
    active_tab:       gr.State


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_results_page(lang: str = "zh") -> ResultsComponents:
    """Render P7 inside the current gr.Blocks context."""
    s = _S.get(lang, _S["zh"])

    # Page header + action buttons
    with gr.Row(elem_classes="page-header"):
        with gr.Column(scale=3):
            gr.HTML(
                f'<div class="page-title">{s["page_title"]}</div>'
                f'<div class="page-subtitle">{s["page_subtitle"]}</div>'
            )
        with gr.Column(scale=2, min_width=280):
            with gr.Row():
                dl_figures_btn = gr.Button(
                    s["dl_figures_btn"],
                    elem_id="btn-dl-figures",
                    elem_classes="btn-secondary",
                    size="sm", interactive=False,
                )
                dl_features_btn = gr.Button(
                    s["dl_features_btn"],
                    elem_id="btn-dl-features",
                    elem_classes="btn-secondary",
                    size="sm", interactive=False,
                )
                gen_report_btn = gr.Button(
                    s["gen_report_btn"],
                    elem_id="btn-gen-report",
                    elem_classes="btn-secondary",
                    size="sm", interactive=False,
                )

    # Status capsule
    status_html = gr.HTML(
        value=(
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'padding:3px 10px;border-radius:12px;border:0.5px solid #e2e8f0;'
            f'background:#f8fafc;font-size:11px;color:#94a3b8;'
            f'margin:4px 24px 8px;">'
            f'{s["status_none"]}</div>'
        ),
        elem_id="results-status-html",
    )

    # Metric cards
    with gr.Row(equal_height=True, elem_classes="page-body"):
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

    # Sub-tab pill nav
    _TAB_LABELS = {
        "dynamic":  s["tab_dynamic"],
        "static":   s["tab_static"],
        "features": s["tab_features"],
        "report":   s["tab_report"],
    }
    tab_btns: list[gr.Button] = []
    with gr.Row(elem_id="results-subtabs"):
        for i, key in enumerate(_SUBTABS):
            btn = gr.Button(
                _TAB_LABELS[key],
                elem_id      = f"results-tab-{key}",
                elem_classes = "results-tab-btn active" if i == 0
                               else "results-tab-btn",
                size         = "sm",
            )
            tab_btns.append(btn)

    # ── Tab 0: Dynamic ────────────────────────────────────────────────────
    with gr.Group(visible=True, elem_id="tab-dynamic") as dynamic_group:
        with gr.Row(elem_classes="page-body"):
            with gr.Column(elem_classes="chart-tile", scale=1):
                gr.HTML(f'<div class="label">{s["chart_polar"]}</div>')
                plot_polar = gr.Plot(
                    label=s["chart_polar"], show_label=False,
                    elem_id="res-plot-polar",
                )
            with gr.Column(elem_classes="chart-tile", scale=1):
                gr.HTML(f'<div class="label">{s["chart_spatial"]}</div>')
                plot_spatial = gr.Plot(
                    label=s["chart_spatial"], show_label=False,
                    elem_id="res-plot-spatial",
                )
        with gr.Row(elem_classes="page-body"):
            with gr.Column(elem_classes="chart-tile", scale=1):
                gr.HTML(f'<div class="label">{s["chart_hist"]}</div>')
                plot_hist = gr.Plot(
                    label=s["chart_hist"], show_label=False,
                    elem_id="res-plot-hist",
                )
            with gr.Column(elem_classes="chart-tile", scale=1):
                gr.HTML(f'<div class="label">{s["chart_events"]}</div>')
                plot_events = gr.Plot(
                    label=s["chart_events"], show_label=False,
                    elem_id="res-plot-events",
                )

    # ── Tab 1: Static Dashboard ───────────────────────────────────────────
    with gr.Group(visible=False, elem_id="tab-static") as static_group:
        with gr.Column(elem_classes="page-body"):
            gr.HTML(
                f'<div style="font-size:12px;color:#94a3b8;'
                f'margin-bottom:10px;">{s["analysis_hint"]}</div>'
            )
            run_analysis_btn = gr.Button(
                s["run_analysis_btn"],
                elem_id="btn-run-analysis",
                elem_classes="btn-secondary",
                size="sm",
            )
            dashboard_img = gr.Image(
                label="", show_label=False,
                interactive=False,
                elem_id="dashboard-img",
                type="pil",
            )

    # ── Tab 2: Feature Summary ────────────────────────────────────────────
    with gr.Group(visible=False, elem_id="tab-features") as features_group:
        with gr.Column(elem_classes="page-body"):
            features_html = gr.HTML(
                value=render_features_table({}, lang),
                elem_id="features-html",
            )

    # ── Tab 3: AI Report ──────────────────────────────────────────────────
    with gr.Group(visible=False, elem_id="tab-report") as report_group:
        with gr.Column(elem_classes="page-body"):
            gen_report_btn_inner = gr.Button(
                s["report_gen_btn"],
                elem_id      = "btn-gen-report-inner",
                elem_classes = "btn-primary-teal",
                size         = "lg",
                interactive  = False,
            )
            report_html = gr.HTML(
                value   = _report_empty_html(lang),
                elem_id = "report-preview-area",
            )
            report_download_btn = gr.Button(
                "⬇ Download",
                elem_id      = "btn-download-report",
                elem_classes = "btn-secondary",
                size         = "sm",
                visible      = False,
            )

    active_tab = gr.State(value="dynamic")

    return ResultsComponents(
        status_html         = status_html,
        metric_step         = metric_step,
        metric_time         = metric_time,
        metric_events       = metric_events,
        metric_polar        = metric_polar,
        metric_clusters     = metric_clusters,
        dl_figures_btn      = dl_figures_btn,
        dl_features_btn     = dl_features_btn,
        gen_report_btn      = gen_report_btn,
        tab_btns            = tab_btns,
        dynamic_group       = dynamic_group,
        plot_polar          = plot_polar,
        plot_spatial        = plot_spatial,
        plot_hist           = plot_hist,
        plot_events         = plot_events,
        static_group        = static_group,
        dashboard_img       = dashboard_img,
        run_analysis_btn    = run_analysis_btn,
        features_group      = features_group,
        features_html       = features_html,
        report_group        = report_group,
        report_html         = report_html,
        gen_report_btn_inner = gen_report_btn_inner,
        report_download_btn  = report_download_btn,
        active_tab          = active_tab,
    )


# ─── Sub-tab switching ────────────────────────────────────────────────────────

def switch_subtab(target: str) -> tuple:
    """
    Return gr.update tuple to switch the active sub-tab.

    Outputs order (8 items):
      dynamic_group, static_group, features_group, report_group,
      tab_btn[dynamic], tab_btn[static], tab_btn[features], tab_btn[report]

    Typical usage in app.py::

        for btn, key in zip(results.tab_btns, _SUBTABS):
            btn.click(
                fn      = lambda k=key: switch_subtab(k),
                outputs = [
                    results.dynamic_group, results.static_group,
                    results.features_group, results.report_group,
                ] + results.tab_btns,
            )
    """
    group_updates = [gr.update(visible=(k == target)) for k in _SUBTABS]
    btn_updates   = [
        gr.update(
            elem_classes="results-tab-btn active" if k == target
                         else "results-tab-btn"
        )
        for k in _SUBTABS
    ]
    return tuple(group_updates + btn_updates)


# ─── Post-run population ─────────────────────────────────────────────────────

def populate_from_run(
    step:     int,
    time_s:   float,
    events:   int,
    polar:    float,
    clusters: int,
    lang:     str = "zh",
    plots:    dict | None = None,
) -> tuple:
    """
    Fill all result components once the simulation finishes.

    Parameters
    ----------
    plots : dict | None
        Keys: "polar", "spatial", "hist", "events" — matplotlib Figure objects.
        If provided, 4 extra gr.update values are appended.

    Returns
    -------
    tuple
        Without plots : 9 gr.update values
            (status_html, step, time, events, polar, clusters,
             dl_figures_btn, dl_features_btn, gen_report_btn)
        With plots    : 13 gr.update values (above + 4 plot updates)
    """
    s = _S.get(lang, _S["zh"])
    status_val = (
        f'<div style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:3px 10px;border-radius:12px;border:0.5px solid #bbf7d0;'
        f'background:#f0fdf4;font-size:11px;color:#16a34a;'
        f'margin:4px 24px 8px;">'
        f'{s["status_done"]}</div>'
    )
    base = (
        gr.update(value=status_val),
        gr.update(value=step),
        gr.update(value=time_s),
        gr.update(value=events),
        gr.update(value=polar),
        gr.update(value=clusters),
        gr.update(interactive=True),   # dl_figures_btn
        gr.update(interactive=True),   # dl_features_btn
        gr.update(interactive=True),   # gen_report_btn
    )
    if plots:
        return base + (
            gr.update(value=plots.get("polar")),
            gr.update(value=plots.get("spatial")),
            gr.update(value=plots.get("hist")),
            gr.update(value=plots.get("events")),
        )
    return base
