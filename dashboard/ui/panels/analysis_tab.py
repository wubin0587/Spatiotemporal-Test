"""
ui/panels/analysis_tab.py

Post-run Analysis tab for the Opinion Dynamics Simulation Dashboard.

Displays the four-panel simulation dashboard figure, a scrollable
feature-summary DataFrame, a Run Analysis button, and a Download
Figures button.

Public API
----------
AnalysisTab
    Dataclass holding all gr.Components for the tab.

build_analysis_tab(lang) -> AnalysisTab
    Render the tab contents in the current Gradio context.

activate_analysis_tab(runner, ui_values) -> tuple
    Run the full analysis pipeline on a completed SimulationRunner
    and return the (figure, dataframe, status_str) tuple suitable
    for use as the return value of the ``run_analysis_btn`` click
    handler.

    Parameters
    ----------
    runner     : SimulationRunner   — must have runner.engine set
    ui_values  : dict               — flat param dict from the panel

    Returns
    -------
    tuple[Figure | None, pd.DataFrame | None, str]
        (dashboard_figure, summary_df, status_markdown)

    Raises
    ------
    Does NOT raise — all exceptions are caught and returned as
    a status string so Gradio can display them gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisTab dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisTab:
    """Holds all gr.Components for the Analysis tab."""

    dashboard_plot:   gr.Plot
    summary_df:       gr.DataFrame
    run_analysis_btn: gr.Button
    download_btn:     gr.DownloadButton
    analysis_status:  gr.Markdown

    # ── Convenience ──────────────────────────────────────────────────────

    def output_list(self) -> list:
        """Outputs list for ``run_analysis_btn.click(outputs=...)``."""
        return [
            self.dashboard_plot,
            self.summary_df,
            self.analysis_status,
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_tab(lang: str = "en") -> AnalysisTab:
    """
    Render the Analysis tab contents inside the current Gradio context.

    Must be called inside an active ``gr.Tab()`` or ``gr.Blocks()`` context.

    Parameters
    ----------
    lang : {"en", "zh"}

    Returns
    -------
    AnalysisTab
    """
    _t = lambda en, zh: zh if lang == "zh" else en

    # ── Dashboard figure ──────────────────────────────────────────────────
    dashboard_plot = gr.Plot(
        label=_t("Simulation Dashboard", "仿真概览"),
    )

    # ── Feature summary table ─────────────────────────────────────────────
    summary_df = gr.DataFrame(
        label=_t("Feature Summary", "特征摘要"),
        elem_id="summary-df",
        wrap=True,
    )

    # ── Controls row ──────────────────────────────────────────────────────
    with gr.Row(elem_id="analysis-controls"):
        run_analysis_btn = gr.Button(
            _t("▶  Run Analysis", "▶  运行分析"),
            variant="primary",
            size="sm",
        )
        download_btn = gr.DownloadButton(
            _t("⬇  Download Figures", "⬇  下载图表"),
            size="sm",
        )

    # ── Status ────────────────────────────────────────────────────────────
    analysis_status = gr.Markdown(
        "",
        elem_id="analysis-status",
    )

    return AnalysisTab(
        dashboard_plot=dashboard_plot,
        summary_df=summary_df,
        run_analysis_btn=run_analysis_btn,
        download_btn=download_btn,
        analysis_status=analysis_status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# activate_analysis_tab
# ─────────────────────────────────────────────────────────────────────────────

def activate_analysis_tab(runner: Any, ui_values: dict) -> tuple:
    """
    Run the analysis pipeline and produce all outputs for the Analysis tab.

    This is the handler function to bind to ``run_analysis_btn.click``:

        analysis_tab.run_analysis_btn.click(
            fn=lambda runner, *vals: activate_analysis_tab(
                runner, dict(zip(param_keys, vals))
            ),
            inputs=[runner_state] + param_inputs,
            outputs=analysis_tab.output_list(),
        )

    Parameters
    ----------
    runner : SimulationRunner
        Must have `runner.engine` set (i.e. simulation has been run).
    ui_values : dict
        Flat dict of all parameter values, as produced by the param panel.

    Returns
    -------
    tuple[Figure | None, pd.DataFrame | None, str]
        Matching ``analysis_tab.output_list()`` order:
        (dashboard_plot, summary_df, analysis_status)
    """
    import pandas as pd

    _no_sim_en = "⚠  No completed simulation — run the simulation first."
    _no_sim_zh = "⚠  无已完成的仿真 — 请先运行仿真。"

    if runner is None or runner.engine is None:
        lang = ui_values.get("output_lang", "en")
        msg  = _no_sim_zh if lang == "zh" else _no_sim_en
        return None, None, msg

    try:
        from core.config_bridge import build_analysis_config_from_ui
        from analysis.manager   import run_analysis

        a_cfg  = build_analysis_config_from_ui(ui_values)
        result = run_analysis(runner.engine, a_cfg)

        # ── Dashboard figure ──────────────────────────────────────────────
        layer_idx = int(ui_values.get("layer_idx", 0))
        fig = runner.get_dashboard_figure(layer_idx=layer_idx)

        # ── Feature summary DataFrame ─────────────────────────────────────
        summary = {}

        # Try pipeline_output dict first
        if hasattr(result, "pipeline_output") and result.pipeline_output:
            summary = result.pipeline_output.get("summary", {})

        # Fallback: flatten top-level numeric fields from result
        if not summary and hasattr(result, "__dict__"):
            for k, v in result.__dict__.items():
                if isinstance(v, (int, float)):
                    summary[k] = v

        if summary:
            rows = [
                (k, f"{v:.6f}" if isinstance(v, float) else str(v))
                for k, v in sorted(summary.items())
            ]
            df = pd.DataFrame(rows, columns=["metric", "value"])
        else:
            df = pd.DataFrame(columns=["metric", "value"])

        # ── Status message ────────────────────────────────────────────────
        n_figs    = len(getattr(result, "figure_paths",  {}))
        n_reports = len(getattr(result, "report_paths",  {}))
        lang      = ui_values.get("output_lang", "en")

        if lang == "zh":
            status = (
                f"✓ 分析完成 — {n_figs} 张图表，"
                f"{n_reports} 份报告，"
                f"{len(df)} 项特征"
            )
        else:
            status = (
                f"✓ Analysis complete — {n_figs} figure(s), "
                f"{n_reports} report(s), "
                f"{len(df)} feature(s)"
            )

        return fig, df, status

    except Exception as exc:
        lang = ui_values.get("output_lang", "en")
        if lang == "zh":
            return None, None, f"⚠  分析失败：{exc}"
        return None, None, f"⚠  Analysis failed: {exc}"
