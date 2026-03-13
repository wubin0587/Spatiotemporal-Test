"""
ui/panels/report_tab.py

AI-assisted Report tab for the Opinion Dynamics Simulation Dashboard.

Lets the user configure report format, optional AI narrative, and
generate a downloadable Markdown / HTML report from the last completed
simulation.

Public API
----------
ReportTab
    Dataclass holding all gr.Components for the tab.

build_report_tab(lang) -> ReportTab
    Render the tab contents inside the current Gradio context.

generate_report(runner, report_opts, ui_values) -> tuple
    Run the report generation pipeline and return output values
    matching ``report_tab.output_list()``.

    Parameters
    ----------
    runner       : SimulationRunner   — must have runner.engine set
    report_opts  : dict               — keys from ReportTab components
                   {report_fmt, report_title, ai_enabled, api_key,
                    ai_model, narrative_mode, theme_name}
    ui_values    : dict               — flat simulation param dict

    Returns
    -------
    tuple[str, str | None, str | None]
        (report_preview_md, download_report_path, download_features_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr


# ─────────────────────────────────────────────────────────────────────────────
# ReportTab dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReportTab:
    """Holds all gr.Components for the Report tab."""

    # Config row
    report_fmt:   gr.Dropdown
    report_title: gr.Textbox

    # AI parser accordion contents
    ai_enabled:     gr.Checkbox
    api_key:        gr.Textbox
    ai_model:       gr.Dropdown
    narrative_mode: gr.Dropdown
    theme_name:     gr.Dropdown

    # Actions
    generate_btn:  gr.Button

    # Outputs
    report_preview:        gr.Markdown
    download_report_btn:   gr.DownloadButton
    download_features_btn: gr.DownloadButton

    # ── Convenience ──────────────────────────────────────────────────────

    def report_inputs(self) -> list:
        """
        Ordered list of components that describe *report options*.
        Pass as ``inputs=`` after runner_state and simulation params.
        """
        return [
            self.report_fmt,
            self.report_title,
            self.ai_enabled,
            self.api_key,
            self.ai_model,
            self.narrative_mode,
            self.theme_name,
        ]

    def output_list(self) -> list:
        """Outputs list for ``generate_btn.click(outputs=...)``."""
        return [
            self.report_preview,
            self.download_report_btn,
            self.download_features_btn,
        ]

    def as_opts_dict(self, *vals) -> dict:
        """
        Convert a flat tuple of values (from report_inputs()) into a
        named dict.  Useful in click handlers:

            def _handler(runner, *rvals, *pvals):
                report_opts = report_tab.as_opts_dict(*rvals[:7])
                ...
        """
        keys = [
            "report_fmt", "report_title",
            "ai_enabled", "api_key", "ai_model",
            "narrative_mode", "theme_name",
        ]
        return dict(zip(keys, vals))


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report_tab(lang: str = "en") -> ReportTab:
    """
    Render the Report tab contents inside the current Gradio context.

    Must be called inside an active ``gr.Tab()`` or ``gr.Blocks()`` context.

    Parameters
    ----------
    lang : {"en", "zh"}

    Returns
    -------
    ReportTab
    """
    _t = lambda en, zh: zh if lang == "zh" else en

    # ── Format + title row ────────────────────────────────────────────────
    with gr.Row():
        report_fmt = gr.Dropdown(
            label=_t("Format", "格式"),
            choices=["md", "html"],
            value="md",
            scale=1,
        )
        report_title = gr.Textbox(
            label=_t("Title (optional)", "标题（可选）"),
            placeholder=_t("auto-generated", "自动生成"),
            scale=3,
        )

    # ── AI Parser accordion ───────────────────────────────────────────────
    with gr.Accordion(
        _t("AI Parser (optional)", "AI 解析（可选）"),
        open=False,
    ):
        ai_enabled = gr.Checkbox(
            label=_t("Enable AI analysis", "启用 AI 分析"),
            value=False,
            info=_t(
                "Uses an external LLM to add narrative interpretation.",
                "使用外部大语言模型生成叙事解读。",
            ),
        )
        api_key = gr.Textbox(
            label="API Key",
            type="password",
            placeholder="sk-...",
        )
        ai_model = gr.Dropdown(
            label="Model",
            choices=[
                "gpt-4o",
                "gpt-4-turbo",
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ],
            value="gpt-4o",
        )
        with gr.Row():
            narrative_mode = gr.Dropdown(
                label=_t("Narrative Mode", "叙事风格"),
                choices=[
                    "",
                    "chronicle",
                    "diagnostic",
                    "comparative",
                    "predictive",
                    "dramatic",
                ],
                value="",
                info=_t(
                    "Leave blank for default narrative.",
                    "留空则使用默认叙事风格。",
                ),
            )
            theme_name = gr.Dropdown(
                label=_t("Theme", "主题"),
                choices=[""],
                value="",
                info=_t(
                    "Optional thematic framing for the narrative.",
                    "可选的叙事主题框架。",
                ),
            )

    # ── Generate button ───────────────────────────────────────────────────
    generate_btn = gr.Button(
        _t("📄  Generate Report", "📄  生成报告"),
        variant="primary",
    )

    # ── Report preview ────────────────────────────────────────────────────
    report_preview = gr.Markdown(
        _t("_No report yet._", "_尚无报告。_"),
        elem_id="report-preview",
    )

    # ── Download buttons ──────────────────────────────────────────────────
    with gr.Row():
        download_report_btn = gr.DownloadButton(
            _t("⬇  Download Report",   "⬇  下载报告"),
            size="sm",
        )
        download_features_btn = gr.DownloadButton(
            _t("⬇  Download Features", "⬇  下载特征数据"),
            size="sm",
        )

    return ReportTab(
        report_fmt=report_fmt,
        report_title=report_title,
        ai_enabled=ai_enabled,
        api_key=api_key,
        ai_model=ai_model,
        narrative_mode=narrative_mode,
        theme_name=theme_name,
        generate_btn=generate_btn,
        report_preview=report_preview,
        download_report_btn=download_report_btn,
        download_features_btn=download_features_btn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# generate_report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    runner: Any,
    report_opts: dict,
    ui_values: dict,
) -> tuple:
    """
    Run the report generation pipeline.

    This function is the implementation backing ``generate_btn.click``.
    It is separated from the component tree so it can be unit-tested.

    Parameters
    ----------
    runner : SimulationRunner
    report_opts : dict
        Keys: report_fmt, report_title, ai_enabled, api_key,
              ai_model, narrative_mode, theme_name
    ui_values : dict
        Full flat simulation parameter dict.

    Returns
    -------
    tuple[str, str | None, str | None]
        Matching ``ReportTab.output_list()`` order:
        (report_preview_md, download_report_path, download_features_path)
    """
    lang = ui_values.get("output_lang", "en")
    _t   = lambda en, zh: zh if lang == "zh" else en

    # ── Guard: no simulation ──────────────────────────────────────────────
    if runner is None or runner.engine is None:
        return (
            _t("_No completed simulation._", "_尚无仿真结果。_"),
            None,
            None,
        )

    # ── Build analysis config ─────────────────────────────────────────────
    try:
        from core.config_bridge import build_analysis_config_from_ui
        from analysis.manager   import run_analysis

        # Merge report options into ui_values for config bridge
        merged = {**ui_values, **report_opts}
        a_cfg  = build_analysis_config_from_ui(merged)

        # Skip figure generation for report-only pass (faster)
        a_cfg["visual"]["enabled"] = False

        # Inject AI parser settings if enabled
        if report_opts.get("ai_enabled"):
            a_cfg.setdefault("parser", {})
            a_cfg["parser"]["enabled"]        = True
            a_cfg["parser"]["api_key"]        = report_opts.get("api_key", "")
            a_cfg["parser"]["model"]          = report_opts.get("ai_model", "gpt-4o")
            a_cfg["parser"]["narrative_mode"] = report_opts.get("narrative_mode", "")
        else:
            a_cfg.setdefault("parser", {})
            a_cfg["parser"]["enabled"] = False

        # Title override
        title = report_opts.get("report_title", "").strip()
        if title:
            a_cfg.setdefault("report", {})
            a_cfg["report"]["title"] = title

        # ── Run ───────────────────────────────────────────────────────────
        result = run_analysis(runner.engine, a_cfg)

        # ── Report text ───────────────────────────────────────────────────
        report_fmt  = report_opts.get("report_fmt", "md")
        report_path = None
        feat_path   = None

        if hasattr(result, "report_paths"):
            report_path = result.report_paths.get(report_fmt) or \
                          result.report_paths.get("md")

        if hasattr(result, "feature_paths"):
            feat_path = result.feature_paths.get("summary_json")

        if report_path:
            try:
                with open(report_path, encoding="utf-8") as f:
                    preview_text = f.read()
                # If HTML, show truncated source in a code fence
                if report_fmt == "html":
                    snippet = preview_text[:3000]
                    preview_text = f"```html\n{snippet}\n```"
            except OSError:
                preview_text = _t(
                    f"_Report saved to `{report_path}` but could not be previewed._",
                    f"_报告已保存至 `{report_path}`，但无法预览。_",
                )
        else:
            preview_text = _t(
                "_Report generated but path not returned._",
                "_报告已生成，但未返回路径。_",
            )

        return preview_text, report_path, feat_path

    except Exception as exc:
        return (
            _t(f"_Error: {exc}_", f"_出错：{exc}_"),
            None,
            None,
        )
