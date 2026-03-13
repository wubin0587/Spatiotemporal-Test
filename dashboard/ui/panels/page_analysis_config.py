"""
ui/panels/page_analysis_config.py

P4 分析配置 — 输出目录、报告语言、AI 解析设置

从原 param_panel.py 的输出/报告区域拆分而来。
AI 解析 Card 在 ai_enabled toggle 打开后展开，API Key 字段默认隐藏。

Layout
------
  Card 1: 输出设置
    output_dir  output_lang
    save_timeseries  save_features_json
    include_trend  layer_idx

  Card 2: AI 解析（toggle 展开）
    ai_enabled  (toggle switch)
    api_key     ai_model
    narrative_style  narrative_theme

  Card 3: 报告格式
    report_format  report_title

  [← 返回]  [保存并继续 →]

Public API
----------
AnalysisConfigComponents
build_analysis_config_page(lang, defaults) -> AnalysisConfigComponents
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr


# ─── Default values ───────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "output_dir":          "output/run_001",
    "output_lang":         "zh",
    "save_timeseries":     True,
    "save_features_json":  True,
    "include_trend":       True,
    "layer_idx":           0,
    "ai_enabled":          False,
    "api_key":             "",
    "ai_model":            "gpt-4o",
    "narrative_style":     "",
    "narrative_theme":     "",
    "report_format":       "md",
    "report_title":        "",
}

# ─── Localised strings ────────────────────────────────────────────────────────

_STRINGS = {
    "zh": {
        "page_title":    "分析配置",
        "page_subtitle": "设置输出目录、报告语言及 AI 解析选项。",
        "card1_title":   "输出设置",
        "output_dir":    "输出目录",
        "output_dir_ph": "例：output/run_001",
        "output_lang":   "报告语言",
        "save_timeseries":    "保存时序数据",
        "save_features_json": "保存特征 JSON",
        "include_trend":      "包含趋势分析",
        "layer_idx":          "主意见层索引",
        "card2_title":   "AI 解析",
        "ai_enabled":    "启用 AI 分析",
        "api_key":       "API Key",
        "api_key_ph":    "sk-…",
        "ai_model":      "模型",
        "narrative_style": "叙事风格",
        "narrative_theme": "叙事主题",
        "card3_title":   "报告格式",
        "report_format": "输出格式",
        "report_title":  "报告标题",
        "report_title_ph": "（留空使用默认标题）",
        "back_btn":  "← 返回",
        "next_btn":  "保存并继续 →",
        "lang_choices": [("中文", "zh"), ("English", "en")],
        "model_choices": [
            ("GPT-4o",          "gpt-4o"),
            ("GPT-4o mini",     "gpt-4o-mini"),
            ("Claude 3.5 Sonnet", "claude-sonnet-4-20250514"),
        ],
        "style_choices": [
            ("— 默认 —", ""),
            ("学术严谨",  "academic"),
            ("科普通俗",  "popular"),
            ("政策简报",  "policy"),
        ],
        "theme_choices": [
            ("— 默认 —", ""),
            ("社会极化",  "polarization"),
            ("信息传播",  "diffusion"),
            ("共识形成",  "consensus"),
        ],
        "format_choices": [
            ("Markdown (.md)",  "md"),
            ("HTML (.html)",    "html"),
            ("纯文本 (.txt)",   "txt"),
        ],
        "ai_disabled_note": "启用后将在仿真完成后自动调用 AI 生成解析报告。",
    },
    "en": {
        "page_title":    "Analysis Config",
        "page_subtitle": "Configure output path, report language and AI interpretation.",
        "card1_title":   "Output Settings",
        "output_dir":    "Output directory",
        "output_dir_ph": "e.g. output/run_001",
        "output_lang":   "Report language",
        "save_timeseries":    "Save timeseries data",
        "save_features_json": "Save features JSON",
        "include_trend":      "Include trend analysis",
        "layer_idx":          "Primary layer index",
        "card2_title":   "AI Interpretation",
        "ai_enabled":    "Enable AI analysis",
        "api_key":       "API Key",
        "api_key_ph":    "sk-…",
        "ai_model":      "Model",
        "narrative_style": "Narrative style",
        "narrative_theme": "Narrative theme",
        "card3_title":   "Report Format",
        "report_format": "Output format",
        "report_title":  "Report title",
        "report_title_ph": "(leave blank for auto-title)",
        "back_btn":  "← Back",
        "next_btn":  "Save & Continue →",
        "lang_choices": [("中文", "zh"), ("English", "en")],
        "model_choices": [
            ("GPT-4o",          "gpt-4o"),
            ("GPT-4o mini",     "gpt-4o-mini"),
            ("Claude 3.5 Sonnet", "claude-sonnet-4-20250514"),
        ],
        "style_choices": [
            ("— Default —", ""),
            ("Academic",    "academic"),
            ("Popular",     "popular"),
            ("Policy brief","policy"),
        ],
        "theme_choices": [
            ("— Default —",      ""),
            ("Social polarisation","polarization"),
            ("Info diffusion",   "diffusion"),
            ("Consensus",        "consensus"),
        ],
        "format_choices": [
            ("Markdown (.md)", "md"),
            ("HTML (.html)",   "html"),
            ("Plain text",     "txt"),
        ],
        "ai_disabled_note": "When enabled, AI will generate an interpretation report after each run.",
    },
}


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfigComponents:
    """All gr.Components from build_analysis_config_page()."""

    output_dir:          gr.Textbox
    output_lang:         gr.Dropdown
    save_timeseries:     gr.Checkbox
    save_features_json:  gr.Checkbox
    include_trend:       gr.Checkbox
    layer_idx:           gr.Number

    ai_enabled:          gr.Checkbox
    ai_settings_group:   gr.Group      # visibility-toggled group
    api_key:             gr.Textbox
    ai_model:            gr.Dropdown
    narrative_style:     gr.Dropdown
    narrative_theme:     gr.Dropdown

    report_format:       gr.Dropdown
    report_title:        gr.Textbox

    back_btn:            gr.Button
    next_btn:            gr.Button

    @property
    def param_components(self) -> dict[str, gr.Component]:
        """Flat dict for sidebar binding and validate_all()."""
        return {
            "output_dir":         self.output_dir,
            "output_lang":        self.output_lang,
            "save_timeseries":    self.save_timeseries,
            "save_features_json": self.save_features_json,
            "include_trend":      self.include_trend,
            "layer_idx":          self.layer_idx,
            "ai_enabled":         self.ai_enabled,
            "api_key":            self.api_key,
            "ai_model":           self.ai_model,
            "narrative_style":    self.narrative_style,
            "narrative_theme":    self.narrative_theme,
            "report_format":      self.report_format,
            "report_title":       self.report_title,
        }


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_analysis_config_page(
    lang:     str = "zh",
    defaults: dict[str, Any] | None = None,
) -> AnalysisConfigComponents:
    """
    Render the analysis config page.

    Parameters
    ----------
    lang : {"zh", "en"}
    defaults : dict | None

    Returns
    -------
    AnalysisConfigComponents
    """
    s  = _STRINGS[lang]
    dv = {**_DEFAULTS, **(defaults or {})}

    with gr.Column(elem_id="page-analysis-config", elem_classes="page-body"):

        # ── Page header ───────────────────────────────────────────────────
        gr.HTML(f"""
<div class="page-header" style="padding:20px 0 16px;">
  <div class="page-title">{s['page_title']}</div>
  <div class="page-subtitle">{s['page_subtitle']}</div>
</div>""")

        # ── Card 1: Output settings ───────────────────────────────────────
        with gr.Group(elem_classes="section-card"):
            gr.HTML(f'<div class="section-card-title">{s["card1_title"]}</div>')

            with gr.Row():
                output_dir = gr.Textbox(
                    label       = s["output_dir"],
                    placeholder = s["output_dir_ph"],
                    value       = dv["output_dir"],
                    interactive = True,
                    scale       = 2,
                )
                output_lang = gr.Dropdown(
                    label       = s["output_lang"],
                    choices     = s["lang_choices"],
                    value       = dv["output_lang"],
                    interactive = True,
                    scale       = 1,
                )

            with gr.Row():
                save_timeseries = gr.Checkbox(
                    label=s["save_timeseries"],
                    value=dv["save_timeseries"],
                    interactive=True, scale=1,
                )
                save_features_json = gr.Checkbox(
                    label=s["save_features_json"],
                    value=dv["save_features_json"],
                    interactive=True, scale=1,
                )
                include_trend = gr.Checkbox(
                    label=s["include_trend"],
                    value=dv["include_trend"],
                    interactive=True, scale=1,
                )

            with gr.Row():
                layer_idx = gr.Number(
                    label=s["layer_idx"],
                    value=dv["layer_idx"],
                    minimum=0, maximum=9, step=1, precision=0,
                    interactive=True, scale=1,
                )
                gr.HTML('<div style="flex:2;"></div>')   # spacer

        # ── Card 2: AI interpretation ─────────────────────────────────────
        with gr.Group(elem_classes="section-card"):
            gr.HTML(f'<div class="section-card-title">{s["card2_title"]}</div>')

            with gr.Row():
                ai_enabled = gr.Checkbox(
                    label       = s["ai_enabled"],
                    info        = s["ai_disabled_note"],
                    value       = dv["ai_enabled"],
                    interactive = True,
                )

            # Collapsible settings — visible only when ai_enabled is True
            with gr.Group(
                visible      = dv["ai_enabled"],
                elem_id      = "ai-settings-group",
                elem_classes = "section-card",
            ) as ai_settings_group:
                with gr.Row():
                    api_key = gr.Textbox(
                        label       = s["api_key"],
                        placeholder = s["api_key_ph"],
                        value       = dv["api_key"],
                        type        = "password",
                        interactive = True,
                        scale       = 2,
                    )
                    ai_model = gr.Dropdown(
                        label       = s["ai_model"],
                        choices     = s["model_choices"],
                        value       = dv["ai_model"],
                        interactive = True,
                        scale       = 1,
                    )
                with gr.Row():
                    narrative_style = gr.Dropdown(
                        label       = s["narrative_style"],
                        choices     = s["style_choices"],
                        value       = dv["narrative_style"],
                        interactive = True,
                        scale       = 1,
                    )
                    narrative_theme = gr.Dropdown(
                        label       = s["narrative_theme"],
                        choices     = s["theme_choices"],
                        value       = dv["narrative_theme"],
                        interactive = True,
                        scale       = 1,
                    )

            # Toggle AI settings group visibility
            ai_enabled.change(
                fn      = lambda enabled: gr.update(visible=enabled),
                inputs  = [ai_enabled],
                outputs = [ai_settings_group],
            )

        # ── Card 3: Report format ─────────────────────────────────────────
        with gr.Group(elem_classes="section-card"):
            gr.HTML(f'<div class="section-card-title">{s["card3_title"]}</div>')
            with gr.Row():
                report_format = gr.Dropdown(
                    label       = s["report_format"],
                    choices     = s["format_choices"],
                    value       = dv["report_format"],
                    interactive = True,
                    scale       = 1,
                )
                report_title = gr.Textbox(
                    label       = s["report_title"],
                    placeholder = s["report_title_ph"],
                    value       = dv["report_title"],
                    interactive = True,
                    scale       = 2,
                )

        # ── Sticky action bar ─────────────────────────────────────────────
        with gr.Row(elem_classes="page-action-bar"):
            back_btn = gr.Button(
                s["back_btn"],
                elem_id      = "btn-analysis-back",
                elem_classes = "btn-secondary",
                size         = "sm",
            )
            next_btn = gr.Button(
                s["next_btn"],
                elem_id      = "btn-analysis-next",
                elem_classes = "btn-primary-teal",
                size         = "sm",
            )

    return AnalysisConfigComponents(
        output_dir=output_dir, output_lang=output_lang,
        save_timeseries=save_timeseries,
        save_features_json=save_features_json,
        include_trend=include_trend, layer_idx=layer_idx,
        ai_enabled=ai_enabled,
        ai_settings_group=ai_settings_group,
        api_key=api_key, ai_model=ai_model,
        narrative_style=narrative_style, narrative_theme=narrative_theme,
        report_format=report_format, report_title=report_title,
        back_btn=back_btn, next_btn=next_btn,
    )
