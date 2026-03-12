"""Report export tab for markdown/html/latex generation."""

from __future__ import annotations

from typing import Any

import gradio as gr

from analysis.manager import run_analysis


_TEXT = {
    "en": {
        "lang": "Report Language",
        "formats": "Report Formats",
        "title": "Report Title",
        "toc": "Include TOC",
        "ai": "AI Parser (optional)",
        "enabled": "Enable AI parsing",
        "api": "API Key",
        "model": "Model",
        "mode": "Narrative Mode",
        "generate": "Generate Report",
        "download": "Generated Files",
        "preview": "Preview",
    },
    "zh": {
        "lang": "报告语言",
        "formats": "输出格式",
        "title": "报告标题",
        "toc": "包含目录",
        "ai": "AI 解析（可选）",
        "enabled": "启用 AI 解析",
        "api": "API 密钥",
        "model": "模型",
        "mode": "叙事模式",
        "generate": "生成报告",
        "download": "生成文件",
        "preview": "预览",
    },
}


def build_report_tab(lang: str = "en") -> dict[str, gr.Component]:
    """Build report tab components."""
    t = _TEXT.get(lang, _TEXT["en"])
    c: dict[str, gr.Component] = {}

    with gr.Row():
        c["report_lang"] = gr.Dropdown(label=t["lang"], choices=["zh", "en"], value="zh")
        c["report_formats"] = gr.CheckboxGroup(label=t["formats"], choices=["md", "html", "latex"], value=["md"])
    c["report_title"] = gr.Textbox(label=t["title"], placeholder="(optional)")
    c["include_toc"] = gr.Checkbox(label=t["toc"], value=True)

    with gr.Accordion(t["ai"], open=False):
        c["ai_enabled"] = gr.Checkbox(label=t["enabled"], value=False)
        c["api_key"] = gr.Textbox(label=t["api"], type="password")
        c["ai_model"] = gr.Dropdown(
            label=t["model"],
            choices=["gpt-4o", "gpt-4-turbo", "claude-3-5-sonnet"],
            value="gpt-4o",
        )
        c["narrative_mode"] = gr.Dropdown(
            label=t["mode"],
            choices=["chronicle", "diagnostic", "comparative", "predictive", "dramatic"],
            value=None,
        )

    c["generate_report_btn"] = gr.Button(t["generate"], variant="primary")
    c["report_files"] = gr.File(label=t["download"], interactive=False)
    c["report_preview"] = gr.Markdown(label=t["preview"])

    return c


def activate_report_tab(engine: Any, analysis_config: dict[str, Any], report_ui: dict[str, Any]) -> tuple[Any, str]:
    """Generate report artifacts from UI-selected options."""
    if engine is None:
        return None, "❌ No completed simulation found."

    cfg = dict(analysis_config)
    cfg.setdefault("output", {})
    cfg.setdefault("report", {})
    cfg.setdefault("parser", {})

    cfg["output"]["lang"] = report_ui.get("report_lang", "zh")
    cfg["report"]["formats"] = report_ui.get("report_formats") or ["md"]
    cfg["report"]["title"] = report_ui.get("report_title") or None
    cfg["report"]["include_toc"] = bool(report_ui.get("include_toc", True))
    cfg["parser"]["enabled"] = bool(report_ui.get("ai_enabled", False))
    cfg["parser"]["api_key"] = report_ui.get("api_key", "")
    cfg["parser"]["model"] = report_ui.get("ai_model", "gpt-4o")
    cfg["parser"]["narrative_mode"] = report_ui.get("narrative_mode") or None

    result = run_analysis(engine, cfg)
    files = list(result.report_paths.values())
    preview = "\n".join(f"- {k}: `{v}`" for k, v in result.report_paths.items()) or "No report generated."
    return files if files else None, preview
