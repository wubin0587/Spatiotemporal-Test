"""
ui/panels/report_tab.py  ── v2 compatibility shim

报告面板内容已拆分到：
  • page_analysis_config.py — 报告格式、标题、AI 配置（配置侧）
  • page_results.py         — AI 报告预览与下载（输出侧，sub-tab 3）

迁移指南
────────
旧::
    from ui.panels.report_tab import build_report_tab
    report = build_report_tab(lang)

新（配置）::
    from ui.panels.page_analysis_config import build_analysis_config_page
    analysis_cfg = build_analysis_config_page(lang)
    # report_format, report_title, ai_enabled, api_key, ...

新（输出预览）::
    from ui.panels.page_results import build_results_page
    results = build_results_page(lang)
    # results.report_html            — Markdown 预览区
    # results.gen_report_btn_inner   — 生成按钮（页内）
    # results.gen_report_btn         — 生成按钮（页头操作栏）
    # results.report_download_btn    — 下载按钮

app.py 报告生成回调示例::
    results.gen_report_btn.click(
        fn      = _generate_report,
        inputs  = [analysis_cfg.param_components["api_key"], ...],
        outputs = [results.report_html, results.report_download_btn],
    )
"""

from __future__ import annotations
import warnings

from ui.panels.page_results import (
    ResultsComponents,
    build_results_page,
)
from ui.panels.page_analysis_config import (
    AnalysisConfigComponents,
    build_analysis_config_page,
)


def build_report_tab(lang: str = "zh"):
    """Deprecated — use build_results_page() and build_analysis_config_page()."""
    warnings.warn(
        "build_report_tab() is deprecated. "
        "Use build_results_page() for preview and "
        "build_analysis_config_page() for report settings.",
        DeprecationWarning, stacklevel=2,
    )
    return build_results_page(lang=lang)


__all__ = [
    "ResultsComponents",         "build_results_page",
    "AnalysisConfigComponents",  "build_analysis_config_page",
    "build_report_tab",
]
