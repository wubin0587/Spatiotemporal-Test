"""
ui/panels/analysis_tab.py  ── v2 compatibility shim

静态分析面板内容已拆分到：
  • page_results.py — 静态仪表盘（sub-tab 1）和特征摘要（sub-tab 2）

迁移指南
────────
旧::
    from ui.panels.analysis_tab import build_analysis_tab
    analysis = build_analysis_tab(lang)

新::
    from ui.panels.page_results import build_results_page, render_features_table
    results = build_results_page(lang)
    # 静态仪表盘: results.dashboard_img, results.run_analysis_btn
    # 特征摘要:   results.features_html
    #             render_features_table(features_dict, lang)

dashboard_plot 对应  → results.dashboard_img
summary_df    对应  → results.features_html  (渲染为 HTML 表格)
run_analysis  对应  → results.run_analysis_btn.click
"""

from __future__ import annotations
import warnings

from ui.panels.page_results import (
    ResultsComponents,
    build_results_page,
    render_features_table,
    populate_from_run,
)


def build_analysis_tab(lang: str = "zh"):
    """Deprecated — use build_results_page() instead."""
    warnings.warn(
        "build_analysis_tab() is deprecated. Use build_results_page().",
        DeprecationWarning, stacklevel=2,
    )
    return build_results_page(lang=lang)


__all__ = [
    "ResultsComponents", "build_results_page",
    "build_analysis_tab", "render_features_table", "populate_from_run",
]
