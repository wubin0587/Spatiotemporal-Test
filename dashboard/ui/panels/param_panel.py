"""
ui/panels/param_panel.py  ── v2 compatibility shim

原始内容已拆分到：
  • page_model_config.py       — 模型、动力学、网络、空间、事件参数
  • page_analysis_config.py    — 输出目录、报告语言、AI 解析
  • page_dashboard_settings.py — 刷新间隔、显示偏好
  • page_intervention.py       — 干预规则

此文件保留以避免外部代码（preset loaders、tests）的 import 路径断裂。
所有公开符号从对应新模块重新导出。

建议直接 import 新模块。此文件将在 v3.0 中移除。
"""

from __future__ import annotations
import warnings

from ui.panels.page_model_config import (
    ModelConfigComponents,
    build_model_config_page,
    _DEFAULTS as _MODEL_DEFAULTS,
)
from ui.panels.page_analysis_config import (
    AnalysisConfigComponents,
    build_analysis_config_page,
    _DEFAULTS as _ANALYSIS_DEFAULTS,
)
from ui.panels.page_dashboard_settings import (
    DashboardSettingsComponents,
    build_dashboard_settings_page,
    _DEFAULTS as _DASHBOARD_DEFAULTS,
)
from ui.panels.page_intervention import (
    InterventionComponents,
    build_intervention_page,
)

# Merged defaults dict for legacy callers
_DEFAULTS: dict = {
    **_MODEL_DEFAULTS,
    **_ANALYSIS_DEFAULTS,
    **_DASHBOARD_DEFAULTS,
}


def build_param_panel(lang: str = "zh", defaults: dict | None = None):
    """
    Deprecated shim.
    Use build_model_config_page() / build_analysis_config_page() directly.
    """
    warnings.warn(
        "build_param_panel() is deprecated; use the individual page builders.",
        DeprecationWarning,
        stacklevel=2,
    )
    mc = build_model_config_page(lang=lang, defaults=defaults or _DEFAULTS)
    return mc.param_components, None


__all__ = [
    "ModelConfigComponents",    "build_model_config_page",
    "AnalysisConfigComponents", "build_analysis_config_page",
    "DashboardSettingsComponents", "build_dashboard_settings_page",
    "InterventionComponents",   "build_intervention_page",
    "_DEFAULTS", "build_param_panel",
]
