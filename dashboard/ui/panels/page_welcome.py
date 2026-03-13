"""
ui/panels/page_welcome.py

P1 首页 — 语言选择 + 项目介绍 + 预设加载

Layout
------
  ┌─────────────────────────────────────────────┐
  │  [右上] 语言切换                               │
  │                                              │
  │  Hero：大标题 + 副标题 + 标签组                │
  │                                              │
  │  功能卡片（3列）                               │
  │                                              │
  │  操作按钮：[开始配置实验] [加载预设方案▾]         │
  │                                              │
  │  底部提示文字                                  │
  └─────────────────────────────────────────────┘

Public API
----------
WelcomeComponents
    Dataclass with all gr.Components.

build_welcome_page(lang) -> WelcomeComponents
    Render the page inside the current gr.Blocks context.
    Must be called inside a gr.Group(visible=...) wrapper in app.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gradio as gr


# ─── Localised strings ────────────────────────────────────────────────────────

_STRINGS = {
    "zh": {
        "title_line1":   "意见动力学",
        "title_line2":   "仿真平台",
        "subtitle": (
            "基于有界信任模型，模拟多智能体在社交网络、"
            "外生事件与内生机制共同作用下的意见演化过程。"
        ),
        "pills": ["有界信任动力学", "Hawkes 级联", "影响力场", "多层意见维度", "空间分布"],
        "card1_title": "模型配置",
        "card1_desc":  "智能体、动力学、网络、空间分布与事件参数的精细控制。",
        "card2_title": "实时监控",
        "card2_desc":  "极化时序、空间分布、意见直方图与事件流实时可视化。",
        "card3_title": "分析报告",
        "card3_desc":  "特征提取、AI 解析与 Markdown/HTML 格式报告生成。",
        "btn_start":   "开始配置实验 →",
        "btn_preset":  "加载预设方案 ▾",
        "hint": (
            "提示：按左侧导航依次完成配置后，"
            "进入「实验运行」页启动仿真。所有配置页均可随时修改。"
        ),
        "preset_names": [
            "默认参数",
            "高极化场景",
            "慢收敛场景",
            "强级联事件",
            "多层意见（3层）",
        ],
        "preset_confirm_title": "加载预设将覆盖当前所有配置，确认继续？",
        "preset_confirm_yes":   "确认加载",
        "preset_confirm_no":    "取消",
        "preset_loaded":        "✓ 已加载预设：",
    },
    "en": {
        "title_line1":   "Opinion Dynamics",
        "title_line2":   "Simulation Platform",
        "subtitle": (
            "Bounded-confidence model for simulating opinion evolution "
            "across multi-agent social networks driven by exogenous events "
            "and endogenous mechanisms."
        ),
        "pills": [
            "Bounded Confidence", "Hawkes Cascade", "Influence Field",
            "Multi-layer Opinions", "Spatial Distribution",
        ],
        "card1_title": "Model Config",
        "card1_desc":  "Fine-grained control over agents, dynamics, network, spatial and event parameters.",
        "card2_title": "Live Monitor",
        "card2_desc":  "Real-time polarisation timeline, spatial scatter, histogram and event stream.",
        "card3_title": "Analysis & Report",
        "card3_desc":  "Feature extraction, AI interpretation and Markdown/HTML report generation.",
        "btn_start":   "Start Configuring →",
        "btn_preset":  "Load Preset ▾",
        "hint": (
            "Tip: complete each config page using the sidebar, "
            "then go to Run Experiment to launch the simulation."
        ),
        "preset_names": [
            "Default",
            "High Polarisation",
            "Slow Convergence",
            "Strong Cascade",
            "Multi-layer (3 layers)",
        ],
        "preset_confirm_title": "Loading a preset will overwrite all current settings. Continue?",
        "preset_confirm_yes":   "Load",
        "preset_confirm_no":    "Cancel",
        "preset_loaded":        "✓ Preset loaded: ",
    },
}

# Available preset keys (must match core/defaults.py preset registry)
PRESET_KEYS: list[str] = [
    "default",
    "high_polarization",
    "slow_convergence",
    "strong_cascade",
    "multilayer_3",
]


# ─── HTML builders ────────────────────────────────────────────────────────────

def _hero_html(lang: str) -> str:
    s = _STRINGS[lang]
    pills_html = "".join(
        f'<span style="'
        f'display:inline-block;padding:3px 10px;border-radius:12px;'
        f'font-size:11px;background:#f1f5f9;border:0.5px solid #e2e8f0;'
        f'color:#475569;margin:0 4px 4px 0;">{p}</span>'
        for p in s["pills"]
    )
    return f"""
<div style="padding-top:8px;">
  <div style="font-size:22px;font-weight:500;color:#0f172a;line-height:1.3;
              font-family:'IBM Plex Sans',sans-serif;">
    {s['title_line1']}<br>{s['title_line2']}
  </div>
  <div style="font-size:13px;color:#475569;margin-top:10px;
              line-height:1.65;max-width:520px;">
    {s['subtitle']}
  </div>
  <div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:0;">
    {pills_html}
  </div>
</div>"""


def _feature_cards_html(lang: str) -> str:
    s = _STRINGS[lang]
    cards = [
        (s["card1_title"], s["card1_desc"], "⚙"),
        (s["card2_title"], s["card2_desc"], "◑"),
        (s["card3_title"], s["card3_desc"], "◎"),
    ]
    cards_html = ""
    for icon, title, desc in [(c[2], c[0], c[1]) for c in cards]:
        cards_html += f"""
<div style="background:#f8fafc;border:0.5px solid #e2e8f0;border-radius:6px;
            padding:14px 16px;flex:1;min-width:140px;">
  <div style="font-size:13px;font-weight:500;color:#0f172a;
              margin-bottom:4px;">{icon} {title}</div>
  <div style="font-size:12px;color:#64748b;line-height:1.55;">{desc}</div>
</div>"""
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap;">{cards_html}</div>'


def _hint_html(lang: str) -> str:
    s = _STRINGS[lang]
    return f"""
<div style="font-size:12px;color:#94a3b8;border-top:0.5px solid #e2e8f0;
            padding-top:12px;line-height:1.6;">
  {s['hint']}
</div>"""


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class WelcomeComponents:
    """
    All gr.Components created by build_welcome_page().

    Attributes
    ----------
    lang_zh_btn : gr.Button
        Language toggle — 中文
    lang_en_btn : gr.Button
        Language toggle — English
    hero_html : gr.HTML
        Dynamic hero text block (title + subtitle + pills).
    feature_html : gr.HTML
        Feature cards row.
    start_btn : gr.Button
        "Start configuring" — navigates to model_config page.
    preset_dropdown : gr.Dropdown
        Preset selection. Value is the preset name string.
    preset_status_md : gr.Markdown
        Short feedback line shown after loading a preset.
    hint_html : gr.HTML
        Bottom hint text.
    """
    lang_zh_btn:      gr.Button
    lang_en_btn:      gr.Button
    hero_html:        gr.HTML
    feature_html:     gr.HTML
    start_btn:        gr.Button
    preset_dropdown:  gr.Dropdown
    preset_status_md: gr.Markdown
    hint_html:        gr.HTML


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_welcome_page(lang: str = "zh") -> WelcomeComponents:
    """
    Render the welcome page inside the current Gradio context.

    Must be called inside an active gr.Blocks() + gr.Group() context.
    The enclosing gr.Group's visible= attribute is managed by app.py.

    Parameters
    ----------
    lang : {"zh", "en"}
        Initial display language.

    Returns
    -------
    WelcomeComponents
    """
    s = _STRINGS[lang]

    with gr.Column(elem_id="page-welcome", elem_classes="page-body"):

        # ── Language toggle (top-right) ────────────────────────────────────
        with gr.Row():
            gr.HTML('<div style="flex:1;"></div>')
            lang_zh_btn = gr.Button(
                "中文",
                elem_id    = "lang-btn-zh",
                elem_classes = "lang-btn selected" if lang == "zh" else "lang-btn",
                size       = "sm",
                min_width  = 60,
            )
            lang_en_btn = gr.Button(
                "English",
                elem_id    = "lang-btn-en",
                elem_classes = "lang-btn selected" if lang == "en" else "lang-btn",
                size       = "sm",
                min_width  = 60,
            )

        # ── Hero ──────────────────────────────────────────────────────────
        hero_html = gr.HTML(
            value   = _hero_html(lang),
            elem_id = "welcome-hero",
        )

        # ── Feature cards ─────────────────────────────────────────────────
        feature_html = gr.HTML(
            value   = _feature_cards_html(lang),
            elem_id = "welcome-features",
        )

        # ── CTA buttons ───────────────────────────────────────────────────
        with gr.Row():
            start_btn = gr.Button(
                s["btn_start"],
                elem_id      = "btn-start-config",
                elem_classes = "btn-primary-teal",
                size         = "lg",
                min_width    = 180,
            )
            preset_dropdown = gr.Dropdown(
                label     = None,
                choices   = list(zip(
                    _STRINGS[lang]["preset_names"],
                    PRESET_KEYS,
                )),
                value     = None,
                multiselect = False,
                allow_custom_value = False,
                interactive = True,
                elem_id   = "preset-dropdown",
                elem_classes = "btn-secondary",
                show_label = False,
                info      = None,
                scale     = 0,
                min_width = 160,
                # Use placeholder text as the button label
            )

        # Preset load status feedback
        preset_status_md = gr.Markdown(
            value     = "",
            visible   = False,
            elem_id   = "preset-status",
        )

        # ── Hint ──────────────────────────────────────────────────────────
        hint_html = gr.HTML(
            value   = _hint_html(lang),
            elem_id = "welcome-hint",
        )

    return WelcomeComponents(
        lang_zh_btn      = lang_zh_btn,
        lang_en_btn      = lang_en_btn,
        hero_html        = hero_html,
        feature_html     = feature_html,
        start_btn        = start_btn,
        preset_dropdown  = preset_dropdown,
        preset_status_md = preset_status_md,
        hint_html        = hint_html,
    )


# ─── Language refresh ─────────────────────────────────────────────────────────

def refresh_lang(lang: str) -> tuple:
    """
    Return updated gr.update values for all language-sensitive components.

    Usage in app.py::

        lang_zh_btn.click(
            fn      = lambda: refresh_lang("zh"),
            outputs = [
                welcome.hero_html,
                welcome.feature_html,
                welcome.hint_html,
                welcome.lang_zh_btn,
                welcome.lang_en_btn,
                welcome.preset_dropdown,
            ],
        )

    Returns
    -------
    tuple of gr.update values (in the order above).
    """
    s = _STRINGS[lang]
    return (
        gr.update(value=_hero_html(lang)),
        gr.update(value=_feature_cards_html(lang)),
        gr.update(value=_hint_html(lang)),
        gr.update(elem_classes="lang-btn selected" if lang == "zh" else "lang-btn"),
        gr.update(elem_classes="lang-btn selected" if lang == "en"  else "lang-btn"),
        gr.update(choices=list(zip(s["preset_names"], PRESET_KEYS))),
    )


# ─── Preset load handler ──────────────────────────────────────────────────────

def on_preset_selected(preset_key: str | None, lang: str) -> tuple:
    """
    Called when the user selects a preset from the dropdown.

    This function validates the key and returns a status message.
    Actual parameter hydration is performed by app.py via
    core/config_bridge.load_preset(preset_key).

    Parameters
    ----------
    preset_key : str | None
        Selected preset key (value side of the Dropdown choices).
    lang : str

    Returns
    -------
    tuple (preset_status_md_value: str, preset_status_visible: bool)
    """
    if not preset_key:
        return "", False

    s = _STRINGS[lang]
    preset_idx = PRESET_KEYS.index(preset_key) if preset_key in PRESET_KEYS else 0
    name = _STRINGS[lang]["preset_names"][preset_idx]
    return f"{s['preset_loaded']}{name}", True
