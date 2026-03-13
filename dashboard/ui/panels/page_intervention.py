"""
ui/panels/page_intervention.py

P5 干预配置 — 干预规则管理

将原 intervention_builder.py 的内容提升为独立页面。
规则列表通过 gr.State 管理，最多 MAX_RULES 条。
每条规则渲染为独立 Card，通过 gr.HTML 动态刷新。

Layout
------
  Page header + 说明文字
  规则卡片列表（gr.HTML 渲染，空状态时显示占位提示）
  [＋ 添加规则]（虚线按钮）
  隐藏的规则编辑表单（每次只编辑一条）
  [← 返回]  [保存并继续 →]

Rule data structure (stored in gr.State as list[dict]):
  {
    "step":      int,    # 在第几步触发
    "target":    str,    # "all" | "group_0" | ... 
    "type":      str,    # "nudge" | "clamp" | "inject"
    "magnitude": float,  # 干预幅度，正负均可
    "layer":     int,    # 目标意见层
  }

Public API
----------
InterventionPageComponents
build_intervention_page(lang, defaults) -> InterventionPageComponents
collect_rules(comps) -> list[dict]
    Convenience function to extract current rules list from gr.State.
    Usage: inputs=[comps.rules_state] in a handler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json

import gradio as gr


MAX_RULES = 5

_DEFAULTS: dict[str, Any] = {
    "rules": [],   # list[dict] — see module docstring
}

_STRINGS = {
    "zh": {
        "page_title":    "干预配置",
        "page_subtitle": "设置仿真期间的人工干预规则。干预规则是可选的，最多 5 条。",
        "empty_state":   "尚未添加干预规则 — 点击下方按钮添加",
        "add_btn":       "＋ 添加规则",
        "add_limit":     "已达上限（5 条）",
        "rule_label":    "规则",
        "del_btn":       "✕ 删除",
        "step_label":    "触发步骤",
        "target_label":  "目标群体",
        "type_label":    "干预类型",
        "magnitude_label": "干预幅度",
        "layer_label":   "目标意见层",
        "back_btn":      "← 返回",
        "next_btn":      "保存并继续 →",
        "target_choices": [
            ("全体智能体", "all"),
            ("群组 0",     "group_0"),
            ("群组 1",     "group_1"),
            ("群组 2",     "group_2"),
        ],
        "type_choices": [
            ("微调 (nudge)",  "nudge"),
            ("钳制 (clamp)",  "clamp"),
            ("注入 (inject)", "inject"),
        ],
        "confirm_delete": "确认删除规则",
    },
    "en": {
        "page_title":    "Intervention Config",
        "page_subtitle": "Define manual interventions during the simulation. Optional, max 5 rules.",
        "empty_state":   "No rules yet — click the button below to add one",
        "add_btn":       "＋ Add Rule",
        "add_limit":     "Limit reached (5 rules)",
        "rule_label":    "Rule",
        "del_btn":       "✕ Remove",
        "step_label":    "Trigger step",
        "target_label":  "Target group",
        "type_label":    "Intervention type",
        "magnitude_label": "Magnitude",
        "layer_label":   "Opinion layer",
        "back_btn":      "← Back",
        "next_btn":      "Save & Continue →",
        "target_choices": [
            ("All agents", "all"),
            ("Group 0",    "group_0"),
            ("Group 1",    "group_1"),
            ("Group 2",    "group_2"),
        ],
        "type_choices": [
            ("Nudge",  "nudge"),
            ("Clamp",  "clamp"),
            ("Inject", "inject"),
        ],
        "confirm_delete": "Confirm delete rule",
    },
}


# ─── HTML renderers ───────────────────────────────────────────────────────────

def _render_rules_html(rules: list[dict], lang: str) -> str:
    """Render the rules list as an HTML string for gr.HTML."""
    s = _STRINGS[lang]

    if not rules:
        return f"""
<div class="intervention-empty">
  {s['empty_state']}
</div>"""

    _TYPE_LABELS = {d[1]: d[0] for d in s["type_choices"]}
    _TARGET_LABELS = {d[1]: d[0] for d in s["target_choices"]}

    cards_html = ""
    for i, rule in enumerate(rules):
        step      = rule.get("step", 1)
        target    = _TARGET_LABELS.get(rule.get("target", "all"), rule.get("target", "all"))
        rtype     = _TYPE_LABELS.get(rule.get("type", "nudge"), rule.get("type", "nudge"))
        magnitude = rule.get("magnitude", 0.1)
        layer     = rule.get("layer", 0)

        cards_html += f"""
<div class="intervention-rule" id="rule-card-{i}">
  <div class="intervention-rule-header">
    <span>{s['rule_label']} {i + 1}</span>
    <button class="intervention-rule-delete"
            onclick="deleteRule({i})">{s['del_btn']}</button>
  </div>
  <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:12px;
              font-family:'IBM Plex Mono',monospace;">
    <span><span style="color:#64748b;">{s['step_label']}</span>
          &nbsp;{step}</span>
    <span><span style="color:#64748b;">{s['target_label']}</span>
          &nbsp;{target}</span>
    <span><span style="color:#64748b;">{s['type_label']}</span>
          &nbsp;{rtype}</span>
    <span><span style="color:#64748b;">{s['magnitude_label']}</span>
          &nbsp;{magnitude:+.3f}</span>
    <span><span style="color:#64748b;">{s['layer_label']}</span>
          &nbsp;{layer}</span>
  </div>
</div>"""

    return cards_html


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class InterventionPageComponents:
    """All gr.Components from build_intervention_page()."""

    rules_state:    gr.State    # list[dict]  — source of truth for rules
    rules_html:     gr.HTML     # rendered rules list
    add_btn:        gr.Button   # "＋ 添加规则"

    # Inline edit form (shown when adding a rule)
    edit_group:       gr.Group
    edit_step:        gr.Number
    edit_target:      gr.Dropdown
    edit_type:        gr.Dropdown
    edit_magnitude:   gr.Number
    edit_layer:       gr.Number
    edit_confirm_btn: gr.Button
    edit_cancel_btn:  gr.Button

    # Delete target index (hidden state)
    delete_idx_state: gr.State  # int | None

    back_btn: gr.Button
    next_btn: gr.Button


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_intervention_page(
    lang:     str = "zh",
    defaults: dict[str, Any] | None = None,
) -> InterventionPageComponents:
    """
    Render the intervention config page.

    Parameters
    ----------
    lang : {"zh", "en"}
    defaults : dict | None
        May contain "rules": list[dict].

    Returns
    -------
    InterventionPageComponents
    """
    s  = _STRINGS[lang]
    dv = {**_DEFAULTS, **(defaults or {})}
    initial_rules: list[dict] = list(dv.get("rules", []))

    with gr.Column(elem_id="page-intervention", elem_classes="page-body"):

        # ── Page header ───────────────────────────────────────────────────
        gr.HTML(f"""
<div class="page-header" style="padding:20px 0 16px;">
  <div class="page-title">{s['page_title']}</div>
  <div class="page-subtitle">{s['page_subtitle']}</div>
</div>""")

        # ── State ─────────────────────────────────────────────────────────
        rules_state     = gr.State(value=initial_rules)
        delete_idx_state = gr.State(value=None)

        # ── Rules list (dynamic HTML) ─────────────────────────────────────
        rules_html = gr.HTML(
            value   = _render_rules_html(initial_rules, lang),
            elem_id = "rules-list-html",
        )

        # ── Add button ────────────────────────────────────────────────────
        add_btn = gr.Button(
            s["add_btn"] if len(initial_rules) < MAX_RULES else s["add_limit"],
            elem_id      = "btn-add-rule",
            interactive  = len(initial_rules) < MAX_RULES,
            size         = "sm",
        )

        # ── Inline edit form (hidden by default) ──────────────────────────
        with gr.Group(
            visible      = False,
            elem_id      = "rule-edit-group",
            elem_classes = "section-card",
        ) as edit_group:
            gr.HTML(f'<div class="section-card-title">'
                    f'{s["rule_label"]} — {s["add_btn"]}</div>')

            with gr.Row():
                edit_step = gr.Number(
                    label=s["step_label"], value=100,
                    minimum=1, maximum=100000, step=1, precision=0,
                    interactive=True, scale=1,
                )
                edit_target = gr.Dropdown(
                    label=s["target_label"],
                    choices=s["target_choices"],
                    value="all",
                    interactive=True, scale=1,
                )
                edit_type = gr.Dropdown(
                    label=s["type_label"],
                    choices=s["type_choices"],
                    value="nudge",
                    interactive=True, scale=1,
                )
            with gr.Row():
                edit_magnitude = gr.Number(
                    label=s["magnitude_label"], value=0.1,
                    minimum=-1.0, maximum=1.0, step=0.01,
                    interactive=True, scale=1,
                )
                edit_layer = gr.Number(
                    label=s["layer_label"], value=0,
                    minimum=0, maximum=9, step=1, precision=0,
                    interactive=True, scale=1,
                )
                gr.HTML('<div style="flex:1;"></div>')

            with gr.Row():
                edit_confirm_btn = gr.Button(
                    "✓ 确认" if lang == "zh" else "✓ Confirm",
                    elem_classes = "btn-primary-teal",
                    size         = "sm",
                )
                edit_cancel_btn = gr.Button(
                    "✕ 取消" if lang == "zh" else "✕ Cancel",
                    elem_classes = "btn-secondary",
                    size         = "sm",
                )

        # ── Event: Add button → show edit form ────────────────────────────
        add_btn.click(
            fn      = lambda: gr.update(visible=True),
            inputs  = [],
            outputs = [edit_group],
        )

        # ── Event: Cancel → hide edit form ────────────────────────────────
        edit_cancel_btn.click(
            fn      = lambda: gr.update(visible=False),
            inputs  = [],
            outputs = [edit_group],
        )

        # ── Event: Confirm → append rule, refresh list ────────────────────
        def _on_confirm(
            rules: list[dict],
            step: float, target: str, rtype: str,
            magnitude: float, layer: float,
        ) -> tuple:
            new_rule = {
                "step":      int(step),
                "target":    target,
                "type":      rtype,
                "magnitude": round(float(magnitude), 4),
                "layer":     int(layer),
            }
            updated = list(rules) + [new_rule]
            at_limit = len(updated) >= MAX_RULES
            return (
                updated,
                gr.update(value=_render_rules_html(updated, lang)),
                gr.update(visible=False),       # hide edit form
                gr.update(                       # update add button
                    value       = s["add_limit"] if at_limit else s["add_btn"],
                    interactive = not at_limit,
                ),
            )

        edit_confirm_btn.click(
            fn      = _on_confirm,
            inputs  = [
                rules_state,
                edit_step, edit_target, edit_type,
                edit_magnitude, edit_layer,
            ],
            outputs = [rules_state, rules_html, edit_group, add_btn],
        )

        # ── JavaScript: delete button inside gr.HTML → hidden Gradio btn ──
        # We render a hidden Gradio button that receives the delete index
        # forwarded from the JS onclick in the rules HTML.
        # app.py binds the JS bridge via a custom JS snippet loaded in the
        # Blocks head.  The pattern:
        #   JS: window._gradioDeleteRuleBtn.click()  +  store idx in hidden state
        #   Python handler reads delete_idx_state, removes the rule.
        #
        # Here we create the hidden delete trigger button.
        delete_trigger_btn = gr.Button(
            "delete-trigger",
            visible    = False,
            elem_id    = "delete-rule-trigger",
        )

        def _on_delete(rules: list[dict], idx: int | None) -> tuple:
            if idx is None or idx < 0 or idx >= len(rules):
                # No-op
                return rules, gr.update(), gr.update()
            updated = [r for i, r in enumerate(rules) if i != idx]
            at_limit = len(updated) >= MAX_RULES
            return (
                updated,
                gr.update(value=_render_rules_html(updated, lang)),
                gr.update(
                    value       = s["add_limit"] if at_limit else s["add_btn"],
                    interactive = not at_limit,
                ),
            )

        delete_trigger_btn.click(
            fn      = _on_delete,
            inputs  = [rules_state, delete_idx_state],
            outputs = [rules_state, rules_html, add_btn],
        )

        # ── Sticky action bar ─────────────────────────────────────────────
        with gr.Row(elem_classes="page-action-bar"):
            back_btn = gr.Button(
                s["back_btn"],
                elem_id      = "btn-intervention-back",
                elem_classes = "btn-secondary",
                size         = "sm",
            )
            next_btn = gr.Button(
                s["next_btn"],
                elem_id      = "btn-intervention-next",
                elem_classes = "btn-primary-teal",
                size         = "sm",
            )

    return InterventionPageComponents(
        rules_state      = rules_state,
        rules_html       = rules_html,
        add_btn          = add_btn,
        edit_group       = edit_group,
        edit_step        = edit_step,
        edit_target      = edit_target,
        edit_type        = edit_type,
        edit_magnitude   = edit_magnitude,
        edit_layer       = edit_layer,
        edit_confirm_btn = edit_confirm_btn,
        edit_cancel_btn  = edit_cancel_btn,
        delete_idx_state = delete_idx_state,
        back_btn         = back_btn,
        next_btn         = next_btn,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def collect_rules(rules_state_value: list[dict]) -> list[dict]:
    """
    Validate and return the rules list from gr.State value.
    Used in app.py to inject intervention_rules into ui_values before
    calling validator.validate_all().

    Parameters
    ----------
    rules_state_value : list[dict]
        The .value of InterventionPageComponents.rules_state,
        captured as an input in a Gradio handler.

    Returns
    -------
    list[dict]
        Validated, sanitised rules list.
    """
    if not isinstance(rules_state_value, list):
        return []
    sanitised = []
    for r in rules_state_value:
        if not isinstance(r, dict):
            continue
        sanitised.append({
            "step":      max(1, int(r.get("step", 1))),
            "target":    str(r.get("target", "all")),
            "type":      str(r.get("type", "nudge")),
            "magnitude": float(r.get("magnitude", 0.1)),
            "layer":     max(0, int(r.get("layer", 0))),
        })
    return sanitised
