"""Dynamic intervention rule builder widgets."""

from __future__ import annotations

import gradio as gr


_TEXT = {
    "en": {
        "title": "Intervention Rules",
        "add": "+ Add Rule",
        "trigger": "Trigger",
        "trigger_value": "Trigger Value",
        "strategy": "Policy",
        "strength": "Strength",
        "max_fire": "Max Fire Count",
        "checkpoint": "Enable Auto Checkpoint",
        "remove": "Remove",
        "empty": "No intervention rules yet.",
        "trigger_choices": ["step", "time", "polarization", "impact", "composite"],
        "policy_choices": ["network_rewire", "opinion_nudge", "event_suppress", "dynamics_param"],
    },
    "zh": {
        "title": "干预规则",
        "add": "+ 添加规则",
        "trigger": "触发器",
        "trigger_value": "触发值",
        "strategy": "策略",
        "strength": "强度",
        "max_fire": "最大触发次数",
        "checkpoint": "自动检查点",
        "remove": "删除",
        "empty": "当前没有干预规则。",
        "trigger_choices": ["step", "time", "polarization", "impact", "composite"],
        "policy_choices": ["network_rewire", "opinion_nudge", "event_suppress", "dynamics_param"],
    },
}


def build_intervention_builder(lang: str = "en", max_rules: int = 3) -> dict[str, gr.Component]:
    """Build a simple multi-rule intervention editor and return all widget refs."""
    t = _TEXT.get(lang, _TEXT["en"])
    comp: dict[str, gr.Component] = {}

    with gr.Accordion(t["title"], open=False):
        comp["add_rule_btn"] = gr.Button(t["add"], size="sm")
        comp["intervention_help"] = gr.Markdown(t["empty"])

        for idx in range(1, max_rules + 1):
            with gr.Group(visible=(idx == 1), elem_classes=["intervention-rule"]):
                comp[f"rule_{idx}_title"] = gr.Markdown(f"**Rule #{idx}**")
                with gr.Row():
                    comp[f"rule_{idx}_trigger"] = gr.Dropdown(
                        label=t["trigger"], choices=t["trigger_choices"], value="step"
                    )
                    comp[f"rule_{idx}_trigger_value"] = gr.Number(
                        label=t["trigger_value"], value=100 * idx, precision=0, minimum=0
                    )
                    comp[f"rule_{idx}_max_fire"] = gr.Number(
                        label=t["max_fire"], value=1, precision=0, minimum=1
                    )
                with gr.Row():
                    comp[f"rule_{idx}_policy"] = gr.Dropdown(
                        label=t["strategy"], choices=t["policy_choices"], value="network_rewire"
                    )
                    comp[f"rule_{idx}_strength"] = gr.Number(
                        label=t["strength"], value=0.2, minimum=0.0, maximum=1.0, precision=3
                    )
                with gr.Row():
                    comp[f"rule_{idx}_checkpoint"] = gr.Checkbox(label=t["checkpoint"], value=True)
                    comp[f"rule_{idx}_remove_btn"] = gr.Button(t["remove"], size="sm", variant="stop")

    return comp
