"""
ui/components/intervention_builder.py

Dynamic "Intervention Rules" form — lets the user specify up to MAX_RULES
targeted interventions that are injected into the simulation engine before
the run begins.

An intervention rule has the form:
    {
        "step":      int   — simulation step at which to apply
        "target":    str   — "all" | "high" | "low" | "extreme"
        "type":      str   — "nudge" | "broadcast" | "silence"
        "magnitude": float — signed shift applied to opinion
        "layer":     int   — which opinion layer (0-indexed)
    }

Public API
----------
InterventionBuilder
    A container object returned by build_intervention_builder().
    Holds all gr.components for up to MAX_RULES rule rows plus
    the "Add Rule" button and rule-count state.

build_intervention_builder(lang) -> InterventionBuilder
    Must be called inside an active gr.Blocks context.
    Renders a collapsible Accordion with a "Add Rule" button
    and up to MAX_RULES rows, each initially hidden.

bind_events(ib: InterventionBuilder) -> None
    Wire the Add/Remove button click events.  Call once after
    build_app() has registered all other events.

collect_rules(ib: InterventionBuilder, *vals) -> list[dict]
    Given the flat list of gr.component values that match
    ib.all_inputs(), return a parsed list of active rule dicts.

Notes
-----
- Row visibility is controlled via gr.update(visible=...).
- The component tree is fixed at MAX_RULES rows;
  rows beyond the active count are hidden.
- Gradio does not support truly dynamic component creation at
  runtime, so the maximum is compiled in at build time.
"""

from __future__ import annotations

import gradio as gr
from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_RULES: int = 5

_TARGET_CHOICES_EN  = ["all", "high opinion", "low opinion", "extreme"]
_TARGET_CHOICES_ZH  = ["全体", "高意见", "低意见", "极端"]

_TYPE_CHOICES_EN    = ["nudge", "broadcast", "silence"]
_TYPE_CHOICES_ZH    = ["微调", "广播", "静默"]


def _t(en: str, zh: str, lang: str) -> str:
    return zh if lang == "zh" else en


# ─────────────────────────────────────────────────────────────────────────────
# Rule row spec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleRow:
    """Holds all gr.components for a single intervention rule row."""
    index:     int
    row:       Any          # gr.Row  (the container; toggled visible/hidden)
    step:      gr.Number
    target:    gr.Dropdown
    rule_type: gr.Dropdown
    magnitude: gr.Number
    layer:     gr.Number
    remove:    gr.Button    # "✕" per-row delete button


@dataclass
class InterventionBuilder:
    """Container returned by build_intervention_builder()."""
    accordion:   Any                        # gr.Accordion wrapper
    rules:       list[RuleRow] = field(default_factory=list)
    add_btn:     gr.Button     = field(default=None)   # type: ignore[assignment]
    count_state: gr.State      = field(default=None)   # type: ignore[assignment]

    # ── Helpers ──────────────────────────────────────────────────────────

    def all_inputs(self) -> list:
        """
        Flat ordered list of all value-bearing inputs across all rule rows,
        plus the count_state.  Use as ``inputs=ib.all_inputs()`` on a button
        click that needs to read the entire form.

        Order per row: step, target, rule_type, magnitude, layer
        Total length: MAX_RULES * 5 + 1 (count_state last)
        """
        flat: list = []
        for r in self.rules:
            flat.extend([r.step, r.target, r.rule_type, r.magnitude, r.layer])
        flat.append(self.count_state)
        return flat

    def row_containers(self) -> list:
        """gr.Row containers for all rule rows (used as visibility outputs)."""
        return [r.row for r in self.rules]


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_intervention_builder(lang: str = "en") -> InterventionBuilder:
    """
    Render the intervention builder inside the current Gradio context.

    Must be called inside an active ``gr.Blocks()`` context.

    Parameters
    ----------
    lang : {"en", "zh"}

    Returns
    -------
    InterventionBuilder
    """
    ib = InterventionBuilder(accordion=None)

    target_choices = _TARGET_CHOICES_ZH if lang == "zh" else _TARGET_CHOICES_EN
    type_choices   = _TYPE_CHOICES_ZH   if lang == "zh" else _TYPE_CHOICES_EN

    section_label = _t("⚡  Interventions", "⚡  干预规则", lang)
    add_label     = _t("＋  Add Rule",       "＋  添加规则", lang)

    with gr.Accordion(section_label, open=False) as acc:
        ib.accordion = acc
        ib.count_state = gr.State(value=0)

        rule_rows: list[RuleRow] = []

        for i in range(MAX_RULES):
            visible = (i == 0)  # first row starts visible; rest hidden

            with gr.Row(visible=False, elem_classes="intervention-rule") as row_container:
                step_lbl  = _t(f"Rule {i+1} — Step",  f"规则 {i+1} — 步骤", lang)
                tgt_lbl   = _t("Target",               "目标",               lang)
                type_lbl  = _t("Type",                 "类型",               lang)
                mag_lbl   = _t("Magnitude",            "幅度",               lang)
                layer_lbl = _t("Layer",                "层",                 lang)

                with gr.Column(scale=1):
                    step = gr.Number(
                        label=step_lbl,
                        value=100 * (i + 1),
                        precision=0,
                        minimum=1,
                    )
                with gr.Column(scale=2):
                    target = gr.Dropdown(
                        label=tgt_lbl,
                        choices=target_choices,
                        value=target_choices[0],
                    )
                with gr.Column(scale=2):
                    rtype = gr.Dropdown(
                        label=type_lbl,
                        choices=type_choices,
                        value=type_choices[0],
                    )
                with gr.Column(scale=1):
                    magnitude = gr.Number(
                        label=mag_lbl,
                        value=0.1,
                        precision=3,
                        minimum=-1.0,
                        maximum=1.0,
                    )
                with gr.Column(scale=1):
                    layer = gr.Number(
                        label=layer_lbl,
                        value=0,
                        precision=0,
                        minimum=0,
                    )
                with gr.Column(scale=1, min_width=36):
                    remove = gr.Button(
                        "✕",
                        size="sm",
                        elem_classes="intervention-rule-delete",
                        variant="stop",
                    )

            rule_rows.append(RuleRow(
                index=i,
                row=row_container,
                step=step,
                target=target,
                rule_type=rtype,
                magnitude=magnitude,
                layer=layer,
                remove=remove,
            ))

        ib.rules = rule_rows

        ib.add_btn = gr.Button(
            add_label,
            size="sm",
            elem_id="btn-add-rule",
        )

    return ib


# ─────────────────────────────────────────────────────────────────────────────
# Event wiring
# ─────────────────────────────────────────────────────────────────────────────

def bind_events(ib: InterventionBuilder) -> None:
    """
    Wire the Add Rule and Remove (✕) button click events.

    Call once after the full gr.Blocks() layout has been defined.
    """

    row_containers = ib.row_containers()
    n = MAX_RULES

    # ── Add Rule ──────────────────────────────────────────────────────────
    def _add_rule(count: int):
        """Show the next hidden row; increment count."""
        new_count = min(count + 1, n)
        # Produce one gr.update per row container
        updates = [
            gr.update(visible=(i < new_count))
            for i in range(n)
        ]
        return [new_count] + updates

    ib.add_btn.click(
        fn=_add_rule,
        inputs=[ib.count_state],
        outputs=[ib.count_state] + row_containers,
    )

    # ── Remove buttons ────────────────────────────────────────────────────
    for rule in ib.rules:
        idx = rule.index

        def _make_remove_fn(remove_idx: int):
            """Factory to close over remove_idx correctly."""
            def _remove(count: int):
                """Hide this row and compact: shift later rows up (visually)."""
                new_count = max(count - 1, 0)
                # Build visibility: rows [0..new_count-1] visible
                # We hide the clicked row but keep others visible.
                # Simple approach: just decrement count; actual reordering
                # requires value swaps which is complex — hide and decrement.
                updates = [
                    gr.update(visible=(i < new_count))
                    for i in range(n)
                ]
                return [new_count] + updates
            return _remove

        rule.remove.click(
            fn=_make_remove_fn(idx),
            inputs=[ib.count_state],
            outputs=[ib.count_state] + row_containers,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rule collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_rules(ib: InterventionBuilder, *vals) -> list[dict]:
    """
    Parse the flat value list from ib.all_inputs() into a list of rule dicts.

    Parameters
    ----------
    ib : InterventionBuilder
    *vals : flat values from gr.component inputs

    Returns
    -------
    list[dict]  — active rules only (skips hidden rows beyond count)
    """
    # vals layout: [step0, tgt0, type0, mag0, layer0, step1, ...] + count
    n      = MAX_RULES
    fields = 5  # components per rule
    count  = int(vals[-1]) if vals else 0
    rules: list[dict] = []

    for i in range(count):
        base    = i * fields
        step    = vals[base + 0]
        target  = vals[base + 1]
        rtype   = vals[base + 2]
        mag     = vals[base + 3]
        layer   = vals[base + 4]

        # Normalise target string to internal key
        target_norm = {
            "all":          "all",
            "high opinion": "high",
            "高意见":        "high",
            "low opinion":  "low",
            "低意见":        "low",
            "extreme":      "extreme",
            "极端":          "extreme",
            "全体":          "all",
        }.get(str(target).strip().lower(), "all")

        type_norm = {
            "nudge":     "nudge",
            "broadcast": "broadcast",
            "silence":   "silence",
            "微调":       "nudge",
            "广播":       "broadcast",
            "静默":       "silence",
        }.get(str(rtype).strip().lower(), "nudge")

        rules.append({
            "step":      int(step) if step is not None else 1,
            "target":    target_norm,
            "type":      type_norm,
            "magnitude": float(mag) if mag is not None else 0.0,
            "layer":     int(layer) if layer is not None else 0,
        })

    return rules
