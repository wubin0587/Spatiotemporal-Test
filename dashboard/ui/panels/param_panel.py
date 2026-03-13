"""
ui/panels/param_panel.py

Left-side parameter panel for the Opinion Dynamics Simulation Dashboard.

This module extracts the param-panel logic from app.py into a standalone,
importable unit so it can be unit-tested and reused independently of the
full app assembly.

Public API
----------
build_param_panel(lang) -> ParamPanel
    Render all parameter accordions in the current Gradio context.
    Returns a ParamPanel dataclass.

ParamPanel
    .components : dict[str, gr.Component]
        Flat dict of all parameter components, keyed by parameter name.
        This is the same dict that app.py passes to the runner as ui_values.

    .keys() / .values() / .items()
        Proxy the underlying .components dict for drop-in compatibility.

    .as_inputs() -> list[gr.Component]
        Ordered list of all components (stable ordering = dict insertion order).

    .as_keys() -> list[str]
        Corresponding key list matching .as_inputs() order.

refresh_labels(panel, lang) -> list[gr.update]
    Return one gr.update(label=..., info=...) per component in as_inputs()
    order, enabling hot-swap language toggle without rebuilding the tree.

Notes
-----
- No sliders — all numeric inputs use gr.Number.
- Section accordions are built with open=True for the two most-used
  sections and open=False for all others.
- The Intervention Builder accordion is rendered at the bottom of the
  panel and wired separately via ui.components.intervention_builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import gradio as gr

from core.defaults import DEFAULTS, get_label, get_info, get_choices
from ui.components.intervention_builder import (
    InterventionBuilder,
    build_intervention_builder,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _num(key: str, lang: str, precision: int = 2, **kw) -> gr.Number:
    return gr.Number(
        label=get_label(key, lang),
        info=get_info(key, lang),
        value=DEFAULTS[key],
        precision=precision,
        **kw,
    )


def _check(key: str, lang: str) -> gr.Checkbox:
    return gr.Checkbox(
        label=get_label(key, lang),
        info=get_info(key, lang),
        value=DEFAULTS[key],
    )


def _drop(key: str, lang: str) -> gr.Dropdown:
    return gr.Dropdown(
        label=get_label(key, lang),
        info=get_info(key, lang),
        choices=get_choices(key),
        value=DEFAULTS[key],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ParamPanel dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParamPanel:
    """Container for all parameter-panel gr.Components."""

    components: dict[str, gr.Component] = field(default_factory=dict)
    intervention_builder: InterventionBuilder = field(default=None)  # type: ignore[assignment]

    # ── dict-like proxy ──────────────────────────────────────────────────

    def __getitem__(self, key: str) -> gr.Component:
        return self.components[key]

    def __contains__(self, key: str) -> bool:
        return key in self.components

    def keys(self) -> list[str]:
        return list(self.components.keys())

    def values(self) -> list[gr.Component]:
        return list(self.components.values())

    def items(self) -> list[tuple[str, gr.Component]]:
        return list(self.components.items())

    def get(self, key: str, default=None):
        return self.components.get(key, default)

    # ── Ordered accessors ────────────────────────────────────────────────

    def as_inputs(self) -> list[gr.Component]:
        """Stable ordered list of all parameter components."""
        return list(self.components.values())

    def as_keys(self) -> list[str]:
        """Corresponding key list matching as_inputs() order."""
        return list(self.components.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_param_panel(lang: str = "en") -> ParamPanel:
    """
    Render the left-side parameter panel inside the current Gradio context.

    Must be called inside an active ``gr.Blocks()`` context and inside a
    ``gr.Column`` that serves as the left pane.

    Parameters
    ----------
    lang : {"en", "zh"}

    Returns
    -------
    ParamPanel
    """
    c: dict[str, gr.Component] = {}

    # ── Preset / YAML row ─────────────────────────────────────────────────
    with gr.Group(elem_id="preset-row"):
        with gr.Row():
            c["preset"] = gr.Dropdown(
                label=get_label("_sec_preset", lang),
                choices=["(none)"],
                value="(none)",
                scale=4,
            )
            c["load_preset_btn"] = gr.Button(
                get_label("_btn_load_preset", lang),
                scale=1, size="sm",
            )
            c["export_yaml_btn"] = gr.Button(
                get_label("_btn_export_yaml", lang),
                scale=1, size="sm",
            )

    # ── Agent & Simulation ────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_agent", lang), open=True):
        with gr.Row():
            c["num_agents"]     = _num("num_agents",     lang, 0, minimum=10,  maximum=5000)
            c["opinion_layers"] = _num("opinion_layers", lang, 0, minimum=1,   maximum=10)
        with gr.Row():
            c["total_steps"]    = _num("total_steps",    lang, 0, minimum=1,   maximum=100000)
            c["seed"]           = _num("seed",           lang, 0, minimum=0)
        with gr.Row():
            c["record_history"] = _check("record_history", lang)
        with gr.Row():
            c["init_type"]      = _drop("init_type",  lang)
            c["init_split"]     = _num("init_split",  lang, 2, minimum=0.0, maximum=1.0)

    # ── Dynamics ──────────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_dynamics", lang), open=True):
        with gr.Row():
            c["epsilon_base"]  = _num("epsilon_base",  lang, 3, minimum=0.01, maximum=1.0)
            c["mu_base"]       = _num("mu_base",       lang, 3, minimum=0.01, maximum=1.0)
        with gr.Row():
            c["alpha_mod"]     = _num("alpha_mod",     lang, 3, minimum=0.0,  maximum=2.0)
            c["beta_mod"]      = _num("beta_mod",      lang, 3, minimum=0.0,  maximum=2.0)
        c["backfire"]          = _check("backfire", lang)

    # ── Influence Field ───────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_field", lang), open=False):
        with gr.Row():
            c["field_alpha"]     = _num("field_alpha",     lang, 2, minimum=0.1)
            c["field_beta"]      = _num("field_beta",      lang, 4, minimum=0.001)
        c["temporal_window"]     = _num("temporal_window", lang, 1, minimum=1.0)

    # ── Topology ──────────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_topo", lang), open=False):
        with gr.Row():
            c["topo_threshold"] = _num("topo_threshold", lang, 3, minimum=0.0, maximum=1.0)
            c["radius_base"]    = _num("radius_base",    lang, 3, minimum=0.01)
        c["radius_dynamic"]     = _num("radius_dynamic", lang, 3, minimum=0.01)

    # ── Network ───────────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_network", lang), open=False):
        c["net_type"] = _drop("net_type", lang)
        with gr.Row():
            c["sw_k"]   = _num("sw_k",  lang, 0, minimum=2)
            c["sw_p"]   = _num("sw_p",  lang, 3, minimum=0.0, maximum=1.0)
        c["sf_m"]       = _num("sf_m",  lang, 0, minimum=1)

    # ── Spatial ───────────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_spatial", lang), open=False):
        c["spatial_type"] = _drop("spatial_type", lang)
        with gr.Row():
            c["n_clusters"]  = _num("n_clusters",  lang, 0, minimum=1)
            c["cluster_std"] = _num("cluster_std", lang, 3, minimum=0.01)

    # ── Exogenous Events ──────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_exo", lang), open=False):
        c["exo_enabled"]  = _check("exo_enabled", lang)
        with gr.Row():
            c["exo_seed"]   = _num("exo_seed",   lang, 0, minimum=0)
            c["exo_lambda"] = _num("exo_lambda", lang, 3, minimum=0.0)
        with gr.Row():
            c["exo_intensity_shape"] = _num("exo_intensity_shape", lang, 2, minimum=0.1)
            c["exo_intensity_min"]   = _num("exo_intensity_min",   lang, 2, minimum=0.0)
        with gr.Row():
            c["exo_polarity_min"] = _num("exo_polarity_min", lang, 2, minimum=-1.0, maximum=0.0)
            c["exo_polarity_max"] = _num("exo_polarity_max", lang, 2, minimum=0.0,  maximum=1.0)
        c["exo_concentration"] = gr.Textbox(
            label=get_label("exo_concentration", lang),
            info=get_info("exo_concentration", lang),
            value=DEFAULTS["exo_concentration"],
            placeholder="e.g. 1,1,1",
        )

    # ── Endogenous Threshold Events ───────────────────────────────────────
    with gr.Accordion(get_label("_sec_endo", lang), open=False):
        c["endo_enabled"] = _check("endo_enabled", lang)
        with gr.Row():
            c["endo_seed"]      = _num("endo_seed",      lang, 0, minimum=0)
            c["endo_threshold"] = _num("endo_threshold", lang, 3, minimum=0.0)
        c["endo_monitor"] = _drop("endo_monitor", lang)
        with gr.Row():
            c["endo_grid"]     = _num("endo_grid",     lang, 0, minimum=2)
            c["endo_cooldown"] = _num("endo_cooldown", lang, 0, minimum=0)
        c["endo_min_agents"]     = _num("endo_min_agents", lang, 0, minimum=1)
        with gr.Row():
            c["endo_base_intensity"] = _num("endo_base_intensity", lang, 2, minimum=0.0)
            c["endo_scale"]          = _num("endo_scale",          lang, 2, minimum=0.0)

    # ── Cascade Events ────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_cascade", lang), open=False):
        c["cascade_enabled"] = _check("cascade_enabled", lang)
        with gr.Row():
            c["cascade_seed"]      = _num("cascade_seed",      lang, 0, minimum=0)
            c["cascade_bg_lambda"] = _num("cascade_bg_lambda", lang, 3, minimum=0.0)
        with gr.Row():
            c["cascade_mu_mult"] = _num("cascade_mu_mult", lang, 3, minimum=0.0, maximum=1.0)
            c["cascade_decay"]   = _num("cascade_decay",   lang, 3, minimum=0.0, maximum=1.0)

    # ── Online Resonance Events ───────────────────────────────────────────
    with gr.Accordion(get_label("_sec_online", lang), open=False):
        c["online_enabled"] = _check("online_enabled", lang)
        with gr.Row():
            c["online_seed"]  = _num("online_seed",  lang, 0, minimum=0)
            c["online_check"] = _num("online_check", lang, 0, minimum=1)
        with gr.Row():
            c["online_smooth"]        = _num("online_smooth",        lang, 0, minimum=1)
            c["online_min_community"] = _num("online_min_community", lang, 0, minimum=1)
        with gr.Row():
            c["online_convergence"] = _num("online_convergence", lang, 4, minimum=0.0)
            c["online_conflict"]    = _num("online_conflict",    lang, 4, minimum=0.0)
        with gr.Row():
            c["online_base"]  = _num("online_base",  lang, 2, minimum=0.0)
            c["online_scale"] = _num("online_scale", lang, 2, minimum=0.0)

    # ── Analysis Output ───────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_analysis", lang), open=False):
        c["output_dir"] = gr.Textbox(
            label=get_label("output_dir", lang),
            info=get_info("output_dir", lang),
            value=DEFAULTS["output_dir"],
        )
        with gr.Row():
            c["output_lang"] = _drop("output_lang", lang)
            c["layer_idx"]   = _num("layer_idx",    lang, 0, minimum=0)
        with gr.Row():
            c["include_trends"]     = _check("include_trends",     lang)
            c["save_timeseries"]    = _check("save_timeseries",    lang)
            c["save_features_json"] = _check("save_features_json", lang)
        c["refresh_every"] = _num("refresh_every", lang, 0, minimum=1)

    # ── Intervention Builder ──────────────────────────────────────────────
    ib = build_intervention_builder(lang=lang)

    return ParamPanel(components=c, intervention_builder=ib)


# ─────────────────────────────────────────────────────────────────────────────
# Language-refresh helper
# ─────────────────────────────────────────────────────────────────────────────

def refresh_labels(panel: ParamPanel, lang: str) -> list:
    """
    Return one gr.update(label=..., info=...) per component in
    panel.as_inputs() order, enabling hot-swap language toggle.

    Components that don't have a matching key in LABELS (e.g. preset,
    load_preset_btn, export_yaml_btn) receive gr.update() (no-op).
    """
    updates = []
    for key in panel.as_keys():
        try:
            lbl  = get_label(key, lang)
            info = get_info(key, lang)
            updates.append(gr.update(label=lbl, info=info))
        except (KeyError, TypeError):
            updates.append(gr.update())
    return updates
