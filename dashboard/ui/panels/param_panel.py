"""Left parameter panel for the simulation dashboard."""

from __future__ import annotations

import gradio as gr

from config.switcher import ConfigSwitcher
from core.defaults import DEFAULTS, get_choices, get_info, get_label
from ui.components.intervention_builder import build_intervention_builder


def _num(key: str, lang: str, precision: int, **kwargs) -> gr.Number:
    return gr.Number(
        label=get_label(key, lang),
        info=get_info(key, lang),
        value=DEFAULTS[key],
        precision=precision,
        **kwargs,
    )


def _drop(key: str, lang: str) -> gr.Dropdown:
    return gr.Dropdown(
        label=get_label(key, lang),
        info=get_info(key, lang),
        choices=get_choices(key),
        value=DEFAULTS[key],
    )


def _check(key: str, lang: str) -> gr.Checkbox:
    return gr.Checkbox(label=get_label(key, lang), info=get_info(key, lang), value=DEFAULTS[key])


def build_param_panel(lang: str = "en") -> dict[str, gr.Component]:
    """Build the full sidebar configuration panel (number-only numeric inputs)."""
    c: dict[str, gr.Component] = {}
    switcher = ConfigSwitcher()

    with gr.Group(elem_id="preset-row"):
        with gr.Row():
            c["ui_lang"] = gr.Dropdown(
                label=get_label("_lang_label", lang),
                choices=[("English", "en"), ("中文", "zh")],
                value=lang,
                scale=2,
            )
            c["preset"] = gr.Dropdown(
                label=get_label("_sec_preset", lang), choices=switcher.list_themes(), value=None, scale=4
            )
        with gr.Row():
            c["load_preset_btn"] = gr.Button(get_label("_btn_load_preset", lang), size="sm")
            c["export_yaml_btn"] = gr.Button(get_label("_btn_export_yaml", lang), size="sm")

    with gr.Accordion(get_label("_sec_agent", lang), open=True):
        with gr.Row():
            c["num_agents"] = _num("num_agents", lang, 0, minimum=10, maximum=5000, step=10)
            c["opinion_layers"] = _num("opinion_layers", lang, 0, minimum=1, maximum=10, step=1)
        with gr.Row():
            c["total_steps"] = _num("total_steps", lang, 0, minimum=1, maximum=100000, step=10)
            c["seed"] = _num("seed", lang, 0, minimum=0, step=1)
        with gr.Row():
            c["record_history"] = _check("record_history", lang)
            c["init_type"] = _drop("init_type", lang)
            c["init_split"] = _num("init_split", lang, 2, minimum=0, maximum=1, step=0.01)

    with gr.Accordion(get_label("_sec_dynamics", lang), open=True):
        with gr.Row():
            c["epsilon_base"] = _num("epsilon_base", lang, 3, minimum=0.01, maximum=1.0, step=0.01)
            c["mu_base"] = _num("mu_base", lang, 3, minimum=0.01, maximum=1.0, step=0.01)
        with gr.Row():
            c["alpha_mod"] = _num("alpha_mod", lang, 3, minimum=0.0, maximum=2.0, step=0.01)
            c["beta_mod"] = _num("beta_mod", lang, 3, minimum=0.0, maximum=2.0, step=0.01)
        c["backfire"] = _check("backfire", lang)

    with gr.Accordion(get_label("_sec_field", lang), open=False):
        with gr.Row():
            c["field_alpha"] = _num("field_alpha", lang, 2, minimum=0.1, step=0.1)
            c["field_beta"] = _num("field_beta", lang, 4, minimum=0.0001, step=0.001)
        c["temporal_window"] = _num("temporal_window", lang, 1, minimum=1, step=1)

    with gr.Accordion(get_label("_sec_topo", lang), open=False):
        with gr.Row():
            c["topo_threshold"] = _num("topo_threshold", lang, 3, minimum=0, maximum=1, step=0.01)
            c["radius_base"] = _num("radius_base", lang, 3, minimum=0.01, step=0.01)
        c["radius_dynamic"] = _num("radius_dynamic", lang, 3, minimum=0.01, step=0.01)

    with gr.Accordion(get_label("_sec_network", lang), open=False):
        c["net_type"] = _drop("net_type", lang)
        with gr.Row():
            c["sw_k"] = _num("sw_k", lang, 0, minimum=2, step=1)
            c["sw_p"] = _num("sw_p", lang, 3, minimum=0, maximum=1, step=0.01)
            c["sf_m"] = _num("sf_m", lang, 0, minimum=1, step=1)

    with gr.Accordion(get_label("_sec_spatial", lang), open=False):
        c["spatial_type"] = _drop("spatial_type", lang)
        with gr.Row():
            c["n_clusters"] = _num("n_clusters", lang, 0, minimum=1, step=1)
            c["cluster_std"] = _num("cluster_std", lang, 3, minimum=0.01, step=0.01)

    with gr.Accordion(get_label("_sec_exo", lang), open=False):
        c["exo_enabled"] = _check("exo_enabled", lang)
        with gr.Row():
            c["exo_seed"] = _num("exo_seed", lang, 0, minimum=0, step=1)
            c["exo_lambda"] = _num("exo_lambda", lang, 3, minimum=0.0, step=0.01)
            c["exo_intensity_shape"] = _num("exo_intensity_shape", lang, 2, minimum=0.1, step=0.1)
        with gr.Row():
            c["exo_intensity_min"] = _num("exo_intensity_min", lang, 2, minimum=0.0, step=0.1)
            c["exo_polarity_min"] = _num("exo_polarity_min", lang, 2, minimum=-1.0, maximum=1.0, step=0.01)
            c["exo_polarity_max"] = _num("exo_polarity_max", lang, 2, minimum=-1.0, maximum=1.0, step=0.01)
        c["exo_concentration"] = gr.Textbox(
            label=get_label("exo_concentration", lang), info=get_info("exo_concentration", lang), value=DEFAULTS["exo_concentration"]
        )

    with gr.Accordion(get_label("_sec_endo", lang), open=False):
        c["endo_enabled"] = _check("endo_enabled", lang)
        with gr.Row():
            c["endo_seed"] = _num("endo_seed", lang, 0, minimum=0, step=1)
            c["endo_threshold"] = _num("endo_threshold", lang, 3, minimum=0.0, maximum=1.0, step=0.01)
            c["endo_grid"] = _num("endo_grid", lang, 0, minimum=2, step=1)
        with gr.Row():
            c["endo_min_agents"] = _num("endo_min_agents", lang, 0, minimum=1, step=1)
            c["endo_cooldown"] = _num("endo_cooldown", lang, 0, minimum=0, step=1)
        with gr.Row():
            c["endo_base_intensity"] = _num("endo_base_intensity", lang, 2, minimum=0.0, step=0.1)
            c["endo_scale"] = _num("endo_scale", lang, 2, minimum=0.0, step=0.1)

    with gr.Accordion(get_label("_sec_cascade", lang), open=False):
        c["cascade_enabled"] = _check("cascade_enabled", lang)
        with gr.Row():
            c["cascade_seed"] = _num("cascade_seed", lang, 0, minimum=0, step=1)
            c["cascade_bg_lambda"] = _num("cascade_bg_lambda", lang, 3, minimum=0.0, step=0.01)
            c["cascade_mu_mult"] = _num("cascade_mu_mult", lang, 3, minimum=0.0, step=0.01)
            c["cascade_decay"] = _num("cascade_decay", lang, 3, minimum=0.0, step=0.01)

    with gr.Accordion(get_label("_sec_online", lang), open=False):
        c["online_enabled"] = _check("online_enabled", lang)
        with gr.Row():
            c["online_seed"] = _num("online_seed", lang, 0, minimum=0, step=1)
            c["online_check"] = _num("online_check", lang, 0, minimum=1, step=1)
            c["online_smooth"] = _num("online_smooth", lang, 0, minimum=1, step=1)
        with gr.Row():
            c["online_min_community"] = _num("online_min_community", lang, 0, minimum=1, step=1)
            c["online_base"] = _num("online_base", lang, 2, minimum=0.0, step=0.1)
            c["online_scale"] = _num("online_scale", lang, 2, minimum=0.0, step=0.1)

    with gr.Accordion(get_label("_sec_analysis", lang), open=False):
        c["output_dir"] = gr.Textbox(label=get_label("output_dir", lang), info=get_info("output_dir", lang), value=DEFAULTS["output_dir"])
        with gr.Row():
            c["output_lang"] = _drop("output_lang", lang)
            c["layer_idx"] = _num("layer_idx", lang, 0, minimum=0, step=1)
            c["refresh_every"] = _num("refresh_every", lang, 0, minimum=1, step=1)
        with gr.Row():
            c["include_trends"] = _check("include_trends", lang)
            c["save_timeseries"] = _check("save_timeseries", lang)
            c["save_features_json"] = _check("save_features_json", lang)

    c.update(build_intervention_builder(lang=lang))
    return c
