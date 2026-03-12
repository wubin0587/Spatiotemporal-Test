"""
app.py

Opinion Dynamics Simulation Dashboard — Gradio entry point.

Launch:
    python app.py
    python app.py --port 7860 --share

Architecture
------------
  Left column  (scale=3): parameter panel — all simulation inputs
  Right column (scale=7): tabbed main panel
    Tab 0: Monitor  — real-time charts + metric cards
    Tab 1: Analysis — post-run dashboard + feature summary
    Tab 2: Report   — AI-assisted report generation

Language toggle (en / zh) in the header hot-swaps all component labels
without a full page reload.

Session isolation
-----------------
One SimulationRunner instance per Gradio session is stored in gr.State.
Never use module-level globals for simulation state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure project root is on the path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr
import numpy as np

from core.defaults    import DEFAULTS, get_label, get_info, get_choices
from core.config_bridge import build_config_from_ui, build_analysis_config_from_ui
from core.runner      import SimulationRunner


# ─────────────────────────────────────────────────────────────────────────────
# CSS  (light, clean, research-tool aesthetic)
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Fonts ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

body, .gradio-container {
    font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif !important;
    background: #f1f5f9 !important;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
#app-header {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 14px 24px 10px;
    margin-bottom: 0;
}
#app-header h1 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
}
#status-badge {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #475569 !important;
    padding: 2px 10px !important;
    border-radius: 12px !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── Left parameter panel ───────────────────────────────────────────────── */
#param-panel {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    min-height: calc(100vh - 80px);
    padding: 0 !important;
    overflow-y: auto;
}
#param-panel .accordion {
    border: none !important;
    border-bottom: 1px solid #f1f5f9 !important;
    border-radius: 0 !important;
    margin: 0 !important;
}
#param-panel .accordion button {
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: #334155 !important;
    padding: 8px 14px !important;
    background: #f8fafc !important;
    letter-spacing: 0.01em;
}
#param-panel label span {
    font-size: 0.73rem !important;
    color: #475569 !important;
}
#param-panel input[type="number"],
#param-panel input[type="text"],
#param-panel select {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    border-color: #e2e8f0 !important;
    border-radius: 4px !important;
}

/* ── Run control row ────────────────────────────────────────────────────── */
#run-controls {
    background: #ffffff;
    border-top: 1px solid #e2e8f0;
    padding: 10px 14px !important;
    position: sticky;
    bottom: 0;
}
#btn-run {
    background: #0d9488 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    border-radius: 5px !important;
}
#btn-run:hover { background: #0f766e !important; }
#btn-stop {
    background: #fee2e2 !important;
    color: #b91c1c !important;
    border: 1px solid #fecaca !important;
    border-radius: 5px !important;
    font-size: 0.78rem !important;
}
#btn-reset {
    background: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 5px !important;
    font-size: 0.78rem !important;
}
#btn-pause {
    background: #fffbeb !important;
    color: #92400e !important;
    border: 1px solid #fde68a !important;
    border-radius: 5px !important;
    font-size: 0.78rem !important;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
.metric-card .wrap {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
}
.metric-card input {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    color: #0f172a !important;
    text-align: center !important;
    border: none !important;
    background: transparent !important;
}
.metric-card label span {
    font-size: 0.68rem !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Status bar ─────────────────────────────────────────────────────────── */
#step-label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #475569 !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
    padding: 4px 10px !important;
}

/* ── Tab titles ─────────────────────────────────────────────────────────── */
.tab-nav button {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
    color: #0d9488 !important;
    border-bottom-color: #0d9488 !important;
}

/* ── Chart images ───────────────────────────────────────────────────────── */
.chart-tile img {
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    width: 100%;
    object-fit: contain;
}
.chart-tile .label {
    font-size: 0.72rem !important;
    color: #64748b !important;
}

/* ── Preset / YAML row ──────────────────────────────────────────────────── */
#preset-row select {
    font-size: 0.76rem !important;
}

/* ── Summary table ──────────────────────────────────────────────────────── */
#summary-df table {
    font-size: 0.76rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Param panel builder
# ─────────────────────────────────────────────────────────────────────────────

def _num(key: str, lang: str, precision: int = 2, **kw) -> gr.Number:
    lbl, info = get_label(key, lang), get_info(key, lang)
    return gr.Number(
        label=lbl, info=info,
        value=DEFAULTS[key],
        precision=precision,
        **kw,
    )


def _check(key: str, lang: str) -> gr.Checkbox:
    lbl, info = get_label(key, lang), get_info(key, lang)
    return gr.Checkbox(label=lbl, info=info, value=DEFAULTS[key])


def _drop(key: str, lang: str) -> gr.Dropdown:
    lbl, info = get_label(key, lang), get_info(key, lang)
    return gr.Dropdown(
        label=lbl, info=info,
        choices=get_choices(key),
        value=DEFAULTS[key],
    )


def build_param_panel(lang: str = "en") -> dict:
    """Build the left-side parameter panel. Returns a flat dict of components."""
    c: dict = {}

    # ── Preset row ─────────────────────────────────────────────────────────
    with gr.Group(elem_id="preset-row"):
        with gr.Row():
            c["preset"] = gr.Dropdown(
                label=get_label("_sec_preset", lang),
                choices=["(none)"],
                value="(none)",
                scale=4,
            )
            c["load_preset_btn"] = gr.Button(
                get_label("_btn_load_preset", lang), scale=1, size="sm")
            c["export_yaml_btn"] = gr.Button(
                get_label("_btn_export_yaml", lang), scale=1, size="sm")

    # ── Agent & Simulation ─────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_agent", lang), open=True):
        with gr.Row():
            c["num_agents"]     = _num("num_agents", lang, 0,
                                       minimum=10, maximum=5000)
            c["opinion_layers"] = _num("opinion_layers", lang, 0,
                                       minimum=1, maximum=10)
        with gr.Row():
            c["total_steps"]    = _num("total_steps", lang, 0,
                                       minimum=1, maximum=100000)
            c["seed"]           = _num("seed", lang, 0, minimum=0)
        with gr.Row():
            c["record_history"] = _check("record_history", lang)
        with gr.Row():
            c["init_type"]      = _drop("init_type", lang)
            c["init_split"]     = _num("init_split", lang, 2,
                                       minimum=0.0, maximum=1.0)

    # ── Dynamics ───────────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_dynamics", lang), open=True):
        with gr.Row():
            c["epsilon_base"]  = _num("epsilon_base", lang, 3,
                                      minimum=0.01, maximum=1.0)
            c["mu_base"]       = _num("mu_base", lang, 3,
                                      minimum=0.01, maximum=1.0)
        with gr.Row():
            c["alpha_mod"]     = _num("alpha_mod", lang, 3,
                                      minimum=0.0, maximum=2.0)
            c["beta_mod"]      = _num("beta_mod", lang, 3,
                                      minimum=0.0, maximum=2.0)
        c["backfire"]          = _check("backfire", lang)

    # ── Influence Field ────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_field", lang), open=False):
        with gr.Row():
            c["field_alpha"]      = _num("field_alpha", lang, 2, minimum=0.1)
            c["field_beta"]       = _num("field_beta", lang, 4, minimum=0.001)
        c["temporal_window"]      = _num("temporal_window", lang, 1, minimum=1.0)

    # ── Topology ───────────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_topo", lang), open=False):
        with gr.Row():
            c["topo_threshold"] = _num("topo_threshold", lang, 3,
                                       minimum=0.0, maximum=1.0)
            c["radius_base"]    = _num("radius_base", lang, 3, minimum=0.01)
        c["radius_dynamic"]     = _num("radius_dynamic", lang, 3, minimum=0.01)

    # ── Network ────────────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_network", lang), open=False):
        c["net_type"] = _drop("net_type", lang)
        with gr.Row():
            c["sw_k"]   = _num("sw_k", lang, 0, minimum=2)
            c["sw_p"]   = _num("sw_p", lang, 3, minimum=0.0, maximum=1.0)
        c["sf_m"]       = _num("sf_m", lang, 0, minimum=1)

    # ── Spatial ────────────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_spatial", lang), open=False):
        c["spatial_type"] = _drop("spatial_type", lang)
        with gr.Row():
            c["n_clusters"]  = _num("n_clusters", lang, 0, minimum=1)
            c["cluster_std"] = _num("cluster_std", lang, 3, minimum=0.01)

    # ── Exogenous Events ───────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_exo", lang), open=False):
        c["exo_enabled"]  = _check("exo_enabled", lang)
        with gr.Row():
            c["exo_seed"]   = _num("exo_seed", lang, 0, minimum=0)
            c["exo_lambda"] = _num("exo_lambda", lang, 3, minimum=0.0)
        with gr.Row():
            c["exo_intensity_shape"] = _num("exo_intensity_shape", lang, 2,
                                            minimum=0.1)
            c["exo_intensity_min"]   = _num("exo_intensity_min", lang, 2,
                                            minimum=0.0)
        with gr.Row():
            c["exo_polarity_min"] = _num("exo_polarity_min", lang, 2,
                                         minimum=-1.0, maximum=0.0)
            c["exo_polarity_max"] = _num("exo_polarity_max", lang, 2,
                                         minimum=0.0, maximum=1.0)
        c["exo_concentration"] = gr.Textbox(
            label=get_label("exo_concentration", lang),
            info=get_info("exo_concentration", lang),
            value=DEFAULTS["exo_concentration"],
            placeholder="e.g. 1,1,1",
        )

    # ── Endogenous Threshold Events ────────────────────────────────────────
    with gr.Accordion(get_label("_sec_endo", lang), open=False):
        c["endo_enabled"] = _check("endo_enabled", lang)
        with gr.Row():
            c["endo_seed"]      = _num("endo_seed", lang, 0, minimum=0)
            c["endo_threshold"] = _num("endo_threshold", lang, 3, minimum=0.0)
        c["endo_monitor"] = _drop("endo_monitor", lang)
        with gr.Row():
            c["endo_grid"]       = _num("endo_grid", lang, 0, minimum=2)
            c["endo_cooldown"]   = _num("endo_cooldown", lang, 0, minimum=0)
        c["endo_min_agents"]     = _num("endo_min_agents", lang, 0, minimum=1)
        with gr.Row():
            c["endo_base_intensity"] = _num("endo_base_intensity", lang, 2,
                                            minimum=0.0)
            c["endo_scale"]          = _num("endo_scale", lang, 2, minimum=0.0)

    # ── Cascade Events ─────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_cascade", lang), open=False):
        c["cascade_enabled"] = _check("cascade_enabled", lang)
        with gr.Row():
            c["cascade_seed"]      = _num("cascade_seed", lang, 0, minimum=0)
            c["cascade_bg_lambda"] = _num("cascade_bg_lambda", lang, 3,
                                          minimum=0.0)
        with gr.Row():
            c["cascade_mu_mult"] = _num("cascade_mu_mult", lang, 3,
                                        minimum=0.0, maximum=1.0)
            c["cascade_decay"]   = _num("cascade_decay", lang, 3,
                                        minimum=0.0, maximum=1.0)

    # ── Online Resonance Events ────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_online", lang), open=False):
        c["online_enabled"] = _check("online_enabled", lang)
        with gr.Row():
            c["online_seed"]  = _num("online_seed", lang, 0, minimum=0)
            c["online_check"] = _num("online_check", lang, 0, minimum=1)
        with gr.Row():
            c["online_smooth"]      = _num("online_smooth", lang, 0, minimum=1)
            c["online_min_community"] = _num("online_min_community", lang, 0,
                                              minimum=1)
        with gr.Row():
            c["online_convergence"] = _num("online_convergence", lang, 4,
                                           minimum=0.0)
            c["online_conflict"]    = _num("online_conflict", lang, 4,
                                           minimum=0.0)
        with gr.Row():
            c["online_base"]  = _num("online_base", lang, 2, minimum=0.0)
            c["online_scale"] = _num("online_scale", lang, 2, minimum=0.0)

    # ── Analysis Output ────────────────────────────────────────────────────
    with gr.Accordion(get_label("_sec_analysis", lang), open=False):
        c["output_dir"]  = gr.Textbox(
            label=get_label("output_dir", lang),
            info=get_info("output_dir", lang),
            value=DEFAULTS["output_dir"],
        )
        with gr.Row():
            c["output_lang"]  = _drop("output_lang", lang)
            c["layer_idx"]    = _num("layer_idx", lang, 0, minimum=0)
        with gr.Row():
            c["include_trends"]      = _check("include_trends", lang)
            c["save_timeseries"]     = _check("save_timeseries", lang)
            c["save_features_json"]  = _check("save_features_json", lang)
        c["refresh_every"] = _num("refresh_every", lang, 0, minimum=1)

    return c


# ─────────────────────────────────────────────────────────────────────────────
# Monitor tab
# ─────────────────────────────────────────────────────────────────────────────

def build_monitor_tab(lang: str = "en") -> dict:
    m: dict = {}

    with gr.Row():
        m["status_md"] = gr.Markdown(
            "● Ready",
            elem_id="step-label",
        )

    # Metric cards
    with gr.Row():
        with gr.Column(min_width=90, elem_classes="metric-card"):
            m["metric_sigma"]     = gr.Number(label="σ polarization",
                                               value=0.0, precision=4,
                                               interactive=False)
        with gr.Column(min_width=90, elem_classes="metric-card"):
            m["metric_mean"]      = gr.Number(label="μ mean opinion",
                                               value=0.0, precision=4,
                                               interactive=False)
        with gr.Column(min_width=90, elem_classes="metric-card"):
            m["metric_impact"]    = gr.Number(label="mean impact",
                                               value=0.0, precision=4,
                                               interactive=False)
        with gr.Column(min_width=90, elem_classes="metric-card"):
            m["metric_events"]    = gr.Number(label="events",
                                               value=0, precision=0,
                                               interactive=False)
        with gr.Column(min_width=90, elem_classes="metric-card"):
            m["metric_consensus"] = gr.Number(label="consensus",
                                               value=0.0, precision=4,
                                               interactive=False)

    # 2 × 2 chart grid
    with gr.Row():
        with gr.Column(elem_classes="chart-tile"):
            m["plot_timeseries"] = gr.Plot(label="Polarization Timeline")
        with gr.Column(elem_classes="chart-tile"):
            m["plot_spatial"]    = gr.Plot(label="Spatial Opinion Distribution")
    with gr.Row():
        with gr.Column(elem_classes="chart-tile"):
            m["plot_histogram"]  = gr.Plot(label="Opinion Histogram")
        with gr.Column(elem_classes="chart-tile"):
            m["plot_events"]     = gr.Plot(label="Event Timeline")

    # Pause / Stop inside the tab (mirrors the sidebar buttons)
    with gr.Row():
        m["pause_btn"] = gr.Button(
            "⏸  Pause", size="sm", interactive=False, elem_id="btn-pause")
        m["stop_btn"]  = gr.Button(
            "⏹  Stop",  size="sm", interactive=False, elem_id="btn-stop",
            variant="stop")

    return m


def monitor_output_list(m: dict) -> list:
    """
    Return the outputs list for run_btn.click(), in the same order
    as SimulationRunner.run_stream() yields.
    """
    return [
        m["status_md"],       # 0
        m["metric_sigma"],    # 1
        m["metric_mean"],     # 2
        m["metric_impact"],   # 3
        m["metric_events"],   # 4
        m["metric_consensus"],# 5
        m["plot_timeseries"], # 6
        m["plot_spatial"],    # 7
        m["plot_histogram"],  # 8
        m["plot_events"],     # 9
        m["pause_btn"],       # 10
        m["stop_btn"],        # 11
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis tab (post-run)
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_tab(lang: str = "en") -> dict:
    a: dict = {}
    a["dashboard_plot"] = gr.Plot(label="Simulation Dashboard")
    a["summary_df"]     = gr.DataFrame(
        label="Feature Summary",
        elem_id="summary-df",
        wrap=True,
    )
    with gr.Row():
        a["run_analysis_btn"] = gr.Button(
            "Run Analysis" if lang == "en" else "运行分析",
            variant="primary", size="sm",
        )
        a["download_btn"] = gr.DownloadButton(
            "Download Figures" if lang == "en" else "下载图表",
            size="sm",
        )
    a["analysis_status"] = gr.Markdown("")
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Report tab
# ─────────────────────────────────────────────────────────────────────────────

def build_report_tab(lang: str = "en") -> dict:
    r: dict = {}
    with gr.Row():
        r["report_fmt"]   = gr.Dropdown(
            label="Format" if lang == "en" else "格式",
            choices=["md", "html"],
            value="md",
        )
        r["report_title"] = gr.Textbox(
            label="Title (optional)" if lang == "en" else "标题（可选）",
            placeholder="auto-generated" if lang == "en" else "自动生成",
        )

    with gr.Accordion("AI Parser (optional)" if lang == "en" else "AI 解析（可选）",
                      open=False):
        r["ai_enabled"]   = gr.Checkbox(
            label="Enable AI analysis" if lang == "en" else "启用 AI 分析",
            value=False,
        )
        r["api_key"]      = gr.Textbox(
            label="API Key", type="password",
            placeholder="sk-..." if lang == "en" else "sk-...",
        )
        r["ai_model"]     = gr.Dropdown(
            label="Model",
            choices=["gpt-4o", "gpt-4-turbo", "claude-sonnet-4-6"],
            value="gpt-4o",
        )
        with gr.Row():
            r["narrative_mode"] = gr.Dropdown(
                label="Narrative Mode" if lang == "en" else "叙事风格",
                choices=["", "chronicle", "diagnostic",
                         "comparative", "predictive", "dramatic"],
                value="",
            )
            r["theme_name"] = gr.Dropdown(
                label="Theme" if lang == "en" else "主题",
                choices=[""],
                value="",
            )

    r["generate_btn"]  = gr.Button(
        get_label("_btn_generate_report", lang) if lang == "zh" else "📄  Generate Report",
        variant="primary",
    )
    r["report_preview"] = gr.Markdown(
        "_No report yet._" if lang == "en" else "_尚无报告。_"
    )
    with gr.Row():
        r["download_report_btn"] = gr.DownloadButton(
            "Download Report" if lang == "en" else "下载报告",
            size="sm",
        )
        r["download_features_btn"] = gr.DownloadButton(
            "Download Features" if lang == "en" else "下载特征数据",
            size="sm",
        )
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Gradio app assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    with gr.Blocks(
        title="Opinion Dynamics Simulation",
        css=_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif"],
            font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace"],
        ),
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────
        with gr.Row(elem_id="app-header"):
            with gr.Column(scale=8):
                gr.Markdown(
                    "# 🔬  Opinion Dynamics Simulation",
                )
            with gr.Column(scale=2):
                lang_toggle = gr.Radio(
                    choices=["en", "zh"],
                    value="en",
                    label="",
                    interactive=True,
                    elem_id="lang-toggle",
                )

        # ── Session state ─────────────────────────────────────────────────
        runner_state = gr.State(lambda: SimulationRunner())
        lang_state   = gr.State("en")

        # ── Main layout: left panel + right tabs ──────────────────────────
        with gr.Row(equal_height=False):

            # ── Left: parameter panel ──────────────────────────────────────
            with gr.Column(scale=3, elem_id="param-panel"):
                params = build_param_panel(lang="en")

                # Run controls (sticky footer)
                with gr.Row(elem_id="run-controls"):
                    run_btn   = gr.Button(
                        "▶  Run Simulation",
                        variant="primary", size="lg",
                        elem_id="btn-run",
                    )
                with gr.Row():
                    pause_sidebar = gr.Button(
                        "⏸  Pause", size="sm", interactive=False,
                        elem_id="btn-pause")
                    stop_sidebar  = gr.Button(
                        "⏹  Stop",  size="sm", interactive=False,
                        variant="stop", elem_id="btn-stop")
                    reset_btn     = gr.Button(
                        "↺  Reset", size="sm",
                        elem_id="btn-reset")

            # ── Right: tabs ────────────────────────────────────────────────
            with gr.Column(scale=7):
                with gr.Tabs() as tabs:

                    with gr.Tab("📡  Monitor"):
                        monitor = build_monitor_tab(lang="en")

                    with gr.Tab("📊  Analysis"):
                        analysis = build_analysis_tab(lang="en")

                    with gr.Tab("📄  Report"):
                        report = build_report_tab(lang="en")

        # ─────────────────────────────────────────────────────────────────
        # Helper: collect all UI param values
        # ─────────────────────────────────────────────────────────────────

        _param_inputs = list(params.values())
        _param_keys   = list(params.keys())

        def _values_to_dict(*vals) -> dict:
            return dict(zip(_param_keys, vals))

        # ─────────────────────────────────────────────────────────────────
        # Run button → streaming generator
        # ─────────────────────────────────────────────────────────────────

        def _run_stream(runner: SimulationRunner, refresh_every: int, *vals):
            ui_v = _values_to_dict(*vals)
            ui_v["refresh_every"] = refresh_every
            yield from runner.run_stream(ui_v, refresh_every=refresh_every)

        run_btn.click(
            fn=_run_stream,
            inputs=[runner_state, params["refresh_every"]] + _param_inputs,
            outputs=monitor_output_list(monitor),
        )

        # ── Pause toggle ──────────────────────────────────────────────────
        def _toggle_pause(runner: SimulationRunner):
            if runner.is_paused:
                runner.resume()
                return gr.update(value="⏸  Pause")
            else:
                runner.pause()
                return gr.update(value="▶  Resume")

        monitor["pause_btn"].click(
            fn=_toggle_pause,
            inputs=[runner_state],
            outputs=[monitor["pause_btn"]],
        )
        pause_sidebar.click(
            fn=_toggle_pause,
            inputs=[runner_state],
            outputs=[pause_sidebar],
        )

        # ── Stop ──────────────────────────────────────────────────────────
        def _stop(runner: SimulationRunner):
            runner.stop()
            return gr.update(value="⏸  Pause", interactive=False)

        monitor["stop_btn"].click(
            fn=_stop, inputs=[runner_state], outputs=[monitor["pause_btn"]])
        stop_sidebar.click(
            fn=_stop, inputs=[runner_state], outputs=[pause_sidebar])

        # ── Reset ─────────────────────────────────────────────────────────
        def _reset(runner: SimulationRunner):
            runner.reset()
            import gradio as _gr
            return (
                "● Ready",
                0.0, 0.0, 0.0, 0, 0.0,
                None, None, None, None,
                _gr.update(value="⏸  Pause", interactive=False),
                _gr.update(interactive=False),
            )

        reset_btn.click(
            fn=_reset,
            inputs=[runner_state],
            outputs=monitor_output_list(monitor),
        )

        # ── Analysis tab: run analysis ─────────────────────────────────────
        def _run_analysis(runner: SimulationRunner, *vals):
            if runner.engine is None:
                return None, "No completed simulation. Run the simulation first.", None

            import pandas as pd
            from core.config_bridge import build_analysis_config_from_ui
            from analysis.manager import run_analysis

            ui_v  = _values_to_dict(*vals)
            a_cfg = build_analysis_config_from_ui(ui_v)

            try:
                result = run_analysis(runner.engine, a_cfg)
                # Dashboard figure
                fig = runner.get_dashboard_figure(
                    layer_idx=int(ui_v.get("layer_idx", 0))
                )
                # Summary dataframe
                summary = result.pipeline_output.get("summary", {})
                rows = [(k, f"{v:.4f}" if isinstance(v, float) else str(v))
                        for k, v in summary.items()]
                df = pd.DataFrame(rows, columns=["metric", "value"])
                status_msg = (
                    f"✓ Analysis complete — "
                    f"{len(result.figure_paths)} figures, "
                    f"{len(result.report_paths)} reports"
                )
                return fig, df, status_msg
            except Exception as exc:
                return None, f"Analysis failed: {exc}", None

        analysis["run_analysis_btn"].click(
            fn=_run_analysis,
            inputs=[runner_state] + _param_inputs,
            outputs=[
                analysis["dashboard_plot"],
                analysis["summary_df"],
                analysis["analysis_status"],
            ],
        )

        # ── Report tab: generate ──────────────────────────────────────────
        def _generate_report(
            runner: SimulationRunner,
            report_fmt, report_title,
            ai_enabled, api_key, ai_model,
            narrative_mode, theme_name,
            *vals,
        ):
            if runner.engine is None:
                return "_No completed simulation._", None, None

            from core.config_bridge import build_analysis_config_from_ui
            from analysis.manager import run_analysis

            ui_v = _values_to_dict(*vals)
            ui_v.update({
                "report_fmt":     report_fmt,
                "report_title":   report_title,
                "ai_enabled":     ai_enabled,
                "api_key":        api_key,
                "ai_model":       ai_model,
                "narrative_mode": narrative_mode,
                "theme_name":     theme_name,
            })

            a_cfg = build_analysis_config_from_ui(ui_v)
            a_cfg["visual"]["enabled"] = False  # skip figures for report-only

            try:
                result = run_analysis(runner.engine, a_cfg)
                md_path = result.report_paths.get("md", "")
                if md_path:
                    text = open(md_path, encoding="utf-8").read()
                else:
                    text = "_Report file not found._"

                feat_path = result.feature_paths.get("summary_json")
                return text, md_path or None, feat_path
            except Exception as exc:
                return f"_Error: {exc}_", None, None

        report["generate_btn"].click(
            fn=_generate_report,
            inputs=[
                runner_state,
                report["report_fmt"], report["report_title"],
                report["ai_enabled"], report["api_key"], report["ai_model"],
                report["narrative_mode"], report["theme_name"],
            ] + _param_inputs,
            outputs=[
                report["report_preview"],
                report["download_report_btn"],
                report["download_features_btn"],
            ],
        )

        # ── Load preset ────────────────────────────────────────────────────
        def _load_preset(name: str):
            if name == "(none)" or not name:
                return [gr.update()] * len(_param_inputs)
            try:
                from config.switcher import ConfigSwitcher
                from core.config_bridge import extract_ui_values_from_config
                switcher = ConfigSwitcher()
                sim_cfg, _ = switcher.resolve_theme(name)
                ui_vals = extract_ui_values_from_config(sim_cfg)
                return [gr.update(value=ui_vals.get(k, DEFAULTS.get(k)))
                        for k in _param_keys]
            except Exception:
                return [gr.update()] * len(_param_inputs)

        params["load_preset_btn"].click(
            fn=_load_preset,
            inputs=[params["preset"]],
            outputs=_param_inputs,
        )

        # ── Export YAML ────────────────────────────────────────────────────
        def _export_yaml(*vals):
            import tempfile, yaml as _yaml
            ui_v = _values_to_dict(*vals)
            cfg  = build_config_from_ui(ui_v)
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                _yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
                return f.name

        params["export_yaml_btn"].click(
            fn=_export_yaml,
            inputs=_param_inputs,
            outputs=[gr.File(visible=False)],
        )

        # ── Populate preset list on load ──────────────────────────────────
        def _init_presets():
            try:
                from config.switcher import ConfigSwitcher
                names = ConfigSwitcher().list_themes()
                return gr.update(choices=["(none)"] + names)
            except Exception:
                return gr.update(choices=["(none)"])

        demo.load(fn=_init_presets, outputs=[params["preset"]])

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opinion Dynamics Simulation Dashboard")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()
