"""
app.py

Opinion Dynamics Simulation Dashboard — Gradio entry point (v2).

Launch:
    python app.py
    python app.py --port 7860 --share

Architecture
------------
Single gr.Blocks with sidebar-driven multi-page navigation.
7 pages are rendered as gr.Group containers; only one is visible at a time.
The sidebar (200px fixed) persists across all pages.

Pages
-----
  P1  home              — Welcome + preset loader
  P2  dashboard_settings— Refresh / display preferences
  P3  model_config      — Model parameters (all Accordions)
  P4  analysis_config   — Output dir, AI, report format
  P5  intervention      — Intervention rules
  P6  experiment        — Checklist (A) + Live monitor (B)
  P7  results           — Dynamic / Static / Features / AI Report

Session isolation
-----------------
One SimulationRunner per Gradio session stored in gr.State.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr

from core.defaults import DEFAULTS
from core.config_bridge import build_config_from_ui, build_analysis_config_from_ui
from core.runner import SimulationRunner
from core.validator import validate_all, summarize

from ui.components.sidebar import (
    SidebarComponents,
    build_sidebar,
    update_status,
    bind_nav_events,
    NAV_ITEMS,
)
from ui.panels.page_welcome import (
    WelcomeComponents,
    build_welcome_page,
    refresh_lang as welcome_refresh_lang,
    on_preset_selected,
    PRESET_KEYS,
)
from ui.panels.page_dashboard_settings import (
    DashboardSettingsComponents,
    build_dashboard_settings_page,
)
from ui.panels.page_model_config import (
    ModelConfigComponents,
    build_model_config_page,
)
from ui.panels.page_analysis_config import (
    AnalysisConfigComponents,
    build_analysis_config_page,
)
from ui.panels.page_intervention import (
    InterventionPageComponents,
    build_intervention_page,
    collect_rules,
)
from ui.panels.page_experiment import (
    ExperimentComponents,
    build_experiment_page,
    render_checklist,
    render_snapshot,
    update_checklist,
    transition_to_monitor,
    transition_to_checklist,
    on_run_complete,
)
from ui.panels.page_results import (
    ResultsComponents,
    build_results_page,
    render_features_table,
    switch_subtab,
    populate_from_run,
    _SUBTABS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page key registry (must match NAV_ITEMS order in sidebar.py)
# ─────────────────────────────────────────────────────────────────────────────

ALL_PAGES = [
    "home",
    "dashboard_settings",
    "model_config",
    "analysis_config",
    "intervention",
    "experiment",
    "results",
]


def switch_page(target: str) -> list:
    """Return 7 gr.update(visible=...) — one per page group."""
    return [gr.update(visible=(p == target)) for p in ALL_PAGES]


# ─────────────────────────────────────────────────────────────────────────────
# CSS path
# ─────────────────────────────────────────────────────────────────────────────

_CSS_PATH = Path(__file__).parent / "assets" / "custom.css"
_CSS = _CSS_PATH.read_text(encoding="utf-8") if _CSS_PATH.exists() else ""


# ─────────────────────────────────────────────────────────────────────────────
# App builder
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    with gr.Blocks(
        title="Opinion Dynamics Simulation",
        css=_CSS,
    ) as demo:

        # ── Session state ─────────────────────────────────────────────────
        runner_state = gr.State(lambda: SimulationRunner())
        lang_state   = gr.State("zh")

        # ── Top bar ───────────────────────────────────────────────────────
        with gr.Row(elem_id="top-bar"):
            gr.HTML(
                '<a id="top-bar-logo" href="#">🔬 Opinion Dynamics</a>'
            )
            top_status_html = gr.HTML(
                value='<div id="top-bar-status">● 就绪</div>',
                elem_id="top-bar-status-wrap",
            )
            gr.HTML('<div id="top-bar-spacer"></div>')
            # Language toggle (thin buttons, right-aligned)
            with gr.Row(elem_id="lang-toggle-group"):
                lang_zh_top = gr.Button(
                    "中文", size="sm",
                    elem_id="top-lang-zh",
                    elem_classes="lang-btn selected",
                )
                lang_en_top = gr.Button(
                    "EN", size="sm",
                    elem_id="top-lang-en",
                    elem_classes="lang-btn",
                )

        # ── App shell: sidebar + content ──────────────────────────────────
        with gr.Row(elem_id="app-shell", equal_height=False):

            # ── Sidebar ───────────────────────────────────────────────────
            with gr.Column(elem_id="sidebar", min_width=200, scale=0):
                sidebar = build_sidebar(lang="zh")

            # ── Page content area ──────────────────────────────────────────
            with gr.Column(elem_id="page-content", scale=1):

                # P1 — Home (visible by default)
                with gr.Group(visible=True, elem_id="page-home") as page_home:
                    welcome = build_welcome_page(lang="zh")

                # P2 — Dashboard Settings
                with gr.Group(visible=False, elem_id="page-dashboard") as page_dashboard:
                    dash_settings = build_dashboard_settings_page(
                        lang="zh", defaults=DEFAULTS
                    )

                # P3 — Model Config
                with gr.Group(visible=False, elem_id="page-model") as page_model:
                    model_cfg = build_model_config_page(
                        lang="zh", defaults=DEFAULTS
                    )

                # P4 — Analysis Config
                with gr.Group(visible=False, elem_id="page-analysis-cfg") as page_analysis_cfg:
                    analysis_cfg = build_analysis_config_page(
                        lang="zh", defaults=DEFAULTS
                    )

                # P5 — Intervention
                with gr.Group(visible=False, elem_id="page-intervention") as page_intervention:
                    interv = build_intervention_page(lang="zh")

                # P6 — Experiment
                with gr.Group(visible=False, elem_id="page-experiment") as page_experiment:
                    experiment = build_experiment_page(lang="zh", defaults=DEFAULTS)

                # P7 — Results
                with gr.Group(visible=False, elem_id="page-results") as page_results_grp:
                    results = build_results_page(lang="zh")

        # ── All page groups list (same order as ALL_PAGES) ─────────────────
        all_page_groups = [
            page_home,
            page_dashboard,
            page_model,
            page_analysis_cfg,
            page_intervention,
            page_experiment,
            page_results_grp,
        ]

        # ── Aggregated param components ────────────────────────────────────
        # Merge all config pages into one flat dict for validator + sidebar.
        # Keys must match core/config_bridge.py.
        all_param_components: dict = {
            **model_cfg.param_components,
            **analysis_cfg.param_components,
            **dash_settings.param_components,
        }
        # intervention_rules comes from gr.State, handled separately
        param_keys   = list(all_param_components.keys())
        param_inputs = list(all_param_components.values())

        def _values_to_dict(*vals) -> dict:
            return dict(zip(param_keys, vals))

        # ──────────────────────────────────────────────────────────────────
        # Sidebar navigation event wiring
        # ──────────────────────────────────────────────────────────────────

        bind_nav_events(
            sidebar           = sidebar,
            page_groups       = all_page_groups,
            switch_fn         = switch_page,
            all_param_inputs  = param_inputs,
            param_keys        = param_keys,
            lang_state        = lang_state,
            runner_state      = runner_state,
        )

        # ──────────────────────────────────────────────────────────────────
        # Language toggle
        # ──────────────────────────────────────────────────────────────────

        def _switch_lang(new_lang: str, active_page: str, *param_vals):
            """Hot-swap all language-sensitive labels."""
            ui_v  = dict(zip(param_keys, param_vals))
            items = validate_all(ui_v)
            s_html, prog = update_status(
                items=items, lang=new_lang, active_page=active_page
            )
            welcome_updates = welcome_refresh_lang(new_lang)
            top_zh_cls = "lang-btn selected" if new_lang == "zh" else "lang-btn"
            top_en_cls = "lang-btn selected" if new_lang == "en"  else "lang-btn"
            return (
                new_lang,          # lang_state
                s_html,            # sidebar.status_html
                prog,              # sidebar.progress_md
                *welcome_updates,  # hero, features, hint, zh-btn, en-btn, preset-dd
                gr.update(elem_classes=top_zh_cls),  # top lang zh button
                gr.update(elem_classes=top_en_cls),  # top lang en button
            )

        _lang_switch_outputs = [
            lang_state,
            sidebar.status_html,
            sidebar.progress_md,
            welcome.hero_html,
            welcome.feature_html,
            welcome.hint_html,
            welcome.lang_zh_btn,
            welcome.lang_en_btn,
            welcome.preset_dropdown,
            lang_zh_top,
            lang_en_top,
        ]

        lang_zh_top.click(
            fn=lambda ap, *pv: _switch_lang("zh", ap, *pv),
            inputs=[sidebar.active_state] + param_inputs,
            outputs=_lang_switch_outputs,
        )
        lang_en_top.click(
            fn=lambda ap, *pv: _switch_lang("en", ap, *pv),
            inputs=[sidebar.active_state] + param_inputs,
            outputs=_lang_switch_outputs,
        )

        # ──────────────────────────────────────────────────────────────────
        # P1 Welcome — Start button → model_config
        # ──────────────────────────────────────────────────────────────────

        welcome.start_btn.click(
            fn=lambda: switch_page("model_config"),
            inputs=[],
            outputs=all_page_groups,
        )

        # ──────────────────────────────────────────────────────────────────
        # P1 Welcome — Preset load
        # ──────────────────────────────────────────────────────────────────

        def _load_preset(preset_key: str, lang: str):
            """Load preset → back-fill all param components."""
            if not preset_key:
                return [gr.update()] * len(param_inputs) + ["", False]
            try:
                from config.switcher import ConfigSwitcher
                from core.config_bridge import extract_ui_values_from_config
                switcher = ConfigSwitcher()
                sim_cfg, _ = switcher.resolve_theme(preset_key)
                ui_vals = extract_ui_values_from_config(sim_cfg)
                updates = [
                    gr.update(value=ui_vals.get(k, DEFAULTS.get(k)))
                    for k in param_keys
                ]
            except Exception:
                updates = [gr.update()] * len(param_inputs)

            status_msg, visible = on_preset_selected(preset_key, lang)
            return updates + [status_msg, visible]

        welcome.preset_dropdown.change(
            fn=_load_preset,
            inputs=[welcome.preset_dropdown, lang_state],
            outputs=param_inputs + [
                welcome.preset_status_md,
                welcome.preset_status_md,  # visible flag handled below
            ],
        )

        # ──────────────────────────────────────────────────────────────────
        # "Save & Continue" nav buttons across config pages
        # ──────────────────────────────────────────────────────────────────

        _page_flow = [
            (dash_settings.save_btn,   "model_config"),
            (model_cfg.next_btn,       "analysis_config"),
            (model_cfg.back_btn,       "dashboard_settings"),
            (analysis_cfg.next_btn,    "intervention"),
            (analysis_cfg.back_btn,    "model_config"),
            (interv.next_btn,          "experiment"),
            (interv.back_btn,          "analysis_config"),
        ]

        for btn, target in _page_flow:
            btn.click(
                fn=lambda t=target: switch_page(t),
                inputs=[],
                outputs=all_page_groups,
            )

        # ──────────────────────────────────────────────────────────────────
        # P6-A Checklist: re-validate whenever experiment page is navigated to
        # (sidebar nav button for "experiment" already calls bind_nav_events,
        #  but we also need to refresh checklist content + snapshot)
        # ──────────────────────────────────────────────────────────────────

        def _refresh_experiment_page(lang: str, rules: list, *param_vals):
            """Re-compute checklist and snapshot when entering P6."""
            ui_v = dict(zip(param_keys, param_vals))
            ui_v["intervention_rules"] = collect_rules(rules)
            items = validate_all(ui_v)
            checklist_update, summary_update, run_btn_update = update_checklist(
                items, lang
            )
            snap_update = gr.update(value=render_snapshot(ui_v, lang))
            return checklist_update, summary_update, run_btn_update, snap_update

        # Bind to the experiment nav button click (sidebar index 5)
        sidebar.nav_buttons[5].click(
            fn=_refresh_experiment_page,
            inputs=[lang_state, interv.rules_state] + param_inputs,
            outputs=[
                experiment.checklist_html,
                experiment.summary_html,
                experiment.run_btn,
                experiment.snapshot_html,
            ],
        )

        # ──────────────────────────────────────────────────────────────────
        # P6-A Export YAML
        # ──────────────────────────────────────────────────────────────────

        def _export_yaml(*vals):
            import yaml as _yaml
            ui_v = _values_to_dict(*vals)
            cfg  = build_config_from_ui(ui_v)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as f:
                _yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
                return f.name

        experiment.export_btn.click(
            fn=_export_yaml,
            inputs=param_inputs,
            outputs=[gr.File(visible=False)],
        )

        # ──────────────────────────────────────────────────────────────────
        # P6-A Confirm & Run → transition to monitor + start streaming
        # ──────────────────────────────────────────────────────────────────

        def _on_confirm_run(runner: SimulationRunner):
            """Switch UI to monitor phase."""
            return transition_to_monitor()

        experiment.run_btn.click(
            fn=_on_confirm_run,
            inputs=[runner_state],
            outputs=[
                experiment.phase_a_group,
                experiment.phase_b_group,
                experiment.run_btn,
                experiment.pause_btn,
                experiment.stop_btn,
            ],
        )

        # Streaming simulation — chained after phase transition
        def _run_stream(runner: SimulationRunner, lang: str, *vals):
            ui_v = _values_to_dict(*vals)
            yield from runner.run_stream(
                ui_v,
                refresh_every=int(ui_v.get("refresh_every", 10)),
            )

        experiment.run_btn.click(
            fn=_run_stream,
            inputs=[runner_state, lang_state] + param_inputs,
            outputs=[
                experiment.status_md,
                experiment.metric_polar,   # sigma → polar metric
                experiment.metric_step,    # mean  → step (reused slot)
                experiment.metric_events,  # impact → events
                experiment.metric_clusters,# events → clusters (reused)
                experiment.metric_time,    # consensus → time (reused)
                experiment.plot_polar,
                experiment.plot_spatial,
                experiment.plot_hist,
                experiment.plot_events,
                experiment.pause_btn,
                experiment.stop_btn,
            ],
        )

        # ──────────────────────────────────────────────────────────────────
        # P6-B Pause / Resume toggle
        # ──────────────────────────────────────────────────────────────────

        def _toggle_pause(runner: SimulationRunner):
            if runner.is_paused:
                runner.resume()
                return gr.update(value="⏸ 暂停")
            else:
                runner.pause()
                return gr.update(value="▶ 继续")

        experiment.pause_btn.click(
            fn=_toggle_pause,
            inputs=[runner_state],
            outputs=[experiment.pause_btn],
        )

        # ──────────────────────────────────────────────────────────────────
        # P6-B Stop
        # ──────────────────────────────────────────────────────────────────

        def _stop(runner: SimulationRunner):
            runner.stop()
            return gr.update(interactive=False)

        experiment.stop_btn.click(
            fn=_stop,
            inputs=[runner_state],
            outputs=[experiment.pause_btn],
        )

        # ──────────────────────────────────────────────────────────────────
        # P6-B "← Reconfigure" link → back to checklist phase
        # ──────────────────────────────────────────────────────────────────

        # This is a gr.HTML element; we hook a hidden button pattern.
        # For simplicity, we expose a visible button that mirrors its intent.
        # (Full JS bridging would require a gr.Button with JS forward.)
        _reconfigure_btn = gr.Button(
            "← 重新配置",
            elem_id="btn-reconfigure",
            elem_classes="btn-secondary",
            size="sm",
            visible=True,
        )

        def _reconfigure(runner: SimulationRunner):
            runner.stop()
            return transition_to_checklist()

        _reconfigure_btn.click(
            fn=_reconfigure,
            inputs=[runner_state],
            outputs=[
                experiment.phase_a_group,
                experiment.phase_b_group,
                experiment.pause_btn,
                experiment.stop_btn,
            ],
        )

        # ──────────────────────────────────────────────────────────────────
        # P6-B View Results → switch page to results
        # ──────────────────────────────────────────────────────────────────

        experiment.view_results_btn.click(
            fn=lambda: switch_page("results"),
            inputs=[],
            outputs=all_page_groups,
        )

        # ──────────────────────────────────────────────────────────────────
        # P7 Results — sub-tab switching
        # ──────────────────────────────────────────────────────────────────

        _subtab_groups = [
            results.dynamic_group,
            results.static_group,
            results.features_group,
            results.report_group,
        ]

        for btn, key in zip(results.tab_btns, _SUBTABS):
            btn.click(
                fn=lambda k=key: switch_subtab(k),
                inputs=[],
                outputs=_subtab_groups + results.tab_btns,
            )

        # ──────────────────────────────────────────────────────────────────
        # P7 Results — Run Analysis (static dashboard + features)
        # ──────────────────────────────────────────────────────────────────

        def _run_analysis(runner: SimulationRunner, *vals):
            import pandas as _pd
            from analysis.manager import run_analysis as _run_analysis_mgr

            if runner.engine is None:
                return None, gr.update(value=render_features_table({}, "zh"))

            ui_v  = _values_to_dict(*vals)
            a_cfg = build_analysis_config_from_ui(ui_v)

            try:
                result = _run_analysis_mgr(runner.engine, a_cfg)
                fig    = runner.get_dashboard_figure(
                    layer_idx=int(ui_v.get("layer_idx", 0))
                )
                summary = result.pipeline_output.get("summary", {})
                feat_html = render_features_table(summary, "zh")
                return fig, gr.update(value=feat_html)
            except Exception as exc:
                err_html = render_features_table({}, "zh")
                return None, gr.update(value=err_html)

        results.run_analysis_btn.click(
            fn=_run_analysis,
            inputs=[runner_state] + param_inputs,
            outputs=[results.dashboard_img, results.features_html],
        )

        # ──────────────────────────────────────────────────────────────────
        # P7 Results — Generate AI Report
        # ──────────────────────────────────────────────────────────────────

        def _generate_report(runner: SimulationRunner, *vals):
            from analysis.manager import run_analysis as _run_analysis_mgr

            if runner.engine is None:
                return gr.update(value="_No simulation data._"), gr.update(visible=False)

            ui_v  = _values_to_dict(*vals)
            a_cfg = build_analysis_config_from_ui(ui_v)
            a_cfg["visual"]["enabled"] = False

            try:
                result = _run_analysis_mgr(runner.engine, a_cfg)
                md_path = result.report_paths.get("md", "")
                text    = open(md_path, encoding="utf-8").read() if md_path else "_Report not found._"
                import markdown as _md
                html_text = f'<div id="report-preview-area">{_md.markdown(text)}</div>'
                return gr.update(value=html_text), gr.update(visible=bool(md_path))
            except Exception as exc:
                return gr.update(value=f"_Error: {exc}_"), gr.update(visible=False)

        results.gen_report_btn.click(
            fn=_generate_report,
            inputs=[runner_state] + param_inputs,
            outputs=[results.report_html, results.report_download_btn],
        )
        results.gen_report_btn_inner.click(
            fn=_generate_report,
            inputs=[runner_state] + param_inputs,
            outputs=[results.report_html, results.report_download_btn],
        )

        # ──────────────────────────────────────────────────────────────────
        # On load: initialise preset list + run initial validation
        # ──────────────────────────────────────────────────────────────────

        def _on_load():
            # Populate preset dropdown
            try:
                from config.switcher import ConfigSwitcher
                names = ConfigSwitcher().list_themes()
                preset_choices = ["(none)"] + names
            except Exception:
                preset_choices = ["(none)"]

            # Initial validation with defaults
            items = validate_all(DEFAULTS)
            s_html, prog = update_status(
                items=items, lang="zh", active_page="home"
            )
            return (
                gr.update(choices=preset_choices),
                s_html,
                prog,
            )

        demo.load(
            fn=_on_load,
            outputs=[
                welcome.preset_dropdown,
                sidebar.status_html,
                sidebar.progress_md,
            ],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opinion Dynamics Simulation Dashboard v2"
    )
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=6657)
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