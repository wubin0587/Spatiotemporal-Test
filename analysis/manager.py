"""
analysis/manager.py

Analysis Manager
----------------
Unified entry point for running single-run simulation analysis.
Reads a parsed YAML config dict and selectively executes feature extraction,
AI parsing, report generation, and visualization steps.

YAML Config Schema
------------------
The config dict (pre-parsed from YAML) has the following structure:

    output:
      dir: "output/run_001"          # root output directory
      formats: ["md", "html"]        # report formats to generate
      lang: "zh"                     # "zh" | "en"
      save_figures: true             # whether to save static plots
      save_timeseries: true          # whether to save .npz timeseries
      save_features_json: true       # whether to save final features as JSON

    feature:
      enabled: true                  # run feature extraction
      layer_idx: 0                   # primary opinion layer for scalar metrics
      include_trends: true           # include trend metrics in summary

    parser:
      enabled: false                 # run AI analysis (requires api_key)
      api_key: ""                    # LLM API key (or set via env OPENAI_API_KEY)
      model: "gpt-4o"
      base_url: null                 # optional custom endpoint
      lang: "zh"
      fmt: "md"
      sections: ["opinion", "spatial", "topo", "event"]
      include_executive_summary: true
      max_tokens: 2048
      narrative_mode: null           # null | "chronicle" | "diagnostic" | "comparative" | "predictive" | "dramatic"
      theme: null                    # null | "concert_crowd" | "political_rally" | ... (auto-detect if null)

    report:
      enabled: true                  # assemble and save report
      formats: ["md"]                # "md" | "html" | "latex"
      include_toc: true
      include_meta: true
      include_snapshot: true
      title: null                    # null = auto-generated

    visual:
      enabled: false                 # generate static matplotlib figures
      dashboard: true                # composite dashboard
      opinion_distribution: true
      spatial_opinions: true
      opinion_timeseries: true
      impact_heatmap: false
      event_timeline: true
      polarization_evolution: true
      network_homophily: false
      dpi: 150

    simulation_meta:                 # optional metadata injected into reports
      n_agents: null
      n_steps: null
      model: null

Usage
-----
    from analysis.manager import run_analysis

    # engine is a completed SimulationEngine / StepExecutor instance
    results = run_analysis(engine, config)

    # Or load config from YAML file:
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    results = run_analysis(engine, config)

Returns
-------
    AnalysisResult  — dataclass with paths and extracted objects for downstream use.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Result Container
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    """
    Container for all outputs produced by run_analysis().

    Attributes
    ----------
    pipeline_output : dict
        Raw output from FeaturePipeline.run(). Always populated if
        feature extraction is enabled.
    report_paths : dict[str, str]
        Maps format → absolute file path of the saved report.
        e.g. {"md": "/output/run_001/report.md", "html": "..."}
    figure_paths : dict[str, str]
        Maps figure name → absolute file path of the saved figure.
    feature_paths : dict[str, str]
        Maps artifact type → absolute file path.
        e.g. {"timeseries_npz": "...", "features_json": "..."}
    parsed_sections : dict[str, str]
        AI-generated section texts (empty if parser disabled).
    config : dict
        The resolved config used for this run.
    errors : list[str]
        Non-fatal errors encountered during the run.
    """
    pipeline_output:  Dict[str, Any]  = field(default_factory=dict)
    report_paths:     Dict[str, str]  = field(default_factory=dict)
    figure_paths:     Dict[str, str]  = field(default_factory=dict)
    feature_paths:    Dict[str, str]  = field(default_factory=dict)
    parsed_sections:  Dict[str, str]  = field(default_factory=dict)
    config:           Dict[str, Any]  = field(default_factory=dict)
    errors:           List[str]       = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# Default Config
# ═════════════════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG: Dict[str, Any] = {
    "output": {
        "dir":                "output",
        "formats":            ["md"],
        "lang":               "zh",
        "save_figures":       True,
        "save_timeseries":    True,
        "save_features_json": True,
    },
    "feature": {
        "enabled":        True,
        "layer_idx":      0,
        "include_trends": True,
    },
    "parser": {
        "enabled":                   False,
        "api_key":                   "",
        "model":                     "gpt-4o",
        "base_url":                  None,
        "lang":                      "zh",
        "fmt":                       "md",
        "sections":                  ["opinion", "spatial", "topo", "event"],
        "include_executive_summary": True,
        "max_tokens":                2048,
        "narrative_mode":            None,
        "theme":                     None,
    },
    "report": {
        "enabled":          True,
        "formats":          ["md"],
        "include_toc":      True,
        "include_meta":     True,
        "include_snapshot": True,
        "title":            None,
    },
    "visual": {
        "enabled":                False,
        "dashboard":              True,
        "opinion_distribution":   True,
        "spatial_opinions":       True,
        "opinion_timeseries":     True,
        "impact_heatmap":         False,
        "event_timeline":         True,
        "polarization_evolution": True,
        "network_homophily":      False,
        "dpi":                    150,
    },
    "simulation_meta": {},
}


def _merge_config(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge user config over defaults.
    Top-level sections are merged key-by-key; unknown top-level keys are
    passed through as-is.
    """
    import copy
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    for section_key, section_val in user.items():
        if isinstance(section_val, dict) and section_key in cfg:
            cfg[section_key].update(section_val)
        else:
            cfg[section_key] = section_val
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Public Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def run_analysis(
    engine: Any,
    config: Dict[str, Any],
) -> AnalysisResult:
    """
    Run the full (or partial) analysis pipeline for a single simulation run.

    Parameters
    ----------
    engine : SimulationEngine / StepExecutor (duck-typed)
        A completed simulation engine exposing:
            .opinion_matrix     np.ndarray (N, L)
            .agent_positions    np.ndarray (N, 2)
            .impact_vector      np.ndarray (N,)
            .network_graph      nx.Graph
            .event_manager      EventManager
            .history            dict  (optional, for timeseries)
            .current_time       float
            .time_step          int
    config : dict
        Parsed YAML config. Missing keys fall back to _DEFAULT_CONFIG.
        See module docstring for full schema.

    Returns
    -------
    AnalysisResult
    """
    cfg    = _merge_config(config)
    result = AnalysisResult(config=cfg)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Feature Extraction ─────────────────────────────────────────────────
    pipeline_output: Dict[str, Any] = {}

    if cfg["feature"]["enabled"]:
        pipeline_output = _run_feature_extraction(engine, cfg, result)
        result.pipeline_output = pipeline_output
    else:
        logger.info("[Manager] Feature extraction disabled — skipping.")

    # ── 2. Save Feature Artifacts ─────────────────────────────────────────────
    if pipeline_output:
        _save_feature_artifacts(pipeline_output, cfg, result)

    # ── 3. AI Parser ──────────────────────────────────────────────────────────
    parsed_sections: Dict[str, str] = {}

    if cfg["parser"]["enabled"] and pipeline_output:
        parsed_sections = _run_parser(pipeline_output, cfg, result)
        result.parsed_sections = parsed_sections
    elif cfg["parser"]["enabled"]:
        msg = "[Manager] Parser enabled but no pipeline output available."
        logger.warning(msg)
        result.errors.append(msg)

    # ── 4. Report Generation ──────────────────────────────────────────────────
    if cfg["report"]["enabled"] and pipeline_output:
        _run_report(
            pipeline_output=pipeline_output,
            parsed_sections=parsed_sections,
            cfg=cfg,
            result=result,
        )
    elif cfg["report"]["enabled"]:
        msg = "[Manager] Report enabled but no pipeline output available."
        logger.warning(msg)
        result.errors.append(msg)

    # ── 5. Visualization ──────────────────────────────────────────────────────
    if cfg["visual"]["enabled"] and pipeline_output:
        _run_visualization(engine, pipeline_output, cfg, result)
    elif cfg["visual"]["enabled"]:
        msg = "[Manager] Visualization enabled but no pipeline output available."
        logger.warning(msg)
        result.errors.append(msg)

    _log_summary(result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Feature Extraction
# ═════════════════════════════════════════════════════════════════════════════

def _run_feature_extraction(
    engine: Any,
    cfg: Dict[str, Any],
    result: AnalysisResult,
) -> Dict[str, Any]:
    """Run FeaturePipeline and return its output dict."""
    from analysis.feature.pipeline import FeaturePipeline

    layer_idx = cfg["feature"].get("layer_idx", 0)

    try:
        pipe   = FeaturePipeline(engine, layer_idx=layer_idx)
        output = pipe.run()

        # Optionally recompute summary without trend metrics
        if not cfg["feature"].get("include_trends", True):
            from analysis.feature.composer import summarize_timeseries
            ts = output.get("timeseries", {})
            if ts:
                output["summary"] = summarize_timeseries(ts, include_trends=False)

        logger.info(
            "[Manager] Feature extraction complete. "
            f"Timeseries keys: {len(output.get('timeseries', {}))}. "
            f"Issues: {output.get('data_issues', [])}"
        )
        return output

    except Exception as exc:
        msg = f"[Manager] Feature extraction failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Save Feature Artifacts
# ═════════════════════════════════════════════════════════════════════════════

def _save_feature_artifacts(
    pipeline_output: Dict[str, Any],
    cfg: Dict[str, Any],
    result: AnalysisResult,
) -> None:
    from analysis.feature.io.exporter import save_summary, save_timeseries

    out_dir = Path(cfg["output"]["dir"])

    if cfg["output"].get("save_features_json", True):
        # Summary stats
        summary_path = out_dir / "features_summary.json"
        try:
            save_summary(pipeline_output.get("summary", {}), summary_path)
            result.feature_paths["summary_json"] = str(summary_path.resolve())
            logger.info(f"[Manager] Saved features summary → {summary_path}")
        except Exception as exc:
            result.errors.append(f"Failed to save features summary: {exc}")

        # Final snapshot features
        final_path = out_dir / "features_final.json"
        try:
            save_summary(pipeline_output.get("final", {}), final_path)
            result.feature_paths["final_json"] = str(final_path.resolve())
            logger.info(f"[Manager] Saved final features → {final_path}")
        except Exception as exc:
            result.errors.append(f"Failed to save final features: {exc}")

    if cfg["output"].get("save_timeseries", True):
        ts = pipeline_output.get("timeseries", {})
        if ts:
            ts_path = out_dir / "timeseries.npz"
            try:
                save_timeseries(ts, ts_path)
                result.feature_paths["timeseries_npz"] = str(ts_path.resolve())
                logger.info(f"[Manager] Saved timeseries → {ts_path}")
            except Exception as exc:
                result.errors.append(f"Failed to save timeseries: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — AI Parser
# ═════════════════════════════════════════════════════════════════════════════

def _run_parser(
    pipeline_output: Dict[str, Any],
    cfg: Dict[str, Any],
    result: AnalysisResult,
) -> Dict[str, str]:
    """
    Run the AI parser.

    Dispatch logic:
      - If narrative_mode or theme is set → composite (theme + narrative) parsing
      - Otherwise → standard ParserClient.parse_pipeline_result()
    """
    from analysis.parser.client import ParserClient

    pcfg    = cfg["parser"]
    api_key = pcfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        msg = "[Manager] Parser enabled but no API key provided (set api_key or OPENAI_API_KEY env var)."
        logger.warning(msg)
        result.errors.append(msg)
        return {}

    try:
        client = ParserClient(
            api_key=api_key,
            lang=pcfg.get("lang", "zh"),
            fmt=pcfg.get("fmt", "md"),
            model=pcfg.get("model", "gpt-4o"),
            base_url=pcfg.get("base_url"),
            sections=pcfg.get("sections", ParserClient.DEFAULT_SECTIONS),
            max_tokens=pcfg.get("max_tokens", 2048),
            verbose=True,
        )

        narrative_mode = pcfg.get("narrative_mode")
        theme_name     = pcfg.get("theme")

        if narrative_mode or theme_name:
            sections = _run_composite_parser(
                client=client,
                pipeline_output=pipeline_output,
                narrative_mode=narrative_mode,
                theme_name=theme_name,
                pcfg=pcfg,
                simulation_meta=cfg.get("simulation_meta") or {},
                result=result,
            )
        else:
            sections = client.parse_pipeline_result(
                pipeline_output=pipeline_output,
                include_executive_summary=pcfg.get("include_executive_summary", True),
                simulation_meta=cfg.get("simulation_meta") or None,
            )

        logger.info(f"[Manager] Parser complete. Sections: {list(sections.keys())}")
        return sections

    except Exception as exc:
        msg = f"[Manager] Parser failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        return {}


def _run_composite_parser(
    client: Any,
    pipeline_output: Dict[str, Any],
    narrative_mode: Optional[str],
    theme_name: Optional[str],
    pcfg: Dict[str, Any],
    simulation_meta: Dict[str, Any],
    result: AnalysisResult,
) -> Dict[str, str]:
    """
    Build composite (theme + narrative mode) prompts for each section,
    then call the LLM backend directly.
    Falls back to standard per-section parsing on any individual failure.
    """
    from analysis.parser.prompts import (
        composite_section_prompt,
        composite_executive_prompt,
        SYSTEM_PROMPT,
    )
    from analysis.parser.themes import ThemeEngine, THEME_REGISTRY
    from analysis.parser.narrative import NarrativeMode
    from analysis.parser.client import _partition_summary

    lang = pcfg.get("lang", "zh")
    fmt  = pcfg.get("fmt", "md")

    # ── Resolve theme ─────────────────────────────────────────────────────────
    theme = None
    if theme_name:
        try:
            theme = ThemeEngine().get(theme_name)
        except KeyError:
            msg = (
                f"[Manager] Unknown theme '{theme_name}'. "
                f"Available: {list(THEME_REGISTRY.keys())}. Falling back to auto-detect."
            )
            logger.warning(msg)
            result.errors.append(msg)

    if theme is None:
        theme = ThemeEngine().detect(pipeline_output.get("summary", {}))
        logger.info(f"[Manager] Auto-detected theme: {theme.name}")

    # ── Resolve narrative mode ────────────────────────────────────────────────
    mode = narrative_mode or NarrativeMode.CHRONICLE

    # ── Parse each section ────────────────────────────────────────────────────
    section_summaries = _partition_summary(pipeline_output.get("summary", {}))
    sections: Dict[str, str] = {}

    for section in pcfg.get("sections", ["opinion", "spatial", "topo", "event"]):
        metrics = section_summaries.get(section) or pipeline_output.get("final", {}).get(section, {})
        if not metrics:
            continue
        try:
            user_prompt = composite_section_prompt(
                section=section,
                metrics=metrics,
                theme=theme,
                narrative_mode=mode,
                lang=lang,
                fmt=fmt,
            )
            sections[section] = client._backend.call(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=pcfg.get("max_tokens", 2048),
            )
        except Exception as exc:
            msg = f"[Manager] Composite parse failed for '{section}': {exc}. Trying standard parse."
            logger.warning(msg)
            result.errors.append(msg)
            try:
                sections[section] = client.parse_section(section, metrics)
            except Exception:
                pass

    # ── Executive summary ─────────────────────────────────────────────────────
    if pcfg.get("include_executive_summary", True) and pipeline_output.get("summary"):
        try:
            exec_prompt = composite_executive_prompt(
                feature_summary=pipeline_output["summary"],
                theme=theme,
                narrative_mode=mode,
                lang=lang,
                fmt=fmt,
                simulation_meta=simulation_meta or None,
            )
            sections["executive_summary"] = client._backend.call(
                system=SYSTEM_PROMPT,
                user=exec_prompt,
                max_tokens=pcfg.get("max_tokens", 2048),
            )
        except Exception as exc:
            msg = f"[Manager] Executive summary (composite) failed: {exc}"
            logger.warning(msg)
            result.errors.append(msg)

    return sections


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Report Generation
# ═════════════════════════════════════════════════════════════════════════════

def _run_report(
    pipeline_output: Dict[str, Any],
    parsed_sections: Dict[str, str],
    cfg: Dict[str, Any],
    result: AnalysisResult,
) -> None:
    from analysis.report.builder import ReportBuilder

    rcfg    = cfg["report"]
    out_dir = Path(cfg["output"]["dir"])
    lang    = cfg["output"].get("lang", "zh")

    # report.formats takes priority; fall back to output.formats
    formats = rcfg.get("formats") or cfg["output"].get("formats", ["md"])

    for fmt in formats:
        try:
            builder = ReportBuilder(
                lang=lang,
                fmt=fmt,
                include_toc=rcfg.get("include_toc", True),
                include_meta=rcfg.get("include_meta", True),
                include_snapshot=rcfg.get("include_snapshot", True),
            )
            doc = builder.build(
                parsed_sections=parsed_sections or None,
                pipeline_output=pipeline_output,
                title=rcfg.get("title"),
                simulation_meta=cfg.get("simulation_meta") or None,
            )

            ext_map  = {"md": "md", "html": "html", "latex": "tex"}
            filename = f"report.{ext_map.get(fmt, fmt)}"
            abs_path = doc.save(str(out_dir / filename))

            result.report_paths[fmt] = abs_path
            logger.info(f"[Manager] Saved {fmt.upper()} report → {abs_path}")

        except Exception as exc:
            msg = f"[Manager] Report generation failed for format '{fmt}': {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — Visualization
# ═════════════════════════════════════════════════════════════════════════════

def _run_visualization(
    engine: Any,
    pipeline_output: Dict[str, Any],
    cfg: Dict[str, Any],
    result: AnalysisResult,
) -> None:
    """Generate and save static matplotlib figures per vcfg flags."""
    vcfg    = cfg["visual"]
    out_dir = Path(cfg["output"]["dir"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi     = vcfg.get("dpi", 150)

    # ── Pull engine state ─────────────────────────────────────────────────────
    positions     = getattr(engine, "agent_positions",  None)
    opinions      = getattr(engine, "opinion_matrix",   None)
    impact        = getattr(engine, "impact_vector",    None)
    history       = getattr(engine, "history",          {}) or {}
    network_graph = getattr(engine, "network_graph",    None)

    if positions is None or opinions is None or impact is None:
        msg = "[Manager] Engine is missing positions/opinions/impact — skipping visualization."
        logger.warning(msg)
        result.errors.append(msg)
        return

    layer_idx        = cfg["feature"].get("layer_idx", 0)
    history_opinions = history.get("opinions",    [])
    history_times    = history.get("time",        [])
    history_impact   = history.get("impact",      [])
    history_events   = history.get("num_events",  [])

    # ── Collect event data ────────────────────────────────────────────────────
    event_times = event_intensities = event_sources = event_locs = None
    try:
        locs, times, intensities, _, _ = engine.event_manager.get_state_vectors()
        if times is not None and len(times) > 0:
            event_times       = np.asarray(times,       dtype=float)
            event_intensities = np.asarray(intensities, dtype=float)
            event_locs        = np.asarray(locs,        dtype=float)
            # Attempt to get source labels; graceful fallback
            try:
                raw_sources = engine.event_manager.get_source_labels()
                event_sources = list(raw_sources)
            except Exception:
                event_sources = ["unknown"] * len(times)
    except Exception:
        pass

    # ── Import visualization (non-interactive backend) ────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        from analysis.visual import static as vis
    except ImportError as exc:
        msg = f"[Manager] Visualization skipped — matplotlib unavailable: {exc}"
        logger.warning(msg)
        result.errors.append(msg)
        return

    def _save(name: str, fig: Any) -> None:
        _save_and_record(fig, out_dir / f"{name}.png", dpi, name, result)

    # ── Dashboard (composite) ─────────────────────────────────────────────────
    if vcfg.get("dashboard", True):
        try:
            fig = vis.plot_simulation_dashboard(
                final_positions   = positions,
                final_opinions    = opinions,
                final_impact      = impact,
                history_opinions  = history_opinions or None,
                history_times     = history_times    or None,
                event_times       = event_times,
                event_intensities = event_intensities,
                event_sources     = event_sources,
                event_locs        = event_locs,
            )
            _save("dashboard", fig)
        except Exception as exc:
            result.errors.append(f"Dashboard figure failed: {exc}")

    # ── Opinion Distribution ──────────────────────────────────────────────────
    if vcfg.get("opinion_distribution", True):
        try:
            _save("opinion_distribution",
                  vis.plot_opinion_distribution(opinions, layer_idx=layer_idx))
        except Exception as exc:
            result.errors.append(f"Opinion distribution figure failed: {exc}")

    # ── Spatial Opinions ──────────────────────────────────────────────────────
    if vcfg.get("spatial_opinions", True):
        try:
            _save("spatial_opinions",
                  vis.plot_spatial_opinions(positions, opinions, impact=impact, layer_idx=layer_idx))
        except Exception as exc:
            result.errors.append(f"Spatial opinions figure failed: {exc}")

    # ── Opinion Timeseries ────────────────────────────────────────────────────
    if vcfg.get("opinion_timeseries", True) and history_opinions and history_times:
        try:
            _save("opinion_timeseries",
                  vis.plot_opinion_timeseries(history_opinions, history_times, layer_idx=layer_idx))
        except Exception as exc:
            result.errors.append(f"Opinion timeseries figure failed: {exc}")

    # ── Impact Heatmap ────────────────────────────────────────────────────────
    if vcfg.get("impact_heatmap", False):
        try:
            _save("impact_heatmap",
                  vis.plot_impact_heatmap(positions, impact, event_locs=event_locs))
        except Exception as exc:
            result.errors.append(f"Impact heatmap figure failed: {exc}")

    # ── Event Timeline ────────────────────────────────────────────────────────
    if vcfg.get("event_timeline", True) and event_times is not None:
        try:
            _save("event_timeline",
                  vis.plot_event_timeline(
                      event_times, event_intensities, event_sources,
                      total_time=getattr(engine, "current_time", None),
                  ))
        except Exception as exc:
            result.errors.append(f"Event timeline figure failed: {exc}")

    # ── Polarization Evolution ────────────────────────────────────────────────
    if vcfg.get("polarization_evolution", True) and history_opinions and history_times:
        try:
            _save("polarization_evolution",
                  vis.plot_polarization_evolution(
                      history_opinions, history_times,
                      history_num_events=history_events or None,
                  ))
        except Exception as exc:
            result.errors.append(f"Polarization evolution figure failed: {exc}")

    # ── Network Homophily ─────────────────────────────────────────────────────
    if vcfg.get("network_homophily", False) and network_graph is not None:
        try:
            edges = list(network_graph.edges())
            _save("network_homophily",
                  vis.plot_network_homophily(positions, opinions, edges, layer_idx=layer_idx))
        except Exception as exc:
            result.errors.append(f"Network homophily figure failed: {exc}")

    logger.info(f"[Manager] Visualization complete. Figures: {list(result.figure_paths.keys())}")


def _save_and_record(
    fig: Any,
    path: Path,
    dpi: int,
    name: str,
    result: AnalysisResult,
) -> None:
    """Save a matplotlib figure and record its path."""
    import matplotlib.pyplot as plt
    try:
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        result.figure_paths[name] = str(path.resolve())
        logger.info(f"[Manager] Saved figure '{name}' → {path}")
    except Exception as exc:
        result.errors.append(f"Failed to save figure '{name}': {exc}")
        try:
            plt.close(fig)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════════

def _log_summary(result: AnalysisResult) -> None:
    logger.info("=" * 60)
    logger.info("[Manager] Analysis run complete.")
    logger.info(f"  Reports:      {list(result.report_paths.keys())}")
    logger.info(f"  Figures:      {list(result.figure_paths.keys())}")
    logger.info(f"  Features:     {list(result.feature_paths.keys())}")
    logger.info(f"  AI sections:  {list(result.parsed_sections.keys())}")
    if result.errors:
        logger.warning(f"  Errors ({len(result.errors)}):")
        for e in result.errors:
            logger.warning(f"    - {e}")
    logger.info("=" * 60)