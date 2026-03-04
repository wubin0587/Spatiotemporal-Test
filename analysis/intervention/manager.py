"""
analysis/intervention/manager.py

Intervention Analysis Manager
-------------------------------
The top-level entry point for running the full intervention analysis pipeline.
Analogous to analysis/manager.py but specialised for before/after comparisons.

Responsibilities
----------------
1. Accept a completed control run and one or more treatment runs.
2. Extract features from all branches (or accept pre-computed pipeline outputs).
3. Compute intervention effect metrics.
4. Optionally run AI-assisted narration via ParserClient / Anthropic SDK.
5. Generate a ReportDocument and save figures.
6. Return an InterventionAnalysisResult.

YAML config schema
------------------
    intervention_analysis:
      output_dir: "output/intervention_001"
      lang: "zh"
      fmt: "md"
      save_figures: true
      save_features_json: true
      ai:
        enabled: false
        api_key: ""
        model: "claude-sonnet-4-20250514"
        max_tokens: 1500
      metrics:
        weights: {}       # optional overrides for effect_score weights
      report:
        include_trajectory: true
        include_attribution: true

Usage
-----
    from analysis.intervention.manager import run_intervention_analysis

    result = run_intervention_analysis(
        control_engine=sim_control._engine,
        treatment_engines={"policy_A": sim_a._engine, "policy_B": sim_b._engine},
        config=yaml_config,
        intervention_manager=mgr,  # optional, for attribution
    )
    print(result.report_paths)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Result Container
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class InterventionAnalysisResult:
    """
    Container for all outputs of run_intervention_analysis().

    Attributes
    ----------
    effect_metrics : dict
        Single-arm or multi-arm effect metrics.
    pipeline_outputs : dict[str, dict]
        Mapping of branch_label → FeaturePipeline.run() output.
        Keys: 'control', and one per treatment label.
    report_paths : dict[str, str]
        Mapping of format → absolute file path of the saved report.
    figure_paths : dict[str, str]
        Mapping of figure name → absolute file path.
    feature_paths : dict[str, str]
        Mapping of artifact type → absolute path.
    ranking : list[dict]
        Multi-arm ranking table (empty for single-arm).
    errors : list[str]
    config : dict
    """
    effect_metrics:   Dict[str, Any]      = field(default_factory=dict)
    pipeline_outputs: Dict[str, Any]      = field(default_factory=dict)
    report_paths:     Dict[str, str]      = field(default_factory=dict)
    figure_paths:     Dict[str, str]      = field(default_factory=dict)
    feature_paths:    Dict[str, str]      = field(default_factory=dict)
    ranking:          List[Dict[str, Any]]= field(default_factory=list)
    errors:           List[str]           = field(default_factory=list)
    config:           Dict[str, Any]      = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Default Config
# ═════════════════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG: Dict[str, Any] = {
    "intervention_analysis": {
        "output_dir":         "output/intervention",
        "lang":               "zh",
        "fmt":                "md",
        "save_figures":       True,
        "save_features_json": True,
        "ai": {
            "enabled":    False,
            "api_key":    "",
            "model":      "claude-sonnet-4-20250514",
            "max_tokens": 1500,
        },
        "metrics": {
            "weights": {},
        },
        "report": {
            "include_trajectory":  True,
            "include_attribution": True,
        },
    }
}


def _merge_config(user: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    ia_user = user.get("intervention_analysis", user)
    ia_cfg = cfg["intervention_analysis"]
    for k, v in ia_user.items():
        if isinstance(v, dict) and k in ia_cfg:
            ia_cfg[k].update(v)
        else:
            ia_cfg[k] = v
    return ia_cfg


# ═════════════════════════════════════════════════════════════════════════════
# Public Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def run_intervention_analysis(
    control_engine: Any,
    treatment_engines: Dict[str, Any],
    config: Dict[str, Any],
    intervention_manager: Optional[Any] = None,
    control_pipeline_output:    Optional[Dict[str, Any]] = None,
    treatment_pipeline_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> InterventionAnalysisResult:
    """
    Run the full intervention analysis pipeline.

    Parameters
    ----------
    control_engine : StepExecutor
        A completed engine for the baseline (no-intervention) run.
    treatment_engines : dict[str, StepExecutor]
        One or more treatment engines.  For single-arm use {'policy': engine}.
    config : dict
        Parsed YAML config (see module docstring for schema).
    intervention_manager : InterventionManager, optional
        If provided, its execution_log and branch_manager are used for attribution.
    control_pipeline_output : dict, optional
        Pre-computed FeaturePipeline.run() for control (skips re-extraction).
    treatment_pipeline_outputs : dict[str, dict], optional
        Pre-computed pipeline outputs for treatments.

    Returns
    -------
    InterventionAnalysisResult
    """
    cfg    = _merge_config(config)
    result = InterventionAnalysisResult(config=cfg)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Feature extraction ─────────────────────────────────────────────────
    pipeline_outputs = _extract_all_features(
        control_engine=control_engine,
        treatment_engines=treatment_engines,
        control_precomputed=control_pipeline_output,
        treatment_precomputed=treatment_pipeline_outputs,
        layer_idx=0,
        result=result,
    )
    result.pipeline_outputs = pipeline_outputs

    if "control" not in pipeline_outputs:
        result.errors.append("Control pipeline output missing — cannot proceed.")
        return result

    control_output = pipeline_outputs["control"]
    treatments_output = {k: v for k, v in pipeline_outputs.items() if k != "control"}

    # ── 2. Effect metrics ─────────────────────────────────────────────────────
    from analysis.intervention.metrics import compute_intervention_effect, rank_interventions

    weights = cfg["metrics"].get("weights") or {}
    weights = weights if weights else None

    if len(treatments_output) == 1:
        label, treat_out = next(iter(treatments_output.items()))
        effect = compute_intervention_effect(control_output, treat_out, weights=weights)
        result.effect_metrics = effect
        result.ranking = [{"label": label, **effect}]
    else:
        result.ranking = rank_interventions(
            control_output, treatments_output, weights=weights
        )
        result.effect_metrics = {
            r["label"]: r for r in result.ranking
        }

    # ── 3. Save feature artifacts ─────────────────────────────────────────────
    if cfg.get("save_features_json", True):
        _save_feature_artifacts(pipeline_outputs, out_dir, result)

    # ── 4. Report generation ──────────────────────────────────────────────────
    _run_report(
        control_output=control_output,
        treatments_output=treatments_output,
        intervention_manager=intervention_manager,
        cfg=cfg,
        result=result,
        out_dir=out_dir,
    )

    # ── 5. Visualization ──────────────────────────────────────────────────────
    if cfg.get("save_figures", True):
        _run_visualization(
            control_output=control_output,
            treatments_output=treatments_output,
            result=result,
            out_dir=out_dir,
            cfg=cfg,
        )

    _log_summary(result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Feature Extraction
# ═════════════════════════════════════════════════════════════════════════════

def _extract_all_features(
    control_engine: Any,
    treatment_engines: Dict[str, Any],
    control_precomputed: Optional[Dict[str, Any]],
    treatment_precomputed: Optional[Dict[str, Dict[str, Any]]],
    layer_idx: int,
    result: InterventionAnalysisResult,
) -> Dict[str, Dict[str, Any]]:
    from analysis.feature.pipeline import FeaturePipeline

    outputs: Dict[str, Dict[str, Any]] = {}

    # Control
    if control_precomputed is not None:
        outputs["control"] = control_precomputed
    else:
        try:
            outputs["control"] = FeaturePipeline(control_engine, layer_idx).run()
        except Exception as exc:
            result.errors.append(f"Control feature extraction failed: {exc}")
            logger.error(f"Control extraction failed: {exc}", exc_info=True)

    # Treatments
    for label, engine in treatment_engines.items():
        if treatment_precomputed and label in treatment_precomputed:
            outputs[label] = treatment_precomputed[label]
        else:
            try:
                outputs[label] = FeaturePipeline(engine, layer_idx).run()
            except Exception as exc:
                result.errors.append(f"Treatment '{label}' extraction failed: {exc}")
                logger.error(f"Treatment '{label}' extraction failed: {exc}", exc_info=True)

    return outputs


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Save Feature Artifacts
# ═════════════════════════════════════════════════════════════════════════════

def _save_feature_artifacts(
    pipeline_outputs: Dict[str, Dict[str, Any]],
    out_dir: Path,
    result: InterventionAnalysisResult,
) -> None:
    from analysis.feature.io.exporter import save_summary

    for label, output in pipeline_outputs.items():
        try:
            path = out_dir / f"features_{label}.json"
            save_summary(output.get("summary", {}), path)
            result.feature_paths[f"{label}_summary_json"] = str(path.resolve())
        except Exception as exc:
            result.errors.append(f"Failed to save features for '{label}': {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Report Generation
# ═════════════════════════════════════════════════════════════════════════════

def _run_report(
    control_output:      Dict[str, Any],
    treatments_output:   Dict[str, Dict[str, Any]],
    intervention_manager: Optional[Any],
    cfg:                 Dict[str, Any],
    result:              InterventionAnalysisResult,
    out_dir:             Path,
) -> None:
    from analysis.intervention.report import InterventionReportBuilder

    lang = cfg.get("lang", "zh")
    fmt  = cfg.get("fmt",  "md")
    ai_cfg = cfg.get("ai", {})

    api_key = ai_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    model   = ai_cfg.get("model", "claude-sonnet-4-20250514")
    max_tok = ai_cfg.get("max_tokens", 1500)

    builder = InterventionReportBuilder(
        lang=lang,
        fmt=fmt,
        include_trajectory=cfg.get("report", {}).get("include_trajectory", True),
        include_attribution=cfg.get("report", {}).get("include_attribution", True),
    )

    try:
        if len(treatments_output) == 1:
            label, treat_out = next(iter(treatments_output.items()))
            doc = builder.build_single(
                control_output=control_output,
                treatment_output=treat_out,
                intervention_label=label,
                execution_log=(intervention_manager.execution_log
                               if intervention_manager else None),
                branch_manager=(intervention_manager.branch_manager
                                if intervention_manager else None),
                api_key=api_key if ai_cfg.get("enabled") else None,
                model=model,
                max_tokens=max_tok,
            )
        else:
            doc = builder.build_multi(
                control_output=control_output,
                treatments=treatments_output,
                api_key=api_key if ai_cfg.get("enabled") else None,
                model=model,
                max_tokens=max_tok,
            )

        ext_map = {"md": ".md", "html": ".html", "latex": ".tex"}
        fname = f"intervention_report{ext_map.get(fmt, '.md')}"
        abs_path = doc.save(str(out_dir / fname))
        result.report_paths[fmt] = abs_path
        logger.info(f"[InterventionManager] Saved report → {abs_path}")

    except Exception as exc:
        result.errors.append(f"Report generation failed: {exc}")
        logger.error(f"Report generation failed: {exc}", exc_info=True)


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Visualization
# ═════════════════════════════════════════════════════════════════════════════

def _run_visualization(
    control_output:    Dict[str, Any],
    treatments_output: Dict[str, Dict[str, Any]],
    result:            InterventionAnalysisResult,
    out_dir:           Path,
    cfg:               Dict[str, Any],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        result.errors.append("matplotlib unavailable — skipping figures.")
        return

    from analysis.intervention.visual import (
        effect_bar_chart,
        trajectory_comparison,
        ranking_heatmap,
    )
    from analysis.intervention.metrics import compute_intervention_effect, rank_interventions
    from analysis.report.outputs.static_tables import _compact_json

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    weights = cfg.get("metrics", {}).get("weights") or None

    def _save(name: str, fig: Any) -> None:
        try:
            path = fig_dir / f"{name}.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            result.figure_paths[name] = str(path.resolve())
        except Exception as exc:
            result.errors.append(f"Figure '{name}' failed: {exc}")

    # Effect bar chart (per treatment)
    for label, treat_out in treatments_output.items():
        try:
            em = compute_intervention_effect(control_output, treat_out, weights=weights)
            fig = effect_bar_chart(em, title=f"Effect Metrics — {label}")
            _save(f"effect_bar_{label}", fig)
        except Exception as exc:
            result.errors.append(f"effect_bar for '{label}' failed: {exc}")

    # Trajectory comparison
    try:
        ts_ctrl = control_output.get("timeseries", {})
        for mk in ["opinion.polarization_std", "opinion.mean_opinion"]:
            ctrl_ts = ts_ctrl.get(mk)
            if ctrl_ts is None:
                continue
            treat_dict = {}
            for lbl, tout in treatments_output.items():
                ts_t = tout.get("timeseries", {}).get(mk)
                if ts_t is not None:
                    treat_dict[lbl] = ts_t
            if treat_dict:
                fig = trajectory_comparison(mk, ctrl_ts, treat_dict)
                safe_name = mk.replace(".", "_")
                _save(f"trajectory_{safe_name}", fig)
    except Exception as exc:
        result.errors.append(f"Trajectory figures failed: {exc}")

    # Ranking heatmap (multi-arm only)
    if len(treatments_output) > 1:
        try:
            from analysis.intervention.engine.comparator import InterventionComparator
            comp = InterventionComparator()
            multi = comp.compare_multi(control_output, treatments_output, weights=weights)
            rows = comp.summary_table(multi)
            fig = ranking_heatmap(rows)
            _save("policy_ranking_heatmap", fig)
        except Exception as exc:
            result.errors.append(f"Ranking heatmap failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════════

def _log_summary(result: InterventionAnalysisResult) -> None:
    logger.info("=" * 60)
    logger.info("[InterventionAnalysisManager] Pipeline complete.")
    logger.info(f"  Reports:  {list(result.report_paths.keys())}")
    logger.info(f"  Figures:  {list(result.figure_paths.keys())}")
    logger.info(f"  Features: {list(result.feature_paths.keys())}")
    if result.errors:
        logger.warning(f"  Errors ({len(result.errors)}):")
        for e in result.errors:
            logger.warning(f"    - {e}")
    logger.info("=" * 60)
