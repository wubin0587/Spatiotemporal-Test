"""
analysis/intervention/engine/comparator.py

Intervention Comparator Engine
--------------------------------
Compares multiple simulation branches (treatment vs control, or policy A vs B)
using both raw feature deltas and statistical summaries.

The comparator is the analytical core used by the InterventionManager
(analysis side) to produce comparison tables, rankings, and effect summaries.

It operates on FeaturePipeline outputs and does not interact with the live
engine at all — it is a pure offline analytical component.

Key capabilities
----------------
1. Pairwise comparison    — treatment vs single control
2. Multi-arm comparison   — N treatment arms vs a shared baseline
3. Metric-level deep-dive — time-series trajectory comparison for a single metric
4. Statistical summary    — mean, std, effect size (Cohen's d) across replicate runs
5. Ranking table          — sort policies by any metric, with effect-size annotation

Usage
-----
    from analysis.intervention.engine.comparator import InterventionComparator

    comp = InterventionComparator()

    # Pairwise
    result = comp.compare(control_output, treatment_output, label="policy_A")

    # Multi-arm
    table = comp.compare_multi(control_output, {
        "policy_A": output_A,
        "policy_B": output_B,
    })

    # Time-series trajectory comparison
    traj = comp.compare_trajectory(
        "opinion.polarization_std",
        control_output, {"A": output_A, "B": output_B}
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..metrics import compute_intervention_effect, rank_interventions

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cohens_d(mean_a: float, std_a: float, mean_b: float, std_b: float) -> float:
    """Compute Cohen's d effect size between two groups."""
    pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2.0)
    if pooled_std < 1e-12:
        return 0.0
    return float((mean_b - mean_a) / pooled_std)


def _effect_label(d: float) -> str:
    """Classify Cohen's d into verbal effect size label."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def _get_ts(pipeline_output: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    return pipeline_output.get("timeseries", {}).get(key)


def _get_final_scalar(
    pipeline_output: Dict[str, Any], section: str, metric: str
) -> Optional[float]:
    val = pipeline_output.get("final", {}).get(section, {}).get(metric)
    return float(val) if val is not None else None


# ═════════════════════════════════════════════════════════════════════════════
# InterventionComparator
# ═════════════════════════════════════════════════════════════════════════════

class InterventionComparator:
    """
    Pure offline comparator for FeaturePipeline outputs.

    Stateless: all methods are pure functions operating on the provided dicts.
    Create once and reuse across many comparisons.
    """

    # ------------------------------------------------------------------
    # 1. Pairwise comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        control:   Dict[str, Any],
        treatment: Dict[str, Any],
        label:     str = "treatment",
        weights:   Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Full pairwise comparison: treatment vs control.

        Returns
        -------
        dict with keys:
            'label', 'effect_metrics', 'final_values_control',
            'final_values_treatment', 'trajectory_summary'
        """
        effect_metrics = compute_intervention_effect(
            control, treatment, weights=weights
        )

        ctrl_vals  = self._extract_final_metrics(control)
        treat_vals = self._extract_final_metrics(treatment)

        return {
            "label":                  label,
            "effect_metrics":         effect_metrics,
            "final_values_control":   ctrl_vals,
            "final_values_treatment": treat_vals,
            "trajectory_summary":     self._trajectory_summary(control, treatment),
        }

    # ------------------------------------------------------------------
    # 2. Multi-arm comparison
    # ------------------------------------------------------------------

    def compare_multi(
        self,
        control:    Dict[str, Any],
        treatments: Dict[str, Dict[str, Any]],
        weights:    Optional[Dict[str, float]] = None,
        sort_by:    str = "effect_score",
    ) -> Dict[str, Any]:
        """
        Compare N treatment arms against a shared control.

        Returns
        -------
        dict with keys:
            'ranking'         – list[dict] sorted by sort_by (best → worst)
            'pairwise'        – dict[label, compare() result]
            'best_label'      – label of the highest-scoring arm
            'worst_label'     – label of the lowest-scoring arm
            'metric_leaders'  – per-metric best arm
        """
        ranking = rank_interventions(
            control, treatments, weights=weights, sort_by=sort_by
        )

        pairwise: Dict[str, Any] = {}
        for lbl, treatment in treatments.items():
            pairwise[lbl] = self.compare(control, treatment, label=lbl, weights=weights)

        best_label  = ranking[0]["label"]  if ranking else ""
        worst_label = ranking[-1]["label"] if ranking else ""

        # Per-metric winner
        metric_keys = [
            "polarization_reduction", "bimodality_reduction",
            "extreme_share_change", "convergence_speed_ratio",
            "effect_score",
        ]
        metric_leaders: Dict[str, str] = {}
        for mk in metric_keys:
            best_val = None
            best_lbl = ""
            for entry in ranking:
                v = entry.get(mk)
                if v is not None and (best_val is None or v > best_val):
                    best_val = v
                    best_lbl = entry["label"]
            if best_lbl:
                metric_leaders[mk] = best_lbl

        return {
            "ranking":        ranking,
            "pairwise":       pairwise,
            "best_label":     best_label,
            "worst_label":    worst_label,
            "metric_leaders": metric_leaders,
        }

    # ------------------------------------------------------------------
    # 3. Time-series trajectory comparison
    # ------------------------------------------------------------------

    def compare_trajectory(
        self,
        metric_key: str,
        control:    Dict[str, Any],
        treatments: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare time-series trajectories for a single metric across branches.

        Parameters
        ----------
        metric_key : str
            Dot-notation key into pipeline 'timeseries' dict.
            E.g. 'opinion.polarization_std', 'opinion.mean_opinion'

        Returns
        -------
        dict with keys:
            'metric_key',
            'control_stats'  – mean/std/trend of the control trajectory
            'treatment_stats'– dict[label, same stats]
            'divergence'     – dict[label, mean absolute divergence from control]
        """
        ts_ctrl = _get_ts(control, metric_key)

        ctrl_stats = self._ts_stats(ts_ctrl) if ts_ctrl is not None else {}

        treat_stats:  Dict[str, Dict[str, float]] = {}
        divergence:   Dict[str, float] = {}

        for lbl, treatment in treatments.items():
            ts_t = _get_ts(treatment, metric_key)
            if ts_t is not None:
                treat_stats[lbl] = self._ts_stats(ts_t)

                # Truncate to shortest common length
                if ts_ctrl is not None:
                    n = min(len(ts_ctrl), len(ts_t))
                    divergence[lbl] = float(
                        np.mean(np.abs(ts_t[:n] - ts_ctrl[:n]))
                    )

        return {
            "metric_key":      metric_key,
            "control_stats":   ctrl_stats,
            "treatment_stats": treat_stats,
            "divergence":      divergence,
        }

    # ------------------------------------------------------------------
    # 4. Replicate-run statistical comparison
    # ------------------------------------------------------------------

    def compare_replicates(
        self,
        control_runs:   List[Dict[str, Any]],
        treatment_runs: List[Dict[str, Any]],
        metric_keys: Optional[List[str]] = None,
        label: str = "treatment",
    ) -> Dict[str, Any]:
        """
        Compare two sets of replicate runs with effect size statistics.

        Parameters
        ----------
        control_runs, treatment_runs : list[dict]
            Each element is a FeaturePipeline.run() output.
        metric_keys : list[str], optional
            Dot-notation keys to compare.  Defaults to key opinion metrics.

        Returns
        -------
        dict[metric_key, dict]
            Per-metric stats including means, stds, Cohen's d, effect label.
        """
        default_keys = [
            "opinion.polarization_std",
            "opinion.mean_opinion",
            "opinion.bimodality_coefficient",
            "opinion.extreme_share",
            "opinion.opinion_entropy",
            "spatial.moran_i",
            "topo.modularity",
        ]
        keys = metric_keys or default_keys

        def _extract_series(runs: List[Dict[str, Any]], key: str) -> np.ndarray:
            parts = key.split(".", 1)
            if len(parts) == 2:
                section, metric = parts
            else:
                section, metric = "opinion", parts[0]
            vals = []
            for r in runs:
                v = r.get("final", {}).get(section, {}).get(metric)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            return np.array(vals, dtype=float)

        results: Dict[str, Any] = {"label": label, "n_control": len(control_runs), "n_treatment": len(treatment_runs)}

        for key in keys:
            ctrl_vals  = _extract_series(control_runs,   key)
            treat_vals = _extract_series(treatment_runs, key)

            if len(ctrl_vals) == 0 or len(treat_vals) == 0:
                continue

            mean_c, std_c = float(np.mean(ctrl_vals)),   float(np.std(ctrl_vals))
            mean_t, std_t = float(np.mean(treat_vals)),  float(np.std(treat_vals))
            d = _cohens_d(mean_c, std_c, mean_t, std_t)

            results[key] = {
                "control_mean":   mean_c,
                "control_std":    std_c,
                "treatment_mean": mean_t,
                "treatment_std":  std_t,
                "delta":          mean_t - mean_c,
                "cohens_d":       d,
                "effect_label":   _effect_label(d),
            }

        return results

    # ------------------------------------------------------------------
    # 5. Summary table (for reporting)
    # ------------------------------------------------------------------

    def summary_table(
        self,
        comparison_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert a multi-arm comparison result into a flat list of row dicts
        suitable for tabular rendering in reports.

        Each row: {'label', 'effect_score', 'polarization_reduction',
                   'bimodality_reduction', 'convergence_speed_ratio',
                   'extreme_share_change', 'opinion_mean_shift'}
        """
        ranking = comparison_result.get("ranking", [])
        columns = [
            "effect_score",
            "polarization_reduction",
            "bimodality_reduction",
            "extreme_share_change",
            "opinion_mean_shift",
            "convergence_speed_ratio",
            "moran_i_change",
            "modularity_change",
        ]

        rows = []
        for rank_idx, entry in enumerate(ranking, 1):
            row: Dict[str, Any] = {
                "rank":  rank_idx,
                "label": entry.get("label", f"arm_{rank_idx}"),
            }
            for col in columns:
                v = entry.get(col)
                row[col] = round(v, 4) if v is not None else None
            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_final_metrics(self, pipeline_output: Dict[str, Any]) -> Dict[str, float]:
        """Flatten final snapshot scalars to a flat dict."""
        final = pipeline_output.get("final", {})
        out: Dict[str, float] = {}
        for section in ("opinion", "spatial", "topo", "network_opinion", "event"):
            data = final.get(section, {})
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (int, float)) and v == v:
                        out[f"{section}.{k}"] = float(v)
        return out

    def _ts_stats(self, ts: np.ndarray) -> Dict[str, float]:
        """Compute descriptive stats for a 1-D time-series array."""
        if ts is None or len(ts) == 0:
            return {}
        valid = ts[~np.isnan(ts)]
        if len(valid) == 0:
            return {}
        n = len(valid)
        slope: float = 0.0
        if n >= 2:
            x = np.linspace(0, 1, n)
            slope = float(np.polyfit(x, valid, 1)[0])
        w = max(1, int(n * 0.1))
        return {
            "mean":       float(np.mean(valid)),
            "std":        float(np.std(valid)),
            "min":        float(np.min(valid)),
            "max":        float(np.max(valid)),
            "start_mean": float(np.mean(valid[:w])),
            "end_mean":   float(np.mean(valid[-w:])),
            "trend_slope": slope,
        }

    def _trajectory_summary(
        self,
        control:   Dict[str, Any],
        treatment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare key trajectories between two branches."""
        keys = [
            "opinion.polarization_std",
            "opinion.mean_opinion",
            "opinion.extreme_share",
        ]
        return {
            k: self.compare_trajectory(k, control, {"treatment": treatment})
            for k in keys
        }
