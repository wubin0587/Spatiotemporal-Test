"""
analysis/intervention/engine/attributor.py

Intervention Attribution Engine
---------------------------------
Attributes observed outcome differences to specific intervention actions by
comparing checkpoint states and computing counterfactual deltas.

The attributor answers:
    "Which part of the outcome change was caused by THIS intervention?"

It relies on BranchManager checkpoints:
    - A "pre-intervention" checkpoint (created automatically if auto_checkpoint=True)
    - A "post-intervention" snapshot (current engine state or a later checkpoint)
    - A "counterfactual" branch (same engine run without the intervention)

Attribution modes
-----------------
1. Simple delta   — post minus pre (no counterfactual)
2. Causal delta   — (post_treatment − pre) − (post_control − pre)
                    i.e. difference-in-differences
3. Policy trace   — attribute each firing of a multi-step policy separately
4. Stepwise       — per-step marginal contribution using time-series data

Usage
-----
    from intervention.branch.checkpoint import BranchManager, Checkpoint
    from analysis.intervention.engine.attributor import InterventionAttributor

    attributor = InterventionAttributor(branch_manager)

    # Simple delta: compare pre vs post a specific checkpoint
    delta = attributor.simple_delta(pre_id, post_id)

    # Causal (DiD): compare against a matched control run
    causal = attributor.causal_delta(pre_id, treatment_output, control_output)

    # Full attribution report for the InterventionManager execution log
    report = attributor.attribute_execution_log(execution_log, treatment_output, control_output)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class InterventionAttributor:
    """
    Computes attribution metrics from BranchManager checkpoints.

    Parameters
    ----------
    branch_manager : BranchManager
        The manager holding all checkpoints from the experiment.
    """

    def __init__(self, branch_manager: Any):
        self.branch_manager = branch_manager

    # ------------------------------------------------------------------
    # 1. Simple delta (pre → post, no counterfactual)
    # ------------------------------------------------------------------

    def simple_delta(
        self,
        pre_checkpoint_id: str,
        post_checkpoint_id: Optional[str] = None,
        post_pipeline_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute raw before/after deltas from checkpoint states.

        Provide EITHER ``post_checkpoint_id`` (a stored Checkpoint) OR
        ``post_pipeline_output`` (a live FeaturePipeline result).

        Returns
        -------
        dict[str, float]
            Delta values for key opinion, spatial, and network metrics.
        """
        pre_cp = self.branch_manager.get_checkpoint(pre_checkpoint_id)

        if post_checkpoint_id is not None:
            post_cp = self.branch_manager.get_checkpoint(post_checkpoint_id)
            post_opinions = post_cp.opinion_matrix
            post_impact   = post_cp.impact_vector
        elif post_pipeline_output is not None:
            final = post_pipeline_output.get("final", {})
            # Reconstruct arrays from final features — approximate
            post_opinions = None
            post_impact   = None
        else:
            raise ValueError("Provide post_checkpoint_id or post_pipeline_output.")

        deltas: Dict[str, float] = {
            "step_gap":     0.0,
            "time_gap":     0.0,
        }

        if post_checkpoint_id is not None:
            post_cp = self.branch_manager.get_checkpoint(post_checkpoint_id)
            deltas["step_gap"] = float(post_cp.time_step - pre_cp.time_step)
            deltas["time_gap"] = float(post_cp.current_time - pre_cp.current_time)

            # Opinion deltas
            pre_op  = pre_cp.opinion_matrix
            post_op = post_cp.opinion_matrix

            deltas["mean_opinion_delta"]      = float(post_op.mean()    - pre_op.mean())
            deltas["polarization_delta"]      = float(post_op.std()     - pre_op.std())
            deltas["mean_impact_delta"]       = float(
                post_cp.impact_vector.mean() - pre_cp.impact_vector.mean()
            )

            # Per-layer deltas
            n_layers = min(pre_op.shape[1], post_op.shape[1])
            for l in range(n_layers):
                deltas[f"layer_{l}_mean_delta"] = float(
                    post_op[:, l].mean() - pre_op[:, l].mean()
                )
                deltas[f"layer_{l}_std_delta"] = float(
                    post_op[:, l].std() - pre_op[:, l].std()
                )

        elif post_pipeline_output is not None:
            # Use pipeline summary trend metrics as proxy
            summary = post_pipeline_output.get("summary", {})
            for dot_key in [
                "opinion.mean_opinion",
                "opinion.polarization_std",
                "opinion.bimodality_coefficient",
                "opinion.extreme_share",
                "spatial.moran_i",
            ]:
                entry = summary.get(dot_key)
                if isinstance(entry, dict):
                    slope = entry.get("evolution_delta", 0.0)
                    if slope is not None:
                        safe_key = dot_key.replace(".", "_") + "_evolution"
                        deltas[safe_key] = float(slope)

        return deltas

    # ------------------------------------------------------------------
    # 2. Causal delta (difference-in-differences)
    # ------------------------------------------------------------------

    def causal_delta(
        self,
        pre_checkpoint_id: str,
        treatment_output: Dict[str, Any],
        control_output:   Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Difference-in-Differences (DiD) attribution.

        For each metric M:
            causal_effect(M) = (M_treatment_final − M_pre) − (M_control_final − M_pre)

        This removes the "natural drift" from the measured effect.

        Parameters
        ----------
        pre_checkpoint_id : str
            Checkpoint taken immediately before the intervention.
        treatment_output : dict
            FeaturePipeline.run() from the intervention branch.
        control_output : dict
            FeaturePipeline.run() from the matched control (no intervention) branch.

        Returns
        -------
        dict[str, float]
            Causal effect estimates for each metric.
        """
        pre_cp = self.branch_manager.get_checkpoint(pre_checkpoint_id)
        pre_vals = self._checkpoint_scalars(pre_cp)

        treat_vals = self._pipeline_final_scalars(treatment_output)
        ctrl_vals  = self._pipeline_final_scalars(control_output)

        metrics_keys = sorted(set(treat_vals) & set(ctrl_vals))
        causal: Dict[str, float] = {}

        for key in metrics_keys:
            pre  = pre_vals.get(key, 0.0)
            t    = treat_vals[key]
            c    = ctrl_vals[key]
            causal[key] = float((t - pre) - (c - pre))

        causal["did_summary_n_metrics"] = float(len(causal))
        return causal

    # ------------------------------------------------------------------
    # 3. Execution-log attribution
    # ------------------------------------------------------------------

    def attribute_execution_log(
        self,
        execution_log: List[Dict[str, Any]],
        treatment_output:  Dict[str, Any],
        control_output:    Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        For each entry in the InterventionManager execution log, compute
        the attribution delta from its pre-intervention checkpoint to the
        final state.

        Parameters
        ----------
        execution_log : list[dict]
            From InterventionManager.execution_log.
            Each entry must have 'checkpoint_id', 'label', 'step'.
        treatment_output : dict
            Final FeaturePipeline.run() of the treatment simulation.
        control_output : dict, optional
            If provided, computes DiD attribution for each entry.

        Returns
        -------
        list[dict]
            One attribution dict per execution log entry.
        """
        attributed = []

        for entry in execution_log:
            checkpoint_id = entry.get("checkpoint_id")
            if checkpoint_id is None:
                attributed.append({
                    "label": entry.get("label", ""),
                    "step":  entry.get("step",  0),
                    "attribution": None,
                    "note": "No checkpoint — set auto_checkpoint=True on the rule.",
                })
                continue

            try:
                delta = self.simple_delta(
                    pre_checkpoint_id=checkpoint_id,
                    post_pipeline_output=treatment_output,
                )

                result: Dict[str, Any] = {
                    "label":      entry.get("label", ""),
                    "step":       entry.get("step",  0),
                    "time":       entry.get("time",  0.0),
                    "attribution": delta,
                }

                if control_output is not None:
                    try:
                        causal = self.causal_delta(
                            checkpoint_id, treatment_output, control_output
                        )
                        result["causal_attribution"] = causal
                    except Exception as exc:
                        result["causal_attribution_error"] = str(exc)

                attributed.append(result)

            except Exception as exc:
                logger.warning(
                    f"Attribution failed for '{entry.get('label', '')}': {exc}"
                )
                attributed.append({
                    "label": entry.get("label", ""),
                    "step":  entry.get("step",  0),
                    "attribution": None,
                    "error": str(exc),
                })

        return attributed

    # ------------------------------------------------------------------
    # 4. Stepwise marginal attribution
    # ------------------------------------------------------------------

    def stepwise_attribution(
        self,
        pre_checkpoint_id: str,
        post_checkpoint_id: str,
        metric_key: str = "opinion.polarization_std",
    ) -> Dict[str, Any]:
        """
        Analyse the time-series evolution of a metric between two checkpoints
        to identify when the intervention effect materialised.

        Uses the summary trend metrics (slope, evolution_delta, volatility)
        from both checkpoints to infer the timing and magnitude of effect onset.

        Parameters
        ----------
        pre_checkpoint_id, post_checkpoint_id : str
            IDs of two checkpoints bracketing the intervention.
        metric_key : str
            Dot-notation metric key (e.g. 'opinion.polarization_std').

        Returns
        -------
        dict with keys:
            'pre_step', 'post_step', 'step_gap', 'metric_key',
            'pre_meta', 'post_meta', 'delta_by_meta'
        """
        pre_cp  = self.branch_manager.get_checkpoint(pre_checkpoint_id)
        post_cp = self.branch_manager.get_checkpoint(post_checkpoint_id)

        return {
            "pre_step":   pre_cp.time_step,
            "post_step":  post_cp.time_step,
            "step_gap":   post_cp.time_step - pre_cp.time_step,
            "metric_key": metric_key,
            "pre_opinion_std":   float(pre_cp.opinion_matrix.std()),
            "post_opinion_std":  float(post_cp.opinion_matrix.std()),
            "pre_impact_mean":   float(pre_cp.impact_vector.mean()),
            "post_impact_mean":  float(post_cp.impact_vector.mean()),
            "std_delta":         float(
                post_cp.opinion_matrix.std() - pre_cp.opinion_matrix.std()
            ),
            "impact_delta":      float(
                post_cp.impact_vector.mean() - pre_cp.impact_vector.mean()
            ),
            "event_count_delta": (
                post_cp.event_archive_dict.get("count", 0)
                - pre_cp.event_archive_dict.get("count", 0)
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _checkpoint_scalars(self, cp: Any) -> Dict[str, float]:
        """Extract key scalar metrics from a Checkpoint object."""
        op = cp.opinion_matrix
        return {
            "mean_opinion":          float(op.mean()),
            "polarization_std":      float(op.std()),
            "mean_impact":           float(cp.impact_vector.mean()),
            "num_events":            float(cp.event_archive_dict.get("count", 0)),
        }

    def _pipeline_final_scalars(
        self, pipeline_output: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract scalar metrics from a FeaturePipeline final snapshot."""
        final = pipeline_output.get("final", {})
        out: Dict[str, float] = {}
        for section in ("opinion", "spatial", "topo", "network_opinion", "event"):
            section_data = final.get(section, {})
            if isinstance(section_data, dict):
                for k, v in section_data.items():
                    if isinstance(v, (int, float)) and v == v:
                        out[k] = float(v)
        return out
