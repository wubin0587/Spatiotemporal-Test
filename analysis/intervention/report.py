"""
analysis/intervention/report.py

Intervention Analysis Report
-----------------------------
Assembles a complete, human-readable report for a single- or multi-arm
intervention experiment.  Supports AI-assisted and static (no-AI) modes.

Integrates with:
    - analysis.intervention.metrics      — effect metric computation
    - analysis.intervention.engine       — attribution + comparison engines
    - analysis.intervention.prompts      — LLM prompt builders
    - analysis.report.builder            — base ReportDocument / ReportBuilder
    - analysis.report.outputs.*          — markdown / html / latex renderers

Usage
-----
    from analysis.intervention.report import InterventionReportBuilder

    builder = InterventionReportBuilder(lang="zh", fmt="md")

    # Single intervention
    doc = builder.build_single(
        control_output=control_pipeline_result,
        treatment_output=treatment_pipeline_result,
        intervention_label="network_rewire_policy",
        execution_log=mgr.execution_log,
        branch_manager=mgr.branch_manager,
        api_key="sk-...",           # optional AI narration
    )
    doc.save("output/intervention_report.md")

    # Multi-arm
    doc = builder.build_multi(
        control_output=control,
        treatments={"A": out_A, "B": out_B},
        api_key="sk-...",
    )
    doc.save("output/comparison_report.md")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from analysis.constants import DEFAULT_LANGUAGE
from analysis.intervention.metrics import (
    compute_intervention_effect,
    rank_interventions,
)
from analysis.intervention.engine.comparator import InterventionComparator
from analysis.report.builder import ReportDocument

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Localization
# ═════════════════════════════════════════════════════════════════════════════

_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "title_single":        "Intervention Analysis Report",
        "title_multi":         "Multi-Arm Intervention Comparison Report",
        "generated":           "Generated",
        "executive_summary":   "Executive Summary",
        "effect_metrics":      "Intervention Effect Metrics",
        "trajectory":          "Metric Trajectory Comparison",
        "attribution":         "Per-Firing Attribution",
        "counterfactual":      "Counterfactual Analysis",
        "robustness":          "Robustness Across Replicates",
        "ranking":             "Policy Ranking",
        "static_mode":         "Static Analysis (No AI)",
        "ai_mode":             "AI-Assisted Analysis",
        "intervention_label":  "Intervention",
        "control_label":       "Control",
        "treatment_label":     "Treatment",
    },
    "zh": {
        "title_single":        "干预分析报告",
        "title_multi":         "多臂干预对比报告",
        "generated":           "生成时间",
        "executive_summary":   "执行摘要",
        "effect_metrics":      "干预效果指标",
        "trajectory":          "指标轨迹对比",
        "attribution":         "逐次触发归因",
        "counterfactual":      "反事实分析",
        "robustness":          "多次运行鲁棒性",
        "ranking":             "策略排名",
        "static_mode":         "静态分析（无AI）",
        "ai_mode":             "AI 辅助分析",
        "intervention_label":  "干预策略",
        "control_label":       "对照组",
        "treatment_label":     "干预组",
    },
}


def _lbl(key: str, lang: str) -> str:
    return _LABELS.get(lang, _LABELS["en"]).get(key, key)


# ═════════════════════════════════════════════════════════════════════════════
# InterventionReportBuilder
# ═════════════════════════════════════════════════════════════════════════════

class InterventionReportBuilder:
    """
    Assembles intervention analysis reports in Markdown, HTML, or LaTeX.

    Parameters
    ----------
    lang : str    "zh" | "en"
    fmt  : str    "md" | "html" | "latex"
    include_trajectory : bool
        Include time-series trajectory comparison section.
    include_attribution : bool
        Include per-firing attribution section (requires execution_log).
    """

    def __init__(
        self,
        lang: str = DEFAULT_LANGUAGE,
        fmt: str = "md",
        include_trajectory: bool = True,
        include_attribution: bool = True,
    ):
        self.lang = lang
        self.fmt  = fmt
        self.include_trajectory  = include_trajectory
        self.include_attribution = include_attribution
        self._comparator = InterventionComparator()

    # ------------------------------------------------------------------
    # Single-intervention report
    # ------------------------------------------------------------------

    def build_single(
        self,
        control_output:      Dict[str, Any],
        treatment_output:    Dict[str, Any],
        intervention_label:  str = "",
        execution_log:       Optional[List[Dict[str, Any]]] = None,
        branch_manager:      Optional[Any] = None,
        title:               Optional[str] = None,
        simulation_meta:     Optional[Dict[str, Any]] = None,
        weights:             Optional[Dict[str, float]] = None,
        api_key:             Optional[str] = None,
        model:               str = "claude-sonnet-4-20250514",
        max_tokens:          int = 1500,
    ) -> ReportDocument:
        """
        Build a single-intervention analysis report.

        Returns a ReportDocument that can be saved directly.
        """
        title = title or _lbl("title_single", self.lang)

        # 1. Compute effect metrics
        effect_metrics = compute_intervention_effect(
            control_output, treatment_output, weights=weights
        )

        # 2. Comparison
        comparison = self._comparator.compare(
            control_output, treatment_output, label=intervention_label or "treatment"
        )

        # 3. Attribution (if log provided)
        attribution_entries: List[Dict[str, Any]] = []
        if execution_log and branch_manager:
            try:
                from analysis.intervention.engine.attributor import InterventionAttributor
                attributor = InterventionAttributor(branch_manager)
                attribution_entries = attributor.attribute_execution_log(
                    execution_log, treatment_output, control_output
                )
            except Exception as exc:
                logger.warning(f"Attribution failed: {exc}")

        # 4. Build section texts
        sections = self._build_single_sections(
            effect_metrics=effect_metrics,
            comparison=comparison,
            attribution_entries=attribution_entries,
            intervention_label=intervention_label,
            simulation_meta=simulation_meta,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
        )

        # 5. Assemble content
        meta = dict(simulation_meta or {})
        meta["intervention"] = intervention_label or "—"
        meta["effect_score"] = f"{effect_metrics.get('effect_score', 0.0):+.4f}"

        content = self._render(title=title, sections=sections, meta=meta)

        return ReportDocument(
            content=content,
            fmt=self.fmt,
            lang=self.lang,
            title=title,
            metadata=meta,
            sections=sections,
        )

    # ------------------------------------------------------------------
    # Multi-arm report
    # ------------------------------------------------------------------

    def build_multi(
        self,
        control_output: Dict[str, Any],
        treatments:     Dict[str, Dict[str, Any]],
        title:          Optional[str] = None,
        simulation_meta: Optional[Dict[str, Any]] = None,
        weights:        Optional[Dict[str, float]] = None,
        api_key:        Optional[str] = None,
        model:          str = "claude-sonnet-4-20250514",
        max_tokens:     int = 2000,
    ) -> ReportDocument:
        """
        Build a multi-arm policy comparison report.
        """
        title = title or _lbl("title_multi", self.lang)

        # Multi-arm comparison
        multi_result = self._comparator.compare_multi(
            control_output, treatments, weights=weights
        )

        summary_rows = self._comparator.summary_table(multi_result)

        sections = self._build_multi_sections(
            multi_result=multi_result,
            summary_rows=summary_rows,
            simulation_meta=simulation_meta,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
        )

        meta = dict(simulation_meta or {})
        meta["n_arms"]     = str(len(treatments))
        meta["best_policy"] = multi_result.get("best_label", "—")

        content = self._render(title=title, sections=sections, meta=meta)

        return ReportDocument(
            content=content,
            fmt=self.fmt,
            lang=self.lang,
            title=title,
            metadata=meta,
            sections=sections,
        )

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_single_sections(
        self,
        effect_metrics:      Dict[str, float],
        comparison:          Dict[str, Any],
        attribution_entries: List[Dict[str, Any]],
        intervention_label:  str,
        simulation_meta:     Optional[Dict[str, Any]],
        api_key:             Optional[str],
        model:               str,
        max_tokens:          int,
    ) -> Dict[str, str]:
        sections: Dict[str, str] = {}

        if api_key:
            sections = self._ai_single_sections(
                effect_metrics=effect_metrics,
                attribution_entries=attribution_entries,
                intervention_label=intervention_label,
                simulation_meta=simulation_meta,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
            )

        # Static fallback / supplement
        sections.setdefault(
            _lbl("effect_metrics", self.lang),
            self._static_effect_table(effect_metrics),
        )

        if self.include_attribution and attribution_entries:
            key = _lbl("attribution", self.lang)
            if key not in sections:
                sections[key] = self._static_attribution_table(attribution_entries)

        if self.include_trajectory:
            traj_key = _lbl("trajectory", self.lang)
            if traj_key not in sections:
                sections[traj_key] = self._static_trajectory_note(comparison)

        return sections

    def _build_multi_sections(
        self,
        multi_result:    Dict[str, Any],
        summary_rows:    List[Dict[str, Any]],
        simulation_meta: Optional[Dict[str, Any]],
        api_key:         Optional[str],
        model:           str,
        max_tokens:      int,
    ) -> Dict[str, str]:
        sections: Dict[str, str] = {}

        if api_key:
            sections = self._ai_multi_sections(
                multi_result=multi_result,
                simulation_meta=simulation_meta,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
            )

        sections.setdefault(
            _lbl("ranking", self.lang),
            self._static_ranking_table(summary_rows),
        )

        return sections

    # ------------------------------------------------------------------
    # AI section generation
    # ------------------------------------------------------------------

    def _ai_single_sections(
        self,
        effect_metrics:     Dict[str, float],
        attribution_entries: List[Dict[str, Any]],
        intervention_label: str,
        simulation_meta:    Optional[Dict[str, Any]],
        api_key:            str,
        model:              str,
        max_tokens:         int,
    ) -> Dict[str, str]:
        from analysis.intervention.prompts import (
            intervention_effect_prompt,
            attribution_report_prompt,
            intervention_executive_summary_prompt,
            INTERVENTION_SYSTEM_PROMPT,
        )

        sections: Dict[str, str] = {}

        def _call(system: str, user: str) -> str:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text
            except Exception as exc:
                logger.warning(f"AI call failed: {exc}")
                return ""

        # Executive summary
        exec_text = _call(
            INTERVENTION_SYSTEM_PROMPT,
            intervention_executive_summary_prompt(
                effect_metrics=effect_metrics,
                intervention_label=intervention_label,
                simulation_meta=simulation_meta,
                lang=self.lang,
                fmt=self.fmt,
            ),
        )
        if exec_text:
            sections[_lbl("executive_summary", self.lang)] = exec_text

        # Effect metrics narrative
        effect_text = _call(
            INTERVENTION_SYSTEM_PROMPT,
            intervention_effect_prompt(
                effect_metrics=effect_metrics,
                intervention_label=intervention_label,
                simulation_meta=simulation_meta,
                lang=self.lang,
                fmt=self.fmt,
            ),
        )
        if effect_text:
            sections[_lbl("effect_metrics", self.lang)] = effect_text

        # Attribution narrative
        if attribution_entries:
            attr_text = _call(
                INTERVENTION_SYSTEM_PROMPT,
                attribution_report_prompt(
                    attributed_log=attribution_entries,
                    overall_effect=effect_metrics,
                    lang=self.lang,
                    fmt=self.fmt,
                ),
            )
            if attr_text:
                sections[_lbl("attribution", self.lang)] = attr_text

        return sections

    def _ai_multi_sections(
        self,
        multi_result:    Dict[str, Any],
        simulation_meta: Optional[Dict[str, Any]],
        api_key:         str,
        model:           str,
        max_tokens:      int,
    ) -> Dict[str, str]:
        from analysis.intervention.prompts import (
            multi_arm_comparison_prompt,
            INTERVENTION_SYSTEM_PROMPT,
        )

        sections: Dict[str, str] = {}

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=INTERVENTION_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": multi_arm_comparison_prompt(
                        ranking=multi_result.get("ranking", []),
                        metric_leaders=multi_result.get("metric_leaders", {}),
                        simulation_meta=simulation_meta,
                        lang=self.lang,
                        fmt=self.fmt,
                    ),
                }],
            )
            text = msg.content[0].text
            if text:
                sections[_lbl("executive_summary", self.lang)] = text
        except Exception as exc:
            logger.warning(f"Multi-arm AI call failed: {exc}")

        return sections

    # ------------------------------------------------------------------
    # Static table formatters
    # ------------------------------------------------------------------

    def _static_effect_table(self, effect_metrics: Dict[str, float]) -> str:
        rows = sorted(
            [(k, v) for k, v in effect_metrics.items()
             if k != "did_summary_n_metrics"],
            key=lambda x: abs(x[1]) if x[1] == x[1] else 0,
            reverse=True,
        )

        if self.fmt == "md":
            lines = ["| Metric | Value | Direction |",
                     "| --- | --- | --- |"]
            for k, v in rows:
                direction = "▲ positive" if v > 0 else ("▼ negative" if v < 0 else "—")
                lines.append(f"| {k} | {v:+.5f} | {direction} |")
            return "\n".join(lines)

        elif self.fmt == "html":
            rows_html = "".join(
                f"<tr><td>{k}</td><td>{v:+.5f}</td>"
                f"<td>{'▲' if v > 0 else '▼' if v < 0 else '—'}</td></tr>"
                for k, v in rows
            )
            return (
                "<table class='metrics-table'>"
                "<thead><tr><th>Metric</th><th>Value</th><th>Direction</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table>"
            )

        else:  # latex / fallback
            body = "\n".join(
                f"{k} & {v:+.5f} & {'positive' if v > 0 else 'negative'} \\\\"
                for k, v in rows
            )
            return (
                "\\begin{tabular}{lrl}\n\\hline\n"
                "Metric & Value & Direction \\\\ \\hline\n"
                f"{body}\n\\hline\n\\end{{tabular}}"
            )

    def _static_ranking_table(self, summary_rows: List[Dict[str, Any]]) -> str:
        cols = ["rank", "label", "effect_score", "polarization_reduction",
                "bimodality_reduction", "extreme_share_change", "convergence_speed_ratio"]

        if self.fmt == "md":
            header = " | ".join(cols)
            sep    = " | ".join("---" for _ in cols)
            lines  = [f"| {header} |", f"| {sep} |"]
            for row in summary_rows:
                cells = " | ".join(
                    str(row.get(c, "—")) for c in cols
                )
                lines.append(f"| {cells} |")
            return "\n".join(lines)

        elif self.fmt == "html":
            ths = "".join(f"<th>{c}</th>" for c in cols)
            trs = "".join(
                "<tr>" + "".join(f"<td>{row.get(c, '—')}</td>" for c in cols) + "</tr>"
                for row in summary_rows
            )
            return (
                "<table class='metrics-table'>"
                f"<thead><tr>{ths}</tr></thead>"
                f"<tbody>{trs}</tbody></table>"
            )

        else:
            col_spec = "l" * len(cols)
            head = " & ".join(cols) + " \\\\"
            body = "\n".join(
                " & ".join(str(row.get(c, "—")) for c in cols) + " \\\\"
                for row in summary_rows
            )
            return (
                f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
                f"{head}\n\\hline\n{body}\n\\hline\n\\end{{tabular}}"
            )

    def _static_attribution_table(
        self, attribution_entries: List[Dict[str, Any]]
    ) -> str:
        if self.fmt == "md":
            lines = ["| Label | Step | std_delta | mean_opinion_delta |",
                     "| --- | --- | --- | --- |"]
            for entry in attribution_entries:
                a = entry.get("attribution") or {}
                lines.append(
                    f"| {entry.get('label', '')} | {entry.get('step', '')} "
                    f"| {a.get('std_delta', '—')} "
                    f"| {a.get('mean_opinion_delta', '—')} |"
                )
            return "\n".join(lines)
        else:
            return str(attribution_entries)

    def _static_trajectory_note(self, comparison: Dict[str, Any]) -> str:
        traj = comparison.get("trajectory_summary", {})
        lines = []
        for key, data in traj.items():
            div = data.get("divergence", {}).get("treatment")
            ctrl_end = (data.get("control_stats") or {}).get("end_mean")
            treat_end = ((data.get("treatment_stats") or {}).get("treatment") or {}).get("end_mean")

            if div is not None and ctrl_end is not None and treat_end is not None:
                if self.fmt == "md":
                    lines.append(
                        f"- **{key}**: control end={ctrl_end:.4f}, "
                        f"treatment end={treat_end:.4f}, "
                        f"divergence={div:.4f}"
                    )
        return "\n".join(lines) if lines else "*(trajectory data unavailable)*"

    # ------------------------------------------------------------------
    # Document rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        title:    str,
        sections: Dict[str, str],
        meta:     Dict[str, Any],
    ) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        if self.fmt == "md":
            return self._render_md(title, sections, meta, ts)
        elif self.fmt == "html":
            return self._render_html(title, sections, meta, ts)
        else:
            return self._render_latex(title, sections, meta, ts)

    def _render_md(
        self,
        title: str,
        sections: Dict[str, str],
        meta: Dict[str, Any],
        ts: str,
    ) -> str:
        lines = [
            f"# {title}",
            f"> {_lbl('generated', self.lang)}: {ts}",
            "",
        ]
        if meta:
            lines += [f"## {_lbl('effect_metrics', self.lang)} — Metadata", ""]
            for k, v in meta.items():
                lines.append(f"- **{k}**: {v}")
            lines += ["", "---", ""]

        for heading, body in sections.items():
            lines += [f"## {heading}", "", body.strip(), ""]

        return "\n".join(lines)

    def _render_html(
        self,
        title: str,
        sections: Dict[str, str],
        meta: Dict[str, Any],
        ts: str,
    ) -> str:
        parts = [
            "<!DOCTYPE html><html><head>",
            f"<meta charset='UTF-8'><title>{title}</title>",
            "</head><body>",
            f"<h1>{title}</h1>",
            f"<p><em>{_lbl('generated', self.lang)}: {ts}</em></p>",
        ]
        if meta:
            parts.append("<ul>")
            for k, v in meta.items():
                parts.append(f"<li><strong>{k}</strong>: {v}</li>")
            parts.append("</ul>")

        for heading, body in sections.items():
            parts += [
                f"<section><h2>{heading}</h2>",
                f"<div>{body.strip()}</div></section>",
            ]

        parts.append("</body></html>")
        return "\n".join(parts)

    def _render_latex(
        self,
        title: str,
        sections: Dict[str, str],
        meta: Dict[str, Any],
        ts: str,
    ) -> str:
        lines = [
            r"\documentclass{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\begin{document}",
            r"\title{" + title + "}",
            r"\date{" + ts + "}",
            r"\maketitle",
        ]
        for heading, body in sections.items():
            lines += [r"\section{" + heading + "}", body.strip(), ""]
        lines.append(r"\end{document}")
        return "\n".join(lines)
