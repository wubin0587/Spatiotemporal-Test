"""
analysis/report/printer.py

Report Printer
--------------
Renders a concise human-readable summary of a ReportDocument (or raw pipeline
output) to stdout. Uses sentence templates from the sentences/ sub-package
for localised plain-text narration of key metrics.

No AI is involved — this is a fast, deterministic terminal summary.

Usage
-----
    from analysis.report.printer import ReportPrinter

    printer = ReportPrinter(lang="zh")
    printer.print_pipeline(pipeline_output)

    # Or after building a report:
    report.print_summary()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .builder import ReportDocument


class ReportPrinter:
    """
    Prints a concise plain-text summary to stdout.

    Parameters
    ----------
    lang : str   "zh" | "en"
    width : int  Terminal width for line wrapping (default 72)
    """

    def __init__(self, lang: str = "zh", width: int = 72):
        self.lang  = lang
        self.width = width
        self._sentences = _load_sentences(lang)

    # ── Primary entry points ─────────────────────────────────────────────────

    def print_document(self, doc: "ReportDocument") -> None:
        """Print summary from a ReportDocument."""
        self._header(doc.title)
        self._meta_block(doc.metadata)
        self._section_list(doc.sections)
        self._footer()

    def print_pipeline(self, pipeline_output: Dict[str, Any]) -> None:
        """Print a quick metric narrative directly from pipeline output."""
        self._header(self._t("summary_title"))
        self._narrate_opinion(pipeline_output)
        self._narrate_spatial(pipeline_output)
        self._narrate_topo(pipeline_output)
        self._narrate_event(pipeline_output)
        self._narrate_stability(pipeline_output)
        self._narrate_network_opinion(pipeline_output)
        self._footer()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _header(self, title: str) -> None:
        bar = "=" * self.width
        print(f"\n{bar}")
        print(f"  {title}")
        print(bar)

    def _footer(self) -> None:
        print("=" * self.width + "\n")

    def _meta_block(self, meta: Dict[str, Any]) -> None:
        if not meta:
            return
        print()
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()

    def _section_list(self, sections: Dict[str, str]) -> None:
        labels = _SECTION_LABELS.get(self.lang, _SECTION_LABELS["en"])
        print()
        for key, text in sections.items():
            label = labels.get(key, key)
            # Print section label + first 200 chars of content
            snippet = text.strip().replace("\n", " ")[:200]
            print(f"  [{label}]")
            print(f"  {snippet}{'...' if len(text.strip()) > 200 else ''}")
            print()

    # ── Metric narrators ────────────────────────────────────────────────────

    def _narrate_opinion(self, pipeline_output: Dict[str, Any]) -> None:
        summary = pipeline_output.get("summary", {})
        final   = pipeline_output.get("final",   {}).get("opinion", {})

        pol_std = _get(summary, "opinion.polarization_std", "mean") \
               or _get_scalar(final, "polarization_std")
        bc      = _get(summary, "opinion.bimodality_coefficient", "mean") \
               or _get_scalar(final, "bimodality_coefficient")
        mean_op = _get(summary, "opinion.mean_opinion", "mean") \
               or _get_scalar(final, "mean_opinion")
        extreme_share = _get(summary, "opinion.extreme_share", "mean") \
                     or _get_scalar(final, "extreme_share")
        final_stability = _get(summary, "opinion.mean_opinion", "final_stability")

        sentences = self._sentences.get("opinion", {})
        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['opinion']} ──")

        if pol_std is not None:
            if pol_std > 0.20:
                print(f"  {sentences.get('high_polarization', '').format(pol_std=pol_std)}")
            else:
                print(f"  {sentences.get('low_polarization', '').format(pol_std=pol_std)}")

        if bc is not None:
            if bc > 0.555:
                print(f"  {sentences.get('bimodal', '').format(bc=bc)}")
            else:
                print(f"  {sentences.get('unimodal', '').format(bc=bc)}")

        if mean_op is not None:
            print(f"  {sentences.get('mean_opinion', '').format(mean_op=mean_op)}")

        if extreme_share is not None and extreme_share > 0.2:
            print(f"  {sentences.get('high_extreme_share', '').format(share=extreme_share)}")

        if final_stability is not None:
            if final_stability < 0.05:
                print(f"  {sentences.get('converged', '')}")
            else:
                print(f"  {sentences.get('volatile', '')}")

    def _narrate_spatial(self, pipeline_output: Dict[str, Any]) -> None:
        summary = pipeline_output.get("summary", {})
        final   = pipeline_output.get("final",   {}).get("spatial", {})

        moran   = _get(summary, "spatial.moran_i", "mean") or _get_scalar(final, "moran_i")
        nni     = _get(summary, "spatial.nearest_neighbor_index", "mean") \
               or _get_scalar(final, "nearest_neighbor_index")
        dx      = _get_scalar(final, "centroid_dx")
        dy      = _get_scalar(final, "centroid_dy")

        sentences = self._sentences.get("spatial", {})
        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['spatial']} ──")

        if moran is not None:
            if moran > 0.2:
                print(f"  {sentences.get('high_moran', '').format(moran=moran)}")
            else:
                print(f"  {sentences.get('low_moran', '').format(moran=moran)}")

        if nni is not None:
            if nni < 1.0:
                print(f"  {sentences.get('clustered', '').format(nni=nni)}")
            else:
                print(f"  {sentences.get('dispersed', '').format(nni=nni)}")

        if dx is not None and dy is not None:
            print(f"  {sentences.get('centroid_drift', '').format(dx=dx, dy=dy)}")

    def _narrate_topo(self, pipeline_output: Dict[str, Any]) -> None:
        summary = pipeline_output.get("summary", {})
        final   = pipeline_output.get("final",   {}).get("topo", {})

        lcc   = _get(summary, "topo.largest_component_ratio", "mean") \
             or _get_scalar(final, "largest_component_ratio")
        mod   = _get(summary, "topo.modularity", "mean") \
             or _get_scalar(final, "modularity")
        gini  = _get(summary, "topo.degree_gini", "mean") \
             or _get_scalar(final, "degree_gini")
        assort = _get(summary, "topo.degree_assortativity", "mean") \
              or _get_scalar(final, "degree_assortativity")

        sentences = self._sentences.get("topo", {})
        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['topo']} ──")

        if lcc is not None:
            if lcc > 0.9:
                print(f"  {sentences.get('connected', '').format(lcc=lcc)}")
            else:
                print(f"  {sentences.get('fragmented', '').format(lcc=lcc)}")

        if mod is not None:
            if mod > 0.3:
                print(f"  {sentences.get('high_modularity', '').format(mod=mod)}")
            else:
                print(f"  {sentences.get('low_modularity', '').format(mod=mod)}")

        if gini is not None and gini > 0.4:
            print(f"  {sentences.get('hub_dominated', '').format(gini=gini)}")

        if assort is not None and assort > 0:
            print(f"  {sentences.get('assortative', '').format(r=assort)}")

    def _narrate_event(self, pipeline_output: Dict[str, Any]) -> None:
        summary = pipeline_output.get("summary", {})
        final   = pipeline_output.get("final",   {}).get("event", {})

        burst = _get(summary, "event.burstiness", "mean") or _get_scalar(final, "burstiness")
        rate  = _get(summary, "event.event_rate",  "mean") or _get_scalar(final, "event_rate")
        max_int = _get(summary, "event.max_intensity", "mean") or _get_scalar(final, "max_intensity")
        gini = _get(summary, "event.temporal_gini", "mean") or _get_scalar(final, "temporal_gini")

        sentences = self._sentences.get("event", {})
        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['event']} ──")

        if burst is not None:
            if burst > 0.3:
                print(f"  {sentences.get('bursty', '').format(burst=burst)}")
            else:
                print(f"  {sentences.get('regular', '').format(burst=burst)}")

        if rate is not None:
            print(f"  {sentences.get('event_rate', '').format(rate=rate)}")

        if max_int is not None and max_int > 0.8:
            print(f"  {sentences.get('high_intensity', '').format(max_int=max_int)}")

        if gini is not None and gini > 0.5:
            print(f"  {sentences.get('concentrated', '').format(gini=gini)}")

    def _narrate_stability(self, pipeline_output: Dict[str, Any]) -> None:
        stability = pipeline_output.get("stability", {})
        if not stability:
            return

        cv = _get_scalar(stability, "cross_run_cv")
        spread = _get_scalar(stability, "inter_run_spread")
        n_runs = _get_int(stability, "n_runs")
        n_converged = _get_int(stability, "n_converged")
        pol_mean = _get_scalar(stability, "polarization_mean")
        pol_std = _get_scalar(stability, "polarization_std")

        sentences = self._sentences.get("stability", {})
        if not sentences:
            return

        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['stability']} ──")

        if cv is not None:
            key = "high_cv" if cv > 0.15 else "low_cv"
            print(f"  {sentences.get(key, '').format(cv=cv)}")

        if spread is not None:
            key = "high_spread" if spread > 0.2 else "low_spread"
            print(f"  {sentences.get(key, '').format(spread=spread)}")

        if n_runs is not None and n_runs > 0 and n_converged is not None:
            if n_converged == 0:
                print(f"  {sentences.get('none_converged', '')}")
            elif n_converged == n_runs:
                print(f"  {sentences.get('all_converged', '').format(n_runs=n_runs)}")
            else:
                print(
                    f"  {sentences.get('partial_converged', '').format(n_runs=n_runs, n_converged=n_converged)}"
                )

        if pol_mean is not None and pol_std is not None:
            key = "stable_polarization" if pol_std <= 0.08 else "unstable_polarization"
            print(f"  {sentences.get(key, '').format(pol_mean=pol_mean, pol_std=pol_std)}")

    def _narrate_network_opinion(self, pipeline_output: Dict[str, Any]) -> None:
        summary = pipeline_output.get("summary", {})
        final = pipeline_output.get("final", {}).get("network_opinion", {})

        corr = _get(summary, "network_opinion.correlation", "mean") \
            or _get_scalar(final, "correlation")
        align = _get(summary, "network_opinion.community_alignment", "mean") \
             or _get_scalar(final, "community_alignment")
        eci = _get(summary, "network_opinion.echo_chamber_index", "mean") \
           or _get_scalar(final, "echo_chamber_index")
        hub_mean = _get(summary, "network_opinion.hub_mean", "mean") \
                or _get_scalar(final, "hub_mean")
        pop_mean = _get(summary, "network_opinion.population_mean", "mean") \
                or _get_scalar(final, "population_mean")
        rewiring = _get(summary, "network_opinion.rewiring_rate", "mean") \
                or _get_scalar(final, "rewiring_rate")
        top_k = _get_int(final, "top_k") or 5

        if all(v is None for v in (corr, align, eci, hub_mean, pop_mean, rewiring)):
            return

        sentences = self._sentences.get("network_opinion", {})
        if not sentences:
            return

        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['network_opinion']} ──")

        if corr is not None:
            if corr > 0.3:
                print(f"  {sentences.get('high_positive_corr', '').format(corr=corr)}")
            elif corr < -0.3:
                print(f"  {sentences.get('high_negative_corr', '').format(corr=corr)}")
            else:
                print(f"  {sentences.get('low_corr', '').format(corr=corr)}")

        if align is not None:
            key = "high_community_alignment" if align > 0.5 else "low_community_alignment"
            print(f"  {sentences.get(key, '').format(align=align)}")

        if hub_mean is not None and pop_mean is not None:
            direction = _hub_direction(hub_mean, pop_mean, self.lang)
            print(
                f"  {sentences.get('hub_influence', '').format(k=top_k, hub_mean=hub_mean, pop_mean=pop_mean, direction=direction)}"
            )

        if eci is not None:
            key = "strong_echo_chamber" if eci > 0.6 else "weak_echo_chamber"
            print(f"  {sentences.get(key, '').format(eci=eci)}")

        if rewiring is not None:
            key = "high_rewiring" if rewiring > 0.1 else "low_rewiring"
            print(f"  {sentences.get(key, '').format(rate=rewiring)}")

    def _t(self, key: str) -> str:
        return self._sentences.get("_meta", {}).get(key, key)


# ═════════════════════════════════════════════════════════════════════════════
# Section label lookup
# ═════════════════════════════════════════════════════════════════════════════

_SECTION_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "executive_summary": "Executive Summary",
        "opinion":           "Opinion Dynamics",
        "spatial":           "Spatial Distribution",
        "topo":              "Network Topology",
        "event":             "Event Stream",
        "stability":         "Cross-run Stability",
        "network_opinion":   "Network–Opinion",
    },
    "zh": {
        "executive_summary": "执行摘要",
        "opinion":           "意见动态",
        "spatial":           "空间分布",
        "topo":              "网络拓扑",
        "event":             "事件流",
        "stability":         "稳定性",
        "network_opinion":   "网络-意见耦合",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Sentence loader
# ═════════════════════════════════════════════════════════════════════════════

def _load_sentences(lang: str) -> Dict[str, Any]:
    """Load the sentence template dict for the given language."""
    if lang == "zh":
        from .sentences.zh_CN import SENTENCES
    else:
        from .sentences.en_US import SENTENCES
    return SENTENCES


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get(summary: Dict[str, Any], key: str, stat: str) -> Optional[float]:
    """Retrieve summary[key][stat] safely."""
    val = summary.get(key)
    if isinstance(val, dict):
        v = val.get(stat)
        return float(v) if v is not None else None
    return None


def _get_scalar(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _get_int(d: Dict[str, Any], key: str) -> Optional[int]:
    v = d.get(key)
    if isinstance(v, (int, float)):
        return int(v)
    return None


def _hub_direction(hub_mean: float, pop_mean: float, lang: str) -> str:
    delta = hub_mean - pop_mean
    if lang == "zh":
        return "引领" if delta > 0.05 else ("滞后于" if delta < -0.05 else "接近")
    return "leading" if delta > 0.05 else ("lagging behind" if delta < -0.05 else "aligned with")
