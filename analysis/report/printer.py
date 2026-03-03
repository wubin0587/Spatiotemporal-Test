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
                print(f"  {sentences.get('bimodal', '').format(bc=bc:.3f)}")
            else:
                print(f"  {sentences.get('unimodal', '').format(bc=bc:.3f)}")

        if mean_op is not None:
            print(f"  {sentences.get('mean_opinion', '').format(mean_op=mean_op:.3f)}")

    def _narrate_spatial(self, pipeline_output: Dict[str, Any]) -> None:
        summary = pipeline_output.get("summary", {})
        final   = pipeline_output.get("final",   {}).get("spatial", {})

        moran   = _get(summary, "spatial.moran_i", "mean") or _get_scalar(final, "moran_i")
        nni     = _get(summary, "spatial.nearest_neighbor_index", "mean") \
               or _get_scalar(final, "nearest_neighbor_index")

        sentences = self._sentences.get("spatial", {})
        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['spatial']} ──")

        if moran is not None:
            if moran > 0.2:
                print(f"  {sentences.get('high_moran', '').format(moran=moran:.3f)}")
            else:
                print(f"  {sentences.get('low_moran', '').format(moran=moran:.3f)}")

        if nni is not None:
            if nni < 1.0:
                print(f"  {sentences.get('clustered', '').format(nni=nni:.3f)}")
            else:
                print(f"  {sentences.get('dispersed', '').format(nni=nni:.3f)}")

    def _narrate_topo(self, pipeline_output: Dict[str, Any]) -> None:
        summary = pipeline_output.get("summary", {})
        final   = pipeline_output.get("final",   {}).get("topo", {})

        lcc   = _get(summary, "topo.largest_component_ratio", "mean") \
             or _get_scalar(final, "largest_component_ratio")
        mod   = _get(summary, "topo.modularity", "mean") \
             or _get_scalar(final, "modularity")

        sentences = self._sentences.get("topo", {})
        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['topo']} ──")

        if lcc is not None:
            if lcc > 0.9:
                print(f"  {sentences.get('connected', '').format(lcc=lcc:.3f)}")
            else:
                print(f"  {sentences.get('fragmented', '').format(lcc=lcc:.3f)}")

        if mod is not None:
            if mod > 0.3:
                print(f"  {sentences.get('high_modularity', '').format(mod=mod:.3f)}")
            else:
                print(f"  {sentences.get('low_modularity', '').format(mod=mod:.3f)}")

    def _narrate_event(self, pipeline_output: Dict[str, Any]) -> None:
        summary = pipeline_output.get("summary", {})
        final   = pipeline_output.get("final",   {}).get("event", {})

        burst = _get(summary, "event.burstiness", "mean") or _get_scalar(final, "burstiness")
        rate  = _get(summary, "event.event_rate",  "mean") or _get_scalar(final, "event_rate")

        sentences = self._sentences.get("event", {})
        print()
        print(f"  ── {_SECTION_LABELS[self.lang]['event']} ──")

        if burst is not None:
            if burst > 0.3:
                print(f"  {sentences.get('bursty', '').format(burst=burst:.3f)}")
            else:
                print(f"  {sentences.get('regular', '').format(burst=burst:.3f)}")

        if rate is not None:
            print(f"  {sentences.get('event_rate', '').format(rate=rate:.3f)}")

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
        "stability":         "Stability",
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
