"""
ui/components/metric_cards.py

Reusable metric card components for the Monitor tab.

A "metric card" is a read-only gr.Number wrapped in a styled column that
shows a single live statistic during simulation.  Cards are always
interactive=False and are updated via streaming yield tuples.

Public API
----------
build_metric_cards(lang) -> MetricCards
    Instantiate all five cards inside the current Gradio context.
    Returns a MetricCards dataclass containing each component.

MetricCards
    .sigma        gr.Number  — opinion standard deviation  (σ)
    .mean         gr.Number  — mean opinion  (μ)
    .impact       gr.Number  — mean external-field impact
    .events       gr.Number  — cumulative event count
    .consensus    gr.Number  — fraction of agents within 0.1 of mean

as_list(cards) -> list[gr.Number]
    Ordered list matching the yield tuple positions 1-5 in runner.py.

Usage
-----
    from ui.components.metric_cards import build_metric_cards, as_list

    with gr.Row():
        cards = build_metric_cards(lang="en")

    # In streaming outputs:
    outputs = [status_md] + as_list(cards) + [...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gradio as gr

from core.defaults import get_label


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

_CARD_KEYS: list[str] = [
    "metric_sigma",
    "metric_mean",
    "metric_impact",
    "metric_events",
    "metric_consensus",
]

# Fallback labels when defaults.py does not carry a key
_FALLBACK_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "metric_sigma":     "σ  polarization",
        "metric_mean":      "μ  mean opinion",
        "metric_impact":    "mean impact",
        "metric_events":    "events",
        "metric_consensus": "consensus",
    },
    "zh": {
        "metric_sigma":     "σ 极化度",
        "metric_mean":      "μ 平均意见",
        "metric_impact":    "平均影响",
        "metric_events":    "事件数",
        "metric_consensus": "共识度",
    },
}

_CARD_PRECISION: dict[str, int] = {
    "metric_sigma":     4,
    "metric_mean":      4,
    "metric_impact":    4,
    "metric_events":    0,
    "metric_consensus": 4,
}

_CARD_DEFAULTS: dict[str, float] = {
    "metric_sigma":     0.0,
    "metric_mean":      0.0,
    "metric_impact":    0.0,
    "metric_events":    0,
    "metric_consensus": 0.0,
}


def _label(key: str, lang: str) -> str:
    """Retrieve label text, falling back to _FALLBACK_LABELS."""
    try:
        return get_label(key, lang)
    except (KeyError, TypeError):
        return _FALLBACK_LABELS.get(lang, _FALLBACK_LABELS["en"]).get(key, key)


# ─────────────────────────────────────────────────────────────────────────────
# MetricCards dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricCards:
    """Container for the five live-metric gr.Number components."""

    sigma:     gr.Number
    mean:      gr.Number
    impact:    gr.Number
    events:    gr.Number
    consensus: gr.Number

    def as_list(self) -> list[gr.Number]:
        """Return ordered list matching runner yield positions 1-5."""
        return [self.sigma, self.mean, self.impact, self.events, self.consensus]

    def reset_updates(self) -> list:
        """Return a list of gr.update() calls that zero all cards."""
        return [
            gr.update(value=0.0),  # sigma
            gr.update(value=0.0),  # mean
            gr.update(value=0.0),  # impact
            gr.update(value=0),    # events
            gr.update(value=0.0),  # consensus
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_metric_cards(lang: str = "en") -> MetricCards:
    """
    Render five metric cards in the current Gradio context.

    Must be called inside an active gr.Blocks() context, typically
    inside a ``with gr.Row():`` block.

    Parameters
    ----------
    lang : {"en", "zh"}
        UI language for card labels.

    Returns
    -------
    MetricCards
        Dataclass holding each gr.Number component.
    """

    def _card(key: str) -> gr.Number:
        with gr.Column(min_width=90, elem_classes="metric-card"):
            return gr.Number(
                label=_label(key, lang),
                value=_CARD_DEFAULTS[key],
                precision=_CARD_PRECISION[key],
                interactive=False,
                show_label=True,
            )

    sigma     = _card("metric_sigma")
    mean      = _card("metric_mean")
    impact    = _card("metric_impact")
    events    = _card("metric_events")
    consensus = _card("metric_consensus")

    return MetricCards(
        sigma=sigma,
        mean=mean,
        impact=impact,
        events=events,
        consensus=consensus,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience alias
# ─────────────────────────────────────────────────────────────────────────────

def as_list(cards: MetricCards) -> list[gr.Number]:
    """Ordered list [sigma, mean, impact, events, consensus]."""
    return cards.as_list()


# ─────────────────────────────────────────────────────────────────────────────
# Label-refresh helper (for language toggle)
# ─────────────────────────────────────────────────────────────────────────────

def refresh_labels(cards: MetricCards, lang: str) -> list:
    """
    Return a list of gr.update(label=...) for each card to hot-swap labels
    after a language toggle without rebuilding the component tree.

    Yields one update per card in as_list() order.
    """
    return [
        gr.update(label=_label(key, lang))
        for key in _CARD_KEYS
    ]
