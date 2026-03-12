"""
core/renderer.py

Matplotlib figure rendering for real-time monitoring and post-run analysis.
All functions use the "Agg" non-interactive backend and return
matplotlib.figure.Figure objects.

Design conventions:
  - Light theme: white/off-white backgrounds, muted grids
  - Teal (#0d9488) as primary accent; amber (#d97706) as secondary
  - Compact figure sizes suitable for gr.Image tiles
  - Every function is self-contained and safe to call mid-simulation
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

# Ensure non-interactive backend before any pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

# ─────────────────────────────────────────────────────────────────────────────
# Theme constants
# ─────────────────────────────────────────────────────────────────────────────

_BG        = "#ffffff"
_SURFACE   = "#f8fafc"
_GRID      = "#e2e8f0"
_TEXT      = "#1e293b"
_SUBTEXT   = "#64748b"
_TEAL      = "#0d9488"
_AMBER     = "#d97706"
_ROSE      = "#e11d48"
_VIOLET    = "#7c3aed"
_SLATE     = "#475569"

_FONT_BODY  = "DejaVu Sans"
_FONT_MONO  = "DejaVu Sans Mono"


def _apply_light_style(ax: plt.Axes, grid: bool = True) -> None:
    """Apply consistent light-theme styling to an Axes object."""
    ax.set_facecolor(_SURFACE)
    ax.figure.patch.set_facecolor(_BG)
    ax.tick_params(colors=_SUBTEXT, labelsize=7)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
        spine.set_linewidth(0.8)
    if grid:
        ax.grid(True, color=_GRID, linewidth=0.6, linestyle="-", alpha=0.8)
        ax.set_axisbelow(True)


def _tight(fig: Figure) -> Figure:
    fig.tight_layout(pad=1.2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Time-series: polarization + impact
# ─────────────────────────────────────────────────────────────────────────────

def render_timeseries(
    times:  Sequence[float],
    sigma:  Sequence[float],
    impact: Optional[Sequence[float]] = None,
    events: Optional[Sequence[int]]   = None,
    figsize: tuple = (5.6, 2.8),
) -> Figure:
    """
    Line chart of polarization (σ) over simulation time.
    Optionally overlays mean impact as a secondary line and
    event count as a faint background bar.

    Parameters
    ----------
    times   : simulation time values (x-axis)
    sigma   : opinion standard deviation per step
    impact  : mean impact per step (optional secondary line)
    events  : cumulative event count per step (optional background bars)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    _apply_light_style(ax)

    t = list(times)
    s = list(sigma)

    if events and max(events) > 0:
        ax2 = ax.twinx()
        ax2.fill_between(t, events, alpha=0.12, color=_VIOLET, linewidth=0)
        ax2.set_ylim(0, max(events) * 4)
        ax2.set_ylabel("events", color=_VIOLET, fontsize=7)
        ax2.tick_params(colors=_VIOLET, labelsize=6)
        ax2.spines["right"].set_edgecolor(_VIOLET)
        for spine in ["top", "left", "bottom"]:
            ax2.spines[spine].set_visible(False)

    ax.plot(t, s, color=_TEAL, linewidth=1.8, label="σ polarization", zorder=3)

    if impact:
        ax.plot(t, impact, color=_AMBER, linewidth=1.2,
                linestyle="--", alpha=0.8, label="mean impact", zorder=2)

    ax.set_xlabel("time", fontsize=8, color=_SUBTEXT)
    ax.set_ylabel("σ", fontsize=8, color=_SUBTEXT)
    ax.legend(fontsize=6.5, framealpha=0.7, loc="upper left")
    ax.set_ylim(bottom=0)

    return _tight(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spatial scatter: agent positions coloured by opinion
# ─────────────────────────────────────────────────────────────────────────────

def render_spatial(
    engine: Any,
    layer_idx: int = 0,
    figsize: tuple = (4.0, 4.0),
) -> Optional[Figure]:
    """
    Scatter plot of agent positions coloured by opinion[layer_idx].
    Marker size is proportional to the impact vector.

    Parameters
    ----------
    engine    : StepExecutor with .agent_positions, .opinion_matrix, .impact_vector
    layer_idx : which opinion layer to visualise
    """
    if engine is None:
        return None

    try:
        pos = np.asarray(engine.agent_positions)
        ops = np.asarray(engine.opinion_matrix)[:, layer_idx]
        imp = np.asarray(engine.impact_vector)
    except Exception:
        return None

    # Normalise impact for scatter size (3–18 pt range)
    imp_norm = imp / (imp.max() + 1e-9)
    sizes = 3.0 + imp_norm * 15.0

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    _apply_light_style(ax, grid=False)

    sc = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=ops, cmap="RdYlBu_r",
        s=sizes, alpha=0.75,
        vmin=0, vmax=1,
        linewidths=0.2, edgecolors=_GRID,
        zorder=2,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.034, pad=0.02)
    cbar.ax.tick_params(labelsize=6, colors=_SUBTEXT)
    cbar.outline.set_edgecolor(_GRID)
    cbar.set_label("opinion", fontsize=7, color=_SUBTEXT)

    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=6)

    return _tight(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Opinion distribution histogram
# ─────────────────────────────────────────────────────────────────────────────

def render_histogram(
    engine: Any,
    layer_idx: int = 0,
    figsize: tuple = (4.2, 2.8),
) -> Optional[Figure]:
    """
    Histogram of the current opinion distribution for a single layer.
    Annotates mean and std.

    Parameters
    ----------
    engine    : StepExecutor with .opinion_matrix
    layer_idx : which opinion layer to plot
    """
    if engine is None:
        return None

    try:
        ops = np.asarray(engine.opinion_matrix)[:, layer_idx]
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    _apply_light_style(ax)

    ax.hist(ops, bins=32, range=(0, 1),
            color=_TEAL, alpha=0.75, edgecolor=_BG, linewidth=0.4, zorder=2)

    mean_v = float(ops.mean())
    std_v  = float(ops.std())
    ax.axvline(mean_v, color=_ROSE, linewidth=1.4, linestyle="-", zorder=3,
               label=f"μ = {mean_v:.3f}")
    ax.axvspan(max(0, mean_v - std_v), min(1, mean_v + std_v),
               alpha=0.08, color=_ROSE, zorder=1)

    ax.set_xlim(0, 1)
    ax.set_xlabel("opinion", fontsize=8, color=_SUBTEXT)
    ax.set_ylabel("agents", fontsize=8, color=_SUBTEXT)
    ax.legend(fontsize=6.5, framealpha=0.7)
    ax.annotate(f"σ = {std_v:.3f}", xy=(0.97, 0.93), xycoords="axes fraction",
                ha="right", va="top", fontsize=6.5, color=_SUBTEXT)

    return _tight(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Event timeline (stem / filled area)
# ─────────────────────────────────────────────────────────────────────────────

def render_events(
    times:      Sequence[float],
    num_events: Sequence[int],
    figsize: tuple = (5.6, 2.2),
) -> Figure:
    """
    Filled area chart of cumulative event count over time.

    Parameters
    ----------
    times      : simulation time values
    num_events : event count per step (cumulative or per-step)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    _apply_light_style(ax)

    t = list(times)
    e = list(num_events)

    ax.fill_between(t, e, alpha=0.22, color=_VIOLET, linewidth=0, zorder=1)
    ax.plot(t, e, color=_VIOLET, linewidth=1.4, zorder=2)

    ax.set_xlabel("time", fontsize=8, color=_SUBTEXT)
    ax.set_ylabel("events", fontsize=8, color=_SUBTEXT)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    return _tight(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Composite 2 × 2 dashboard (post-run, high-quality)
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(
    engine: Any,
    h_time:   list[float],
    h_sigma:  list[float],
    h_impact: list[float],
    h_events: list[int],
    layer_idx: int = 0,
    figsize: tuple = (11.0, 8.0),
) -> Figure:
    """
    2 × 2 composite figure combining all four chart types at higher resolution.
    Intended for the Analysis tab after simulation completes.

    Parameters
    ----------
    engine    : completed StepExecutor
    h_*       : lightweight history buffers collected during run
    layer_idx : opinion layer for spatial / histogram panels
    """
    fig = plt.figure(figsize=figsize, dpi=130)
    fig.patch.set_facecolor(_BG)

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                          left=0.07, right=0.97, top=0.93, bottom=0.07)

    # ── (0,0) Polarization time-series ────────────────────────────────────
    ax_ts = fig.add_subplot(gs[0, 0])
    _apply_light_style(ax_ts)
    ax_ts.plot(h_time, h_sigma,  color=_TEAL,  linewidth=1.8, label="σ")
    ax_ts.plot(h_time, h_impact, color=_AMBER, linewidth=1.2,
               linestyle="--", alpha=0.85, label="mean impact")
    ax_ts.set_xlabel("time", fontsize=8, color=_SUBTEXT)
    ax_ts.set_ylabel("value", fontsize=8, color=_SUBTEXT)
    ax_ts.set_title("Polarization & Impact", fontsize=9, color=_TEXT, pad=4)
    ax_ts.legend(fontsize=7, framealpha=0.7)
    ax_ts.set_ylim(bottom=0)

    # ── (0,1) Spatial scatter ──────────────────────────────────────────────
    ax_sp = fig.add_subplot(gs[0, 1])
    _apply_light_style(ax_sp, grid=False)
    if engine is not None:
        try:
            pos = np.asarray(engine.agent_positions)
            ops = np.asarray(engine.opinion_matrix)[:, layer_idx]
            imp = np.asarray(engine.impact_vector)
            imp_norm = imp / (imp.max() + 1e-9)
            sizes = 3.0 + imp_norm * 18.0
            sc = ax_sp.scatter(pos[:, 0], pos[:, 1], c=ops, cmap="RdYlBu_r",
                               s=sizes, alpha=0.75, vmin=0, vmax=1,
                               linewidths=0.2, edgecolors=_GRID)
            cbar = fig.colorbar(sc, ax=ax_sp, fraction=0.034, pad=0.02)
            cbar.ax.tick_params(labelsize=6, colors=_SUBTEXT)
            cbar.outline.set_edgecolor(_GRID)
        except Exception:
            ax_sp.text(0.5, 0.5, "data unavailable", transform=ax_sp.transAxes,
                       ha="center", va="center", color=_SUBTEXT, fontsize=8)
    ax_sp.set_aspect("equal")
    ax_sp.set_title("Spatial Opinion Distribution", fontsize=9, color=_TEXT, pad=4)
    ax_sp.set_xlim(-0.02, 1.02)
    ax_sp.set_ylim(-0.02, 1.02)

    # ── (1,0) Opinion histogram ────────────────────────────────────────────
    ax_hi = fig.add_subplot(gs[1, 0])
    _apply_light_style(ax_hi)
    if engine is not None:
        try:
            ops_all = np.asarray(engine.opinion_matrix)[:, layer_idx]
            ax_hi.hist(ops_all, bins=32, range=(0, 1),
                       color=_TEAL, alpha=0.75, edgecolor=_BG, linewidth=0.3)
            mean_v = float(ops_all.mean())
            ax_hi.axvline(mean_v, color=_ROSE, linewidth=1.5, linestyle="-",
                          label=f"μ={mean_v:.3f}")
            ax_hi.legend(fontsize=7, framealpha=0.7)
        except Exception:
            pass
    ax_hi.set_xlim(0, 1)
    ax_hi.set_xlabel("opinion", fontsize=8, color=_SUBTEXT)
    ax_hi.set_ylabel("count", fontsize=8, color=_SUBTEXT)
    ax_hi.set_title("Opinion Distribution", fontsize=9, color=_TEXT, pad=4)

    # ── (1,1) Event timeline ───────────────────────────────────────────────
    ax_ev = fig.add_subplot(gs[1, 1])
    _apply_light_style(ax_ev)
    if h_time and h_events:
        ax_ev.fill_between(h_time, h_events, alpha=0.22, color=_VIOLET, linewidth=0)
        ax_ev.plot(h_time, h_events, color=_VIOLET, linewidth=1.4)
    ax_ev.set_xlabel("time", fontsize=8, color=_SUBTEXT)
    ax_ev.set_ylabel("events", fontsize=8, color=_SUBTEXT)
    ax_ev.set_title("Event Timeline", fontsize=9, color=_TEXT, pad=4)
    ax_ev.set_ylim(bottom=0)
    ax_ev.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    return fig
