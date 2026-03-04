"""
analysis/intervention/visual.py

Intervention Visualization Module
-----------------------------------
Static matplotlib figures for visualising intervention effects and
treatment-vs-control comparisons.

All functions follow the same conventions as analysis/visual/static.py:
- Pure functions returning matplotlib Figure objects
- DEFAULT_PALETTE / DEFAULT_OPINION_CMAP imported from static.py
- Optional save_path parameter
- No classes, no global state

Figure types
------------
1. effect_bar_chart           — horizontal bar chart of delta metrics
2. trajectory_comparison      — treatment vs control time-series overlay
3. opinion_shift_violin       — violin plots of final opinion distributions
4. ranking_heatmap            — colour-coded policy ranking table
5. attribution_waterfall      — waterfall chart for per-firing attribution
6. phase_portrait_comparison  — 2D phase space before/after intervention
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from analysis.visual.static import DEFAULT_PALETTE, DEFAULT_OPINION_CMAP, apply_style, save_figure


# ═════════════════════════════════════════════════════════════════════════════
# 1. Effect bar chart
# ═════════════════════════════════════════════════════════════════════════════

def effect_bar_chart(
    effect_metrics: Dict[str, float],
    title: str = "Intervention Effect Metrics",
    figsize: Tuple[int, int] = (9, 6),
    palette: Optional[Dict[str, str]] = None,
    exclude_keys: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart showing signed delta values for each effect metric.

    Positive bars (green/teal) = beneficial / convergence effect.
    Negative bars (red/rose)   = harmful / polarisation effect.
    The 'effect_score' key is highlighted separately if present.
    """
    palette = palette or DEFAULT_PALETTE
    skip = set(exclude_keys or []) | {"effect_score", "did_summary_n_metrics"}

    items = [(k, v) for k, v in effect_metrics.items()
             if k not in skip and isinstance(v, (int, float)) and v == v]
    items.sort(key=lambda x: x[1])

    if not items:
        fig, ax = plt.subplots(figsize=figsize)
        apply_style(fig, [ax], palette)
        ax.text(0.5, 0.5, "No metrics to display", ha="center", va="center",
                color=palette["text_dim"], transform=ax.transAxes)
        return fig

    labels = [k.replace("_", " ") for k, _ in items]
    values = [v for _, v in items]
    colors = [palette["teal"] if v >= 0 else palette["rose"] for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    bars = ax.barh(labels, values, color=colors, alpha=0.82, height=0.65)

    # Value annotations
    for bar, val in zip(bars, values):
        x_offset = 0.003 if val >= 0 else -0.003
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + x_offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center", ha=ha,
            color=palette["text"], fontsize=7.5, fontfamily="monospace",
        )

    ax.axvline(0, color=palette["border"], linewidth=1.2, zorder=5)

    # Annotate composite score if present
    score = effect_metrics.get("effect_score")
    if score is not None:
        score_color = palette["amber"] if score >= 0 else palette["rose"]
        ax.text(
            0.98, 0.02,
            f"Effect Score: {score:+.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            color=score_color, fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=palette["surface"],
                      edgecolor=palette["border"]),
        )

    ax.set_xlabel("Δ (treatment − control)", fontsize=9)
    ax.set_title(title, color=palette["text"], fontsize=11, pad=12)
    ax.margins(x=0.15)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette["bg"])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2. Trajectory comparison
# ═════════════════════════════════════════════════════════════════════════════

def trajectory_comparison(
    metric_key: str,
    control_ts: np.ndarray,
    treatment_ts_dict: Dict[str, np.ndarray],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    palette: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay time-series trajectories for a single metric across branches.

    Parameters
    ----------
    control_ts : np.ndarray (T,)
        Time-series from the control branch.
    treatment_ts_dict : dict[label, np.ndarray]
        Time-series arrays from each treatment branch.
    """
    palette = palette or DEFAULT_PALETTE
    title = title or f"Trajectory: {metric_key.replace('.', ' › ')}"

    treatment_colors = [
        palette["amber"], palette["teal"], palette["violet"],
        palette["rose"],  palette["sky"],  palette["lime"],
    ]

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    t_ctrl = np.arange(len(control_ts))
    ax.plot(t_ctrl, control_ts, color=palette["text_dim"], linewidth=2.5,
            linestyle="--", label="Control", alpha=0.9, zorder=5)

    for i, (lbl, ts) in enumerate(treatment_ts_dict.items()):
        color = treatment_colors[i % len(treatment_colors)]
        t_t = np.arange(len(ts))
        ax.plot(t_t, ts, color=color, linewidth=2, label=lbl, alpha=0.85)

    ax.set_xlabel("Step", fontsize=9)
    y_label = metric_key.split(".")[-1].replace("_", " ").title()
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(title, color=palette["text"], fontsize=11, pad=12)
    ax.legend(fontsize=8, facecolor=palette["surface"],
              edgecolor=palette["border"], labelcolor=palette["text"])

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette["bg"])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3. Opinion shift violin plot
# ═════════════════════════════════════════════════════════════════════════════

def opinion_shift_violin(
    pre_opinions:  np.ndarray,
    post_control:  np.ndarray,
    post_treatment_dict: Dict[str, np.ndarray],
    layer_idx: int = 0,
    title: str = "Opinion Distribution Shift",
    figsize: Tuple[int, int] = (10, 5),
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Violin plots comparing opinion distributions: pre / control / each treatment.

    Parameters
    ----------
    pre_opinions : np.ndarray (N, L) or (N,)
        Opinions at the pre-intervention checkpoint.
    post_control : np.ndarray (N, L) or (N,)
        Final opinions of the control branch.
    post_treatment_dict : dict[label, np.ndarray]
        Final opinions for each treatment arm.
    """
    palette = palette or DEFAULT_PALETTE

    def _extract(op: np.ndarray) -> np.ndarray:
        if op.ndim == 2:
            return op[:, layer_idx].astype(float)
        return op.astype(float)

    groups = [("Pre", _extract(pre_opinions)), ("Control", _extract(post_control))]
    for lbl, op in post_treatment_dict.items():
        groups.append((lbl, _extract(op)))

    group_colors = [
        palette["text_dim"],
        palette["sky"],
        palette["amber"],
        palette["teal"],
        palette["violet"],
        palette["rose"],
    ]

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    positions = list(range(1, len(groups) + 1))
    labels = [g[0] for g in groups]
    data   = [g[1] for g in groups]

    parts = ax.violinplot(data, positions=positions, showmedians=True,
                          showextrema=True, widths=0.65)

    for i, pc in enumerate(parts["bodies"]):
        c = group_colors[i % len(group_colors)]
        pc.set_facecolor(c)
        pc.set_edgecolor(palette["border"])
        pc.set_alpha(0.65)

    for artist_key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[artist_key].set_color(palette["text_dim"])
        parts[artist_key].set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Opinion Value", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, color=palette["text"], fontsize=11, pad=12)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette["bg"])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 4. Ranking heatmap
# ═════════════════════════════════════════════════════════════════════════════

def ranking_heatmap(
    summary_rows: List[Dict[str, Any]],
    metric_columns: Optional[List[str]] = None,
    title: str = "Policy Ranking Comparison",
    figsize: Tuple[int, int] = (11, 6),
    palette: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Colour-coded table of policy scores across metrics.

    Each cell is coloured by its rank within its column:
        green = best in column, red = worst in column.

    Parameters
    ----------
    summary_rows : list[dict]
        Output of InterventionComparator.summary_table().
    metric_columns : list[str], optional
        Which columns to display.  Defaults to standard effect metrics.
    """
    palette = palette or DEFAULT_PALETTE

    default_cols = [
        "effect_score", "polarization_reduction", "bimodality_reduction",
        "extreme_share_change", "convergence_speed_ratio",
        "opinion_mean_shift", "moran_i_change",
    ]
    cols = metric_columns or default_cols
    # Filter to columns that actually exist in at least one row
    cols = [c for c in cols if any(c in r for r in summary_rows)]

    if not summary_rows or not cols:
        fig, ax = plt.subplots(figsize=figsize)
        apply_style(fig, [ax], palette)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color=palette["text_dim"], transform=ax.transAxes)
        return fig

    row_labels = [r.get("label", f"policy_{i}") for i, r in enumerate(summary_rows)]
    matrix = np.full((len(summary_rows), len(cols)), np.nan)

    for i, row in enumerate(summary_rows):
        for j, col in enumerate(cols):
            v = row.get(col)
            if v is not None:
                try:
                    matrix[i, j] = float(v)
                except (TypeError, ValueError):
                    pass

    # Normalise each column to [0, 1] for colour mapping
    norm_matrix = np.full_like(matrix, 0.5)
    for j in range(matrix.shape[1]):
        col_vals = matrix[:, j]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) < 2:
            continue
        col_min, col_max = valid.min(), valid.max()
        span = col_max - col_min
        if span > 1e-10:
            norm_matrix[:, j] = (col_vals - col_min) / span

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "intervention_heatmap",
        [palette["rose"], palette["surface"], palette["teal"]],
        N=256,
    )

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    im = ax.imshow(norm_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate cells with actual values
    for i in range(len(summary_rows)):
        for j in range(len(cols)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        color=palette["text"], fontsize=7.5,
                        fontfamily="monospace")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, color=palette["text"], fontsize=11, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors=palette["text_dim"], labelsize=7)
    cbar.set_label("Relative rank (higher = better)", color=palette["text_dim"], fontsize=7)
    cbar.outline.set_edgecolor(palette["border"])

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette["bg"])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 5. Attribution waterfall chart
# ═════════════════════════════════════════════════════════════════════════════

def attribution_waterfall(
    attributed_log: List[Dict[str, Any]],
    metric: str = "std_delta",
    title: str = "Per-Firing Attribution Waterfall",
    figsize: Tuple[int, int] = (10, 5),
    palette: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Waterfall chart showing the cumulative contribution of each intervention firing.

    Each bar represents one policy firing.  Running total is shown as a line.

    Parameters
    ----------
    attributed_log : list[dict]
        Output of InterventionAttributor.attribute_execution_log().
    metric : str
        Key within each entry's 'attribution' dict to visualise.
    """
    palette = palette or DEFAULT_PALETTE

    labels = []
    values = []

    for entry in attributed_log:
        attribution = entry.get("attribution") or {}
        v = attribution.get(metric)
        if v is not None:
            lbl = entry.get("label", "")
            step = entry.get("step", 0)
            labels.append(f"{lbl}\n(t={step})")
            values.append(float(v))

    if not values:
        fig, ax = plt.subplots(figsize=figsize)
        apply_style(fig, [ax], palette)
        ax.text(0.5, 0.5, f"No attribution data for '{metric}'",
                ha="center", va="center",
                color=palette["text_dim"], transform=ax.transAxes)
        return fig

    running = np.cumsum(values)
    bottoms = np.concatenate([[0], running[:-1]])

    colors = [palette["teal"] if v >= 0 else palette["rose"] for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    x = np.arange(len(values))
    ax.bar(x, values, bottom=bottoms, color=colors, alpha=0.8,
           width=0.6, edgecolor=palette["border"])

    # Running total line
    ax.plot(x, running, color=palette["amber"], linewidth=2,
            marker="o", markersize=5, zorder=5, label="Cumulative")

    ax.axhline(0, color=palette["border"], linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"Δ {metric.replace('_', ' ')}", fontsize=9)
    ax.set_title(title, color=palette["text"], fontsize=11, pad=12)
    ax.legend(fontsize=8, facecolor=palette["surface"],
              edgecolor=palette["border"], labelcolor=palette["text"])

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette["bg"])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 6. Phase portrait comparison
# ═════════════════════════════════════════════════════════════════════════════

def phase_portrait_comparison(
    pre_opinions:   np.ndarray,
    post_control:   np.ndarray,
    post_treatment: np.ndarray,
    layer_x: int = 0,
    layer_y: int = 1,
    title: str = "Opinion Phase Portrait: Pre / Control / Treatment",
    figsize: Tuple[int, int] = (14, 5),
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side scatter plots in opinion phase space (layer_x vs layer_y)
    for three states: pre-intervention, control, and treatment.

    Requires opinions with at least 2 layers (N, L≥2).
    """
    palette = palette or DEFAULT_PALETTE
    opinion_cmap = opinion_cmap or DEFAULT_OPINION_CMAP

    def _slice(op: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if op.ndim == 1:
            ox = op
            oy = op
        else:
            ox = op[:, layer_x % op.shape[1]]
            oy = op[:, layer_y % op.shape[1]]
        return ox, oy

    states = [
        ("Pre-Intervention", pre_opinions,   palette["text_dim"]),
        ("Control",          post_control,   palette["sky"]),
        ("Treatment",        post_treatment, palette["amber"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    apply_style(fig, axes, palette)

    for ax, (lbl, op, _) in zip(axes, states):
        ox, oy = _slice(op)
        # Colour by mean opinion across layers as a proxy
        color_val = (ox + oy) / 2.0
        sc = ax.scatter(ox, oy, c=color_val, cmap=opinion_cmap,
                        s=8, alpha=0.65, linewidths=0, vmin=0, vmax=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Layer {layer_x}", fontsize=8)
        ax.set_ylabel(f"Layer {layer_y}", fontsize=8)
        ax.set_title(lbl, color=palette["text"], fontsize=10)

        mu = ox.mean()
        sigma = ox.std()
        ax.text(0.02, 0.97, f"μ={mu:.3f}\nσ={sigma:.3f}",
                transform=ax.transAxes, va="top",
                color=palette["amber"], fontsize=8, fontfamily="monospace")

    fig.suptitle(title, color=palette["text"], fontsize=12,
                 fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, palette["bg"])
    return fig
