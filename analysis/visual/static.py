"""
analysis/visual/static.py

Static Visualization Module for Opinion Dynamics Simulation
------------------------------------------------------------
Provides publication-quality static plots for analyzing simulation results.
All functions accept standard numpy arrays and return matplotlib figures.

Design Philosophy:
- All color options (palettes, colormaps) are configurable via external parameters.
- Built-in figure saving mechanism via `save_path`.
- Pure functional approach (no classes).
- Every plot is self-contained and can be used in papers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Tuple

# =========================================================================
# Default Color & Style System (Used as fallbacks if not provided)
# =========================================================================

DEFAULT_PALETTE = {
    'bg':        '#0d0f14',
    'surface':   '#141720',
    'border':    '#1e2233',
    'text':      '#e8ecf4',
    'text_dim':  '#6b7280',
    'amber':     '#f59e0b',
    'teal':      '#14b8a6',
    'rose':      '#f43f5e',
    'violet':    '#8b5cf6',
    'sky':       '#38bdf8',
    'lime':      '#84cc16',
    'exogenous': '#f59e0b',
    'endogenous':'#14b8a6',
    'cascade':   '#8b5cf6',
}

DEFAULT_OPINION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'opinion', ['#2563eb', '#64748b', '#f59e0b'], N=256
)

DEFAULT_IMPACT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'impact', ['#0d0f14', '#1e3a5f', '#0ea5e9', '#f59e0b', '#ef4444'], N=256
)

DEFAULT_SOURCE_COLORS = {
    'exogenous':           DEFAULT_PALETTE['amber'],
    'endogenous_threshold': DEFAULT_PALETTE['teal'],
    'cascade':             DEFAULT_PALETTE['violet'],
    'unknown':             DEFAULT_PALETTE['text_dim'],
}


def apply_style(fig: plt.Figure, axes, palette: Dict[str, str]):
    """Apply consistent style to figure and axes using the provided palette."""
    fig.patch.set_facecolor(palette['bg'])
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(palette['surface'])
        ax.tick_params(colors=palette['text_dim'], labelsize=8)
        ax.xaxis.label.set_color(palette['text_dim'])
        ax.yaxis.label.set_color(palette['text_dim'])
        if ax.get_title():
            ax.title.set_color(palette['text'])
        for spine in ax.spines.values():
            spine.set_edgecolor(palette['border'])
        ax.grid(True, color=palette['border'], linewidth=0.5, alpha=0.6)


def save_figure(fig: plt.Figure, save_path: str, facecolor: str):
    """Helper function to save figure to disk."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=facecolor, transparent=False)


# =========================================================================
# Plotting Functions
# =========================================================================

def plot_opinion_distribution(
    opinions: np.ndarray,
    layer_idx: int = 0,
    bins: int = 40,
    title: str = "Opinion Distribution",
    figsize: Tuple[int, int] = (8, 4),
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Histogram of opinion values for a given layer with KDE overlay.
    """
    palette = palette or DEFAULT_PALETTE
    opinion_cmap = opinion_cmap or DEFAULT_OPINION_CMAP

    data = opinions[:, layer_idx] if opinions.ndim == 2 else opinions.ravel()

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    # Histogram
    counts, edges, patches = ax.hist(
        data, bins=bins, range=(0, 1),
        color=palette['amber'], alpha=0.6, edgecolor=palette['bg'], linewidth=0.5
    )

    # Color-code bars by position
    for patch, left in zip(patches, edges[:-1]):
        c = opinion_cmap(left)
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data, bw_method=0.1)
    xs = np.linspace(0, 1, 300)
    ys = kde(xs) * len(data) * (edges[1] - edges[0])
    ax.plot(xs, ys, color=palette['sky'], linewidth=2, alpha=0.9)

    # Mean / std annotations
    mu, sigma = data.mean(), data.std()
    ax.axvline(mu, color=palette['rose'], linewidth=1.5, linestyle='--', alpha=0.8)
    ax.text(
        mu + 0.02, counts.max() * 0.9,
        f'μ={mu:.3f}\nσ={sigma:.3f}',
        color=palette['text'], fontsize=8,
        va='top', fontfamily='monospace'
    )

    ax.set_xlabel('Opinion Value', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(title, color=palette['text'], fontsize=11, pad=12)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette['bg'])
    return fig


def plot_spatial_opinions(
    positions: np.ndarray,
    opinions: np.ndarray,
    impact: Optional[np.ndarray] = None,
    layer_idx: int = 0,
    title: str = "Agent Spatial Distribution",
    figsize: Tuple[int, int] = (7, 7),
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot of agents in 2D space colored by opinion value.
    Optionally scales marker size by impact field.
    """
    palette = palette or DEFAULT_PALETTE
    opinion_cmap = opinion_cmap or DEFAULT_OPINION_CMAP

    color_vals = opinions[:, layer_idx] if opinions.ndim == 2 else opinions.ravel()

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    if impact is not None:
        norm_impact = impact / (impact.max() + 1e-9)
        sizes = 10 + norm_impact * 60
    else:
        sizes = 12

    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=color_vals, cmap=opinion_cmap,
        s=sizes, alpha=0.75, linewidths=0,
        vmin=0, vmax=1
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=palette['text_dim'], labelsize=7)
    cbar.set_label('Opinion', color=palette['text_dim'], fontsize=8)
    cbar.outline.set_edgecolor(palette['border'])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_title(title, color=palette['text'], fontsize=11, pad=12)
    ax.set_aspect('equal')

    if impact is not None:
        for val, label in [(0.0, 'Low Impact'), (0.5, 'Mid Impact'), (1.0, 'High Impact')]:
            ax.scatter([], [], s=10 + val * 60, c=[palette['text_dim']],
                       alpha=0.5, label=label)
        ax.legend(fontsize=7, loc='lower right', facecolor=palette['surface'], 
                  edgecolor=palette['border'], labelcolor=palette['text_dim'])

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette['bg'])
    return fig


def plot_opinion_timeseries(
    history_opinions: List[np.ndarray],
    history_times: List[float],
    layer_idx: int = 0,
    n_sample: int = 50,
    title: str = "Opinion Trajectories",
    figsize: Tuple[int, int] = (10, 5),
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Spaghetti plot of individual opinion trajectories over time.
    """
    palette = palette or DEFAULT_PALETTE
    opinion_cmap = opinion_cmap or DEFAULT_OPINION_CMAP

    T = len(history_times)
    N = history_opinions[0].shape[0]

    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=min(n_sample, N), replace=False)
    traj = np.array([history_opinions[t][:, layer_idx] for t in range(T)])

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    final_ops = traj[-1]
    for i in idx:
        color = opinion_cmap(final_ops[i])
        ax.plot(history_times, traj[:, i], color=color, alpha=0.25, linewidth=0.8)

    ax.plot(history_times, traj.mean(axis=1),
            color=palette['sky'], linewidth=2.5, label='Population Mean', zorder=5)

    mean_t = traj.mean(axis=1)
    std_t = traj.std(axis=1)
    ax.fill_between(history_times, mean_t - std_t, mean_t + std_t,
                    color=palette['sky'], alpha=0.1)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Time', fontsize=9)
    ax.set_ylabel('Opinion Value', fontsize=9)
    ax.set_title(title, color=palette['text'], fontsize=11, pad=12)
    ax.legend(fontsize=8, facecolor=palette['surface'], edgecolor=palette['border'], labelcolor=palette['text'])

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette['bg'])
    return fig


def plot_impact_heatmap(
    positions: np.ndarray,
    impact: np.ndarray,
    event_locs: Optional[np.ndarray] = None,
    grid_res: int = 60,
    title: str = "Impact Field Intensity",
    figsize: Tuple[int, int] = (7, 7),
    palette: Optional[Dict[str, str]] = None,
    impact_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    2D heatmap of the impact field interpolated onto a regular grid.
    """
    palette = palette or DEFAULT_PALETTE
    impact_cmap = impact_cmap or DEFAULT_IMPACT_CMAP

    from scipy.interpolate import griddata

    xi = np.linspace(0, 1, grid_res)
    yi = np.linspace(0, 1, grid_res)
    xx, yy = np.meshgrid(xi, yi)

    zz = griddata(positions, impact, (xx, yy), method='linear', fill_value=0)
    zz = np.clip(zz, 0, None)

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    im = ax.imshow(
        zz, origin='lower', extent=[0, 1, 0, 1],
        cmap=impact_cmap, aspect='equal',
        vmin=0, vmax=zz.max() if zz.max() > 0 else 1
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=palette['text_dim'], labelsize=7)
    cbar.set_label('Impact I(t)', color=palette['text_dim'], fontsize=8)
    cbar.outline.set_edgecolor(palette['border'])

    if event_locs is not None and len(event_locs) > 0:
        ax.scatter(event_locs[:, 0], event_locs[:, 1],
                   marker='*', s=120, c=palette['rose'],
                   edgecolors='white', linewidths=0.5, zorder=10, label='Events')
        ax.legend(fontsize=8, facecolor=palette['surface'], edgecolor=palette['border'], labelcolor=palette['text'])

    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_title(title, color=palette['text'], fontsize=11, pad=12)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette['bg'])
    return fig


def plot_event_timeline(
    event_times: np.ndarray,
    event_intensities: np.ndarray,
    event_sources: List[str],
    total_time: Optional[float] = None,
    title: str = "Event Timeline",
    figsize: Tuple[int, int] = (12, 4),
    palette: Optional[Dict[str, str]] = None,
    source_colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Stem plot of events over time, colored by source and sized by intensity.
    """
    palette = palette or DEFAULT_PALETTE
    source_colors = source_colors or DEFAULT_SOURCE_COLORS

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    if total_time is None:
        total_time = event_times.max() * 1.05 if len(event_times) > 0 else 100

    max_int = event_intensities.max() if event_intensities.max() > 0 else 1
    norm_int = event_intensities / max_int

    for src, color in source_colors.items():
        mask = np.array(event_sources) == src
        if not mask.any():
            continue
        t = event_times[mask]
        s = norm_int[mask]

        for ti, si in zip(t, s):
            ax.plot([ti, ti], [0, si], color=color, linewidth=0.8, alpha=0.5)

        ax.scatter(t, s, s=30 + s * 80, c=[color] * mask.sum(),
                   alpha=0.85, zorder=5, label=src.replace('_', ' ').title())

    ax.axhline(0, color=palette['border'], linewidth=1)
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Time', fontsize=9)
    ax.set_ylabel('Normalized Intensity', fontsize=9)
    ax.set_title(title, color=palette['text'], fontsize=11, pad=12)

    ax.legend(fontsize=8, facecolor=palette['surface'], edgecolor=palette['border'], 
              labelcolor=palette['text'], ncol=3, loc='upper right')

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette['bg'])
    return fig


def plot_polarization_evolution(
    history_opinions: List[np.ndarray],
    history_times: List[float],
    history_num_events: Optional[List[int]] = None,
    title: str = "Polarization Over Time",
    figsize: Tuple[int, int] = (10, 5),
    palette: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Dual-axis plot: polarization (std) + event activity over time.
    """
    palette = palette or DEFAULT_PALETTE

    T = len(history_times)
    n_layers = history_opinions[0].shape[1]
    pol_matrix = np.array([history_opinions[t].std(axis=0) for t in range(T)])

    layer_colors = [palette['amber'], palette['teal'], palette['violet'], palette['rose'], palette['sky']]

    if history_num_events is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        axes = [ax1]

    apply_style(fig, axes, palette)

    for l in range(n_layers):
        c = layer_colors[l % len(layer_colors)]
        ax1.plot(history_times, pol_matrix[:, l], color=c, linewidth=2, label=f'Layer {l}', alpha=0.9)

    ax1.set_ylabel('Polarization (σ)', fontsize=9)
    ax1.set_title(title, color=palette['text'], fontsize=11, pad=12)
    ax1.set_ylim(0, 0.55)
    ax1.legend(fontsize=8, facecolor=palette['surface'], edgecolor=palette['border'], labelcolor=palette['text'])

    if history_num_events is not None:
        ax2.bar(history_times, history_num_events, color=palette['rose'], alpha=0.6, width=0.9)
        ax2.set_ylabel('Events', fontsize=8)
        ax2.set_xlabel('Time', fontsize=9)
    else:
        ax1.set_xlabel('Time', fontsize=9)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette['bg'])
    return fig


def plot_simulation_dashboard(
    final_positions: np.ndarray,
    final_opinions: np.ndarray,
    final_impact: np.ndarray,
    history_opinions: Optional[List[np.ndarray]] = None,
    history_times: Optional[List[float]] = None,
    event_times: Optional[np.ndarray] = None,
    event_intensities: Optional[np.ndarray] = None,
    event_sources: Optional[List[str]] = None,
    event_locs: Optional[np.ndarray] = None,
    title: str = "Simulation Summary Dashboard",
    figsize: Tuple[int, int] = (16, 12),
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    impact_cmap: Optional[mcolors.Colormap] = None,
    source_colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Composite dashboard with spatial map, opinion distribution,
    trajectory lines, and event timeline.
    """
    palette = palette or DEFAULT_PALETTE
    opinion_cmap = opinion_cmap or DEFAULT_OPINION_CMAP
    impact_cmap = impact_cmap or DEFAULT_IMPACT_CMAP
    source_colors = source_colors or DEFAULT_SOURCE_COLORS

    fig = plt.figure(figsize=figsize, facecolor=palette['bg'])
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35, left=0.06, right=0.97, top=0.92, bottom=0.06)

    axes = {
        'spatial':  fig.add_subplot(gs[0:2, 0]),
        'impact':   fig.add_subplot(gs[0:2, 1]),
        'dist':     fig.add_subplot(gs[0, 2]),
        'pol':      fig.add_subplot(gs[1, 2]),
        'timeline': fig.add_subplot(gs[2, :]),
    }
    apply_style(fig, list(axes.values()), palette)

    # --- Spatial opinions ---
    ax = axes['spatial']
    scatter = ax.scatter(
        final_positions[:, 0], final_positions[:, 1],
        c=final_opinions[:, 0], cmap=opinion_cmap,
        s=8, alpha=0.7, linewidths=0, vmin=0, vmax=1
    )
    if event_locs is not None and len(event_locs) > 0:
        ax.scatter(event_locs[:, 0], event_locs[:, 1], marker='*', s=80, c=palette['rose'], zorder=10, alpha=0.8)
    ax.set_title('Agent Opinions (Layer 0)', color=palette['text'], fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=palette['text_dim'], labelsize=6)
    cbar.outline.set_edgecolor(palette['border'])

    # --- Impact heatmap ---
    ax = axes['impact']
    from scipy.interpolate import griddata
    xi = np.linspace(0, 1, 50); yi = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata(final_positions, final_impact, (xx, yy), method='linear', fill_value=0)
    ax.imshow(np.clip(zz, 0, None), origin='lower', extent=[0, 1, 0, 1], cmap=impact_cmap, aspect='equal', vmin=0, vmax=max(zz.max(), 0.1))
    ax.set_title('Impact Field I(t)', color=palette['text'], fontsize=9)
    ax.set_aspect('equal')

    # --- Opinion distribution ---
    ax = axes['dist']
    data = final_opinions[:, 0]
    ax.hist(data, bins=30, range=(0, 1), color=palette['amber'], alpha=0.6, edgecolor=palette['bg'], linewidth=0.3)
    ax.axvline(data.mean(), color=palette['rose'], linewidth=1.5, linestyle='--')
    ax.set_title('Opinion Distribution', color=palette['text'], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Value', fontsize=7); ax.set_ylabel('Count', fontsize=7)

    # --- Polarization ---
    ax = axes['pol']
    if history_opinions is not None and history_times is not None:
        pol = [o.std() for o in history_opinions]
        ax.plot(history_times, pol, color=palette['teal'], linewidth=2)
        ax.fill_between(history_times, 0, pol, color=palette['teal'], alpha=0.15)
    else:
        ax.text(0.5, 0.5, 'No history', ha='center', va='center', color=palette['text_dim'], fontsize=9, transform=ax.transAxes)
    ax.set_title('Polarization σ(t)', color=palette['text'], fontsize=9)
    ax.set_xlabel('Time', fontsize=7); ax.set_ylabel('σ', fontsize=7)

    # --- Event timeline ---
    ax = axes['timeline']
    if (event_times is not None and event_intensities is not None and event_sources is not None and len(event_times) > 0):
        total_t = event_times.max() * 1.05
        max_int = event_intensities.max() if event_intensities.max() > 0 else 1
        norm_int = event_intensities / max_int
        for src, color in source_colors.items():
            mask = np.array(event_sources) == src
            if not mask.any():
                continue
            t = event_times[mask]
            s = norm_int[mask]
            for ti, si in zip(t, s):
                ax.plot([ti, ti], [0, si], color=color, linewidth=0.6, alpha=0.4)
            ax.scatter(t, s, s=15 + s * 50, c=[color] * mask.sum(), alpha=0.8, zorder=5, label=src.replace('_', ' ').title())
        ax.set_xlim(0, total_t)
        ax.legend(fontsize=7, facecolor=palette['surface'], edgecolor=palette['border'], labelcolor=palette['text'], ncol=3, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No events', ha='center', va='center', color=palette['text_dim'], fontsize=9, transform=ax.transAxes)
    ax.set_title('Event Timeline', color=palette['text'], fontsize=9)
    ax.set_xlabel('Time', fontsize=8); ax.set_ylabel('Intensity', fontsize=8)
    ax.axhline(0, color=palette['border'], linewidth=0.8)

    fig.text(0.5, 0.96, title, ha='center', va='top', color=palette['text'], fontsize=14, fontweight='bold', fontfamily='monospace')

    if save_path:
        save_figure(fig, save_path, palette['bg'])
    return fig


def plot_network_homophily(
    positions: np.ndarray,
    opinions: np.ndarray,
    edges: List[Tuple[int, int]],
    layer_idx: int = 0,
    max_edges: int = 2000,
    title: str = "Network Opinion Homophily",
    figsize: Tuple[int, int] = (7, 7),
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the network where edge color encodes opinion agreement.
    """
    palette = palette or DEFAULT_PALETTE
    opinion_cmap = opinion_cmap or DEFAULT_OPINION_CMAP

    op = opinions[:, layer_idx] if opinions.ndim == 2 else opinions

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    rng = np.random.default_rng(0)
    if len(edges) > max_edges:
        idx = rng.choice(len(edges), size=max_edges, replace=False)
        edges = [edges[i] for i in idx]

    segments = []
    colors = []
    for (i, j) in edges:
        segments.append([positions[i], positions[j]])
        diff = abs(op[i] - op[j])
        c = (1 - diff, diff * 0.25, diff, 0.3)
        colors.append(c)

    lc = LineCollection(segments, colors=colors, linewidths=0.5, zorder=1)
    ax.add_collection(lc)

    ax.scatter(positions[:, 0], positions[:, 1], c=op, cmap=opinion_cmap, s=10, alpha=0.85, linewidths=0, zorder=5, vmin=0, vmax=1)

    sm = plt.cm.ScalarMappable(cmap=opinion_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=palette['text_dim'], labelsize=7)
    cbar.set_label('Opinion', color=palette['text_dim'], fontsize=8)
    cbar.outline.set_edgecolor(palette['border'])

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=9); ax.set_ylabel('Y', fontsize=9)
    ax.set_title(title, color=palette['text'], fontsize=11, pad=12)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, palette['bg'])
    return fig