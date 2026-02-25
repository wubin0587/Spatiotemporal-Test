"""
analysis/visual/static.py

Static Visualization Module for Opinion Dynamics Simulation
------------------------------------------------------------
Provides publication-quality static plots for analyzing simulation results.
All functions accept standard numpy arrays and return matplotlib figures.

Design Philosophy:
- Dark scientific aesthetic with amber/teal accent palette
- Every plot is self-contained and can be used in papers
- Consistent visual language across all chart types
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Any, Tuple, Sequence

# =========================================================================
# 1. Color & Style System
# =========================================================================

# Simulation color palette
PALETTE = {
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

# Gradient for opinions [0→1]: deep blue → neutral gray → deep amber
OPINION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'opinion',
    ['#2563eb', '#64748b', '#f59e0b'],
    N=256
)

# Gradient for impact intensity
IMPACT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'impact',
    ['#0d0f14', '#1e3a5f', '#0ea5e9', '#f59e0b', '#ef4444'],
    N=256
)

SOURCE_COLORS = {
    'exogenous':           PALETTE['amber'],
    'endogenous_threshold': PALETTE['teal'],
    'cascade':             PALETTE['violet'],
    'unknown':             PALETTE['text_dim'],
}


def _apply_dark_style(fig: plt.Figure, axes):
    """Apply consistent dark scientific style to figure and axes."""
    fig.patch.set_facecolor(PALETTE['bg'])
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(PALETTE['surface'])
        ax.tick_params(colors=PALETTE['text_dim'], labelsize=8)
        ax.xaxis.label.set_color(PALETTE['text_dim'])
        ax.yaxis.label.set_color(PALETTE['text_dim'])
        if ax.get_title():
            ax.title.set_color(PALETTE['text'])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE['border'])
        ax.grid(True, color=PALETTE['border'], linewidth=0.5, alpha=0.6)


# =========================================================================
# 2. Opinion Distribution
# =========================================================================

def plot_opinion_distribution(
    opinions: np.ndarray,
    layer_idx: int = 0,
    bins: int = 40,
    title: str = "Opinion Distribution",
    figsize: Tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Histogram of opinion values for a given layer with KDE overlay.

    Args:
        opinions: Shape (N, L) or (N,)
        layer_idx: Which opinion dimension to plot
        bins: Number of histogram bins
        title: Plot title

    Returns:
        matplotlib Figure
    """
    if opinions.ndim == 2:
        data = opinions[:, layer_idx]
    else:
        data = opinions.ravel()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    # Histogram
    counts, edges, patches = ax.hist(
        data, bins=bins, range=(0, 1),
        color=PALETTE['amber'], alpha=0.6, edgecolor=PALETTE['bg'], linewidth=0.5
    )

    # Color-code bars by position (blue→amber gradient)
    for patch, left in zip(patches, edges[:-1]):
        c = OPINION_CMAP(left)
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data, bw_method=0.1)
    xs = np.linspace(0, 1, 300)
    ys = kde(xs) * len(data) * (edges[1] - edges[0])
    ax.plot(xs, ys, color=PALETTE['sky'], linewidth=2, alpha=0.9)

    # Mean / std annotations
    mu, sigma = data.mean(), data.std()
    ax.axvline(mu, color=PALETTE['rose'], linewidth=1.5, linestyle='--', alpha=0.8)
    ax.text(
        mu + 0.02, counts.max() * 0.9,
        f'μ={mu:.3f}\nσ={sigma:.3f}',
        color=PALETTE['text'], fontsize=8,
        va='top', fontfamily='monospace'
    )

    ax.set_xlabel('Opinion Value', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(title, color=PALETTE['text'], fontsize=11, pad=12)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    return fig


# =========================================================================
# 3. Spatial Agent Map
# =========================================================================

def plot_spatial_opinions(
    positions: np.ndarray,
    opinions: np.ndarray,
    impact: Optional[np.ndarray] = None,
    layer_idx: int = 0,
    title: str = "Agent Spatial Distribution",
    figsize: Tuple[int, int] = (7, 7),
) -> plt.Figure:
    """
    Scatter plot of agents in 2D space colored by opinion value.
    Optionally scales marker size by impact field.

    Args:
        positions: Shape (N, 2)
        opinions: Shape (N, L) or (N,)
        impact: Shape (N,) — if given, scales marker sizes
        layer_idx: Opinion layer to color-code
        title: Plot title

    Returns:
        matplotlib Figure
    """
    if opinions.ndim == 2:
        color_vals = opinions[:, layer_idx]
    else:
        color_vals = opinions.ravel()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    # Base marker sizes
    if impact is not None:
        norm_impact = impact / (impact.max() + 1e-9)
        sizes = 10 + norm_impact * 60
    else:
        sizes = 12

    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=color_vals, cmap=OPINION_CMAP,
        s=sizes, alpha=0.75, linewidths=0,
        vmin=0, vmax=1
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=PALETTE['text_dim'], labelsize=7)
    cbar.set_label('Opinion', color=PALETTE['text_dim'], fontsize=8)
    cbar.outline.set_edgecolor(PALETTE['border'])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_title(title, color=PALETTE['text'], fontsize=11, pad=12)
    ax.set_aspect('equal')

    if impact is not None:
        # Legend for marker size
        for val, label in [(0.0, 'Low Impact'), (0.5, 'Mid Impact'), (1.0, 'High Impact')]:
            ax.scatter([], [], s=10 + val * 60, c=[PALETTE['text_dim']],
                       alpha=0.5, label=label)
        leg = ax.legend(fontsize=7, loc='lower right',
                        facecolor=PALETTE['surface'], edgecolor=PALETTE['border'],
                        labelcolor=PALETTE['text_dim'])

    fig.tight_layout()
    return fig


# =========================================================================
# 4. Opinion Time-Series
# =========================================================================

def plot_opinion_timeseries(
    history_opinions: List[np.ndarray],
    history_times: List[float],
    layer_idx: int = 0,
    n_sample: int = 50,
    title: str = "Opinion Trajectories",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Spaghetti plot of individual opinion trajectories over time.

    Args:
        history_opinions: List of (N, L) arrays at each recorded step
        history_times: Corresponding time values
        layer_idx: Opinion dimension to trace
        n_sample: Number of random agents to trace (for readability)

    Returns:
        matplotlib Figure
    """
    T = len(history_times)
    N = history_opinions[0].shape[0]

    # Sample agents
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=min(n_sample, N), replace=False)

    # Build trajectory matrix
    traj = np.array([history_opinions[t][:, layer_idx] for t in range(T)])  # (T, N)

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    # Color each trajectory by final opinion
    final_ops = traj[-1]
    for i in idx:
        color = OPINION_CMAP(final_ops[i])
        ax.plot(history_times, traj[:, i],
                color=color, alpha=0.25, linewidth=0.8)

    # Population mean
    ax.plot(history_times, traj.mean(axis=1),
            color=PALETTE['sky'], linewidth=2.5, label='Population Mean', zorder=5)

    # Std band
    mean_t = traj.mean(axis=1)
    std_t = traj.std(axis=1)
    ax.fill_between(history_times, mean_t - std_t, mean_t + std_t,
                    color=PALETTE['sky'], alpha=0.1)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Time', fontsize=9)
    ax.set_ylabel('Opinion Value', fontsize=9)
    ax.set_title(title, color=PALETTE['text'], fontsize=11, pad=12)

    leg = ax.legend(fontsize=8, facecolor=PALETTE['surface'],
                    edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])

    fig.tight_layout()
    return fig


# =========================================================================
# 5. Impact Field Heatmap
# =========================================================================

def plot_impact_heatmap(
    positions: np.ndarray,
    impact: np.ndarray,
    event_locs: Optional[np.ndarray] = None,
    grid_res: int = 60,
    title: str = "Impact Field Intensity",
    figsize: Tuple[int, int] = (7, 7),
) -> plt.Figure:
    """
    2D heatmap of the impact field interpolated onto a regular grid.

    Args:
        positions: Agent positions (N, 2)
        impact: Impact values (N,)
        event_locs: Event locations to overlay (M, 2)
        grid_res: Resolution of the interpolation grid

    Returns:
        matplotlib Figure
    """
    from scipy.interpolate import griddata

    xi = np.linspace(0, 1, grid_res)
    yi = np.linspace(0, 1, grid_res)
    xx, yy = np.meshgrid(xi, yi)

    zz = griddata(positions, impact, (xx, yy), method='linear', fill_value=0)
    zz = np.clip(zz, 0, None)

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    im = ax.imshow(
        zz, origin='lower', extent=[0, 1, 0, 1],
        cmap=IMPACT_CMAP, aspect='equal',
        vmin=0, vmax=zz.max() if zz.max() > 0 else 1
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=PALETTE['text_dim'], labelsize=7)
    cbar.set_label('Impact I(t)', color=PALETTE['text_dim'], fontsize=8)
    cbar.outline.set_edgecolor(PALETTE['border'])

    # Overlay event epicenters
    if event_locs is not None and len(event_locs) > 0:
        ax.scatter(event_locs[:, 0], event_locs[:, 1],
                   marker='*', s=120, c=PALETTE['rose'],
                   edgecolors='white', linewidths=0.5,
                   zorder=10, label='Events')
        ax.legend(fontsize=8, facecolor=PALETTE['surface'],
                  edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])

    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_title(title, color=PALETTE['text'], fontsize=11, pad=12)

    fig.tight_layout()
    return fig


# =========================================================================
# 6. Event Timeline
# =========================================================================

def plot_event_timeline(
    event_times: np.ndarray,
    event_intensities: np.ndarray,
    event_sources: List[str],
    total_time: Optional[float] = None,
    title: str = "Event Timeline",
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Stem plot of events over time, colored by source and sized by intensity.

    Args:
        event_times: Event times (M,)
        event_intensities: Event intensities (M,)
        event_sources: Event source labels (M,)
        total_time: Maximum time axis value

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    if total_time is None:
        total_time = event_times.max() * 1.05 if len(event_times) > 0 else 100

    # Normalize intensity for marker sizes
    max_int = event_intensities.max() if event_intensities.max() > 0 else 1
    norm_int = event_intensities / max_int

    for src, color in SOURCE_COLORS.items():
        mask = np.array(event_sources) == src
        if not mask.any():
            continue
        t = event_times[mask]
        s = norm_int[mask]

        # Stem lines
        for ti, si in zip(t, s):
            ax.plot([ti, ti], [0, si], color=color, linewidth=0.8, alpha=0.5)

        # Dots
        ax.scatter(t, s, s=30 + s * 80, c=[color] * mask.sum(),
                   alpha=0.85, zorder=5, label=src.replace('_', ' ').title())

    ax.axhline(0, color=PALETTE['border'], linewidth=1)
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Time', fontsize=9)
    ax.set_ylabel('Normalized Intensity', fontsize=9)
    ax.set_title(title, color=PALETTE['text'], fontsize=11, pad=12)

    leg = ax.legend(fontsize=8, facecolor=PALETTE['surface'],
                    edgecolor=PALETTE['border'], labelcolor=PALETTE['text'],
                    ncol=3, loc='upper right')

    fig.tight_layout()
    return fig


# =========================================================================
# 7. Polarization Evolution
# =========================================================================

def plot_polarization_evolution(
    history_opinions: List[np.ndarray],
    history_times: List[float],
    history_num_events: Optional[List[int]] = None,
    title: str = "Polarization Over Time",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Dual-axis plot: polarization (std) + event activity over time.

    Args:
        history_opinions: List of (N, L) arrays
        history_times: Time values
        history_num_events: Event counts per step (optional)

    Returns:
        matplotlib Figure
    """
    T = len(history_times)

    # Compute per-layer polarization
    n_layers = history_opinions[0].shape[1]
    pol_matrix = np.array([history_opinions[t].std(axis=0) for t in range(T)])  # (T, L)

    layer_colors = [PALETTE['amber'], PALETTE['teal'], PALETTE['violet'],
                    PALETTE['rose'], PALETTE['sky']]

    if history_num_events is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        axes = [ax1]

    _apply_dark_style(fig, axes)

    for l in range(n_layers):
        c = layer_colors[l % len(layer_colors)]
        ax1.plot(history_times, pol_matrix[:, l],
                 color=c, linewidth=2, label=f'Layer {l}', alpha=0.9)

    ax1.set_ylabel('Polarization (σ)', fontsize=9)
    ax1.set_title(title, color=PALETTE['text'], fontsize=11, pad=12)
    ax1.set_ylim(0, 0.55)
    leg = ax1.legend(fontsize=8, facecolor=PALETTE['surface'],
                     edgecolor=PALETTE['border'], labelcolor=PALETTE['text'])

    if history_num_events is not None:
        ax2.bar(history_times, history_num_events,
                color=PALETTE['rose'], alpha=0.6, width=0.9)
        ax2.set_ylabel('Events', fontsize=8)
        ax2.set_xlabel('Time', fontsize=9)
    else:
        ax1.set_xlabel('Time', fontsize=9)

    fig.tight_layout()
    return fig


# =========================================================================
# 8. Simulation Dashboard (Composite)
# =========================================================================

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
) -> plt.Figure:
    """
    Composite dashboard with spatial map, opinion distribution,
    trajectory lines, and event timeline.

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize, facecolor=PALETTE['bg'])
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.06, right=0.97, top=0.92, bottom=0.06)

    axes = {
        'spatial':  fig.add_subplot(gs[0:2, 0]),
        'impact':   fig.add_subplot(gs[0:2, 1]),
        'dist':     fig.add_subplot(gs[0, 2]),
        'pol':      fig.add_subplot(gs[1, 2]),
        'timeline': fig.add_subplot(gs[2, :]),
    }
    _apply_dark_style(fig, list(axes.values()))

    # --- Spatial opinions ---
    ax = axes['spatial']
    scatter = ax.scatter(
        final_positions[:, 0], final_positions[:, 1],
        c=final_opinions[:, 0], cmap=OPINION_CMAP,
        s=8, alpha=0.7, linewidths=0, vmin=0, vmax=1
    )
    if event_locs is not None and len(event_locs) > 0:
        ax.scatter(event_locs[:, 0], event_locs[:, 1],
                   marker='*', s=80, c=PALETTE['rose'], zorder=10, alpha=0.8)
    ax.set_title('Agent Opinions (Layer 0)', color=PALETTE['text'], fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=PALETTE['text_dim'], labelsize=6)
    cbar.outline.set_edgecolor(PALETTE['border'])

    # --- Impact heatmap ---
    ax = axes['impact']
    from scipy.interpolate import griddata
    xi = np.linspace(0, 1, 50); yi = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata(final_positions, final_impact, (xx, yy), method='linear', fill_value=0)
    ax.imshow(np.clip(zz, 0, None), origin='lower', extent=[0, 1, 0, 1],
              cmap=IMPACT_CMAP, aspect='equal',
              vmin=0, vmax=max(zz.max(), 0.1))
    ax.set_title('Impact Field I(t)', color=PALETTE['text'], fontsize=9)
    ax.set_aspect('equal')

    # --- Opinion distribution ---
    ax = axes['dist']
    data = final_opinions[:, 0]
    ax.hist(data, bins=30, range=(0, 1), color=PALETTE['amber'], alpha=0.6,
            edgecolor=PALETTE['bg'], linewidth=0.3)
    ax.axvline(data.mean(), color=PALETTE['rose'], linewidth=1.5, linestyle='--')
    ax.set_title('Opinion Distribution', color=PALETTE['text'], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Value', fontsize=7); ax.set_ylabel('Count', fontsize=7)

    # --- Polarization ---
    ax = axes['pol']
    if history_opinions is not None and history_times is not None:
        pol = [o.std() for o in history_opinions]
        ax.plot(history_times, pol, color=PALETTE['teal'], linewidth=2)
        ax.fill_between(history_times, 0, pol, color=PALETTE['teal'], alpha=0.15)
    else:
        ax.text(0.5, 0.5, 'No history', ha='center', va='center',
                color=PALETTE['text_dim'], fontsize=9, transform=ax.transAxes)
    ax.set_title('Polarization σ(t)', color=PALETTE['text'], fontsize=9)
    ax.set_xlabel('Time', fontsize=7); ax.set_ylabel('σ', fontsize=7)

    # --- Event timeline ---
    ax = axes['timeline']
    if (event_times is not None and event_intensities is not None
            and event_sources is not None and len(event_times) > 0):
        total_t = event_times.max() * 1.05
        max_int = event_intensities.max() if event_intensities.max() > 0 else 1
        norm_int = event_intensities / max_int
        for src, color in SOURCE_COLORS.items():
            mask = np.array(event_sources) == src
            if not mask.any():
                continue
            t = event_times[mask]
            s = norm_int[mask]
            for ti, si in zip(t, s):
                ax.plot([ti, ti], [0, si], color=color, linewidth=0.6, alpha=0.4)
            ax.scatter(t, s, s=15 + s * 50, c=[color] * mask.sum(), alpha=0.8,
                       zorder=5, label=src.replace('_', ' ').title())
        ax.set_xlim(0, total_t)
        leg = ax.legend(fontsize=7, facecolor=PALETTE['surface'],
                        edgecolor=PALETTE['border'], labelcolor=PALETTE['text'],
                        ncol=3, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No events', ha='center', va='center',
                color=PALETTE['text_dim'], fontsize=9, transform=ax.transAxes)
    ax.set_title('Event Timeline', color=PALETTE['text'], fontsize=9)
    ax.set_xlabel('Time', fontsize=8); ax.set_ylabel('Intensity', fontsize=8)
    ax.axhline(0, color=PALETTE['border'], linewidth=0.8)

    # Global title
    fig.text(0.5, 0.96, title, ha='center', va='top',
             color=PALETTE['text'], fontsize=14, fontweight='bold',
             fontfamily='monospace')

    return fig


# =========================================================================
# 9. Network Homophily Map
# =========================================================================

def plot_network_homophily(
    positions: np.ndarray,
    opinions: np.ndarray,
    edges: List[Tuple[int, int]],
    layer_idx: int = 0,
    max_edges: int = 2000,
    title: str = "Network Opinion Homophily",
    figsize: Tuple[int, int] = (7, 7),
) -> plt.Figure:
    """
    Visualize the network where edge color encodes opinion agreement.

    Args:
        positions: Agent positions (N, 2)
        opinions: Agent opinions (N, L)
        edges: List of (i, j) edge tuples
        layer_idx: Opinion dimension to measure
        max_edges: Cap on edges to draw for performance

    Returns:
        matplotlib Figure
    """
    op = opinions[:, layer_idx] if opinions.ndim == 2 else opinions

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    # Sample edges if too many
    rng = np.random.default_rng(0)
    if len(edges) > max_edges:
        idx = rng.choice(len(edges), size=max_edges, replace=False)
        edges = [edges[i] for i in idx]

    # Build line segments with disagreement coloring
    segments = []
    colors = []
    for (i, j) in edges:
        segments.append([positions[i], positions[j]])
        # Disagreement ∈ [0,1] → map to color
        diff = abs(op[i] - op[j])
        # 0=agree(teal), 1=disagree(rose)
        c = (1 - diff, diff * 0.25, diff, 0.3)
        colors.append(c)

    lc = LineCollection(segments, colors=colors, linewidths=0.5, zorder=1)
    ax.add_collection(lc)

    # Nodes
    ax.scatter(positions[:, 0], positions[:, 1],
               c=op, cmap=OPINION_CMAP, s=10, alpha=0.85,
               linewidths=0, zorder=5, vmin=0, vmax=1)

    # Colorbar proxy
    sm = plt.cm.ScalarMappable(cmap=OPINION_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=PALETTE['text_dim'], labelsize=7)
    cbar.set_label('Opinion', color=PALETTE['text_dim'], fontsize=8)
    cbar.outline.set_edgecolor(PALETTE['border'])

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=9); ax.set_ylabel('Y', fontsize=9)
    ax.set_title(title, color=PALETTE['text'], fontsize=11, pad=12)

    fig.tight_layout()
    return fig
