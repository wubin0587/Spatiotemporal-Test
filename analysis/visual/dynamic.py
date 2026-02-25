"""
analysis/visual/dynamic.py

Dynamic Visualization Module — Animated Simulation Playback
------------------------------------------------------------
Provides frame-by-frame animations of opinion dynamics, impact fields,
and event cascades. 

Design Philosophy:
- Pure functional approach (no classes).
- All color options (palettes, colormaps) are configurable via external parameters.
- Built-in video/gif export mechanism via `save_path`.
- Output formats: matplotlib FuncAnimation (for Jupyter), .mp4 / .gif via writer.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from typing import List, Optional, Tuple, Dict, Any

try:
    from scipy.interpolate import griddata
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Import default style configurations from the refactored static.py
from .static import (
    DEFAULT_PALETTE,
    DEFAULT_OPINION_CMAP,
    DEFAULT_IMPACT_CMAP,
    DEFAULT_SOURCE_COLORS,
    apply_style
)


# =========================================================================
# 1. Agent Opinion Evolution Animation
# =========================================================================

def animate_opinion_evolution(
    positions: np.ndarray,
    history_opinions: List[np.ndarray],
    history_times: List[float],
    history_events: Optional[List[List[Dict[str, Any]]]] = None,
    layer_idx: int = 0,
    figsize: Tuple[int, int] = (8, 8),
    fps: int = 15,
    title: str = "Opinion Dynamics",
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    source_colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    dpi: int = 120,
    writer: str = 'ffmpeg'
) -> animation.FuncAnimation:
    """
    Animates agents on a 2D spatial map colored by their opinion value.
    Events are shown as ripple rings when they trigger.
    """
    palette = palette or DEFAULT_PALETTE
    opinion_cmap = opinion_cmap or DEFAULT_OPINION_CMAP
    source_colors = source_colors or DEFAULT_SOURCE_COLORS
    history_events = history_events or [[] for _ in history_times]
    n_frames = len(history_times)

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    op0 = history_opinions[0][:, layer_idx]
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=op0, cmap=opinion_cmap, s=12, alpha=0.8,
        linewidths=0, vmin=0, vmax=1
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=palette['text_dim'], labelsize=7)
    cbar.set_label('Opinion', color=palette['text_dim'], fontsize=8)
    cbar.outline.set_edgecolor(palette['border'])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_title(title, color=palette['text'], fontsize=11, pad=12)

    time_text = ax.text(
        0.02, 0.97, '', transform=ax.transAxes,
        color=palette['amber'], fontsize=9, va='top',
        fontfamily='monospace'
    )

    std_text = ax.text(
        0.98, 0.97, '', transform=ax.transAxes,
        color=palette['teal'], fontsize=9, va='top', ha='right',
        fontfamily='monospace'
    )

    # Event ripple artists (reused pool)
    ripple_circles = []
    for _ in range(10):
        c = plt.Circle((0.5, 0.5), 0, fill=False,
                       edgecolor=palette['rose'], linewidth=0, alpha=0)
        ax.add_patch(c)
        ripple_circles.append(c)

    def init():
        scatter.set_array(history_opinions[0][:, layer_idx])
        time_text.set_text('')
        std_text.set_text('')
        for c in ripple_circles:
            c.set_radius(0)
            c.set_alpha(0)
        return [scatter, time_text, std_text] + ripple_circles

    def update(frame):
        ops = history_opinions[frame][:, layer_idx]
        scatter.set_array(ops)

        t = history_times[frame]
        time_text.set_text(f't = {t:.1f}')
        std_text.set_text(f'σ = {ops.std():.3f}')

        # Draw event ripples
        events = history_events[frame]
        for k, rc in enumerate(ripple_circles):
            if k < len(events):
                ev = events[k]
                loc = ev.get('loc', (0.5, 0.5))
                intensity = ev.get('intensity', 1.0)
                src = ev.get('source', 'unknown')
                color = source_colors.get(src, palette['rose'])
                
                rc.center = loc
                rc.set_radius(min(0.04 + intensity * 0.005, 0.25))
                rc.set_edgecolor(color)
                rc.set_linewidth(1.5)
                rc.set_alpha(0.6)
            else:
                rc.set_alpha(0)

        return [scatter, time_text, std_text] + ripple_circles

    interval = max(1, int(1000 / fps))
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True, interval=interval
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        anim.save(save_path, fps=fps, dpi=dpi, writer=writer,
                  savefig_kwargs={'facecolor': palette['bg']})
        print(f"[Animator] Saved Opinion Evolution → {save_path}")

    return anim


# =========================================================================
# 2. Impact Field Animation
# =========================================================================

def animate_impact_field(
    positions: np.ndarray,
    history_impact: List[np.ndarray],
    history_times: List[float],
    event_locs_per_step: Optional[List[np.ndarray]] = None,
    grid_res: int = 40,
    figsize: Tuple[int, int] = (7, 7),
    fps: int = 12,
    title: str = "Impact Field I(t)",
    palette: Optional[Dict[str, str]] = None,
    impact_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None,
    dpi: int = 100,
    writer: str = 'ffmpeg'
) -> animation.FuncAnimation:
    """
    Animates the 2D impact field heatmap over time.
    """
    palette = palette or DEFAULT_PALETTE
    impact_cmap = impact_cmap or DEFAULT_IMPACT_CMAP
    event_locs = event_locs_per_step or [None] * len(history_times)
    n_frames = len(history_times)

    def _interpolate(impact: np.ndarray) -> np.ndarray:
        if not _HAS_SCIPY:
            return np.zeros((grid_res, grid_res))
        xi = np.linspace(0, 1, grid_res)
        yi = np.linspace(0, 1, grid_res)
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata(positions, impact, (xx, yy), method='linear', fill_value=0)
        return np.clip(zz, 0, None)

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, [ax], palette)

    all_max = max(imp.max() for imp in history_impact if imp.max() > 0)
    all_max = max(all_max, 0.1)

    z0 = _interpolate(history_impact[0])
    im = ax.imshow(
        z0, origin='lower', extent=[0, 1, 0, 1],
        cmap=impact_cmap, aspect='equal',
        vmin=0, vmax=all_max, animated=True
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=palette['text_dim'], labelsize=7)
    cbar.set_label('Impact I(t)', color=palette['text_dim'], fontsize=8)
    cbar.outline.set_edgecolor(palette['border'])

    event_scatter = ax.scatter([], [], marker='*', s=100,
                               c=palette['rose'], zorder=10, alpha=0.9)

    time_text = ax.text(
        0.02, 0.97, '', transform=ax.transAxes,
        color=palette['amber'], fontsize=9, va='top',
        fontfamily='monospace'
    )

    ax.set_title(title, color=palette['text'], fontsize=11, pad=12)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)

    def init():
        im.set_data(_interpolate(history_impact[0]))
        event_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return [im, event_scatter, time_text]

    def update(frame):
        zz = _interpolate(history_impact[frame])
        im.set_data(zz)

        locs = event_locs[frame]
        if locs is not None and len(locs) > 0:
            event_scatter.set_offsets(locs)
        else:
            event_scatter.set_offsets(np.empty((0, 2)))

        t = history_times[frame]
        mean_impact = history_impact[frame].mean()
        time_text.set_text(f't={t:.1f}  μI={mean_impact:.2f}')
        return [im, event_scatter, time_text]

    interval = max(1, int(1000 / fps))
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True, interval=interval
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        anim.save(save_path, fps=fps, dpi=dpi, writer=writer,
                  savefig_kwargs={'facecolor': palette['bg']})
        print(f"[Animator] Saved Impact Field → {save_path}")

    return anim


# =========================================================================
# 3. Dual-Panel "Algorithm vs Reality" Animation
# =========================================================================

def animate_dual_mode(
    positions: np.ndarray,
    history_opinions: List[np.ndarray],
    history_impact: List[np.ndarray],
    history_times: List[float],
    layer_idx: int = 0,
    mode_threshold: float = 0.25,
    figsize: Tuple[int, int] = (14, 6),
    fps: int = 12,
    title: str = "Algorithm Mode vs Reality Mode",
    palette: Optional[Dict[str, str]] = None,
    opinion_cmap: Optional[mcolors.Colormap] = None,
    save_path: Optional[str] = None,
    dpi: int = 120,
    writer: str = 'ffmpeg'
) -> animation.FuncAnimation:
    """
    Side-by-side animation showing:
    LEFT:  Agent mode (algorithm vs reality/event-driven)
    RIGHT: Opinion distribution over time
    """
    palette = palette or DEFAULT_PALETTE
    opinion_cmap = opinion_cmap or DEFAULT_OPINION_CMAP
    n_frames = len(history_times)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
    apply_style(fig, [ax_left, ax_right], palette)

    N = len(positions)
    impact0 = history_impact[0]
    modes0 = (impact0 >= mode_threshold).astype(float)

    mode_cmap = mcolors.LinearSegmentedColormap.from_list(
        'mode', [palette['sky'], palette['amber']], N=2
    )

    # LEFT: Mode scatter
    scatter_mode = ax_left.scatter(
        positions[:, 0], positions[:, 1],
        c=modes0, cmap=mode_cmap, s=12, alpha=0.8,
        linewidths=0, vmin=0, vmax=1
    )
    ax_left.set_title('Agent Mode', color=palette['text'], fontsize=10)
    ax_left.set_aspect('equal')
    ax_left.set_xlim(-0.02, 1.02); ax_left.set_ylim(-0.02, 1.02)

    algo_patch = mpatches.Patch(color=palette['sky'], label='Algorithm Mode')
    real_patch = mpatches.Patch(color=palette['amber'], label='Reality Mode')
    ax_left.legend(handles=[algo_patch, real_patch], fontsize=7,
                   facecolor=palette['surface'], edgecolor=palette['border'],
                   labelcolor=palette['text'])

    # RIGHT: Histogram
    op0 = history_opinions[0][:, layer_idx]
    bins = np.linspace(0, 1, 31)
    bar_vals, _, bars = ax_right.hist(op0, bins=bins, color=palette['amber'],
                                      alpha=0.7, edgecolor=palette['bg'], lw=0.3)
    
    for bar, left in zip(bars, bins[:-1]):
        bar.set_facecolor(opinion_cmap(left))
        bar.set_alpha(0.75)

    vline = ax_right.axvline(op0.mean(), color=palette['rose'],
                             linewidth=1.5, linestyle='--')
    ax_right.set_xlim(0, 1)
    ax_right.set_xlabel('Opinion Value', fontsize=9)
    ax_right.set_ylabel('Count', fontsize=9)
    ax_right.set_title('Opinion Distribution', color=palette['text'], fontsize=10)

    # Texts
    stats_text = ax_right.text(
        0.98, 0.95, '', transform=ax_right.transAxes,
        ha='right', va='top', color=palette['teal'], fontsize=8,
        fontfamily='monospace'
    )
    time_text = ax_left.text(
        0.02, 0.97, '', transform=ax_left.transAxes,
        color=palette['amber'], fontsize=9, va='top',
        fontfamily='monospace'
    )
    ratio_text = ax_left.text(
        0.98, 0.97, '', transform=ax_left.transAxes,
        color=palette['teal'], fontsize=8, va='top', ha='right',
        fontfamily='monospace'
    )

    fig.suptitle(title, color=palette['text'], fontsize=12, fontweight='bold', y=1.01)

    def init():
        scatter_mode.set_array(history_impact[0] >= mode_threshold)
        time_text.set_text('')
        stats_text.set_text('')
        ratio_text.set_text('')
        return [scatter_mode, vline, stats_text, time_text, ratio_text]

    def update(frame):
        impact = history_impact[frame]
        ops = history_opinions[frame][:, layer_idx]
        modes = (impact >= mode_threshold).astype(float)
        n_active = int(modes.sum())

        scatter_mode.set_array(modes)

        # Redraw histogram
        vals, _ = np.histogram(ops, bins=bins)
        for bar, v in zip(bars, vals):
            bar.set_height(v)

        vline.set_xdata([ops.mean(), ops.mean()])

        t = history_times[frame]
        time_text.set_text(f't = {t:.1f}')
        ratio_text.set_text(f'Reality: {n_active}/{N}')
        stats_text.set_text(f'μ={ops.mean():.3f}\nσ={ops.std():.3f}')

        return [scatter_mode, vline, stats_text, time_text, ratio_text] + list(bars)

    interval = max(1, int(1000 / fps))
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=False, interval=interval
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        anim.save(save_path, fps=fps, dpi=dpi, writer=writer,
                  savefig_kwargs={'facecolor': palette['bg']})
        print(f"[Animator] Saved Dual Mode → {save_path}")

    return anim