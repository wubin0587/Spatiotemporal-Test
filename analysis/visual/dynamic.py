"""
analysis/visual/dynamic.py

Dynamic Visualization Module — Animated Simulation Playback
------------------------------------------------------------
Provides frame-by-frame animations of opinion dynamics, impact fields,
and event cascades. All animators use the same dark aesthetic as static.py.

Output formats:
- matplotlib FuncAnimation (for Jupyter / display)
- .mp4 / .gif file export via writer
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection
from typing import List, Optional, Tuple, Callable, Dict, Any

# Reuse the palette & colormaps from static
from .static import (
    PALETTE, OPINION_CMAP, IMPACT_CMAP, SOURCE_COLORS,
    _apply_dark_style
)

try:
    from scipy.interpolate import griddata
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# =========================================================================
# 1. Agent Opinion Evolution Animation
# =========================================================================

class OpinionEvolutionAnimator:
    """
    Animates agents on a 2D spatial map colored by their opinion value.
    Events are shown as ripple rings when they trigger.

    Usage:
        animator = OpinionEvolutionAnimator(
            positions=pos,
            history_opinions=history,
            history_times=times
        )
        anim = animator.build()
        anim.save("opinion_evolution.mp4", fps=20, dpi=150)
    """

    def __init__(
        self,
        positions: np.ndarray,
        history_opinions: List[np.ndarray],
        history_times: List[float],
        history_events: Optional[List[List[Dict]]] = None,
        layer_idx: int = 0,
        figsize: Tuple[int, int] = (8, 8),
        fps: int = 15,
        title: str = "Opinion Dynamics",
    ):
        """
        Args:
            positions: Agent positions (N, 2) — assumed static
            history_opinions: List of (N, L) arrays, one per recorded step
            history_times: Timestamps for each frame
            history_events: Optional list of lists of event dicts per step
                            Each event dict: {'loc': (x,y), 'intensity': float, 'source': str}
            layer_idx: Which opinion dimension to animate
            fps: Animation playback speed
        """
        self.positions = positions
        self.history_opinions = history_opinions
        self.history_times = history_times
        self.history_events = history_events or [[] for _ in history_times]
        self.layer_idx = layer_idx
        self.figsize = figsize
        self.fps = fps
        self.title = title
        self.n_frames = len(history_times)
        self._fig = None
        self._anim = None

    def build(self) -> animation.FuncAnimation:
        """Construct and return the FuncAnimation object."""
        fig, ax = plt.subplots(figsize=self.figsize)
        _apply_dark_style(fig, [ax])
        self._fig = fig

        op0 = self.history_opinions[0][:, self.layer_idx]
        scatter = ax.scatter(
            self.positions[:, 0], self.positions[:, 1],
            c=op0, cmap=OPINION_CMAP, s=12, alpha=0.8,
            linewidths=0, vmin=0, vmax=1
        )

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors=PALETTE['text_dim'], labelsize=7)
        cbar.set_label('Opinion', color=PALETTE['text_dim'], fontsize=8)
        cbar.outline.set_edgecolor(PALETTE['border'])

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)

        time_text = ax.text(
            0.02, 0.97, '', transform=ax.transAxes,
            color=PALETTE['amber'], fontsize=9, va='top',
            fontfamily='monospace'
        )

        std_text = ax.text(
            0.98, 0.97, '', transform=ax.transAxes,
            color=PALETTE['teal'], fontsize=9, va='top', ha='right',
            fontfamily='monospace'
        )

        # Event ripple artists (reused pool)
        ripple_circles = []
        for _ in range(10):
            c = plt.Circle((0.5, 0.5), 0, fill=False,
                           edgecolor=PALETTE['rose'], linewidth=0, alpha=0)
            ax.add_patch(c)
            ripple_circles.append(c)

        ax.set_title(self.title, color=PALETTE['text'], fontsize=11, pad=12)

        def init():
            scatter.set_array(self.history_opinions[0][:, self.layer_idx])
            time_text.set_text('')
            std_text.set_text('')
            for c in ripple_circles:
                c.set_radius(0)
                c.set_alpha(0)
            return [scatter, time_text, std_text] + ripple_circles

        def update(frame):
            ops = self.history_opinions[frame][:, self.layer_idx]
            scatter.set_array(ops)

            t = self.history_times[frame]
            time_text.set_text(f't = {t:.1f}')
            std_text.set_text(f'σ = {ops.std():.3f}')

            # Draw event ripples
            events = self.history_events[frame]
            for k, rc in enumerate(ripple_circles):
                if k < len(events):
                    ev = events[k]
                    loc = ev.get('loc', (0.5, 0.5))
                    intensity = ev.get('intensity', 1.0)
                    src = ev.get('source', 'unknown')
                    color = SOURCE_COLORS.get(src, PALETTE['rose'])
                    rc.center = loc
                    rc.set_radius(min(0.04 + intensity * 0.005, 0.25))
                    rc.set_edgecolor(color)
                    rc.set_linewidth(1.5)
                    rc.set_alpha(0.6)
                else:
                    rc.set_alpha(0)

            return [scatter, time_text, std_text] + ripple_circles

        interval = max(1, int(1000 / self.fps))
        self._anim = animation.FuncAnimation(
            fig, update, frames=self.n_frames,
            init_func=init, blit=True, interval=interval
        )
        return self._anim

    def save(self, filepath: str, fps: Optional[int] = None,
             dpi: int = 120, writer: str = 'ffmpeg'):
        """Save animation to file."""
        if self._anim is None:
            self.build()
        fps = fps or self.fps
        self._anim.save(filepath, fps=fps, dpi=dpi, writer=writer,
                        savefig_kwargs={'facecolor': PALETTE['bg']})
        print(f"[Animator] Saved → {filepath}")


# =========================================================================
# 2. Impact Field Animation
# =========================================================================

class ImpactFieldAnimator:
    """
    Animates the 2D impact field heatmap over time.

    Usage:
        animator = ImpactFieldAnimator(positions, history_impact, times)
        anim = animator.build()
    """

    def __init__(
        self,
        positions: np.ndarray,
        history_impact: List[np.ndarray],
        history_times: List[float],
        event_locs_per_step: Optional[List[np.ndarray]] = None,
        grid_res: int = 40,
        figsize: Tuple[int, int] = (7, 7),
        fps: int = 12,
        title: str = "Impact Field I(t)",
    ):
        self.positions = positions
        self.history_impact = history_impact
        self.history_times = history_times
        self.event_locs = event_locs_per_step or [None] * len(history_times)
        self.grid_res = grid_res
        self.figsize = figsize
        self.fps = fps
        self.title = title
        self._fig = None
        self._anim = None

    def _interpolate(self, impact: np.ndarray) -> np.ndarray:
        if not _HAS_SCIPY:
            return np.zeros((self.grid_res, self.grid_res))
        xi = np.linspace(0, 1, self.grid_res)
        yi = np.linspace(0, 1, self.grid_res)
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata(self.positions, impact, (xx, yy), method='linear', fill_value=0)
        return np.clip(zz, 0, None)

    def build(self) -> animation.FuncAnimation:
        fig, ax = plt.subplots(figsize=self.figsize)
        _apply_dark_style(fig, [ax])
        self._fig = fig

        # Compute global max for consistent colorbar
        all_max = max(imp.max() for imp in self.history_impact if imp.max() > 0)
        all_max = max(all_max, 0.1)

        z0 = self._interpolate(self.history_impact[0])
        im = ax.imshow(
            z0, origin='lower', extent=[0, 1, 0, 1],
            cmap=IMPACT_CMAP, aspect='equal',
            vmin=0, vmax=all_max, animated=True
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors=PALETTE['text_dim'], labelsize=7)
        cbar.set_label('Impact I(t)', color=PALETTE['text_dim'], fontsize=8)
        cbar.outline.set_edgecolor(PALETTE['border'])

        event_scatter = ax.scatter([], [], marker='*', s=100,
                                   c=PALETTE['rose'], zorder=10, alpha=0.9)

        time_text = ax.text(
            0.02, 0.97, '', transform=ax.transAxes,
            color=PALETTE['amber'], fontsize=9, va='top',
            fontfamily='monospace'
        )

        ax.set_title(self.title, color=PALETTE['text'], fontsize=11, pad=12)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)

        def init():
            im.set_data(self._interpolate(self.history_impact[0]))
            event_scatter.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return [im, event_scatter, time_text]

        def update(frame):
            zz = self._interpolate(self.history_impact[frame])
            im.set_data(zz)

            locs = self.event_locs[frame]
            if locs is not None and len(locs) > 0:
                event_scatter.set_offsets(locs)
            else:
                event_scatter.set_offsets(np.empty((0, 2)))

            t = self.history_times[frame]
            mean_impact = self.history_impact[frame].mean()
            time_text.set_text(f't={t:.1f}  μI={mean_impact:.2f}')
            return [im, event_scatter, time_text]

        interval = max(1, int(1000 / self.fps))
        self._anim = animation.FuncAnimation(
            fig, update, frames=len(self.history_times),
            init_func=init, blit=True, interval=interval
        )
        return self._anim

    def save(self, filepath: str, fps: Optional[int] = None, dpi: int = 100):
        if self._anim is None:
            self.build()
        fps = fps or self.fps
        self._anim.save(filepath, fps=fps, dpi=dpi,
                        savefig_kwargs={'facecolor': PALETTE['bg']})
        print(f"[ImpactAnimator] Saved → {filepath}")


# =========================================================================
# 3. Dual-Panel "Algorithm vs Reality" Animation
# =========================================================================

class DualModeAnimator:
    """
    Side-by-side animation showing:
    LEFT:  Agent mode (blue=algorithm, amber=reality/event-driven)
    RIGHT: Opinion distribution over time

    This makes the core "filter bubble vs reality shock" mechanism visible.
    """

    def __init__(
        self,
        positions: np.ndarray,
        history_opinions: List[np.ndarray],
        history_impact: List[np.ndarray],
        history_times: List[float],
        layer_idx: int = 0,
        mode_threshold: float = 0.25,
        figsize: Tuple[int, int] = (14, 6),
        fps: int = 12,
        title: str = "Algorithm Mode vs Reality Mode",
    ):
        self.positions = positions
        self.history_opinions = history_opinions
        self.history_impact = history_impact
        self.history_times = history_times
        self.layer_idx = layer_idx
        self.threshold = mode_threshold
        self.figsize = figsize
        self.fps = fps
        self.title = title
        self._fig = None
        self._anim = None

    def build(self) -> animation.FuncAnimation:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=self.figsize)
        _apply_dark_style(fig, [ax_left, ax_right])
        self._fig = fig

        N = len(self.positions)
        impact0 = self.history_impact[0]
        modes0 = (impact0 >= self.threshold).astype(float)

        # Mode colormap: 0=algorithm(blue) → 1=reality(amber)
        MODE_CMAP = mcolors.LinearSegmentedColormap.from_list(
            'mode', [PALETTE['sky'], PALETTE['amber']], N=2
        )

        # LEFT: Mode scatter
        scatter_mode = ax_left.scatter(
            self.positions[:, 0], self.positions[:, 1],
            c=modes0, cmap=MODE_CMAP, s=12, alpha=0.8,
            linewidths=0, vmin=0, vmax=1
        )
        ax_left.set_title('Agent Mode', color=PALETTE['text'], fontsize=10)
        ax_left.set_aspect('equal')
        ax_left.set_xlim(-0.02, 1.02); ax_left.set_ylim(-0.02, 1.02)

        # Mode legend patches
        algo_patch = mpatches.Patch(color=PALETTE['sky'], label='Algorithm Mode')
        real_patch = mpatches.Patch(color=PALETTE['amber'], label='Reality Mode')
        ax_left.legend(handles=[algo_patch, real_patch], fontsize=7,
                       facecolor=PALETTE['surface'], edgecolor=PALETTE['border'],
                       labelcolor=PALETTE['text'])

        # RIGHT: Histogram
        op0 = self.history_opinions[0][:, self.layer_idx]
        bins = np.linspace(0, 1, 31)
        bar_vals, _, bars = ax_right.hist(op0, bins=bins, color=PALETTE['amber'],
                                           alpha=0.7, edgecolor=PALETTE['bg'], lw=0.3)
        # Color bars
        for bar, left in zip(bars, bins[:-1]):
            bar.set_facecolor(OPINION_CMAP(left))
            bar.set_alpha(0.75)

        vline = ax_right.axvline(op0.mean(), color=PALETTE['rose'],
                                 linewidth=1.5, linestyle='--')
        ax_right.set_xlim(0, 1)
        ax_right.set_xlabel('Opinion Value', fontsize=9)
        ax_right.set_ylabel('Count', fontsize=9)
        ax_right.set_title('Opinion Distribution', color=PALETTE['text'], fontsize=10)

        # Stats text
        stats_text = ax_right.text(
            0.98, 0.95, '', transform=ax_right.transAxes,
            ha='right', va='top', color=PALETTE['teal'], fontsize=8,
            fontfamily='monospace'
        )

        # Time text
        time_text = ax_left.text(
            0.02, 0.97, '', transform=ax_left.transAxes,
            color=PALETTE['amber'], fontsize=9, va='top',
            fontfamily='monospace'
        )

        # Mode ratio text
        ratio_text = ax_left.text(
            0.98, 0.97, '', transform=ax_left.transAxes,
            color=PALETTE['teal'], fontsize=8, va='top', ha='right',
            fontfamily='monospace'
        )

        fig.suptitle(self.title, color=PALETTE['text'], fontsize=12,
                     fontweight='bold', y=1.01)

        def init():
            scatter_mode.set_array(self.history_impact[0] >= self.threshold)
            time_text.set_text('')
            stats_text.set_text('')
            ratio_text.set_text('')
            return [scatter_mode, vline, stats_text, time_text, ratio_text]

        def update(frame):
            impact = self.history_impact[frame]
            ops = self.history_opinions[frame][:, self.layer_idx]
            modes = (impact >= self.threshold).astype(float)
            n_active = int(modes.sum())

            scatter_mode.set_array(modes)

            # Redraw histogram
            vals, _ = np.histogram(ops, bins=bins)
            for bar, v in zip(bars, vals):
                bar.set_height(v)

            vline.set_xdata([ops.mean(), ops.mean()])

            t = self.history_times[frame]
            time_text.set_text(f't = {t:.1f}')
            ratio_text.set_text(f'Reality: {n_active}/{N}')
            stats_text.set_text(f'μ={ops.mean():.3f}\nσ={ops.std():.3f}')

            return [scatter_mode, vline, stats_text, time_text, ratio_text] + list(bars)

        interval = max(1, int(1000 / self.fps))
        self._anim = animation.FuncAnimation(
            fig, update, frames=len(self.history_times),
            init_func=init, blit=False, interval=interval
        )
        return self._anim

    def save(self, filepath: str, fps: Optional[int] = None, dpi: int = 120):
        if self._anim is None:
            self.build()
        fps = fps or self.fps
        self._anim.save(filepath, fps=fps, dpi=dpi,
                        savefig_kwargs={'facecolor': PALETTE['bg']})
        print(f"[DualModeAnimator] Saved → {filepath}")
