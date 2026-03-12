"""
core/runner.py

SimulationRunner — wraps SimulationFacade and provides a Gradio-compatible
streaming generator interface with stop / pause control.

Usage pattern
-------------
Each Gradio session should hold one SimulationRunner instance in gr.State.
Global sharing is unsafe for multi-user deployments.

    runner = SimulationRunner()

    # In a gr.Button.click() handler bound as a generator:
    for update_tuple in runner.run_stream(ui_values, refresh_every=10):
        yield update_tuple

Stop / pause
------------
  runner.stop()   — sets stop flag; generator exits after next step
  runner.pause()  — generator idles (sleep loop) until resume()
  runner.resume() — clears pause flag
  runner.reset()  — clears history buffers, does NOT destroy the engine

Yield tuple order (must match monitor_tab.py get_output_list())
--------------------------------------------------------------
  0   status_md          str     step label markdown
  1   metric_sigma       float   opinion std
  2   metric_mean        float   mean opinion
  3   metric_impact      float   mean impact
  4   metric_events      int     cumulative event count
  5   metric_consensus   float   fraction of agents within 0.1 of mean
  6   plot_timeseries    Figure  σ + impact over time
  7   plot_spatial       Figure  agent scatter
  8   plot_histogram     Figure  opinion distribution
  9   plot_events        Figure  event timeline
  10  pause_btn_update   gr.update
  11  stop_btn_update    gr.update
"""

from __future__ import annotations

import time
import threading
from typing import Any, Generator, Optional

import numpy as np

# Lazy imports — only required when simulation actually runs
# (avoids import errors when this module is loaded in isolation)
_SimulationFacade = None
_build_config     = None
_renderer         = None


def _ensure_imports() -> None:
    global _SimulationFacade, _build_config, _renderer
    if _SimulationFacade is None:
        from models.engine.facade import SimulationFacade as _SF
        _SimulationFacade = _SF
    if _build_config is None:
        from core.config_bridge import build_config_from_ui as _bc
        _build_config = _bc
    if _renderer is None:
        import core.renderer as _r
        _renderer = _r


# ─────────────────────────────────────────────────────────────────────────────
# SimulationRunner
# ─────────────────────────────────────────────────────────────────────────────

class SimulationRunner:
    """
    Encapsulates one simulation lifecycle for a single Gradio session.

    The instance should be stored in gr.State so it survives across
    Gradio event callbacks.
    """

    def __init__(self) -> None:
        self._stop_flag:  bool = False
        self._pause_flag: bool = False

        self.sim:    Optional[Any] = None   # SimulationFacade
        self.engine: Optional[Any] = None   # StepExecutor (sim._engine)

        # Lightweight scalar history (no matrix data)
        self._h_time:   list[float] = []
        self._h_sigma:  list[float] = []
        self._h_impact: list[float] = []
        self._h_events: list[int]   = []
        self._last_figures: list[Any] = []

        # Thread-safe access to pause/stop flags
        self._lock = threading.Lock()

    # ─── Control interface ────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the running generator to exit after the current step."""
        with self._lock:
            self._stop_flag = True

    def pause(self) -> None:
        """Signal the running generator to idle until resume() is called."""
        with self._lock:
            self._pause_flag = True

    def resume(self) -> None:
        """Clear the pause flag so the generator continues."""
        with self._lock:
            self._pause_flag = False

    def _close_figures(self) -> None:
        """Close cached matplotlib figures to avoid figure leaks."""
        if not self._last_figures:
            return
        try:
            import matplotlib.pyplot as plt
            for fig in self._last_figures:
                if fig is not None:
                    plt.close(fig)
        except Exception:
            pass
        finally:
            self._last_figures.clear()

    def reset(self) -> None:
        """
        Clear history buffers and release the current simulation instance.
        Does not affect stop/pause flags.
        """
        self._close_figures()
        self._h_time.clear()
        self._h_sigma.clear()
        self._h_impact.clear()
        self._h_events.clear()
        self.sim    = None
        self.engine = None

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._pause_flag

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            return self._stop_flag

    # ─── Streaming generator ─────────────────────────────────────────────────

    def run_stream(
        self,
        ui_values:     dict[str, Any],
        refresh_every: int = 10,
    ) -> Generator[tuple, None, None]:
        """
        Main streaming generator.  Bind to a gr.Button.click() handler
        with outputs matching the yield tuple order documented at the
        top of this module.

        Parameters
        ----------
        ui_values     : flat dict of current UI component values
        refresh_every : number of steps between each chart refresh / yield
        """
        _ensure_imports()

        # Reset stop/pause flags at the start of each new run
        with self._lock:
            self._stop_flag  = False
            self._pause_flag = False

        # Build config and initialise simulation
        try:
            config = _build_config(ui_values)
            self.sim = _SimulationFacade.from_config_dict(config)
            self.sim.initialize()
            self.engine = self.sim._engine
        except Exception as exc:
            yield _error_tuple(f"Initialisation failed: {exc}")
            return

        total = int(ui_values.get("total_steps", 500))
        refresh_every = max(1, int(refresh_every))

        self._h_time.clear()
        self._h_sigma.clear()
        self._h_impact.clear()
        self._h_events.clear()

        # ── Step loop ────────────────────────────────────────────────────────
        for step_i in range(total):

            # Check stop
            if self.is_stopped:
                break

            # Check pause (busy-wait with short sleep)
            while self.is_paused and not self.is_stopped:
                time.sleep(0.05)

            if self.is_stopped:
                break

            # Execute one step
            try:
                stats = self.sim.step()
            except Exception as exc:
                yield _error_tuple(f"Step {step_i + 1} failed: {exc}")
                return

            # Accumulate lightweight history
            self._h_time.append(float(stats.get("time", step_i)))
            self._h_sigma.append(float(stats.get("opinion_std",
                                  np.std(self.engine.opinion_matrix) if self.engine else 0.0)))
            self._h_impact.append(float(stats.get("mean_impact", 0.0)))
            self._h_events.append(int(stats.get("num_events", 0)))

            # Yield update on refresh cadence or final step
            is_last = (step_i == total - 1) or self.is_stopped
            if (step_i + 1) % refresh_every == 0 or is_last:
                yield self._build_update_tuple(
                    step=step_i + 1,
                    total=total,
                    stats=stats,
                    active=not is_last,
                )

        # ── Final yield: unlock buttons ───────────────────────────────────────
        try:
            import gradio as gr
            yield (
                _step_label(total, total, stats if 'stats' in dir() else {}),
                None, None, None, None, None,
                None, None, None, None,
                gr.update(value="⏸  Pause", interactive=False),
                gr.update(interactive=False),
            )
        except Exception:
            pass

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _build_update_tuple(
        self,
        step:   int,
        total:  int,
        stats:  dict[str, Any],
        active: bool,
    ) -> tuple:
        """Render figures and assemble the yield tuple."""
        import gradio as gr

        # Metrics
        if self.engine is not None:
            ops   = np.asarray(self.engine.opinion_matrix)
            sigma = float(np.std(ops))
            mean  = float(np.mean(ops))
            imp   = float(np.mean(self.engine.impact_vector))
            # Consensus: fraction within 0.1 of the per-layer mean
            per_mean = np.mean(ops, axis=0)
            dist     = np.mean(np.abs(ops - per_mean), axis=1)
            consensus = float(np.mean(dist < 0.1))
        else:
            sigma = mean = imp = consensus = 0.0

        events = self._h_events[-1] if self._h_events else 0
        time_v = self._h_time[-1]   if self._h_time   else 0.0

        status = (
            f"**Step** {step} / {total}"
            f"  ·  **Time** {time_v:.2f}"
            f"  ·  **Events** {events}"
        )

        # Figures
        self._close_figures()
        fig_ts   = _renderer.render_timeseries(
            self._h_time, self._h_sigma, self._h_impact, self._h_events)
        fig_sp   = _renderer.render_spatial(self.engine)
        fig_hist = _renderer.render_histogram(self.engine)
        fig_ev   = _renderer.render_events(self._h_time, self._h_events)

        self._last_figures = [fig_ts, fig_sp, fig_hist, fig_ev]

        return (
            status,          # 0  status_md
            sigma,           # 1  metric_sigma
            mean,            # 2  metric_mean
            imp,             # 3  metric_impact
            events,          # 4  metric_events
            consensus,       # 5  metric_consensus
            fig_ts,          # 6  plot_timeseries
            fig_sp,          # 7  plot_spatial
            fig_hist,        # 8  plot_histogram
            fig_ev,          # 9  plot_events
            gr.update(value="⏸  Pause", interactive=active),   # 10 pause_btn
            gr.update(interactive=active),                      # 11 stop_btn
        )

    def get_dashboard_figure(self, layer_idx: int = 0):
        """
        Render the high-quality composite dashboard figure.
        Call after run completes for the Analysis tab.
        """
        _ensure_imports()
        if self.engine is None:
            return None
        self._close_figures()
        return _renderer.render_dashboard(
            engine   = self.engine,
            h_time   = self._h_time,
            h_sigma  = self._h_sigma,
            h_impact = self._h_impact,
            h_events = self._h_events,
            layer_idx = layer_idx,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _step_label(step: int, total: int, stats: dict) -> str:
    time_v  = float(stats.get("time", 0.0))
    events  = int(stats.get("num_events", 0))
    return (
        f"**Step** {step} / {total}"
        f"  ·  **Time** {time_v:.2f}"
        f"  ·  **Events** {events}"
    )


def _error_tuple(message: str) -> tuple:
    """Yield an error status with empty figure slots."""
    try:
        import gradio as gr
        return (
            f"❌ **Error:** {message}",
            None, None, None, None, None,
            None, None, None, None,
            gr.update(value="⏸  Pause", interactive=False),
            gr.update(interactive=False),
        )
    except Exception:
        return (f"Error: {message}",) + (None,) * 11
