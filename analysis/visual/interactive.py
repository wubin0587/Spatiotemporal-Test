"""
analysis/visual/interactive.py

Interactive Visualization Module — HTML/JS Dashboard
-----------------------------------------------------
Generates self-contained HTML dashboards using Plotly (via plotly.graph_objects).
No server needed — everything is embedded in a single .html file.

Features:
- Animated spatial opinion map with slider
- Interactive event timeline with zoom/pan
- Hoverable agent details
- Cross-filtered views

Requires: plotly (pip install plotly)
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from .static import PALETTE, SOURCE_COLORS


def _require_plotly():
    if not _HAS_PLOTLY:
        raise ImportError(
            "plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )


# =========================================================================
# Theme configuration for Plotly
# =========================================================================

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor=PALETTE['bg'],
    plot_bgcolor=PALETTE['surface'],
    font=dict(color=PALETTE['text'], family='monospace', size=11),
    xaxis=dict(
        gridcolor=PALETTE['border'], linecolor=PALETTE['border'],
        tickfont=dict(color=PALETTE['text_dim']),
        zerolinecolor=PALETTE['border'],
    ),
    yaxis=dict(
        gridcolor=PALETTE['border'], linecolor=PALETTE['border'],
        tickfont=dict(color=PALETTE['text_dim']),
        zerolinecolor=PALETTE['border'],
    ),
    coloraxis_colorbar=dict(
        outlinecolor=PALETTE['border'],
        tickfont=dict(color=PALETTE['text_dim']),
    ),
    hoverlabel=dict(
        bgcolor=PALETTE['surface'],
        bordercolor=PALETTE['border'],
        font=dict(color=PALETTE['text'], family='monospace'),
    ),
    margin=dict(l=50, r=30, t=60, b=50),
)

OPINION_COLORSCALE = [
    [0.0,  '#2563eb'],
    [0.5,  '#64748b'],
    [1.0,  '#f59e0b'],
]

IMPACT_COLORSCALE = [
    [0.0,  PALETTE['bg']],
    [0.2,  '#1e3a5f'],
    [0.5,  '#0ea5e9'],
    [0.8,  '#f59e0b'],
    [1.0,  '#ef4444'],
]

SOURCE_COLOR_MAP = {
    'exogenous':            PALETTE['amber'],
    'endogenous_threshold': PALETTE['teal'],
    'cascade':              PALETTE['violet'],
    'unknown':              PALETTE['text_dim'],
}


# =========================================================================
# 1. Animated Spatial Opinion Map
# =========================================================================

def interactive_opinion_map(
    positions: np.ndarray,
    history_opinions: List[np.ndarray],
    history_times: List[float],
    layer_idx: int = 0,
    frame_stride: int = 1,
    agent_labels: Optional[List[str]] = None,
    title: str = "Interactive Opinion Map",
) -> 'go.Figure':
    """
    Interactive animated scatter plot of agent opinions in 2D space.

    Args:
        positions: (N, 2) array of agent coordinates
        history_opinions: List of (N, L) opinion arrays
        history_times: Corresponding timestamps
        layer_idx: Opinion dimension to display
        frame_stride: Skip every N frames for performance

    Returns:
        plotly Figure object (call .show() or .write_html(...))
    """
    _require_plotly()

    N = len(positions)
    if agent_labels is None:
        agent_labels = [f'Agent {i}' for i in range(N)]

    frames_idx = list(range(0, len(history_times), frame_stride))

    # Build animation frames
    frames = []
    for fi in frames_idx:
        op = history_opinions[fi][:, layer_idx]
        frame_data = go.Scatter(
            x=positions[:, 0], y=positions[:, 1],
            mode='markers',
            marker=dict(
                color=op,
                colorscale=OPINION_COLORSCALE,
                cmin=0, cmax=1,
                size=7, opacity=0.8,
                line=dict(width=0),
            ),
            text=[f'{lbl}<br>Opinion: {op[i]:.3f}' for i, lbl in enumerate(agent_labels)],
            hoverinfo='text',
        )
        frames.append(go.Frame(
            data=[frame_data],
            name=str(history_times[fi]),
            layout=go.Layout(
                annotations=[dict(
                    x=0.02, y=0.97, xref='paper', yref='paper',
                    text=f't = {history_times[fi]:.1f}  |  σ = {history_opinions[fi][:, layer_idx].std():.3f}',
                    showarrow=False,
                    font=dict(color=PALETTE['amber'], size=12, family='monospace'),
                    bgcolor=PALETTE['surface'], bordercolor=PALETTE['border'],
                    borderwidth=1, borderpad=6,
                )]
            )
        ))

    # Initial frame data
    op0 = history_opinions[0][:, layer_idx]
    initial_trace = go.Scatter(
        x=positions[:, 0], y=positions[:, 1],
        mode='markers',
        marker=dict(
            color=op0,
            colorscale=OPINION_COLORSCALE,
            cmin=0, cmax=1,
            size=7, opacity=0.8,
            showscale=True,
            colorbar=dict(
                title='Opinion',
                titlefont=dict(color=PALETTE['text_dim']),
                tickfont=dict(color=PALETTE['text_dim']),
                outlinecolor=PALETTE['border'],
                x=1.02,
            ),
        ),
        text=[f'{lbl}<br>Opinion: {op0[i]:.3f}' for i, lbl in enumerate(agent_labels)],
        hoverinfo='text',
        name='Agents',
    )

    # Slider steps
    sliders = [dict(
        active=0,
        currentvalue=dict(prefix='t = ', font=dict(color=PALETTE['text'], size=11)),
        pad=dict(b=10, t=10),
        bgcolor=PALETTE['surface'],
        bordercolor=PALETTE['border'],
        font=dict(color=PALETTE['text_dim']),
        steps=[dict(
            args=[[str(history_times[fi])],
                  dict(frame=dict(duration=80, redraw=True), mode='immediate')],
            label=f'{history_times[fi]:.0f}',
            method='animate',
        ) for fi in frames_idx]
    )]

    layout = go.Layout(
        title=dict(text=title, font=dict(color=PALETTE['text'], size=14), x=0.5),
        xaxis=dict(range=[-0.02, 1.02], title='X', **PLOTLY_LAYOUT_BASE['xaxis']),
        yaxis=dict(range=[-0.02, 1.02], title='Y', scaleanchor='x',
                   **PLOTLY_LAYOUT_BASE['yaxis']),
        sliders=sliders,
        updatemenus=[dict(
            type='buttons', showactive=False,
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(frame=dict(duration=60, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')]),
            ],
            x=0, y=0, xanchor='left', yanchor='top',
            pad=dict(r=10, t=10),
            bgcolor=PALETTE['surface'],
            bordercolor=PALETTE['border'],
            font=dict(color=PALETTE['text']),
        )],
        **{k: v for k, v in PLOTLY_LAYOUT_BASE.items()
           if k not in ('xaxis', 'yaxis')},
        height=650,
    )

    fig = go.Figure(data=[initial_trace], layout=layout, frames=frames)
    return fig


# =========================================================================
# 2. Interactive Event Timeline
# =========================================================================

def interactive_event_timeline(
    event_times: np.ndarray,
    event_intensities: np.ndarray,
    event_sources: List[str],
    event_locs: Optional[np.ndarray] = None,
    event_polarities: Optional[np.ndarray] = None,
    title: str = "Event Timeline",
) -> 'go.Figure':
    """
    Interactive timeline with hover details and source-colored markers.

    Args:
        event_times: (M,) event timestamps
        event_intensities: (M,) intensity values
        event_sources: (M,) source label strings
        event_locs: (M, 2) locations (for hover text)
        event_polarities: (M,) polarity values

    Returns:
        plotly Figure
    """
    _require_plotly()

    if event_locs is None:
        event_locs = np.zeros((len(event_times), 2))
    if event_polarities is None:
        event_polarities = np.zeros(len(event_times))

    fig = go.Figure()

    for src, color in SOURCE_COLOR_MAP.items():
        mask = np.array(event_sources) == src
        if not mask.any():
            continue

        t = event_times[mask]
        s = event_intensities[mask]
        locs = event_locs[mask]
        pols = event_polarities[mask]
        norm_s = s / (event_intensities.max() + 1e-9)

        hover_text = [
            f'<b>{src}</b><br>'
            f'Time: {t[k]:.1f}<br>'
            f'Intensity: {s[k]:.2f}<br>'
            f'Location: ({locs[k,0]:.3f}, {locs[k,1]:.3f})<br>'
            f'Polarity: {pols[k]:.3f}'
            for k in range(len(t))
        ]

        # Stem lines
        for k in range(len(t)):
            fig.add_shape(
                type='line',
                x0=t[k], x1=t[k], y0=0, y1=norm_s[k],
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.4,
            )

        # Markers
        fig.add_trace(go.Scatter(
            x=t, y=norm_s,
            mode='markers',
            name=src.replace('_', ' ').title(),
            marker=dict(
                color=color,
                size=8 + norm_s * 16,
                opacity=0.85,
                line=dict(width=0),
            ),
            text=hover_text,
            hoverinfo='text',
        ))

    # Zero baseline
    fig.add_hline(y=0, line=dict(color=PALETTE['border'], width=1))

    layout_args = dict(
        title=dict(text=title, font=dict(color=PALETTE['text'], size=13), x=0.5),
        **PLOTLY_LAYOUT_BASE,
        xaxis=dict(title='Time', **PLOTLY_LAYOUT_BASE['xaxis']),
        yaxis=dict(title='Normalized Intensity', range=[-0.05, 1.2],
                   **PLOTLY_LAYOUT_BASE['yaxis']),
        legend=dict(
            bgcolor=PALETTE['surface'], bordercolor=PALETTE['border'],
            font=dict(color=PALETTE['text']),
        ),
        height=400,
    )
    fig.update_layout(**layout_args)
    return fig


# =========================================================================
# 3. Full Interactive Dashboard
# =========================================================================

def interactive_dashboard(
    positions: np.ndarray,
    final_opinions: np.ndarray,
    final_impact: np.ndarray,
    history_opinions: Optional[List[np.ndarray]] = None,
    history_times: Optional[List[float]] = None,
    history_impact: Optional[List[np.ndarray]] = None,
    event_times: Optional[np.ndarray] = None,
    event_intensities: Optional[np.ndarray] = None,
    event_sources: Optional[List[str]] = None,
    event_locs: Optional[np.ndarray] = None,
    layer_idx: int = 0,
    title: str = "Simulation Dashboard",
    output_html: Optional[str] = None,
) -> 'go.Figure':
    """
    Full interactive HTML dashboard with:
    - Top-left: Spatial opinion scatter (final)
    - Top-right: Opinion distribution histogram
    - Middle-left: Impact scatter (final)
    - Middle-right: Polarization over time
    - Bottom: Event timeline

    Args:
        output_html: If provided, saves the dashboard to this path.

    Returns:
        plotly Figure
    """
    _require_plotly()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Agent Opinions (Final)', 'Opinion Distribution',
            'Impact Field (Final)', 'Polarization over Time',
            'Event Timeline', '',
        ],
        specs=[
            [{'type': 'scatter'}, {'type': 'histogram'}],
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'colspan': 2, 'type': 'scatter'}, None],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        row_heights=[0.35, 0.30, 0.25],
    )

    op = final_opinions[:, layer_idx] if final_opinions.ndim == 2 else final_opinions

    # --- Row 1 Left: Opinion scatter ---
    fig.add_trace(go.Scatter(
        x=positions[:, 0], y=positions[:, 1],
        mode='markers',
        marker=dict(color=op, colorscale=OPINION_COLORSCALE,
                    cmin=0, cmax=1, size=5, opacity=0.75,
                    showscale=True,
                    colorbar=dict(
                        x=0.44, title='Opinion',
                        titlefont=dict(color=PALETTE['text_dim']),
                        tickfont=dict(color=PALETTE['text_dim']),
                        outlinecolor=PALETTE['border'], len=0.32, y=0.82,
                    )),
        text=[f'Agent {i}<br>Opinion: {op[i]:.3f}' for i in range(len(op))],
        hoverinfo='text', name='Agents',
    ), row=1, col=1)

    # --- Row 1 Right: Histogram ---
    fig.add_trace(go.Histogram(
        x=op, nbinsx=30, name='Distribution',
        marker=dict(color=PALETTE['amber'], opacity=0.7,
                    line=dict(color=PALETTE['bg'], width=0.3)),
        xbins=dict(start=0, end=1, size=1/30),
    ), row=1, col=2)
    fig.add_vline(x=op.mean(), line_color=PALETTE['rose'],
                  line_dash='dash', line_width=2, row=1, col=2)

    # --- Row 2 Left: Impact scatter ---
    norm_impact = final_impact / (final_impact.max() + 1e-9)
    fig.add_trace(go.Scatter(
        x=positions[:, 0], y=positions[:, 1],
        mode='markers',
        marker=dict(
            color=final_impact,
            colorscale=IMPACT_COLORSCALE,
            cmin=0, cmax=final_impact.max(),
            size=5 + norm_impact * 12,
            opacity=0.8,
            showscale=True,
            colorbar=dict(
                x=0.44, title='I(t)',
                titlefont=dict(color=PALETTE['text_dim']),
                tickfont=dict(color=PALETTE['text_dim']),
                outlinecolor=PALETTE['border'], len=0.26, y=0.48,
            )
        ),
        text=[f'Agent {i}<br>Impact: {final_impact[i]:.3f}' for i in range(len(final_impact))],
        hoverinfo='text', name='Impact',
    ), row=2, col=1)

    # --- Row 2 Right: Polarization ---
    if history_opinions is not None and history_times is not None:
        pol = [o.std() for o in history_opinions]
        fig.add_trace(go.Scatter(
            x=history_times, y=pol,
            mode='lines', name='σ(t)',
            line=dict(color=PALETTE['teal'], width=2),
            fill='tozeroy', fillcolor=f"rgba(20,184,166,0.1)",
        ), row=2, col=2)
        if history_impact is not None:
            mean_impact_t = [imp.mean() for imp in history_impact]
            fig.add_trace(go.Scatter(
                x=history_times, y=mean_impact_t,
                mode='lines', name='Mean Impact',
                line=dict(color=PALETTE['amber'], width=1.5, dash='dot'),
                yaxis='y5',
            ), row=2, col=2)

    # --- Row 3: Event timeline ---
    if (event_times is not None and event_intensities is not None
            and event_sources is not None and len(event_times) > 0):
        max_int = event_intensities.max() if event_intensities.max() > 0 else 1
        norm_e = event_intensities / max_int
        e_locs = event_locs if event_locs is not None else np.zeros((len(event_times), 2))

        for src, color in SOURCE_COLOR_MAP.items():
            mask = np.array(event_sources) == src
            if not mask.any():
                continue
            t = event_times[mask]
            s = norm_e[mask]
            locs = e_locs[mask]

            hover = [
                f'<b>{src}</b><br>t={t[k]:.1f}<br>I={event_intensities[mask][k]:.2f}'
                f'<br>loc=({locs[k,0]:.2f},{locs[k,1]:.2f})'
                for k in range(len(t))
            ]

            fig.add_trace(go.Scatter(
                x=t, y=s, mode='markers',
                name=src.replace('_', ' ').title(),
                marker=dict(color=color, size=7 + s * 12, opacity=0.85),
                text=hover, hoverinfo='text',
            ), row=3, col=1)

    # Global layout
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE['text'], size=15), x=0.5),
        paper_bgcolor=PALETTE['bg'],
        plot_bgcolor=PALETTE['surface'],
        font=dict(color=PALETTE['text'], family='monospace', size=10),
        legend=dict(bgcolor=PALETTE['surface'], bordercolor=PALETTE['border'],
                    font=dict(color=PALETTE['text'])),
        hoverlabel=dict(bgcolor=PALETTE['surface'], bordercolor=PALETTE['border'],
                        font=dict(color=PALETTE['text'], family='monospace')),
        height=1000,
    )

    # Style all axes
    for key in fig.layout:
        if key.startswith('xaxis') or key.startswith('yaxis'):
            fig.layout[key].update(
                gridcolor=PALETTE['border'],
                linecolor=PALETTE['border'],
                tickfont=dict(color=PALETTE['text_dim']),
                zerolinecolor=PALETTE['border'],
            )

    for annotation in fig.layout.annotations:
        annotation.font.color = PALETTE['text_dim']
        annotation.font.size = 11

    if output_html:
        fig.write_html(output_html, include_plotlyjs='cdn', full_html=True)
        print(f"[Dashboard] Saved → {output_html}")

    return fig


# =========================================================================
# 4. Opinion Phase-Space Plot (Interactive)
# =========================================================================

def interactive_phase_space(
    opinions: np.ndarray,
    impact: np.ndarray,
    layer_x: int = 0,
    layer_y: int = 1,
    title: str = "Opinion Phase Space",
) -> 'go.Figure':
    """
    Scatter of layer_x vs layer_y opinions, colored by impact.
    Reveals cross-layer correlations induced by events.
    """
    _require_plotly()

    if opinions.ndim < 2 or opinions.shape[1] < 2:
        raise ValueError("opinions must be (N, L) with L >= 2")

    ox = opinions[:, layer_x]
    oy = opinions[:, layer_y]

    fig = go.Figure(go.Scatter(
        x=ox, y=oy, mode='markers',
        marker=dict(
            color=impact, colorscale=IMPACT_COLORSCALE,
            cmin=0, cmax=impact.max(),
            size=6, opacity=0.7,
            showscale=True,
            colorbar=dict(
                title='Impact I(t)',
                titlefont=dict(color=PALETTE['text_dim']),
                tickfont=dict(color=PALETTE['text_dim']),
                outlinecolor=PALETTE['border'],
            )
        ),
        text=[f'Opinion[{layer_x}]={ox[i]:.3f}<br>Opinion[{layer_y}]={oy[i]:.3f}<br>Impact={impact[i]:.3f}'
              for i in range(len(ox))],
        hoverinfo='text', name='Agents',
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE['text'], size=13), x=0.5),
        xaxis=dict(title=f'Opinion Layer {layer_x}', range=[0, 1],
                   **PLOTLY_LAYOUT_BASE['xaxis']),
        yaxis=dict(title=f'Opinion Layer {layer_y}', range=[0, 1],
                   scaleanchor='x', **PLOTLY_LAYOUT_BASE['yaxis']),
        **{k: v for k, v in PLOTLY_LAYOUT_BASE.items()
           if k not in ('xaxis', 'yaxis')},
        height=600,
    )
    return fig
