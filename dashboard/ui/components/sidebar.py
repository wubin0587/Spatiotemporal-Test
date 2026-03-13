"""
ui/components/sidebar.py

Persistent left-side navigation sidebar for the Opinion Dynamics
Simulation Dashboard.

Renders 7 navigation items with per-item status dots, a configuration
progress bar, and a version footer.  The sidebar is built once inside
the gr.Blocks context and remains visible across all page switches.

Public API
----------
SidebarComponents
    Dataclass holding all gr.Components created by build_sidebar().

build_sidebar(lang) -> SidebarComponents
    Render the sidebar in the current Gradio context.
    Must be called inside an active gr.Blocks() + gr.Column() context.

update_status(items, lang) -> tuple
    Given a list[CheckItem] from validator.validate_all(), return
    the (status_html, progress_md) update tuple for the two dynamic
    sidebar outputs.

    Use as the return value of a gr.on(...) handler:
        outputs=[sidebar.status_html, sidebar.progress_md]

bind_nav_events(sidebar, page_groups, switch_fn) -> None
    Wire all 7 nav buttons to the page-switch function.
    Call once after the full layout is defined.

NAV_ITEMS
    Ordered list of (page_key, icon, label_en, label_zh) for all pages.
    Shared with app.py to ensure key consistency.

Design
------
- Status dots: green (ok), amber (warn), red (error), slate (idle/locked).
- Progress bar: teal fill, 4px height, updates via gr.HTML.
- "实验结果" is initially disabled (locked) until a simulation completes;
  the lock is released by passing unlock_results=True to update_status().
- The sidebar does NOT hold simulation state — it is a pure display layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gradio as gr

# Lazy import to avoid circular dependency
_CheckItem = None


def _ensure_check_item():
    global _CheckItem
    if _CheckItem is None:
        from core.validator import CheckItem as _CI
        _CheckItem = _CI


# ─────────────────────────────────────────────────────────────────────────────
# Navigation item registry
# ─────────────────────────────────────────────────────────────────────────────

# (page_key, icon, label_en, label_zh, has_status_dot, initially_locked)
NAV_ITEMS: list[tuple[str, str, str, str, bool, bool]] = [
    ("home",              "⌂",  "Home",              "首页",       False, False),
    ("dashboard_settings","◈",  "Dashboard Settings","仪表盘设置",  True,  False),
    ("model_config",      "⚙",  "Model Config",      "模型配置",   True,  False),
    ("analysis_config",   "◑",  "Analysis Config",   "分析配置",   True,  False),
    ("intervention",      "⚡", "Interventions",     "干预配置",   True,  False),
    ("experiment",        "▶",  "Run Experiment",    "实验运行",   False, False),
    ("results",           "◎",  "Results",           "实验结果",   False, True),
]

# page_key → index for fast lookup
_PAGE_INDEX: dict[str, int] = {item[0]: i for i, item in enumerate(NAV_ITEMS)}

# Config-page keys that carry status dots
_CONFIG_PAGES: list[str] = [
    item[0] for item in NAV_ITEMS if item[4]  # has_status_dot
]

# Separator: drawn between index 4 (intervention) and index 5 (experiment)
_SEPARATOR_AFTER_INDEX = 4


# ─────────────────────────────────────────────────────────────────────────────
# Status dot color constants
# ─────────────────────────────────────────────────────────────────────────────

_DOT_OK       = "#1D9E75"   # green
_DOT_WARN     = "#BA7517"   # amber
_DOT_ERROR    = "#dc2626"   # red
_DOT_IDLE     = "#cbd5e1"   # slate-300  (not yet configured)
_DOT_LOCKED   = "#e2e8f0"   # slate-200  (page not yet unlocked)


# ─────────────────────────────────────────────────────────────────────────────
# SidebarComponents dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SidebarComponents:
    """
    Holds all gr.Components created by build_sidebar().

    Attributes
    ----------
    nav_buttons : list[gr.Button]
        7 navigation buttons in NAV_ITEMS order.
        Use nav_buttons[_PAGE_INDEX[page_key]] to get a specific button.
    status_html : gr.HTML
        Dynamic status-dot area.  Updated by update_status().
    progress_md : gr.Markdown
        Progress bar + label below the navigation items.
        Updated by update_status().
    active_state : gr.State
        Currently active page key.  Updated on every nav click.
    """
    nav_buttons:  list[gr.Button]
    status_html:  gr.HTML
    progress_md:  gr.Markdown
    active_state: gr.State

    # ── Convenience ──────────────────────────────────────────────────────────

    def button_for(self, page_key: str) -> gr.Button:
        """Return the nav button for the given page key."""
        idx = _PAGE_INDEX.get(page_key)
        if idx is None:
            raise KeyError(f"Unknown page key: {page_key!r}")
        return self.nav_buttons[idx]

    def all_outputs(self) -> list:
        """
        All dynamic output components.
        Use as outputs= in a handler that updates sidebar state.
        """
        return [self.status_html, self.progress_md, self.active_state]


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dot_html(color: str, animate: bool = False) -> str:
    """Render a single status dot as an inline HTML span."""
    pulse = (
        " animation:pulse-warn 2s ease-in-out infinite;"
        if animate else ""
    )
    return (
        f'<span style="'
        f'display:inline-block;'
        f'width:6px;height:6px;border-radius:50%;'
        f'background:{color};flex-shrink:0;'
        f'{pulse}'
        f'"></span>'
    )


def _build_status_html(
    dot_colors: dict[str, str],
    active_page: str,
    lang: str,
) -> str:
    """
    Build the full HTML string for all 7 nav items with their dots.

    This replaces the entire sidebar nav area on every status update,
    ensuring active-state highlighting and dot colors are always in sync.

    Parameters
    ----------
    dot_colors : dict[str, str]
        {page_key: hex_color}  for items with status dots.
    active_page : str
        Currently active page key (receives active styling).
    lang : str
        "en" | "zh"
    """
    lang_idx = 2 if lang == "en" else 3  # index into NAV_ITEMS tuple

    items_html = ""
    for i, (page_key, icon, label_en, label_zh, has_dot, locked) in enumerate(NAV_ITEMS):

        label = label_en if lang == "en" else label_zh
        is_active = (page_key == active_page)

        # ── Separator before experiment/results group ─────────────────────
        if i == _SEPARATOR_AFTER_INDEX + 1:
            items_html += (
                '<div style="height:1px;background:#f1f5f9;'
                'margin:6px 14px;"></div>'
            )

        # ── Active / hover styles ─────────────────────────────────────────
        if is_active:
            bg    = "#f0fdfa"
            color = "#0f172a"
            weight = "500"
            border_left = "border-left:2px solid #0d9488;"
            padding_left = "12px"   # compensate for border
        elif locked:
            bg    = "transparent"
            color = "#94a3b8"
            weight = "400"
            border_left = ""
            padding_left = "14px"
        else:
            bg    = "transparent"
            color = "#475569"
            weight = "400"
            border_left = ""
            padding_left = "14px"

        cursor = "not-allowed" if locked else "pointer"

        # ── Status dot ───────────────────────────────────────────────────
        if has_dot:
            dot_color  = dot_colors.get(page_key, _DOT_IDLE)
            animate    = (dot_color == _DOT_WARN)
            dot        = _dot_html(dot_color, animate=animate)
        elif locked:
            dot = _dot_html(_DOT_LOCKED)
        else:
            dot = ""

        dot_spacer = (
            '<span style="width:6px;height:6px;flex-shrink:0;'
            'display:inline-block;"></span>'
            if (not has_dot and not locked) else ""
        )

        items_html += (
            f'<div style="'
            f'display:flex;align-items:center;gap:8px;'
            f'padding:8px {padding_left};'
            f'border-radius:6px;'
            f'background:{bg};'
            f'{border_left}'
            f'cursor:{cursor};'
            f'font-size:13px;font-weight:{weight};color:{color};'
            f'transition:background 0.12s ease;'
            f'">'
            f'<span style="font-size:14px;width:16px;text-align:center;'
            f'flex-shrink:0">{icon}</span>'
            f'{dot}{dot_spacer}'
            f'<span>{label}</span>'
            f'</div>'
        )

    return f'<div style="display:flex;flex-direction:column;gap:2px;">{items_html}</div>'


def _build_progress_html(
    n_ok:    int,
    n_warn:  int,
    n_error: int,
    total:   int,
    lang:    str,
) -> str:
    """
    Render the progress bar + label as an HTML string for gr.HTML.

    Shown in the bottom area of the sidebar, above the version footer.
    """
    pct = int(100 * n_ok / total) if total else 0

    if lang == "zh":
        lbl_title = "配置完成度"
        if n_error:
            lbl_detail = f"{n_error} 项错误 · {n_warn} 项警告"
            bar_color  = "#dc2626"
        elif n_warn:
            lbl_detail = f"{n_ok}/{total} 项完成 · {n_warn} 项警告"
            bar_color  = "#d97706"
        else:
            lbl_detail = f"{n_ok}/{total} 项已完成" if n_ok < total else "全部配置完成"
            bar_color  = "#0d9488"
    else:
        lbl_title = "Config Progress"
        if n_error:
            lbl_detail = f"{n_error} error(s) · {n_warn} warning(s)"
            bar_color  = "#dc2626"
        elif n_warn:
            lbl_detail = f"{n_ok}/{total} done · {n_warn} warning(s)"
            bar_color  = "#d97706"
        else:
            lbl_detail = f"{n_ok}/{total} complete" if n_ok < total else "All configured"
            bar_color  = "#0d9488"

    return (
        f'<div style="padding:10px 14px;">'
        f'<div style="font-size:11px;font-weight:500;color:#64748b;'
        f'margin-bottom:5px;">{lbl_title}</div>'
        f'<div style="height:4px;background:#e2e8f0;border-radius:2px;'
        f'margin-bottom:5px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{bar_color};'
        f'border-radius:2px;transition:width 0.3s ease;"></div>'
        f'</div>'
        f'<div style="font-size:11px;color:{bar_color};">{lbl_detail}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sidebar(lang: str = "en") -> SidebarComponents:
    """
    Render the persistent sidebar inside the current Gradio context.

    Must be called inside an active gr.Blocks() context, inside a
    gr.Column(width=200, elem_id="sidebar") block.

    Parameters
    ----------
    lang : {"en", "zh"}

    Returns
    -------
    SidebarComponents
    """
    lang_idx = 2 if lang == "en" else 3

    # ── App branding ──────────────────────────────────────────────────────
    with gr.Row(elem_classes="sidebar-brand"):
        gr.HTML(
            '<div style="padding:12px 14px 10px;">'
            '<div style="font-size:13px;font-weight:500;color:#0f172a;">'
            '🔬 Opinion Dynamics</div>'
            '<div style="font-size:11px;color:#64748b;margin-top:2px;">'
            'Simulation Platform</div>'
            '</div>'
        )

    # ── Navigation buttons (invisible, overlaid on HTML) ──────────────────
    # We render the visual nav in gr.HTML (for full styling control) and
    # maintain 7 invisible gr.Buttons for Gradio event binding.
    # The buttons are positioned absolutely over the corresponding rows
    # via CSS class "sidebar-btn-overlay".
    nav_buttons: list[gr.Button] = []
    for page_key, icon, label_en, label_zh, has_dot, locked in NAV_ITEMS:
        label = label_en if lang == "en" else label_zh
        btn = gr.Button(
            label,
            elem_id=f"nav-btn-{page_key}",
            elem_classes="sidebar-nav-btn",
            size="sm",
            interactive=not locked,
            visible=True,
        )
        nav_buttons.append(btn)

    # ── Visual nav HTML (replaces button visuals) ─────────────────────────
    initial_dots = {page_key: _DOT_IDLE for page_key in _CONFIG_PAGES}
    initial_html = _build_status_html(
        dot_colors  = initial_dots,
        active_page = "home",
        lang        = lang,
    )
    status_html = gr.HTML(
        value       = initial_html,
        elem_id     = "sidebar-nav-html",
    )

    # ── Separator ─────────────────────────────────────────────────────────
    gr.HTML('<div style="height:1px;background:#e2e8f0;margin:4px 0;"></div>')

    # ── Progress area ─────────────────────────────────────────────────────
    n_config = len(_CONFIG_PAGES)
    initial_progress = _build_progress_html(
        n_ok=0, n_warn=0, n_error=0, total=n_config, lang=lang
    )
    progress_md = gr.Markdown(
        value   = initial_progress,
        elem_id = "sidebar-progress",
    )

    # ── Version footer ────────────────────────────────────────────────────
    gr.HTML(
        '<div style="padding:8px 14px;font-size:11px;color:#94a3b8;">'
        'v2.0.0-beta</div>'
    )

    # ── Active page state ─────────────────────────────────────────────────
    active_state = gr.State(value="home")

    return SidebarComponents(
        nav_buttons  = nav_buttons,
        status_html  = status_html,
        progress_md  = progress_md,
        active_state = active_state,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic update
# ─────────────────────────────────────────────────────────────────────────────

def update_status(
    items:          list,      # list[CheckItem] — typed loosely to avoid import
    lang:           str  = "en",
    active_page:    str  = "home",
    unlock_results: bool = False,
) -> tuple:
    """
    Compute updated sidebar HTML and progress markdown.

    Parameters
    ----------
    items : list[CheckItem]
        Output of validator.validate_all().
    lang : str
        Current UI language.
    active_page : str
        Currently visible page key.
    unlock_results : bool
        If True, the "实验结果" nav item is rendered as enabled.

    Returns
    -------
    tuple[str, str]
        (status_html_value, progress_md_value)
        Assign to sidebar.status_html and sidebar.progress_md.
    """
    from core.validator import (
        STATUS_OK, STATUS_WARN, STATUS_ERROR, summarize
    )

    # ── Map CheckItem statuses to dot colors ──────────────────────────────
    # Build a page_key → color dict for config pages.
    # Validator group keys don't map 1:1 to page keys because model_config
    # covers three groups (agent_simulation, dynamics, network_spatial,
    # events).  We take the worst status across all groups for a page.

    _GROUP_TO_PAGE: dict[str, str] = {
        "agent_simulation": "model_config",
        "dynamics":          "model_config",
        "network_spatial":   "model_config",
        "events":            "model_config",
        "intervention":      "intervention",
        "analysis_config":   "analysis_config",
    }

    # dashboard_settings has no validator (all optional); default to idle
    page_worst: dict[str, str] = {pk: _DOT_IDLE for pk in _CONFIG_PAGES}

    _STATUS_RANK = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_ERROR: 2}
    _STATUS_DOT  = {STATUS_OK: _DOT_OK, STATUS_WARN: _DOT_WARN, STATUS_ERROR: _DOT_ERROR}

    for item in items:
        page = _GROUP_TO_PAGE.get(item.group)
        if page is None:
            continue
        current_dot   = page_worst.get(page, _DOT_IDLE)
        current_rank  = 0 if current_dot == _DOT_IDLE else (
            1 if current_dot == _DOT_WARN else (
            2 if current_dot == _DOT_ERROR else 0))
        new_rank = _STATUS_RANK.get(item.status, 0)
        if new_rank >= current_rank:
            page_worst[page] = _STATUS_DOT.get(item.status, _DOT_IDLE)

    # dashboard_settings: no validators, mark OK if it was visited
    # (We don't track "visited" state here; keep as IDLE to avoid confusion)

    # ── Unlock results ────────────────────────────────────────────────────
    # We handle this in the HTML builder by passing unlock_results; the
    # "results" key is not in _CONFIG_PAGES, so dots don't apply —
    # but we need to reflect the locked/unlocked state in the nav HTML.
    # We pass it via a patched copy of NAV_ITEMS (structural, no mutation).
    nav_items_effective = list(NAV_ITEMS)
    if unlock_results:
        # Replace the results entry's initially_locked flag
        nav_items_effective[_PAGE_INDEX["results"]] = (
            "results", "◎", "Results", "实验结果", False, False
        )

    # ── Build HTML ────────────────────────────────────────────────────────
    # Temporarily patch the global for _build_status_html to use
    import core.sidebar as _self
    _orig = _self.NAV_ITEMS
    _self.NAV_ITEMS = nav_items_effective

    html_val = _build_status_html(
        dot_colors  = page_worst,
        active_page = active_page,
        lang        = lang,
    )

    _self.NAV_ITEMS = _orig  # restore

    # ── Build progress ────────────────────────────────────────────────────
    summary     = summarize(items)
    progress_val = _build_progress_html(
        n_ok    = summary["n_ok"],
        n_warn  = summary["n_warn"],
        n_error = summary["n_error"],
        total   = len(items),
        lang    = lang,
    )

    return html_val, progress_val


# ─────────────────────────────────────────────────────────────────────────────
# Event wiring
# ─────────────────────────────────────────────────────────────────────────────

def bind_nav_events(
    sidebar:      SidebarComponents,
    page_groups:  list,          # list[gr.Group], len == 7, in NAV_ITEMS order
    switch_fn,                   # callable(page_key: str) -> list[gr.update]
    all_param_inputs: list,      # list[gr.Component] for status re-computation
    param_keys:       list[str], # matching all_param_inputs
    lang_state:       gr.State,
    runner_state:     gr.State,
) -> None:
    """
    Wire all sidebar navigation buttons to the page-switch function
    and set up status-update triggers.

    Call once after the full gr.Blocks layout is defined.

    Parameters
    ----------
    sidebar : SidebarComponents
    page_groups : list[gr.Group]
        7 page container groups in NAV_ITEMS order.
    switch_fn : callable
        (page_key: str) -> list[gr.update(visible=...)]
        Returns one gr.update per page group.
    all_param_inputs : list[gr.Component]
        All parameter inputs (for status re-computation on change).
    param_keys : list[str]
        Keys matching all_param_inputs.
    lang_state : gr.State
        Current language ("en" | "zh").
    runner_state : gr.State
        SimulationRunner instance (for detecting completed runs).
    """
    from core.validator import validate_all

    # ── Nav button clicks ─────────────────────────────────────────────────
    for btn, (page_key, *_rest) in zip(sidebar.nav_buttons, NAV_ITEMS):

        def _make_click_fn(pk: str):
            def _click(lang: str, *param_vals):
                # Switch page
                page_updates = switch_fn(pk)
                # Recompute status
                ui_v   = dict(zip(param_keys, param_vals))
                items  = validate_all(ui_v)
                s_html, prog = update_status(
                    items       = items,
                    lang        = lang,
                    active_page = pk,
                )
                return page_updates + [s_html, prog, pk]
            return _click

        btn.click(
            fn      = _make_click_fn(page_key),
            inputs  = [lang_state] + all_param_inputs,
            outputs = page_groups + [
                sidebar.status_html,
                sidebar.progress_md,
                sidebar.active_state,
            ],
        )

    # ── Param change → status re-computation ─────────────────────────────
    # Trigger on blur (change) of any parameter input.
    # We use gr.on() with all inputs to batch the updates.
    def _on_param_change(lang: str, active_page: str, *param_vals):
        from core.validator import validate_all
        ui_v  = dict(zip(param_keys, param_vals))
        items = validate_all(ui_v)
        s_html, prog = update_status(
            items       = items,
            lang        = lang,
            active_page = active_page,
        )
        return s_html, prog

    gr.on(
        triggers = [c.change for c in all_param_inputs],
        fn       = _on_param_change,
        inputs   = [lang_state, sidebar.active_state] + all_param_inputs,
        outputs  = [sidebar.status_html, sidebar.progress_md],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Language refresh
# ─────────────────────────────────────────────────────────────────────────────

def refresh_lang(
    sidebar:    SidebarComponents,
    lang:       str,
    active_page: str,
    dot_colors: dict[str, str] | None = None,
) -> tuple:
    """
    Rebuild sidebar HTML for a language change without re-running validators.

    Call when the language toggle fires, before validators have re-run.

    Parameters
    ----------
    sidebar : SidebarComponents
    lang : str
    active_page : str
    dot_colors : dict | None
        Current dot colors. If None, all config dots are reset to IDLE.

    Returns
    -------
    tuple[str]  — (status_html_value,)
    """
    if dot_colors is None:
        dot_colors = {pk: _DOT_IDLE for pk in _CONFIG_PAGES}

    html_val = _build_status_html(
        dot_colors  = dot_colors,
        active_page = active_page,
        lang        = lang,
    )
    return (html_val,)
