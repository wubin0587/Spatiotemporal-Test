"""
core/config_bridge.py

Bidirectional converter between flat UI value dicts and the nested
SimulationFacade config dict schema.

Three public functions:
  build_config_from_ui(v)         -> complete nested config dict
  build_analysis_config_from_ui(v)-> analysis config dict
  extract_ui_values_from_config(c)-> flat UI dict (for preset back-fill)
"""

from __future__ import annotations

import copy
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _int(v: Any, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v) if v is not None else default


def _parse_concentration(raw: str, n_layers: int) -> list[float]:
    """
    Parse a comma-separated string into a float list of length n_layers.
    Pads or truncates as needed.
    """
    try:
        values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        values = []
    # Pad with 1.0 if too short
    while len(values) < n_layers:
        values.append(1.0)
    return values[:n_layers]


# ─────────────────────────────────────────────────────────────────────────────
# UI → Simulation config
# ─────────────────────────────────────────────────────────────────────────────

def build_config_from_ui(v: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a flat dict of UI component values into a complete
    SimulationFacade config dict.

    Parameters
    ----------
    v : dict
        {component_key: current_value} — all keys from param_panel.py.

    Returns
    -------
    dict
        Nested config ready for SimulationFacade.from_config_dict().
    """
    n_agents = _int(v.get("num_agents"), 150)
    n_layers = _int(v.get("opinion_layers"), 3)
    concentration = _parse_concentration(v.get("exo_concentration", "1,1,1"), n_layers)

    # ── Engine ────────────────────────────────────────────────────────────────
    engine_cfg: dict[str, Any] = {
        "interface": {
            "agents": {
                "num_agents":     n_agents,
                "opinion_layers": n_layers,
                "initial_opinions": {
                    "type":   v.get("init_type", "polarized"),
                    "params": {"split": _float(v.get("init_split"), 0.5)},
                },
            },
            "simulation": {
                "total_steps":    _int(v.get("total_steps"), 500),
                "seed":           _int(v.get("seed"), 42),
                "record_history": _bool(v.get("record_history"), True),
            },
        },
        "maths": {
            "dynamics": {
                "epsilon_base": _float(v.get("epsilon_base"), 0.25),
                "mu_base":      _float(v.get("mu_base"), 0.35),
                "alpha_mod":    _float(v.get("alpha_mod"), 0.25),
                "beta_mod":     _float(v.get("beta_mod"), 0.15),
                "backfire":     _bool(v.get("backfire"), False),
            },
            "field": {
                "alpha":           _float(v.get("field_alpha", v.get("gamma_field")), 6.0),
                "beta":            _float(v.get("field_beta", v.get("delta_decay")), 0.08),
                "temporal_window": _float(v.get("temporal_window"), 100.0),
            },
            "topo": {
                "threshold":      _float(v.get("topo_threshold"), 0.3),
                "radius_base":    _float(v.get("radius_base"), 0.06),
                "radius_dynamic": _float(v.get("radius_dynamic"), 0.15),
            },
        },
    }

    # ── Networks ──────────────────────────────────────────────────────────────
    net_type = v.get("net_type", "small_world")
    net_params: dict[str, Any] = {"n": n_agents}
    if net_type == "small_world":
        net_params["k"] = _int(v.get("sw_k"), 6)
        net_params["p"] = _float(v.get("sw_p"), 0.1)
    elif net_type == "scale_free":
        net_params["m"] = _int(v.get("sf_m"), 3)
    elif net_type == "random":
        net_params["p"] = _float(v.get("sw_p"), 0.1)

    networks_cfg: dict[str, Any] = {
        "builder": {
            "layers": [
                {
                    "name":   "social",
                    "type":   net_type,
                    "params": net_params,
                }
            ]
        }
    }

    # ── Spatial ───────────────────────────────────────────────────────────────
    spatial_cfg: dict[str, Any] = {
        "distribution": {
            "type":        v.get("spatial_type", "clustered"),
            "n_clusters":  _int(v.get("n_clusters"), 4),
            "cluster_std": _float(v.get("cluster_spread", v.get("cluster_std")), 0.1),
        }
    }

    # ── Events ────────────────────────────────────────────────────────────────
    events_cfg: dict[str, Any] = {
        "generation": {
            # Exogenous
            "exogenous": {
                "enabled": _bool(v.get("exo_enabled"), True),
                "seed":    _int(v.get("exo_seed"), 43),
                "time_trigger": {
                    "type":        "poisson",
                    "lambda_rate": _float(v.get("exo_lambda"), 0.25),
                },
                "attributes": {
                    "location":  {"type": "uniform"},
                    "intensity": {
                        "type":    "pareto",
                        "shape":   _float(v.get("exo_intensity_shape"), 2.5),
                        "min_val": _float(v.get("exo_magnitude", v.get("exo_intensity_min")), 4.0),
                    },
                    "content": {
                        "topic_dim":     n_layers,
                        "concentration": concentration,
                    },
                    "polarity": {
                        "type": "uniform",
                        "min":  _float(v.get("exo_polarity_min"), -0.5),
                        "max":  _float(v.get("exo_polarity_max"), 0.5),
                    },
                    "diffusion": {
                        "type":     "log_normal",
                        "log_mean": -2.0,
                        "log_std":  0.5,
                    },
                    "lifecycle": {
                        "type":       "bimodal",
                        "fast_prob":  0.9,
                        "fast_range": [2, 5],
                        "slow_range": [10, 20],
                    },
                },
            },

            # Endogenous Threshold
            "endogenous_threshold": {
                "enabled":            _bool(v.get("endo_enabled"), True),
                "seed":               _int(v.get("endo_seed"), 44),
                "monitor_attribute":  v.get("endo_monitor", "opinion_extremism"),
                "critical_threshold": _float(v.get("endo_threshold"), 0.12),
                "grid_resolution":    _int(v.get("endo_grid"), 8),
                "min_agents_in_cell": _int(v.get("endo_min_agents"), 2),
                "cooldown":           _int(v.get("endo_cooldown"), 5),
                "attributes": {
                    "intensity": {
                        "base_value":   _float(v.get("endo_base_intensity"), 8.0),
                        "scale_factor": _float(v.get("endo_scale"), 4.0),
                    },
                    "content": {
                        "topic_dim":        n_layers,
                        "amplify_dominant": True,
                    },
                    "polarity":  {"type": "dynamic"},
                    "diffusion": {
                        "min_sigma":  0.1,
                        "max_sigma":  0.3,
                        "var_min":    0.001,
                        "var_max":    0.01,
                        "size_factor": _float(v.get("endo_radius"), 0.1),
                    },
                    "lifecycle": {
                        "type":      "uniform",
                        "min_sigma": 5.0,
                        "max_sigma": 10.0,
                    },
                },
            },

            # Cascade
            "endogenous_cascade": {
                "enabled":           _bool(v.get("cascade_enabled"), True),
                "seed":              _int(v.get("cascade_seed"), 45),
                "background_lambda": _float(v.get("cascade_bg_lambda"), 0.0),
                "mu_multiplier":     _float(v.get("cascade_mu_mult"), 0.6),
                "attributes": {
                    "intensity": {
                        "cascade_decay": _float(v.get("cascade_decay"), 0.5),
                    },
                    "diffusion": {
                        "inherit_from_parent": True,
                        "spatial_mutation":    0.04,
                    },
                    "lifecycle": {
                        "type":      "uniform",
                        "min_sigma": 2.0,
                        "max_sigma": 5.0,
                    },
                },
            },

            # Online Resonance
            "online_resonance": {
                "enabled":               _bool(v.get("online_enabled"), True),
                "seed":                  _int(v.get("online_seed"), 46),
                "check_interval":        _int(v.get("online_check"), 2),
                "smoothing_window":      _int(v.get("online_smooth"), 4),
                "convergence_threshold": _float(v.get("online_convergence"), 0.01),
                "conflict_threshold":    _float(v.get("online_conflict"), 0.01),
                "min_community_size":    _int(v.get("online_min_community"), 3),
                "layer_weights":         [1.0] * n_layers,
                "attributes": {
                    "intensity": {
                        "base_value": _float(v.get("online_base"), 4.0),
                        "size_scale": _float(v.get("online_scale"), 8.0),
                    },
                    "diffusion": {
                        "dispersion_scale": 1.0,
                        "min_sigma":        0.03,
                        "max_sigma":        0.3,
                    },
                    "lifecycle": {
                        "type":      "uniform",
                        "min_sigma": 3.0,
                        "max_sigma": 8.0,
                    },
                },
            },
        }
    }

    return {
        "engine":   engine_cfg,
        "networks": networks_cfg,
        "spatial":  spatial_cfg,
        "events":   events_cfg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI → Analysis config
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_config_from_ui(v: dict[str, Any]) -> dict[str, Any]:
    """
    Build an analysis config dict from UI values.

    Parameters
    ----------
    v : dict
        Must include the analysis-output section keys from param_panel.py.
        AI parser fields (api_key, model, narrative_mode, theme) are optional.

    Returns
    -------
    dict
        Config for run_analysis(engine, config).
    """
    return {
        "output": {
            "dir":                v.get("output_dir", "output/run_001"),
            "formats":            ["md"],
            "lang":               v.get("output_lang", "zh"),
            "save_figures":       True,
            "save_timeseries":    _bool(v.get("save_timeseries"), True),
            "save_features_json": _bool(v.get("save_features_json"), True),
        },
        "feature": {
            "enabled":        True,
            "layer_idx":      _int(v.get("layer_idx", v.get("primary_layer")), 0),
            "include_trends": _bool(v.get("include_trends"), True),
        },
        "parser": {
            "enabled":                   _bool(v.get("ai_enabled"), False),
            "api_key":                   v.get("api_key", ""),
            "model":                     v.get("ai_model", "gpt-4o"),
            "lang":                      v.get("output_lang", "zh"),
            "fmt":                       "md",
            "sections":                  ["opinion", "spatial", "topo", "event"],
            "include_executive_summary": True,
            "max_tokens":                _int(v.get("ai_max_tokens"), 2048),
            "narrative_mode":            v.get("narrative_mode") or None,
            "theme":                     v.get("theme_name") or None,
        },
        "report": {
            "enabled":          True,
            "formats":          [v.get("report_fmt", "md")],
            "include_toc":      True,
            "include_meta":     True,
            "include_snapshot": True,
            "title":            v.get("report_title") or None,
        },
        "visual": {
            "enabled":               True,
            "dashboard":             True,
            "opinion_distribution":  True,
            "spatial_opinions":      True,
            "opinion_timeseries":    True,
            "impact_heatmap":        False,
            "event_timeline":        True,
            "polarization_evolution": True,
            "network_homophily":     False,
            "dpi":                   150,
        },
        "simulation_meta": {
            "n_agents": _int(v.get("num_agents"), 0),
            "n_steps":  _int(v.get("total_steps"), 0),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulation config → flat UI dict  (for preset back-fill)
# ─────────────────────────────────────────────────────────────────────────────

def _get(d: dict, *path: str, default: Any = None) -> Any:
    """Safely navigate a nested dict by a sequence of keys."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
        if cur is None:
            return default
    return cur


def extract_ui_values_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Reverse-convert a full SimulationFacade config dict into a flat UI dict
    that can be used to back-fill param_panel.py components.

    Parameters
    ----------
    config : dict
        Full nested config (as returned by ConfigSwitcher or SimulationFacade).

    Returns
    -------
    dict
        {component_key: value} suitable for gr.update(value=...) calls.
    """
    ifc  = _get(config, "engine", "interface") or {}
    math = _get(config, "engine", "maths")     or {}
    gen  = _get(config, "events", "generation") or {}

    agents = _get(ifc, "agents")     or {}
    sim    = _get(ifc, "simulation") or {}
    dyn    = _get(math, "dynamics")  or {}
    fld    = _get(math, "field")     or {}
    topo   = _get(math, "topo")      or {}

    net_layers = _get(config, "networks", "builder", "layers") or []
    net_layer  = net_layers[0] if net_layers else {}
    net_type   = net_layer.get("type", "small_world")
    net_params = net_layer.get("params") or {}

    spatial = _get(config, "spatial", "distribution") or {}

    exo     = _get(gen, "exogenous") or {}
    endo    = _get(gen, "endogenous_threshold") or {}
    cascade = _get(gen, "endogenous_cascade") or {}
    online  = _get(gen, "online_resonance") or {}

    init_op = _get(agents, "initial_opinions") or {}
    conc    = _get(exo, "attributes", "content", "concentration") or [1, 1, 1]

    return {
        # Agent & Simulation
        "num_agents":        agents.get("num_agents", 150),
        "opinion_layers":    agents.get("opinion_layers", 3),
        "total_steps":       sim.get("total_steps", 500),
        "seed":              sim.get("seed", 42),
        "record_history":    sim.get("record_history", True),
        "init_type":         init_op.get("type", "polarized"),
        "init_split":        _get(init_op, "params", "split") or 0.5,
        # Dynamics
        "epsilon_base":      dyn.get("epsilon_base", 0.25),
        "mu_base":           dyn.get("mu_base", 0.35),
        "alpha_mod":         dyn.get("alpha_mod", 0.25),
        "beta_mod":          dyn.get("beta_mod", 0.15),
        "backfire":          dyn.get("backfire", False),
        # Field
        "field_alpha":       fld.get("alpha", 6.0),
        "field_beta":        fld.get("beta", 0.08),
        "gamma_field":       fld.get("alpha", 6.0),
        "delta_decay":       fld.get("beta", 0.08),
        "temporal_window":   fld.get("temporal_window", 100.0),
        # Topo
        "topo_threshold":    topo.get("threshold", 0.3),
        "radius_base":       topo.get("radius_base", 0.06),
        "radius_dynamic":    topo.get("radius_dynamic", 0.15),
        # Network
        "net_type":          net_type,
        "sw_k":              net_params.get("k", 6),
        "sw_p":              net_params.get("p", 0.1),
        "sf_m":              net_params.get("m", 3),
        # Spatial
        "spatial_type":      spatial.get("type", "clustered"),
        "n_clusters":        spatial.get("n_clusters", 4),
        "cluster_std":       spatial.get("cluster_std", 0.1),
        "cluster_spread":    spatial.get("cluster_std", 0.1),
        # Exogenous
        "exo_enabled":       exo.get("enabled", True),
        "exo_seed":          exo.get("seed", 43),
        "exo_lambda":        _get(exo, "time_trigger", "lambda_rate") or 0.25,
        "exo_intensity_shape": _get(exo, "attributes", "intensity", "shape") or 2.5,
        "exo_intensity_min": _get(exo, "attributes", "intensity", "min_val") or 4.0,
        "exo_magnitude":     _get(exo, "attributes", "intensity", "min_val") or 4.0,
        "exo_polarity_min":  _get(exo, "attributes", "polarity", "min") or -0.5,
        "exo_polarity_max":  _get(exo, "attributes", "polarity", "max") or 0.5,
        "exo_concentration": ",".join(str(c) for c in conc),
        # Endogenous Threshold
        "endo_enabled":      endo.get("enabled", True),
        "endo_seed":         endo.get("seed", 44),
        "endo_monitor":      endo.get("monitor_attribute", "opinion_extremism"),
        "endo_threshold":    endo.get("critical_threshold", 0.12),
        "endo_grid":         endo.get("grid_resolution", 8),
        "endo_min_agents":   endo.get("min_agents_in_cell", 2),
        "endo_cooldown":     endo.get("cooldown", 5),
        "endo_base_intensity": _get(endo, "attributes", "intensity", "base_value") or 8.0,
        "endo_scale":        _get(endo, "attributes", "intensity", "scale_factor") or 4.0,
        "endo_radius":       _get(endo, "attributes", "diffusion", "size_factor") or 0.1,
        # Cascade
        "cascade_enabled":   cascade.get("enabled", True),
        "cascade_seed":      cascade.get("seed", 45),
        "cascade_bg_lambda": cascade.get("background_lambda", 0.0),
        "cascade_mu_mult":   cascade.get("mu_multiplier", 0.6),
        "cascade_decay":     _get(cascade, "attributes", "intensity", "cascade_decay") or 0.5,
        # Online Resonance
        "online_enabled":    online.get("enabled", True),
        "online_seed":       online.get("seed", 46),
        "online_check":      online.get("check_interval", 2),
        "online_smooth":     online.get("smoothing_window", 4),
        "online_convergence": online.get("convergence_threshold", 0.01),
        "online_conflict":   online.get("conflict_threshold", 0.01),
        "online_min_community": online.get("min_community_size", 3),
        "online_base":       _get(online, "attributes", "intensity", "base_value") or 4.0,
        "online_scale":      _get(online, "attributes", "intensity", "size_scale") or 8.0,
    }
