"""
config/switcher.py

ConfigSwitcher — Unified configuration assembly entry point.

Supports four input modes (can be mixed arbitrarily):
  1. Complete config dict      -> Passed through directly, can be overlaid with _overrides_
  2. _presets_ selection       -> Loaded from the yamls/ dictionary library and mounted
  3. Direct config blocks      -> Deep merged with presets, higher priority
  4. _overrides_ dot-path      -> Highest priority, precisely modifies any leaf node

Priority (from lowest to highest):
  base skeleton < _presets_ < direct blocks < _overrides_

External usage:
  from config import switcher
  config = switcher.resolve({...})
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Mount point table
# key  : group name (the string passed to _presets_)
# value: dot-path to mount onto the full config, None means merge directly to root
# ---------------------------------------------------------------------------
MOUNT_POINTS: dict[str, str | None] = {
    # engine — initial opinions distribution type
    "engine.interface":                 "engine.interface.agents.initial_opinions",

    # engine — mathematical parameter blocks (each block is an independent file)
    "engine.maths/dynamics":            "engine.maths.dynamics",
    "engine.maths/field":               "engine.maths.field",
    "engine.maths/topo":                "engine.maths.topo",

    # events — whole block configurations for three types of generators
    "events/generate.exp":              "events.generation.exogenous",
    "events/generate.imp":              "events.generation.endogenous_threshold",
    "events/generate.cascade":          "events.generation.endogenous_cascade",

    # events — diffusion/lifecycle distributions (usually embedded, but can be selected separately)
    "events/generate.dist.spatial":     "events.generation.exogenous.attributes.diffusion",
    "events/generate.dist.time":        "events.generation.exogenous.attributes.lifecycle",

    # networks
    "networks.builder":                 "networks.builder",

    # spatial
    "spatial.distributions":            "spatial.distribution",

    # intervention
    "intervention/manager":             "intervention",
    "intervention/triggers":            None,  # Used separately, not mounted automatically
    "intervention/policies":            None,  # Used separately, not mounted automatically

    # analyse
    "analyse":                          "analyse",
}


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _yaml_root() -> Path:
    """Locate the yamls/ directory (the yamls/ at the same level as this file)."""
    here = Path(__file__).parent
    candidate = here / "yamls"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find the yamls/ dictionary directory. Expected location: {candidate}"
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, patch: dict) -> dict:
    """
    Recursive deep merge. 'patch' overwrites 'base'; dict types are merged recursively, 
    others are overwritten directly.
    Does not modify the original objects.
    """
    result = copy.deepcopy(base)
    for k, v in patch.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _set_by_path(d: dict, path: str, value: Any) -> None:
    """
    Set a value by dot-path, automatically creating intermediate dict layers.
    e.g. _set_by_path(d, "engine.maths.dynamics.epsilon_base", 0.35)
    """
    keys = path.split(".")
    cur = d
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _get_by_path(d: dict, path: str, default: Any = None) -> Any:
    """Get a value by dot-path, returning default if the path does not exist."""
    keys = path.split(".")
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _mount(config: dict, mount_path: str | None, patch: dict) -> dict:
    """Deep merge the patch into the config at the mount_path location."""
    if mount_path is None:
        return _deep_merge(config, patch)
    existing = _get_by_path(config, mount_path)
    merged = _deep_merge(existing, patch) if isinstance(existing, dict) else patch
    result = copy.deepcopy(config)
    _set_by_path(result, mount_path, merged)
    return result


# ---------------------------------------------------------------------------
# ConfigSwitcher
# ---------------------------------------------------------------------------

class ConfigSwitcher:
    """
    Unified configuration assembler.

    Parameters
    ----------
    yamls_root : str | Path | None
        Root directory of the yamls/ dictionary library. If None, it is automatically 
        inferred (the yamls/ at the same level as this file).
    """

    def __init__(self, yamls_root: str | Path | None = None) -> None:
        self._root: Path = Path(yamls_root) if yamls_root else _yaml_root()
        base_path = self._root.parent / "yamls" / "base.yaml"
        self._base: dict = _load_yaml(base_path) if base_path.exists() else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, spec: dict) -> dict:
        """
        Resolve the spec into a complete config dict.

        Reserved fields in spec
        -----------------------
        _presets_   : dict[group, choice]
            Select presets from the dictionary library and mount them according to MOUNT_POINTS.
            e.g. {"networks.builder": "small_world",
                  "spatial.distributions": "clustered"}

        _overrides_ : dict[dot.path, value]
            Precise dot-path overrides, supporting arbitrary depths.
            e.g. {"engine.maths.dynamics.epsilon_base": 0.35,
                  "networks.builder.params.k": 8}

        Other top-level keys in spec are treated as direct config blocks and deep merged 
        with presets (higher priority than presets, lower than _overrides_).

        Priority (from lowest to highest)
        ---------------------------------
        base skeleton < _presets_ < direct blocks < _overrides_

        Returns
        -------
        dict  A complete, ready-to-use config dict that can be passed directly to SimulationFacade.
        """
        spec = copy.deepcopy(spec)
        presets   = spec.pop("_presets_",   {})
        overrides = spec.pop("_overrides_", {})

        # 1. Base skeleton
        config = copy.deepcopy(self._base)

        # 2. Apply presets
        for group, choice in presets.items():
            patch = self._load_preset(group, choice)
            mount = MOUNT_POINTS.get(group)
            config = _mount(config, mount, patch)

        # 3. Direct config blocks (higher than presets)
        if spec:
            config = _deep_merge(config, spec)

        # 4. Dot-path overrides (highest priority)
        for path, value in overrides.items():
            _set_by_path(config, path, value)

        return config

    def load_preset(self, group: str, choice: str) -> dict:
        """
        Directly load a preset dict without merging.
        Can be used to inspect preset content or for manual assembly.
        """
        return self._load_preset(group, choice)

    def get(self, config: dict, path: str, default: Any = None) -> Any:
        """Get a value by dot-path from an already resolved config."""
        return _get_by_path(config, path, default)

    def set(self, config: dict, path: str, value: Any) -> dict:
        """
        Set a value by dot-path on an already resolved config, returning a new dict 
        (does not modify the original object).
        Can be used for chained modifications:
            cfg = switcher.resolve({...})
            cfg = switcher.set(cfg, "engine.maths.dynamics.epsilon_base", 0.4)
        """
        result = copy.deepcopy(config)
        _set_by_path(result, path, value)
        return result

    def patch(self, config: dict, patches: dict[str, Any]) -> dict:
        """
        Batch dot-path value setting, equivalent to calling set() multiple times.
        e.g. switcher.patch(cfg, {
                 "engine.maths.dynamics.epsilon_base": 0.4,
                 "engine.maths.dynamics.mu_base": 0.25,
             })
        """
        result = copy.deepcopy(config)
        for path, value in patches.items():
            _set_by_path(result, path, value)
        return result

    def list_groups(self) -> list[str]:
        """List all registered group names (keys of MOUNT_POINTS)."""
        return list(MOUNT_POINTS.keys())

    def list_choices(self, group: str) -> list[str]:
        """List all available choices under a specific group (yaml filenames, excluding extensions)."""
        group_dir = self._group_dir(group)
        if not group_dir.exists():
            return[]
        return sorted(p.stem for p in group_dir.glob("*.yaml"))

    def available(self) -> dict[str, list[str]]:
        """Return a complete index of all group -> [choices]."""
        return {g: self.list_choices(g) for g in MOUNT_POINTS}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_dir(self, group: str) -> Path:
        """
        Map a group name to a directory path under yamls/.

        Naming style:
          "networks.builder"      -> yamls/networks.builder/
          "events/generate.exp"   -> yamls/events/  (files named with generate.exp)
          "engine.maths/dynamics" -> yamls/engine.maths/
        """
        if "/" in group:
            parts = group.split("/")
            return self._root.joinpath(*parts[:-1])
        return self._root / group

    def _load_preset(self, group: str, choice: str) -> dict:
        """
        Load the yaml file corresponding to the group + choice.

        Lookup order (returns the first existing one):
          1. {root}/{group_dir}/{choice}.yaml                 Standard: subdirectory + choice filename
          2. {root}/{group_dir}/{file_prefix}/{choice}.yaml   Nested subdirectory
          3. {root}/{group_dir}/{file_prefix}.yaml            Group itself is a single file (ignores choice)
        """
        candidates: list[Path] =[]

        if "/" in group:
            parts    = group.split("/")
            base_dir = self._root.joinpath(*parts[:-1])
            prefix   = parts[-1]
            candidates +=[
                base_dir / f"{prefix}" / f"{choice}.yaml",    # events/generate.exp/uniform.yaml
                base_dir / f"{choice}.yaml",                  # events/uniform.yaml (fallback)
                base_dir / f"{prefix}.yaml",                  # events/generate.exp.yaml (single file)
            ]
        else:
            candidates +=[
                self._root / group / f"{choice}.yaml",        # networks.builder/small_world.yaml
                self._root / group / choice / "default.yaml", # rare case
            ]

        for path in candidates:
            if path.exists():
                return _load_yaml(path)

        raise FileNotFoundError(
            f"Cannot find preset group='{group}' choice='{choice}'.\n"
            f"Tried paths:\n" + "\n".join(f"  {p}" for p in candidates)
        )