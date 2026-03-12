"""
config/switcher.py

ConfigSwitcher - unified configuration assembly entry point.

Supports four input modes (freely combinable):
  1. Full config dict           -> passed through as-is, _overrides_ still applied
  2. _presets_ preset selection -> loaded from the yamls/ library and mounted
  3. Inline config blocks       -> deep-merged on top of presets (higher priority)
  4. _overrides_ dot-path write -> highest priority, overwrites any leaf value

Priority order (lowest to highest):
  base skeleton < _presets_ < inline blocks < _overrides_

Public usage:
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
# key  : group name (string passed to _presets_)
# value: dot-path within the full config where the preset is mounted;
#        None means merge directly into the root
# ---------------------------------------------------------------------------
MOUNT_POINTS: dict[str, str | None] = {
    # engine - initial opinion distribution type
    "engine.interface":              "engine.interface.agents.initial_opinions",

    # engine - math parameter blocks (one file each)
    "engine.maths/dynamics":         "engine.maths.dynamics",
    "engine.maths/field":            "engine.maths.field",
    "engine.maths/topo":             "engine.maths.topo",

    # events - generator blocks
    "events/generate.exp":           "events.generation.exogenous",
    "events/generate.imp":           "events.generation.endogenous_threshold",
    "events/generate.cascade":       "events.generation.endogenous_cascade",
    "events/generate.online":        "events.generation.online_resonance",

    # events - diffusion / lifecycle sub-distributions (embeddable or standalone)
    "events/generate.dist.spatial":  "events.generation.exogenous.attributes.diffusion",
    "events/generate.dist.time":     "events.generation.exogenous.attributes.lifecycle",

    # networks
    "networks.builder":              "networks.builder",

    # spatial
    "spatial.distributions":         "spatial.distribution",

    # intervention
    "intervention/manager":          "intervention",
    "intervention/triggers":         None,  # standalone use only, not auto-mounted
    "intervention/policies":         None,  # standalone use only, not auto-mounted

    # analyse
    "analyse":                       "analyse",
}


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _yaml_root() -> Path:
    """Locate the yamls/ directory (sibling of this file)."""
    here = Path(__file__).parent
    candidate = here / "yamls"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"yamls/ library directory not found; expected: {candidate}"
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, patch: dict) -> dict:
    """
    Recursively deep-merge two dicts. patch overrides base; dicts are merged
    recursively, all other types are overwritten. Neither input is modified.
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
    Write a value at a dot-separated path, creating intermediate dicts as needed.
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
    """Read a value at a dot-separated path; return default if not found."""
    keys = path.split(".")
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _mount(config: dict, mount_path: str | None, patch: dict) -> dict:
    """Deep-merge patch into config at the given mount_path."""
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
        Root directory of the yamls/ library. Defaults to the yamls/ folder
        that is a sibling of this file.
    """

    def __init__(self, yamls_root: str | Path | None = None) -> None:
        self._root: Path = Path(yamls_root) if yamls_root else _yaml_root()

        base_path = self._root.parent / "yamls" / "base.yaml"
        self._base: dict = _load_yaml(base_path) if base_path.exists() else {}

        # themes/ sits alongside yamls/
        self._themes_root: Path = self._root.parent / "themes"

        # shared analysis config used by all themes
        shared_analyse_path = self._themes_root / "_shared_analyse.yaml"
        self._shared_analyse: dict = (
            _load_yaml(shared_analyse_path) if shared_analyse_path.exists() else {}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, spec: dict) -> dict:
        """
        Resolve a spec dict into a complete, ready-to-use config dict.

        Reserved keys in spec
        ---------------------
        _presets_   : dict[group, choice]
            Select presets from the library; mounted according to MOUNT_POINTS.
            e.g. {"networks.builder": "small_world",
                  "spatial.distributions": "clustered"}

        _overrides_ : dict[dot.path, value]
            Dot-path overwrites applied last (highest priority).
            e.g. {"engine.maths.dynamics.epsilon_base": 0.35,
                  "networks.builder.params.k": 8}

        All other top-level keys are treated as inline config blocks and
        deep-merged on top of presets (higher priority than presets,
        lower than _overrides_).

        Priority (lowest to highest)
        ----------------------------
        base skeleton < _presets_ < inline blocks < _overrides_

        Returns
        -------
        dict  Complete config ready for SimulationFacade.from_config_dict()
        """
        spec = copy.deepcopy(spec)
        presets   = spec.pop("_presets_",   {})
        overrides = spec.pop("_overrides_", {})

        # 1. start from the base skeleton
        config = copy.deepcopy(self._base)

        # 2. apply presets
        for group, choice in presets.items():
            patch = self._load_preset(group, choice)
            mount = MOUNT_POINTS.get(group)
            config = _mount(config, mount, patch)

        # 3. merge inline blocks (higher priority than presets)
        if spec:
            config = _deep_merge(config, spec)

        # 4. apply dot-path overrides (highest priority)
        for path, value in overrides.items():
            _set_by_path(config, path, value)

        return config

    def resolve_theme(
        self,
        name: str,
        extra_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """
        Load a named theme and return a fully assembled (sim_config, analysis_config)
        pair.

        Parameters
        ----------
        name : str
            Theme filename without extension, e.g. "concert", "radicalization".
        extra_overrides : dict | None
            Additional dot-path overrides; merged on top of the theme's own
            _overrides block (highest priority).

        Returns
        -------
        sim_config : dict
            Complete simulation config for SimulationFacade.from_config_dict()
        analysis_config : dict
            Complete analysis config for run_analysis()

        Examples
        --------
        sim_cfg, ana_cfg = switcher.resolve_theme("concert")
        sim_cfg, ana_cfg = switcher.resolve_theme(
            "radicalization",
            extra_overrides={"engine.interface.agents.num_agents": 500},
        )
        """
        theme = self._load_theme(name)

        # build simulation spec
        spec: dict = {}
        if "_presets" in theme:
            spec["_presets_"] = theme["_presets"]

        overrides_dict: dict = dict(theme.get("_overrides", {}) or {})
        if extra_overrides:
            overrides_dict.update(extra_overrides)
        if overrides_dict:
            spec["_overrides_"] = overrides_dict

        sim_config = self.resolve(spec)

        # build analysis config: shared base + theme-specific overrides
        analysis_config = copy.deepcopy(self._shared_analyse)
        for path, value in (theme.get("_analyse_overrides") or {}).items():
            _set_by_path(analysis_config, path, value)

        return sim_config, analysis_config

    def load_theme(self, name: str) -> dict:
        """Return the raw theme dict without merging (useful for inspection)."""
        return self._load_theme(name)

    def list_themes(self) -> list[str]:
        """List all available theme names (files starting with _ are excluded)."""
        if not self._themes_root.exists():
            return []
        return sorted(
            p.stem
            for p in self._themes_root.glob("*.yaml")
            if not p.stem.startswith("_")
        )

    def theme_meta(self, name: str) -> dict:
        """Return the _meta block of a theme (label, description, tags, etc.)."""
        return self._load_theme(name).get("_meta", {})

    def all_theme_metas(self) -> dict[str, dict]:
        """Return a {name: _meta} index for all themes."""
        return {name: self.theme_meta(name) for name in self.list_themes()}

    def load_preset(self, group: str, choice: str) -> dict:
        """Return a raw preset dict without merging (useful for inspection)."""
        return self._load_preset(group, choice)

    def get(self, config: dict, path: str, default: Any = None) -> Any:
        """Read a value from a resolved config by dot-path."""
        return _get_by_path(config, path, default)

    def set(self, config: dict, path: str, value: Any) -> dict:
        """
        Write a single value into a resolved config by dot-path.
        Returns a new dict; the original is not modified.

        Example
        -------
        cfg = switcher.set(cfg, "engine.maths.dynamics.epsilon_base", 0.4)
        """
        result = copy.deepcopy(config)
        _set_by_path(result, path, value)
        return result

    def patch(self, config: dict, patches: dict[str, Any]) -> dict:
        """
        Write multiple dot-path values into a resolved config at once.
        Returns a new dict; the original is not modified.

        Example
        -------
        cfg = switcher.patch(cfg, {
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
        """List all available choices for a group (yaml filenames without extension)."""
        group_dir = self._group_dir(group)
        if not group_dir.exists():
            return []
        return sorted(p.stem for p in group_dir.glob("*.yaml"))

    def available(self) -> dict[str, list[str]]:
        """Return a full {group: [choices]} index of the yaml library."""
        return {g: self.list_choices(g) for g in MOUNT_POINTS}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_theme(self, name: str) -> dict:
        """Load themes/{name}.yaml; raise FileNotFoundError if missing."""
        path = self._themes_root / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Theme '{name}' not found.\n"
                f"Expected path: {path}\n"
                f"Available themes: {self.list_themes()}"
            )
        return _load_yaml(path)

    def _group_dir(self, group: str) -> Path:
        """
        Map a group name to its directory inside yamls/.

        Naming conventions supported:
          "networks.builder"      -> yamls/networks.builder/
          "events/generate.exp"  -> yamls/events/   (file prefix = generate.exp)
          "engine.maths/dynamics" -> yamls/engine.maths/
        """
        if "/" in group:
            parts = group.split("/")
            return self._root.joinpath(*parts[:-1])
        return self._root / group

    def _load_preset(self, group: str, choice: str) -> dict:
        """
        Load the yaml file for a given group + choice pair.

        Search order (first match wins):
          1. {root}/{group_dir}/{choice}.yaml            standard sub-directory layout
          2. {root}/{group_dir}/{file_prefix}/{choice}.yaml  nested sub-directory
          3. {root}/{group_dir}/{file_prefix}.yaml       single-file group (choice ignored)
        """
        candidates: list[Path] = []

        if "/" in group:
            parts    = group.split("/")
            base_dir = self._root.joinpath(*parts[:-1])
            prefix   = parts[-1]
            candidates += [
                base_dir / prefix / f"{choice}.yaml",  # e.g. events/generate.exp/uniform.yaml
                base_dir / f"{choice}.yaml",            # fallback: events/uniform.yaml
                base_dir / f"{prefix}.yaml",            # single-file group
            ]
        else:
            candidates += [
                self._root / group / f"{choice}.yaml",         # e.g. networks.builder/small_world.yaml
                self._root / group / choice / "default.yaml",  # rarely used nested layout
            ]

        for path in candidates:
            if path.exists():
                return _load_yaml(path)

        raise FileNotFoundError(
            f"Preset not found: group='{group}' choice='{choice}'.\n"
            "Tried paths:\n" + "\n".join(f"  {p}" for p in candidates)
        )
