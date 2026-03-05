"""
config/switcher.py

ConfigSwitcher — 统一配置组装入口。

支持四种输入模式（可任意混合）：
  1. 完整 config dict          → 直接透传，可叠加 _overrides_
  2. _presets_ 选择预设        → 从 yamls/ 字典库加载并挂载
  3. 直接写入任意 config 块    → 与预设深合并，优先级更高
  4. _overrides_ 点路径覆盖    → 最高优先级，精确修改任意叶节点

优先级（从低到高）：
  base骨架 < _presets_ < 直接写的块 < _overrides_

对外使用：
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
# key  : group 名（传给 _presets_ 的字符串）
# value: 挂载到完整 config 的点路径，None 表示直接 merge 到根
# ---------------------------------------------------------------------------
MOUNT_POINTS: dict[str, str | None] = {
    # engine — initial opinions 分布类型
    "engine.interface":                 "engine.interface.agents.initial_opinions",

    # engine — 数学参数块（每块独立文件）
    "engine.maths/dynamics":            "engine.maths.dynamics",
    "engine.maths/field":               "engine.maths.field",
    "engine.maths/topo":                "engine.maths.topo",

    # events — 三类生成器整块配置
    "events/generate.exp":              "events.generation.exogenous",
    "events/generate.imp":              "events.generation.endogenous_threshold",
    "events/generate.cascade":          "events.generation.endogenous_cascade",

    # events — 扩散/生命周期分布（通常内嵌，也可单独选）
    "events/generate.dist.spatial":     "events.generation.exogenous.attributes.diffusion",
    "events/generate.dist.time":        "events.generation.exogenous.attributes.lifecycle",

    # networks
    "networks.builder":                 "networks.builder",

    # spatial
    "spatial.distributions":            "spatial.distribution",

    # intervention
    "intervention/manager":             "intervention",
    "intervention/triggers":            None,  # 单独使用，不自动挂载
    "intervention/policies":            None,  # 单独使用，不自动挂载

    # analyse
    "analyse":                          "analyse",
}


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _yaml_root() -> Path:
    """定位 yamls/ 目录（与本文件同级的 yamls/）。"""
    here = Path(__file__).parent
    candidate = here / "yamls"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"找不到 yamls/ 字典库目录，期望位置：{candidate}"
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, patch: dict) -> dict:
    """
    递归深合并。patch 覆盖 base；dict 类型递归合并，其余直接覆盖。
    不修改原始对象。
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
    按点路径写入值，自动创建中间层 dict。
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
    """按点路径读取值，路径不存在时返回 default。"""
    keys = path.split(".")
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _mount(config: dict, mount_path: str | None, patch: dict) -> dict:
    """将 patch 深合并到 config 的 mount_path 位置。"""
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
    统一配置组装器。

    Parameters
    ----------
    yamls_root : str | Path | None
        yamls/ 字典库根目录。None 则自动推断（与本文件同级的 yamls/）。
    """

    def __init__(self, yamls_root: str | Path | None = None) -> None:
        self._root: Path = Path(yamls_root) if yamls_root else _yaml_root()
        base_path = self._root.parent / "yamls" / "base.yaml"
        self._base: dict = _load_yaml(base_path) if base_path.exists() else {}
        # themes/ 与 yamls/ 同级
        self._themes_root: Path = self._root.parent / "themes"
        # 共享分析配置
        shared_analyse_path = self._themes_root / "_shared_analyse.yaml"
        self._shared_analyse: dict = (
            _load_yaml(shared_analyse_path) if shared_analyse_path.exists() else {}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, spec: dict) -> dict:
        """
        将 spec 解析为完整 config dict。

        spec 保留字段
        -------------
        _presets_   : dict[group, choice]
            从字典库选预设，按 MOUNT_POINTS 挂载。
            e.g. {"networks.builder": "small_world",
                  "spatial.distributions": "clustered"}

        _overrides_ : dict[dot.path, value]
            点路径精确覆盖，支持任意深度。
            e.g. {"engine.maths.dynamics.epsilon_base": 0.35,
                  "networks.builder.params.k": 8}

        spec 中其余顶层 key 视为直接写入的 config 块，与预设深合并
        （优先级高于预设，低于 _overrides_）。

        优先级（从低到高）
        -----------------
        base骨架 < _presets_ < 直接写的块 < _overrides_

        Returns
        -------
        dict  完整可用的 config dict，可直接传给 SimulationFacade
        """
        spec = copy.deepcopy(spec)
        presets   = spec.pop("_presets_",   {})
        overrides = spec.pop("_overrides_", {})

        # 1. 基础骨架
        config = copy.deepcopy(self._base)

        # 2. 应用预设
        for group, choice in presets.items():
            patch = self._load_preset(group, choice)
            mount = MOUNT_POINTS.get(group)
            config = _mount(config, mount, patch)

        # 3. 直接写入的块（高于预设）
        if spec:
            config = _deep_merge(config, spec)

        # 4. 点路径覆盖（最高优先级）
        for path, value in overrides.items():
            _set_by_path(config, path, value)

        return config

    def resolve_theme(
        self,
        name: str,
        extra_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """
        按 theme 名称加载并组装完整仿真 config 与分析 config。

        Parameters
        ----------
        name : str
            theme 文件名（不含 .yaml），e.g. "concert", "radicalization"
        extra_overrides : dict | None
            额外的点路径覆盖，优先级最高，叠加在 theme._overrides 之上。

        Returns
        -------
        sim_config : dict
            完整仿真配置，可直接传给 SimulationFacade.from_config_dict()
        analysis_config : dict
            完整分析配置，可直接传给 run_analysis()

        Example
        -------
        sim_cfg, ana_cfg = switcher.resolve_theme("concert")
        sim_cfg, ana_cfg = switcher.resolve_theme(
            "radicalization",
            extra_overrides={"engine.interface.agents.num_agents": 500},
        )
        """
        theme = self._load_theme(name)

        # --- 构造仿真 spec ---
        spec: dict = {}
        if "_presets" in theme:
            spec["_presets_"] = theme["_presets"]
        overrides_dict: dict = dict(theme.get("_overrides", {}) or {})
        if extra_overrides:
            overrides_dict.update(extra_overrides)
        if overrides_dict:
            spec["_overrides_"] = overrides_dict

        sim_config = self.resolve(spec)

        # --- 构造分析 config ---
        analysis_config = copy.deepcopy(self._shared_analyse)
        for path, value in (theme.get("_analyse_overrides") or {}).items():
            _set_by_path(analysis_config, path, value)

        return sim_config, analysis_config

    def load_theme(self, name: str) -> dict:
        """直接返回 theme 原始 dict，不做 merge，供检查用。"""
        return self._load_theme(name)

    def list_themes(self) -> list[str]:
        """列出所有可用 theme 名称（不含下划线开头的内部文件）。"""
        if not self._themes_root.exists():
            return []
        return sorted(
            p.stem
            for p in self._themes_root.glob("*.yaml")
            if not p.stem.startswith("_")
        )

    def theme_meta(self, name: str) -> dict:
        """返回 theme 的 _meta 块（label、description、tags 等）。"""
        return self._load_theme(name).get("_meta", {})

    def all_theme_metas(self) -> dict[str, dict]:
        """返回所有 theme 的 _meta 索引，方便程序化选择 theme。"""
        return {name: self.theme_meta(name) for name in self.list_themes()}

    def load_preset(self, group: str, choice: str) -> dict:
        """
        直接加载某个预设 dict，不做 merge。
        可用于检查预设内容或手动组装。
        """
        return self._load_preset(group, choice)

    def get(self, config: dict, path: str, default: Any = None) -> Any:
        """从已 resolve 的 config 中按点路径读取值。"""
        return _get_by_path(config, path, default)

    def set(self, config: dict, path: str, value: Any) -> dict:
        """
        对已 resolve 的 config 按点路径写入值，返回新 dict（不修改原对象）。
        可用于链式修改：
            cfg = switcher.resolve({...})
            cfg = switcher.set(cfg, "engine.maths.dynamics.epsilon_base", 0.4)
        """
        result = copy.deepcopy(config)
        _set_by_path(result, path, value)
        return result

    def patch(self, config: dict, patches: dict[str, Any]) -> dict:
        """
        批量点路径写入，等价于多次调用 set()。
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
        """列出所有已注册的 group 名（MOUNT_POINTS 的 key）。"""
        return list(MOUNT_POINTS.keys())

    def list_choices(self, group: str) -> list[str]:
        """列出某 group 下所有可用 choice（yaml 文件名，不含扩展名）。"""
        group_dir = self._group_dir(group)
        if not group_dir.exists():
            return []
        return sorted(p.stem for p in group_dir.glob("*.yaml"))

    def available(self) -> dict[str, list[str]]:
        """返回所有 group → [choices] 的完整索引。"""
        return {g: self.list_choices(g) for g in MOUNT_POINTS}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_theme(self, name: str) -> dict:
        """加载 themes/{name}.yaml，文件不存在时报错。"""
        path = self._themes_root / f"{name}.yaml"
        if not path.exists():
            available = self.list_themes()
            raise FileNotFoundError(
                f"找不到 theme '{name}'。\n"
                f"期望路径：{path}\n"
                f"可用 themes：{available}"
            )
        return _load_yaml(path)

    def _group_dir(self, group: str) -> Path:
        """
        将 group 名映射到 yamls/ 下的目录路径。

        命名风格：
          "networks.builder"      → yamls/networks.builder/
          "events/generate.exp"   → yamls/events/  （文件以 generate.exp 命名）
          "engine.maths/dynamics" → yamls/engine.maths/
        """
        if "/" in group:
            parts = group.split("/")
            return self._root.joinpath(*parts[:-1])
        return self._root / group

    def _load_preset(self, group: str, choice: str) -> dict:
        """
        加载 group + choice 对应的 yaml 文件。

        查找顺序（取第一个存在的）：
          1. {root}/{group_dir}/{choice}.yaml      标准：子目录 + choice 文件名
          2. {root}/{group_dir}/{file_prefix}/{choice}.yaml  嵌套子目录
          3. {root}/{group_dir}/{file_prefix}.yaml  group 本身是单文件（忽略 choice）
        """
        candidates: list[Path] = []

        if "/" in group:
            parts    = group.split("/")
            base_dir = self._root.joinpath(*parts[:-1])
            prefix   = parts[-1]
            candidates += [
                base_dir / f"{prefix}" / f"{choice}.yaml",   # events/generate.exp/uniform.yaml
                base_dir / f"{choice}.yaml",                  # events/uniform.yaml (fallback)
                base_dir / f"{prefix}.yaml",                  # events/generate.exp.yaml (单文件)
            ]
        else:
            candidates += [
                self._root / group / f"{choice}.yaml",        # networks.builder/small_world.yaml
                self._root / group / choice / "default.yaml", # 极少情况
            ]

        for path in candidates:
            if path.exists():
                return _load_yaml(path)

        raise FileNotFoundError(
            f"找不到预设 group='{group}' choice='{choice}'。\n"
            f"已尝试路径：\n" + "\n".join(f"  {p}" for p in candidates)
        )
