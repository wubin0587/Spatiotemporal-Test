"""
core/validator.py

Pre-run parameter validation for the Opinion Dynamics Simulation Dashboard.

Validates all simulation parameters grouped into 6 logical sections,
returning structured CheckItem results consumed by the P6-A checklist UI.

Public API
----------
CheckItem
    Dataclass representing a single validation result for one config group.

validate_all(ui_values) -> list[CheckItem]
    Run all validators and return one CheckItem per group.

validate_group(group, ui_values) -> CheckItem
    Validate a single group by name (for partial re-validation on page change).

STATUS constants: STATUS_OK, STATUS_WARN, STATUS_ERROR

Design
------
- No Gradio imports — pure Python, fully unit-testable.
- Validators are registered functions, easy to extend.
- Each validator receives the full ui_values dict and returns
  (status, detail, extra_hints) where extra_hints is a list of
  (field_key, message) pairs for field-level annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Status constants
# ─────────────────────────────────────────────────────────────────────────────

STATUS_OK    = "ok"
STATUS_WARN  = "warn"
STATUS_ERROR = "error"


# ─────────────────────────────────────────────────────────────────────────────
# CheckItem dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckItem:
    """
    Validation result for one configuration group.

    Attributes
    ----------
    group : str
        Internal key matching NAV_ITEMS in sidebar.py.
        One of: "model_config", "dynamics", "network_spatial",
                "events", "intervention", "analysis_config"
    status : str
        STATUS_OK | STATUS_WARN | STATUS_ERROR
    title : str
        Short display title for the checklist row (≤ 30 chars).
    detail : str
        One-line explanation shown below the title (≤ 60 chars).
    target_page : str
        Page key to navigate to when "去修改" is clicked.
        Matches page keys used in app.py switch_page().
    field_hints : list[tuple[str, str]]
        Per-field (key, message) pairs for inline input annotation.
        Empty list when status is OK.
    """
    group:       str
    status:      str
    title:       str
    detail:      str
    target_page: str
    field_hints: list[tuple[str, str]] = field(default_factory=list)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def is_ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def is_warn(self) -> bool:
        return self.status == STATUS_WARN

    @property
    def is_error(self) -> bool:
        return self.status == STATUS_ERROR

    @property
    def blocks_run(self) -> bool:
        """True if this item should prevent simulation from starting."""
        return self.status == STATUS_ERROR


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
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


def _str(v: Any, default: str = "") -> str:
    return str(v).strip() if v is not None else default


# ─────────────────────────────────────────────────────────────────────────────
# Group validator functions
# Each function: (ui_values: dict) -> (status, title, detail, field_hints)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_agent_simulation(v: dict) -> tuple:
    """
    Group: Agent & Simulation parameters.

    Errors
    ------
    - num_agents <= 0
    - total_steps <= 0
    - opinion_layers <= 0

    Warnings
    --------
    - num_agents > 2000  (performance warning)
    - total_steps > 10000  (long run warning)
    - seed == 0  (non-determinism hint)
    """
    num_agents    = _int(v.get("num_agents"), 0)
    total_steps   = _int(v.get("total_steps"), 0)
    opinion_layers = _int(v.get("opinion_layers"), 0)

    hints: list[tuple[str, str]] = []

    # ── Errors ────────────────────────────────────────────────────────────────
    if num_agents <= 0:
        hints.append(("num_agents", "必须大于 0"))
    if total_steps <= 0:
        hints.append(("total_steps", "必须大于 0"))
    if opinion_layers <= 0:
        hints.append(("opinion_layers", "必须大于 0"))

    if any(s == STATUS_ERROR for s in
           [("num_agents" in [h[0] for h in hints]),
            ("total_steps" in [h[0] for h in hints])]):
        pass  # will be caught below

    error_keys = {h[0] for h in hints}
    if error_keys:
        detail = "、".join(
            {"num_agents": "智能体数", "total_steps": "总步数",
             "opinion_layers": "意见层数"}.get(k, k)
            for k in sorted(error_keys)
        ) + " 配置无效"
        return STATUS_ERROR, "智能体与仿真参数", detail, hints

    # ── Warnings ──────────────────────────────────────────────────────────────
    warn_parts: list[str] = []
    if num_agents > 2000:
        hints.append(("num_agents", f"{num_agents} 个智能体可能导致运行缓慢"))
        warn_parts.append(f"{num_agents} 个智能体（性能警告）")
    if total_steps > 10000:
        hints.append(("total_steps", f"{total_steps} 步运行时间较长"))
        warn_parts.append(f"{total_steps} 步（耗时较长）")
    if _int(v.get("seed"), -1) == 0:
        hints.append(("seed", "seed=0 为有效种子，但建议使用非零值"))
        warn_parts.append("seed=0")

    if warn_parts:
        detail = "；".join(warn_parts[:2])  # max 2 in detail line
        return STATUS_WARN, "智能体与仿真参数", detail, hints

    # ── OK ────────────────────────────────────────────────────────────────────
    detail = (
        f"{num_agents} 个智能体 · {opinion_layers} 层意见 · {total_steps} 步"
    )
    return STATUS_OK, "智能体与仿真参数", detail, []


def _validate_dynamics(v: dict) -> tuple:
    """
    Group: Dynamics parameters.

    Errors
    ------
    - epsilon_base not in (0, 1]
    - mu_base not in (0, 1]

    Warnings
    --------
    - alpha_mod > 1.5  (strong event amplification)
    - beta_mod > 1.5   (strong decay modulation)
    - epsilon_base < 0.05  (very narrow tolerance, may cause isolation)
    """
    epsilon = _float(v.get("epsilon_base"), -1.0)
    mu      = _float(v.get("mu_base"),      -1.0)
    alpha   = _float(v.get("alpha_mod"),     0.0)
    beta    = _float(v.get("beta_mod"),      0.0)

    hints: list[tuple[str, str]] = []

    # ── Errors ────────────────────────────────────────────────────────────────
    if not (0 < epsilon <= 1.0):
        hints.append(("epsilon_base", f"需在 (0, 1]，当前 {epsilon:.3f}"))
    if not (0 < mu <= 1.0):
        hints.append(("mu_base", f"需在 (0, 1]，当前 {mu:.3f}"))

    if hints:
        detail = "ε 或 μ 超出有效范围"
        return STATUS_ERROR, "动力学参数", detail, hints

    # ── Warnings ──────────────────────────────────────────────────────────────
    warn_parts: list[str] = []
    if alpha > 1.5:
        hints.append(("alpha_mod", f"α={alpha:.2f} 较大，事件影响可能过强"))
        warn_parts.append(f"α={alpha:.2f} 较大")
    if beta > 1.5:
        hints.append(("beta_mod", f"β={beta:.2f} 较大，衰减调制较强"))
        warn_parts.append(f"β={beta:.2f} 较大")
    if epsilon < 0.05:
        hints.append(("epsilon_base",
                       f"ε={epsilon:.3f} 极小，智能体可能快速孤立"))
        warn_parts.append(f"ε={epsilon:.3f} 极小")

    if warn_parts:
        detail = "；".join(warn_parts[:2])
        return STATUS_WARN, "动力学参数", detail, hints

    backfire = _bool(v.get("backfire"), False)
    detail = f"ε={epsilon:.3f} · μ={mu:.3f}" + ("  · 回火效应已启用" if backfire else "")
    return STATUS_OK, "动力学参数", detail, []


def _validate_network_spatial(v: dict) -> tuple:
    """
    Group: Network + Spatial distribution.

    Errors
    ------
    - sw_k >= num_agents  (small-world impossible)
    - sf_m < 1

    Warnings
    --------
    - sw_p == 0  (no rewiring, regular lattice)
    - sw_p == 1  (full rewiring, equivalent to random)
    - n_clusters > num_agents / 5  (too many clusters for agents)
    """
    net_type   = _str(v.get("net_type"), "small_world")
    num_agents = _int(v.get("num_agents"), 150)
    sw_k       = _int(v.get("sw_k"), 6)
    sw_p       = _float(v.get("sw_p"), 0.1)
    sf_m       = _int(v.get("sf_m"), 3)
    n_clusters = _int(v.get("n_clusters"), 4)

    hints: list[tuple[str, str]] = []

    # ── Errors ────────────────────────────────────────────────────────────────
    if net_type == "small_world" and sw_k >= num_agents:
        hints.append(("sw_k",
                       f"k={sw_k} ≥ 智能体数 {num_agents}，小世界网络无法构建"))
    if net_type == "scale_free" and sf_m < 1:
        hints.append(("sf_m", "m 必须 ≥ 1"))

    if hints:
        return STATUS_ERROR, "网络与空间配置", hints[0][1], hints

    # ── Warnings ──────────────────────────────────────────────────────────────
    warn_parts: list[str] = []
    if net_type == "small_world":
        if sw_p == 0.0:
            hints.append(("sw_p", "p=0 为规则格子，无随机长程连接"))
            warn_parts.append("p=0（规则格子）")
        elif sw_p == 1.0:
            hints.append(("sw_p", "p=1 等同于随机网络"))
            warn_parts.append("p=1（随机网络）")

    if n_clusters > max(1, num_agents // 5):
        hints.append(("n_clusters",
                       f"{n_clusters} 个簇对 {num_agents} 个智能体可能过多"))
        warn_parts.append(f"{n_clusters} 个空间簇偏多")

    if warn_parts:
        detail = "；".join(warn_parts[:2])
        return STATUS_WARN, "网络与空间配置", detail, hints

    spatial_type = _str(v.get("spatial_type"), "clustered")
    detail = f"{net_type} 网络 · {spatial_type} 空间分布"
    return STATUS_OK, "网络与空间配置", detail, []


def _validate_events(v: dict) -> tuple:
    """
    Group: Event generators.

    Errors
    ------
    - All four event types disabled simultaneously

    Warnings
    --------
    - exo_enabled but exo_lambda == 0  (enabled but rate=0)
    - cascade_enabled but cascade_bg_lambda == 0 and cascade_mu_mult == 0
    - online_enabled but online_convergence == 0 and online_conflict == 0
    """
    exo_en     = _bool(v.get("exo_enabled"),     True)
    endo_en    = _bool(v.get("endo_enabled"),     True)
    cascade_en = _bool(v.get("cascade_enabled"),  True)
    online_en  = _bool(v.get("online_enabled"),   True)

    hints: list[tuple[str, str]] = []

    # ── Errors ────────────────────────────────────────────────────────────────
    if not any([exo_en, endo_en, cascade_en, online_en]):
        hints.append(("exo_enabled", "至少需要启用一类事件生成器"))
        return STATUS_ERROR, "事件配置", "所有事件生成器均已禁用", hints

    # ── Warnings ──────────────────────────────────────────────────────────────
    warn_parts: list[str] = []

    if exo_en and _float(v.get("exo_lambda"), -1.0) == 0.0:
        hints.append(("exo_lambda", "已启用但 λ=0，不会生成外生事件"))
        warn_parts.append("外生事件 λ=0")

    if cascade_en:
        mu_mult = _float(v.get("cascade_mu_mult"), 0.6)
        bg_lam  = _float(v.get("cascade_bg_lambda"), 0.0)
        if mu_mult == 0.0 and bg_lam == 0.0:
            hints.append(("cascade_mu_mult", "μ 乘数和背景率均为 0，级联不会触发"))
            warn_parts.append("级联事件参数为零")

    if online_en:
        conv = _float(v.get("online_convergence"), 0.01)
        conf = _float(v.get("online_conflict"),    0.01)
        if conv == 0.0 and conf == 0.0:
            hints.append(("online_convergence", "收敛与冲突阈值均为 0，在线共鸣不会触发"))
            warn_parts.append("在线共鸣阈值为零")

    if warn_parts:
        detail = "；".join(warn_parts[:2])
        return STATUS_WARN, "事件配置", detail, hints

    enabled_names = []
    if exo_en:     enabled_names.append("外生")
    if endo_en:    enabled_names.append("内生阈值")
    if cascade_en: enabled_names.append("级联")
    if online_en:  enabled_names.append("在线共鸣")
    detail = " · ".join(enabled_names) + " 已启用"
    return STATUS_OK, "事件配置", detail, []


def _validate_intervention(v: dict) -> tuple:
    """
    Group: Intervention rules.

    This validator inspects the intervention rule data embedded in ui_values
    under the key "intervention_rules" (list[dict]) if present, or falls back
    to checking individual rule component keys injected by collect_rules().

    Errors
    ------
    (none — interventions are fully optional)

    Warnings
    --------
    - Any rule's step > total_steps
    - Any rule's layer >= opinion_layers
    - Duplicate steps across rules
    """
    total_steps    = _int(v.get("total_steps"), 500)
    opinion_layers = _int(v.get("opinion_layers"), 3)

    # Accept pre-parsed rules list or an empty fallback
    rules: list[dict] = v.get("intervention_rules") or []

    hints: list[tuple[str, str]] = []

    if not rules:
        return STATUS_OK, "干预配置", "未添加干预规则（可选）", []

    # ── Warnings ──────────────────────────────────────────────────────────────
    warn_parts: list[str] = []
    steps_seen: list[int] = []

    for i, rule in enumerate(rules):
        rule_step  = _int(rule.get("step"),  1)
        rule_layer = _int(rule.get("layer"), 0)
        label      = f"规则 {i + 1}"

        if rule_step > total_steps:
            hints.append((f"intervention_step_{i}",
                           f"{label}：步骤 {rule_step} 超出总步数 {total_steps}"))
            warn_parts.append(f"{label} 步骤越界")

        if rule_layer >= opinion_layers:
            hints.append((f"intervention_layer_{i}",
                           f"{label}：层 {rule_layer} 超出范围（共 {opinion_layers} 层）"))
            warn_parts.append(f"{label} 层越界")

        if rule_step in steps_seen:
            hints.append((f"intervention_step_{i}",
                           f"{label}：步骤 {rule_step} 与其他规则重复"))
            warn_parts.append(f"{label} 步骤重复")
        steps_seen.append(rule_step)

    if warn_parts:
        detail = "；".join(warn_parts[:2])
        if len(warn_parts) > 2:
            detail += f" 等 {len(warn_parts)} 项"
        return STATUS_WARN, "干预配置", detail, hints

    detail = f"{len(rules)} 条规则，步骤范围合法"
    return STATUS_OK, "干预配置", detail, []


def _validate_analysis_config(v: dict) -> tuple:
    """
    Group: Analysis output configuration.

    Errors
    ------
    - output_dir is empty string

    Warnings
    --------
    - ai_enabled but api_key is empty
    - layer_idx >= opinion_layers
    """
    output_dir     = _str(v.get("output_dir"), "")
    ai_enabled     = _bool(v.get("ai_enabled"), False)
    api_key        = _str(v.get("api_key"), "")
    layer_idx      = _int(v.get("layer_idx"), 0)
    opinion_layers = _int(v.get("opinion_layers"), 3)

    hints: list[tuple[str, str]] = []

    # ── Errors ────────────────────────────────────────────────────────────────
    if not output_dir:
        hints.append(("output_dir", "输出目录不能为空"))
        return STATUS_ERROR, "分析与输出配置", "输出目录未填写", hints

    # ── Warnings ──────────────────────────────────────────────────────────────
    warn_parts: list[str] = []

    if ai_enabled and not api_key:
        hints.append(("api_key", "已启用 AI 解析但未填写 API Key"))
        warn_parts.append("AI 解析未配置 Key")

    if layer_idx >= opinion_layers:
        hints.append(("layer_idx",
                       f"主意见层 {layer_idx} 超出范围（共 {opinion_layers} 层）"))
        warn_parts.append(f"主意见层 {layer_idx} 越界")

    if warn_parts:
        detail = "；".join(warn_parts[:2])
        return STATUS_WARN, "分析与输出配置", detail, hints

    lang   = _str(v.get("output_lang"), "zh")
    detail = f"输出至 {output_dir}  · 语言: {lang}"
    return STATUS_OK, "分析与输出配置", detail, []


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

# (group_key, validator_fn, target_page)
_VALIDATORS: list[tuple[str, Callable, str]] = [
    ("agent_simulation",  _validate_agent_simulation,  "model_config"),
    ("dynamics",          _validate_dynamics,           "model_config"),
    ("network_spatial",   _validate_network_spatial,    "model_config"),
    ("events",            _validate_events,             "model_config"),
    ("intervention",      _validate_intervention,       "intervention"),
    ("analysis_config",   _validate_analysis_config,    "analysis_config"),
]

_VALIDATOR_MAP: dict[str, tuple[Callable, str]] = {
    key: (fn, page) for key, fn, page in _VALIDATORS
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_all(ui_values: dict[str, Any]) -> list[CheckItem]:
    """
    Run all validators against the current UI parameter dict.

    Parameters
    ----------
    ui_values : dict
        Flat dict of {component_key: value} as produced by the param panel.
        Optionally contains "intervention_rules": list[dict] for pre-parsed
        intervention data; if absent, intervention validator returns OK.

    Returns
    -------
    list[CheckItem]
        One item per group, in registry order (top-to-bottom in the checklist).
    """
    results: list[CheckItem] = []

    for group_key, validator_fn, target_page in _VALIDATORS:
        try:
            status, title, detail, field_hints = validator_fn(ui_values)
        except Exception as exc:
            # Defensive: a crashing validator should not block the UI
            status      = STATUS_WARN
            title       = group_key
            detail      = f"校验时发生意外错误：{exc}"
            field_hints = []

        results.append(CheckItem(
            group       = group_key,
            status      = status,
            title       = title,
            detail      = detail,
            target_page = target_page,
            field_hints = field_hints,
        ))

    return results


def validate_group(group: str, ui_values: dict[str, Any]) -> CheckItem:
    """
    Validate a single configuration group by key.

    Useful for partial re-validation when only one page's values change,
    avoiding the cost of running all validators.

    Parameters
    ----------
    group : str
        One of the group keys in the registry.
    ui_values : dict
        Full flat parameter dict (all keys available, only the relevant
        subset will be read by the group's validator).

    Returns
    -------
    CheckItem

    Raises
    ------
    KeyError
        If group key is not registered.
    """
    if group not in _VALIDATOR_MAP:
        raise KeyError(f"Unknown validation group: {group!r}. "
                       f"Available: {list(_VALIDATOR_MAP)}")

    validator_fn, target_page = _VALIDATOR_MAP[group]

    try:
        status, title, detail, field_hints = validator_fn(ui_values)
    except Exception as exc:
        status      = STATUS_WARN
        title       = group
        detail      = f"校验时发生意外错误：{exc}"
        field_hints = []

    return CheckItem(
        group       = group,
        status      = status,
        title       = title,
        detail      = detail,
        target_page = target_page,
        field_hints = field_hints,
    )


def summarize(items: list[CheckItem]) -> dict[str, Any]:
    """
    Summarise a list of CheckItems into aggregate counts.

    Returns
    -------
    dict with keys:
        n_ok    : int
        n_warn  : int
        n_error : int
        blocks_run : bool   — True if any item is STATUS_ERROR
        progress   : float  — fraction of OK items (0.0–1.0)
        label_zh   : str    — human-readable summary in Chinese
    """
    n_ok    = sum(1 for it in items if it.is_ok)
    n_warn  = sum(1 for it in items if it.is_warn)
    n_error = sum(1 for it in items if it.is_error)
    total   = len(items)

    progress   = n_ok / total if total else 0.0
    blocks_run = n_error > 0

    if n_error:
        label_zh = f"{n_error} 项错误 · {n_warn} 项警告 · {n_ok} 项通过"
    elif n_warn:
        label_zh = f"{n_warn} 项警告 · {n_ok} 项通过"
    else:
        label_zh = f"全部 {n_ok} 项通过"

    return {
        "n_ok":      n_ok,
        "n_warn":    n_warn,
        "n_error":   n_error,
        "blocks_run": blocks_run,
        "progress":  progress,
        "label_zh":  label_zh,
    }
