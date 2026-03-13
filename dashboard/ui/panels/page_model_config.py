"""
ui/panels/page_model_config.py

P3 模型配置 — 从 param_panel.py 拆分而来

包含 4 组 Accordion，覆盖原 param_panel.py 中的模型相关参数：
  1. 智能体与仿真基础（原 Basic / Simulation 区域）
  2. 动力学参数     （原 Dynamics 区域）
  3. 网络与空间     （原 Network + Spatial 区域）
  4. 事件配置       （原 Events 区域）

分析输出相关参数（output_dir, ai_enabled 等）已移至 page_analysis_config.py。
干预规则已移至 page_intervention.py。

Layout
------
  Page header
  ┌── Accordion: 智能体与仿真 ──────────────────────────┐
  │  num_agents  total_steps  opinion_layers  seed      │
  └────────────────────────────────────────────────────┘
  ┌── Accordion: 动力学参数 ────────────────────────────┐
  │  epsilon_base  mu_base  alpha_mod  beta_mod         │
  │  backfire  delta_decay  gamma_field                 │
  └────────────────────────────────────────────────────┘
  ┌── Accordion: 网络与空间 ────────────────────────────┐
  │  net_type  sw_k  sw_p  sf_m  n_clusters            │
  │  spatial_type  cluster_spread                       │
  └────────────────────────────────────────────────────┘
  ┌── Accordion: 事件配置 ──────────────────────────────┐
  │  exo_enabled  exo_lambda  exo_magnitude             │
  │  endo_enabled  endo_threshold  endo_radius          │
  │  cascade_enabled  cascade_bg_lambda  cascade_mu_mult│
  │  online_enabled  online_convergence  online_conflict│
  └────────────────────────────────────────────────────┘
  [sticky bottom bar: ← 返回 | 保存并继续 →]

Public API
----------
ModelConfigComponents
    Dataclass with all gr.Components.  .param_components is the flat
    {key: component} dict consumed by sidebar status binding and
    validator.validate_all().

build_model_config_page(lang, defaults) -> ModelConfigComponents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gradio as gr


# ─── Default values ───────────────────────────────────────────────────────────
# Mirrors core/defaults.py — kept here so page is independently runnable.
# In production, pass defaults=config_bridge.get_defaults() from app.py.

_DEFAULTS: dict[str, Any] = {
    # Agent / simulation
    "num_agents":    150,
    "total_steps":   500,
    "opinion_layers": 3,
    "seed":           42,

    # Dynamics
    "epsilon_base":  0.25,
    "mu_base":       0.35,
    "alpha_mod":     0.25,
    "beta_mod":      0.15,
    "backfire":      False,
    "delta_decay":   0.01,
    "gamma_field":   0.05,

    # Network
    "net_type":      "small_world",   # "small_world" | "scale_free" | "random"
    "sw_k":           6,
    "sw_p":           0.1,
    "sf_m":           3,

    # Spatial
    "n_clusters":     4,
    "spatial_type":  "clustered",     # "clustered" | "random" | "ring"
    "cluster_spread": 0.3,

    # Events
    "exo_enabled":         True,
    "exo_lambda":          0.25,
    "exo_magnitude":       0.3,
    "endo_enabled":        True,
    "endo_threshold":      0.6,
    "endo_radius":         0.15,
    "cascade_enabled":     True,
    "cascade_bg_lambda":   0.05,
    "cascade_mu_mult":     0.6,
    "online_enabled":      True,
    "online_convergence":  0.02,
    "online_conflict":     0.015,
}

# ─── Localised strings ────────────────────────────────────────────────────────

_STRINGS = {
    "zh": {
        "page_title":    "模型配置",
        "page_subtitle": "配置仿真的核心参数。所有参数均可在实验启动前随时修改。",
        "sec_agent":     "智能体与仿真基础",
        "sec_dynamics":  "动力学参数",
        "sec_network":   "网络与空间分布",
        "sec_events":    "事件配置",
        # Agent
        "num_agents":    "智能体数量",
        "total_steps":   "仿真总步数",
        "opinion_layers":"意见层数",
        "seed":          "随机种子",
        # Dynamics
        "epsilon_base":  "信任阈值 ε",
        "mu_base":       "更新速率 μ",
        "alpha_mod":     "事件增益 α",
        "beta_mod":      "衰减调制 β",
        "backfire":      "启用回火效应",
        "delta_decay":   "场衰减 δ",
        "gamma_field":   "场强 γ",
        # Network
        "net_type":      "网络类型",
        "sw_k":          "SW 近邻数 k",
        "sw_p":          "SW 重连概率 p",
        "sf_m":          "SF 新连接数 m",
        "n_clusters":    "空间簇数",
        "spatial_type":  "空间分布类型",
        "cluster_spread":"簇内扩散度",
        # Events
        "exo_enabled":        "启用外生事件",
        "exo_lambda":         "外生事件率 λ",
        "exo_magnitude":      "事件幅度",
        "endo_enabled":       "启用内生阈值事件",
        "endo_threshold":     "极化触发阈值",
        "endo_radius":        "影响半径",
        "cascade_enabled":    "启用 Hawkes 级联",
        "cascade_bg_lambda":  "背景率 λ₀",
        "cascade_mu_mult":    "激励乘数 μ",
        "online_enabled":     "启用在线共鸣",
        "online_convergence": "收敛触发阈值",
        "online_conflict":    "冲突触发阈值",
        # Nav
        "back_btn":      "← 返回",
        "next_btn":      "保存并继续 →",
        # Net type choices
        "net_choices": [
            ("小世界网络", "small_world"),
            ("无标度网络", "scale_free"),
            ("随机网络",   "random"),
        ],
        "spatial_choices": [
            ("聚类分布", "clustered"),
            ("随机分布", "random"),
            ("环形分布", "ring"),
        ],
    },
    "en": {
        "page_title":    "Model Config",
        "page_subtitle": "Core simulation parameters. All values can be changed before running.",
        "sec_agent":     "Agents & Simulation",
        "sec_dynamics":  "Dynamics Parameters",
        "sec_network":   "Network & Spatial",
        "sec_events":    "Event Configuration",
        # Agent
        "num_agents":    "Number of agents",
        "total_steps":   "Total steps",
        "opinion_layers":"Opinion layers",
        "seed":          "Random seed",
        # Dynamics
        "epsilon_base":  "Trust threshold ε",
        "mu_base":       "Update rate μ",
        "alpha_mod":     "Event gain α",
        "beta_mod":      "Decay modulation β",
        "backfire":      "Enable backfire effect",
        "delta_decay":   "Field decay δ",
        "gamma_field":   "Field strength γ",
        # Network
        "net_type":      "Network type",
        "sw_k":          "SW neighbours k",
        "sw_p":          "SW rewire prob p",
        "sf_m":          "SF new edges m",
        "n_clusters":    "Spatial clusters",
        "spatial_type":  "Spatial layout",
        "cluster_spread":"Cluster spread",
        # Events
        "exo_enabled":        "Enable exogenous events",
        "exo_lambda":         "Exogenous rate λ",
        "exo_magnitude":      "Event magnitude",
        "endo_enabled":       "Enable endogenous threshold",
        "endo_threshold":     "Polarisation trigger",
        "endo_radius":        "Influence radius",
        "cascade_enabled":    "Enable Hawkes cascade",
        "cascade_bg_lambda":  "Background rate λ₀",
        "cascade_mu_mult":    "Excitation multiplier μ",
        "online_enabled":     "Enable online resonance",
        "online_convergence": "Convergence threshold",
        "online_conflict":    "Conflict threshold",
        # Nav
        "back_btn":      "← Back",
        "next_btn":      "Save & Continue →",
        # Choices
        "net_choices": [
            ("Small world", "small_world"),
            ("Scale-free",  "scale_free"),
            ("Random",      "random"),
        ],
        "spatial_choices": [
            ("Clustered", "clustered"),
            ("Random",    "random"),
            ("Ring",      "ring"),
        ],
    },
}


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class ModelConfigComponents:
    """All gr.Components from build_model_config_page()."""

    # Agent / simulation
    num_agents:    gr.Number
    total_steps:   gr.Number
    opinion_layers: gr.Number
    seed:          gr.Number

    # Dynamics
    epsilon_base:  gr.Number
    mu_base:       gr.Number
    alpha_mod:     gr.Number
    beta_mod:      gr.Number
    backfire:      gr.Checkbox
    delta_decay:   gr.Number
    gamma_field:   gr.Number

    # Network
    net_type:      gr.Dropdown
    sw_k:          gr.Number
    sw_p:          gr.Number
    sf_m:          gr.Number

    # Spatial
    n_clusters:    gr.Number
    spatial_type:  gr.Dropdown
    cluster_spread: gr.Number

    # Events
    exo_enabled:        gr.Checkbox
    exo_lambda:         gr.Number
    exo_magnitude:      gr.Number
    endo_enabled:       gr.Checkbox
    endo_threshold:     gr.Number
    endo_radius:        gr.Number
    cascade_enabled:    gr.Checkbox
    cascade_bg_lambda:  gr.Number
    cascade_mu_mult:    gr.Number
    online_enabled:     gr.Checkbox
    online_convergence: gr.Number
    online_conflict:    gr.Number

    # Nav buttons
    back_btn: gr.Button
    next_btn: gr.Button

    @property
    def param_components(self) -> dict[str, gr.Component]:
        """
        Flat dict {key: component} for all model parameters.
        Consumed by sidebar status binding and validator.
        Keys match _DEFAULTS and core/config_bridge.py.
        """
        return {
            "num_agents":    self.num_agents,
            "total_steps":   self.total_steps,
            "opinion_layers": self.opinion_layers,
            "seed":          self.seed,
            "epsilon_base":  self.epsilon_base,
            "mu_base":       self.mu_base,
            "alpha_mod":     self.alpha_mod,
            "beta_mod":      self.beta_mod,
            "backfire":      self.backfire,
            "delta_decay":   self.delta_decay,
            "gamma_field":   self.gamma_field,
            "net_type":      self.net_type,
            "sw_k":          self.sw_k,
            "sw_p":          self.sw_p,
            "sf_m":          self.sf_m,
            "n_clusters":    self.n_clusters,
            "spatial_type":  self.spatial_type,
            "cluster_spread": self.cluster_spread,
            "exo_enabled":        self.exo_enabled,
            "exo_lambda":         self.exo_lambda,
            "exo_magnitude":      self.exo_magnitude,
            "endo_enabled":       self.endo_enabled,
            "endo_threshold":     self.endo_threshold,
            "endo_radius":        self.endo_radius,
            "cascade_enabled":    self.cascade_enabled,
            "cascade_bg_lambda":  self.cascade_bg_lambda,
            "cascade_mu_mult":    self.cascade_mu_mult,
            "online_enabled":     self.online_enabled,
            "online_convergence": self.online_convergence,
            "online_conflict":    self.online_conflict,
        }


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_model_config_page(
    lang:     str = "zh",
    defaults: dict[str, Any] | None = None,
) -> ModelConfigComponents:
    """
    Render the model config page.

    Must be called inside an active gr.Blocks() + gr.Group() context.

    Parameters
    ----------
    lang : {"zh", "en"}
    defaults : dict | None
        Overrides for default parameter values.

    Returns
    -------
    ModelConfigComponents
    """
    s  = _STRINGS[lang]
    dv = {**_DEFAULTS, **(defaults or {})}

    with gr.Column(
        elem_id      = "page-model-config",
        elem_classes = "page-body",
    ):
        # ── Page header ───────────────────────────────────────────────────
        gr.HTML(f"""
<div class="page-header" style="padding:20px 0 16px;">
  <div class="page-title">{s['page_title']}</div>
  <div class="page-subtitle">{s['page_subtitle']}</div>
</div>""")

        # ── Accordion 1: Agents & Simulation ──────────────────────────────
        with gr.Accordion(s["sec_agent"], open=True, elem_classes="model-accordion"):
            with gr.Row():
                num_agents = gr.Number(
                    label=s["num_agents"], value=dv["num_agents"],
                    minimum=1, maximum=10000, step=1, precision=0,
                    interactive=True, scale=1,
                )
                total_steps = gr.Number(
                    label=s["total_steps"], value=dv["total_steps"],
                    minimum=1, maximum=100000, step=1, precision=0,
                    interactive=True, scale=1,
                )
            with gr.Row():
                opinion_layers = gr.Number(
                    label=s["opinion_layers"], value=dv["opinion_layers"],
                    minimum=1, maximum=10, step=1, precision=0,
                    interactive=True, scale=1,
                )
                seed = gr.Number(
                    label=s["seed"], value=dv["seed"],
                    minimum=0, step=1, precision=0,
                    interactive=True, scale=1,
                )

        # ── Accordion 2: Dynamics ─────────────────────────────────────────
        with gr.Accordion(s["sec_dynamics"], open=False, elem_classes="model-accordion"):
            with gr.Row():
                epsilon_base = gr.Number(
                    label=s["epsilon_base"], value=dv["epsilon_base"],
                    minimum=0.001, maximum=1.0, step=0.01,
                    interactive=True, scale=1,
                )
                mu_base = gr.Number(
                    label=s["mu_base"], value=dv["mu_base"],
                    minimum=0.001, maximum=1.0, step=0.01,
                    interactive=True, scale=1,
                )
            with gr.Row():
                alpha_mod = gr.Number(
                    label=s["alpha_mod"], value=dv["alpha_mod"],
                    minimum=0.0, maximum=5.0, step=0.05,
                    interactive=True, scale=1,
                )
                beta_mod = gr.Number(
                    label=s["beta_mod"], value=dv["beta_mod"],
                    minimum=0.0, maximum=5.0, step=0.05,
                    interactive=True, scale=1,
                )
            with gr.Row():
                delta_decay = gr.Number(
                    label=s["delta_decay"], value=dv["delta_decay"],
                    minimum=0.0, maximum=1.0, step=0.001,
                    interactive=True, scale=1,
                )
                gamma_field = gr.Number(
                    label=s["gamma_field"], value=dv["gamma_field"],
                    minimum=0.0, maximum=1.0, step=0.005,
                    interactive=True, scale=1,
                )
            backfire = gr.Checkbox(
                label=s["backfire"], value=dv["backfire"],
                interactive=True,
            )

        # ── Accordion 3: Network & Spatial ────────────────────────────────
        with gr.Accordion(s["sec_network"], open=False, elem_classes="model-accordion"):
            with gr.Row():
                net_type = gr.Dropdown(
                    label=s["net_type"],
                    choices=s["net_choices"],
                    value=dv["net_type"],
                    interactive=True,
                    scale=1,
                )
                spatial_type = gr.Dropdown(
                    label=s["spatial_type"],
                    choices=s["spatial_choices"],
                    value=dv["spatial_type"],
                    interactive=True,
                    scale=1,
                )
            with gr.Row():
                sw_k = gr.Number(
                    label=s["sw_k"], value=dv["sw_k"],
                    minimum=2, maximum=50, step=1, precision=0,
                    interactive=True, scale=1,
                )
                sw_p = gr.Number(
                    label=s["sw_p"], value=dv["sw_p"],
                    minimum=0.0, maximum=1.0, step=0.01,
                    interactive=True, scale=1,
                )
                sf_m = gr.Number(
                    label=s["sf_m"], value=dv["sf_m"],
                    minimum=1, maximum=20, step=1, precision=0,
                    interactive=True, scale=1,
                )
            with gr.Row():
                n_clusters = gr.Number(
                    label=s["n_clusters"], value=dv["n_clusters"],
                    minimum=1, maximum=50, step=1, precision=0,
                    interactive=True, scale=1,
                )
                cluster_spread = gr.Number(
                    label=s["cluster_spread"], value=dv["cluster_spread"],
                    minimum=0.01, maximum=1.0, step=0.01,
                    interactive=True, scale=1,
                )

        # ── Accordion 4: Events ───────────────────────────────────────────
        with gr.Accordion(s["sec_events"], open=False, elem_classes="model-accordion"):

            # Exogenous
            gr.HTML('<div style="font-size:11px;color:#64748b;margin:4px 0 6px;'
                    'text-transform:uppercase;letter-spacing:.05em;">外生事件</div>')
            with gr.Row():
                exo_enabled = gr.Checkbox(
                    label=s["exo_enabled"], value=dv["exo_enabled"],
                    interactive=True, scale=1,
                )
            with gr.Row():
                exo_lambda = gr.Number(
                    label=s["exo_lambda"], value=dv["exo_lambda"],
                    minimum=0.0, maximum=10.0, step=0.01,
                    interactive=True, scale=1,
                )
                exo_magnitude = gr.Number(
                    label=s["exo_magnitude"], value=dv["exo_magnitude"],
                    minimum=0.0, maximum=1.0, step=0.01,
                    interactive=True, scale=1,
                )

            # Endogenous
            gr.HTML('<div style="font-size:11px;color:#64748b;margin:10px 0 6px;'
                    'text-transform:uppercase;letter-spacing:.05em;">内生阈值事件</div>')
            with gr.Row():
                endo_enabled = gr.Checkbox(
                    label=s["endo_enabled"], value=dv["endo_enabled"],
                    interactive=True, scale=1,
                )
            with gr.Row():
                endo_threshold = gr.Number(
                    label=s["endo_threshold"], value=dv["endo_threshold"],
                    minimum=0.0, maximum=1.0, step=0.01,
                    interactive=True, scale=1,
                )
                endo_radius = gr.Number(
                    label=s["endo_radius"], value=dv["endo_radius"],
                    minimum=0.0, maximum=1.0, step=0.01,
                    interactive=True, scale=1,
                )

            # Hawkes cascade
            gr.HTML('<div style="font-size:11px;color:#64748b;margin:10px 0 6px;'
                    'text-transform:uppercase;letter-spacing:.05em;">Hawkes 级联</div>')
            with gr.Row():
                cascade_enabled = gr.Checkbox(
                    label=s["cascade_enabled"], value=dv["cascade_enabled"],
                    interactive=True, scale=1,
                )
            with gr.Row():
                cascade_bg_lambda = gr.Number(
                    label=s["cascade_bg_lambda"], value=dv["cascade_bg_lambda"],
                    minimum=0.0, maximum=5.0, step=0.01,
                    interactive=True, scale=1,
                )
                cascade_mu_mult = gr.Number(
                    label=s["cascade_mu_mult"], value=dv["cascade_mu_mult"],
                    minimum=0.0, maximum=5.0, step=0.01,
                    interactive=True, scale=1,
                )

            # Online resonance
            gr.HTML('<div style="font-size:11px;color:#64748b;margin:10px 0 6px;'
                    'text-transform:uppercase;letter-spacing:.05em;">在线共鸣</div>')
            with gr.Row():
                online_enabled = gr.Checkbox(
                    label=s["online_enabled"], value=dv["online_enabled"],
                    interactive=True, scale=1,
                )
            with gr.Row():
                online_convergence = gr.Number(
                    label=s["online_convergence"], value=dv["online_convergence"],
                    minimum=0.0, maximum=1.0, step=0.001,
                    interactive=True, scale=1,
                )
                online_conflict = gr.Number(
                    label=s["online_conflict"], value=dv["online_conflict"],
                    minimum=0.0, maximum=1.0, step=0.001,
                    interactive=True, scale=1,
                )

        # ── Sticky action bar ─────────────────────────────────────────────
        with gr.Row(elem_classes="page-action-bar"):
            back_btn = gr.Button(
                s["back_btn"],
                elem_id      = "btn-model-back",
                elem_classes = "btn-secondary",
                size         = "sm",
            )
            next_btn = gr.Button(
                s["next_btn"],
                elem_id      = "btn-model-next",
                elem_classes = "btn-primary-teal",
                size         = "sm",
            )

    return ModelConfigComponents(
        num_agents=num_agents, total_steps=total_steps,
        opinion_layers=opinion_layers, seed=seed,
        epsilon_base=epsilon_base, mu_base=mu_base,
        alpha_mod=alpha_mod, beta_mod=beta_mod,
        backfire=backfire, delta_decay=delta_decay, gamma_field=gamma_field,
        net_type=net_type, sw_k=sw_k, sw_p=sw_p, sf_m=sf_m,
        n_clusters=n_clusters, spatial_type=spatial_type,
        cluster_spread=cluster_spread,
        exo_enabled=exo_enabled, exo_lambda=exo_lambda,
        exo_magnitude=exo_magnitude,
        endo_enabled=endo_enabled, endo_threshold=endo_threshold,
        endo_radius=endo_radius,
        cascade_enabled=cascade_enabled,
        cascade_bg_lambda=cascade_bg_lambda, cascade_mu_mult=cascade_mu_mult,
        online_enabled=online_enabled,
        online_convergence=online_convergence, online_conflict=online_conflict,
        back_btn=back_btn, next_btn=next_btn,
    )
