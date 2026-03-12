"""
core/defaults.py

Central registry of all simulation parameter defaults and UI labels.
All default values are extracted from test.py::_make_config() and
analysis/manager.py::_DEFAULT_CONFIG.

Supports bilingual labels: "zh" (Chinese) and "en" (English).
"""

from __future__ import annotations
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Default parameter values
# Keys correspond 1:1 to component keys in param_panel.py
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    # Agent & Simulation
    "num_agents":           150,
    "opinion_layers":       3,
    "total_steps":          500,
    "seed":                 42,
    "record_history":       True,
    "init_type":            "polarized",
    "init_split":           0.5,
    # Dynamics
    "epsilon_base":         0.25,
    "mu_base":              0.35,
    "alpha_mod":            0.25,
    "beta_mod":             0.15,
    "backfire":             False,
    # Influence Field
    "field_alpha":          6.0,
    "field_beta":           0.08,
    "temporal_window":      100.0,
    # Topology
    "topo_threshold":       0.3,
    "radius_base":          0.06,
    "radius_dynamic":       0.15,
    # Network
    "net_type":             "small_world",
    "sw_k":                 6,
    "sw_p":                 0.1,
    "sf_m":                 3,
    # Spatial Distribution
    "spatial_type":         "clustered",
    "n_clusters":           4,
    "cluster_std":          0.1,
    # Exogenous Events
    "exo_enabled":          True,
    "exo_seed":             43,
    "exo_lambda":           0.25,
    "exo_intensity_shape":  2.5,
    "exo_intensity_min":    4.0,
    "exo_polarity_min":     -0.5,
    "exo_polarity_max":     0.5,
    "exo_concentration":    "1,1,1",
    # Endogenous Threshold Events
    "endo_enabled":         True,
    "endo_seed":            44,
    "endo_monitor":         "opinion_extremism",
    "endo_threshold":       0.12,
    "endo_grid":            8,
    "endo_min_agents":      2,
    "endo_cooldown":        5,
    "endo_base_intensity":  8.0,
    "endo_scale":           4.0,
    # Cascade Events
    "cascade_enabled":      True,
    "cascade_seed":         45,
    "cascade_bg_lambda":    0.0,
    "cascade_mu_mult":      0.6,
    "cascade_decay":        0.5,
    # Online Resonance Events
    "online_enabled":       True,
    "online_seed":          46,
    "online_check":         2,
    "online_smooth":        4,
    "online_convergence":   0.01,
    "online_conflict":      0.01,
    "online_min_community": 3,
    "online_base":          4.0,
    "online_scale":         8.0,
    # Analysis Output
    "output_dir":           "output/run_001",
    "output_lang":          "zh",
    "layer_idx":            0,
    "include_trends":       True,
    "save_timeseries":      True,
    "save_features_json":   True,
    "refresh_every":        10,
}

# ─────────────────────────────────────────────────────────────────────────────
# Bilingual UI labels
# LABELS[lang][key] = (label_text, info_text)
# ─────────────────────────────────────────────────────────────────────────────

LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "en": {
        # Section headings
        "_sec_agent":           ("Agent & Simulation",          ""),
        "_sec_dynamics":        ("Dynamics",                    ""),
        "_sec_field":           ("Influence Field",             ""),
        "_sec_topo":            ("Topology",                    ""),
        "_sec_network":         ("Network",                     ""),
        "_sec_spatial":         ("Spatial Distribution",        ""),
        "_sec_exo":             ("Exogenous Events",            ""),
        "_sec_endo":            ("Endogenous Threshold Events", ""),
        "_sec_cascade":         ("Cascade Events",              ""),
        "_sec_online":          ("Online Resonance Events",     ""),
        "_sec_analysis":        ("Analysis Output",             ""),
        "_sec_preset":          ("Scenario Presets",            ""),
        "_sec_intervention":    ("Intervention Rules",          ""),
        # Agent & Simulation
        "num_agents":           ("Agent Count",       "Total number of agents in the simulation"),
        "opinion_layers":       ("Opinion Layers",    "Number of opinion dimensions per agent"),
        "total_steps":          ("Total Steps",       "Number of simulation steps to execute"),
        "seed":                 ("Random Seed",       "Master seed for reproducibility"),
        "record_history":       ("Record History",    "Store full timeseries (increases memory usage)"),
        "init_type":            ("Initial Opinions",  "Distribution type for agent opinion initialization"),
        "init_split":           ("Polarization Split","Fraction in each camp (polarized mode only)"),
        # Dynamics
        "epsilon_base":         ("Tolerance ε",       "Bounded confidence threshold — agents interact only within ε"),
        "mu_base":              ("Influence μ",        "Opinion update rate per interaction"),
        "alpha_mod":            ("α Modulation",      "Event amplitude scaling coefficient"),
        "beta_mod":             ("β Modulation",      "Decay rate modulation coefficient"),
        "backfire":             ("Backfire Effect",    "Enable opinion backfire on strong disagreement"),
        # Influence Field
        "field_alpha":          ("Spatial Decay α",   "Rate of spatial impact field decay"),
        "field_beta":           ("Temporal Decay β",  "Rate of temporal impact field decay"),
        "temporal_window":      ("Temporal Window",   "Time window for impact accumulation"),
        # Topology
        "topo_threshold":       ("Similarity Thr.",   "Minimum similarity for interaction eligibility"),
        "radius_base":          ("Base Radius",       "Fixed interaction radius"),
        "radius_dynamic":       ("Dynamic Radius",    "Additional radius from event-driven expansion"),
        # Network
        "net_type":             ("Network Type",      "Social network topology"),
        "sw_k":                 ("Neighbors k",       "Initial nearest neighbors (small-world)"),
        "sw_p":                 ("Rewire Prob p",      "Edge rewiring probability (small-world)"),
        "sf_m":                 ("Edges m",           "New edges per node (scale-free)"),
        # Spatial
        "spatial_type":         ("Distribution",      "Agent spatial placement pattern"),
        "n_clusters":           ("Cluster Count",     "Number of spatial clusters"),
        "cluster_std":          ("Cluster Spread",    "Standard deviation within each cluster"),
        # Exogenous Events
        "exo_enabled":          ("Enable",            "Enable exogenous (external) event generation"),
        "exo_seed":             ("Seed",              "Random seed for exogenous generator"),
        "exo_lambda":           ("Poisson Rate λ",    "Mean events per time unit (Poisson process)"),
        "exo_intensity_shape":  ("Pareto Shape",      "Shape parameter of intensity distribution"),
        "exo_intensity_min":    ("Min Intensity",     "Minimum event intensity value"),
        "exo_polarity_min":     ("Polarity Min",      "Lower bound of event polarity"),
        "exo_polarity_max":     ("Polarity Max",      "Upper bound of event polarity"),
        "exo_concentration":    ("Dirichlet Conc.",   "Comma-separated concentration params (length = opinion layers)"),
        # Endogenous Threshold
        "endo_enabled":         ("Enable",            "Enable endogenous threshold-triggered events"),
        "endo_seed":            ("Seed",              "Random seed for threshold generator"),
        "endo_monitor":         ("Monitor Attribute", "Spatial attribute to monitor for threshold crossing"),
        "endo_threshold":       ("Critical Threshold","Threshold value that triggers event generation"),
        "endo_grid":            ("Grid Resolution",   "NxN spatial grid for attribute monitoring"),
        "endo_min_agents":      ("Min Agents/Cell",   "Minimum agents in cell to allow triggering"),
        "endo_cooldown":        ("Cooldown Steps",    "Minimum steps between events from same cell"),
        "endo_base_intensity":  ("Base Intensity",    "Baseline event intensity"),
        "endo_scale":           ("Intensity Scale",   "Scaling factor applied to base intensity"),
        # Cascade
        "cascade_enabled":      ("Enable",            "Enable Hawkes-process cascade events"),
        "cascade_seed":         ("Seed",              "Random seed for cascade generator"),
        "cascade_bg_lambda":    ("Background λ",      "Background rate (0 = purely reactive)"),
        "cascade_mu_mult":      ("μ Multiplier",      "Branching ratio for cascade propagation"),
        "cascade_decay":        ("Cascade Decay",     "Intensity decay rate for child events"),
        # Online Resonance
        "online_enabled":       ("Enable",            "Enable community-resonance event generation"),
        "online_seed":          ("Seed",              "Random seed for online resonance generator"),
        "online_check":         ("Check Interval",    "Steps between community state evaluations"),
        "online_smooth":        ("Smoothing Window",  "Moving-average window for smoothing metrics"),
        "online_convergence":   ("Convergence Thr.",  "Threshold to detect opinion convergence"),
        "online_conflict":      ("Conflict Thr.",     "Threshold to detect opinion conflict"),
        "online_min_community": ("Min Community",     "Minimum community size for event eligibility"),
        "online_base":          ("Base Intensity",    "Baseline resonance event intensity"),
        "online_scale":         ("Size Scale",        "Community-size scaling factor"),
        # Analysis
        "output_dir":           ("Output Directory",  "Root directory for all saved outputs"),
        "output_lang":          ("Report Language",   "Language for generated text reports"),
        "layer_idx":            ("Primary Layer",     "Opinion layer index used for scalar metrics"),
        "include_trends":       ("Include Trends",    "Add trend metrics to feature summary"),
        "save_timeseries":      ("Save Timeseries",   "Save full timeseries to .npz file"),
        "save_features_json":   ("Save Features JSON","Save extracted features as JSON"),
        "refresh_every":        ("Refresh Every N",   "Chart refresh interval (steps) during simulation"),
        # UI chrome
        "_btn_run":             ("▶  Run Simulation", ""),
        "_btn_stop":            ("⏹  Stop",           ""),
        "_btn_reset":           ("↺  Reset",          ""),
        "_btn_load_preset":     ("Load",              ""),
        "_btn_export_yaml":     ("Export YAML",       ""),
        "_tab_monitor":         ("Monitor",           ""),
        "_tab_analysis":        ("Analysis",          ""),
        "_tab_report":          ("Report",            ""),
        "_preset_label":        ("Preset",            "Load a scenario preset"),
        "_lang_label":          ("Interface Language",""),
        "_status_ready":        ("● Ready",           ""),
        "_status_running":      ("⚙ Running",         ""),
        "_status_done":         ("✓ Complete",        ""),
        "_status_stopped":      ("■ Stopped",         ""),
        "_step_label_fmt":      ("Step {step}/{total}  ·  Time {time:.2f}  ·  Events {events}", ""),
    },
    "zh": {
        # Section headings
        "_sec_agent":           ("智能体与仿真",          ""),
        "_sec_dynamics":        ("动力学参数",            ""),
        "_sec_field":           ("影响力场",              ""),
        "_sec_topo":            ("拓扑参数",              ""),
        "_sec_network":         ("网络配置",              ""),
        "_sec_spatial":         ("空间分布",              ""),
        "_sec_exo":             ("外生事件",              ""),
        "_sec_endo":            ("内生阈值事件",           ""),
        "_sec_cascade":         ("级联事件",              ""),
        "_sec_online":          ("在线共鸣事件",           ""),
        "_sec_analysis":        ("分析输出配置",           ""),
        "_sec_preset":          ("场景预设",              ""),
        "_sec_intervention":    ("干预规则",              ""),
        # Agent & Simulation
        "num_agents":           ("智能体数量",   "仿真中的智能体总数"),
        "opinion_layers":       ("意见层数",     "每个智能体的意见维度数"),
        "total_steps":          ("总步数",       "仿真执行的总步数"),
        "seed":                 ("随机种子",     "用于可复现性的主随机种子"),
        "record_history":       ("记录历史",     "保存完整时间序列（会增加内存占用）"),
        "init_type":            ("初始意见分布", "智能体意见的初始化分布类型"),
        "init_split":           ("极化分裂比例", "每阵营比例（仅 polarized 模式有效）"),
        # Dynamics
        "epsilon_base":         ("容忍度 ε",    "有界信任阈值"),
        "mu_base":              ("影响强度 μ",  "每次交互的意见更新幅度"),
        "alpha_mod":            ("α 调制系数",  "事件振幅缩放系数"),
        "beta_mod":             ("β 调制系数",  "衰减率调制系数"),
        "backfire":             ("回火效应",    "在强烈不同意时启用意见回火"),
        # Influence Field
        "field_alpha":          ("空间衰减 α",  "影响力场的空间衰减率"),
        "field_beta":           ("时间衰减 β",  "影响力场的时间衰减率"),
        "temporal_window":      ("时间窗口",    "影响力累积的时间窗口"),
        # Topology
        "topo_threshold":       ("相似度阈值",  "允许交互的最低相似度"),
        "radius_base":          ("基础半径",    "固定交互半径"),
        "radius_dynamic":       ("动态半径",    "事件驱动的额外扩展半径"),
        # Network
        "net_type":             ("网络类型",    "社交网络拓扑结构"),
        "sw_k":                 ("近邻数 k",    "初始最近邻数（小世界）"),
        "sw_p":                 ("重连概率 p",  "边重连概率（小世界）"),
        "sf_m":                 ("边数 m",      "每个新节点的边数（无标度）"),
        # Spatial
        "spatial_type":         ("分布类型",    "智能体空间分布模式"),
        "n_clusters":           ("簇数量",      "空间簇的数量"),
        "cluster_std":          ("簇内标准差",  "每个簇内的标准差"),
        # Exogenous Events
        "exo_enabled":          ("启用",        "启用外生（外部）事件生成"),
        "exo_seed":             ("随机种子",    "外生生成器的随机种子"),
        "exo_lambda":           ("泊松率 λ",    "每单位时间平均事件数"),
        "exo_intensity_shape":  ("Pareto 形状", "强度分布的形状参数"),
        "exo_intensity_min":    ("最小强度",    "事件强度最小值"),
        "exo_polarity_min":     ("极性下限",    "事件极性的下界"),
        "exo_polarity_max":     ("极性上限",    "事件极性的上界"),
        "exo_concentration":    ("Dirichlet 浓度","逗号分隔浓度参数（长度=意见层数）"),
        # Endogenous Threshold
        "endo_enabled":         ("启用",        "启用内生阈值触发事件"),
        "endo_seed":            ("随机种子",    "阈值生成器的随机种子"),
        "endo_monitor":         ("监测属性",    "用于监测阈值穿越的空间属性"),
        "endo_threshold":       ("临界阈值",    "触发事件生成的阈值"),
        "endo_grid":            ("网格分辨率",  "用于属性监测的 NxN 空间网格"),
        "endo_min_agents":      ("最少智能体数","允许触发的单元最少智能体数"),
        "endo_cooldown":        ("冷却步数",    "同一单元两次触发之间的最少步数"),
        "endo_base_intensity":  ("基础强度",    "事件基础强度"),
        "endo_scale":           ("强度缩放",    "应用于基础强度的缩放因子"),
        # Cascade
        "cascade_enabled":      ("启用",        "启用 Hawkes 过程级联事件"),
        "cascade_seed":         ("随机种子",    "级联生成器的随机种子"),
        "cascade_bg_lambda":    ("背景 λ",      "背景率（0 = 纯反应式）"),
        "cascade_mu_mult":      ("μ 乘数",      "级联传播的分支比"),
        "cascade_decay":        ("级联衰减",    "子事件强度衰减率"),
        # Online Resonance
        "online_enabled":       ("启用",        "启用社区共鸣事件生成"),
        "online_seed":          ("随机种子",    "在线共鸣生成器的随机种子"),
        "online_check":         ("检测间隔",    "两次社区状态评估之间的步数"),
        "online_smooth":        ("平滑窗口",    "指标平滑的移动平均窗口"),
        "online_convergence":   ("收敛阈值",    "检测意见收敛的阈值"),
        "online_conflict":      ("冲突阈值",    "检测意见冲突的阈值"),
        "online_min_community": ("最小社区规模","可触发事件的社区最小规模"),
        "online_base":          ("基础强度",    "共鸣事件基础强度"),
        "online_scale":         ("规模缩放",    "社区规模缩放因子"),
        # Analysis
        "output_dir":           ("输出目录",    "所有输出文件的根目录"),
        "output_lang":          ("报告语言",    "生成文本报告的语言"),
        "layer_idx":            ("主要意见层",  "用于标量指标的意见层索引"),
        "include_trends":       ("包含趋势指标","在特征摘要中添加趋势指标"),
        "save_timeseries":      ("保存时间序列","将完整时间序列保存为 .npz"),
        "save_features_json":   ("保存特征 JSON","将提取的特征保存为 JSON"),
        "refresh_every":        ("刷新步间隔",  "仿真期间图表刷新步数间隔"),
        # UI chrome
        "_btn_run":             ("▶  运行仿真",  ""),
        "_btn_stop":            ("⏹  停止",     ""),
        "_btn_reset":           ("↺  重置",     ""),
        "_btn_load_preset":     ("加载",        ""),
        "_btn_export_yaml":     ("导出 YAML",   ""),
        "_tab_monitor":         ("实时监控",    ""),
        "_tab_analysis":        ("结果分析",    ""),
        "_tab_report":          ("报告导出",    ""),
        "_preset_label":        ("场景预设",    "加载仿真场景预设"),
        "_lang_label":          ("界面语言",    ""),
        "_status_ready":        ("● 就绪",      ""),
        "_status_running":      ("⚙ 运行中",    ""),
        "_status_done":         ("✓ 完成",      ""),
        "_status_stopped":      ("■ 已停止",    ""),
        "_step_label_fmt":      ("步骤 {step}/{total}  ·  时间 {time:.2f}  ·  事件 {events}", ""),
    },
}


def get_label(key: str, lang: str = "en") -> str:
    """Return the UI label for a given key and language."""
    return LABELS.get(lang, LABELS["en"]).get(key, (key, ""))[0]


def get_info(key: str, lang: str = "en") -> str:
    """Return the info/tooltip text for a given key and language."""
    return LABELS.get(lang, LABELS["en"]).get(key, ("", ""))[1]


def get_label_and_info(key: str, lang: str = "en") -> tuple[str, str]:
    """Return (label, info) tuple for a given key and language."""
    return LABELS.get(lang, LABELS["en"]).get(key, (key, ""))


# ─────────────────────────────────────────────────────────────────────────────
# Dropdown choice lists
# ─────────────────────────────────────────────────────────────────────────────

CHOICES: dict[str, list] = {
    "init_type":    ["polarized", "uniform", "random", "clustered"],
    "net_type":     ["small_world", "random", "scale_free", "complete"],
    "spatial_type": ["clustered", "uniform", "grid", "ring"],
    "endo_monitor": ["opinion_extremism", "opinion_variance", "density"],
    "output_lang":  ["zh", "en"],
}


def get_choices(key: str) -> list:
    """Return dropdown choices for a given parameter key."""
    return CHOICES.get(key, [])
