"""
analysis/parser/themes.py

Simulation Theme Engine
-----------------------
Maps simulation feature patterns onto real-world narrative "themes"
(e.g., concert crowd, public safety incident, political rally).

Two operational modes:
    1. AUTO  — infer the best-fit theme from feature summary statistics.
    2. RANDOM — sample a theme at random (useful for creative/exploratory analysis).

Each theme provides:
    - A scene description injected into AI prompts.
    - Role mappings (agents → crowd members / citizens / voters …).
    - Domain-specific interpretation hints for each metric family.
    - A thematic system prompt override (appended to SYSTEM_PROMPT in client.py).

Usage
-----
    from analysis.parser.themes import ThemeEngine, get_theme_prompt_override

    engine = ThemeEngine()

    # Auto-detect from pipeline summary
    theme = engine.detect(pipeline_output["summary"])
    print(theme.name)           # e.g. "concert_crowd"
    print(theme.scene_desc)     # natural language scene context

    # Inject into a section prompt
    override = get_theme_prompt_override(theme, lang="zh", fmt="md")

    # Random theme
    random_theme = engine.random_theme()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# Theme Definition
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Theme:
    """
    A narrative theme that contextualises simulation metrics.

    Attributes
    ----------
    name : str
        Internal identifier (snake_case). Used as dict key.
    label_zh : str
        Display name in Chinese.
    label_en : str
        Display name in English.
    scene_desc_zh : str
        One-paragraph scene description injected into Chinese prompts.
    scene_desc_en : str
        One-paragraph scene description injected into English prompts.
    agent_role_zh : str
        What agents represent in this theme (Chinese). e.g. "观众"
    agent_role_en : str
        What agents represent in this theme (English). e.g. "concertgoers"
    opinion_interp : Dict[str, str]
        Maps opinion metric names to domain-specific interpretations.
        e.g. {"polarization_std": "Crowd emotional spread"}
    spatial_interp : Dict[str, str]
        Domain-specific interpretations for spatial metrics.
    event_interp : Dict[str, str]
        Domain-specific interpretations for event metrics.
    detection_hints : Dict[str, Any]
        Feature thresholds used by ThemeEngine.detect() to score this theme.
        Keys are flat summary keys (e.g. "opinion.polarization_std.mean").
        Values are (low, high) tuples — theme scores +1 if value is in range.
    """
    name: str
    label_zh: str
    label_en: str
    scene_desc_zh: str
    scene_desc_en: str
    agent_role_zh: str
    agent_role_en: str
    opinion_interp: Dict[str, str] = field(default_factory=dict)
    spatial_interp: Dict[str, str] = field(default_factory=dict)
    event_interp: Dict[str, str] = field(default_factory=dict)
    detection_hints: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Theme Catalogue
# ═════════════════════════════════════════════════════════════════════════════

_THEME_CATALOGUE: List[Theme] = [

    # ── 1. Concert Crowd ─────────────────────────────────────────────────────
    Theme(
        name="concert_crowd",
        label_zh="演唱会人群",
        label_en="Concert Crowd",
        scene_desc_zh=(
            "这是一场大型露天演唱会现场。数千名观众聚集在表演区周围，"
            "他们的情绪状态随着音乐节奏、灯光效果和周围人群的反应实时变化。"
            "外部事件（如高潮曲目、突发烟花）引发情绪波动并迅速在人群中扩散。"
            "空间聚集和情绪同步是该场景的核心动态。"
        ),
        scene_desc_en=(
            "A large open-air concert venue. Thousands of attendees cluster around "
            "the performance area; their emotional states shift in real time with "
            "the music tempo, lighting, and reactions of those nearby. "
            "External triggers (climactic songs, pyrotechnics) produce emotional surges "
            "that cascade rapidly through the crowd. "
            "Spatial clustering and emotional synchronisation are the defining dynamics."
        ),
        agent_role_zh="观众",
        agent_role_en="concertgoers",
        opinion_interp={
            "mean_opinion":           "人群整体情绪水平（0=冷漠, 1=狂热）",
            "polarization_std":       "观众情绪分化程度——高值表示部分人狂热、部分人冷静",
            "bimodality_coefficient": "情绪是否形成两极（狂热群体 vs 旁观者）",
            "extreme_share":          "高度投入观众的比例",
            "opinion_entropy":        "情绪多样性——低值表示全场同步",
        },
        spatial_interp={
            "nearest_neighbor_index": "人群聚集密度——值<1表示形成小圈子",
            "moran_i":                "相邻观众情绪相关性——高值反映情绪传染效应",
            "radius_of_gyration":     "人群扩散范围——收缩表示向舞台聚拢",
        },
        event_interp={
            "burstiness":     "爆发性事件（如安可、返场）的节奏",
            "temporal_gini":  "高潮时刻的集中度",
            "intensity_mean": "事件对情绪的平均冲击强度",
        },
        detection_hints={
            # Concert: high spatial clustering, moderate-high opinion mean, bursty events
            "spatial.nearest_neighbor_index.mean": (0.0, 0.85),
            "opinion.mean_opinion.mean":           (0.45, 0.85),
            "event.burstiness.mean":               (0.2, 1.0),
            "spatial.moran_i.mean":                (0.2, 1.0),
        },
    ),

    # ── 2. Public Safety Incident ─────────────────────────────────────────────
    Theme(
        name="public_safety_incident",
        label_zh="治安突发事件",
        label_en="Public Safety Incident",
        scene_desc_zh=(
            "城市公共空间中发生了一起突发治安事件（如群体冲突、紧急疏散或抗议集会）。"
            "居民和路人的行为模式受到恐惧、信息传播和应急响应的多重影响。"
            "事件以高强度短暂爆发为特征，随后人群出现明显的空间重组和意见分化。"
            "部分人群聚集围观，另一部分迅速疏散，形成显著的空间极化。"
        ),
        scene_desc_en=(
            "A sudden public safety incident in an urban space — a crowd confrontation, "
            "emergency evacuation, or protest gathering. Residents and bystanders are "
            "shaped by fear, information cascades, and emergency response. "
            "The incident is characterised by a high-intensity burst followed by rapid "
            "spatial reorganisation and opinion divergence: some agents converge on the "
            "scene while others disperse, producing strong spatial polarisation."
        ),
        agent_role_zh="市民 / 目击者",
        agent_role_en="citizens / bystanders",
        opinion_interp={
            "mean_opinion":           "整体社会安全感（0=极度恐慌, 1=平静）",
            "polarization_std":       "恐慌情绪的分化——分值高表示社会情绪撕裂",
            "bimodality_coefficient": "是否形成恐慌群体与稳定群体的双峰分布",
            "extreme_share":          "处于极端恐慌或极端平静状态的人口比例",
            "opinion_entropy":        "信息混乱程度——高熵表示谣言与真相并存",
        },
        spatial_interp={
            "nearest_neighbor_index": "人群聚集或逃散程度",
            "moran_i":                "恐慌情绪的空间传染性",
            "radius_of_gyration":     "人群疏散半径——增大表示逃散行为扩展",
            "centroid_x":             "人群重心偏移方向（可能指向疏散通道）",
        },
        event_interp={
            "burstiness":     "突发事件的冲击性（高值=单次强烈冲击）",
            "temporal_gini":  "事件集中度——高值表示短时间内爆发大量事件",
            "intensity_max":  "单次最高冲击强度（关键事件的严重程度）",
        },
        detection_hints={
            # Safety incident: high polarization, high burstiness, high intensity
            "opinion.polarization_std.mean":       (0.18, 0.5),
            "event.burstiness.mean":               (0.35, 1.0),
            "event.intensity_max.mean":            (0.5, 1.0),
            "opinion.bimodality_coefficient.mean": (0.4, 2.0),
        },
    ),

    # ── 3. Political Rally / Election ────────────────────────────────────────
    Theme(
        name="political_rally",
        label_zh="政治集会 / 选举",
        label_en="Political Rally / Election",
        scene_desc_zh=(
            "选举季或政治集会背景下，选民在社交网络中形成意见阵营。"
            "媒体事件、候选人辩论和社交媒体热点构成外部冲击，"
            "推动意见从多元分布向两极对立演化。"
            "网络同质性（支持者抱团）和跨阵营对话的缺乏是关键结构特征。"
        ),
        scene_desc_en=(
            "An election season or political rally context in which voters form "
            "opinion camps across a social network. Media events, candidate debates, "
            "and social-media trending topics act as exogenous shocks that push "
            "opinion distributions from diversity toward bimodal polarisation. "
            "Network homophily (supporters clustering together) and the scarcity of "
            "cross-camp dialogue are the defining structural features."
        ),
        agent_role_zh="选民 / 支持者",
        agent_role_en="voters / supporters",
        opinion_interp={
            "mean_opinion":           "整体民意倾向（0=反对派, 1=支持派）",
            "polarization_std":       "政治极化程度——高值表示社会严重撕裂",
            "bimodality_coefficient": "是否形成泾渭分明的两个阵营（BC>0.555为强极化）",
            "extreme_share":          "极端立场支持者的比例",
            "edge_homophily_score":   "社交网络同质性——高值表示信息茧房严重",
        },
        spatial_interp={
            "moran_i":                "政治立场的地理聚集性（选区分化）",
            "nearest_neighbor_index": "支持者的地理集聚程度",
        },
        event_interp={
            "burstiness":     "舆论爆发节奏（辩论、丑闻等关键事件的冲击模式）",
            "intensity_mean": "媒体事件对民意的平均影响强度",
            "temporal_gini":  "舆论事件的时间集中度",
        },
        detection_hints={
            # Political: strong bimodality, high homophily, moderate burstiness
            "opinion.bimodality_coefficient.mean": (0.45, 2.0),
            "opinion.polarization_std.mean":       (0.15, 0.5),
            "opinion.edge_homophily_score.mean":   (0.55, 1.0),
        },
    ),

    # ── 4. Social Media Cascade ───────────────────────────────────────────────
    Theme(
        name="social_media_cascade",
        label_zh="社交媒体舆论级联",
        label_en="Social Media Cascade",
        scene_desc_zh=(
            "一个热点话题在社交媒体平台上迅速扩散。"
            "信息通过网络结构级联传播，少数意见领袖节点主导舆论走向。"
            "外部事件（爆料、辟谣、平台算法推送）不断扰动意见生态，"
            "形成典型的先快速极化、后部分回归的动态模式。"
        ),
        scene_desc_en=(
            "A trending topic spreads rapidly across a social media platform. "
            "Information cascades through network structure; a small number of "
            "opinion-leader hubs dominate the narrative trajectory. "
            "Exogenous shocks (leaks, rebuttals, algorithmic amplification) "
            "continuously perturb the opinion ecosystem, producing the signature "
            "pattern of rapid initial polarisation followed by partial reversion."
        ),
        agent_role_zh="网络用户 / 意见节点",
        agent_role_en="users / opinion nodes",
        opinion_interp={
            "mean_opinion":           "整体舆论倾向",
            "opinion_entropy":        "信息多样性——低熵表示一边倒的舆论场",
            "polarization_std":       "舆论分化烈度",
            "bimodality_coefficient": "是否形成支持与反对的双极结构",
            "extreme_share":          "极端言论用户的比例",
        },
        spatial_interp={
            "moran_i":         "相邻用户观点一致性（社区内回声室效应）",
            "spatial_entropy": "用户地理分布的均匀程度",
        },
        event_interp={
            "burstiness":     "级联爆发的突发性（高值=病毒式传播）",
            "temporal_gini":  "传播时间集中度——高值表示短暂爆发后迅速消退",
            "intensity_max":  "单条内容的最大传播冲击力",
            "event_rate":     "信息事件频率（内容发布密度）",
        },
        detection_hints={
            # Social media: very bursty, high event rate, hub-dominated (high degree_gini)
            "event.burstiness.mean":      (0.3, 1.0),
            "event.event_rate.mean":      (0.5, 1.0),
            "topo.degree_gini.mean":      (0.3, 1.0),
            "opinion.opinion_entropy.mean": (0.0, 1.8),
        },
    ),

    # ── 5. Community Governance ───────────────────────────────────────────────
    Theme(
        name="community_governance",
        label_zh="社区治理与公众参与",
        label_en="Community Governance",
        scene_desc_zh=(
            "一个城市社区正在就重大公共议题（如城市规划、环境政策）进行集体决策。"
            "居民通过邻里网络和公众论坛交换意见，意见逐渐在反复商议中趋于收敛。"
            "外部干预（政策公告、专家咨询）间歇性改变意见格局。"
            "该场景的典型特征是缓慢的意见整合与持续的小范围分歧共存。"
        ),
        scene_desc_en=(
            "An urban community deliberating on a major public issue — urban planning, "
            "environmental policy, neighbourhood governance. Residents exchange views "
            "through neighbourhood networks and public forums; opinions gradually "
            "converge through repeated deliberation. External interventions "
            "(policy announcements, expert consultations) intermittently reshape the "
            "landscape. The signature pattern is slow opinion integration coexisting "
            "with persistent minority dissent."
        ),
        agent_role_zh="社区居民 / 利益相关方",
        agent_role_en="residents / stakeholders",
        opinion_interp={
            "mean_opinion":           "社区整体共识倾向",
            "polarization_std":       "意见分散程度——随时间下降表示达成共识",
            "opinion_entropy":        "意见多样性——治理过程中熵的下降反映共识形成",
            "bimodality_coefficient": "是否存在顽固的少数反对派",
            "edge_homophily_score":   "邻里间意见的相似程度",
        },
        spatial_interp={
            "moran_i":                "意见的地理聚集——高值表示区域间存在明显立场差异",
            "nearest_neighbor_index": "居民互动的空间密度",
            "radius_of_gyration":     "意见影响的地理扩散范围",
        },
        event_interp={
            "burstiness":     "政策冲击的规律性（低值=有序推进，高值=突发干预）",
            "intensity_mean": "外部干预的平均影响强度",
            "event_rate":     "公众事件的发生频率",
        },
        detection_hints={
            # Governance: low polarization trend (converging), low burstiness, moderate clustering
            "opinion.polarization_std.trend_slope":  (-1.0, -0.01),
            "event.burstiness.mean":                 (-1.0, 0.2),
            "topo.average_clustering.mean":          (0.2, 1.0),
        },
    ),

    # ── 6. Epidemic / Rumour Spread ───────────────────────────────────────────
    Theme(
        name="epidemic_rumour",
        label_zh="流行病 / 谣言传播",
        label_en="Epidemic / Rumour Spread",
        scene_desc_zh=(
            "一种观念、恐慌或谣言正在人群中以类似传染病的方式扩散。"
            "初始阶段感染者（持极端意见者）比例极低，随后出现指数级增长，"
            "最终因群体免疫（持怀疑态度者形成防火墙）或外部辟谣而平息。"
            "传播网络的结构（是否存在超级传播节点）决定最终扩散范围。"
        ),
        scene_desc_en=(
            "A belief, panic, or rumour spreading through a population in a manner "
            "analogous to an epidemic. The initial share of 'infected' agents "
            "(extreme opinion holders) is tiny; exponential growth follows until "
            "herd resistance (sceptics forming a firewall) or external debunking "
            "halts propagation. Network structure — particularly the presence of "
            "super-spreader hubs — determines the ultimate reach."
        ),
        agent_role_zh="易感者 / 传播者 / 免疫者",
        agent_role_en="susceptible / spreading / immune agents",
        opinion_interp={
            "mean_opinion":           "谣言/观念的整体渗透率",
            "extreme_share":          "已被'感染'（持极端意见）的人口比例——对应流行病的感染率",
            "polarization_std":       "传播过程中的意见分化——峰值对应传播高峰期",
            "bimodality_coefficient": "感染者与未感染者的双峰分离程度",
            "opinion_entropy":        "传播初期熵高（混沌），收敛后熵低",
        },
        spatial_interp={
            "moran_i":                "传播的空间聚集性——高值表示存在传播中心",
            "nearest_neighbor_index": "传播者的空间密度",
        },
        event_interp={
            "burstiness":     "传播爆发的冲击性",
            "event_rate":     "传播事件的频率（对应R0有效繁殖数）",
            "intensity_max":  "最强传播事件的冲击力（超级传播者事件）",
        },
        detection_hints={
            # Epidemic: rapidly growing extreme_share, high spatial autocorrelation
            "opinion.extreme_share.trend_slope":  (0.01, 1.0),
            "spatial.moran_i.mean":               (0.15, 1.0),
            "event.event_rate.mean":              (0.3, 1.0),
        },
    ),

    # ── 7. Financial Market ───────────────────────────────────────────────────
    Theme(
        name="financial_market",
        label_zh="金融市场情绪",
        label_en="Financial Market Sentiment",
        scene_desc_zh=(
            "金融市场中，投资者的情绪（看涨/看跌预期）在价格信号和新闻事件的驱动下动态演化。"
            "羊群效应推动意见快速同向聚集，而少数逆向投资者构成分歧来源。"
            "市场崩盘或急涨对应高强度的突发事件，随后形成新的均衡或持续震荡。"
        ),
        scene_desc_en=(
            "In financial markets, investor sentiment (bullish/bearish expectations) "
            "evolves dynamically driven by price signals and news events. "
            "Herding behaviour drives rapid homogenisation of opinions; "
            "contrarian investors provide the primary source of disagreement. "
            "Market crashes or surges correspond to high-intensity exogenous shocks, "
            "after which the system finds a new equilibrium or enters sustained oscillation."
        ),
        agent_role_zh="投资者 / 交易者",
        agent_role_en="investors / traders",
        opinion_interp={
            "mean_opinion":           "市场整体情绪（0=极度悲观, 1=极度乐观）",
            "polarization_std":       "多空分歧程度",
            "bimodality_coefficient": "是否形成牛市派与熊市派的对立",
            "opinion_entropy":        "市场信息多样性——低熵表示共识形成（可能是泡沫前兆）",
            "extreme_share":          "极端仓位（满仓做多或做空）投资者比例",
        },
        spatial_interp={
            "moran_i": "情绪的机构聚集性（同类机构倾向相似仓位）",
        },
        event_interp={
            "burstiness":     "市场波动的爆发性（高值=闪崩/急拉风险）",
            "intensity_max":  "最大单次冲击（黑天鹅事件烈度）",
            "temporal_gini":  "波动的时间集中度",
            "event_rate":     "市场事件（财报、政策）的发生频率",
        },
        detection_hints={
            # Market: high volatility in opinions, high event intensity
            "opinion.opinion_variance.volatility": (0.01, 1.0),
            "event.intensity_max.mean":            (0.4, 1.0),
            "event.burstiness.mean":               (0.1, 1.0),
        },
    ),
]

# Build lookup dict for fast access
THEME_REGISTRY: Dict[str, Theme] = {t.name: t for t in _THEME_CATALOGUE}


# ═════════════════════════════════════════════════════════════════════════════
# ThemeEngine
# ═════════════════════════════════════════════════════════════════════════════

class ThemeEngine:
    """
    Detects or samples the most appropriate narrative theme for a simulation run.

    Parameters
    ----------
    catalogue : list[Theme], optional
        Custom theme list. Defaults to the built-in _THEME_CATALOGUE.
    score_threshold : float
        Minimum detection score to accept auto-detection.
        If no theme scores above this, falls back to "generic".
    """

    def __init__(
        self,
        catalogue: Optional[List[Theme]] = None,
        score_threshold: float = 1.0,
    ):
        self._themes = catalogue or _THEME_CATALOGUE
        self._threshold = score_threshold

    # ── Detection ────────────────────────────────────────────────────────────

    def detect(self, summary: Dict[str, Any]) -> Theme:
        """
        Infer the best-fit theme from pipeline summary statistics.

        Scoring:
            For each theme, iterate its detection_hints.
            If summary[key]["mean"] (or direct float) falls within (low, high),
            add 1 to the theme's score.
            The theme with the highest score wins.

        Parameters
        ----------
        summary : dict
            Output of summarize_timeseries() — keys like "opinion.polarization_std",
            values are dicts with "mean", "std", "trend_slope", etc.

        Returns
        -------
        Theme  — best-fit theme (or generic fallback).
        """
        scores: Dict[str, float] = {t.name: 0.0 for t in self._themes}

        for theme in self._themes:
            for hint_key, (lo, hi) in theme.detection_hints.items():
                val = _extract_hint_value(summary, hint_key)
                if val is not None and lo <= val <= hi:
                    scores[theme.name] += 1.0

        best_name = max(scores, key=lambda k: scores[k])
        best_score = scores[best_name]

        if best_score < self._threshold:
            return _generic_theme()

        return THEME_REGISTRY.get(best_name, _generic_theme())

    def rank_themes(self, summary: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Return all themes ranked by detection score (descending).

        Useful for debugging or when multiple themes are plausible.

        Returns
        -------
        list of (theme_name, score) tuples.
        """
        scores: Dict[str, float] = {}
        for theme in self._themes:
            score = 0.0
            for hint_key, (lo, hi) in theme.detection_hints.items():
                val = _extract_hint_value(summary, hint_key)
                if val is not None and lo <= val <= hi:
                    score += 1.0
            scores[theme.name] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── Random Sampling ───────────────────────────────────────────────────────

    def random_theme(self, seed: Optional[int] = None) -> Theme:
        """
        Return a uniformly random theme from the catalogue.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Theme
        """
        rng = random.Random(seed)
        return rng.choice(self._themes)

    # ── Convenience ───────────────────────────────────────────────────────────

    def get(self, name: str) -> Theme:
        """Retrieve a theme by name. Raises KeyError if not found."""
        if name not in THEME_REGISTRY:
            raise KeyError(
                f"Theme '{name}' not found. "
                f"Available: {list(THEME_REGISTRY.keys())}"
            )
        return THEME_REGISTRY[name]

    def list_themes(self) -> List[str]:
        """Return all registered theme names."""
        return [t.name for t in self._themes]


# ═════════════════════════════════════════════════════════════════════════════
# Prompt Construction Helpers
# ═════════════════════════════════════════════════════════════════════════════

def get_theme_prompt_override(
    theme: Theme,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """
    Build the theme context block injected into section prompts.

    The returned string is prepended to any section prompt to give the LLM
    a concrete real-world frame of reference.

    Parameters
    ----------
    theme : Theme
    lang  : "zh" | "en"
    fmt   : "md" | "html" | "latex"

    Returns
    -------
    str — formatted context block ready for prompt injection.
    """
    if lang == "zh":
        scene   = theme.scene_desc_zh
        label   = theme.label_zh
        role    = theme.agent_role_zh
        op_map  = _format_interp_block(theme.opinion_interp,  "意见指标映射", fmt)
        sp_map  = _format_interp_block(theme.spatial_interp,  "空间指标映射", fmt)
        ev_map  = _format_interp_block(theme.event_interp,    "事件指标映射", fmt)
        heading = "## 场景背景"
        role_lbl = "智能体角色"
    else:
        scene   = theme.scene_desc_en
        label   = theme.label_en
        role    = theme.agent_role_en
        op_map  = _format_interp_block(theme.opinion_interp,  "Opinion Metric Mappings", fmt)
        sp_map  = _format_interp_block(theme.spatial_interp,  "Spatial Metric Mappings", fmt)
        ev_map  = _format_interp_block(theme.event_interp,    "Event Metric Mappings",   fmt)
        heading = "## Scene Context"
        role_lbl = "Agent Role"

    lines = [
        f"{heading}: {label}",
        "",
        scene,
        "",
        f"**{role_lbl}**: {role}",
        "",
    ]
    if op_map:
        lines += [op_map, ""]
    if sp_map:
        lines += [sp_map, ""]
    if ev_map:
        lines += [ev_map, ""]

    lines += [
        "---",
        ("请结合上述场景背景和指标映射，将所有数值解读为真实事件的模拟表现，"
         "而非抽象的统计数字。" if lang == "zh" else
         "Using the scene context and metric mappings above, interpret all numeric "
         "values as simulated manifestations of real events, not abstract statistics."),
        "",
    ]
    return "\n".join(lines)


def get_theme_system_addendum(theme: Theme, lang: str = "zh") -> str:
    """
    Short addendum appended to SYSTEM_PROMPT when a theme is active.
    Reminds the LLM to maintain thematic consistency throughout.
    """
    if lang == "zh":
        return (
            f"\n\n## 主题模式已激活：{theme.label_zh}\n"
            f"在本次分析中，你将所有仿真结果解读为「{theme.label_zh}」场景下的真实动态。"
            f"使用与该场景相符的专业术语和叙事框架。"
            f"智能体在报告中应被称为「{theme.agent_role_zh}」，而非「智能体」。"
        )
    else:
        return (
            f"\n\n## Theme Mode Active: {theme.label_en}\n"
            f"For this analysis, interpret all simulation results as real-world dynamics "
            f"within a '{theme.label_en}' scenario. "
            f"Use domain-appropriate terminology and narrative framing. "
            f"Refer to agents as '{theme.agent_role_en}', not 'agents'."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _extract_hint_value(summary: Dict[str, Any], key: str) -> Optional[float]:
    """
    Extract a scalar value from a summary dict using a dot-notation key.

    Key format: "section.metric_name.stat_name"
    e.g.: "opinion.polarization_std.mean"
          "event.burstiness.mean"
          "opinion.polarization_std.trend_slope"

    Falls back to direct float value if the sub-key is absent.
    """
    parts = key.split(".")
    if len(parts) == 3:
        section, metric, stat = parts
        full_key = f"{section}.{metric}"
        val = summary.get(full_key)
        if isinstance(val, dict):
            return val.get(stat)
        if isinstance(val, (int, float)):
            return float(val)
    elif len(parts) == 2:
        val = summary.get(key)
        if isinstance(val, dict):
            return val.get("mean")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _format_interp_block(
    interp: Dict[str, str],
    heading: str,
    fmt: str,
) -> str:
    """Render an interpretation mapping as a formatted list."""
    if not interp:
        return ""
    if fmt == "md":
        lines = [f"**{heading}**"]
        for metric, desc in interp.items():
            lines.append(f"- `{metric}`: {desc}")
        return "\n".join(lines)
    elif fmt == "html":
        items = "".join(f"<li><code>{m}</code>: {d}</li>" for m, d in interp.items())
        return f"<p><strong>{heading}</strong></p><ul>{items}</ul>"
    elif fmt == "latex":
        items = "\n".join(
            r"\item \texttt{" + m.replace("_", r"\_") + "}: " + d
            for m, d in interp.items()
        )
        return (
            r"\textbf{" + heading + "}\n"
            r"\begin{itemize}" + "\n" + items + "\n" + r"\end{itemize}"
        )
    return ""


def _generic_theme() -> Theme:
    """Fallback theme when no specific theme is detected."""
    return Theme(
        name="generic",
        label_zh="通用仿真",
        label_en="Generic Simulation",
        scene_desc_zh="一个智能体意见动力学仿真，未识别出特定现实场景。",
        scene_desc_en="An agent-based opinion dynamics simulation with no specific real-world context identified.",
        agent_role_zh="智能体",
        agent_role_en="agents",
    )
