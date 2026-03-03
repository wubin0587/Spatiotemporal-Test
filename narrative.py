"""
analysis/parser/narrative.py

Narrative Prompt Engine
-----------------------
Converts simulation feature data into story-driven narrative text.

Unlike prompts.py (which focuses on analytical interpretation of metrics)
and themes.py (which provides real-world scene context), narrative.py is
concerned with *how* information is told — the arc, voice, and rhetorical
structure of the generated text.

Narrative modes
---------------
    "chronicle"   — Time-ordered account: what happened, when, how it evolved.
    "diagnostic"  — Problem-first structure: what went wrong / right and why.
    "comparative" — Before/after or A-vs-B contrast framing.
    "predictive"  — Forward-looking: what the current state implies about the future.
    "dramatic"    — Tension-resolution arc (peak polarisation as dramatic climax).

Each mode produces a prompt that instructs the LLM to adopt a specific
narrative structure, independent of the underlying metrics or theme.

Integration
-----------
    NarrativeBuilder is called by ParserClient after theme injection:

        theme_context = get_theme_prompt_override(theme, lang, fmt)
        narrative_ctx = NarrativeBuilder(mode="chronicle", lang=lang, fmt=fmt)
        prompt = narrative_ctx.build(section, metrics, theme_context)

        result = llm.call(SYSTEM_PROMPT + get_theme_system_addendum(theme), prompt)

Usage
-----
    from analysis.parser.narrative import NarrativeBuilder, NarrativeMode

    builder = NarrativeBuilder(mode="dramatic", lang="zh", fmt="md")
    prompt = builder.build(
        section="opinion",
        metrics={"polarization_std": {...}, "bimodality_coefficient": {...}},
        theme_context=theme_override_str,   # from themes.get_theme_prompt_override()
    )
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import json


# ═════════════════════════════════════════════════════════════════════════════
# Narrative Mode Registry
# ═════════════════════════════════════════════════════════════════════════════

class NarrativeMode:
    """Namespace of available narrative mode identifiers."""
    CHRONICLE   = "chronicle"
    DIAGNOSTIC  = "diagnostic"
    COMPARATIVE = "comparative"
    PREDICTIVE  = "predictive"
    DRAMATIC    = "dramatic"

    ALL = [CHRONICLE, DIAGNOSTIC, COMPARATIVE, PREDICTIVE, DRAMATIC]


# ── Mode descriptions (shown to the LLM to establish narrative contract) ────

_MODE_DESC_ZH: Dict[str, str] = {
    NarrativeMode.CHRONICLE: (
        "按时间顺序讲述仿真的演化过程。"
        "从初始状态出发，描述关键转折点，最终抵达终态。"
        "叙事结构：【起点】→【发展】→【关键事件】→【当前状态】。"
        "时态以过去式为主，辅以现在完成时描述持续影响。"
    ),
    NarrativeMode.DIAGNOSTIC: (
        "以问题诊断的方式组织分析。"
        "首先点明核心现象（意见极化、空间分异、网络脆断等），"
        "随后追溯成因机制，最后评估后果与影响。"
        "叙事结构：【现象识别】→【成因分析】→【机制解释】→【影响评估】。"
        "采用客观、精准的专业分析语气。"
    ),
    NarrativeMode.COMPARATIVE: (
        "以对比为核心叙事手段。"
        "对比可以是：早期 vs 晚期、算法模式 vs 现实模式、不同群体之间、"
        "或不同指标维度之间的差异。"
        "叙事结构：【基准描述】→【对比对象描述】→【差异分析】→【差异的意义】。"
        "善用'而'、'相比之下'、'截然不同'等转折连接词。"
    ),
    NarrativeMode.PREDICTIVE: (
        "以当前状态为起点，推演未来趋势。"
        "基于趋势指标（trend_slope、final_stability、evolution_delta），"
        "判断系统是否趋于收敛、继续极化或进入混沌状态。"
        "叙事结构：【当前快照】→【趋势信号】→【路径预测】→【条件判断与风险提示】。"
        "语气应表达合理的不确定性，避免过度确定的预言。"
    ),
    NarrativeMode.DRAMATIC: (
        "以戏剧化的叙事张力组织内容。"
        "将仿真过程呈现为一个有冲突、有高潮、有结局的故事。"
        "极化高峰是戏剧高潮，稳定化是结局，初始状态是铺垫。"
        "叙事结构：【背景铺垫】→【矛盾升级】→【高潮爆发】→【结局与余震】。"
        "语言可适当富有感染力，但不失科学严谨性。"
    ),
}

_MODE_DESC_EN: Dict[str, str] = {
    NarrativeMode.CHRONICLE: (
        "Tell the story of the simulation in chronological order. "
        "Begin with the initial state, trace key turning points, and arrive at the final state. "
        "Structure: [Origin] → [Development] → [Key Events] → [Current State]. "
        "Use past tense primarily, with present perfect for lasting effects."
    ),
    NarrativeMode.DIAGNOSTIC: (
        "Organise the analysis as a problem diagnosis. "
        "Lead with the core phenomenon (polarisation, spatial segregation, network fragmentation), "
        "then trace causal mechanisms, then evaluate consequences. "
        "Structure: [Phenomenon] → [Root Causes] → [Mechanism] → [Impact Assessment]. "
        "Adopt an objective, precise, analytical register."
    ),
    NarrativeMode.COMPARATIVE: (
        "Use contrast as the central narrative device. "
        "Comparisons may be: early vs late, algorithm mode vs reality mode, group vs group, "
        "or dimension vs dimension. "
        "Structure: [Baseline] → [Contrast Subject] → [Difference Analysis] → [Significance]. "
        "Use transition phrases: 'whereas', 'by contrast', 'strikingly different'."
    ),
    NarrativeMode.PREDICTIVE: (
        "Start from the current state and extrapolate future trajectories. "
        "Use trend indicators (trend_slope, final_stability, evolution_delta) to judge whether "
        "the system is converging, continuing to polarise, or entering oscillation. "
        "Structure: [Current Snapshot] → [Trend Signals] → [Path Projection] → [Conditions & Risks]. "
        "Express appropriate epistemic uncertainty; avoid overconfident predictions."
    ),
    NarrativeMode.DRAMATIC: (
        "Organise content with dramatic narrative tension. "
        "Present the simulation as a story with conflict, climax, and resolution. "
        "The polarisation peak is the dramatic climax; stabilisation is the denouement; "
        "the initial state is the exposition. "
        "Structure: [Setup] → [Escalation] → [Climax] → [Resolution & Aftermath]. "
        "Language may be evocative but must remain scientifically grounded."
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# Section-Specific Narrative Guidance
# ═════════════════════════════════════════════════════════════════════════════

_SECTION_NARRATIVE_HINTS_ZH: Dict[str, Dict[str, str]] = {
    "opinion": {
        NarrativeMode.CHRONICLE:   "描述意见分布如何从初始多样性演化至当前格局，点明每次重大转变的时间点。",
        NarrativeMode.DIAGNOSTIC:  "诊断意见极化的根源：是同质性网络导致的回声室，还是外部事件冲击？",
        NarrativeMode.COMPARATIVE: "对比仿真早期与晚期的意见分布特征，量化变化程度。",
        NarrativeMode.PREDICTIVE:  "基于trend_slope和final_stability，预测意见是否将继续极化或趋于稳定。",
        NarrativeMode.DRAMATIC:    "以极化峰值为高潮，讲述意见从多元走向撕裂的戏剧性过程。",
    },
    "spatial": {
        NarrativeMode.CHRONICLE:   "描述人群空间分布的动态演化，包括聚集、扩散和重心偏移。",
        NarrativeMode.DIAGNOSTIC:  "分析空间聚集与意见极化之间的因果关系，是地理隔离强化了观点分化吗？",
        NarrativeMode.COMPARATIVE: "对比高影响区域与低影响区域的空间特征差异。",
        NarrativeMode.PREDICTIVE:  "基于radius_of_gyration和centroid漂移趋势，预测空间格局走向。",
        NarrativeMode.DRAMATIC:    "以空间重组作为群体分裂的戏剧性具象，描述回声室的地理形成过程。",
    },
    "topo": {
        NarrativeMode.CHRONICLE:   "追踪网络拓扑从初始结构到当前状态的演化轨迹。",
        NarrativeMode.DIAGNOSTIC:  "诊断网络结构对意见传播的放大或抑制效应。",
        NarrativeMode.COMPARATIVE: "对比不同社区子网络的拓扑特征差异。",
        NarrativeMode.PREDICTIVE:  "基于当前网络结构预测信息传播效率的未来变化。",
        NarrativeMode.DRAMATIC:    "以网络断裂（社区分裂）为高潮，描述连接性从整合到碎片化的过程。",
    },
    "event": {
        NarrativeMode.CHRONICLE:   "按时间顺序记录关键事件序列，描述每次事件对系统的冲击和余震。",
        NarrativeMode.DIAGNOSTIC:  "诊断哪类事件是意见格局变化的主要驱动力。",
        NarrativeMode.COMPARATIVE: "对比不同类型或不同强度事件的效果差异。",
        NarrativeMode.PREDICTIVE:  "基于burstiness和temporal_gini预测未来事件的可能节奏和强度。",
        NarrativeMode.DRAMATIC:    "以最高强度事件为戏剧高潮，描述'平静-爆发-余震'的完整弧线。",
    },
}

_SECTION_NARRATIVE_HINTS_EN: Dict[str, Dict[str, str]] = {
    "opinion": {
        NarrativeMode.CHRONICLE:   "Describe how the opinion distribution evolved from initial diversity to its current pattern, marking each major inflection point.",
        NarrativeMode.DIAGNOSTIC:  "Diagnose the root cause of opinion polarisation: echo-chamber effects from homophilous networks, or exogenous event shocks?",
        NarrativeMode.COMPARATIVE: "Contrast the early and late opinion distribution characteristics; quantify the degree of change.",
        NarrativeMode.PREDICTIVE:  "Using trend_slope and final_stability, project whether opinion will continue polarising or stabilise.",
        NarrativeMode.DRAMATIC:    "Use the polarisation peak as the climax; narrate the dramatic journey from diversity to rupture.",
    },
    "spatial": {
        NarrativeMode.CHRONICLE:   "Describe the dynamic spatial evolution of the population: clustering, dispersal, and centroid migration.",
        NarrativeMode.DIAGNOSTIC:  "Analyse the causal link between spatial clustering and opinion polarisation — does geographic isolation amplify divergence?",
        NarrativeMode.COMPARATIVE: "Contrast spatial characteristics of high-impact versus low-impact zones.",
        NarrativeMode.PREDICTIVE:  "Use radius_of_gyration and centroid drift trends to project future spatial configurations.",
        NarrativeMode.DRAMATIC:    "Use spatial reorganisation as a concrete manifestation of group fragmentation; narrate the geographic formation of echo chambers.",
    },
    "topo": {
        NarrativeMode.CHRONICLE:   "Trace the network topology's evolution from initial structure to its current state.",
        NarrativeMode.DIAGNOSTIC:  "Diagnose how network structure amplifies or suppresses opinion propagation.",
        NarrativeMode.COMPARATIVE: "Contrast topological characteristics across different community sub-networks.",
        NarrativeMode.PREDICTIVE:  "Use current network structure to project future changes in information-flow efficiency.",
        NarrativeMode.DRAMATIC:    "Use network fragmentation (community split) as the climax: narrate connectivity's journey from integration to atomisation.",
    },
    "event": {
        NarrativeMode.CHRONICLE:   "Catalogue key events in chronological order; describe each event's impact and aftershocks.",
        NarrativeMode.DIAGNOSTIC:  "Diagnose which event types are the primary drivers of opinion-landscape shifts.",
        NarrativeMode.COMPARATIVE: "Contrast the effects of different event types or intensity levels.",
        NarrativeMode.PREDICTIVE:  "Using burstiness and temporal_gini, forecast likely future event rhythm and intensity.",
        NarrativeMode.DRAMATIC:    "Use the highest-intensity event as the dramatic peak; narrate the full 'calm → eruption → aftershock' arc.",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# NarrativeBuilder
# ═════════════════════════════════════════════════════════════════════════════

class NarrativeBuilder:
    """
    Constructs narrative-mode-aware prompts for AI section analysis.

    Parameters
    ----------
    mode : str
        One of NarrativeMode.ALL. Default: "chronicle".
    lang : str
        "zh" | "en". Default: "zh".
    fmt : str
        "md" | "html" | "latex". Default: "md".
    word_count_hint : int
        Approximate target word count for the generated narrative.
        Injected as a soft instruction to the LLM.
    """

    def __init__(
        self,
        mode: str = NarrativeMode.CHRONICLE,
        lang: str = "zh",
        fmt: str = "md",
        word_count_hint: int = 300,
    ):
        if mode not in NarrativeMode.ALL:
            raise ValueError(
                f"Unknown narrative mode '{mode}'. "
                f"Choose from: {NarrativeMode.ALL}"
            )
        self.mode            = mode
        self.lang            = lang
        self.fmt             = fmt
        self.word_count_hint = word_count_hint

    # ── Primary builder ───────────────────────────────────────────────────────

    def build(
        self,
        section: str,
        metrics: Dict[str, Any],
        theme_context: str = "",
        extra_context: str = "",
    ) -> str:
        """
        Build a full narrative prompt for a given section and metrics dict.

        Parameters
        ----------
        section : str
            "opinion" | "spatial" | "topo" | "event" | custom
        metrics : dict
            Feature summary stats for this section.
        theme_context : str
            Output of get_theme_prompt_override() from themes.py.
            Injected before the narrative instructions.
        extra_context : str
            Any additional user-supplied context.

        Returns
        -------
        str — ready-to-use LLM user prompt.
        """
        mode_desc      = self._mode_desc()
        section_hint   = self._section_hint(section)
        format_instr   = self._format_instruction()
        lang_instr     = self._lang_instruction()
        metrics_block  = _fmt_metrics(metrics)
        wc_hint        = self._word_count_instruction()

        parts = []

        # 1. Language + format directives
        parts.append(f"{lang_instr}\n{format_instr}")

        # 2. Theme context (if provided)
        if theme_context.strip():
            parts.append(theme_context.strip())

        # 3. Narrative mode instruction
        parts.append(self._narrative_mode_block(mode_desc))

        # 4. Section-specific narrative guidance
        if section_hint:
            parts.append(self._section_guidance_block(section_hint, section))

        # 5. Metrics data
        parts.append(self._metrics_block(metrics_block, section))

        # 6. Extra context
        if extra_context.strip():
            parts.append(extra_context.strip())

        # 7. Final instruction
        parts.append(self._final_instruction(wc_hint))

        return "\n\n".join(parts)

    # ── Executive narrative ────────────────────────────────────────────────────

    def build_executive_narrative(
        self,
        full_summary: Dict[str, Any],
        theme_context: str = "",
        simulation_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build an executive-level narrative prompt covering all metric families.

        Used to generate the top-level summary section of a report.
        """
        mode_desc     = self._mode_desc()
        format_instr  = self._format_instruction()
        lang_instr    = self._lang_instruction()
        meta_block    = _fmt_metrics(simulation_meta) if simulation_meta else "N/A"
        summary_block = _fmt_metrics(full_summary)
        wc_hint       = self._word_count_instruction(multiplier=2)

        parts = []
        parts.append(f"{lang_instr}\n{format_instr}")

        if theme_context.strip():
            parts.append(theme_context.strip())

        parts.append(self._narrative_mode_block(mode_desc))

        if self.lang == "zh":
            parts.append(
                "## 任务：撰写仿真执行摘要\n\n"
                "以下是完整的特征汇总统计，涵盖意见、空间、拓扑和事件四个维度。\n"
                "请综合所有维度，撰写一份连贯的叙事性执行摘要。"
            )
        else:
            parts.append(
                "## Task: Write Simulation Executive Narrative\n\n"
                "Below is the full feature summary spanning opinion, spatial, topology, and event dimensions.\n"
                "Synthesise all dimensions into a coherent narrative executive summary."
            )

        if simulation_meta:
            meta_label = "### 仿真元数据" if self.lang == "zh" else "### Simulation Metadata"
            parts.append(f"{meta_label}\n```json\n{meta_block}\n```")

        summary_label = "### 完整特征汇总" if self.lang == "zh" else "### Full Feature Summary"
        parts.append(f"{summary_label}\n```json\n{summary_block}\n```")
        parts.append(self._final_instruction(wc_hint))

        return "\n\n".join(parts)

    # ── Transition narrative ───────────────────────────────────────────────────

    def build_transition_narrative(
        self,
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
        event_description: str = "",
        section: str = "opinion",
    ) -> str:
        """
        Build a narrative prompt describing a state transition (before → after an event).

        Useful for annotating critical moments in the simulation history.
        """
        lang_instr   = self._lang_instruction()
        format_instr = self._format_instruction()
        before_block = _fmt_metrics(before_metrics)
        after_block  = _fmt_metrics(after_metrics)

        if self.lang == "zh":
            task = (
                "## 任务：描述状态转变\n\n"
                "请以叙事方式描述以下事件前后系统状态的变化。"
            )
            before_lbl = "### 事件前状态"
            after_lbl  = "### 事件后状态"
            event_lbl  = "### 触发事件"
            instr = (
                "聚焦于：变化幅度、变化方向、最受影响的维度。"
                f"采用「{_MODE_DESC_ZH.get(self.mode, '')}」的叙事风格。"
            )
        else:
            task = (
                "## Task: Narrate a State Transition\n\n"
                "Describe in narrative form the change in system state across the following event."
            )
            before_lbl = "### Pre-Event State"
            after_lbl  = "### Post-Event State"
            event_lbl  = "### Triggering Event"
            instr = (
                "Focus on: magnitude of change, direction of change, most affected dimensions. "
                f"Use the narrative style of '{_MODE_DESC_EN.get(self.mode, '')}'."
            )

        parts = [
            f"{lang_instr}\n{format_instr}",
            task,
        ]
        if event_description:
            parts.append(f"{event_lbl}\n{event_description}")

        parts += [
            f"{before_lbl}\n```json\n{before_block}\n```",
            f"{after_lbl}\n```json\n{after_block}\n```",
            instr,
        ]

        return "\n\n".join(parts)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mode_desc(self) -> str:
        return (
            _MODE_DESC_ZH.get(self.mode, "")
            if self.lang == "zh"
            else _MODE_DESC_EN.get(self.mode, "")
        )

    def _section_hint(self, section: str) -> str:
        hints = (
            _SECTION_NARRATIVE_HINTS_ZH
            if self.lang == "zh"
            else _SECTION_NARRATIVE_HINTS_EN
        )
        return hints.get(section, {}).get(self.mode, "")

    def _lang_instruction(self) -> str:
        return (
            "请用简体中文撰写你的完整回复。"
            if self.lang == "zh"
            else "Write your entire response in English."
        )

    def _format_instruction(self) -> str:
        _fmt_map = {
            "md": (
                "格式：使用标准 Markdown。用 ## 作二级标题，**粗体** 标注关键术语，"
                "行文以段落为主，避免滥用列表。"
                if self.lang == "zh" else
                "Format: standard Markdown. Use ## for section headers, **bold** for key terms. "
                "Prefer paragraphs over bullet lists."
            ),
            "html": (
                "格式：输出纯 HTML 片段（无 <html>/<body> 包裹）。"
                "用 <h2> 作节标题，<strong> 标注关键词，<p> 组织段落。"
                if self.lang == "zh" else
                "Format: pure HTML fragment (no <html>/<body> wrapper). "
                "Use <h2> for headers, <strong> for key terms, <p> for paragraphs."
            ),
            "latex": (
                "格式：输出 LaTeX 片段（无 \\documentclass 前导）。"
                "用 \\section{} 作节标题，\\textbf{} 标注关键词。"
                if self.lang == "zh" else
                "Format: LaTeX fragment (no \\documentclass preamble). "
                "Use \\section{} for headers, \\textbf{} for key terms."
            ),
        }
        return _fmt_map.get(self.fmt, _fmt_map["md"])

    def _word_count_instruction(self, multiplier: float = 1.0) -> str:
        wc = int(self.word_count_hint * multiplier)
        if self.lang == "zh":
            return f"篇幅目标约 {wc} 字，内容充实而不冗余。"
        return f"Target approximately {wc} words. Be substantive but not verbose."

    def _narrative_mode_block(self, mode_desc: str) -> str:
        if self.lang == "zh":
            mode_name = {
                NarrativeMode.CHRONICLE:   "编年体叙事",
                NarrativeMode.DIAGNOSTIC:  "诊断式分析",
                NarrativeMode.COMPARATIVE: "对比式叙事",
                NarrativeMode.PREDICTIVE:  "预测式推演",
                NarrativeMode.DRAMATIC:    "戏剧化叙事",
            }.get(self.mode, self.mode)
            return (
                f"## 叙事风格：{mode_name}\n\n"
                f"{mode_desc}"
            )
        else:
            mode_name = {
                NarrativeMode.CHRONICLE:   "Chronicle",
                NarrativeMode.DIAGNOSTIC:  "Diagnostic",
                NarrativeMode.COMPARATIVE: "Comparative",
                NarrativeMode.PREDICTIVE:  "Predictive",
                NarrativeMode.DRAMATIC:    "Dramatic",
            }.get(self.mode, self.mode)
            return (
                f"## Narrative Mode: {mode_name}\n\n"
                f"{mode_desc}"
            )

    def _section_guidance_block(self, hint: str, section: str) -> str:
        if self.lang == "zh":
            return f"### 本节叙事重点（{section}）\n{hint}"
        return f"### Section Narrative Focus ({section})\n{hint}"

    def _metrics_block(self, metrics_block: str, section: str) -> str:
        if self.lang == "zh":
            return f"### 指标数据（{section}）\n```json\n{metrics_block}\n```"
        return f"### Metric Data ({section})\n```json\n{metrics_block}\n```"

    def _final_instruction(self, wc_hint: str) -> str:
        if self.lang == "zh":
            return (
                "---\n"
                "请严格遵循上述叙事风格和格式要求，以下是你的分析：\n"
                f"{wc_hint}\n"
                "每个句子都应传递新的洞察，避免重复描述同一事实。"
            )
        return (
            "---\n"
            "Strictly follow the narrative mode and format requirements above. "
            "Here is your analysis:\n"
            f"{wc_hint}\n"
            "Every sentence should deliver new insight; avoid restating the same fact."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Standalone Narrative Prompt Functions
# ═════════════════════════════════════════════════════════════════════════════
# These can be imported directly without instantiating NarrativeBuilder,
# mirroring the pattern in prompts.py for use in SECTION_PROMPT_REGISTRY.

def chronicle_section_prompt(
    section: str,
    metrics: Dict[str, Any],
    theme_context: str = "",
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """Chronicle narrative for a single section."""
    return NarrativeBuilder(NarrativeMode.CHRONICLE, lang, fmt).build(
        section, metrics, theme_context
    )


def diagnostic_section_prompt(
    section: str,
    metrics: Dict[str, Any],
    theme_context: str = "",
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """Diagnostic narrative for a single section."""
    return NarrativeBuilder(NarrativeMode.DIAGNOSTIC, lang, fmt).build(
        section, metrics, theme_context
    )


def dramatic_summary_prompt(
    full_summary: Dict[str, Any],
    theme_context: str = "",
    simulation_meta: Optional[Dict[str, Any]] = None,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """Dramatic executive summary narrative."""
    return NarrativeBuilder(NarrativeMode.DRAMATIC, lang, fmt).build_executive_narrative(
        full_summary, theme_context, simulation_meta
    )


def predictive_summary_prompt(
    full_summary: Dict[str, Any],
    theme_context: str = "",
    simulation_meta: Optional[Dict[str, Any]] = None,
    lang: str = "zh",
    fmt: str = "md",
) -> str:
    """Predictive / forward-looking executive summary."""
    return NarrativeBuilder(NarrativeMode.PREDICTIVE, lang, fmt).build_executive_narrative(
        full_summary, theme_context, simulation_meta
    )


# ═════════════════════════════════════════════════════════════════════════════
# Narrative Registry
# ═════════════════════════════════════════════════════════════════════════════

#: Maps (section, mode) → prompt function — for external dispatch.
NARRATIVE_PROMPT_REGISTRY: Dict[str, Any] = {
    NarrativeMode.CHRONICLE:   chronicle_section_prompt,
    NarrativeMode.DIAGNOSTIC:  diagnostic_section_prompt,
    NarrativeMode.DRAMATIC:    dramatic_summary_prompt,
    NarrativeMode.PREDICTIVE:  predictive_summary_prompt,
}


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_metrics(d: Any, indent: int = 2) -> str:
    """JSON-serialise a metrics dict, handling numpy types."""
    class _SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                import numpy as np
                if isinstance(obj, np.integer):  return int(obj)
                if isinstance(obj, np.floating): return round(float(obj), 6)
                if isinstance(obj, np.ndarray):  return obj.tolist()
            except ImportError:
                pass
            if isinstance(obj, float):
                return round(obj, 6)
            return str(obj)

    return json.dumps(d, indent=indent, cls=_SafeEncoder, ensure_ascii=False)
