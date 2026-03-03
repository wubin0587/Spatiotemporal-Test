"""
analysis/report/sentences/zh_CN.py

中文句子模板，用于 ReportPrinter 终端快速摘要输出。
每个值为带命名占位符的格式字符串。

这些模板仅供终端快速检查，不用于正式报告正文。
AI Parser 负责生成完整的叙事段落。

可自由扩展；printer.py 只使用已知的键，新增键不会破坏现有逻辑。
"""

SENTENCES: dict = {

    "_meta": {
        "summary_title": "仿真快速摘要",
    },

    "opinion": {
        "high_polarization": (
            "意见极化程度较高（σ = {pol_std:.3f}），"
            "群体内部存在显著的意见分化。"
        ),
        "low_polarization": (
            "意见极化程度较低（σ = {pol_std:.3f}），"
            "群体意见相对集中。"
        ),
        "bimodal": (
            "双峰系数 BC = {bc:.3f}（> 0.555），"
            "意见分布呈现明显的两极化结构，存在两个对立阵营。"
        ),
        "unimodal": (
            "双峰系数 BC = {bc:.3f}（< 0.555），"
            "意见分布大体呈单峰形态。"
        ),
        "mean_opinion": (
            "群体意见均值为 {mean_op:.3f}"
            "（0 = 完全反对，1 = 完全支持）。"
        ),
        "high_extreme_share": (
            "持极端意见的智能体占比达 {share:.1%}。"
        ),
        "converged": (
            "终态稳定性较低，仿真已基本收敛。"
        ),
        "volatile": (
            "终态稳定性较高，仿真结束时意见仍持续波动。"
        ),
    },

    "spatial": {
        "high_moran": (
            "Moran's I = {moran:.3f}，空间自相关显著。"
            "相似意见在地理上聚集，存在回声室效应。"
        ),
        "low_moran": (
            "Moran's I = {moran:.3f}，空间自相关较弱。"
            "意见在空间上无明显聚集规律。"
        ),
        "clustered": (
            "最近邻指数 = {nni:.3f}（< 1.0），智能体在空间上形成聚集。"
        ),
        "dispersed": (
            "最近邻指数 = {nni:.3f}（> 1.0），智能体在空间上较为分散。"
        ),
        "centroid_drift": (
            "仿真期间，群体重心漂移了 ({dx:.3f}, {dy:.3f})。"
        ),
    },

    "topo": {
        "connected": (
            "最大连通子图覆盖 {lcc:.1%} 的节点，网络连通性良好。"
        ),
        "fragmented": (
            "最大连通子图仅覆盖 {lcc:.1%} 的节点，网络存在明显碎片化。"
        ),
        "high_modularity": (
            "模块度 = {mod:.3f}（> 0.3），检测到显著的社区结构。"
        ),
        "low_modularity": (
            "模块度 = {mod:.3f}，社区结构较弱或不明显。"
        ),
        "hub_dominated": (
            "度基尼系数 = {gini:.3f}，网络由少数枢纽节点主导。"
        ),
        "assortative": (
            "度同配系数 = {r:.3f}，节点倾向于与度数相近的节点相连。"
        ),
    },

    "event": {
        "bursty": (
            "突发性指数 = {burst:.3f}（> 0.3），事件以不规则爆发方式到达。"
        ),
        "regular": (
            "突发性指数 = {burst:.3f}，事件发生节奏较为规律。"
        ),
        "event_rate": (
            "平均事件发生率：每单位时间 {rate:.3f} 次。"
        ),
        "high_intensity": (
            "事件峰值强度 = {max_int:.3f}，仿真中发生了至少一次高影响冲击。"
        ),
        "concentrated": (
            "时间基尼系数 = {gini:.3f}，事件高度集中于少数时间窗口内。"
        ),
    },

    "stability": {
        "high_cv":   "跨次运行变异系数 CV = {cv:.3f}（> 0.15），结果在不同运行间波动明显，建议基于集成均值得出结论。",
        "low_cv":    "跨次运行变异系数 CV = {cv:.3f}，结果在不同运行间高度可复现。",
        "high_spread": "跨次意见散度 = {spread:.3f}，仿真结果对初始条件或随机噪声较为敏感。",
        "low_spread":  "跨次意见散度 = {spread:.3f}，不同随机种子下结果鲁棒性良好。",
        "all_converged":     "全部 {n_runs:d} 次运行均在终止步之前收敛，系统能可靠地到达稳定吸引子。",
        "partial_converged": "{n_runs:d} 次运行中有 {n_converged:d} 次收敛，部分轨迹在仿真结束时仍未稳定。",
        "none_converged":    "在给定仿真时域内无任何运行收敛，系统动态可能需要更长的时间窗口。",
        "stable_polarization":   "终态极化在各次运行间保持一致（均值 σ = {pol_mean:.3f}，跨次标准差 = {pol_std:.3f}）。",
        "unstable_polarization": "终态极化在各次运行间差异显著（均值 σ = {pol_mean:.3f}，跨次标准差 = {pol_std:.3f}），表明系统可能处于双稳态或混沌区间。",
    },

    "network_opinion": {
        "high_positive_corr":       "网络-意见相关系数 = {corr:.3f}，度数较高的节点持有显著更极端或更趋同的意见。",
        "high_negative_corr":       "网络-意见相关系数 = {corr:.3f}，度数较高的节点倾向于持温和或对立意见。",
        "low_corr":                  "网络-意见相关系数 = {corr:.3f}，网络中心性与意见值之间无显著关联。",
        "high_community_alignment": "社区-意见对齐度 = {align:.3f}，网络社区结构与意见聚类高度吻合。",
        "low_community_alignment":  "社区-意见对齐度 = {align:.3f}，网络结构与意见聚类基本相互独立。",
        "hub_influence":             "前 {k:d} 个枢纽节点的意见均值为 {hub_mean:.3f}，整体均值为 {pop_mean:.3f}——枢纽节点正在{direction}共识形成。",
        "strong_echo_chamber":       "回声室指数 = {eci:.3f}（> 0.6），智能体主要与意见相近的邻居互动。",
        "weak_echo_chamber":         "回声室指数 = {eci:.3f}，跨意见交互较为普遍，回声室效应较弱。",
        "high_rewiring":             "网络重连率 = {rate:.3f}，拓扑结构随意见动态发生了显著的共演化变化。",
        "low_rewiring":              "网络重连率 = {rate:.3f}，仿真全程拓扑结构基本保持稳定。",
    },
}
