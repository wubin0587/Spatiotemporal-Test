"""
analysis/report/sentences/en_US.py

English sentence templates for ReportPrinter terminal output.
Each value is a format string accepting named placeholders.

These are intentionally brief — they are for quick terminal inspection,
not publication-quality text. The AI parser produces the full narratives.

Extend freely; printer.py only uses keys it knows about, so new keys
added here will not break anything until printer.py references them.
"""

SENTENCES: dict = {

    "_meta": {
        "summary_title": "Simulation Quick Summary",
    },

    "opinion": {
        "high_polarization": (
            "Polarization is elevated (σ = {pol_std:.3f}). "
            "The population shows significant opinion divergence."
        ),
        "low_polarization": (
            "Polarization is low (σ = {pol_std:.3f}). "
            "Opinions are relatively concentrated."
        ),
        "bimodal": (
            "The bimodality coefficient (BC = {bc:.3f}) exceeds 0.555, "
            "indicating two distinct opinion camps have formed."
        ),
        "unimodal": (
            "The bimodality coefficient (BC = {bc:.3f}) is below 0.555, "
            "suggesting a roughly unimodal distribution."
        ),
        "mean_opinion": (
            "Mean opinion sits at {mean_op:.3f} "
            "(0 = fully opposed, 1 = fully aligned)."
        ),
        "high_extreme_share": (
            "Extreme opinion holders make up {share:.1%} of the population."
        ),
        "converged": (
            "Final stability is low, suggesting the system has largely converged."
        ),
        "volatile": (
            "Final stability is high; opinions remain volatile at simulation end."
        ),
    },

    "spatial": {
        "high_moran": (
            "Moran's I = {moran:.3f} — strong spatial autocorrelation. "
            "Similar opinions cluster geographically (echo-chamber pattern)."
        ),
        "low_moran": (
            "Moran's I = {moran:.3f} — weak spatial autocorrelation. "
            "Opinions are distributed without strong geographic clustering."
        ),
        "clustered": (
            "Nearest-neighbor index = {nni:.3f} < 1.0: agents form spatial clusters."
        ),
        "dispersed": (
            "Nearest-neighbor index = {nni:.3f} > 1.0: agents are spatially dispersed."
        ),
        "centroid_drift": (
            "The population centroid drifted by ({dx:.3f}, {dy:.3f}) over the run."
        ),
    },

    "topo": {
        "connected": (
            "The largest connected component covers {lcc:.1%} of nodes — "
            "the network is highly connected."
        ),
        "fragmented": (
            "The largest connected component covers only {lcc:.1%} of nodes — "
            "the network is fragmented."
        ),
        "high_modularity": (
            "Modularity = {mod:.3f} (> 0.3): meaningful community structure detected."
        ),
        "low_modularity": (
            "Modularity = {mod:.3f}: community structure is weak or absent."
        ),
        "hub_dominated": (
            "Degree Gini = {gini:.3f}: network is hub-dominated."
        ),
        "assortative": (
            "Degree assortativity = {r:.3f}: nodes preferentially connect to "
            "similarly-connected peers."
        ),
    },

    "event": {
        "bursty": (
            "Burstiness index = {burst:.3f} (> 0.3): events arrive in irregular bursts."
        ),
        "regular": (
            "Burstiness index = {burst:.3f}: event timing is relatively regular."
        ),
        "event_rate": (
            "Average event rate: {rate:.3f} events per time unit."
        ),
        "high_intensity": (
            "Peak event intensity = {max_int:.3f}: at least one high-impact shock occurred."
        ),
        "concentrated": (
            "Temporal Gini = {gini:.3f}: events are concentrated in short time windows."
        ),
    },

    "stability": {
        "high_cv": (
            "Cross-run CV = {cv:.3f} (> 0.15): results show substantial run-to-run "
            "variability — conclusions should be drawn from ensemble averages."
        ),
        "low_cv": (
            "Cross-run CV = {cv:.3f}: results are highly reproducible across runs."
        ),
        "high_spread": (
            "Inter-run opinion spread = {spread:.3f}: the simulation outcome is "
            "sensitive to initial conditions or stochastic noise."
        ),
        "low_spread": (
            "Inter-run opinion spread = {spread:.3f}: outcomes are robust across "
            "different random seeds."
        ),
        "all_converged": (
            "All {n_runs:d} runs converged before the final step — "
            "the system reliably reaches a stable attractor."
        ),
        "partial_converged": (
            "{n_converged:d} of {n_runs:d} runs converged. "
            "Some trajectories remained unsettled at simulation end."
        ),
        "none_converged": (
            "No runs converged within the simulation horizon — "
            "the system dynamics may require a longer time window."
        ),
        "stable_polarization": (
            "Final polarization is consistent across runs "
            "(mean σ = {pol_mean:.3f}, inter-run std = {pol_std:.3f})."
        ),
        "unstable_polarization": (
            "Final polarization varies substantially across runs "
            "(mean σ = {pol_mean:.3f}, inter-run std = {pol_std:.3f}), "
            "suggesting a bistable or chaotic regime."
        ),
    },

    "network_opinion": {
        "high_positive_corr": (
            "Network–opinion correlation = {corr:.3f}: high-degree nodes "
            "hold significantly more extreme or aligned opinions."
        ),
        "high_negative_corr": (
            "Network–opinion correlation = {corr:.3f}: high-degree nodes "
            "tend toward moderate or opposing opinions."
        ),
        "low_corr": (
            "Network–opinion correlation = {corr:.3f}: "
            "no strong link between network centrality and opinion value."
        ),
        "high_community_alignment": (
            "Community–opinion alignment = {align:.3f}: "
            "network communities correspond closely to opinion clusters."
        ),
        "low_community_alignment": (
            "Community–opinion alignment = {align:.3f}: "
            "network structure and opinion clusters are largely independent."
        ),
        "hub_influence": (
            "Top-{k:d} hubs carry mean opinion {hub_mean:.3f} vs "
            "population mean {pop_mean:.3f} — hubs are pulling {direction} the consensus."
        ),
        "strong_echo_chamber": (
            "Echo-chamber index = {eci:.3f} (> 0.6): agents predominantly "
            "interact with opinion-similar neighbours."
        ),
        "weak_echo_chamber": (
            "Echo-chamber index = {eci:.3f}: cross-opinion interaction is "
            "relatively common; echo chambers are weak."
        ),
        "high_rewiring": (
            "Network rewiring rate = {rate:.3f}: significant structural "
            "change co-evolved with opinion dynamics."
        ),
        "low_rewiring": (
            "Network rewiring rate = {rate:.3f}: topology remained largely "
            "static throughout the simulation."
        ),
    },
}
