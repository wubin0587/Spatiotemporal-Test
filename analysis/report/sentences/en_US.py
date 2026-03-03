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
}
