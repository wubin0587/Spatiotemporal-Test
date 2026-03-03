"""
analysis/parser/client.py

AI Parser Client
----------------
Manages communication with the LLM API and orchestrates multi-section
analysis of simulation feature data.

Usage
-----
    from analysis.parser.client import ParserClient

    client = ParserClient(api_key="...", lang="zh", fmt="md")

    # Parse a full feature pipeline result
    result = client.parse_pipeline_result(pipeline_output)

    # Parse a single section
    opinion_text = client.parse_section("opinion", summary["opinion"])

    # Compare two runs
    comparison = client.compare_runs(summary_a, summary_b, label_a="baseline", label_b="tuned")

Data Contract
-------------
Input:  pipeline output from analysis.feature.pipeline.run_feature_pipeline()
        {
            "final":      dict   – features of last snapshot
            "timeseries": dict   – per-key np.ndarray time series
            "summary":    dict   – per-key statistics (mean, std, trend_slope, ...)
            "data_issues": list  – structural warnings
        }

Output: ParsedResult  –  dict[section_name -> str (rendered text)]
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from analysis.feature.multi_run import MultiRunResult

from .prompts import (
    SYSTEM_PROMPT,
    SECTION_PROMPT_REGISTRY,
    trend_summary_prompt,
    comparative_prompt,
    narrative_section_prompt,
    multi_run_opinion_prompt,
    stability_analysis_prompt,
    parameter_comparison_prompt,
    _fmt_dict,
)


# ═════════════════════════════════════════════════════════════════════════════
# LLM Backend Abstraction
# ═════════════════════════════════════════════════════════════════════════════

class _LLMBackend:
    """
    Thin wrapper around an LLM API.
    Defaults to OpenAI-compatible chat completion.
    Swap out _call() to support other providers (Anthropic, local, etc.).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_key     = api_key
        self.model       = model
        self.base_url    = base_url
        self.timeout     = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """
        Send a system+user message pair to the LLM and return the text response.

        Raises
        ------
        RuntimeError  if all retries are exhausted.
        """
        for attempt in range(self.max_retries):
            try:
                return self._call(system, user, max_tokens)
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"LLM call failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.retry_delay * (attempt + 1))
        return ""  # unreachable

    def _call(self, system: str, user: str, max_tokens: int) -> str:
        """
        Concrete API call. Override this method to support other providers.

        Default: OpenAI-compatible chat completion via `openai` package.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for the default LLM backend. "
                "Install with: pip install openai\n"
                "Or subclass _LLMBackend and override _call() for another provider."
            )

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,   # low temperature for analytical consistency
        )
        return response.choices[0].message.content.strip()


# ═════════════════════════════════════════════════════════════════════════════
# Public Parser Client
# ═════════════════════════════════════════════════════════════════════════════

class ParserClient:
    """
    High-level client that maps feature summaries → AI-generated analysis text.

    Parameters
    ----------
    api_key    : str    LLM API key.
    lang       : str    Output language: "zh" | "en"  (default "zh")
    fmt        : str    Output format:   "md" | "html" | "latex"  (default "md")
    model      : str    LLM model name (default "gpt-4o")
    base_url   : str    Optional custom API base URL (for proxies / local models)
    sections   : list   Which sections to parse. Default: all registered sections.
                        Possible values: "opinion", "spatial", "topo", "event", "trend_summary"
    max_tokens : int    Max tokens per LLM response (default 2048)
    verbose    : bool   If True, print section names as they are processed.
    """

    #: Default section order for full pipeline parse
    DEFAULT_SECTIONS: List[str] = ["opinion", "spatial", "topo", "event"]

    def __init__(
        self,
        api_key: str,
        lang: str = "zh",
        fmt: str = "md",
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        sections: Optional[List[str]] = None,
        max_tokens: int = 2048,
        verbose: bool = False,
        _backend: Optional[_LLMBackend] = None,  # injection for testing
    ):
        self.lang       = lang
        self.fmt        = fmt
        self.sections   = sections or self.DEFAULT_SECTIONS
        self.max_tokens = max_tokens
        self.verbose    = verbose

        self._backend = _backend or _LLMBackend(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    # ── Primary Entry Points ─────────────────────────────────────────────────

    def parse_pipeline_result(
        self,
        pipeline_output: Dict[str, Any],
        include_executive_summary: bool = True,
        simulation_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Parse a full output from run_feature_pipeline() into section texts.

        Parameters
        ----------
        pipeline_output : dict
            Output of FeaturePipeline.run() / run_feature_pipeline().
            Must contain keys: "summary", "final", "data_issues".
        include_executive_summary : bool
            If True, prepend an AI-generated executive summary section.
        simulation_meta : dict, optional
            Extra metadata to include in the executive summary prompt
            (e.g., n_agents, n_steps, config params).

        Returns
        -------
        dict[str, str]
            Keys: section names + optionally "executive_summary".
            Values: rendered text in the requested language and format.
        """
        summary = pipeline_output.get("summary", {})
        final   = pipeline_output.get("final", {})

        # Partition summary by section prefix (e.g., "opinion.mean_opinion" -> "opinion")
        section_summaries = _partition_summary(summary)

        results: Dict[str, str] = {}

        # 1. Per-section analysis
        for section in self.sections:
            sec_data = section_summaries.get(section, {})
            if not sec_data:
                # Fall back to final snapshot values for this section
                sec_data = final.get(section, {})
            if sec_data:
                if self.verbose:
                    print(f"[ParserClient] Parsing section: {section}")
                results[section] = self.parse_section(section, sec_data)

        # 2. Executive summary (uses the full summary dict)
        if include_executive_summary and summary:
            if self.verbose:
                print("[ParserClient] Generating executive summary...")
            results["executive_summary"] = self._call_trend_summary(
                summary, simulation_meta
            )

        return results

    def parse_section(
        self,
        section: str,
        metrics: Dict[str, Any],
        context: str = "",
    ) -> str:
        """
        Parse a single named section.

        Parameters
        ----------
        section : str
            One of the keys in SECTION_PROMPT_REGISTRY, or any custom name.
        metrics : dict
            Metric values or summary stats for this section.
        context : str
            Optional extra context injected into the prompt (for custom sections).

        Returns
        -------
        str  –  AI-generated analysis text.
        """
        prompt_fn = SECTION_PROMPT_REGISTRY.get(section)

        if prompt_fn is not None and section not in ("trend_summary", "comparative"):
            user_prompt = prompt_fn(metrics, lang=self.lang, fmt=self.fmt)
        else:
            # Unknown section: use generic fallback
            user_prompt = narrative_section_prompt(
                section_name=section,
                metrics=metrics,
                context=context,
                lang=self.lang,
                fmt=self.fmt,
            )

        return self._backend.call(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=self.max_tokens,
        )

    def compare_runs(
        self,
        summary_a: Dict[str, Any],
        summary_b: Dict[str, Any],
        label_a: str = "Run A",
        label_b: str = "Run B",
    ) -> str:
        """
        Generate a comparative analysis of two simulation runs.

        Parameters
        ----------
        summary_a, summary_b : dict
            Feature summaries from summarize_timeseries() for each run.
        label_a, label_b : str
            Human-readable names for the two runs.

        Returns
        -------
        str  –  Comparative analysis text.
        """
        user_prompt = comparative_prompt(
            summary_a=summary_a,
            summary_b=summary_b,
            label_a=label_a,
            label_b=label_b,
            lang=self.lang,
            fmt=self.fmt,
        )
        return self._backend.call(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=self.max_tokens,
        )


    def parse_multi_run_result(
        self,
        multi_run_result: MultiRunResult,
        include_stability_analysis: bool = True,
        include_executive_summary: bool = True,
        simulation_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Parse an aggregated MultiRunResult into section narratives."""
        section_summaries = _partition_summary(multi_run_result.mean_summary)
        results: Dict[str, str] = {}

        for section in self.sections:
            sec_mean = section_summaries.get(section, {})
            if not sec_mean:
                continue

            sec_std = _partition_summary(multi_run_result.std_summary).get(section, {})
            sec_ci95 = _partition_summary(multi_run_result.ci95_summary).get(section, {})
            enriched = _enrich_with_multi_run_stats(sec_mean, sec_std, sec_ci95)

            if section == "opinion":
                user_prompt = multi_run_opinion_prompt(
                    mean_summary=enriched,
                    std_summary=sec_std,
                    ci95_summary=sec_ci95,
                    n_runs=multi_run_result.n_runs,
                    lang=self.lang,
                    fmt=self.fmt,
                )
                results[section] = self._backend.call(
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    max_tokens=self.max_tokens,
                )
            else:
                results[section] = self.parse_section(section, enriched)

        if include_stability_analysis:
            user_prompt = stability_analysis_prompt(
                cv_summary=multi_run_result.cv_summary,
                consensus_scores=multi_run_result.consensus_score,
                n_runs=multi_run_result.n_runs,
                lang=self.lang,
                fmt=self.fmt,
            )
            results["stability"] = self._backend.call(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=self.max_tokens,
            )

        if include_executive_summary and multi_run_result.mean_summary:
            results["executive_summary"] = self._call_trend_summary(
                multi_run_result.mean_summary,
                simulation_meta,
            )

        return results

    def compare_parameter_sweeps(
        self,
        results_map: Dict[str, MultiRunResult],
        focus_metrics: Optional[List[str]] = None,
    ) -> str:
        """Compare multiple parameter configurations, each represented by MultiRunResult."""
        sweep_results = {k: v.mean_summary for k, v in results_map.items()}
        sweep_stds = {k: v.std_summary for k, v in results_map.items()}

        user_prompt = parameter_comparison_prompt(
            sweep_results=sweep_results,
            sweep_stds=sweep_stds,
            focus_metrics=focus_metrics,
            lang=self.lang,
            fmt=self.fmt,
        )
        return self._backend.call(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=self.max_tokens,
        )
    def parse_custom(
        self,
        section_name: str,
        metrics: Dict[str, Any],
        context: str = "",
    ) -> str:
        """Convenience alias for parse_section() with a custom section name."""
        return self.parse_section(section_name, metrics, context=context)

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _call_trend_summary(
        self,
        full_summary: Dict[str, Any],
        simulation_meta: Optional[Dict[str, Any]],
    ) -> str:
        user_prompt = trend_summary_prompt(
            feature_summary=full_summary,
            lang=self.lang,
            fmt=self.fmt,
            simulation_meta=simulation_meta,
        )
        return self._backend.call(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=self.max_tokens,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _partition_summary(
    summary: Dict[str, Any],
    known_sections: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Split a flat summary dict (keyed as "section.metric_name") into nested
    section dicts.

    Example
    -------
    Input:  {"opinion.polarization_std": {...}, "spatial.moran_i": {...}}
    Output: {"opinion": {"polarization_std": {...}}, "spatial": {"moran_i": {...}}}

    Keys without a dot separator are placed under "other".
    """
    known_sections = known_sections or list(SECTION_PROMPT_REGISTRY.keys())
    partitioned: Dict[str, Dict[str, Any]] = {}

    for key, val in summary.items():
        if "." in key:
            section, metric = key.split(".", 1)
        else:
            section, metric = "other", key

        partitioned.setdefault(section, {})[metric] = val

    return partitioned

def _enrich_with_multi_run_stats(
    mean_summary: Dict[str, Any],
    std_summary: Dict[str, Any],
    ci95_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Inject run-level uncertainty fields into summary metric dicts."""
    out: Dict[str, Any] = {}
    for metric, payload in mean_summary.items():
        if isinstance(payload, dict):
            merged = dict(payload)
            merged["run_std"] = std_summary.get(metric, {})
            merged["ci95"] = ci95_summary.get(metric, {})
            out[metric] = merged
        else:
            out[metric] = {
                "mean": payload,
                "run_std": std_summary.get(metric),
                "ci95": ci95_summary.get(metric),
            }
    return out
