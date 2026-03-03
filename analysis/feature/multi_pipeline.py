"""
analysis/feature/multi_pipeline.py

Orchestration helper for repeated simulation runs and cross-run aggregation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from .multi import MultiRunResult, aggregate_runs
from .pipeline import FeaturePipeline


class MultiRunPipeline:
    """Run repeated simulations with same params and aggregate summaries."""

    def run_from_factory(
        self,
        engine_factory: Callable[[], Any],
        n_runs: int = 10,
        layer_idx: int = 0,
        parallel: bool = False,
        seed_list: Optional[List[int]] = None,
    ) -> MultiRunResult:
        """
        Factory mode.

        `engine_factory` can optionally accept a single seed argument.
        If `seed_list` is given, each run is called with the matching seed.
        """
        if n_runs <= 0:
            return MultiRunResult(run_summaries=[], run_finals=[], n_runs=0, layer_idx=layer_idx)

        seeds = seed_list if seed_list is not None else [None] * n_runs
        if len(seeds) < n_runs:
            raise ValueError("seed_list length must be >= n_runs.")

        def _run_once(seed: Optional[int]) -> Dict[str, Any]:
            engine = engine_factory(seed) if seed is not None else engine_factory()
            return FeaturePipeline(engine, layer_idx=layer_idx).run()

        run_results: List[Dict[str, Any]] = []

        if parallel:
            with ThreadPoolExecutor() as ex:
                futures = [ex.submit(_run_once, seeds[i]) for i in range(n_runs)]
                for fut in as_completed(futures):
                    run_results.append(fut.result())
        else:
            for i in range(n_runs):
                run_results.append(_run_once(seeds[i]))

        return aggregate_runs(run_results, layer_idx=layer_idx)

    def run_from_results(
        self,
        run_results: List[Dict[str, Any]],
        layer_idx: int = 0,
    ) -> MultiRunResult:
        """Offline mode: aggregate precomputed `FeaturePipeline.run()` results."""
        return aggregate_runs(run_results, layer_idx=layer_idx)
