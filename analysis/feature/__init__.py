"""analysis/feature — public API."""

from .extractor import (
    extract_all_features,
    extract_event_features,
    extract_network_opinion_features,
    extract_opinion_features,
    extract_spatial_features,
    extract_topo_features,
)
from .composer import (
    compose_multilayer,
    compose_timeseries,
    flatten_step,
    summarize_timeseries,
)
from .pipeline import FeaturePipeline, run_feature_pipeline

__all__ = [
    # extractor
    "extract_all_features",
    "extract_event_features",
    "extract_network_opinion_features",
    "extract_opinion_features",
    "extract_spatial_features",
    "extract_topo_features",
    # composer
    "compose_multilayer",
    "compose_timeseries",
    "flatten_step",
    "summarize_timeseries",
    # pipeline
    "FeaturePipeline",
    "run_feature_pipeline",
]
