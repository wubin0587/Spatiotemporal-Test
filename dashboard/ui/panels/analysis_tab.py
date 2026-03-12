"""Post-run analysis tab and activation helpers."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from analysis.manager import run_analysis
from core import renderer


_TEXT = {
    "en": {
        "summary": "Summary Metrics",
        "dashboard": "Composite Dashboard",
        "gallery": "Analysis Figures",
        "download": "Download Figures (.zip)",
        "features": "features_final.json",
        "empty": "Run simulation first, then analysis will appear here.",
    },
    "zh": {
        "summary": "关键指标汇总",
        "dashboard": "综合仪表盘",
        "gallery": "分析图像",
        "download": "下载图表（.zip）",
        "features": "features_final.json",
        "empty": "请先完成仿真，再查看分析结果。",
    },
}


def build_analysis_tab(lang: str = "en") -> dict[str, gr.Component]:
    """Build analysis tab widgets."""
    t = _TEXT.get(lang, _TEXT["en"])
    c: dict[str, gr.Component] = {}

    c["summary_table"] = gr.Dataframe(label=t["summary"], interactive=False, elem_id="summary-df")
    c["dashboard_img"] = gr.Plot(label=t["dashboard"])
    c["figure_gallery"] = gr.Gallery(label=t["gallery"], columns=3, height=300)
    c["download_zip_btn"] = gr.File(label=t["download"], interactive=False)
    c["features_json_viewer"] = gr.Code(label=t["features"], language="json", interactive=False)
    c["analysis_status"] = gr.Markdown(t["empty"])

    return c


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _zip_figures(figure_paths: dict[str, str], output_dir: str | Path) -> str | None:
    if not figure_paths:
        return None
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    zip_path = out_root / "analysis_figures.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, file_path in figure_paths.items():
            p = Path(file_path)
            if p.exists() and p.is_file():
                zf.write(p, arcname=f"{key}{p.suffix}")
    return str(zip_path)


def activate_analysis_tab(engine: Any, analysis_config: dict[str, Any], layer_idx: int = 0) -> tuple[Any, ...]:
    """Run analysis pipeline and return values in build_analysis_tab() order."""
    if engine is None:
        msg = "No simulation engine found."
        return None, None, [], None, "{}", f"❌ {msg}"

    result = run_analysis(engine, analysis_config)
    summary = result.pipeline_output.get("summary", {})
    summary_df = pd.DataFrame([summary]) if isinstance(summary, dict) else pd.DataFrame(summary)

    dashboard_fig = renderer.render_dashboard(
        engine=engine,
        h_time=list(getattr(engine, "history", {}).get("time", [])),
        h_sigma=list(getattr(engine, "history", {}).get("opinion_std", [])),
        h_impact=list(getattr(engine, "history", {}).get("mean_impact", [])),
        h_events=list(getattr(engine, "history", {}).get("num_events", [])),
        layer_idx=int(layer_idx),
    )

    gallery_items = [(path, name) for name, path in result.figure_paths.items() if Path(path).exists()]
    zip_path = _zip_figures(result.figure_paths, result.config.get("output", {}).get("dir", tempfile.gettempdir()))

    features_json = result.pipeline_output.get("final_features") or result.pipeline_output
    status = "✅ Analysis completed" if not result.errors else f"⚠️ Analysis completed with warnings: {'; '.join(result.errors)}"

    return summary_df, dashboard_fig, gallery_items, zip_path, _safe_json(features_json), status
