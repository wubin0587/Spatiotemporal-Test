"""Shared constants for analysis output and rendering defaults."""

from pathlib import Path

# Repository root directory (one level above ``analysis`` package).
ROOT_DIR = Path(__file__).resolve().parent.parent

# Default directory for analysis artifacts.
OUTPUT_DIR = ROOT_DIR / "output"

# Default color theme used by analysis visualizations.
DEFAULT_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "background": "#ffffff",
    "text": "#111111",
}

# Default language used by report and parser modules.
DEFAULT_LANGUAGE = "zh"
