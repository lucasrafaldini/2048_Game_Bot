"""Visualization and dashboard for 2048 benchmarks."""

from .charts import (
    create_score_distribution_plot,
    create_max_tile_comparison,
    create_win_rate_chart,
    create_performance_scatter,
    create_gameplay_animation,
    create_comparison_dashboard,
    create_summary_table,
    save_all_charts,
)
from .streamlit_app import main as run_dashboard

__all__ = [
    "create_score_distribution_plot",
    "create_max_tile_comparison",
    "create_win_rate_chart",
    "create_performance_scatter",
    "create_gameplay_animation",
    "create_comparison_dashboard",
    "create_summary_table",
    "save_all_charts",
    "run_dashboard",
]