"""Visualization and dashboard for 2048 benchmarks."""

from .charts import (
    create_comparison_dashboard,
    create_gameplay_animation,
    create_max_tile_comparison,
    create_performance_scatter,
    create_score_distribution_plot,
    create_summary_table,
    create_win_rate_chart,
    save_all_charts,
)


def run_dashboard(*args, **kwargs):
    """Launch the Streamlit dashboard (lazy import keeps Streamlit out of
    the package's import path, avoiding bare-mode ScriptRunContext
    warnings)."""
    from .streamlit_app import main
    return main(*args, **kwargs)


__all__ = [
    "create_comparison_dashboard",
    "create_gameplay_animation",
    "create_max_tile_comparison",
    "create_performance_scatter",
    "create_score_distribution_plot",
    "create_summary_table",
    "create_win_rate_chart",
    "run_dashboard",
    "save_all_charts",
]
