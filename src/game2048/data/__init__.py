"""Data storage and analysis for benchmarks."""

from .storage import BenchmarkDatabase, ParquetStore, create_database, create_parquet_store
from .analysis import (
    StatisticalTest,
    compare_agents,
    pairwise_comparison,
    compute_confidence_intervals,
    analyze_tile_distribution,
    learning_curve_analysis,
    compute_effect_sizes,
    generate_report,
)

__all__ = [
    "BenchmarkDatabase",
    "ParquetStore",
    "create_database",
    "create_parquet_store",
    "StatisticalTest",
    "compare_agents",
    "pairwise_comparison",
    "compute_confidence_intervals",
    "analyze_tile_distribution",
    "learning_curve_analysis",
    "compute_effect_sizes",
    "generate_report",
]