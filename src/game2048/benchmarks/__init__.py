"""Benchmarking framework for 2048 agents."""

from .metrics import BenchmarkResult, ComparisonResult, GameMetrics
from .runner import BenchmarkConfig, BenchmarkRunner, run_quick_benchmark


def benchmark_cli():
    """Entry point for the benchmark CLI (lazy import avoids heavy deps and
    keeps `python -m game2048.benchmarks.cli` free of re-import warnings)."""
    from .cli import app
    return app


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ComparisonResult",
    "GameMetrics",
    "benchmark_cli",
    "run_quick_benchmark",
]
