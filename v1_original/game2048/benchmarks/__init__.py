"""Benchmarking framework for 2048 agents."""

from .metrics import BenchmarkResult, ComparisonResult, GameMetrics
from .runner import BenchmarkConfig, BenchmarkRunner, run_quick_benchmark
from .cli import app as benchmark_cli

__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "GameMetrics",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "run_quick_benchmark",
    "benchmark_cli",
]