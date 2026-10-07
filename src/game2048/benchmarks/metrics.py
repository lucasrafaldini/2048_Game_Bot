"""Metrics and data structures for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class GameMetrics:
    """Metrics for a single game."""
    agent_name: str
    seed: int
    final_score: int
    max_tile: int
    moves: int
    duration_seconds: float
    state: str  # "won", "lost", "playing"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Per-move statistics
    move_times: list[float] = field(default_factory=list)
    nodes_evaluated: list[int] = field(default_factory=list)
    
    # Board snapshots (optional, for analysis)
    board_history: list[np.ndarray] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "seed": self.seed,
            "final_score": self.final_score,
            "max_tile": self.max_tile,
            "moves": self.moves,
            "duration_seconds": self.duration_seconds,
            "state": self.state,
            "timestamp": self.timestamp,
            "avg_move_time": np.mean(self.move_times) if self.move_times else 0,
            "max_move_time": np.max(self.move_times) if self.move_times else 0,
            "total_nodes": sum(self.nodes_evaluated),
            "avg_nodes_per_move": np.mean(self.nodes_evaluated) if self.nodes_evaluated else 0,
        }
    
    @property
    def won(self) -> bool:
        return self.state == "won"
    
    @property
    def max_tile_log2(self) -> int:
        import math
        return int(math.log2(self.max_tile)) if self.max_tile > 0 else 0


@dataclass
class BenchmarkResult:
    """Aggregated results for a benchmark run."""
    agent_name: str
    num_games: int
    seeds: list[int]
    
    # Aggregated metrics
    games: list[GameMetrics] = field(default_factory=list)
    
    # Summary statistics
    win_rate: float = 0.0
    avg_score: float = 0.0
    median_score: float = 0.0
    std_score: float = 0.0
    max_score: int = 0
    min_score: int = 0
    
    avg_max_tile: float = 0.0
    max_tile_distribution: dict[int, int] = field(default_factory=dict)  # log2 -> count
    
    avg_moves: float = 0.0
    avg_duration: float = 0.0
    avg_move_time: float = 0.0
    
    # Percentiles
    score_percentiles: dict[int, float] = field(default_factory=dict)
    
    def add_game(self, metrics: GameMetrics) -> None:
        self.games.append(metrics)
        self._recompute()
    
    def _recompute(self) -> None:
        if not self.games:
            return
        
        scores = [g.final_score for g in self.games]
        max_tiles = [g.max_tile for g in self.games]
        moves = [g.moves for g in self.games]
        durations = [g.duration_seconds for g in self.games]
        move_times = [g.move_times for g in self.games if g.move_times]
        
        self.num_games = len(self.games)
        self.win_rate = sum(1 for g in self.games if g.won) / self.num_games
        self.avg_score = float(np.mean(scores))
        self.median_score = float(np.median(scores))
        self.std_score = float(np.std(scores))
        self.max_score = int(np.max(scores))
        self.min_score = int(np.min(scores))
        
        self.avg_max_tile = float(np.mean(max_tiles))
        self.max_tile_distribution = {}
        for mt in max_tiles:
            log2 = int(np.log2(mt)) if mt > 0 else 0
            self.max_tile_distribution[log2] = self.max_tile_distribution.get(log2, 0) + 1
        
        self.avg_moves = float(np.mean(moves))
        self.avg_duration = float(np.mean(durations))
        
        all_move_times = [t for mts in move_times for t in mts]
        self.avg_move_time = float(np.mean(all_move_times)) if all_move_times else 0.0
        
        # Percentiles
        for p in [10, 25, 50, 75, 90, 95, 99]:
            self.score_percentiles[p] = float(np.percentile(scores, p))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "num_games": self.num_games,
            "seeds": self.seeds,
            "win_rate": self.win_rate,
            "avg_score": self.avg_score,
            "median_score": self.median_score,
            "std_score": self.std_score,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "avg_max_tile": self.avg_max_tile,
            "max_tile_distribution": self.max_tile_distribution,
            "avg_moves": self.avg_moves,
            "avg_duration": self.avg_duration,
            "avg_move_time": self.avg_move_time,
            "score_percentiles": self.score_percentiles,
            "games": [g.to_dict() for g in self.games],
        }


@dataclass
class ComparisonResult:
    """Results comparing multiple agents."""
    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult) -> None:
        self.results[result.agent_name] = result
    
    def get_ranking(self, metric: str = "avg_score") -> list[tuple[str, float]]:
        """Get agents ranked by metric."""
        ranking = []
        for name, result in self.results.items():
            val = getattr(result, metric, 0)
            ranking.append((name, val))
        return sorted(ranking, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> dict[str, Any]:
        return {name: result.to_dict() for name, result in self.results.items()}