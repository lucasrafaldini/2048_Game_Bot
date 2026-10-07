"""Benchmark runner for evaluating agents."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from .metrics import BenchmarkResult, ComparisonResult, GameMetrics
from ..core.game import Game2048
from ..core.constants import Direction
from ..ai.base import BaseAgent


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    num_games: int = 100
    seeds: Optional[list[int]] = None
    max_moves: int = 10000
    parallel: bool = True
    num_workers: Optional[int] = None
    record_history: bool = False
    verbose: bool = False


def _run_single_game(args: tuple) -> GameMetrics:
    """Run a single game (for parallel execution)."""
    agent_factory, seed, max_moves, record_history = args
    
    # Create agent instance
    agent = agent_factory()
    
    # Create game
    game = Game2048(seed=seed, record_history=record_history)
    
    # Track move times and nodes
    move_times = []
    nodes_evaluated = []
    
    original_get_move = agent.get_move
    
    def timed_get_move(board):
        start = time.perf_counter()
        move = original_get_move(board)
        elapsed = time.perf_counter() - start
        move_times.append(elapsed)
        stats = agent.get_stats()
        nodes_evaluated.append(stats.get("nodes_evaluated", 0))
        return move
    
    agent.get_move = timed_get_move
    
    # Play game
    result = game.play(agent, max_moves=max_moves)
    
    return GameMetrics(
        agent_name=agent.name,
        seed=seed,
        final_score=result.final_score,
        max_tile=result.max_tile,
        moves=result.moves,
        duration_seconds=result.duration_seconds,
        state=result.state,
        move_times=move_times,
        nodes_evaluated=nodes_evaluated,
        board_history=result.board_history if record_history else [],
    )


class BenchmarkRunner:
    """Run benchmarks for one or more agents."""
    
    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self.console = Console()
    
    def run(
        self,
        agent_factory: Callable[[], BaseAgent],
        agent_name: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run benchmark for a single agent."""
        name = agent_name or agent_factory().name
        seeds = self.config.seeds or list(range(self.config.num_games))
        
        self.console.print(f"\n[bold cyan]Benchmarking {name}[/bold cyan]")
        self.console.print(f"Games: {len(seeds)} | Max moves: {self.config.max_moves}")
        
        result = BenchmarkResult(agent_name=name, num_games=0, seeds=seeds)
        
        if self.config.parallel and len(seeds) > 1:
            result = self._run_parallel(agent_factory, seeds, name)
        else:
            result = self._run_sequential(agent_factory, seeds, name)
        
        self._print_summary(result)
        return result
    
    def _run_sequential(
        self,
        agent_factory: Callable[[], BaseAgent],
        seeds: list[int],
        name: str,
    ) -> BenchmarkResult:
        result = BenchmarkResult(agent_name=name, num_games=0, seeds=seeds)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Running {name}", total=len(seeds))
            
            for seed in seeds:
                metrics = _run_single_game((
                    agent_factory, seed, self.config.max_moves, self.config.record_history
                ))
                result.add_game(metrics)
                progress.advance(task)
                
                if self.config.verbose:
                    self.console.print(
                        f"  Seed {seed}: Score={metrics.final_score}, "
                        f"Max={metrics.max_tile}, Moves={metrics.moves}, "
                        f"State={metrics.state}"
                    )
        
        return result
    
    def _run_parallel(
        self,
        agent_factory: Callable[[], BaseAgent],
        seeds: list[int],
        name: str,
    ) -> BenchmarkResult:
        result = BenchmarkResult(agent_name=name, num_games=0, seeds=seeds)
        
        args_list = [
            (agent_factory, seed, self.config.max_moves, self.config.record_history)
            for seed in seeds
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Running {name}", total=len(seeds))
            
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {executor.submit(_run_single_game, args): args[1] for args in args_list}
                
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        metrics = future.result()
                        result.add_game(metrics)
                    except Exception as e:
                        self.console.print(f"[red]Error on seed {seed}: {e}[/red]")
                    progress.advance(task)
        
        return result
    
    def compare(
        self,
        agent_factories: dict[str, Callable[[], BaseAgent]],
    ) -> ComparisonResult:
        """Run benchmarks for multiple agents and compare."""
        self.console.print("\n[bold magenta]Running Comparison Benchmark[/bold magenta]")
        
        comparison = ComparisonResult()
        
        for name, factory in agent_factories.items():
            result = self.run(factory, name)
            comparison.add_result(result)
        
        self._print_comparison(comparison)
        return comparison
    
    def _print_summary(self, result: BenchmarkResult) -> None:
        table = Table(title=f"Results: {result.agent_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Games Played", str(result.num_games))
        table.add_row("Win Rate", f"{result.win_rate:.1%}")
        table.add_row("Avg Score", f"{result.avg_score:,.0f}")
        table.add_row("Median Score", f"{result.median_score:,.0f}")
        table.add_row("Std Score", f"{result.std_score:,.0f}")
        table.add_row("Max Score", f"{result.max_score:,}")
        table.add_row("Min Score", f"{result.min_score:,}")
        table.add_row("Avg Max Tile", f"{result.avg_max_tile:,.0f}")
        table.add_row("Avg Moves", f"{result.avg_moves:.0f}")
        table.add_row("Avg Duration", f"{result.avg_duration:.2f}s")
        table.add_row("Avg Move Time", f"{result.avg_move_time*1000:.2f}ms")
        
        # Max tile distribution
        dist_str = ", ".join(f"2^{k}: {v}" for k, v in sorted(result.max_tile_distribution.items()))
        table.add_row("Max Tile Dist.", dist_str)
        
        self.console.print(table)
    
    def _print_comparison(self, comparison: ComparisonResult) -> None:
        table = Table(title="Agent Comparison")
        table.add_column("Agent", style="cyan")
        table.add_column("Win Rate", style="green")
        table.add_column("Avg Score", style="yellow")
        table.add_column("Median", style="yellow")
        table.add_column("Max Tile", style="magenta")
        table.add_column("Avg Moves", style="blue")
        table.add_column("Move Time (ms)", style="blue")
        
        # Sort by avg score
        ranked = comparison.get_ranking("avg_score")
        
        for name, _ in ranked:
            result = comparison.results[name]
            table.add_row(
                name,
                f"{result.win_rate:.1%}",
                f"{result.avg_score:,.0f}",
                f"{result.median_score:,.0f}",
                f"{result.avg_max_tile:,.0f}",
                f"{result.avg_moves:.0f}",
                f"{result.avg_move_time*1000:.2f}",
            )
        
        self.console.print(table)


def run_quick_benchmark(agent_name: str = "expectimax", num_games: int = 10) -> BenchmarkResult:
    """Quick benchmark for testing."""
    from ..ai import get_agent
    
    config = BenchmarkConfig(
        num_games=num_games,
        parallel=False,
        verbose=True,
    )
    runner = BenchmarkRunner(config)
    return runner.run(lambda: get_agent(agent_name), agent_name)