"""CLI for running benchmarks."""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .runner import BenchmarkConfig, BenchmarkRunner, run_quick_benchmark
from ..ai import get_agent, list_agents, AGENT_REGISTRY

app = typer.Typer(
    name="game2048-benchmark",
    help="Benchmark 2048 AI agents",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    agent: str = typer.Argument(..., help="Agent name (use 'list' to see available)"),
    games: int = typer.Option(100, "-n", "--games", help="Number of games to play"),
    seeds: Optional[str] = typer.Option(None, "-s", "--seeds", help="Comma-separated list of seeds"),
    max_moves: int = typer.Option(10000, "-m", "--max-moves", help="Maximum moves per game"),
    parallel: bool = typer.Option(True, "-p/--no-parallel", help="Run in parallel"),
    workers: Optional[int] = typer.Option(None, "-w", "--workers", help="Number of worker processes"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    history: bool = typer.Option(False, "--history", help="Record board history"),
):
    """Run benchmark for a single agent."""
    if agent == "list":
        list_agents_cmd()
        return
    
    try:
        agent_factory = lambda: get_agent(agent)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    seed_list = None
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    
    config = BenchmarkConfig(
        num_games=games,
        seeds=seed_list,
        max_moves=max_moves,
        parallel=parallel,
        num_workers=workers,
        verbose=verbose,
        record_history=history,
    )
    
    runner = BenchmarkRunner(config)
    result = runner.run(agent_factory, agent)
    
    if output:
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
# Module-level factory for pickling
def _make_agent_factory(agent_name: str):
    """Create a picklable factory function for an agent."""
    from game2048.ai import get_agent
    return get_agent(agent_name)


def _agent_factory(agent_name: str):
    """Picklable factory function."""
    return get_agent(agent_name)


@app.command()
def compare(
    agents: str = typer.Argument(..., help="Comma-separated agent names"),
    games: int = typer.Option(100, "-n", "--games", help="Number of games per agent"),
    seeds: Optional[str] = typer.Option(None, "-s", "--seeds", help="Comma-separated list of seeds"),
    max_moves: int = typer.Option(10000, "-m", "--max-moves", help="Maximum moves per game"),
    parallel: bool = typer.Option(True, "-p/--no-parallel", help="Run in parallel"),
    workers: Optional[int] = typer.Option(None, "-w", "--workers", help="Number of worker processes"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """Compare multiple agents."""
    agent_names = [a.strip() for a in agents.split(",")]
    
    factories = {}
    for name in agent_names:
        try:
            # Use functools.partial to create a picklable factory
            import functools
            factories[name] = functools.partial(_agent_factory, name)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    seed_list = None
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    
    config = BenchmarkConfig(
        num_games=games,
        seeds=seed_list,
        max_moves=max_moves,
        parallel=parallel,
        num_workers=workers,
        verbose=verbose,
    )
    
    runner = BenchmarkRunner(config)
    comparison = runner.compare(factories)
    
    if output:
        with open(output, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def quick(
    agent: str = typer.Option("expectimax", "-a", "--agent", help="Agent to test"),
    games: int = typer.Option(10, "-n", "--games", help="Number of games"),
):
    """Quick test benchmark."""
    run_quick_benchmark(agent, games)


@app.command()
def list():
    """List available agents."""
    list_agents_cmd()


def list_agents_cmd() -> None:
    """List all available agents."""
    table = Table(title="Available Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    
    descriptions = {
        "random": "Random valid moves",
        "greedy": "Greedy heuristic (1-step lookahead)",
        "snake": "Snake strategy (corner max tile)",
        "monotonic": "Monotonicity-focused heuristic",
        "lookahead-2": "2-step expectimax lookahead",
        "lookahead-3": "3-step expectimax lookahead",
        "expectimax": "Expectimax depth 6, 0.1s time limit",
        "expectimax-fast": "Expectimax depth 4, 0.05s time limit",
        "expectimax-deep": "Expectimax depth 8, 0.5s time limit",
        "mcts": "MCTS 100 simulations, 0.1s time limit",
        "mcts-fast": "MCTS 50 simulations, 0.05s time limit",
        "mcts-deep": "MCTS 500 simulations, 0.5s time limit",
    }
    
    for name in sorted(AGENT_REGISTRY.keys()):
        desc = descriptions.get(name, "Custom agent")
        table.add_row(name, desc)
    
    console.print(table)


@app.command()
def config(
    output: Path = typer.Option("benchmark.yaml", "-o", "--output", help="Output config file"),
):
    """Generate a sample benchmark configuration file."""
    config = {
        "benchmark": {
            "num_games": 100,
            "max_moves": 10000,
            "parallel": True,
            "num_workers": 4,
            "record_history": False,
        },
        "agents": {
            "expectimax": {"depth": 6, "time_limit": 0.1},
            "expectimax-fast": {"depth": 4, "time_limit": 0.05},
            "mcts": {"simulations": 100, "time_limit": 0.1},
            "greedy": {},
            "snake": {},
        },
        "comparison": {
            "agents": ["expectimax", "mcts", "greedy", "snake"],
            "num_games": 50,
        },
    }
    
    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    console.print(f"[green]Config template saved to {output}[/green]")


@app.command()
def from_config(
    config_file: Path = typer.Argument(..., help="Benchmark config YAML file"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
):
    """Run benchmark from config file."""
    import functools
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    bench_config = config.get("benchmark", {})
    runner_config = BenchmarkConfig(**bench_config)
    runner = BenchmarkRunner(runner_config)
    
    # Single agent runs
    agents_config = config.get("agents", {})
    for name, params in agents_config.items():
        try:
            if name in AGENT_REGISTRY:
                factory = functools.partial(_agent_factory, name)
            else:
                console.print(f"[yellow]Unknown agent: {name}[/yellow]")
                continue
            
            runner.run(factory, name)
        except Exception as e:
            console.print(f"[red]Error running {name}: {e}[/red]")
    
    # Comparison
    compare_config = config.get("comparison", {})
    if compare_config:
        agent_names = compare_config.get("agents", [])
        factories = {name: functools.partial(_agent_factory, name) for name in agent_names}
        comparison = runner.compare(factories)
        
        if output:
            with open(output, "w") as f:
                json.dump(comparison.to_dict(), f, indent=2)
            console.print(f"[green]Comparison saved to {output}[/green]")


if __name__ == "__main__":
    app()