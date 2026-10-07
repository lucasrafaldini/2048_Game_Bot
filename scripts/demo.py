#!/usr/bin/env python3
"""
Demo script showing the complete 2048 AI benchmark workflow.
"""

from game2048 import (
    Game2048,
    get_agent,
    list_agents,
    BenchmarkRunner,
    BenchmarkConfig,
)
from game2048.data import generate_report
from game2048.visualization import create_comparison_dashboard, create_summary_table
import pandas as pd


def demo_single_game():
    """Play a single game with an AI agent."""
    print("=" * 60)
    print("DEMO: Single Game with Expectimax Agent")
    print("=" * 60)
    
    game = Game2048(seed=42)
    agent = get_agent("expectimax-fast")
    
    print(f"Agent: {agent.name}")
    print(f"Initial board:")
    print(game.board)
    print()
    
    result = game.play(agent, max_moves=1000, verbose=False)
    
    print(f"Final Score: {result.final_score:,}")
    print(f"Max Tile: {result.max_tile}")
    print(f"Moves: {result.moves}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"State: {result.state}")
    print()
    
    return result


def demo_quick_benchmark():
    """Run a quick benchmark comparing multiple agents."""
    print("=" * 60)
    print("DEMO: Quick Benchmark (3 games each)")
    print("=" * 60)
    
    config = BenchmarkConfig(num_games=3, parallel=False, verbose=True)
    runner = BenchmarkRunner(config)
    
    agents = {
        "random": lambda: get_agent("random"),
        "greedy": lambda: get_agent("greedy"),
        "snake": lambda: get_agent("snake"),
        "expectimax-fast": lambda: get_agent("expectimax-fast"),
    }
    
    comparison = runner.compare(agents)
    
    print("\nComparison Results:")
    print("-" * 60)
    
    # Show summary table
    df = create_summary_table(comparison.to_dict())
    print(df.to_string(index=False))
    print()
    
    return comparison


def demo_statistical_analysis():
    """Demonstrate statistical analysis capabilities."""
    print("=" * 60)
    print("DEMO: Statistical Analysis")
    print("=" * 60)
    
    # Run a larger benchmark for meaningful statistics (sequential to avoid pickle issues)
    config = BenchmarkConfig(num_games=20, parallel=False)
    runner = BenchmarkRunner(config)
    
    agents = {
        "greedy": lambda: get_agent("greedy"),
        "snake": lambda: get_agent("snake"),
        "expectimax-fast": lambda: get_agent("expectimax-fast"),
    }
    
    comparison = runner.compare(agents)
    
    # Convert to DataFrame for analysis
    all_games = []
    for agent_name, result in comparison.results.items():
        for game in result.games:
            game_dict = game.to_dict()
            game_dict["agent_name"] = agent_name
            all_games.append(game_dict)
    
    df = pd.DataFrame(all_games)
    
    # Generate report
    report = generate_report(df)
    print(report)
    
    return df


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("=" * 60)
    print("DEMO: Visualization")
    print("=" * 60)
    
    config = BenchmarkConfig(num_games=10, parallel=True, num_workers=2)
    runner = BenchmarkRunner(config)
    
    agents = {
        "greedy": lambda: get_agent("greedy"),
        "snake": lambda: get_agent("snake"),
    }
    
    comparison = runner.compare(agents)
    
    # Create dashboard
    fig = create_comparison_dashboard(comparison.to_dict())
    
    # Save as HTML
    fig.write_html("demo_dashboard.html")
    print("Dashboard saved to demo_dashboard.html")
    print("Open in browser to view interactive charts")
    print()
    
    return fig


def main():
    print("🎮 2048 AI Benchmark Framework - Demo")
    print()
    
    # Show available agents
    print("Available agents:")
    for name in list_agents():
        print(f"  - {name}")
    print()
    
    # Run demos
    demo_single_game()
    demo_quick_benchmark()
    demo_statistical_analysis()
    demo_visualization()
    
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  - Run full benchmarks: game2048-benchmark compare 'expectimax,mcts,greedy,snake' -n 100")
    print("  - Launch dashboard: game2048-dashboard")
    print("  - Train RL agents: pip install game2048[rl] && python -m game2048.ai.rl train_dqn")


if __name__ == "__main__":
    main()