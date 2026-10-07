"""
game2048 - Modern 2048 Game Engine with AI Agents and Benchmarking

A complete rewrite of the classic 2048 bot with:
- Pure Python game engine (no external dependencies for core)
- Multiple AI agents: Random, Greedy, Expectimax, MCTS, RL
- Comprehensive benchmarking framework
- Data science tooling: statistics, visualization, storage
- Streamlit dashboard for interactive analysis

Quick Start:
    >>> from game2048 import Game2048, get_agent
    >>> game = Game2048()
    >>> agent = get_agent("expectimax")
    >>> result = game.play(agent)
    >>> print(f"Score: {result.final_score}, Max Tile: {result.max_tile}")

Run Benchmarks:
    $ game2048-benchmark compare "expectimax,mcts,greedy,snake" -n 100

Launch Dashboard:
    $ game2048-dashboard
"""

from .core import (
    Board2048,
    Game2048,
    GameResult,
    Direction,
    GameState,
    BOARD_SIZE,
    WIN_TILE,
    simulate_move,
)
from .ai import (
    Agent,
    BaseAgent,
    HeuristicAgent,
    RandomAgent,
    GreedyAgent,
    NStepLookaheadAgent,
    WeightedGreedyAgent,
    ExpectimaxAgent,
    MCTSAgent,
    get_agent,
    list_agents,
    AGENT_REGISTRY,
    create_snake_agent,
    create_monotonic_agent,
    create_expectimax_agent,
    create_mcts_agent,
)
from .benchmarks import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkResult,
    ComparisonResult,
    GameMetrics,
    run_quick_benchmark,
)
from .visualization import (
    create_score_distribution_plot,
    create_max_tile_comparison,
    create_win_rate_chart,
    create_performance_scatter,
    create_comparison_dashboard,
    create_summary_table,
    run_dashboard,
)
from .data import (
    BenchmarkDatabase,
    ParquetStore,
    compare_agents,
    pairwise_comparison,
    compute_confidence_intervals,
    generate_report,
)
from .utils import (
    setup_logging,
    get_logger,
    format_board,
    calculate_tile_entropy,
    calculate_monotonicity,
    calculate_smoothness,
    get_game_phase,
)

__version__ = "2.0.0"

__all__ = [
    # Core
    "Board2048",
    "Game2048",
    "GameResult",
    "Direction",
    "GameState",
    "BOARD_SIZE",
    "WIN_TILE",
    "simulate_move",
    # AI
    "Agent",
    "BaseAgent",
    "HeuristicAgent",
    "RandomAgent",
    "GreedyAgent",
    "NStepLookaheadAgent",
    "WeightedGreedyAgent",
    "ExpectimaxAgent",
    "MCTSAgent",
    "get_agent",
    "list_agents",
    "AGENT_REGISTRY",
    "create_snake_agent",
    "create_monotonic_agent",
    "create_expectimax_agent",
    "create_mcts_agent",
    # Benchmarks
    "BenchmarkConfig",
    "BenchmarkRunner",
    "BenchmarkResult",
    "ComparisonResult",
    "GameMetrics",
    "run_quick_benchmark",
    # Visualization
    "create_score_distribution_plot",
    "create_max_tile_comparison",
    "create_win_rate_chart",
    "create_performance_scatter",
    "create_comparison_dashboard",
    "create_summary_table",
    "run_dashboard",
    # Data
    "BenchmarkDatabase",
    "ParquetStore",
    "compare_agents",
    "pairwise_comparison",
    "compute_confidence_intervals",
    "generate_report",
    # Utils
    "setup_logging",
    "get_logger",
    "format_board",
    "calculate_tile_entropy",
    "calculate_monotonicity",
    "calculate_smoothness",
    "get_game_phase",
]