# game2048 - Modern 2048 AI Benchmark Framework

A complete rewrite of the classic 2048 bot with modern Python architecture, multiple AI agents, comprehensive benchmarking, and data science tooling.

## Features

- 🎮 **Pure Python Game Engine** - No external dependencies for core gameplay
- 🤖 **Multiple AI Agents** - Random, Greedy, Expectimax, MCTS, RL (DQN/PPO)
- 📊 **Benchmarking Framework** - Parallel execution, statistical analysis, CI
- 📈 **Visualization** - Plotly charts, Streamlit dashboard
- 💾 **Data Storage** - SQLite + Parquet for large-scale experiments
- 🔬 **Data Science** - Statistical tests, effect sizes, confidence intervals

## Installation

```bash
# Core dependencies
pip install -e .

# With RL support (requires PyTorch)
pip install -e .[rl]

# Development dependencies
pip install -e .[dev]

# Everything
pip install -e .[all]
```

## Quick Start

### Play a Game

```python
from game2048 import Game2048, get_agent

# Create game and agent
game = Game2048(seed=42)
agent = get_agent("expectimax")  # Gold-standard AI

# Play
result = game.play(agent, verbose=True)
print(f"Final Score: {result.final_score}")
print(f"Max Tile: {result.max_tile}")
print(f"Moves: {result.moves}")
```

### Run Benchmarks

```bash
# Quick test
game2048-benchmark quick -a expectimax -n 10

# Single agent
game2048-benchmark run expectimax -n 100 -o results.json

# Compare multiple agents
game2048-benchmark compare "expectimax,mcts,greedy,snake" -n 50 -o comparison.json
```

### High-Speed Simulation (bitboard engine)

The fast bitboard engine is a self-contained, ultra-fast 2048 implementation
verified move-by-move against the reference engine - ideal for accuracy
testing, high-speed turn simulation and statistical analysis.

```bash
# High-speed benchmark: verifies accuracy first, then simulates 1000 games
game2048-benchmark fast --policy greedy -n 1000 --seed 42 -o fast_results.json

# Disable the accuracy pre-check
game2048-benchmark fast --policy expectimax -n 200 --no-verify
```

```python
from game2048 import FastGame, greedy_policy, batch_simulate, summarize_results

# Single high-speed game (~45k moves/s, fully reproducible)
game = FastGame(seed=42)
result = game.play(greedy_policy)
print(f"Score: {result.final_score}, Max: {result.max_tile}, State: {result.state}")

# Batch simulation with statistics (parallel across CPU cores)
results = batch_simulate(greedy_policy, n_games=1000, seed=0)
summary = summarize_results(results)
print(f"Mean score: {summary.score_mean:.0f} ± {summary.score_ci95}")
print(f"P(reach 2048): {summary.win_rate:.1%}")

# Accuracy verification against the reference engine (Board2048)
from game2048.benchmarks.accuracy import run_accuracy_suite
report = run_accuracy_suite(n_fuzz_boards=2000, n_replay_games=20)
print(report.summary())  # "PASS: ... 0 mismatches"
```

How it works:
- Board is a single 64-bit integer (16 nibbles = tile exponents)
- Moves use precomputed 65536-entry row tables (4 lookups per move)
- Zero dependencies (stdlib only); spawns logged for exact replay
- Accuracy harness replays full games turn-by-turn on the reference engine

### Launch Dashboard

```bash
game2048-dashboard
# Then open http://localhost:8501
```

## Available Agents

| Agent | Description | Strength |
|-------|-------------|----------|
| `random` | Random valid moves | Baseline |
| `greedy` | 1-step heuristic lookahead | Weak |
| `snake` | Corner-max snake strategy | Medium |
| `monotonic` | Monotonicity-focused | Medium |
| `lookahead-2` | 2-step expectimax | Medium |
| `lookahead-3` | 3-step expectimax | Strong |
| `expectimax` | Full expectimax (depth 6, 0.1s) | **Strongest** |
| `expectimax-fast` | Expectimax (depth 4, 0.05s) | Strong |
| `expectimax-deep` | Expectimax (depth 8, 0.5s) | **Strongest** |
| `mcts` | MCTS (100 sims, 0.1s) | Strong |
| `mcts-fast` | MCTS (50 sims, 0.05s) | Medium |
| `mcts-deep` | MCTS (500 sims, 0.5s) | Strong |

## Architecture

```
src/game2048/
├── core/           # Game engine (board, game, constants)
├── ai/             # AI agents (heuristic, expectimax, mcts, rl)
├── benchmarks/     # Benchmark runner, metrics, CLI
├── visualization/  # Charts, Streamlit dashboard
├── data/           # Storage (SQLite, Parquet), analysis
├── env.py          # Gymnasium environment for RL
└── utils/          # Helper functions
```

## Advanced Usage

### Custom Agent

```python
from game2048 import BaseAgent, Direction, Board2048

class MyAgent(BaseAgent):
    def get_move(self, board: Board2048) -> Direction:
        # Your logic here
        return Direction.UP

# Use in benchmark
from game2048.benchmarks import BenchmarkRunner, BenchmarkConfig
runner = BenchmarkRunner(BenchmarkConfig(num_games=100))
result = runner.run(MyAgent, "MyAgent")
```

### Statistical Analysis

```python
from game2048.data import pairwise_comparison, generate_report
import pandas as pd

# Load results
df = pd.read_parquet("benchmark_data/*.parquet")

# Compare all agents
comparison = pairwise_comparison(df, metric="final_score")
print(comparison)

# Full report
report = generate_report(df)
print(report)
```

### Custom Heuristic Weights

```python
from game2048.ai import WeightedGreedyAgent

agent = WeightedGreedyAgent(weights={
    "empty": 3.0,
    "monotonicity": 2.0,
    "smoothness": 0.5,
    "corner_max": 5.0,  # Strongly prefer corner max
})
```

### RL Training (Optional)

```bash
# Train DQN
python -m game2048.ai.rl train_dqn --timesteps 1000000 --save models/dqn

# Train PPO
python -m game2048.ai.rl train_ppo --timesteps 2000000 --save models/ppo
```

## Benchmark Results (Typical)

| Agent | Win Rate | Avg Score | Max Tile | Avg Move Time |
|-------|----------|-----------|----------|---------------|
| Random | 0% | ~1,000 | 64 | <1ms |
| Greedy | ~5% | ~15,000 | 256 | <1ms |
| Snake | ~30% | ~45,000 | 512 | <1ms |
| Expectimax (fast) | ~80% | ~180,000 | 1024 | ~50ms |
| Expectimax (deep) | ~95% | ~350,000 | 2048 | ~500ms |
| MCTS | ~70% | ~150,000 | 1024 | ~100ms |

*Results vary by hardware and configuration.*

## Project Structure

```
2048_game_bot/
├── src/game2048/          # Main package
├── tests/                 # Unit tests
├── configs/               # Config templates
├── scripts/               # Helper scripts
├── notebooks/             # Jupyter notebooks
├── benchmark_results/     # Output directory
├── pyproject.toml         # Package config
└── README.md              # This file
```

## Development

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
ruff format .

# Type check
mypy src/game2048

# Run pre-commit
pre-commit run --all-files
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original concept by Gabriele Cirulli
- Expectimax implementation inspired by various 2048 AI research
- The PC Geek's YouTube tutorials for the original bot approach