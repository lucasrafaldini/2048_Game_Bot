"""AI agents for 2048."""

from .base import Agent, BaseAgent, HeuristicAgent
from .heuristic import (
    RandomAgent,
    GreedyAgent,
    NStepLookaheadAgent,
    WeightedGreedyAgent,
    create_snake_agent,
    create_monotonic_agent,
)
from .expectimax import ExpectimaxAgent, OptimizedExpectimaxAgent, ExpectimaxConfig, create_expectimax_agent
from .mcts import MCTSAgent, MCTSConfig, create_mcts_agent

# RL agents (optional)
try:
    from .rl import RLAgent, DQNAgent, PPOAgent, create_random_rl_agent, train_dqn, train_ppo
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False
    RLAgent = DQNAgent = PPOAgent = None
    create_random_rl_agent = train_dqn = train_ppo = None

__all__ = [
    "Agent",
    "BaseAgent",
    "HeuristicAgent",
    "RandomAgent",
    "GreedyAgent",
    "NStepLookaheadAgent",
    "WeightedGreedyAgent",
    "create_snake_agent",
    "create_monotonic_agent",
    "ExpectimaxAgent",
    "OptimizedExpectimaxAgent",
    "ExpectimaxConfig",
    "create_expectimax_agent",
    "MCTSAgent",
    "MCTSConfig",
    "create_mcts_agent",
]

if _RL_AVAILABLE:
    __all__ += [
        "RLAgent",
        "DQNAgent",
        "PPOAgent",
        "create_random_rl_agent",
        "train_dqn",
        "train_ppo",
    ]

# Registry of all available agents
AGENT_REGISTRY = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "snake": create_snake_agent,
    "monotonic": create_monotonic_agent,
    "lookahead-2": lambda: NStepLookaheadAgent(depth=2),
    "lookahead-3": lambda: NStepLookaheadAgent(depth=3),
    "expectimax": lambda: create_expectimax_agent(depth=6, time_limit=0.1),
    "expectimax-fast": lambda: create_expectimax_agent(depth=4, time_limit=0.05),
    "expectimax-deep": lambda: create_expectimax_agent(depth=8, time_limit=0.5),
    "mcts": lambda: create_mcts_agent(simulations=100, time_limit=0.1),
    "mcts-fast": lambda: create_mcts_agent(simulations=50, time_limit=0.05),
    "mcts-deep": lambda: create_mcts_agent(simulations=500, time_limit=0.5),
}


def get_agent(name: str) -> BaseAgent:
    """Get agent by name from registry."""
    if name not in AGENT_REGISTRY:
        available = ", ".join(sorted(AGENT_REGISTRY.keys()))
        raise ValueError(f"Unknown agent: {name}. Available: {available}")
    return AGENT_REGISTRY[name]()


def list_agents() -> list[str]:
    """List all available agent names."""
    return sorted(AGENT_REGISTRY.keys())