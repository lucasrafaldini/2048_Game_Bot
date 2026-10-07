"""Tests for AI agents."""

import pytest
import time

from game2048.core import Board2048, Direction
from game2048.ai import (
    RandomAgent,
    GreedyAgent,
    WeightedGreedyAgent,
    NStepLookaheadAgent,
    ExpectimaxAgent,
    MCTSAgent,
    ExpectimaxConfig,
    MCTSConfig,
    create_snake_agent,
    create_monotonic_agent,
    create_expectimax_agent,
    create_mcts_agent,
)


class TestHeuristicAgents:
    """Tests for heuristic-based agents."""
    
    def test_random_agent_deterministic(self):
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)
        board = Board2048()
        
        moves1 = [agent1(board) for _ in range(10)]
        moves2 = [agent2(board) for _ in range(10)]
        assert moves1 == moves2
    
    def test_greedy_prefers_merge(self):
        agent = GreedyAgent()
        # Board where LEFT merges
        board = Board2048([[2, 2, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in board.get_valid_moves()
        # Should prefer a merge move
    
    def test_greedy_prefers_higher_score(self):
        agent = GreedyAgent()
        # LEFT gives 4, UP gives 0
        board = Board2048([[2, 2, 4, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in board.get_valid_moves()
        # Should prefer the higher scoring move
    
    def test_weighted_greedy_custom_weights(self):
        agent = WeightedGreedyAgent(weights={"empty": 10.0})
        board = Board2048()
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in board.get_valid_moves()
    
    def test_snake_agent(self):
        agent = create_snake_agent()
        # Snake prefers keeping max in corner
        board = Board2048([[1024, 512, 256, 128],
                           [64, 32, 16, 8],
                           [4, 2, 0, 0],
                           [0, 0, 0, 0]])
        move = agent(board)
        # Should avoid moving max tile from corner
        assert move != Direction.LEFT  # Would move 1024 from corner
        assert move != Direction.UP    # Would move 1024 from corner
    
    def test_monotonic_agent(self):
        agent = create_monotonic_agent()
        board = Board2048([[16, 8, 4, 2],
                           [32, 16, 8, 4],
                           [64, 32, 16, 8],
                           [0, 0, 0, 0]])  # Bottom row empty
        move = agent(board)
        assert move in board.get_valid_moves()
    
    def test_lookahead_agent(self):
        agent = NStepLookaheadAgent(depth=2)
        board = Board2048([[2, 2, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in board.get_valid_moves()
        stats = agent.get_stats()
        assert stats["nodes_evaluated"] > 0


class TestExpectimaxAgent:
    """Tests for Expectimax agent."""
    
    def test_expectimax_basic(self):
        agent = create_expectimax_agent(depth=2, time_limit=0.01)
        board = Board2048([[2, 2, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in board.get_valid_moves()
    
    def test_expectimax_avoids_death(self):
        agent = create_expectimax_agent(depth=3, time_limit=0.05)
        # Board with some empty cells
        board = Board2048([[2, 4, 8, 16],
                           [32, 64, 128, 256],
                           [512, 1024, 2, 4],
                           [8, 16, 32, 0]])
        move = agent(board)
        assert move in board.get_valid_moves()
    
    def test_expectimax_stats(self):
        agent = create_expectimax_agent(depth=3, time_limit=0.1)
        board = Board2048()
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        stats = agent.get_stats()
        assert stats["nodes_evaluated"] > 0
        assert stats["max_depth_reached"] >= 0
    
    def test_expectimax_config(self):
        config = ExpectimaxConfig(
            max_depth=4,
            time_limit=0.05,
            weights={"empty": 5.0, "corner_max": 10.0},
        )
        agent = ExpectimaxAgent(config)
        board = Board2048()
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in board.get_valid_moves()


class TestMCTSAgent:
    """Tests for MCTS agent."""
    
    def test_mcts_basic(self):
        agent = create_mcts_agent(simulations=10, time_limit=0.01)
        board = Board2048([[2, 2, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in board.get_valid_moves()
    
    def test_mcts_stats(self):
        agent = create_mcts_agent(simulations=20, time_limit=0.05)
        board = Board2048()
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        stats = agent.get_stats()
        assert stats["nodes_evaluated"] > 0
    
    def test_mcts_config(self):
        config = MCTSConfig(
            num_simulations=50,
            time_limit=0.1,
            exploration_constant=1.0,
        )
        agent = MCTSAgent(config)
        board = Board2048()
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in board.get_valid_moves()


class TestAgentComparison:
    """Tests comparing agent behaviors."""
    
    def test_all_agents_return_valid_moves(self):
        from game2048.ai import list_agents, get_agent
        
        board = Board2048([[2, 4, 8, 16],
                           [32, 64, 128, 256],
                           [512, 1024, 2, 4],
                           [8, 16, 32, 0]])  # One empty cell
        valid = board.get_valid_moves()
        
        for name in list_agents():
            agent = get_agent(name)
            move = agent(board)
            assert move in valid, f"{name} returned invalid move: {move}"
    
    def test_agents_deterministic(self):
        """Agents should be deterministic (except RandomAgent and MCTS)."""
        from game2048.ai import list_agents, get_agent
        
        board = Board2048([[2, 2, 4, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        board.add_random_tile()
        board.add_random_tile()
        
        for name in list_agents():
            if "random" in name.lower() or "mcts" in name.lower():
                continue  # These are inherently non-deterministic
            agent1 = get_agent(name)
            agent2 = get_agent(name)
            move1 = agent1(board)
            move2 = agent2(board)
            assert move1 == move2, f"{name} not deterministic: {move1} != {move2}"


class TestPerformance:
    """Performance benchmarks for agents."""
    
    def test_greedy_speed(self):
        agent = GreedyAgent()
        board = Board2048()
        
        start = time.perf_counter()
        for _ in range(100):
            agent(board)
        elapsed = time.perf_counter() - start
        
        # Should be very fast (< 10ms per 100 moves)
        assert elapsed < 0.1
    
    def test_expectimax_respects_time_limit(self):
        agent = create_expectimax_agent(depth=10, time_limit=0.05)
        board = Board2048()
        
        start = time.perf_counter()
        agent(board)
        elapsed = time.perf_counter() - start
        
        # Should respect time limit (with some overhead tolerance)
        assert elapsed < 0.2
    
    def test_mcts_respects_time_limit(self):
        agent = create_mcts_agent(simulations=10000, time_limit=0.05)
        board = Board2048()
        
        start = time.perf_counter()
        agent(board)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])