"""Tests for 2048 game engine."""

import pytest
import numpy as np

from game2048.core import (
    Board2048,
    Game2048,
    Direction,
    GameState,
    simulate_move,
)
from game2048.ai import (
    RandomAgent,
    GreedyAgent,
    get_agent,
    list_agents,
)
from game2048.utils import (
    format_board,
    calculate_monotonicity,
    calculate_smoothness,
    get_game_phase,
)


class TestBoard2048:
    """Tests for Board2048 core logic."""
    
    def test_empty_board(self):
        board = Board2048()
        assert board.max_tile == 0
        assert board.score == 0
        assert len(board.empty_cells) == 16
        assert board.get_state() == "playing"
    
    def test_initial_tiles(self):
        board = Board2048()
        board.add_random_tile()
        board.add_random_tile()
        assert len(board.empty_cells) == 14
        assert board.max_tile in (2, 4)
    
    def test_move_left_merge(self):
        # [2, 2, 0, 0] -> [4, 0, 0, 0]
        board = Board2048([[2, 2, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        changed, gained = board.move(Direction.LEFT)
        assert changed
        assert gained == 4
        assert board.grid[0] == [4, 0, 0, 0]
    
    def test_move_left_no_merge(self):
        # [2, 4, 0, 0] -> [2, 4, 0, 0] (already compressed, no change)
        board = Board2048([[2, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        changed, gained = board.move(Direction.LEFT)
        assert not changed  # No change - already compressed
        assert gained == 0
        assert board.grid[0] == [2, 4, 0, 0]
    
    def test_move_right(self):
        board = Board2048([[0, 0, 2, 2],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        changed, gained = board.move(Direction.RIGHT)
        assert changed
        assert gained == 4
        assert board.grid[0] == [0, 0, 0, 4]
    
    def test_move_up(self):
        board = Board2048([[0, 2, 0, 0],
                           [0, 2, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        changed, gained = board.move(Direction.UP)
        assert changed
        assert gained == 4
        assert board.grid[0][1] == 4
        assert board.grid[1][1] == 0
    
    def test_move_down(self):
        board = Board2048([[0, 0, 0, 0],
                           [0, 2, 0, 0],
                           [0, 2, 0, 0],
                           [0, 0, 0, 0]])
        changed, gained = board.move(Direction.DOWN)
        assert changed
        assert gained == 4
        assert board.grid[3][1] == 4
        assert board.grid[2][1] == 0
    
    def test_chain_merge(self):
        # [2, 2, 4, 4] -> [4, 8, 0, 0]
        board = Board2048([[2, 2, 4, 4],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        changed, gained = board.move(Direction.LEFT)
        assert changed
        assert gained == 12  # 4 + 8
        assert board.grid[0] == [4, 8, 0, 0]
    
    def test_no_move_possible(self):
        # Full board with no merges
        board = Board2048([[2, 4, 2, 4],
                           [4, 2, 4, 2],
                           [2, 4, 2, 4],
                           [4, 2, 4, 2]])
        assert board.is_full
        valid = board.get_valid_moves()
        assert len(valid) == 0
        assert board.get_state() == "lost"
    
    def test_win_state(self):
        board = Board2048([[2048, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        assert board.get_state() == "won"
    
    def test_copy(self):
        board = Board2048([[2, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], score=10)
        copy = board.copy()
        assert copy.grid == board.grid
        assert copy.score == board.score
        # Modify original
        board.move(Direction.LEFT)
        # Copy should be unchanged
        assert copy.grid == [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert copy.score == 10
    
    def test_numpy_conversion(self):
        board = Board2048([[2, 4, 8, 16],
                           [32, 64, 128, 256],
                           [512, 1024, 2048, 0],
                           [0, 0, 0, 0]])
        arr = board.to_numpy()
        assert arr.shape == (4, 4)
        assert arr[0, 0] == 2
        assert arr[2, 2] == 2048
        
        # Round trip
        board2 = Board2048.from_numpy(arr)
        assert board2.grid == board.grid
    
    def test_simulate_move(self):
        board = Board2048([[2, 2, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        new_board, gained = simulate_move(board, Direction.LEFT)
        assert gained == 4
        assert new_board.grid[0] == [4, 0, 0, 0]
        # Original unchanged
        assert board.grid[0] == [2, 2, 0, 0]


class TestGame2048:
    """Tests for Game2048 orchestration."""
    
    def test_reset(self):
        game = Game2048(seed=42)
        game.play(RandomAgent(), max_moves=10)
        game.reset(seed=42)
        # Should be deterministic with same seed
        assert len(game.board.empty_cells) == 14
    
    def test_play_random(self):
        game = Game2048(seed=123)
        result = game.play(RandomAgent(), max_moves=100)
        assert result.moves > 0
        assert result.final_score >= 0
        assert result.state in ("won", "lost", "playing")
    
    def test_play_greedy(self):
        game = Game2048(seed=456)
        result = game.play(GreedyAgent(), max_moves=100)
        assert result.moves > 0
        assert result.final_score >= 0


class TestAgents:
    """Tests for AI agents."""
    
    def test_random_agent(self):
        agent = RandomAgent(seed=42)
        board = Board2048()
        board.add_random_tile()
        board.add_random_tile()
        move = agent(board)
        assert move in Direction
        assert move in board.get_valid_moves()
    
    def test_greedy_agent(self):
        agent = GreedyAgent()
        # Board where LEFT merges both pairs for maximum score
        board = Board2048([[2, 2, 4, 4],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        move = agent(board)
        # LEFT merges both pairs: 2+2=4, 4+4=8 -> [4, 8, 0, 0]
        # This should be strongly preferred
        assert move in board.get_valid_moves()
    
    def test_get_agent(self):
        for name in list_agents():
            agent = get_agent(name)
            assert agent is not None
            assert hasattr(agent, 'get_move')
            assert hasattr(agent, 'name')
    
    def test_expectimax_agent(self):
        agent = get_agent("expectimax-fast")
        board = Board2048([[2, 2, 4, 4],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        move = agent(board)
        assert move in board.get_valid_moves()
        stats = agent.get_stats()
        assert stats["nodes_evaluated"] > 0


class TestUtils:
    """Tests for utility functions."""
    
    def test_format_board(self):
        board = [[2, 4, 0, 8],
                 [0, 0, 16, 0],
                 [32, 0, 0, 0],
                 [0, 0, 0, 0]]
        formatted = format_board(board)
        assert "2" in formatted
        assert "4" in formatted
        assert "." in formatted
    
    def test_monotonicity(self):
        # Perfectly monotonic decreasing left-to-right
        board = [[16, 8, 4, 2],
                 [32, 16, 8, 4],
                 [64, 32, 16, 8],
                 [128, 64, 32, 16]]
        score = calculate_monotonicity(board)
        assert score > 0
    
    def test_smoothness(self):
        # Smooth board
        board = [[1024, 512, 256, 128],
                 [512, 256, 128, 64],
                 [256, 128, 64, 32],
                 [128, 64, 32, 16]]
        score = calculate_smoothness(board)
        assert score < 0  # Negative because differences exist
        
        # Less smooth
        board2 = [[1024, 2, 1024, 2],
                  [2, 1024, 2, 1024],
                  [1024, 2, 1024, 2],
                  [2, 1024, 2, 1024]]
        score2 = calculate_smoothness(board2)
        assert score2 < score  # More negative = less smooth
    
    def test_game_phase(self):
        # Early game
        board = [[2, 0, 0, 0],
                 [0, 4, 0, 0],
                 [0, 0, 2, 0],
                 [0, 0, 0, 0]]
        assert get_game_phase(board) == "early"
        
        # Mid game (3-6 empty cells)
        board = [[1024, 512, 256, 128],
                 [64, 32, 16, 8],
                 [4, 2, 0, 0],
                 [0, 0, 0, 0]]
        assert get_game_phase(board) == "mid"
        
        # Late game (<= 2 empty)
        board = [[1024, 512, 256, 128],
                 [64, 32, 16, 8],
                 [4, 2, 2, 2],
                 [2, 2, 2, 2]]
        assert get_game_phase(board) == "late"
        
        # End game
        board = [[2048, 1024, 512, 256],
                 [128, 64, 32, 16],
                 [8, 4, 2, 2],
                 [0, 0, 0, 0]]
        assert get_game_phase(board) == "end"


class TestConstants:
    """Tests for constants and enums."""
    
    def test_direction_vectors(self):
        assert Direction.UP.vector == (-1, 0)
        assert Direction.RIGHT.vector == (0, 1)
        assert Direction.DOWN.vector == (1, 0)
        assert Direction.LEFT.vector == (0, -1)
    
    def test_direction_opposite(self):
        assert Direction.UP.opposite() == Direction.DOWN
        assert Direction.LEFT.opposite() == Direction.RIGHT
    
    def test_board_size(self):
        assert GameState.__args__ == ("playing", "won", "lost")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])