"""Base classes for AI agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from ..core.board import Board2048
from ..core.constants import Direction


class Agent(Protocol):
    """Protocol for game agents."""
    
    def __call__(self, board: Board2048) -> Direction:
        ...
    
    @property
    def name(self) -> str:
        ...


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str | None = None):
        self._name = name or self.__class__.__name__
        self._nodes_evaluated = 0
        self._max_depth_reached = 0
    
    @property
    def name(self) -> str:
        return self._name
    
    @abstractmethod
    def get_move(self, board: Board2048) -> Direction:
        """Get the best move for the given board."""
        pass
    
    def __call__(self, board: Board2048) -> Direction:
        self._nodes_evaluated = 0
        self._max_depth_reached = 0
        return self.get_move(board)
    
    def get_stats(self) -> dict:
        """Return search statistics."""
        return {
            "nodes_evaluated": self._nodes_evaluated,
            "max_depth_reached": self._max_depth_reached,
        }
    
    def reset_stats(self) -> None:
        self._nodes_evaluated = 0
        self._max_depth_reached = 0


class HeuristicAgent(BaseAgent):
    """Base class for heuristic-based agents."""
    
    def __init__(self, name: str | None = None):
        super().__init__(name)
        self.weights = {
            "empty": 2.7,
            "monotonicity": 1.0,
            "smoothness": 0.1,
            "max_tile": 1.0,
            "merges": 1.5,
            "edge_max": 0.5,
            "corner_max": 1.0,
        }
    
    def evaluate(self, board: Board2048) -> float:
        """Evaluate board position. Higher is better."""
        return (
            self.weights["empty"] * self._empty_score(board) +
            self.weights["monotonicity"] * self._monotonicity_score(board) +
            self.weights["smoothness"] * self._smoothness_score(board) +
            self.weights["max_tile"] * self._max_tile_score(board) +
            self.weights["merges"] * self._merge_potential(board) +
            self.weights["edge_max"] * self._edge_max_score(board) +
            self.weights["corner_max"] * self._corner_max_score(board)
        )
    
    def _empty_score(self, board: Board2048) -> float:
        """Score based on number of empty cells."""
        return len(board.empty_cells)
    
    def _monotonicity_score(self, board: Board2048) -> float:
        """Score based on monotonicity (tiles decreasing in a direction)."""
        grid = board.grid
        scores = [0, 0, 0, 0]  # up, down, left, right
        
        # Check rows (left/right)
        for r in range(4):
            row = grid[r]
            for c in range(3):
                if row[c] >= row[c + 1] and row[c] != 0:
                    scores[2] += row[c] - row[c + 1]  # left
                if row[c] <= row[c + 1] and row[c + 1] != 0:
                    scores[3] += row[c + 1] - row[c]  # right
        
        # Check columns (up/down)
        for c in range(4):
            for r in range(3):
                if grid[r][c] >= grid[r + 1][c] and grid[r][c] != 0:
                    scores[0] += grid[r][c] - grid[r + 1][c]  # up
                if grid[r][c] <= grid[r + 1][c] and grid[r + 1][c] != 0:
                    scores[1] += grid[r + 1][c] - grid[r][c]  # down
        
        return max(scores)
    
    def _smoothness_score(self, board: Board2048) -> float:
        """Score based on smoothness (similar adjacent tiles)."""
        grid = board.grid
        smoothness = 0
        for r in range(4):
            for c in range(4):
                if grid[r][c] == 0:
                    continue
                val = grid[r][c]
                # Check right neighbor
                if c + 1 < 4 and grid[r][c + 1] != 0:
                    smoothness -= abs(val - grid[r][c + 1])
                # Check down neighbor
                if r + 1 < 4 and grid[r + 1][c] != 0:
                    smoothness -= abs(val - grid[r + 1][c])
        return smoothness
    
    def _max_tile_score(self, board: Board2048) -> float:
        """Score based on max tile (log scale)."""
        import math
        return math.log2(board.max_tile) if board.max_tile > 0 else 0
    
    def _merge_potential(self, board: Board2048) -> float:
        """Score based on potential merges."""
        grid = board.grid
        merges = 0
        for r in range(4):
            for c in range(4):
                if grid[r][c] == 0:
                    continue
                if c + 1 < 4 and grid[r][c] == grid[r][c + 1]:
                    merges += 1
                if r + 1 < 4 and grid[r][c] == grid[r + 1][c]:
                    merges += 1
        return merges
    
    def _edge_max_score(self, board: Board2048) -> float:
        """Score for max tile on edge."""
        grid = board.grid
        max_val = board.max_tile
        edge_score = 0
        # Check edges
        for c in range(4):
            if grid[0][c] == max_val:
                edge_score += 1
            if grid[3][c] == max_val:
                edge_score += 1
        for r in range(4):
            if grid[r][0] == max_val:
                edge_score += 1
            if grid[r][3] == max_val:
                edge_score += 1
        return edge_score
    
    def _corner_max_score(self, board: Board2048) -> float:
        """Score for max tile in corner (ideal for snake strategy)."""
        grid = board.grid
        max_val = board.max_tile
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for r, c in corners:
            if grid[r][c] == max_val:
                return 1.0
        return 0.0
    
    def get_move(self, board: Board2048) -> Direction:
        """Default greedy move selection based on heuristic evaluation."""
        from ..core.board import simulate_move
        from ..core.constants import Direction
        
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return Direction.UP
        
        best_move = valid_moves[0]
        best_score = float('-inf')
        
        for move in valid_moves:
            new_board, _ = simulate_move(board, move)
            score = self.evaluate(new_board)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move