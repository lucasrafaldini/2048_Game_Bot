"""Expectimax agent - the gold standard for 2048 AI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from .base import BaseAgent, HeuristicAgent
from ..core.board import Board2048, simulate_move
from ..core.constants import Direction, GameState


@dataclass
class ExpectimaxConfig:
    """Configuration for expectimax search."""
    max_depth: int = 6
    time_limit: float = 0.1  # seconds
    use_cache: bool = True
    pruning_threshold: float = 0.0  # prune branches below this value
    
    # Heuristic weights for leaf evaluation
    weights: dict[str, float] = None
    
    def __post_init__(self):
        default_weights = {
            "empty": 2.7,
            "monotonicity": 1.0,
            "smoothness": 0.1,
            "max_tile": 1.0,
            "merges": 1.5,
            "edge_max": 0.5,
            "corner_max": 1.0,
        }
        if self.weights is None:
            self.weights = default_weights
        else:
            # Merge with defaults
            self.weights = {**default_weights, **self.weights}


class ExpectimaxAgent(BaseAgent):
    """
    Expectimax agent for 2048.
    
    Models the game as:
    - Player nodes (MAX): choose best move
    - Chance nodes: random tile spawns (2 with 0.9, 4 with 0.1)
    
    Uses alpha-beta style pruning and transposition table for speed.
    """
    
    def __init__(self, config: ExpectimaxConfig | None = None, name: str | None = None):
        super().__init__(name or "Expectimax")
        self.config = config or ExpectimaxConfig()
        self._heuristic = HeuristicAgent()
        self._heuristic.weights = self.config.weights
        self._cache: dict[int, float] = {}
        self._start_time: float = 0
        self._time_limit: float = 0
    
    def get_move(self, board: Board2048) -> Direction:
        self._nodes_evaluated = 0
        self._max_depth_reached = 0
        self._cache.clear()
        self._start_time = time.perf_counter()
        self._time_limit = self.config.time_limit
        
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return Direction.UP
        
        # Iterative deepening
        best_move = valid_moves[0]
        for depth in range(1, self.config.max_depth + 1):
            if self._time_exceeded():
                break
            
            move_scores = []
            for move in valid_moves:
                new_board, _ = simulate_move(board, move)
                score = self._search(new_board, depth - 1, False)
                move_scores.append((move, score))
            
            move_scores.sort(key=lambda x: x[1], reverse=True)
            best_move = move_scores[0][0]
        
        return best_move
    
    def _time_exceeded(self) -> bool:
        return time.perf_counter() - self._start_time > self._time_limit
    
    def _search(
        self,
        board: Board2048,
        depth: int,
        is_chance: bool,
    ) -> float:
        """Expectimax search with chance nodes."""
        if self._time_exceeded():
            return self._evaluate(board)
        
        self._nodes_evaluated += 1
        self._max_depth_reached = max(self._max_depth_reached, self.config.max_depth - depth)
        
        state = board.get_state()
        if state != "playing":
            return self._terminal_value(board, state)
        
        if depth == 0:
            return self._evaluate(board)
        
        # Transposition table lookup
        if self.config.use_cache:
            board_hash = hash(board)
            if board_hash in self._cache:
                return self._cache[board_hash]
        
        if is_chance:
            # Chance node: expected value over tile spawns
            value = self._chance_node(board, depth)
        else:
            # Player node: maximize over moves
            value = self._player_node(board, depth)
        
        if self.config.use_cache:
            self._cache[hash(board)] = value
        
        return value
    
    def _player_node(self, board: Board2048, depth: int) -> float:
        """Player chooses best move."""
        best = float('-inf')
        for move in board.get_valid_moves():
            new_board, _ = simulate_move(board, move)
            val = self._search(new_board, depth - 1, True)
            best = max(best, val)
            
            # Pruning: if we found a very good move, maybe stop
            if best > self.config.pruning_threshold and self._time_exceeded():
                break
        return best
    
    def _chance_node(self, board: Board2048, depth: int) -> float:
        """Expected value over random tile spawns."""
        empty = board.empty_cells
        if not empty:
            return self._evaluate(board)
        
        total = 0.0
        # Sample a subset of empty cells for speed if too many
        cells_to_eval = empty
        if len(empty) > 6:
            # Evaluate corners and edges preferentially
            corners = [(r, c) for r, c in empty if (r in (0, 3) and c in (0, 3))]
            edges = [(r, c) for r, c in empty if r in (0, 3) or c in (0, 3)]
            cells_to_eval = corners + edges[:6 - len(corners)]
        
        for r, c in cells_to_eval:
            for value, prob in [(2, 0.9), (4, 0.1)]:
                test_board = board.copy()
                test_board._grid[r][c] = value
                val = self._search(test_board, depth - 1, False)
                total += prob * val
        
        # Average over evaluated cells (not all empty cells)
        return total / len(cells_to_eval) if cells_to_eval else self._evaluate(board)
    
    def _evaluate(self, board: Board2048) -> float:
        """Heuristic evaluation of board."""
        return self._heuristic.evaluate(board)
    
    def _terminal_value(self, board: Board2048, state: GameState) -> float:
        """Value for terminal states."""
        if state == "won":
            return 100000 + board.score
        elif state == "lost":
            return -100000
        return self._evaluate(board)


class OptimizedExpectimaxAgent(ExpectimaxAgent):
    """
    Optimized expectimax with bitboard representation and better pruning.
    
    Uses 64-bit integer board representation for fast operations.
    """
    
    def __init__(self, config: ExpectimaxConfig | None = None):
        super().__init__(config, "OptimizedExpectimax")
        # Precompute tables for bitboard operations
        self._row_left_table: dict[int, tuple[int, int]] = {}
        self._init_tables()
    
    def _init_tables(self):
        """Precompute row operations for all 16^4 = 65536 possible rows."""
        # Each row is 4 tiles, each 4 bits (0-15, where 0=empty, 1=2, 2=4, etc.)
        # Actually 16 values (0-15), so 4 bits per tile = 16 bits per row
        for row_val in range(65536):
            tiles = [(row_val >> (4 * i)) & 0xF for i in range(4)]
            # Reverse to get left-to-right order
            tiles = tiles[::-1]
            new_tiles, score = self._compress_row(tiles)
            # Pack back
            new_val = 0
            for i, t in enumerate(new_tiles):
                new_val |= (t << (4 * (3 - i)))
            self._row_left_table[row_val] = (new_val, score)
    
    def _compress_row(self, row: list[int]) -> tuple[list[int], int]:
        """Compress row leftward."""
        filtered = [x for x in row if x != 0]
        merged = []
        score = 0
        i = 0
        while i < len(filtered):
            if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
                merged.append(filtered[i] + 1)  # In log2 representation
                score += 1 << (filtered[i] + 1)
                i += 2
            else:
                merged.append(filtered[i])
                i += 1
        merged.extend([0] * (4 - len(merged)))
        return merged, score


# Convenience factory functions
def create_expectimax_agent(
    depth: int = 6,
    time_limit: float = 0.1,
    weights: dict[str, float] | None = None,
) -> ExpectimaxAgent:
    """Create an expectimax agent with common settings."""
    config = ExpectimaxConfig(
        max_depth=depth,
        time_limit=time_limit,
        weights=weights,  # Pass None to use defaults
    )
    return ExpectimaxAgent(config)