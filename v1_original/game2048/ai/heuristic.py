"""Heuristic-based agents for 2048."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .base import Agent, BaseAgent, HeuristicAgent
from ..core.board import Board2048, simulate_move
from ..core.constants import Direction


class RandomAgent(BaseAgent):
    """Agent that picks random valid moves."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Random")
        self._rng = random.Random(seed)
    
    def get_move(self, board: Board2048) -> Direction:
        valid = board.get_valid_moves()
        return self._rng.choice(valid) if valid else Direction.UP


class GreedyAgent(HeuristicAgent):
    """Agent that picks the move with highest immediate heuristic score."""
    
    def __init__(self, name: str = "Greedy"):
        super().__init__(name)
    
    def get_move(self, board: Board2048) -> Direction:
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


class NStepLookaheadAgent(HeuristicAgent):
    """Agent that looks N moves ahead using expectimax."""
    
    def __init__(self, depth: int = 3, name: str | None = None):
        super().__init__(name or f"Lookahead-{depth}")
        self.depth = depth
    
    def get_move(self, board: Board2048) -> Direction:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return Direction.UP
        
        best_move = valid_moves[0]
        best_value = float('-inf')
        
        for move in valid_moves:
            value = self._expectimax(board, move, self.depth, True)
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def _expectimax(
        self,
        board: Board2048,
        move: Direction,
        depth: int,
        is_player: bool,
    ) -> float:
        self._nodes_evaluated += 1
        self._max_depth_reached = max(self._max_depth_reached, self.depth - depth)
        
        new_board, _ = simulate_move(board, move)
        
        if depth == 0 or new_board.get_state() != "playing":
            return self.evaluate(new_board)
        
        if is_player:
            # Player's turn: maximize
            best = float('-inf')
            for m in new_board.get_valid_moves():
                val = self._expectimax(new_board, m, depth - 1, False)
                best = max(best, val)
            return best
        else:
            # Chance node: average over possible tile spawns
            empty = new_board.empty_cells
            if not empty:
                return self.evaluate(new_board)
            
            total = 0.0
            for r, c in empty:
                for value, prob in [(2, 0.9), (4, 0.1)]:
                    test_board = new_board.copy()
                    test_board._grid[r][c] = value
                    # For chance nodes, evaluate all possible next moves
                    moves = test_board.get_valid_moves()
                    if not moves:
                        total += prob * self.evaluate(test_board)
                    else:
                        move_vals = [self._expectimax(test_board, m, depth - 1, True) for m in moves]
                        total += prob * max(move_vals)
            return total / len(empty)


class WeightedGreedyAgent(HeuristicAgent):
    """Greedy agent with configurable weights."""
    
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        name: str = "WeightedGreedy",
    ):
        super().__init__(name)
        if weights:
            self.weights.update(weights)
    
    def get_move(self, board: Board2048) -> Direction:
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


def create_snake_agent() -> WeightedGreedyAgent:
    """Create agent optimized for snake strategy (max tile in corner)."""
    return WeightedGreedyAgent(
        weights={
            "empty": 2.7,
            "monotonicity": 1.5,
            "smoothness": 0.2,
            "max_tile": 1.0,
            "merges": 2.0,
            "edge_max": 1.0,
            "corner_max": 3.0,
        },
        name="SnakeGreedy",
    )


def create_monotonic_agent() -> WeightedGreedyAgent:
    """Create agent optimized for monotonicity."""
    return WeightedGreedyAgent(
        weights={
            "empty": 1.0,
            "monotonicity": 3.0,
            "smoothness": 0.5,
            "max_tile": 0.5,
            "merges": 1.0,
            "edge_max": 0.5,
            "corner_max": 0.5,
        },
        name="MonotonicGreedy",
    )