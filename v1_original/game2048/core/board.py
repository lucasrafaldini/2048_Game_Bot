"""Core 2048 board logic - pure Python, no dependencies."""

from __future__ import annotations

import copy
import random
from typing import Literal

import numpy as np

from .constants import (
    BOARD_SIZE,
    Direction,
    NEW_TILE_PROBABILITIES,
    NEW_TILE_VALUES,
    Board,
    GameState,
)


class Board2048:
    """
    2048 game board with pure Python logic.
    
    Uses numpy internally for performance but exposes pure Python API.
    """

    __slots__ = ("_grid", "_score", "_moves", "_history")

    def __init__(self, grid: Board | None = None, score: int = 0):
        if grid is None:
            self._grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        else:
            self._grid = [row[:] for row in grid]
        self._score = score
        self._moves = 0
        self._history: list[tuple[Board, int, Direction]] = []

    @property
    def grid(self) -> Board:
        return [row[:] for row in self._grid]

    @property
    def score(self) -> int:
        return self._score

    @property
    def moves(self) -> int:
        return self._moves

    @property
    def max_tile(self) -> int:
        return max(max(row) for row in self._grid)

    @property
    def empty_cells(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self._grid[r][c] == 0]

    @property
    def is_full(self) -> bool:
        return all(self._grid[r][c] != 0 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))

    def copy(self) -> Board2048:
        """Create a deep copy of the board."""
        new_board = Board2048(self._grid, self._score)
        new_board._moves = self._moves
        new_board._history = self._history.copy()
        return new_board

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for ML/analysis."""
        return np.array(self._grid, dtype=np.int32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, score: int = 0) -> Board2048:
        """Create board from numpy array."""
        return cls(arr.tolist(), score)

    def _compress_row(self, row: list[int]) -> tuple[list[int], int]:
        """Compress a row leftward, returning new row and score gained."""
        # Filter non-zero
        filtered = [x for x in row if x != 0]
        # Merge
        merged = []
        score_gained = 0
        i = 0
        while i < len(filtered):
            if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
                merged.append(filtered[i] * 2)
                score_gained += filtered[i] * 2
                i += 2
            else:
                merged.append(filtered[i])
                i += 1
        # Pad with zeros
        merged.extend([0] * (BOARD_SIZE - len(merged)))
        return merged, score_gained

    def _move_left(self) -> tuple[bool, int]:
        """Execute left move. Returns (changed, score_gained)."""
        changed = False
        total_score = 0
        for r in range(BOARD_SIZE):
            new_row, gained = self._compress_row(self._grid[r])
            if new_row != self._grid[r]:
                changed = True
                self._grid[r] = new_row
            total_score += gained
        return changed, total_score

    def move(self, direction: Direction) -> tuple[bool, int]:
        """
        Execute a move in the given direction.
        
        Returns:
            (changed, score_gained): Whether board changed and points scored.
        """
        # Save state for undo
        self._history.append((self.grid, self._score, direction))
        
        # Rotate grid to make move always "left"
        rotated = self._rotate_to_left(direction)
        self._grid = rotated
        
        changed, gained = self._move_left()
        
        # Rotate back
        self._grid = self._rotate_from_left(direction)
        
        if changed:
            self._score += gained
            self._moves += 1
            
        return changed, gained

    def _rotate_to_left(self, direction: Direction) -> Board:
        """Rotate grid so move becomes leftward."""
        if direction == Direction.LEFT:
            return [row[:] for row in self._grid]
        elif direction == Direction.UP:
            # Transpose: columns become rows (moving UP = moving LEFT on transposed)
            return [[self._grid[c][r] for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)]
        elif direction == Direction.RIGHT:
            # Reverse each row: moving RIGHT = moving LEFT on reversed rows
            return [row[::-1] for row in self._grid]
        else:  # DOWN
            # Reverse rows then transpose: moving DOWN = moving LEFT on (reversed rows transposed)
            reversed_grid = [row[:] for row in self._grid[::-1]]
            return [[reversed_grid[c][r] for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)]

    def _rotate_from_left(self, direction: Direction) -> Board:
        """Rotate grid back from leftward move."""
        if direction == Direction.LEFT:
            return self._grid
        elif direction == Direction.UP:
            # Transpose back
            return [[self._grid[c][r] for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)]
        elif direction == Direction.RIGHT:
            # Reverse each row back
            return [row[::-1] for row in self._grid]
        else:  # DOWN
            # Transpose back then reverse rows
            transposed = [[self._grid[c][r] for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)]
            return transposed[::-1]

    def add_random_tile(self) -> tuple[int, int] | None:
        """Add a random tile (2 or 4) to an empty cell. Returns position or None."""
        empty = self.empty_cells
        if not empty:
            return None
        r, c = random.choices(empty, k=1)[0]
        value = random.choices(NEW_TILE_VALUES, weights=NEW_TILE_PROBABILITIES, k=1)[0]
        self._grid[r][c] = value
        return (r, c)

    def get_state(self) -> GameState:
        """Get current game state."""
        if self.max_tile >= 2048:
            return "won"
        if not self.is_full:
            return "playing"
        # Check if any moves possible
        for d in Direction:
            if self.can_move(d):
                return "playing"
        return "lost"

    def can_move(self, direction: Direction) -> bool:
        """Check if a move would change the board."""
        test_board = self.copy()
        changed, _ = test_board.move(direction)
        return changed

    def get_valid_moves(self) -> list[Direction]:
        """Get all valid moves."""
        return [d for d in Direction if self.can_move(d)]

    def undo(self) -> bool:
        """Undo last move. Returns True if successful."""
        if not self._history:
            return False
        self._grid, self._score, _ = self._history.pop()
        self._moves -= 1
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board2048):
            return NotImplemented
        return self._grid == other._grid

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self._grid))

    def __repr__(self) -> str:
        return f"Board2048(max={self.max_tile}, score={self._score}, moves={self._moves})"

    def __str__(self) -> str:
        lines = []
        for row in self._grid:
            lines.append(" ".join(f"{x:4d}" if x else "    ." for x in row))
        return "\n".join(lines)


def simulate_move(board: Board2048, direction: Direction) -> tuple[Board2048, int]:
    """
    Simulate a move without modifying the original board.
    
    Returns:
        (new_board, score_gained)
    """
    new_board = board.copy()
    changed, gained = new_board.move(direction)
    return new_board, gained if changed else 0