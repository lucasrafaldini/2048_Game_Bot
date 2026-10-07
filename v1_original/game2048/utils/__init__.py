"""Utility functions for 2048."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich: bool = True,
) -> None:
    """Configure logging with Rich handler."""
    handlers = []
    
    if rich:
        handlers.append(RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        ))
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


def format_board(board: list[list[int]]) -> str:
    """Format board as pretty string."""
    lines = []
    for row in board:
        lines.append(" ".join(f"{x:4d}" if x else "    ." for x in row))
    return "\n".join(lines)


def board_to_string(board: list[list[int]], compact: bool = False) -> str:
    """Convert board to string representation."""
    if compact:
        return "".join(str(x) if x else "." for row in board for x in row)
    return format_board(board)


def parse_board_string(s: str) -> list[list[int]]:
    """Parse board from string."""
    board = [[0]*4 for _ in range(4)]
    idx = 0
    for ch in s:
        if ch.isdigit():
            val = int(ch)
            if val > 0:
                board[idx // 4][idx % 4] = 2 ** val
            idx += 1
        elif ch == '.':
            idx += 1
    return board


def calculate_tile_entropy(board: list[list[int]]) -> float:
    """Calculate entropy of tile distribution."""
    import math
    tiles = [x for row in board for x in row if x > 0]
    if not tiles:
        return 0.0
    
    counts = {}
    for t in tiles:
        counts[t] = counts.get(t, 0) + 1
    
    total = len(tiles)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy


def calculate_monotonicity(board: list[list[int]]) -> float:
    """Calculate monotonicity score (higher = more monotonic)."""
    score = 0
    
    # Rows
    for r in range(4):
        row = board[r]
        for c in range(3):
            if row[c] >= row[c+1] and row[c] > 0:
                score += row[c] - row[c+1]
    
    # Columns
    for c in range(4):
        for r in range(3):
            if board[r][c] >= board[r+1][c] and board[r][c] > 0:
                score += board[r][c] - board[r+1][c]
    
    return score


def calculate_smoothness(board: list[list[int]]) -> float:
    """Calculate smoothness (negative sum of differences)."""
    smooth = 0
    for r in range(4):
        for c in range(4):
            if board[r][c] == 0:
                continue
            if c + 1 < 4 and board[r][c+1] > 0:
                smooth -= abs(board[r][c] - board[r][c+1])
            if r + 1 < 4 and board[r+1][c] > 0:
                smooth -= abs(board[r][c] - board[r+1][c])
    return smooth


def is_valid_move(board: list[list[int]], direction: int) -> bool:
    """Check if move would change the board."""
    from .core.board import Board2048, simulate_move
    from .core.constants import Direction
    
    b = Board2048(board)
    changed, _ = simulate_move(b, Direction(direction))
    return changed


def get_game_phase(board: list[list[int]]) -> str:
    """Determine game phase: early, mid, late, end."""
    empty = sum(1 for row in board for x in row if x == 0)
    max_tile = max(max(row) for row in board)
    
    if max_tile >= 2048:
        return "end"
    elif empty <= 2:
        return "late"
    elif empty <= 6:
        return "mid"
    else:
        return "early"