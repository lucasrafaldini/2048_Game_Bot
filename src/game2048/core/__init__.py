"""Core 2048 game engine."""

from .board import Board2048, simulate_move
from .constants import BOARD_SIZE, WIN_TILE, Board, Direction, GameState
from .fast import (
    FastGame,
    FastResult,
    apply_move_bits,
    batch_simulate,
    bits_to_grid,
    can_move_bits,
    greedy_policy,
    grid_to_bits,
    make_expectimax_policy,
    random_policy,
    snake_policy,
    valid_moves_bits,
)
from .game import Game2048, GameResult, play_cli

__all__ = [
    "BOARD_SIZE",
    "WIN_TILE",
    "Board",
    "Board2048",
    "Direction",
    "FastGame",
    "FastResult",
    "Game2048",
    "GameResult",
    "GameState",
    "apply_move_bits",
    "batch_simulate",
    "bits_to_grid",
    "can_move_bits",
    "greedy_policy",
    "grid_to_bits",
    "make_expectimax_policy",
    "play_cli",
    "random_policy",
    "simulate_move",
    "snake_policy",
    "valid_moves_bits",
]
