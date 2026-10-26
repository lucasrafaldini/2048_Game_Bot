"""Core 2048 game engine."""

from .board import Board2048, simulate_move
from .constants import Direction, Board, GameState, BOARD_SIZE, WIN_TILE
from .game import Game2048, GameResult, play_cli
from .fast import (
    FastGame,
    FastResult,
    grid_to_bits,
    bits_to_grid,
    apply_move_bits,
    can_move_bits,
    valid_moves_bits,
    random_policy,
    greedy_policy,
    snake_policy,
    make_expectimax_policy,
    batch_simulate,
)

__all__ = [
    "Board2048",
    "simulate_move",
    "Direction",
    "Board",
    "GameState",
    "BOARD_SIZE",
    "WIN_TILE",
    "Game2048",
    "GameResult",
    "play_cli",
    "FastGame",
    "FastResult",
    "grid_to_bits",
    "bits_to_grid",
    "apply_move_bits",
    "can_move_bits",
    "valid_moves_bits",
    "random_policy",
    "greedy_policy",
    "snake_policy",
    "make_expectimax_policy",
    "batch_simulate",
]