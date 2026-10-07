"""Core 2048 game engine."""

from .board import Board2048, simulate_move
from .constants import Direction, Board, GameState, BOARD_SIZE, WIN_TILE
from .game import Game2048, GameResult, play_cli

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
]