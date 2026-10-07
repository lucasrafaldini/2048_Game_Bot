"""Constants and types for 2048 game."""

from enum import IntEnum
from typing import Literal


class Direction(IntEnum):
    """Game move directions."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def name_str(self) -> str:
        return self.name.lower()

    @property
    def vector(self) -> tuple[int, int]:
        """Return (dr, dc) vector for this direction."""
        return {
            Direction.UP: (-1, 0),
            Direction.RIGHT: (0, 1),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
        }[self]

    def opposite(self) -> "Direction":
        return Direction((self.value + 2) % 4)


# Type aliases
Board = list[list[int]]  # 4x4 grid
Move = Direction
GameState = Literal["playing", "won", "lost"]

# Game configuration
BOARD_SIZE = 4
WIN_TILE = 2048
INITIAL_TILES = 2
NEW_TILE_VALUES = (2, 4)
NEW_TILE_PROBABILITIES = (0.9, 0.1)  # 90% 2, 10% 4

# Heuristic weights for scoring
HEURISTIC_WEIGHTS = {
    "monotonicity": 1.0,
    "smoothness": 0.1,
    "empty_tiles": 2.7,
    "max_tile": 1.0,
    "merge_potential": 1.5,
}