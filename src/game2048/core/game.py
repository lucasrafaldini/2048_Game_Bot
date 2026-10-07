"""High-level 2048 game orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .board import Board2048
from .constants import Direction, GameState


@dataclass
class GameResult:
    """Result of a completed game."""
    final_score: int
    max_tile: int
    moves: int
    duration_seconds: float
    state: GameState
    board_history: list[np.ndarray] = field(default_factory=list)
    move_history: list[Direction] = field(default_factory=list)
    tile_spawn_history: list[tuple[int, int, int]] = field(default_factory=list)  # (r, c, value)

    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "max_tile": self.max_tile,
            "moves": self.moves,
            "duration_seconds": self.duration_seconds,
            "state": self.state,
            "board_history": [b.tolist() for b in self.board_history],
            "move_history": [m.name for m in self.move_history],
            "tile_spawn_history": self.tile_spawn_history,
        }


class Game2048:
    """
    Main game class that orchestrates gameplay.
    
    Supports both interactive play and headless simulation for benchmarks.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        record_history: bool = False,
    ):
        self.board = Board2048()
        self._rng = np.random.default_rng(seed)
        self._record_history = record_history
        self._result: Optional[GameResult] = None
        self._start_time: Optional[float] = None

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset game to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.board = Board2048()
        self._result = None
        self._start_time = time.perf_counter()
        # Add initial tiles
        self._add_initial_tiles()

    def _add_initial_tiles(self) -> None:
        """Add initial random tiles."""
        for _ in range(2):
            self._spawn_tile()

    def _spawn_tile(self) -> Optional[tuple[int, int, int]]:
        """Spawn a random tile. Returns (r, c, value) or None."""
        empty = self.board.empty_cells
        if not empty:
            return None
        idx = self._rng.choice(len(empty))
        r, c = empty[idx]
        value = self._rng.choice([2, 4], p=[0.9, 0.1])
        self.board._grid[r][c] = value
        return (r, c, value)

    def step(self, direction: Direction) -> tuple[bool, GameState]:
        """
        Execute one game step.
        
        Returns:
            (valid_move, game_state)
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
            
        changed, _ = self.board.move(direction)
        
        if self._record_history:
            self._result.move_history.append(direction)
            self._result.board_history.append(self.board.to_numpy())
        
        if changed:
            spawn = self._spawn_tile()
            if spawn and self._record_history:
                self._result.tile_spawn_history.append(spawn)
        
        state = self.board.get_state()
        return changed, state

    def play(
        self,
        agent: Callable[[Board2048], Direction],
        max_moves: int = 10000,
        verbose: bool = False,
    ) -> GameResult:
        """
        Play a full game using the given agent.
        
        Args:
            agent: Function that takes a Board2048 and returns a Direction
            max_moves: Maximum moves before giving up
            verbose: Print progress
            
        Returns:
            GameResult with statistics
        """
        self.reset()
        
        self._result = GameResult(
            final_score=0,
            max_tile=0,
            moves=0,
            duration_seconds=0.0,
            state="playing",
        )
        
        if self._record_history:
            self._result.board_history.append(self.board.to_numpy())
        
        for move_num in range(max_moves):
            direction = agent(self.board)
            changed, state = self.step(direction)
            
            if verbose:
                print(f"Move {move_num + 1}: {direction.name} -> {state}")
                print(self.board)
                print()
            
            if state != "playing":
                break
        
        duration = time.perf_counter() - (self._start_time or time.perf_counter())
        
        self._result.final_score = self.board.score
        self._result.max_tile = self.board.max_tile
        self._result.moves = self.board.moves
        self._result.duration_seconds = duration
        self._result.state = state
        
        return self._result

    def play_interactive(self) -> GameResult:
        """Play interactively with keyboard input (for testing)."""
        import sys
        
        print("Use WASD or arrow keys to move. Q to quit.")
        print(self.board)
        
        self.reset()
        self._result = GameResult(0, 0, 0, 0.0, "playing")
        self._start_time = time.perf_counter()
        
        key_map = {
            'w': Direction.UP, 'a': Direction.LEFT, 's': Direction.DOWN, 'd': Direction.RIGHT,
            'W': Direction.UP, 'A': Direction.LEFT, 'S': Direction.DOWN, 'D': Direction.RIGHT,
        }
        
        try:
            while True:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q' or key == 'Q':
                        break
                    if key in key_map:
                        changed, state = self.step(key_map[key])
                        print(f"\n{key_map[key].name}:")
                        print(self.board)
                        if state != "playing":
                            print(f"Game {state}! Score: {self.board.score}")
                            break
        except KeyboardInterrupt:
            pass
        
        duration = time.perf_counter() - self._start_time
        self._result.final_score = self.board.score
        self._result.max_tile = self.board.max_tile
        self._result.moves = self.board.moves
        self._result.duration_seconds = duration
        self._result.state = self.board.get_state()
        
        return self._result


def play_cli() -> None:
    """CLI entry point for interactive play."""
    import typer
    from rich.console import Console
    from rich.table import Table
    
    app = typer.Typer()
    console = Console()
    
    @app.command()
    def play(
        seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
        agent: str = typer.Option("human", help="Agent type: human, random, greedy"),
    ):
        """Play 2048 in terminal."""
        from ..ai.heuristic import GreedyAgent, RandomAgent
        
        game = Game2048(seed=seed)
        
        if agent == "random":
            agent_fn = RandomAgent()
        elif agent == "greedy":
            agent_fn = GreedyAgent()
        else:
            # Human play
            console.print("[bold green]2048 Game[/bold green]")
            console.print("Use W/A/S/D to move, Q to quit")
            console.print(str(game.board))
            
            key_map = {
                'w': Direction.UP, 'a': Direction.LEFT, 's': Direction.DOWN, 'd': Direction.RIGHT,
            }
            
            try:
                while True:
                    key = console.input("Move: ").strip().lower()
                    if key == 'q':
                        break
                    if key in key_map:
                        changed, state = game.step(key_map[key])
                        console.print(str(game.board))
                        console.print(f"Score: {game.board.score} | Max: {game.board.max_tile}")
                        if state != "playing":
                            console.print(f"[bold red]Game {state}![/bold red]")
                            break
            except KeyboardInterrupt:
                pass
            return
        
        # AI play
        result = game.play(agent_fn, verbose=True)
        console.print(f"\n[bold]Final Score: {result.final_score}[/bold]")
        console.print(f"Max Tile: {result.max_tile}")
        console.print(f"Moves: {result.moves}")
        console.print(f"Duration: {result.duration_seconds:.2f}s")
    
    app()


if __name__ == "__main__":
    play_cli()