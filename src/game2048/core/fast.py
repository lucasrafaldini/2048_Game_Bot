"""High-speed, self-contained 2048 engine (bitboard).

Design goals
------------
* **Self-contained**: only ``random`` + ``dataclasses`` from stdlib. No numpy,
  no scipy - safe to embed anywhere.
* **Fast**: board is a single 64-bit integer (16 nibbles, one per cell). Each
  nibble stores the *exponent* of the tile (0 = empty, 1 = tile 2, 2 = tile 4,
  ..., 15 = tile 32768). Moves use precomputed 16-bit row tables, so a move is
  just 4 table lookups + packing.
* **Reproducible**: seeded ``random.Random`` per game.
* **Verifiable**: results can be cross-checked move-by-move against the
  reference engine (``Board2048``) - see ``game2048.benchmarks.accuracy``.

Typical throughput: >100k moves/s single core with the random policy.

Example
-------
>>> from game2048.core.fast import FastGame, random_policy
>>> game = FastGame(seed=42)
>>> result = game.play(random_policy)
>>> result.final_score > 0
True
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .constants import Direction

__all__ = [
    "FastGame",
    "FastResult",
    "grid_to_bits",
    "bits_to_grid",
    "apply_move_bits",
    "can_move_bits",
    "expectimax_bits",
    "random_policy",
    "greedy_policy",
    "snake_policy",
    "expectimax_policy",
    "batch_simulate",
]

# ---------------------------------------------------------------------------
# Bitboard primitives
# ---------------------------------------------------------------------------
# Nibble i (0..15) = cell (row r=i//4, col c=i%4), value = 2**exponent.
# Row r occupies bits [16*r, 16*r+16).

MAX_EXPONENT = 15  # tile 32768; cannot be merged further in 4 bits

_ROW_MASK = 0xFFFF
_ROW_LEFT: list[int] = []
_ROW_SCORE: list[int] = []
_ROW_REVERSED: list[int] = []


def _compress_row_exponents(exps: list[int]) -> tuple[list[int], int]:
    """Compress/merge one row (as tile exponents) leftward.

    Mirrors ``Board2048._compress_row`` semantics in exponent space:
    merge the leftmost equal pair first; a merged tile cannot merge again
    within the same move.
    """
    filtered = [e for e in exps if e != 0]
    out: list[int] = []
    score = 0
    i = 0
    n = len(filtered)
    while i < n:
        e = filtered[i]
        if i + 1 < n and filtered[i + 1] == e and e < MAX_EXPONENT:
            out.append(e + 1)
            score += 1 << (e + 1)  # merged tile value
            i += 2
        else:
            out.append(e)
            i += 1
    out.extend([0] * (4 - len(out)))
    return out, score


def _build_tables() -> None:
    global _ROW_LEFT, _ROW_SCORE, _ROW_REVERSED
    if _ROW_LEFT:
        return
    left = [0] * 65536
    score = [0] * 65536
    rev = [0] * 65536
    for row in range(65536):
        exps = [(row >> (4 * i)) & 0xF for i in range(4)]
        merged, gained = _compress_row_exponents(exps)
        packed = 0
        for i, e in enumerate(merged):
            packed |= e << (4 * i)
        left[row] = packed
        score[row] = gained
        # reverse nibble order (col 0 <-> col 3)
        rev[row] = ((row & 0x000F) << 12) | ((row & 0x00F0) << 4) | (
            (row & 0x0F00) >> 4
        ) | ((row & 0xF000) >> 12)
    _ROW_LEFT, _ROW_SCORE, _ROW_REVERSED = left, score, rev


_build_tables()


def _row16(bits: int, r: int) -> int:
    return (bits >> (16 * r)) & _ROW_MASK


def _place_row(bits: int, r: int, row16: int) -> int:
    shift = 16 * r
    return (bits & ~(_ROW_MASK << shift)) | (row16 << shift)


def _transpose(bits: int) -> int:
    """Transpose the 4x4 nibble matrix."""
    t = 0
    for i in range(16):
        nib = (bits >> (4 * i)) & 0xF
        j = (i % 4) * 4 + (i // 4)
        t |= nib << (4 * j)
    return t


def grid_to_bits(grid: list[list[int]]) -> int:
    """Convert a 4x4 value grid (tile values like 2, 4, 8...) to a bitboard."""
    bits = 0
    for r in range(4):
        for c in range(4):
            v = grid[r][c]
            if v:
                exp = v.bit_length() - 1
                if v != (1 << exp):
                    raise ValueError(f"Tile {v} is not a power of two")
                bits |= exp << (4 * (r * 4 + c))
    return bits


def bits_to_grid(bits: int) -> list[list[int]]:
    """Convert a bitboard back to a 4x4 value grid."""
    grid = [[0] * 4 for _ in range(4)]
    for i in range(16):
        e = (bits >> (4 * i)) & 0xF
        if e:
            grid[i // 4][i % 4] = 1 << e
    return grid


def _move_rows_left(bits: int) -> tuple[int, int]:
    """Apply move-left to every row. Returns (new_bits, score_gained)."""
    new_bits = 0
    gained = 0
    for r in range(4):
        row = _row16(bits, r)
        new_bits |= _ROW_LEFT[row] << (16 * r)
        gained += _ROW_SCORE[row]
    return new_bits, gained


def apply_move_bits(bits: int, direction: Direction) -> tuple[int, int, bool]:
    """
    Apply a move to a bitboard.

    Returns:
        (new_bits, score_gained, moved)
    """
    if direction == Direction.LEFT:
        new_bits, gained = _move_rows_left(bits)
    elif direction == Direction.RIGHT:
        rev = 0
        for r in range(4):
            rev |= _ROW_REVERSED[_row16(bits, r)] << (16 * r)
        moved_rev, gained = _move_rows_left(rev)
        new_bits = 0
        for r in range(4):
            new_bits |= _ROW_REVERSED[_row16(moved_rev, r)] << (16 * r)
    elif direction == Direction.UP:
        t = _transpose(bits)
        moved_t, gained = _move_rows_left(t)
        new_bits = _transpose(moved_t)
    elif direction == Direction.DOWN:
        t = _transpose(bits)
        rev = 0
        for r in range(4):
            rev |= _ROW_REVERSED[_row16(t, r)] << (16 * r)
        moved_rev, gained = _move_rows_left(rev)
        unrev = 0
        for r in range(4):
            unrev |= _ROW_REVERSED[_row16(moved_rev, r)] << (16 * r)
        new_bits = _transpose(unrev)
    else:
        raise ValueError(f"Invalid direction: {direction}")
    return new_bits, gained, new_bits != bits


def valid_moves_bits(bits: int) -> list[Direction]:
    """Directions that would change the board."""
    return [d for d in Direction if apply_move_bits(bits, d)[2]]


def can_move_bits(bits: int) -> bool:
    """Whether any move is possible."""
    return any(apply_move_bits(bits, d)[2] for d in Direction)


def empty_positions(bits: int) -> list[int]:
    """Nibble indices of empty cells."""
    return [i for i in range(16) if not (bits >> (4 * i)) & 0xF]


def max_exponent(bits: int) -> int:
    """Largest tile exponent on the board (0 if empty)."""
    m = 0
    for i in range(16):
        e = (bits >> (4 * i)) & 0xF
        if e > m:
            m = e
    return m


# ---------------------------------------------------------------------------
# Fast heuristic evaluation (row-memoized)
# ---------------------------------------------------------------------------

_HEUR_CACHE: dict[int, float] = {}
_SNAKE_WEIGHTS = (
    65536, 32768, 16384, 8192,
    512, 1024, 2048, 4096,
    256, 128, 64, 32,
    2, 4, 8, 16,
)


def _row_heuristic(row: int) -> float:
    """Heuristic for a single 16-bit row (memoized)."""
    cached = _HEUR_CACHE.get(row)
    if cached is not None:
        return cached
    exps = [(row >> (4 * i)) & 0xF for i in range(4)]
    empty = sum(1 for e in exps if e == 0)
    # Monotonicity: penalize non-monotonic sequences.
    inc = dec = 0.0
    for i in range(3):
        a, b = exps[i], exps[i + 1]
        if a > b:
            dec += a - b
        else:
            inc += b - a
    mono = 1.0 - (inc + dec) / 8.0
    # Merge potential: adjacent equal non-zero exponents.
    merges = sum(1 for i in range(3) if exps[i] != 0 and exps[i] == exps[i + 1])
    value = 2.7 * empty + 6.0 * mono + 1.5 * merges
    _HEUR_CACHE[row] = value
    return value


def evaluate_bits(bits: int) -> float:
    """Fast positional evaluation of a bitboard (rows + columns)."""
    t = _transpose(bits)
    score = (
        _row_heuristic(_row16(bits, 0))
        + _row_heuristic(_row16(bits, 1))
        + _row_heuristic(_row16(bits, 2))
        + _row_heuristic(_row16(bits, 3))
        + _row_heuristic(_row16(t, 0))
        + _row_heuristic(_row16(t, 1))
        + _row_heuristic(_row16(t, 2))
        + _row_heuristic(_row16(t, 3))
    )
    # Corner preference for the max tile.
    me = max_exponent(bits)
    if me:
        for corner in (0, 3, 12, 15):
            if (bits >> (4 * corner)) & 0xF == me:
                score += 3.0
                break
    return score


def evaluate_snake(bits: int) -> float:
    """Snake-pattern weighted evaluation (corner-snake strategy)."""
    s = 0.0
    for i in range(16):
        e = (bits >> (4 * i)) & 0xF
        if e:
            s += _SNAKE_WEIGHTS[i] * (2 ** e)
    return s


# ---------------------------------------------------------------------------
# Self-contained policies (fast, picklable-friendly)
# ---------------------------------------------------------------------------

def random_policy(bits: int) -> Direction:
    """Random valid move.

    Uses the global ``random`` module: seed it (``random.seed(...)``) for
    reproducible runs.
    """
    moves = valid_moves_bits(bits)
    if not moves:
        return Direction.UP
    return random.choice(moves)


def greedy_policy(bits: int) -> Direction:
    """1-ply greedy on the bitboard heuristic."""
    best_dir = Direction.UP
    best_score = -float("inf")
    for d in Direction:
        nb, _, moved = apply_move_bits(bits, d)
        if not moved:
            continue
        s = evaluate_bits(nb)
        if s > best_score:
            best_score = s
            best_dir = d
    return best_dir


def snake_policy(bits: int) -> Direction:
    """1-ply greedy on the snake-pattern weights."""
    best_dir = Direction.UP
    best_score = -float("inf")
    for d in Direction:
        nb, _, moved = apply_move_bits(bits, d)
        if not moved:
            continue
        s = evaluate_snake(nb)
        if s > best_score:
            best_score = s
            best_dir = d
    return best_dir


def expectimax_bits(bits: int, depth: int = 3) -> float:
    """Expectimax search on the bitboard. Returns expected evaluation."""
    moves = valid_moves_bits(bits)
    if depth == 0 or not moves:
        return evaluate_bits(bits)
    best = -float("inf")
    for d in moves:
        nb, _, _ = apply_move_bits(bits, d)
        # Chance node: average over tile spawns (90% 2, 10% 4).
        empties = empty_positions(nb)
        if not empties:
            v = evaluate_bits(nb)
        else:
            total = 0.0
            for pos in empties:
                with_two = nb | (1 << (4 * pos))
                with_four = nb | (2 << (4 * pos))
                total += 0.9 * expectimax_bits(with_two, depth - 1) + \
                    0.1 * expectimax_bits(with_four, depth - 1)
            v = total / len(empties)
        if v > best:
            best = v
    return best


def make_expectimax_policy(depth: int = 2) -> Callable[[int], Direction]:
    """Build a bitboard expectimax policy with a given search depth."""
    def policy(bits: int) -> Direction:
        best_dir = Direction.UP
        best_score = -float("inf")
        for d in Direction:
            nb, _, moved = apply_move_bits(bits, d)
            if not moved:
                continue
            empties = empty_positions(nb)
            if empties:
                total = 0.0
                for pos in empties:
                    total += 0.9 * expectimax_bits(nb | (1 << (4 * pos)), depth) \
                        + 0.1 * expectimax_bits(nb | (2 << (4 * pos)), depth)
                s = total / len(empties)
            else:
                s = evaluate_bits(nb)
            if s > best_score:
                best_score = s
                best_dir = d
        return best_dir
    return policy


expectimax_policy = make_expectimax_policy(2)


# ---------------------------------------------------------------------------
# FastGame - self-contained high-speed game
# ---------------------------------------------------------------------------

@dataclass
class FastResult:
    """Result of a completed fast game."""
    final_score: int
    max_tile: int
    moves: int
    state: str
    duration_seconds: float = 0.0
    score_curve: list[int] = field(default_factory=list)   # cumulative score per move
    max_tile_curve: list[int] = field(default_factory=list)  # max tile after each move

    @property
    def max_exponent(self) -> int:
        return self.max_tile.bit_length() - 1 if self.max_tile else 0


class FastGame:
    """
    Self-contained 2048 game optimized for high-speed simulation.

    Semantics mirror the reference engine (``Game2048``/``Board2048``):
    * 90% spawn a 2, 10% spawn a 4;
    * score gained per move = sum of merged tile values;
    * state is "won" once a 2048 tile exists (matching reference behavior),
      "lost" when no moves remain.

    Args:
        seed: RNG seed for reproducibility.
        stop_on_win: Stop the game when 2048 is reached (default True,
            matching the reference engine).
    """

    __slots__ = ("_rng", "_bits", "_score", "_moves", "_stop_on_win",
                 "_score_curve", "_max_tile_curve", "_recording", "_spawn_log")

    def __init__(self, seed: Optional[int] = None, stop_on_win: bool = True):
        self._rng = random.Random(seed)
        self._stop_on_win = stop_on_win
        self._bits = 0
        self._score = 0
        self._moves = 0
        self._recording = False
        self._score_curve: list[int] = []
        self._max_tile_curve: list[int] = []
        self._spawn_log: list[Optional[tuple[int, int, int]]] = []
        self._spawn_initial()

    # -- state ---------------------------------------------------------------
    @property
    def bits(self) -> int:
        return self._bits

    @property
    def grid(self) -> list[list[int]]:
        return bits_to_grid(self._bits)

    @property
    def score(self) -> int:
        return self._score

    @property
    def moves(self) -> int:
        return self._moves

    @property
    def max_tile(self) -> int:
        e = max_exponent(self._bits)
        return (1 << e) if e else 0

    @property
    def rng(self) -> random.Random:
        return self._rng

    @property
    def spawns(self) -> list[Optional[tuple[int, int, int]]]:
        """Spawn log: 2 initial tiles + one entry per attempted spawn after
        each moved turn. Entries may be None (board full)."""
        return list(self._spawn_log)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset to a fresh game."""
        if seed is not None:
            self._rng = random.Random(seed)
        self._bits = 0
        self._score = 0
        self._moves = 0
        self._score_curve = []
        self._max_tile_curve = []
        self._spawn_log = []
        self._spawn_initial()

    def _spawn_initial(self) -> None:
        for _ in range(2):
            self._spawn_tile()

    def _spawn_tile(self) -> Optional[tuple[int, int, int]]:
        """Spawn a random tile. Returns (r, c, value) or None.

        Every spawn attempt is appended to ``spawns`` (including None), so
        full games can be replayed deterministically on other engines.
        """
        empties = empty_positions(self._bits)
        if not empties:
            self._spawn_log.append(None)
            return None
        nib = empties[self._rng.randrange(len(empties))]
        value = self._rng.choices((2, 4), weights=(9, 1))[0]
        exp = value.bit_length() - 1
        self._bits |= exp << (4 * nib)
        spawn = (nib // 4, nib % 4, value)
        self._spawn_log.append(spawn)
        return spawn

    def state(self) -> str:
        """Game state: 'won', 'playing' or 'lost' (mirrors Board2048.get_state)."""
        e = max_exponent(self._bits)
        if e >= 11:  # 2048
            return "won"
        if not can_move_bits(self._bits):
            return "lost"
        return "playing"

    # -- gameplay ------------------------------------------------------------
    def step(self, direction: Direction) -> tuple[bool, int, str]:
        """
        Execute one game step.

        Returns:
            (moved, score_gained, state)
        """
        new_bits, gained, moved = apply_move_bits(self._bits, direction)
        if moved:
            self._bits = new_bits
            self._score += gained
            self._moves += 1
            self._spawn_tile()
            if self._recording:
                self._score_curve.append(self._score)
                self._max_tile_curve.append(self.max_tile)
        return moved, gained, self.state()

    def play(
        self,
        policy: Callable[[int], Direction],
        max_moves: int = 100_000,
        record_curves: bool = False,
    ) -> FastResult:
        """
        Play a full game with the given policy.

        Args:
            policy: ``Callable[[int], Direction]`` - bitboard in, move out.
            max_moves: Safety cap.
            record_curves: Record per-move score/max-tile curves.

        Returns:
            FastResult with final statistics.
        """
        start = time.perf_counter()
        self._recording = record_curves
        state = self.state()
        # If already won/lost at start (e.g. reset() then play), still play
        # unless stop_on_win and won.
        if not (self._stop_on_win and state == "won"):
            for _ in range(max_moves):
                direction = policy(self._bits)
                moved, _, state = self.step(direction)
                if state != "playing":
                    if state == "won" and not self._stop_on_win:
                        continue
                    break
        duration = time.perf_counter() - start
        return FastResult(
            final_score=self._score,
            max_tile=self.max_tile,
            moves=self._moves,
            state=state,
            duration_seconds=duration,
            score_curve=self._score_curve if record_curves else [],
            max_tile_curve=self._max_tile_curve if record_curves else [],
        )

    def play_many(
        self,
        policy: Callable[[int], Direction],
        n_games: int,
        max_moves: int = 100_000,
    ) -> list[FastResult]:
        """Play ``n_games`` sequential games, reseeding each time."""
        results = []
        base_seed = self._rng.randrange(1 << 30)
        for i in range(n_games):
            self.reset(seed=base_seed + i)
            results.append(self.play(policy, max_moves=max_moves))
        return results


# ---------------------------------------------------------------------------
# Batch simulation (parallel)
# ---------------------------------------------------------------------------

def _batch_worker(args: tuple) -> FastResult:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    policy, seed, max_moves, record_curves = args
    # Seed the global RNG so stochastic policies (e.g. random_policy) are
    # reproducible per game.
    random.seed(seed)
    game = FastGame(seed=seed)
    return game.play(policy, max_moves=max_moves, record_curves=record_curves)


def _resolve_policy(factory_or_policy: Callable) -> Callable[[int], Direction]:
    """Accept either a zero-arg factory (returns a policy) or a policy itself."""
    import inspect

    if callable(factory_or_policy):
        try:
            params = [
                p for p in inspect.signature(factory_or_policy).parameters.values()
                if p.default is inspect.Parameter.empty
            ]
            if not params:
                return factory_or_policy()
        except (TypeError, ValueError):
            pass
    return factory_or_policy


def batch_simulate(
    policy_factory: Callable[[], Callable[[int], Direction]],
    n_games: int = 1000,
    seed: int = 0,
    workers: Optional[int] = None,
    max_moves: int = 100_000,
    record_curves: bool = False,
    parallel: bool = True,
) -> list[FastResult]:
    """
    Simulate many games at high speed, optionally in parallel.

    Args:
        policy_factory: Either a zero-arg callable returning a policy
            (``Callable[[int], Direction]``) or a policy function directly
            (``policy(bits) -> Direction``). For parallel execution it must
            be picklable (module-level function or ``functools.partial`` -
            not a lambda).
        n_games: Number of games.
        seed: Base seed; game i uses seed + i.
        workers: Process count (defaults to CPU count).
        max_moves: Safety cap per game.
        record_curves: Record per-move curves (more memory).
        parallel: Use ProcessPoolExecutor when True.

    Returns:
        List of FastResult, one per game.
    """
    jobs = [
        (_resolve_policy(policy_factory), seed + i, max_moves, record_curves)
        for i in range(n_games)
    ]
    if not parallel or n_games < 8:
        return [_batch_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_batch_worker, jobs))
