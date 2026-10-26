"""Accuracy harness and statistics for the fast bitboard engine.

Two complementary verification strategies:

1. **Move-level fuzzing** (``fuzz_moves_equivalence``): generate thousands of
   random boards, apply every direction on both the fast bitboard engine and
   the reference engine (``Board2048``), and assert identical resulting grids
   and score gains.

2. **Game-level replay** (``replay_games_equivalence``): play full games with
   the fast engine recording every tile spawn, then replay the same move +
   spawn sequence on the reference engine and assert identical state after
   every single turn (score, move count, max tile, grid).

Plus statistical summaries for batches of games (``summarize_results``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.board import Board2048
from ..core.constants import Direction
from ..core.fast import (
    FastGame,
    FastResult,
    apply_move_bits,
    bits_to_grid,
    grid_to_bits,
)

__all__ = [
    "AccuracyReport",
    "FastSummary",
    "fuzz_moves_equivalence",
    "replay_games_equivalence",
    "run_accuracy_suite",
    "summarize_results",
    "tile_reach_probabilities",
    "turn_statistics",
]


# ---------------------------------------------------------------------------
# Accuracy verification
# ---------------------------------------------------------------------------

@dataclass
class AccuracyReport:
    """Result of an accuracy verification run."""
    boards_tested: int = 0
    directions_tested: int = 0
    games_tested: int = 0
    turns_compared: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict:
        return {
            "boards_tested": self.boards_tested,
            "directions_tested": self.directions_tested,
            "games_tested": self.games_tested,
            "turns_compared": self.turns_compared,
            "mismatches": self.mismatches[:20],
            "total_mismatches": len(self.mismatches),
            "passed": self.passed,
        }

    def summary(self) -> str:
        if self.passed:
            return (
                f"PASS: {self.directions_tested} move comparisons across "
                f"{self.boards_tested} boards + {self.turns_compared} replayed "
                f"turns across {self.games_tested} games - 0 mismatches"
            )
        return f"FAIL: {len(self.mismatches)} mismatches"


def _random_board(rng: random.Random, max_exp: int = 9) -> list[list[int]]:
    """Generate a plausible random board (biased toward empty cells)."""
    grid = [[0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            roll = rng.random()
            if roll < 0.35:
                continue  # empty
            exp = rng.choices(
                range(1, max_exp + 1),
                weights=[16 // (2 ** (e - 1)) for e in range(1, max_exp + 1)],
            )[0]
            grid[r][c] = 1 << exp
    return grid


def fuzz_moves_equivalence(
    n_boards: int = 2000,
    seed: int = 42,
    directions: list[Direction] | None = None,
) -> AccuracyReport:
    """
    Fuzz-test: compare fast engine move results against the reference engine.

    For each random board and each direction, apply the move on both engines
    and compare (resulting grid, score gained). Mirrors exactly the rules
    implemented in ``Board2048``.
    """
    report = AccuracyReport(boards_tested=n_boards)
    rng = random.Random(seed)
    dirs = directions or list(Direction)
    report.directions_tested = n_boards * len(dirs)

    for _ in range(n_boards):
        grid = _random_board(rng)
        for d in dirs:
            # Reference engine
            ref = Board2048(grid)
            ref_changed, ref_gained = ref.move(d)
            ref_grid = ref.grid
            # Fast engine
            bits = grid_to_bits(grid)
            fast_bits, fast_gained, fast_changed = apply_move_bits(bits, d)
            fast_grid = bits_to_grid(fast_bits)
            # Compare
            if ref_grid != fast_grid or (ref_gained if ref_changed else 0) != (fast_gained if fast_changed else 0) or ref_changed != fast_changed:
                report.mismatches.append(
                    f"board={grid} dir={d.name} ref_grid={ref_grid} "
                    f"fast_grid={fast_grid} ref_gained={ref_gained} fast_gained={fast_gained}"
                )
    return report


def replay_games_equivalence(
    policy: Callable[[int], Direction],
    n_games: int = 20,
    seed: int = 1234,
    max_moves: int = 10_000,
) -> AccuracyReport:
    """
    Replay-test: full-game equivalence between fast and reference engines.

    The fast game is driven manually while recording, turn by turn: the move
    chosen, the board after the turn (move + spawn applied), the cumulative
    score and the spawned tile. The reference engine (``Board2048``) then
    replays the exact same sequence - identical forced spawns included - and
    the two engines must agree on grid and score after *every* turn.
    """
    report = AccuracyReport(games_tested=n_games)

    for i in range(n_games):
        # --- Fast engine game (recorded) ---
        fast = FastGame(seed=seed + i)
        spawns: list[tuple[int, int, int] | None] = list(fast.spawns)  # 2 initial
        move_log: list[Direction] = []
        grid_log: list[list[list[int]]] = []
        score_log: list[int] = []
        state = "playing"
        while state == "playing" and fast.moves < max_moves:
            d = policy(fast.bits)
            moved, _, state = fast.step(d)
            if not moved:
                if state != "playing":
                    break
                # Invalid move while still playable = policy bug, stop game.
                report.mismatches.append(
                    f"game={i}: policy returned invalid move {d.name} "
                    f"with playable board"
                )
                break
            move_log.append(d)
            grid_log.append(fast.grid)
            score_log.append(fast.score)
            spawns.append(fast.spawns[-1])

        # --- Reference replay with forced identical spawns ---
        ref = Board2048()
        report.turns_compared += len(move_log)
        ok = True
        for spawn in spawns[:2]:
            if spawn:
                ref._grid[spawn[0]][spawn[1]] = spawn[2]
        for idx, d in enumerate(move_log):
            ref_changed, _ = ref.move(d)
            if not ref_changed:
                report.mismatches.append(
                    f"game={i} turn={idx}: reference rejected move {d.name} "
                    "but fast engine accepted it"
                )
                ok = False
                break
            spawn = spawns[2 + idx]
            if spawn:
                ref._grid[spawn[0]][spawn[1]] = spawn[2]
            # Turn-by-turn comparison
            if ref.grid != grid_log[idx]:
                report.mismatches.append(
                    f"game={i} turn={idx} ({d.name}): grid mismatch\n"
                    f"ref={ref.grid}\nfast={grid_log[idx]}"
                )
                ok = False
                break
            if ref.score != score_log[idx]:
                report.mismatches.append(
                    f"game={i} turn={idx} ({d.name}): score mismatch "
                    f"ref={ref.score} fast={score_log[idx]}"
                )
                ok = False
                break
        # Final aggregate comparison
        if ok:
            ref_state = ref.get_state()
            if ref_state != state:
                report.mismatches.append(
                    f"game={i}: final state ref={ref_state} fast={state}"
                )
            if ref.max_tile != fast.max_tile:
                report.mismatches.append(
                    f"game={i}: final max tile ref={ref.max_tile} fast={fast.max_tile}"
                )
    return report


def run_accuracy_suite(
    n_fuzz_boards: int = 2000,
    n_replay_games: int = 20,
    seed: int = 7,
) -> AccuracyReport:
    """Run both verification strategies and merge their reports."""
    fuzz = fuzz_moves_equivalence(n_boards=n_fuzz_boards, seed=seed)
    replay = replay_games_equivalence(greedy_policy_safe(), n_games=n_replay_games, seed=seed + 1)
    return AccuracyReport(
        boards_tested=fuzz.boards_tested,
        directions_tested=fuzz.directions_tested,
        games_tested=replay.games_tested,
        turns_compared=replay.turns_compared,
        mismatches=fuzz.mismatches + replay.mismatches,
    )


def greedy_policy_safe() -> Callable[[int], Direction]:
    """Import-lazy greedy policy (avoids circular import at module load)."""
    from ..core.fast import greedy_policy
    return greedy_policy


# ---------------------------------------------------------------------------
# Statistics for batches of games
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


@dataclass
class FastSummary:
    """Statistical summary of a batch of fast games."""
    games: int
    score_mean: float
    score_median: float
    score_std: float
    score_ci95: tuple[float, float]
    score_max: int
    score_min: int
    moves_mean: float
    max_tile_distribution: dict[int, int]
    win_rate: float  # reached 2048
    loss_rate: float
    duration_mean: float
    moves_per_second: float

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "score_mean": round(self.score_mean, 2),
            "score_median": self.score_median,
            "score_std": round(self.score_std, 2),
            "score_ci95": [round(x, 2) for x in self.score_ci95],
            "score_max": self.score_max,
            "score_min": self.score_min,
            "moves_mean": round(self.moves_mean, 2),
            "max_tile_distribution": {
                str(k): v for k, v in sorted(self.max_tile_distribution.items())
            },
            "win_rate": round(self.win_rate, 4),
            "loss_rate": round(self.loss_rate, 4),
            "duration_mean": round(self.duration_mean, 4),
            "moves_per_second": round(self.moves_per_second, 1),
        }


def summarize_results(results: list[FastResult]) -> FastSummary:
    """Compute statistical summary over a batch of fast games."""
    if not results:
        raise ValueError("No results to summarize")
    scores = [float(r.final_score) for r in results]
    moves = [float(r.moves) for r in results]
    durations = [r.duration_seconds for r in results]
    n = len(scores)
    mean = _mean(scores)
    std = _std(scores)
    half = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    dist: dict[int, int] = {}
    for r in results:
        dist[r.max_tile] = dist.get(r.max_tile, 0) + 1
    total_moves = sum(moves)
    total_time = sum(durations)
    sorted_scores = sorted(scores)
    return FastSummary(
        games=n,
        score_mean=mean,
        score_median=sorted_scores[n // 2] if n % 2 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2,
        score_std=std,
        score_ci95=(mean - half, mean + half),
        score_max=int(max(scores)),
        score_min=int(min(scores)),
        moves_mean=_mean(moves),
        max_tile_distribution=dist,
        win_rate=sum(1 for r in results if r.max_tile >= 2048) / n,
        loss_rate=sum(1 for r in results if r.state == "lost") / n,
        duration_mean=_mean(durations),
        moves_per_second=total_moves / total_time if total_time > 0 else 0.0,
    )


def tile_reach_probabilities(results: list[FastResult]) -> dict[int, float]:
    """P(max_tile >= t) for milestone tiles."""
    if not results:
        return {}
    n = len(results)
    return {
        tile: sum(1 for r in results if r.max_tile >= tile) / n
        for tile in (128, 256, 512, 1024, 2048)
    }


def turn_statistics(results: list[FastResult]) -> dict:
    """
    Turn-by-turn statistics across games (requires ``record_curves=True``).

    Returns per-turn averages of cumulative score and max tile for the first
    ``min_len`` turns common to all games, plus aggregate survival curve
    (fraction of games still alive at each turn).
    """
    curves = [r.score_curve for r in results if r.score_curve]
    max_curves = [r.max_tile_curve for r in results if r.max_tile_curve]
    if not curves:
        return {"error": "no curves recorded (use record_curves=True)"}
    min_len = min(len(c) for c in curves)
    survival = [
        sum(1 for c in curves if len(c) > t) / len(curves)
        for t in range(min_len)
    ]
    return {
        "games_with_curves": len(curves),
        "turns_compared": min_len,
        "score_at_turn": {
            t: round(_mean([c[t] for c in curves]), 1)
            for t in _sample_turns(min_len)
        },
        "max_tile_at_turn": {
            t: int(max(max_curves, key=lambda c: c[t])[t]) if max_curves else 0
            for t in _sample_turns(min_len)
        },
        "survival_sample": {
            t: round(survival[t], 3) for t in _sample_turns(min_len)
        },
    }


def _sample_turns(n: int, samples: int = 20) -> list[int]:
    if n <= samples:
        return list(range(n))
    step = n / samples
    return sorted({int(i * step) for i in range(samples)} | {n - 1})
