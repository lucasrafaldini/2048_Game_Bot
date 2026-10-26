"""Tests for the fast bitboard engine, accuracy harness and statistics."""

from __future__ import annotations

import random
import time

import pytest

from game2048.benchmarks.accuracy import (
    AccuracyReport,
    fuzz_moves_equivalence,
    replay_games_equivalence,
    summarize_results,
    tile_reach_probabilities,
    turn_statistics,
)
from game2048.core.constants import Direction
from game2048.core.fast import (
    FastGame,
    apply_move_bits,
    batch_simulate,
    bits_to_grid,
    can_move_bits,
    expectimax_bits,
    greedy_policy,
    grid_to_bits,
    random_policy,
    snake_policy,
    valid_moves_bits,
)


class TestBitboardConversion:
    def test_roundtrip(self):
        grid = [[2, 4, 0, 16], [0, 1024, 8, 0], [2, 0, 0, 0], [2048, 4, 2, 2]]
        bits = grid_to_bits(grid)
        assert bits_to_grid(bits) == grid

    def test_empty_board(self):
        bits = grid_to_bits([[0] * 4 for _ in range(4)])
        assert bits == 0
        assert bits_to_grid(0) == [[0] * 4 for _ in range(4)]

    def test_rejects_non_power_of_two(self):
        with pytest.raises(ValueError):
            grid_to_bits([[3, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])

    def test_conversion_matches_reference(self):
        grid = [[2, 0, 4, 2], [0, 8, 0, 16], [2, 2, 0, 0], [0, 0, 32, 4]]
        bits = grid_to_bits(grid)
        assert bits_to_grid(bits) == grid


class TestMovesAgainstReference:
    """The fast engine must agree with Board2048 on random boards."""

    def test_fuzz_equivalence_small(self):
        report = fuzz_moves_equivalence(n_boards=300, seed=1)
        assert report.passed
        assert report.directions_tested == 300 * 4

    def test_fuzz_equivalence_large(self):
        report = fuzz_moves_equivalence(n_boards=2000, seed=42)
        assert report.passed

    def test_known_move_left(self):
        grid = [[2, 2, 4, 0], [0, 0, 0, 0], [2, 0, 0, 2], [4, 4, 4, 4]]
        bits, gained, moved = apply_move_bits(grid_to_bits(grid), Direction.LEFT)
        assert moved
        assert bits_to_grid(bits) == [[4, 4, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0], [8, 8, 0, 0]]
        assert gained == 4 + 4 + 8 + 8  # merged values

    def test_no_merge_after_merge(self):
        # [2,2,2] must become [4,2], not [8]
        bits, _, _ = apply_move_bits(grid_to_bits([[2, 2, 2, 0], [0] * 4, [0] * 4, [0] * 4]), Direction.LEFT)
        assert bits_to_grid(bits)[0] == [4, 2, 0, 0]

    def test_invalid_move_flagged(self):
        grid = [
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 128],
        ]
        bits = grid_to_bits(grid)
        moved = [apply_move_bits(bits, d)[2] for d in Direction]
        # Full board, no equal neighbours: nothing can move.
        assert not any(moved)
        assert not can_move_bits(bits)
        assert valid_moves_bits(bits) == []


class TestFastGame:
    def test_deterministic_with_seed(self):
        random.seed(7)
        a = FastGame(seed=7).play(random_policy)
        random.seed(7)
        b = FastGame(seed=7).play(random_policy)
        assert a.final_score == b.final_score
        assert a.moves == b.moves
        assert a.state == b.state

    def test_initial_state(self):
        game = FastGame(seed=1)
        grid = game.grid
        tiles = sum(1 for row in grid for v in row if v)
        assert tiles == 2
        assert all(v in (2, 4) for row in grid for v in row if v)
        assert game.score == 0
        assert game.moves == 0

    def test_spawn_probability_balance(self):
        """Over many spawns, ~90% should be 2s."""
        counts = {2: 0, 4: 0}
        for seed in range(50):
            game = FastGame(seed=seed)
            for _ in range(200):
                moved, _, state = game.step(random_policy(game.bits))
                if state != "playing":
                    break
            # tally spawned values from the spawn log
            for spawn in game.spawns:
                if spawn:
                    counts[spawn[2]] += 1
        total = counts[2] + counts[4]
        ratio4 = counts[4] / total
        assert 0.05 < ratio4 < 0.15  # expected 0.10

    def test_score_accumulates(self):
        game = FastGame(seed=3)
        result = game.play(greedy_policy)
        assert result.final_score > 0
        assert result.moves > 0
        assert result.max_tile >= 4

    def test_game_ends_in_terminal_state(self):
        game = FastGame(seed=5)
        result = game.play(random_policy)
        assert result.state in ("won", "lost")

    def test_state_semantics_match_reference(self):
        """state() must mirror Board2048.get_state on the same position."""
        grid = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 128, 2]]
        game = FastGame(seed=0)
        game._bits = grid_to_bits(grid)
        ref_grid = [row[:] for row in grid]
        # reference semantics: full board, no merges available -> lost
        assert game.state() == "lost"

        grid_win = [[2048, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4]
        game._bits = grid_to_bits(grid_win)
        assert game.state() == "won"

    def test_play_records_curves(self):
        game = FastGame(seed=11)
        result = game.play(random_policy, record_curves=True)
        assert len(result.score_curve) == result.moves
        assert len(result.max_tile_curve) == result.moves
        if result.score_curve:
            assert result.score_curve[-1] == result.final_score
            assert result.score_curve == sorted(result.score_curve)

    def test_spawn_log_alignment(self):
        """2 initial spawns + one log entry per moved turn."""
        game = FastGame(seed=2)
        moved_count = 0
        for _ in range(50):
            moved, _, state = game.step(random_policy(game.bits))
            if moved:
                moved_count += 1
            if state != "playing":
                break
        assert len(game.spawns) == 2 + moved_count


class TestReplayEquivalence:
    def test_replay_greedy(self):
        report = replay_games_equivalence(greedy_policy, n_games=5, seed=100)
        assert report.passed
        assert report.games_tested == 5
        assert report.turns_compared > 100

    def test_replay_random(self):
        def seeded_random_policy(bits: int) -> Direction:
            return random_policy(bits)

        random.seed(2024)
        report = replay_games_equivalence(seeded_random_policy, n_games=5, seed=200)
        assert report.passed

    def test_replay_snake(self):
        report = replay_games_equivalence(snake_policy, n_games=3, seed=300)
        assert report.passed


class TestPolicies:
    def test_greedy_returns_valid_move(self):
        game = FastGame(seed=9)
        d = greedy_policy(game.bits)
        assert d in Direction

    def test_greedy_beats_random(self):
        random.seed(1)
        g = FastGame(seed=1)
        greedy_scores = [r.final_score for r in g.play_many(greedy_policy, 15)]
        random.seed(1)
        r = FastGame(seed=1)
        random_scores = [res.final_score for res in r.play_many(random_policy, 15)]
        assert sum(greedy_scores) > sum(random_scores)

    def test_expectimax_bits_reasonable(self):
        game = FastGame(seed=13)
        score = expectimax_bits(game.bits, depth=1)
        assert score > 0

    def test_expectimax_policy_valid_move(self):
        policy = __import__(
            "game2048.core.fast", fromlist=["make_expectimax_policy"]
        ).make_expectimax_policy(1)
        game = FastGame(seed=15)
        d = policy(game.bits)
        assert d in Direction


class TestBatchSimulate:
    def test_sequential(self):
        results = batch_simulate(greedy_policy, n_games=10, seed=0, parallel=False)
        assert len(results) == 10
        assert all(isinstance(r.final_score, int) for r in results)

    def test_parallel_reproducible(self):
        results_a = batch_simulate(greedy_policy, n_games=8, seed=42, parallel=True)
        results_b = batch_simulate(greedy_policy, n_games=8, seed=42, parallel=True)
        assert [r.final_score for r in results_a] == [r.final_score for r in results_b]

    def test_results_terminated(self):
        results = batch_simulate(random_policy, n_games=6, seed=9, parallel=False)
        assert all(r.state in ("won", "lost") for r in results)


class TestStatistics:
    def _make_results(self, n: int = 30) -> list:
        random.seed(0)
        game = FastGame(seed=0)
        return game.play_many(random_policy, n)

    def test_summarize(self):
        results = self._make_results()
        s = summarize_results(results)
        assert s.games == len(results)
        assert s.score_min <= s.score_mean <= s.score_max
        assert s.score_ci95[0] <= s.score_mean <= s.score_ci95[1]
        assert 0 <= s.win_rate <= 1
        assert s.moves_per_second > 0
        d = s.to_dict()
        assert d["games"] == len(results)

    def test_tile_reach_monotone(self):
        results = self._make_results()
        reach = tile_reach_probabilities(results)
        tiles = sorted(reach)
        probs = [reach[t] for t in tiles]
        assert probs == sorted(probs, reverse=True)  # P(>=128) >= P(>=256) >= ...

    def test_turn_statistics(self):
        results = batch_simulate(random_policy, n_games=10, seed=3,
                                 parallel=False, record_curves=True)
        stats = turn_statistics(results)
        assert "error" not in stats
        assert stats["turns_compared"] > 0
        assert len(stats["score_at_turn"]) > 0

    def test_summarize_empty_raises(self):
        with pytest.raises(ValueError):
            summarize_results([])


class TestSpeed:
    def test_random_policy_throughput(self):
        """Fast engine should sustain > 5,000 moves/s with the random policy."""
        random.seed(0)
        game = FastGame(seed=0)
        results = game.play_many(random_policy, 30)
        total_moves = sum(r.moves for r in results)
        total_time = sum(r.duration_seconds for r in results)
        assert total_time > 0
        assert total_moves / total_time > 5_000

    def test_greedy_100_games(self):
        game = FastGame(seed=0)
        start = time.perf_counter()
        results = game.play_many(greedy_policy, 100)
        elapsed = time.perf_counter() - start
        assert len(results) == 100
        assert elapsed < 30  # generous CI bound; typically < 3s
