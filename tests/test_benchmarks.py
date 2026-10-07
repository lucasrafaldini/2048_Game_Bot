"""Tests for benchmark framework."""

import pytest
import tempfile
from pathlib import Path

from game2048.benchmarks import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkResult,
    ComparisonResult,
    GameMetrics,
    run_quick_benchmark,
)
from game2048.ai import get_agent
from game2048.data import BenchmarkDatabase, ParquetStore


def greedy_factory():
    return get_agent("greedy")


class TestBenchmarkMetrics:
    """Tests for metrics data structures."""
    
    def test_game_metrics(self):
        metrics = GameMetrics(
            agent_name="test",
            seed=42,
            final_score=1000,
            max_tile=256,
            moves=50,
            duration_seconds=1.5,
            state="lost",
        )
        assert metrics.won is False
        assert metrics.max_tile_log2 == 8
    
    def test_game_metrics_won(self):
        metrics = GameMetrics(
            agent_name="test",
            seed=42,
            final_score=10000,
            max_tile=2048,
            moves=200,
            duration_seconds=5.0,
            state="won",
        )
        assert metrics.won is True
        assert metrics.max_tile_log2 == 11
    
    def test_benchmark_result(self):
        result = BenchmarkResult(agent_name="test", num_games=0, seeds=[1, 2, 3])
        
        for i, seed in enumerate([1, 2, 3]):
            metrics = GameMetrics(
                agent_name="test",
                seed=seed,
                final_score=1000 * (i + 1),
                max_tile=256,
                moves=50,
                duration_seconds=1.0,
                state="lost",
            )
            result.add_game(metrics)
        
        assert result.num_games == 3
        assert result.avg_score == 2000
        assert result.median_score == 2000
        assert result.max_score == 3000
        assert result.min_score == 1000
    
    def test_comparison_result(self):
        comp = ComparisonResult()
        
        result1 = BenchmarkResult(agent_name="A", num_games=2, seeds=[1, 2])
        result1.add_game(GameMetrics("A", 1, 1000, 256, 50, 1.0, "lost"))
        result1.add_game(GameMetrics("A", 2, 2000, 512, 60, 1.5, "lost"))
        
        result2 = BenchmarkResult(agent_name="B", num_games=2, seeds=[1, 2])
        result2.add_game(GameMetrics("B", 1, 1500, 256, 55, 1.2, "lost"))
        result2.add_game(GameMetrics("B", 2, 2500, 512, 65, 1.8, "lost"))
        
        comp.add_result(result1)
        comp.add_result(result2)
        
        ranking = comp.get_ranking("avg_score")
        assert ranking[0][0] == "B"
        assert ranking[1][0] == "A"


class TestBenchmarkRunner:
    """Tests for benchmark runner."""
    
    def test_quick_benchmark(self):
        result = run_quick_benchmark("greedy", num_games=3)
        assert result.num_games == 3
        assert result.agent_name == "greedy"
        assert len(result.games) == 3
    
    def test_runner_sequential(self):
        config = BenchmarkConfig(num_games=3, parallel=False, verbose=True)
        runner = BenchmarkRunner(config)
        result = runner.run(lambda: get_agent("greedy"), "greedy")
        
        assert result.num_games == 3
        assert result.agent_name == "greedy"
    
    def test_runner_parallel(self):
        config = BenchmarkConfig(num_games=3, parallel=True, num_workers=2)
        runner = BenchmarkRunner(config)
        result = runner.run(greedy_factory, "greedy")
        
        assert result.num_games == 3
    
    def test_compare_agents(self):
        config = BenchmarkConfig(num_games=2, parallel=False)
        runner = BenchmarkRunner(config)
        
        factories = {
            "greedy": lambda: get_agent("greedy"),
            "snake": lambda: get_agent("snake"),
        }
        
        comparison = runner.compare(factories)
        assert "greedy" in comparison.results
        assert "snake" in comparison.results


class TestDatabase:
    """Tests for data storage."""
    
    def test_sqlite_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = BenchmarkDatabase(db_path)
            
            # Save a run
            result = {
                "agent_name": "test",
                "timestamp": "2024-01-01T00:00:00",
                "num_games": 2,
                "win_rate": 0.0,
                "avg_score": 1500,
                "median_score": 1500,
                "std_score": 500,
                "max_score": 2000,
                "min_score": 1000,
                "avg_max_tile": 256,
                "avg_moves": 50,
                "avg_duration": 1.0,
                "avg_move_time": 0.001,
                "games": [
                    {
                        "seed": 1,
                        "final_score": 1000,
                        "max_tile": 256,
                        "moves": 40,
                        "duration_seconds": 0.8,
                        "state": "lost",
                        "move_times": [0.001] * 40,
                        "nodes_evaluated": [0] * 40,
                        "board_history": [],
                    },
                    {
                        "seed": 2,
                        "final_score": 2000,
                        "max_tile": 512,
                        "moves": 60,
                        "duration_seconds": 1.2,
                        "state": "lost",
                        "move_times": [0.001] * 60,
                        "nodes_evaluated": [0] * 60,
                        "board_history": [],
                    },
                ],
            }
            
            run_id = db.save_run(result, {"num_games": 2})
            assert run_id == 1
            
            # Load runs
            runs = db.load_runs("test")
            assert len(runs) == 1
            assert runs[0]["agent_name"] == "test"
            
            # Load games
            games = db.load_games(run_id)
            assert len(games) == 2
            assert games[0]["final_score"] == 1000
            
            # DataFrame
            df = db.to_dataframe()
            assert len(df) == 2
            assert "agent_name" in df.columns
    
    def test_parquet_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetStore(tmpdir)
            
            result = {
                "agent_name": "test",
                "timestamp": "2024-01-01T00:00:00",
                "num_games": 1,
                "win_rate": 0.0,
                "avg_score": 1000,
                "median_score": 1000,
                "std_score": 0,
                "max_score": 1000,
                "min_score": 1000,
                "avg_max_tile": 256,
                "avg_moves": 50,
                "avg_duration": 1.0,
                "avg_move_time": 0.001,
                "games": [
                    {
                        "seed": 1,
                        "final_score": 1000,
                        "max_tile": 256,
                        "moves": 50,
                        "duration_seconds": 1.0,
                        "state": "lost",
                    },
                ],
            }
            
            store.save_run(result, {}, "test_run_1")
            
            loaded = store.load_run("test_run_1")
            assert loaded["agent_name"] == "test"
            assert loaded["num_games"] == 1
            
            # List runs
            runs = store.list_runs()
            assert len(runs) == 1


class TestAnalysis:
    """Tests for statistical analysis."""
    
    def test_pairwise_comparison(self):
        import pandas as pd
        from game2048.data import pairwise_comparison
        
        df = pd.DataFrame({
            "agent_name": ["A"] * 10 + ["B"] * 10,
            "final_score": [1000] * 10 + [2000] * 10,
        })
        
        result = pairwise_comparison(df, "final_score")
        assert len(result) == 1
        assert bool(result.iloc[0]["significant"]) is True
    
    def test_confidence_intervals(self):
        import pandas as pd
        from game2048.data import compute_confidence_intervals
        
        df = pd.DataFrame({
            "agent_name": ["A"] * 10 + ["B"] * 10,
            "final_score": [1000] * 10 + [2000] * 10,
        })
        
        ci = compute_confidence_intervals(df, "final_score")
        assert len(ci) == 2
        assert ci.iloc[0]["mean"] == 1000
        assert ci.iloc[1]["mean"] == 2000
    
    def test_generate_report(self):
        import pandas as pd
        from game2048.data import generate_report
        
        df = pd.DataFrame({
            "agent_name": ["A"] * 10 + ["B"] * 10,
            "final_score": [1000] * 10 + [2000] * 10,
            "max_tile": [256] * 10 + [512] * 10,
            "seed": list(range(10)) * 2,
        })
        
        report = generate_report(df)
        assert "A" in report
        assert "B" in report
        assert "CONFIDENCE INTERVALS" in report
        assert "PAIRWISE COMPARISONS" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])