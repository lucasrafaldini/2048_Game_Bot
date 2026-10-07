"""Data storage for benchmark results using SQLite and Parquet."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class BenchmarkDatabase:
    """SQLite database for storing benchmark results."""
    
    def __init__(self, db_path: str | Path = "benchmarks.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    num_games INTEGER NOT NULL,
                    win_rate REAL,
                    avg_score REAL,
                    median_score REAL,
                    std_score REAL,
                    max_score INTEGER,
                    min_score INTEGER,
                    avg_max_tile REAL,
                    avg_moves REAL,
                    avg_duration REAL,
                    avg_move_time REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    final_score INTEGER NOT NULL,
                    max_tile INTEGER NOT NULL,
                    moves INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    state TEXT NOT NULL,
                    move_times_json TEXT,
                    nodes_evaluated_json TEXT,
                    board_history_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_agent 
                ON runs (agent_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_games_run 
                ON games (run_id)
            """)
    
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def save_run(self, result: dict, config: dict) -> int:
        """Save a benchmark run and return run ID."""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO runs (
                    agent_name, config_json, timestamp, num_games,
                    win_rate, avg_score, median_score, std_score,
                    max_score, min_score, avg_max_tile,
                    avg_moves, avg_duration, avg_move_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result["agent_name"],
                json.dumps(config),
                result.get("timestamp", ""),
                result["num_games"],
                result.get("win_rate"),
                result.get("avg_score"),
                result.get("median_score"),
                result.get("std_score"),
                result.get("max_score"),
                result.get("min_score"),
                result.get("avg_max_tile"),
                result.get("avg_moves"),
                result.get("avg_duration"),
                result.get("avg_move_time"),
            ))
            run_id = cursor.lastrowid
            
            # Save individual games
            for game in result.get("games", []):
                conn.execute("""
                    INSERT INTO games (
                        run_id, seed, final_score, max_tile, moves,
                        duration_seconds, state, move_times_json,
                        nodes_evaluated_json, board_history_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    game["seed"],
                    game["final_score"],
                    game["max_tile"],
                    game["moves"],
                    game["duration_seconds"],
                    game["state"],
                    json.dumps(game.get("move_times", [])),
                    json.dumps(game.get("nodes_evaluated", [])),
                    json.dumps([b.tolist() for b in game.get("board_history", [])]),
                ))
            
            return run_id
    
    def load_runs(self, agent_name: Optional[str] = None) -> list[dict]:
        """Load all runs, optionally filtered by agent."""
        with self._connect() as conn:
            if agent_name:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE agent_name = ? ORDER BY timestamp DESC",
                    (agent_name,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY timestamp DESC"
                ).fetchall()
            
            return [dict(row) for row in rows]
    
    def load_games(self, run_id: int) -> list[dict]:
        """Load all games for a run."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM games WHERE run_id = ? ORDER BY seed",
                (run_id,)
            ).fetchall()
            
            games = []
            for row in rows:
                game = dict(row)
                game["move_times"] = json.loads(game["move_times_json"])
                game["nodes_evaluated"] = json.loads(game["nodes_evaluated_json"])
                game["board_history"] = json.loads(game["board_history_json"])
                games.append(game)
            
            return games
    
    def to_dataframe(self, agent_name: Optional[str] = None) -> pd.DataFrame:
        """Load all games as a pandas DataFrame."""
        with self._connect() as conn:
            if agent_name:
                query = """
                    SELECT g.*, r.agent_name, r.timestamp as run_timestamp
                    FROM games g
                    JOIN runs r ON g.run_id = r.id
                    WHERE r.agent_name = ?
                """
                df = pd.read_sql_query(query, conn, params=(agent_name,))
            else:
                query = """
                    SELECT g.*, r.agent_name, r.timestamp as run_timestamp
                    FROM games g
                    JOIN runs r ON g.run_id = r.id
                """
                df = pd.read_sql_query(query, conn)
            
            return df
    
    def get_agent_summary(self) -> pd.DataFrame:
        """Get summary statistics per agent."""
        with self._connect() as conn:
            return pd.read_sql_query("""
                SELECT 
                    agent_name,
                    COUNT(*) as num_runs,
                    SUM(num_games) as total_games,
                    AVG(win_rate) as avg_win_rate,
                    AVG(avg_score) as overall_avg_score,
                    MAX(max_score) as best_score,
                    AVG(avg_max_tile) as avg_max_tile,
                    AVG(avg_move_time) as avg_move_time
                FROM runs
                GROUP BY agent_name
                ORDER BY overall_avg_score DESC
            """, conn)


class ParquetStore:
    """Parquet-based storage for large-scale benchmark data."""
    
    def __init__(self, base_path: str | Path = "benchmark_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_run(self, result: dict, config: dict, run_id: str) -> None:
        """Save run as Parquet files."""
        run_dir = self.base_path / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save metadata
        meta = {
            "agent_name": result["agent_name"],
            "config": config,
            "timestamp": result.get("timestamp"),
            "summary": {k: v for k, v in result.items() if k != "games"},
        }
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        
        # Save games as Parquet
        if result.get("games"):
            df = pd.DataFrame(result["games"])
            df.to_parquet(run_dir / "games.parquet", index=False)
    
    def load_run(self, run_id: str) -> dict:
        """Load run from Parquet files."""
        run_dir = self.base_path / run_id
        
        meta = json.loads((run_dir / "metadata.json").read_text())
        games_df = pd.read_parquet(run_dir / "games.parquet")
        
        return {
            **meta["summary"],
            "agent_name": meta["agent_name"],
            "timestamp": meta["timestamp"],
            "games": games_df.to_dict("records"),
        }
    
    def load_all(self) -> dict:
        """Load all runs as a combined DataFrame."""
        all_games = []
        for run_dir in self.base_path.iterdir():
            if run_dir.is_dir() and (run_dir / "games.parquet").exists():
                df = pd.read_parquet(run_dir / "games.parquet")
                meta = json.loads((run_dir / "metadata.json").read_text())
                df["agent_name"] = meta["agent_name"]
                df["run_timestamp"] = meta["timestamp"]
                all_games.append(df)
        
        if all_games:
            return pd.concat(all_games, ignore_index=True)
        return pd.DataFrame()
    
    def list_runs(self) -> list[dict]:
        """List all available runs."""
        runs = []
        for run_dir in self.base_path.iterdir():
            if run_dir.is_dir() and (run_dir / "metadata.json").exists():
                meta = json.loads((run_dir / "metadata.json").read_text())
                runs.append({
                    "run_id": run_dir.name,
                    "agent_name": meta["agent_name"],
                    "timestamp": meta["timestamp"],
                    "num_games": meta["summary"].get("num_games", 0),
                })
        return sorted(runs, key=lambda x: x["timestamp"], reverse=True)


def create_database(db_path: str = "benchmarks.db") -> BenchmarkDatabase:
    """Factory function to create database."""
    return BenchmarkDatabase(db_path)


def create_parquet_store(base_path: str = "benchmark_data") -> ParquetStore:
    """Factory function to create Parquet store."""
    return ParquetStore(base_path)