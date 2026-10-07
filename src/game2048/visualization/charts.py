"""Visualization charts for 2048 benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_score_distribution_plot(
    results: dict,
    title: str = "Score Distribution by Agent",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Create violin/box plot of score distributions."""
    fig = go.Figure()
    
    for agent_name, data in results.items():
        scores = [g["final_score"] for g in data.get("games", [])]
        if not scores:
            continue
        
        fig.add_trace(go.Violin(
            y=scores,
            name=agent_name,
            box_visible=True,
            meanline_visible=True,
            opacity=0.7,
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title="Score",
        xaxis_title="Agent",
        template="plotly_white",
        height=500,
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def create_max_tile_comparison(
    results: dict,
    title: str = "Max Tile Distribution",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Create stacked bar chart of max tile distributions."""
    agents = []
    tile_data = {}
    
    for agent_name, data in results.items():
        dist = data.get("max_tile_distribution", {})
        agents.append(agent_name)
        for tile_log2, count in dist.items():
            tile = 2 ** tile_log2
            if tile not in tile_data:
                tile_data[tile] = []
            tile_data[tile].append(count)
    
    fig = go.Figure()
    
    for tile in sorted(tile_data.keys(), reverse=True):
        fig.add_trace(go.Bar(
            x=agents,
            y=tile_data[tile],
            name=f"{tile}",
            text=[str(v) if v > 0 else "" for v in tile_data[tile]],
            textposition="inside",
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Agent",
        yaxis_title="Number of Games",
        barmode="stack",
        template="plotly_white",
        height=500,
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def create_win_rate_chart(
    results: dict,
    title: str = "Win Rate by Agent",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Create win rate bar chart."""
    agents = []
    win_rates = []
    games_played = []
    
    for agent_name, data in results.items():
        agents.append(agent_name)
        win_rates.append(data.get("win_rate", 0) * 100)
        games_played.append(data.get("num_games", 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=agents,
        y=win_rates,
        text=[f"{wr:.1f}%" for wr in win_rates],
        textposition="outside",
        marker_color="green",
        name="Win Rate",
    ))
    
    # Add games played as secondary axis
    fig.add_trace(go.Scatter(
        x=agents,
        y=games_played,
        mode="lines+markers+text",
        text=[str(g) for g in games_played],
        textposition="top center",
        name="Games Played",
        yaxis="y2",
        line=dict(color="gray", dash="dot"),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Agent",
        yaxis=dict(title="Win Rate (%)", range=[0, max(win_rates) * 1.3 if win_rates else 100]),
        yaxis2=dict(title="Games Played", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        height=500,
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def create_performance_scatter(
    results: dict,
    title: str = "Score vs Move Time",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Create scatter plot of score vs move time."""
    fig = go.Figure()
    
    for agent_name, data in results.items():
        games = data.get("games", [])
        if not games:
            continue
        
        scores = [g["final_score"] for g in games]
        move_times = [g.get("avg_move_time", 0) * 1000 for g in games]  # ms
        
        fig.add_trace(go.Scatter(
            x=move_times,
            y=scores,
            mode="markers",
            name=agent_name,
            marker=dict(size=8, opacity=0.6),
            text=[f"Seed: {g['seed']}<br>Max: {g['max_tile']}" for g in games],
            hovertemplate="%{text}<br>Move Time: %{x:.2f}ms<br>Score: %{y:,}<extra></extra>",
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Average Move Time (ms)",
        yaxis_title="Final Score",
        template="plotly_white",
        height=500,
        xaxis_type="log",
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def create_summary_table(results: dict) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    rows = []
    for agent_name, data in results.items():
        rows.append({
            "Agent": agent_name,
            "Games": data.get("num_games", 0),
            "Win Rate": f"{data.get('win_rate', 0):.1%}",
            "Avg Score": f"{data.get('avg_score', 0):,.0f}",
            "Median": f"{data.get('median_score', 0):,.0f}",
            "Std": f"{data.get('std_score', 0):,.0f}",
            "Max Score": f"{data.get('max_score', 0):,}",
            "Avg Max Tile": f"{data.get('avg_max_tile', 0):,.0f}",
            "Avg Moves": f"{data.get('avg_moves', 0):.0f}",
            "Avg Duration (s)": f"{data.get('avg_duration', 0):.2f}",
            "Move Time (ms)": f"{data.get('avg_move_time', 0) * 1000:.2f}",
        })
    return pd.DataFrame(rows)


def create_gameplay_animation(
    board_history: list,
    title: str = "Gameplay",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Create animated heatmap of gameplay."""
    if not board_history:
        return go.Figure()
    
    frames = []
    for i, board in enumerate(board_history):
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=board,
                colorscale="Viridis",
                zmin=0,
                zmax=np.max(board_history) if board_history else 2048,
                showscale=(i == 0),
            )],
            name=f"Move {i}",
        ))
    
    fig = go.Figure(
        data=[go.Heatmap(
            z=board_history[0],
            colorscale="Viridis",
            zmin=0,
            zmax=np.max(board_history) if board_history else 2048,
        )],
        frames=frames,
    )
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 500}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]},
            ],
        }],
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def create_comparison_dashboard(
    results: dict,
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Create comprehensive comparison dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Score Distribution", "Max Tile Distribution", "Win Rate", "Performance"),
        specs=[
            [{"type": "violin"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )
    
    # Score distribution (violin)
    for agent_name, data in results.items():
        scores = [g["final_score"] for g in data.get("games", [])]
        if scores:
            fig.add_trace(go.Violin(
                y=scores, name=agent_name, showlegend=True,
                box_visible=True, meanline_visible=True, opacity=0.6,
            ), row=1, col=1)
    
    # Max tile distribution (stacked bar)
    agents = list(results.keys())
    tile_data = {}
    for agent_name, data in results.items():
        for tile_log2, count in data.get("max_tile_distribution", {}).items():
            tile = 2 ** tile_log2
            if tile not in tile_data:
                tile_data[tile] = []
            tile_data[tile].append(count)
    
    for tile in sorted(tile_data.keys(), reverse=True):
        fig.add_trace(go.Bar(
            x=agents, y=tile_data[tile], name=f"{tile}", showlegend=True,
        ), row=1, col=2)
    
    # Win rate
    win_rates = [data.get("win_rate", 0) * 100 for data in results.values()]
    fig.add_trace(go.Bar(
        x=agents, y=win_rates, name="Win Rate", showlegend=False,
        text=[f"{wr:.1f}%" for wr in win_rates], textposition="outside",
    ), row=2, col=1)
    
    # Performance scatter
    for agent_name, data in results.items():
        games = data.get("games", [])
        if games:
            scores = [g["final_score"] for g in games]
            move_times = [g.get("avg_move_time", 0) * 1000 for g in games]
            fig.add_trace(go.Scatter(
                x=move_times, y=scores, mode="markers", name=agent_name,
                marker=dict(size=6, opacity=0.5), showlegend=False,
            ), row=2, col=2)
    
    fig.update_layout(
        title="2048 Agent Benchmark Dashboard",
        template="plotly_white",
        height=900,
        barmode="stack",
    )
    
    fig.update_xaxes(title_text="Agent", row=1, col=2)
    fig.update_xaxes(title_text="Agent", row=2, col=1)
    fig.update_xaxes(title_text="Move Time (ms)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Games", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=2)
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def save_all_charts(
    results: dict,
    output_dir: Path,
    prefix: str = "",
) -> list[Path]:
    """Generate and save all charts to directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    
    charts = [
        ("score_dist", create_score_distribution_plot),
        ("max_tiles", create_max_tile_comparison),
        ("win_rate", create_win_rate_chart),
        ("performance", create_performance_scatter),
        ("dashboard", create_comparison_dashboard),
    ]
    
    for name, func in charts:
        path = output_dir / f"{prefix}{name}.html"
        func(results, save_path=path)
        saved.append(path)
    
    # Also save summary table
    df = create_summary_table(results)
    csv_path = output_dir / f"{prefix}summary.csv"
    df.to_csv(csv_path, index=False)
    saved.append(csv_path)
    
    return saved