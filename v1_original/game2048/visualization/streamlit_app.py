"""Streamlit dashboard for 2048 benchmark visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .charts import (
    create_score_distribution_plot,
    create_max_tile_comparison,
    create_win_rate_chart,
    create_performance_scatter,
    create_comparison_dashboard,
    create_summary_table,
)


st.set_page_config(
    page_title="2048 AI Benchmark Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_results(file_path: Path) -> dict:
    """Load benchmark results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def load_multiple_results(directory: Path) -> dict:
    """Load all JSON result files from a directory."""
    results = {}
    for json_file in directory.glob("*.json"):
        try:
            data = load_results(json_file)
            # Use agent name from data or filename
            agent_name = data.get("agent_name", json_file.stem)
            results[agent_name] = data
        except Exception as e:
            st.error(f"Error loading {json_file}: {e}")
    return results


def render_sidebar() -> dict:
    """Render sidebar controls and return selected options."""
    st.sidebar.title("🎮 2048 Benchmark")
    
    # Data source
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio(
        "Select data source",
        ["Upload JSON", "Directory", "Run Benchmark"],
    )
    
    results = {}
    
    if data_source == "Upload JSON":
        uploaded_files = st.sidebar.file_uploader(
            "Upload benchmark result JSON files",
            type="json",
            accept_multiple_files=True,
        )
        if uploaded_files:
            for f in uploaded_files:
                data = json.load(f)
                agent_name = data.get("agent_name", f.name)
                results[agent_name] = data
    
    elif data_source == "Directory":
        dir_path = st.sidebar.text_input("Results directory", "benchmark_results")
        if st.sidebar.button("Load from Directory"):
            results = load_multiple_results(Path(dir_path))
            if results:
                st.sidebar.success(f"Loaded {len(results)} agents")
            else:
                st.sidebar.warning("No valid JSON files found")
    
    elif data_source == "Run Benchmark":
        st.sidebar.info("Run benchmarks via CLI:\n`game2048-benchmark run expectimax -n 100`")
    
    # Visualization options
    st.sidebar.header("Visualization")
    show_distributions = st.sidebar.checkbox("Score Distributions", True)
    show_max_tiles = st.sidebar.checkbox("Max Tile Distribution", True)
    show_win_rate = st.sidebar.checkbox("Win Rate", True)
    show_performance = st.sidebar.checkbox("Performance Scatter", True)
    show_dashboard = st.sidebar.checkbox("Full Dashboard", True)
    show_table = st.sidebar.checkbox("Summary Table", True)
    
    # Export options
    st.sidebar.header("Export")
    if st.sidebar.button("Export Charts as HTML"):
        if results:
            export_charts(results)
            st.sidebar.success("Charts exported!")
        else:
            st.sidebar.warning("No results to export")
    
    return {
        "results": results,
        "show_distributions": show_distributions,
        "show_max_tiles": show_max_tiles,
        "show_win_rate": show_win_rate,
        "show_performance": show_performance,
        "show_dashboard": show_dashboard,
        "show_table": show_table,
    }


def render_main(results: dict, options: dict) -> None:
    """Render main dashboard content."""
    if not results:
        st.info("👈 Load benchmark results from the sidebar to visualize")
        st.markdown("""
        ## 2048 AI Benchmark Dashboard
        
        This dashboard visualizes benchmark results for 2048 AI agents.
        
        ### Getting Started
        1. Run benchmarks via CLI:
           ```bash
           game2048-benchmark compare "expectimax,mcts,greedy,snake" -n 100
           ```
        2. Load the generated JSON files using the sidebar
        3. Explore the visualizations
        
        ### Available Agents
        - **Random**: Random valid moves (baseline)
        - **Greedy**: 1-step heuristic lookahead
        - **Snake**: Corner-max strategy
        - **Monotonic**: Monotonicity-focused
        - **Expectimax**: Gold-standard adversarial search
        - **MCTS**: Monte Carlo Tree Search
        """)
        return
    
    # Overview metrics
    st.header("📊 Overview")
    
    cols = st.columns(4)
    for i, (agent_name, data) in enumerate(results.items()):
        with cols[i % 4]:
            st.metric(
                label=agent_name,
                value=f"{data.get('avg_score', 0):,.0f}",
                delta=f"Win: {data.get('win_rate', 0):.1%}",
            )
    
    # Full dashboard
    if options["show_dashboard"]:
        st.header("📈 Full Dashboard")
        fig = create_comparison_dashboard(results)
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual charts
    col1, col2 = st.columns(2)
    
    if options["show_distributions"]:
        with col1:
            st.subheader("Score Distributions")
            fig = create_score_distribution_plot(results)
            st.plotly_chart(fig, use_container_width=True)
    
    if options["show_max_tiles"]:
        with col2:
            st.subheader("Max Tile Distribution")
            fig = create_max_tile_comparison(results)
            st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    if options["show_win_rate"]:
        with col3:
            st.subheader("Win Rate")
            fig = create_win_rate_chart(results)
            st.plotly_chart(fig, use_container_width=True)
    
    if options["show_performance"]:
        with col4:
            st.subheader("Score vs Move Time")
            fig = create_performance_scatter(results)
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    if options["show_table"]:
        st.header("📋 Summary Statistics")
        df = create_summary_table(results)
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "benchmark_summary.csv",
            "text/csv",
        )
    
    # Detailed game explorer
    if len(results) == 1:
        render_game_explorer(list(results.values())[0])


def render_game_explorer(data: dict) -> None:
    """Render detailed game explorer for single agent."""
    st.header("🔍 Game Explorer")
    
    games = data.get("games", [])
    if not games:
        st.info("No game history available. Run benchmark with `--history` flag.")
        return
    
    # Game selector
    game_idx = st.selectbox(
        "Select game",
        range(len(games)),
        format_func=lambda i: f"Game {i+1} (Seed: {games[i]['seed']}, Score: {games[i]['final_score']:,}, Max: {games[i]['max_tile']})",
    )
    
    game = games[game_idx]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{game['final_score']:,}")
    with col2:
        st.metric("Max Tile", game['max_tile'])
    with col3:
        st.metric("Moves", game['moves'])
    
    # Board history animation
    if game.get("board_history"):
        st.subheader("Gameplay Animation")
        boards = game["board_history"]
        if boards:
            fig = go.Figure(
                data=[go.Heatmap(
                    z=boards[0],
                    colorscale="Viridis",
                    zmin=0,
                    zmax=max(max(b.max() for b in boards), 2048),
                )],
                frames=[go.Frame(
                    data=[go.Heatmap(z=b)],
                    name=f"Move {i}",
                ) for i, b in enumerate(boards)],
            )
            fig.update_layout(
                height=400,
                updatemenus=[{
                    "buttons": [
                        {"label": "▶", "method": "animate", "args": [None, {"frame": {"duration": 300}}]},
                        {"label": "⏸", "method": "animate", "args": [[None], {"mode": "immediate"}]},
                    ],
                }],
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Move time analysis
    if game.get("move_times"):
        st.subheader("Move Time Analysis")
        move_df = pd.DataFrame({
            "Move": range(1, len(game["move_times"]) + 1),
            "Time (ms)": [t * 1000 for t in game["move_times"]],
        })
        fig = px.line(move_df, x="Move", y="Time (ms)", title="Move Time per Turn")
        st.plotly_chart(fig, use_container_width=True)
    
    # Nodes evaluated
    if game.get("nodes_evaluated"):
        st.subheader("Search Nodes Evaluated")
        nodes_df = pd.DataFrame({
            "Move": range(1, len(game["nodes_evaluated"]) + 1),
            "Nodes": game["nodes_evaluated"],
        })
        fig = px.bar(nodes_df, x="Move", y="Nodes", title="Nodes Evaluated per Move")
        st.plotly_chart(fig, use_container_width=True)


def export_charts(results: dict, output_dir: str = "benchmark_charts") -> None:
    """Export all charts as HTML files."""
    from .charts import save_all_charts
    save_all_charts(results, Path(output_dir))


def main() -> None:
    """Main entry point for Streamlit app."""
    # Custom CSS
    st.markdown("""
        <style>
        .main > div { padding-top: 2rem; }
        .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
        </style>
    """, unsafe_allow_html=True)
    
    options = render_sidebar()
    render_main(options["results"], options)


if __name__ == "__main__":
    main()