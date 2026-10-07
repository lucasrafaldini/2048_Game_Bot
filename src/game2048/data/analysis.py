"""Statistical analysis utilities for benchmark data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatisticalTest:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    interpretation: str = ""


def compare_agents(
    df: pd.DataFrame,
    agent_a: str,
    agent_b: str,
    metric: str = "final_score",
    test: str = "mannwhitney",
    alpha: float = 0.05,
) -> StatisticalTest:
    """
    Compare two agents statistically.
    
    Args:
        df: DataFrame with columns ['agent_name', metric]
        agent_a: Name of first agent
        agent_b: Name of second agent
        metric: Column to compare
        test: Test type ('mannwhitney', 'ttest', 'ks')
        alpha: Significance level
    
    Returns:
        StatisticalTest result
    """
    data_a = df[df["agent_name"] == agent_a][metric].dropna()
    data_b = df[df["agent_name"] == agent_b][metric].dropna()
    
    if len(data_a) < 2 or len(data_b) < 2:
        return StatisticalTest(
            test_name=test,
            statistic=0,
            p_value=1,
            significant=False,
            interpretation="Insufficient data",
        )
    
    if test == "mannwhitney":
        # Non-parametric, doesn't assume normal distribution
        stat, p = stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
        # Effect size: rank-biserial correlation
        effect = 1 - (2 * stat) / (len(data_a) * len(data_b))
    elif test == "ttest":
        # Parametric, assumes normal distribution
        stat, p = stats.ttest_ind(data_a, data_b, equal_var=False)
        # Cohen's d
        pooled_std = np.sqrt((data_a.var() + data_b.var()) / 2)
        effect = (data_a.mean() - data_b.mean()) / pooled_std if pooled_std > 0 else 0
    elif test == "ks":
        # Kolmogorov-Smirnov test for distribution difference
        stat, p = stats.ks_2samp(data_a, data_b)
        effect = stat
    else:
        raise ValueError(f"Unknown test: {test}")
    
    significant = p < alpha
    
    if test == "mannwhitney":
        interpretation = (
            f"{agent_a} {'significantly outperforms' if data_a.median() > data_b.median() else 'underperforms'} "
            f"{agent_b} (p={p:.4f})"
            if significant else f"No significant difference between {agent_a} and {agent_b} (p={p:.4f})"
        )
    else:
        interpretation = (
            f"Significant difference (p={p:.4f})"
            if significant else f"No significant difference (p={p:.4f})"
        )
    
    return StatisticalTest(
        test_name=test,
        statistic=stat,
        p_value=p,
        significant=significant,
        effect_size=effect,
        interpretation=interpretation,
    )


def pairwise_comparison(
    df: pd.DataFrame,
    metric: str = "final_score",
    test: str = "mannwhitney",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run pairwise comparisons between all agents."""
    agents = df["agent_name"].unique()
    results = []
    
    for i, a in enumerate(agents):
        for b in agents[i+1:]:
            test_result = compare_agents(df, a, b, metric, test, alpha)
            results.append({
                "agent_a": a,
                "agent_b": b,
                "test": test_result.test_name,
                "statistic": test_result.statistic,
                "p_value": test_result.p_value,
                "significant": test_result.significant,
                "effect_size": test_result.effect_size,
                "interpretation": test_result.interpretation,
            })
    
    return pd.DataFrame(results)


def compute_confidence_intervals(
    df: pd.DataFrame,
    metric: str = "final_score",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute confidence intervals for each agent."""
    results = []
    
    for agent in df["agent_name"].unique():
        data = df[df["agent_name"] == agent][metric].dropna()
        n = len(data)
        mean = data.mean()
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
        
        results.append({
            "agent": agent,
            "n": n,
            "mean": mean,
            "median": data.median(),
            "std": data.std(),
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "ci_width": ci[1] - ci[0],
        })
    
    return pd.DataFrame(results)


def analyze_tile_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze max tile distribution per agent."""
    results = []
    
    for agent in df["agent_name"].unique():
        agent_df = df[df["agent_name"] == agent]
        total = len(agent_df)
        
        for tile_log2 in range(1, 12):  # up to 2048
            tile = 2 ** tile_log2
            count = len(agent_df[agent_df["max_tile"] >= tile])
            pct = count / total * 100 if total > 0 else 0
            results.append({
                "agent": agent,
                "tile": tile,
                "tile_log2": tile_log2,
                "count": count,
                "percentage": pct,
            })
    
    return pd.DataFrame(results)


def learning_curve_analysis(
    df: pd.DataFrame,
    metric: str = "final_score",
    window: int = 10,
) -> pd.DataFrame:
    """Analyze performance over time (by seed order)."""
    results = []
    
    for agent in df["agent_name"].unique():
        agent_df = df[df["agent_name"] == agent].sort_values("seed")
        scores = agent_df[metric].values
        
        if len(scores) >= window:
            rolling_mean = pd.Series(scores).rolling(window).mean()
            rolling_std = pd.Series(scores).rolling(window).std()
            
            for i in range(window-1, len(scores)):
                results.append({
                    "agent": agent,
                    "game_idx": i,
                    "seed": agent_df.iloc[i]["seed"],
                    "rolling_mean": rolling_mean.iloc[i],
                    "rolling_std": rolling_std.iloc[i],
                })
    
    return pd.DataFrame(results)


def compute_effect_sizes(
    df: pd.DataFrame,
    baseline_agent: str,
    metric: str = "final_score",
) -> pd.DataFrame:
    """Compute Cohen's d effect sizes relative to baseline."""
    baseline_data = df[df["agent_name"] == baseline_agent][metric].dropna()
    baseline_mean = baseline_data.mean()
    baseline_std = baseline_data.std()
    
    results = []
    
    for agent in df["agent_name"].unique():
        if agent == baseline_agent:
            continue
        
        agent_data = df[df["agent_name"] == agent][metric].dropna()
        
        # Cohen's d
        pooled_std = np.sqrt((baseline_std**2 + agent_data.std()**2) / 2)
        d = (agent_data.mean() - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        # Hedges' g (corrected for small sample sizes)
        n1, n2 = len(baseline_data), len(agent_data)
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        g = d * correction
        
        results.append({
            "agent": agent,
            "cohens_d": d,
            "hedges_g": g,
            "interpretation": (
                "Large" if abs(g) > 0.8 else
                "Medium" if abs(g) > 0.5 else
                "Small" if abs(g) > 0.2 else
                "Negligible"
            ),
        })
    
    return pd.DataFrame(results)


def generate_report(
    df: pd.DataFrame,
    metric: str = "final_score",
) -> str:
    """Generate a text report of the analysis."""
    lines = [
        "=" * 60,
        "2048 BENCHMARK ANALYSIS REPORT",
        "=" * 60,
        "",
    ]
    
    # Summary table
    ci_df = compute_confidence_intervals(df, metric)
    lines.append("CONFIDENCE INTERVALS (95%):")
    lines.append("-" * 60)
    for _, row in ci_df.iterrows():
        lines.append(
            f"{row['agent']:20s} | n={row['n']:3d} | "
            f"mean={row['mean']:8.0f} | "
            f"CI=[{row['ci_lower']:8.0f}, {row['ci_upper']:8.0f}]"
        )
    lines.append("")
    
    # Pairwise comparisons
    pw_df = pairwise_comparison(df, metric)
    lines.append("PAIRWISE COMPARISONS (Mann-Whitney U):")
    lines.append("-" * 60)
    for _, row in pw_df.iterrows():
        sig = "✓" if row["significant"] else "✗"
        lines.append(
            f"{sig} {row['agent_a']:15s} vs {row['agent_b']:15s} | "
            f"p={row['p_value']:.4f} | "
            f"effect={row['effect_size']:.3f}"
        )
    lines.append("")
    
    # Effect sizes vs best
    best_agent = ci_df.loc[ci_df["mean"].idxmax(), "agent"]
    es_df = compute_effect_sizes(df, best_agent, metric)
    lines.append(f"EFFECT SIZES vs BEST ({best_agent}):")
    lines.append("-" * 60)
    for _, row in es_df.iterrows():
        lines.append(
            f"{row['agent']:20s} | d={row['cohens_d']:6.3f} | "
            f"g={row['hedges_g']:6.3f} | {row['interpretation']}"
        )
    lines.append("")
    
    # Tile distribution
    tile_df = analyze_tile_distribution(df)
    lines.append("MAX TILE ACHIEVEMENT RATES:")
    lines.append("-" * 60)
    for agent in df["agent_name"].unique():
        agent_tiles = tile_df[tile_df["agent"] == agent]
        line = f"{agent:20s} | "
        for _, row in agent_tiles.iterrows():
            if row["percentage"] > 0:
                line += f"{row['tile']}: {row['percentage']:.1f}%  "
        lines.append(line)
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)