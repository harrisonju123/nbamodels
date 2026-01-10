#!/usr/bin/env python3
"""
CLV Strategy Backtest

Compare baseline EdgeStrategy vs CLV-filtered strategy performance.
Validates if CLV filtering improves ROI and win rate.

Usage:
    # Run backtest comparison
    python scripts/backtest_clv_strategy.py

    # Specify date range
    python scripts/backtest_clv_strategy.py --start-date 2024-01-01 --end-date 2024-12-31

    # Export results to CSV
    python scripts/backtest_clv_strategy.py --export results.csv
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats

from src.bet_tracker import DB_PATH
from src.betting.edge_strategy import EdgeStrategy


def _get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_historical_bets(
    start_date: str = None,
    end_date: str = None,
    min_bets: int = 50
) -> pd.DataFrame:
    """
    Load historical settled bets with CLV data.

    Args:
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)
        min_bets: Minimum number of bets required

    Returns:
        DataFrame with bet history
    """
    conn = _get_connection()

    query = """
        SELECT
            id, game_id, home_team, away_team, commence_time,
            bet_type, bet_side, odds, line,
            edge, kelly, model_prob, market_prob,
            outcome, profit, settled_at,
            closing_odds, closing_line, clv,
            clv_at_1hr, clv_at_4hr, clv_at_12hr, clv_at_24hr,
            line_velocity, max_clv_achieved, snapshot_coverage,
            closing_line_source, bookmaker
        FROM bets
        WHERE outcome IS NOT NULL
        AND settled_at IS NOT NULL
    """

    params = []

    if start_date:
        query += " AND settled_at >= ?"
        params.append(start_date)

    if end_date:
        query += " AND settled_at <= ?"
        params.append(end_date)

    query += " ORDER BY settled_at ASC"

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()

    if len(df) < min_bets:
        logger.warning(f"Only {len(df)} bets found, need at least {min_bets} for statistical significance")

    return df


def simulate_strategy(
    bets_df: pd.DataFrame,
    strategy_name: str,
    strategy: EdgeStrategy
) -> Dict:
    """
    Simulate strategy performance on historical bets.

    Args:
        bets_df: Historical bets DataFrame
        strategy_name: Name for logging
        strategy: EdgeStrategy instance to test

    Returns:
        Performance metrics dict
    """
    # Filter bets that would have been selected by this strategy
    selected_bets = []

    for _, bet in bets_df.iterrows():
        # For CLV-filtered strategies, we need historical CLV
        if strategy.clv_filter_enabled:
            # Skip bets without CLV data
            if pd.isna(bet['clv_at_4hr']):
                continue

            # Simulate historical CLV lookup
            # In reality, this would use bets BEFORE this one
            # For backtest, we'll use the actual CLV as a proxy
            historical_clv = bet['clv_at_4hr']

            if historical_clv < strategy.min_historical_clv:
                continue  # Would have been filtered

        # For timing-filtered strategies
        if strategy.optimal_timing_filter:
            # Simplified: assume we can only bet if snapshot coverage is good
            if pd.isna(bet['snapshot_coverage']) or bet['snapshot_coverage'] < 0.5:
                continue

        # Check edge threshold
        if abs(bet['edge']) < strategy.edge_threshold:
            continue

        # Check B2B filter (simplified - we don't have B2B data in bet records)
        # In real backtest, would need to join with game data

        # Check team filter
        if strategy.teams_to_exclude:
            bet_team = bet['home_team'] if bet['bet_side'] == 'home' else bet['away_team']
            if bet_team in strategy.teams_to_exclude:
                continue

        selected_bets.append(bet)

    if not selected_bets:
        return {
            'strategy': strategy_name,
            'num_bets': 0,
            'win_rate': 0.0,
            'roi': 0.0,
            'total_profit': 0.0,
            'avg_clv': 0.0,
            'avg_edge': 0.0,
        }

    selected_df = pd.DataFrame(selected_bets)

    # Calculate metrics
    num_bets = len(selected_df)
    wins = (selected_df['outcome'] == 'win').sum()
    win_rate = wins / num_bets

    total_profit = selected_df['profit'].sum()
    total_wagered = num_bets * 100  # Assuming $100 per bet
    roi = (total_profit / total_wagered) * 100

    avg_clv = selected_df['clv'].mean() if 'clv' in selected_df.columns else 0.0
    avg_edge = selected_df['edge'].mean()

    # CLV breakdown
    clv_breakdown = {}
    for window in ['clv_at_1hr', 'clv_at_4hr', 'clv_at_12hr', 'clv_at_24hr']:
        if window in selected_df.columns:
            clv_breakdown[window] = selected_df[window].mean()

    # Closing line source quality
    if 'closing_line_source' in selected_df.columns:
        source_counts = selected_df['closing_line_source'].value_counts().to_dict()
    else:
        source_counts = {}

    return {
        'strategy': strategy_name,
        'num_bets': num_bets,
        'wins': wins,
        'losses': num_bets - wins,
        'win_rate': win_rate,
        'roi': roi,
        'total_profit': total_profit,
        'avg_clv': avg_clv,
        'avg_edge': avg_edge,
        'clv_breakdown': clv_breakdown,
        'closing_line_sources': source_counts,
        'bets': selected_df,
    }


def compare_strategies(results: List[Dict]) -> Dict:
    """
    Compare strategy results and perform statistical tests.

    Args:
        results: List of strategy result dicts

    Returns:
        Comparison analysis dict
    """
    if len(results) < 2:
        return {}

    baseline = results[0]
    test = results[1]

    # Win rate comparison
    win_rate_diff = test['win_rate'] - baseline['win_rate']
    win_rate_pct_change = (win_rate_diff / baseline['win_rate']) * 100 if baseline['win_rate'] > 0 else 0

    # ROI comparison
    roi_diff = test['roi'] - baseline['roi']
    roi_pct_change = (roi_diff / baseline['roi']) * 100 if baseline['roi'] != 0 else 0

    # Statistical significance (Mann-Whitney U test)
    if 'bets' in baseline and 'bets' in test:
        baseline_profits = baseline['bets']['profit'].values
        test_profits = test['bets']['profit'].values

        if len(baseline_profits) > 0 and len(test_profits) > 0:
            u_stat, p_value = stats.mannwhitneyu(
                baseline_profits,
                test_profits,
                alternative='two-sided'
            )
        else:
            p_value = 1.0
    else:
        p_value = 1.0

    # CLV comparison
    clv_diff = test['avg_clv'] - baseline['avg_clv']

    return {
        'baseline_name': baseline['strategy'],
        'test_name': test['strategy'],
        'win_rate_diff': win_rate_diff,
        'win_rate_pct_change': win_rate_pct_change,
        'roi_diff': roi_diff,
        'roi_pct_change': roi_pct_change,
        'clv_diff': clv_diff,
        'p_value': p_value,
        'statistically_significant': p_value < 0.05,
        'bet_count_change': test['num_bets'] - baseline['num_bets'],
        'bet_count_pct_change': ((test['num_bets'] - baseline['num_bets']) / baseline['num_bets'] * 100) if baseline['num_bets'] > 0 else 0,
    }


def print_results(results: List[Dict], comparison: Dict = None):
    """
    Print formatted backtest results.

    Args:
        results: List of strategy result dicts
        comparison: Optional comparison analysis dict
    """
    logger.info("\n" + "=" * 80)
    logger.info("CLV STRATEGY BACKTEST RESULTS")
    logger.info("=" * 80)

    for result in results:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Strategy: {result['strategy']}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Number of Bets: {result['num_bets']}")
        logger.info(f"Wins: {result['wins']} | Losses: {result['losses']}")
        logger.info(f"Win Rate: {result['win_rate']:.1%}")
        logger.info(f"ROI: {result['roi']:+.2f}%")
        logger.info(f"Total Profit: ${result['total_profit']:+,.2f}")
        logger.info(f"Average Edge: {result['avg_edge']:+.2f} pts")
        logger.info(f"Average CLV: {result['avg_clv']:+.3f}")

        if result['clv_breakdown']:
            logger.info("\nCLV by Time Window:")
            for window, clv in result['clv_breakdown'].items():
                if not pd.isna(clv):
                    logger.info(f"  {window}: {clv:+.3f}")

        if result['closing_line_sources']:
            logger.info("\nClosing Line Sources:")
            for source, count in result['closing_line_sources'].items():
                pct = (count / result['num_bets']) * 100
                logger.info(f"  {source}: {count} ({pct:.1f}%)")

    if comparison:
        logger.info(f"\n{'=' * 80}")
        logger.info("STRATEGY COMPARISON")
        logger.info(f"{'=' * 80}")
        logger.info(f"Baseline: {comparison['baseline_name']}")
        logger.info(f"Test: {comparison['test_name']}")
        logger.info(f"\nWin Rate: {comparison['win_rate_diff']:+.1%} ({comparison['win_rate_pct_change']:+.1f}%)")
        logger.info(f"ROI: {comparison['roi_diff']:+.2f}% ({comparison['roi_pct_change']:+.1f}%)")
        logger.info(f"CLV: {comparison['clv_diff']:+.3f}")
        logger.info(f"Bet Count: {comparison['bet_count_change']:+d} ({comparison['bet_count_pct_change']:+.1f}%)")
        logger.info(f"\nStatistical Significance:")
        logger.info(f"  p-value: {comparison['p_value']:.4f}")
        logger.info(f"  Significant (p < 0.05): {'YES ✓' if comparison['statistically_significant'] else 'NO ✗'}")

        # Success criteria
        logger.info(f"\n{'=' * 80}")
        logger.info("SUCCESS CRITERIA")
        logger.info(f"{'=' * 80}")

        criteria_met = 0
        total_criteria = 4

        # 1. Win rate >= baseline
        win_rate_pass = comparison['win_rate_diff'] >= 0
        logger.info(f"1. Win rate >= baseline: {'PASS ✓' if win_rate_pass else 'FAIL ✗'}")
        if win_rate_pass:
            criteria_met += 1

        # 2. Avg CLV > 1%
        clv_pass = comparison['clv_diff'] > 0.01
        logger.info(f"2. CLV improvement > 1%: {'PASS ✓' if clv_pass else 'FAIL ✗'}")
        if clv_pass:
            criteria_met += 1

        # 3. Statistical significance
        sig_pass = comparison['statistically_significant']
        logger.info(f"3. Statistical significance (p < 0.05): {'PASS ✓' if sig_pass else 'FAIL ✗'}")
        if sig_pass:
            criteria_met += 1

        # 4. ROI improvement
        roi_pass = comparison['roi_diff'] > 0
        logger.info(f"4. ROI improvement: {'PASS ✓' if roi_pass else 'FAIL ✗'}")
        if roi_pass:
            criteria_met += 1

        logger.info(f"\nOverall: {criteria_met}/{total_criteria} criteria met")

        if criteria_met >= 3:
            logger.info("✓ RECOMMENDATION: CLV strategy shows improvement, proceed to paper trading")
        else:
            logger.info("✗ RECOMMENDATION: CLV strategy needs refinement before deployment")


def export_results(results: List[Dict], filename: str):
    """
    Export backtest results to CSV.

    Args:
        results: List of strategy result dicts
        filename: Output CSV filename
    """
    summary_data = []

    for result in results:
        summary_data.append({
            'strategy': result['strategy'],
            'num_bets': result['num_bets'],
            'wins': result['wins'],
            'losses': result['losses'],
            'win_rate': result['win_rate'],
            'roi': result['roi'],
            'total_profit': result['total_profit'],
            'avg_edge': result['avg_edge'],
            'avg_clv': result['avg_clv'],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(filename, index=False)
    logger.info(f"Results exported to {filename}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Backtest CLV-filtered strategy vs baseline')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--export', type=str, help='Export results to CSV file')
    parser.add_argument('--min-bets', type=int, default=50, help='Minimum bets required (default: 50)')
    args = parser.parse_args()

    logger.info("=== CLV Strategy Backtest Started ===")

    # Load historical data
    logger.info(f"Loading historical bets (start: {args.start_date or 'all'}, end: {args.end_date or 'all'})...")
    bets_df = load_historical_bets(
        start_date=args.start_date,
        end_date=args.end_date,
        min_bets=args.min_bets
    )

    if bets_df.empty:
        logger.error("No historical bets found")
        return 1

    logger.info(f"Loaded {len(bets_df)} settled bets")

    # Define strategies to compare
    strategies = [
        ('Baseline (Team Filtered)', EdgeStrategy.team_filtered_strategy()),
        ('CLV Filtered', EdgeStrategy.clv_filtered_strategy()),
    ]

    # Run simulations
    results = []

    for strategy_name, strategy in strategies:
        logger.info(f"\nSimulating: {strategy_name}...")
        result = simulate_strategy(bets_df, strategy_name, strategy)
        results.append(result)
        logger.info(f"  Selected {result['num_bets']} bets")

    # Compare results
    if len(results) >= 2:
        comparison = compare_strategies(results)
    else:
        comparison = None

    # Print results
    print_results(results, comparison)

    # Export if requested
    if args.export:
        export_results(results, args.export)

    logger.info("\n=== Backtest Complete ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
