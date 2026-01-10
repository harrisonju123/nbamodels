#!/usr/bin/env python3
"""
Line Movement Timing Analysis

Analyzes historical odds data to identify:
1. When lines move the most (high volatility windows)
2. When lines are most stable (low volatility windows)
3. Optimal betting windows to capture best prices

Uses 405 days of historical odds snapshots (2022-2026).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from collections import defaultdict


def load_historical_odds(days_back: int = 90) -> pd.DataFrame:
    """Load historical odds data."""
    logger.info(f"Loading historical odds (last {days_back} days)...")

    odds_files = sorted(glob.glob('data/historical_odds/odds_*.parquet'))

    if not odds_files:
        logger.error("No historical odds files found")
        return pd.DataFrame()

    # Filter to recent files
    if days_back:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        odds_files = [
            f for f in odds_files
            if datetime.strptime(f.split('/')[-1].replace('odds_', '').replace('.parquet', ''), '%Y-%m-%d') >= cutoff_date
        ]

    logger.info(f"Loading {len(odds_files)} files...")

    # Load all files
    dfs = []
    for f in odds_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    odds = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(odds):,} odds records across {odds['game_id'].nunique()} games")

    return odds


def calculate_line_movement(odds: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate line movement for each game.

    Returns DataFrame with:
    - game_id, market, bookmaker
    - hours_before_game
    - line_movement (change from opening)
    - volatility (std of line over time)
    """
    logger.info("Calculating line movement...")

    # Filter to spread market only for simplicity
    spread = odds[odds['market'] == 'spread'].copy()

    if len(spread) == 0:
        logger.warning("No spread data found")
        return pd.DataFrame()

    # Parse timestamps
    spread['last_update'] = pd.to_datetime(spread['last_update'])
    spread['commence_time'] = pd.to_datetime(spread['commence_time'])

    # Calculate hours before game
    spread['hours_before_game'] = (
        (spread['commence_time'] - spread['last_update']).dt.total_seconds() / 3600
    )

    # Filter to reasonable time windows (0-72 hours before game)
    spread = spread[
        (spread['hours_before_game'] >= 0) &
        (spread['hours_before_game'] <= 72)
    ].copy()

    logger.info(f"Analyzing {len(spread):,} spread records")

    # For each game/bookmaker/side, track line movement
    movements = []

    for (game_id, bookmaker, team), group in spread.groupby(['game_id', 'bookmaker', 'team']):
        if len(group) < 2:
            continue

        # Sort by time
        group = group.sort_values('last_update')

        opening_line = group.iloc[-1]['line']  # Earliest snapshot (furthest from game)
        closing_line = group.iloc[0]['line']   # Latest snapshot (closest to game)

        # Calculate movement
        line_change = closing_line - opening_line if pd.notna(opening_line) and pd.notna(closing_line) else 0
        volatility = group['line'].std() if len(group) > 1 else 0

        # Track snapshots by time window
        for _, row in group.iterrows():
            movements.append({
                'game_id': game_id,
                'bookmaker': bookmaker,
                'team': team,
                'hours_before': row['hours_before_game'],
                'line': row['line'],
                'opening_line': opening_line,
                'closing_line': closing_line,
                'line_movement': line_change,
                'volatility': volatility,
                'timestamp': row['last_update'],
                'commence_time': row['commence_time']
            })

    movements_df = pd.DataFrame(movements)
    logger.info(f"Calculated movement for {len(movements_df):,} snapshots")

    return movements_df


def analyze_timing_windows(movements: pd.DataFrame) -> dict:
    """
    Analyze line behavior in different timing windows.

    Windows:
    - 48-72hr: Very early (opening lines)
    - 24-48hr: Early
    - 12-24hr: Mid
    - 4-12hr: Late
    - 1-4hr: Very late
    - 0-1hr: Closing
    """
    logger.info("\nAnalyzing timing windows...")

    # Define windows
    windows = [
        ('48-72hr', 48, 72),
        ('24-48hr', 24, 48),
        ('12-24hr', 12, 24),
        ('4-12hr', 4, 12),
        ('1-4hr', 1, 4),
        ('0-1hr', 0, 1),
    ]

    results = {}

    for window_name, min_hrs, max_hrs in windows:
        window_data = movements[
            (movements['hours_before'] >= min_hrs) &
            (movements['hours_before'] < max_hrs)
        ]

        if len(window_data) == 0:
            continue

        # Calculate stats
        stats = {
            'window': window_name,
            'n_snapshots': len(window_data),
            'n_games': window_data['game_id'].nunique(),
            'avg_line_movement': window_data.groupby('game_id')['line_movement'].first().abs().mean(),
            'avg_volatility': window_data.groupby('game_id')['volatility'].first().mean(),
            'pct_lines_moved': (window_data.groupby('game_id')['line_movement'].first().abs() > 0.5).mean() * 100,
        }

        results[window_name] = stats

        logger.info(f"\n{window_name}:")
        logger.info(f"  Games: {stats['n_games']:,}")
        logger.info(f"  Avg movement: {stats['avg_line_movement']:.3f} points")
        logger.info(f"  Avg volatility: {stats['avg_volatility']:.3f}")
        logger.info(f"  Lines moved >0.5: {stats['pct_lines_moved']:.1f}%")

    return results


def analyze_hour_of_day_patterns(movements: pd.DataFrame) -> pd.DataFrame:
    """Analyze line behavior by hour of day."""
    logger.info("\nAnalyzing hour-of-day patterns...")

    movements['hour'] = movements['timestamp'].dt.hour

    hour_stats = movements.groupby('hour').agg({
        'line': lambda x: x.diff().abs().mean(),  # Avg line change per hour
        'game_id': 'count'
    }).reset_index()

    hour_stats.columns = ['hour', 'avg_line_change_per_update', 'n_updates']

    # Find volatile vs stable hours
    volatile_hours = hour_stats.nlargest(5, 'avg_line_change_per_update')
    stable_hours = hour_stats.nsmallest(5, 'avg_line_change_per_update')

    logger.info("\nMost volatile hours (high line movement):")
    for _, row in volatile_hours.iterrows():
        logger.info(f"  {int(row['hour']):02d}:00 - Avg change: {row['avg_line_change_per_update']:.4f}")

    logger.info("\nMost stable hours (low line movement):")
    for _, row in stable_hours.iterrows():
        logger.info(f"  {int(row['hour']):02d}:00 - Avg change: {row['avg_line_change_per_update']:.4f}")

    return hour_stats


def generate_recommendations(window_stats: dict, hour_stats: pd.DataFrame):
    """Generate actionable betting timing recommendations."""
    logger.info("\n" + "=" * 80)
    logger.info("BETTING TIMING RECOMMENDATIONS")
    logger.info("=" * 80)

    # Find best timing window (lowest movement = most stable prices)
    if window_stats:
        best_window = min(window_stats.items(), key=lambda x: x[1]['avg_line_movement'])
        worst_window = max(window_stats.items(), key=lambda x: x[1]['avg_line_movement'])

        logger.info(f"\n📊 Best Timing Window: {best_window[0]}")
        logger.info(f"   - Lowest average movement: {best_window[1]['avg_line_movement']:.3f} points")
        logger.info(f"   - Most stable prices")
        logger.info(f"   - Only {best_window[1]['pct_lines_moved']:.1f}% of lines move >0.5 points")

        logger.info(f"\n⚠️  Worst Timing Window: {worst_window[0]}")
        logger.info(f"   - Highest average movement: {worst_window[1]['avg_line_movement']:.3f} points")
        logger.info(f"   - Most volatile prices")
        logger.info(f"   - {worst_window[1]['pct_lines_moved']:.1f}% of lines move >0.5 points")

    # Find best hours
    if not hour_stats.empty:
        best_hours = hour_stats.nsmallest(3, 'avg_line_change_per_update')

        logger.info("\n⏰ Best Hours to Bet (most stable):")
        for _, row in best_hours.iterrows():
            logger.info(f"   {int(row['hour']):02d}:00 - Low volatility")

    logger.info("\n💡 STRATEGIC RECOMMENDATIONS:")
    logger.info("\n1. TIMING STRATEGY:")
    if 'avg_line_movement' in best_window[1]:
        if '24-48hr' in best_window[0]:
            logger.info("   ✅ Bet 24-48 hours before games for most stable prices")
        elif '12-24hr' in best_window[0]:
            logger.info("   ✅ Bet 12-24 hours before games for balance of info and stability")
        elif '4-12hr' in best_window[0]:
            logger.info("   ✅ Bet 4-12 hours before games to incorporate late info")
        else:
            logger.info(f"   ✅ {best_window[0]} window shows best price stability")

    logger.info("\n2. AVOID:")
    logger.info("   ❌ Betting right after lines open (high volatility)")
    logger.info("   ❌ Betting in final hour (closing line already efficient)")
    logger.info("   ❌ Betting during peak hours when sharp action moves lines")

    logger.info("\n3. IMPLEMENTATION:")
    logger.info("   • Set up automated bet placement during optimal windows")
    logger.info("   • Monitor line movement velocity before betting")
    logger.info("   • If line moving quickly, wait for stability")
    logger.info("   • Compare your bet time to optimal window for CLV analysis")

    logger.info("\n" + "=" * 80)


def main():
    """Run line movement timing analysis."""
    logger.info("=" * 80)
    logger.info("LINE MOVEMENT TIMING ANALYSIS")
    logger.info("=" * 80)

    # Load data
    odds = load_historical_odds(days_back=180)  # Last 6 months

    if odds.empty:
        logger.error("No data loaded")
        return 1

    # Calculate movements
    movements = calculate_line_movement(odds)

    if movements.empty:
        logger.error("No movement data calculated")
        return 1

    # Analyze timing windows
    window_stats = analyze_timing_windows(movements)

    # Analyze hour patterns
    hour_stats = analyze_hour_of_day_patterns(movements)

    # Generate recommendations
    generate_recommendations(window_stats, hour_stats)

    logger.info("\n✅ Analysis complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
