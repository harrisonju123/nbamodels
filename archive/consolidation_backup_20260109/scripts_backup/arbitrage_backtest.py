#!/usr/bin/env python3
"""
Arbitrage Backtest - Scan Historical Odds for Arbitrage Opportunities

Tests whether arbitrage opportunities exist in historical NBA odds data
by scanning 11 bookmakers across 3 markets (moneyline, spread, total).

Usage:
    python scripts/arbitrage_backtest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from collections import defaultdict

from src.betting.strategies.arbitrage_strategy import ArbitrageStrategy


def load_all_historical_odds():
    """
    Load all parquet files from data/historical_odds/.

    Returns:
        DataFrame with all historical odds data
    """
    odds_dir = Path("data/historical_odds")

    if not odds_dir.exists():
        logger.error(f"Historical odds directory not found: {odds_dir}")
        return pd.DataFrame()

    logger.info(f"Loading historical odds from {odds_dir}")

    all_odds = []
    parquet_files = list(odds_dir.glob("odds_*.parquet"))

    if not parquet_files:
        logger.error("No parquet files found in historical_odds/")
        return pd.DataFrame()

    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            all_odds.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {parquet_file}: {e}")

    if not all_odds:
        logger.error("No valid odds data loaded")
        return pd.DataFrame()

    combined = pd.concat(all_odds, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} odds records from {len(parquet_files)} files")

    return combined


def analyze_arb_opportunities(arbs, odds_df):
    """
    Analyze arbitrage opportunities and generate detailed report.

    Args:
        arbs: List of arbitrage opportunities
        odds_df: Original odds DataFrame

    Returns:
        Dict with analysis results
    """
    if not arbs:
        return {
            'total_arbs': 0,
            'by_market': {},
            'by_bookmaker_pair': {},
            'profit_stats': {},
        }

    # Convert to DataFrame for easier analysis
    arbs_df = pd.DataFrame(arbs)

    # Count by market
    by_market = arbs_df.groupby('market').size().to_dict()

    # Count by bookmaker pairs
    arbs_df['bookmaker_pair'] = arbs_df.apply(
        lambda row: f"{row['bookmaker1']} / {row['bookmaker2']}",
        axis=1
    )
    by_bookmaker_pair = arbs_df['bookmaker_pair'].value_counts().head(10).to_dict()

    # Profit statistics
    profit_stats = {
        'min': arbs_df['profit_pct'].min(),
        'max': arbs_df['profit_pct'].max(),
        'mean': arbs_df['profit_pct'].mean(),
        'median': arbs_df['profit_pct'].median(),
        'std': arbs_df['profit_pct'].std(),
    }

    # Unique games
    unique_games = arbs_df['game_id'].nunique()
    total_games = odds_df['game_id'].nunique()

    return {
        'total_arbs': len(arbs),
        'unique_games': unique_games,
        'total_games': total_games,
        'games_with_arbs_pct': (unique_games / total_games * 100) if total_games > 0 else 0,
        'by_market': by_market,
        'by_bookmaker_pair': by_bookmaker_pair,
        'profit_stats': profit_stats,
        'arbs_df': arbs_df,
    }


def run_backtest():
    """Run comprehensive arbitrage backtest."""

    logger.info("=" * 80)
    logger.info("ARBITRAGE STRATEGY BACKTEST")
    logger.info("=" * 80)

    # Load historical odds data
    odds_df = load_all_historical_odds()

    if len(odds_df) == 0:
        logger.error("No odds data loaded - cannot run backtest")
        return

    # Display data summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total odds records: {len(odds_df):,}")
    logger.info(f"Unique games: {odds_df['game_id'].nunique()}")
    logger.info(f"Unique bookmakers: {odds_df['bookmaker'].nunique()}")
    logger.info(f"Bookmakers: {', '.join(sorted(odds_df['bookmaker'].unique()))}")
    logger.info(f"Markets: {', '.join(sorted(odds_df['market'].unique()))}")
    logger.info(f"Date range: {odds_df['commence_time'].min()} to {odds_df['commence_time'].max()}")

    # Test different profit thresholds
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.03]

    logger.info("\n" + "=" * 80)
    logger.info("SCANNING FOR ARBITRAGE OPPORTUNITIES")
    logger.info("=" * 80)

    results = {}

    for min_profit in thresholds:
        logger.info(f"\nTesting min_profit threshold: {min_profit:.1%}")

        # Initialize strategy
        strategy = ArbitrageStrategy(min_arb_profit=min_profit)

        # Scan all markets
        all_arbs = []
        for market in ['moneyline', 'spread', 'total']:
            arbs = strategy.find_arbitrage(odds_df, market)
            all_arbs.extend(arbs)

        # Analyze results
        analysis = analyze_arb_opportunities(all_arbs, odds_df)
        results[min_profit] = analysis

        logger.info(f"  Found {analysis['total_arbs']} arbitrage opportunities")
        logger.info(f"  Games with arbs: {analysis['unique_games']}/{analysis['total_games']} ({analysis['games_with_arbs_pct']:.1f}%)")

        if analysis['total_arbs'] > 0:
            logger.info(f"  Avg profit: {analysis['profit_stats']['mean']:.2%}")
            logger.info(f"  Profit range: {analysis['profit_stats']['min']:.2%} - {analysis['profit_stats']['max']:.2%}")

    # Detailed report for best threshold (1%)
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED ANALYSIS (1% THRESHOLD)")
    logger.info("=" * 80)

    best_analysis = results[0.01]

    if best_analysis['total_arbs'] == 0:
        logger.warning("NO ARBITRAGE OPPORTUNITIES FOUND AT 1% THRESHOLD")
        logger.warning("This suggests NBA markets are very efficient with these bookmakers.")
        logger.warning("Recommendation: SKIP arbitrage strategy, focus on other strategies.")
    else:
        logger.info(f"\n✅ {best_analysis['total_arbs']} arbitrage opportunities found")

        logger.info("\nBy Market:")
        for market, count in best_analysis['by_market'].items():
            pct = count / best_analysis['total_arbs'] * 100
            logger.info(f"  {market}: {count} ({pct:.1f}%)")

        logger.info("\nTop 10 Bookmaker Pairs:")
        for pair, count in list(best_analysis['by_bookmaker_pair'].items())[:10]:
            logger.info(f"  {pair}: {count} arbs")

        logger.info("\nProfit Statistics:")
        stats = best_analysis['profit_stats']
        logger.info(f"  Mean: {stats['mean']:.2%}")
        logger.info(f"  Median: {stats['median']:.2%}")
        logger.info(f"  Std Dev: {stats['std']:.2%}")
        logger.info(f"  Min: {stats['min']:.2%}")
        logger.info(f"  Max: {stats['max']:.2%}")

        logger.info("\nTop 10 Most Profitable Arbs:")
        arbs_df = best_analysis['arbs_df'].nlargest(10, 'profit_pct')
        for idx, arb in arbs_df.iterrows():
            logger.info(
                f"  {arb['market']}: {arb['profit_pct']:.2%} profit "
                f"({arb['bookmaker1']} / {arb['bookmaker2']})"
            )

    # Summary and recommendation
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY & RECOMMENDATION")
    logger.info("=" * 80)

    logger.info("\nArbitrage Opportunities by Threshold:")
    for threshold, analysis in results.items():
        logger.info(f"  {threshold:.1%}: {analysis['total_arbs']} arbs")

    # Decision criteria
    REQUIRED_MIN_ARBS = 10
    REQUIRED_AVG_PROFIT = 0.01  # 1%

    if best_analysis['total_arbs'] >= REQUIRED_MIN_ARBS:
        if best_analysis['profit_stats'].get('mean', 0) >= REQUIRED_AVG_PROFIT:
            logger.success("\n✅ RECOMMENDATION: ENABLE ARBITRAGE STRATEGY")
            logger.info(f"   - {best_analysis['total_arbs']} opportunities found (>{REQUIRED_MIN_ARBS} required)")
            logger.info(f"   - {best_analysis['profit_stats']['mean']:.2%} average profit (>{REQUIRED_AVG_PROFIT:.1%} required)")
            logger.info(f"   - Arbs in {best_analysis['unique_games']}/{best_analysis['total_games']} games ({best_analysis['games_with_arbs_pct']:.1f}%)")
            logger.info("\n   Next steps:")
            logger.info("   1. Enable arbitrage in config/multi_strategy_config.yaml")
            logger.info("   2. Set min_arb_profit: 0.01 (1%)")
            logger.info("   3. Allocate 25% of bankroll")
            logger.info("   4. Test with daily_multi_strategy_pipeline.py --dry-run")
        else:
            logger.warning("\n⚠️  MARGINAL: Found arbs but profit margins too low")
            logger.warning(f"   Average profit: {best_analysis['profit_stats']['mean']:.2%} (need >{REQUIRED_AVG_PROFIT:.1%})")
            logger.warning("   Consider increasing threshold or skipping arbitrage")
    else:
        logger.warning("\n❌ RECOMMENDATION: SKIP ARBITRAGE STRATEGY")
        logger.warning(f"   - Only {best_analysis['total_arbs']} opportunities found (<{REQUIRED_MIN_ARBS} required)")
        logger.warning("   - NBA markets appear too efficient for arbitrage")
        logger.warning("   - Focus on model-based strategies (spread, B2B, player props)")


if __name__ == "__main__":
    try:
        run_backtest()
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
