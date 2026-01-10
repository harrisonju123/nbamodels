#!/usr/bin/env python3
"""
Collect Historical Odds for Backtesting

Fetches historical NBA game odds for 2023-2024 seasons to enable proper model backtesting.

Usage:
    python scripts/collect_historical_odds_for_backtest.py
    python scripts/collect_historical_odds_for_backtest.py --season 2023
    python scripts/collect_historical_odds_for_backtest.py --dry-run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from time import sleep

from src.data.odds_api import OddsAPIClient


# NBA season date ranges (regular season only)
SEASON_DATES = {
    2023: {
        'start': '2022-10-18',  # 2022-23 season start
        'end': '2023-04-09'     # 2022-23 regular season end
    },
    2024: {
        'start': '2023-10-24',  # 2023-24 season start
        'end': '2024-04-14'     # 2023-24 regular season end
    }
}


def get_game_dates_from_season(season: int) -> list:
    """
    Get all game dates for a season.

    Args:
        season: Season year (2023 or 2024)

    Returns:
        List of date strings in YYYY-MM-DD format
    """
    if season not in SEASON_DATES:
        raise ValueError(f"Season {season} not configured. Available: {list(SEASON_DATES.keys())}")

    dates = SEASON_DATES[season]
    start = datetime.strptime(dates['start'], '%Y-%m-%d')
    end = datetime.strptime(dates['end'], '%Y-%m-%d')

    # Get all dates in range
    all_dates = []
    current = start
    while current <= end:
        all_dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    return all_dates


def fetch_odds_for_date(client: OddsAPIClient, date_str: str, dry_run: bool = False) -> pd.DataFrame:
    """
    Fetch and cache historical odds for a specific date.

    Args:
        client: OddsAPIClient instance
        date_str: Date in YYYY-MM-DD format
        dry_run: If True, skip API calls

    Returns:
        DataFrame with odds data (empty if cached or dry run)
    """
    cache_dir = Path("data/historical_odds")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"odds_{date_str}.parquet"

    # Check cache first
    if cache_file.exists():
        try:
            odds_df = pd.read_parquet(cache_file)
            logger.debug(f"  ✓ Cached ({len(odds_df)} records)")
            return pd.DataFrame()  # Return empty to indicate cached
        except Exception as e:
            logger.warning(f"  ⚠️  Cache corrupted, will re-fetch: {e}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would fetch odds from API")
        return pd.DataFrame()

    # Fetch from API
    try:
        logger.info(f"  Fetching from Odds API...")
        odds_df = client.get_historical_odds(date_str, markets=['spreads'])

        if not odds_df.empty:
            # Save to cache
            odds_df.to_parquet(cache_file)
            logger.success(f"  ✓ Fetched {len(odds_df)} records ({len(odds_df['game_id'].unique())} games)")
            return odds_df
        else:
            logger.warning(f"  ⚠️  No games found")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Collect historical odds for backtesting')
    parser.add_argument('--season', type=int, choices=[2023, 2024],
                       help='Season to collect (default: both)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be fetched without API calls')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API calls in seconds (default: 1.0)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("📊 COLLECT HISTORICAL ODDS FOR BACKTESTING")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("⚠️  DRY RUN MODE - No API calls will be made")

    # Determine seasons to process
    seasons = [args.season] if args.season else [2023, 2024]

    # Initialize API client
    client = OddsAPIClient()
    if not client.api_key:
        logger.error("❌ ODDS_API_KEY not found in environment")
        logger.error("   Set it in .env file or environment variables")
        return 1

    # Collect dates for each season
    all_dates = []
    for season in seasons:
        season_dates = get_game_dates_from_season(season)
        all_dates.extend(season_dates)
        logger.info(f"\n📅 Season {season}: {len(season_dates)} dates")
        logger.info(f"   {season_dates[0]} to {season_dates[-1]}")

    logger.info(f"\n📊 Total dates to process: {len(all_dates)}")

    # Check cache status
    cache_dir = Path("data/historical_odds")
    cached_count = 0
    if cache_dir.exists():
        cached_files = list(cache_dir.glob("odds_*.parquet"))
        cached_count = len(cached_files)
        logger.info(f"💾 Already cached: {cached_count} dates")

    # Process each date
    logger.info(f"\n{'='*80}")
    logger.info("🔄 FETCHING ODDS")
    logger.info("=" * 80)

    total_fetched = 0
    total_cached = 0
    total_failed = 0
    total_games = 0

    for i, date_str in enumerate(all_dates, 1):
        logger.info(f"\n[{i}/{len(all_dates)}] {date_str}")

        odds_df = fetch_odds_for_date(client, date_str, dry_run=args.dry_run)

        if odds_df.empty:
            # Either cached or failed
            cache_file = cache_dir / f"odds_{date_str}.parquet"
            if cache_file.exists():
                total_cached += 1
            else:
                total_failed += 1
        else:
            total_fetched += 1
            total_games += len(odds_df['game_id'].unique())

            # Rate limiting
            if i < len(all_dates) and not args.dry_run:
                sleep(args.delay)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ COLLECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Total dates: {len(all_dates)}")
    logger.info(f"   ✓ Newly fetched: {total_fetched}")
    logger.info(f"   💾 Already cached: {total_cached}")
    logger.info(f"   ✗ Failed/No games: {total_failed}")
    if total_games > 0:
        logger.info(f"   🏀 Total games: {total_games}")

    if args.dry_run:
        logger.info(f"\n💡 This was a dry run. Run without --dry-run to fetch from API.")
        logger.info(f"   ⚠️  Note: Historical odds require a PAID Odds API plan")
    else:
        logger.info(f"\n💾 Odds data saved to: data/historical_odds/")
        logger.info(f"\n📈 Next step:")
        logger.info(f"   python scripts/backtest_model_comparison.py --seasons 2023 2024")

    return 0


if __name__ == "__main__":
    sys.exit(main())
