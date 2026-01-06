#!/usr/bin/env python3
"""
Backfill Historical Odds

Fetch historical odds from The Odds API and update existing bets with correct market_prob.

This fixes the issue where market_prob was hardcoded to 0.5 instead of calculated from odds.

Usage:
    python scripts/backfill_historical_odds.py --dry-run  # Preview changes
    python scripts/backfill_historical_odds.py            # Update database
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
from loguru import logger

from src.data.odds_api import OddsAPIClient
from src.bet_tracker import american_to_implied_prob


# Team name mapping (abbreviation to full name)
TEAM_NAME_MAP = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def get_bets_needing_update() -> pd.DataFrame:
    """Get all bets that need market_prob updates."""
    conn = sqlite3.connect("data/bets/bets.db")

    query = """
    SELECT
        id,
        game_id,
        home_team,
        away_team,
        commence_time,
        bet_type,
        bet_side,
        odds,
        line,
        model_prob,
        market_prob,
        edge,
        outcome
    FROM bets
    WHERE market_prob IS NOT NULL
    ORDER BY commence_time
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert commence_time to datetime
    df['commence_time'] = pd.to_datetime(df['commence_time'], format='ISO8601')
    df['game_date'] = df['commence_time'].dt.date

    return df


def fetch_historical_odds_for_date(client: OddsAPIClient, date_str: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical odds for a specific date (with caching).

    Args:
        client: OddsAPIClient instance
        date_str: Date in YYYY-MM-DD format
        use_cache: If True, check cache before API call

    Returns:
        DataFrame with historical odds
    """
    # Check cache first
    cache_dir = Path("data/historical_odds")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"odds_{date_str}.parquet"

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached odds for {date_str}...")
        try:
            odds_df = pd.read_parquet(cache_file)
            logger.success(f"  ✓ Loaded {len(odds_df)} odds records from cache")
            return odds_df
        except Exception as e:
            logger.warning(f"  ⚠️  Cache read failed: {e}, fetching from API")

    # Fetch from API
    try:
        logger.info(f"Fetching historical odds for {date_str} from API...")
        odds_df = client.get_historical_odds(date_str, markets=['h2h', 'spreads', 'totals'])

        if not odds_df.empty:
            logger.success(f"  ✓ Fetched {len(odds_df)} odds records for {date_str}")

            # Save to cache
            try:
                odds_df.to_parquet(cache_file)
                logger.debug(f"  💾 Cached to {cache_file}")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to cache: {e}")
        else:
            logger.warning(f"  ⚠️  No odds found for {date_str}")

        return odds_df

    except Exception as e:
        logger.error(f"  ✗ Error fetching odds for {date_str}: {e}")
        return pd.DataFrame()


def match_bet_to_odds(bet: pd.Series, historical_odds: pd.DataFrame) -> Dict:
    """
    Match a bet to historical odds and calculate correct market_prob.

    Args:
        bet: Bet record from database
        historical_odds: DataFrame with historical odds

    Returns:
        Dict with updated values or None if no match found
    """
    # Convert team abbreviations to full names for matching
    home_full = TEAM_NAME_MAP.get(bet['home_team'], bet['home_team'])
    away_full = TEAM_NAME_MAP.get(bet['away_team'], bet['away_team'])

    # Filter odds for this game - try both home/away combinations
    # (bet data may have teams reversed)
    game_odds = historical_odds[
        ((historical_odds['home_team'] == home_full) & (historical_odds['away_team'] == away_full)) |
        ((historical_odds['home_team'] == away_full) & (historical_odds['away_team'] == home_full))
    ]

    if game_odds.empty:
        logger.debug(f"No odds found for {away_full} @ {home_full} or {home_full} @ {away_full}")
        return None

    # Determine if teams are reversed in the odds data
    first_game = game_odds.iloc[0]
    teams_reversed = (first_game['home_team'] == away_full)

    # Match based on bet type
    if bet['bet_type'] == 'spread':
        # Find spread odds for the correct side
        spread_odds = game_odds[game_odds['market'] == 'spread']

        # Adjust bet side if teams are reversed
        if teams_reversed:
            actual_side = 'away' if bet['bet_side'] == 'home' else 'home'
        else:
            actual_side = bet['bet_side']

        side_odds = spread_odds[spread_odds['team'] == actual_side]

        if not side_odds.empty:
            # Use average odds across bookmakers
            avg_odds = side_odds['odds'].mean()
            avg_line = side_odds['line'].mean()

            return {
                'odds': avg_odds,
                'line': avg_line,
                'market_prob': american_to_implied_prob(avg_odds),
                'edge': bet['model_prob'] - american_to_implied_prob(avg_odds)
            }

    elif bet['bet_type'] == 'moneyline':
        # Find moneyline odds
        ml_odds = game_odds[game_odds['market'] == 'moneyline']

        # Adjust bet side if teams are reversed
        if teams_reversed:
            actual_side = 'away' if bet['bet_side'] == 'home' else 'home'
        else:
            actual_side = bet['bet_side']

        side_odds = ml_odds[ml_odds['team'] == actual_side]

        if not side_odds.empty:
            avg_odds = side_odds['odds'].mean()

            return {
                'odds': avg_odds,
                'line': None,
                'market_prob': american_to_implied_prob(avg_odds),
                'edge': bet['model_prob'] - american_to_implied_prob(avg_odds)
            }

    elif bet['bet_type'] == 'totals':
        # Find totals odds
        total_odds = game_odds[game_odds['market'] == 'total']

        side_odds = total_odds[total_odds['team'] == bet['bet_side']]

        if not side_odds.empty:
            avg_odds = side_odds['odds'].mean()
            avg_line = side_odds['line'].mean()

            return {
                'odds': avg_odds,
                'line': avg_line,
                'market_prob': american_to_implied_prob(avg_odds),
                'edge': bet['model_prob'] - american_to_implied_prob(avg_odds)
            }

    return None


def update_bet_in_database(bet_id: str, updates: Dict, dry_run: bool = False) -> bool:
    """
    Update a bet record with corrected values.

    Args:
        bet_id: Bet ID
        updates: Dict with fields to update
        dry_run: If True, don't actually update

    Returns:
        True if successful
    """
    if dry_run:
        logger.info(f"  [DRY RUN] Would update bet {bet_id}:")
        for key, value in updates.items():
            if key == 'market_prob' or key == 'edge':
                logger.info(f"    {key}: {value:.4f} ({value*100:.2f}%)")
            elif key == 'odds':
                logger.info(f"    {key}: {value:+.0f}")
            else:
                logger.info(f"    {key}: {value}")
        return True

    conn = sqlite3.connect("data/bets/bets.db")

    try:
        # Build UPDATE query
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        query = f"UPDATE bets SET {set_clause} WHERE id = ?"

        values = list(updates.values()) + [bet_id]

        conn.execute(query, values)
        conn.commit()

        logger.debug(f"  ✓ Updated bet {bet_id}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Error updating bet {bet_id}: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Backfill historical odds for existing bets')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without updating database')
    parser.add_argument('--limit', type=int, help='Limit number of dates to process (for testing)')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache and fetch fresh data from API')
    args = parser.parse_args()

    use_cache = not args.no_cache

    logger.info("=" * 80)
    logger.info("🔄 BACKFILL HISTORICAL ODDS")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("⚠️  DRY RUN MODE - No database updates will be made")

    # Initialize Odds API client
    client = OddsAPIClient()

    # Get all bets needing updates
    logger.info("\n1️⃣  Loading bets from database...")
    bets_df = get_bets_needing_update()
    logger.info(f"   ✓ Found {len(bets_df)} bets")

    if bets_df.empty:
        logger.info("No bets to update - exiting")
        return 0

    # Get unique game dates
    unique_dates = sorted(bets_df['game_date'].unique())
    logger.info(f"   📅 Date range: {unique_dates[0]} to {unique_dates[-1]}")
    logger.info(f"   📅 Total dates: {len(unique_dates)}")

    if args.limit:
        unique_dates = unique_dates[:args.limit]
        logger.info(f"   ⚠️  Limited to first {args.limit} dates")

    # Fetch historical odds for each date
    logger.info(f"\n2️⃣  Fetching historical odds from Odds API...")
    logger.info(f"   ⚠️  Note: Historical odds require a paid API plan")

    all_historical_odds = {}
    failed_dates = []

    for i, date in enumerate(unique_dates, 1):
        date_str = str(date)
        logger.info(f"\n   [{i}/{len(unique_dates)}] {date_str}")

        odds_df = fetch_historical_odds_for_date(client, date_str, use_cache=use_cache)

        if not odds_df.empty:
            all_historical_odds[date_str] = odds_df
        else:
            failed_dates.append(date_str)

    if not all_historical_odds:
        logger.error("\n❌ No historical odds retrieved. Check:")
        logger.error("   1. Your Odds API key is valid")
        logger.error("   2. You have a paid plan with historical odds access")
        logger.error("   3. The dates are correct")
        return 1

    logger.success(f"\n   ✓ Successfully fetched odds for {len(all_historical_odds)} dates")

    if failed_dates:
        logger.warning(f"   ⚠️  Failed to fetch {len(failed_dates)} dates")

    # Match bets to historical odds and update
    logger.info(f"\n3️⃣  Matching bets to historical odds...")

    updated_count = 0
    no_match_count = 0
    error_count = 0

    for _, bet in bets_df.iterrows():
        date_str = str(bet['game_date'])

        if date_str not in all_historical_odds:
            logger.debug(f"No historical odds for {date_str}")
            no_match_count += 1
            continue

        historical_odds = all_historical_odds[date_str]

        # Match bet to odds
        updates = match_bet_to_odds(bet, historical_odds)

        if updates:
            # Update database
            success = update_bet_in_database(bet['id'], updates, dry_run=args.dry_run)

            if success:
                updated_count += 1
            else:
                error_count += 1
        else:
            no_match_count += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ BACKFILL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Total bets: {len(bets_df)}")
    logger.info(f"   ✓ Updated: {updated_count}")
    logger.info(f"   ⚠️  No match: {no_match_count}")
    logger.info(f"   ✗ Errors: {error_count}")

    if args.dry_run:
        logger.info(f"\n💡 This was a dry run. Run without --dry-run to actually update the database.")
    else:
        logger.info(f"\n💾 Database updated successfully!")
        logger.info(f"\n📈 Check updated data:")
        logger.info(f"   python -m streamlit run analytics_dashboard.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
