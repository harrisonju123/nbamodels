#!/usr/bin/env python3
"""
Closing Line Capture Script

Captures provisional closing lines 30-60 minutes before game start.
This provides a backup closing line in case post-game snapshots fail.

Run every 15 minutes via cron:
    */15 * * * * cd /path/to/nbamodels && python scripts/capture_closing_lines.py >> logs/closing_lines.log 2>&1

For manual execution:
    python scripts/capture_closing_lines.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import pandas as pd
from loguru import logger

from src.data.odds_api import OddsAPIClient
from src.bet_tracker import DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_pending_bets_near_gametime(
    min_minutes: int = 30,
    max_minutes: int = 60
) -> List[Dict]:
    """
    Get pending bets for games starting in 30-60 minutes.

    Args:
        min_minutes: Minimum minutes before game start
        max_minutes: Maximum minutes before game start

    Returns:
        List of bet records that need closing lines
    """
    conn = _get_connection()
    now = datetime.now(timezone.utc)
    min_time = now + timedelta(minutes=min_minutes)
    max_time = now + timedelta(minutes=max_minutes)

    query = """
        SELECT
            id, game_id, bet_type, bet_side, line, odds, bookmaker, commence_time
        FROM bets
        WHERE outcome IS NULL
        AND provisional_closing_odds IS NULL
        AND commence_time >= ?
        AND commence_time <= ?
    """

    bets = conn.execute(query, (min_time.isoformat(), max_time.isoformat())).fetchall()
    conn.close()

    return [dict(bet) for bet in bets]


def fetch_current_odds_for_games(game_ids: List[str]) -> pd.DataFrame:
    """
    Fetch current odds for specific games.

    Args:
        game_ids: List of game IDs to fetch odds for

    Returns:
        DataFrame with current odds
    """
    client = OddsAPIClient()

    try:
        # Fetch all current odds
        odds_df = client.get_current_odds(markets=["h2h", "spreads", "totals"])

        if odds_df.empty:
            return pd.DataFrame()

        # Filter to only games we care about
        filtered = odds_df[odds_df['game_id'].isin(game_ids)]

        return filtered

    except Exception as e:
        logger.error(f"Error fetching current odds: {e}")
        return pd.DataFrame()


def update_provisional_closing_line(
    bet_id: str,
    closing_odds: Optional[float],
    closing_line: Optional[float]
) -> bool:
    """
    Update bet record with provisional closing line.

    Args:
        bet_id: Bet ID to update
        closing_odds: Provisional closing odds
        closing_line: Provisional closing line (None for moneyline)

    Returns:
        True if successful
    """
    conn = _get_connection()

    try:
        conn.execute("""
            UPDATE bets
            SET provisional_closing_odds = ?,
                provisional_closing_line = ?
            WHERE id = ?
        """, (closing_odds, closing_line, bet_id))

        conn.commit()
        logger.debug(f"Updated provisional closing for bet {bet_id}: odds={closing_odds}, line={closing_line}")
        return True

    except Exception as e:
        logger.error(f"Error updating bet {bet_id}: {e}")
        return False

    finally:
        conn.close()


def match_bet_to_odds(bet: Dict, odds_df: pd.DataFrame) -> Optional[Dict]:
    """
    Find matching odds for a bet.

    Args:
        bet: Bet record
        odds_df: DataFrame with current odds

    Returns:
        Dict with matched odds, or None if not found
    """
    # Filter by game_id
    game_odds = odds_df[odds_df['game_id'] == bet['game_id']]

    if game_odds.empty:
        return None

    # Map bet_type to market
    market_mapping = {
        'moneyline': 'h2h',
        'spread': 'spreads',
        'totals': 'totals'
    }
    market = market_mapping.get(bet['bet_type'])

    if not market:
        return None

    # Filter by market
    market_odds = game_odds[game_odds['market'] == market]

    if market_odds.empty:
        return None

    # Filter by bookmaker if specified
    if bet.get('bookmaker'):
        bookmaker_odds = market_odds[market_odds['bookmaker'] == bet['bookmaker']]
        if not bookmaker_odds.empty:
            market_odds = bookmaker_odds
        else:
            # Fall back to any bookmaker if preferred not available
            logger.debug(f"Bookmaker {bet['bookmaker']} not available, using alternative")

    # Extract odds based on bet type and side
    if bet['bet_type'] == 'moneyline':
        if bet['bet_side'] == 'home':
            match = market_odds[market_odds['home_odds'].notna()]
            if not match.empty:
                return {
                    'odds': float(match.iloc[0]['home_odds']),
                    'line': None
                }
        elif bet['bet_side'] == 'away':
            match = market_odds[market_odds['away_odds'].notna()]
            if not match.empty:
                return {
                    'odds': float(match.iloc[0]['away_odds']),
                    'line': None
                }

    elif bet['bet_type'] == 'spread':
        # Match by team side
        side_odds = market_odds[market_odds['team'] == bet['bet_side']]
        if not side_odds.empty:
            row = side_odds.iloc[0]
            if pd.notna(row.get('odds')) and pd.notna(row.get('line')):
                return {
                    'odds': float(row['odds']),
                    'line': float(row['line'])
                }

    elif bet['bet_type'] == 'totals':
        if bet['bet_side'] == 'over':
            match = market_odds[market_odds['over_odds'].notna()]
            if not match.empty:
                row = match.iloc[0]
                return {
                    'odds': float(row['over_odds']),
                    'line': float(row['line']) if pd.notna(row.get('line')) else None
                }
        elif bet['bet_side'] == 'under':
            match = market_odds[market_odds['under_odds'].notna()]
            if not match.empty:
                row = match.iloc[0]
                return {
                    'odds': float(row['under_odds']),
                    'line': float(row['line']) if pd.notna(row.get('line')) else None
                }

    return None


def main():
    """Main execution function."""
    logger.info("=== Closing Line Capture Started ===")

    # Get pending bets near game time
    pending_bets = get_pending_bets_near_gametime(min_minutes=30, max_minutes=60)

    if not pending_bets:
        logger.info("No pending bets near game time (30-60 min window)")
        return 0

    logger.info(f"Found {len(pending_bets)} pending bets near game time")

    # Get unique game IDs
    game_ids = list(set([bet['game_id'] for bet in pending_bets]))
    logger.info(f"Fetching odds for {len(game_ids)} games")

    # Fetch current odds
    odds_df = fetch_current_odds_for_games(game_ids)

    if odds_df.empty:
        logger.warning("No odds data available from API")
        return 1

    # Process each bet
    success_count = 0
    failed_count = 0

    for bet in pending_bets:
        # Match bet to current odds
        matched_odds = match_bet_to_odds(bet, odds_df)

        if matched_odds:
            # Update provisional closing line
            if update_provisional_closing_line(
                bet['id'],
                matched_odds['odds'],
                matched_odds['line']
            ):
                success_count += 1
                logger.info(
                    f"Captured closing line for {bet['id']}: "
                    f"odds={matched_odds['odds']}, line={matched_odds['line']}"
                )
            else:
                failed_count += 1
        else:
            logger.warning(f"No matching odds found for bet {bet['id']}")
            failed_count += 1

    logger.info(f"=== Closing Line Capture Complete: {success_count} succeeded, {failed_count} failed ===")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
