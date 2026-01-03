#!/usr/bin/env python3
"""
Validate and Finalize Closing Lines

Post-game script that selects the best closing line source for each bet.
Uses a priority system: snapshot > API provisional > opening line > NULL.

Run daily at 6:15 AM via cron (after populate_clv_data.py):
    15 6 * * * cd /path/to/nbamodels && python scripts/validate_closing_lines.py >> logs/validate_closing.log 2>&1

For manual execution:
    python scripts/validate_closing_lines.py

Usage:
    # Validate last 48 hours (default)
    python scripts/validate_closing_lines.py

    # Validate last 7 days
    python scripts/validate_closing_lines.py --days 7

    # Force revalidate all bets
    python scripts/validate_closing_lines.py --force
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from loguru import logger
import pandas as pd

from src.bet_tracker import DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_settled_bets_needing_validation(
    lookback_days: int = 2,
    force_revalidate: bool = False
) -> List[Dict]:
    """
    Get settled bets that need closing line validation.

    Args:
        lookback_days: How many days back to look for settled bets
        force_revalidate: If True, revalidate all bets

    Returns:
        List of bet records
    """
    conn = _get_connection()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    if force_revalidate:
        # Revalidate all settled bets in time window
        query = """
            SELECT *
            FROM bets
            WHERE outcome IS NOT NULL
            AND settled_at >= ?
            ORDER BY settled_at DESC
        """
        params = (cutoff.isoformat(),)
    else:
        # Only validate bets without closing_line_source
        query = """
            SELECT *
            FROM bets
            WHERE outcome IS NOT NULL
            AND settled_at >= ?
            AND closing_line_source IS NULL
            ORDER BY settled_at DESC
        """
        params = (cutoff.isoformat(),)

    bets = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(bet) for bet in bets]


def get_snapshot_closing_line(
    game_id: str,
    bet_type: str,
    bet_side: str,
    bookmaker: Optional[str],
    commence_time: str
) -> Optional[Tuple[float, Optional[float]]]:
    """
    Get closing line from snapshots (last snapshot before game).

    Args:
        game_id: Game ID
        bet_type: Bet type
        bet_side: Bet side
        bookmaker: Preferred bookmaker (optional)
        commence_time: Game commence time

    Returns:
        Tuple of (odds, line) or None if not found
    """
    conn = _get_connection()

    # Query for last snapshot before game
    query = """
        SELECT odds, line, bookmaker
        FROM line_snapshots
        WHERE game_id = ?
        AND bet_type = ?
        AND side = ?
        AND snapshot_time < ?
        ORDER BY snapshot_time DESC
        LIMIT 10
    """

    snapshots = conn.execute(
        query,
        (game_id, bet_type, bet_side, commence_time)
    ).fetchall()

    conn.close()

    if not snapshots:
        return None

    # Prefer specific bookmaker if specified
    if bookmaker:
        for snap in snapshots:
            if snap['bookmaker'] == bookmaker:
                return (float(snap['odds']), float(snap['line']) if snap['line'] is not None else None)

    # Fall back to first (most recent) snapshot
    snap = snapshots[0]
    return (float(snap['odds']), float(snap['line']) if snap['line'] is not None else None)


def get_opening_line(
    game_id: str,
    bet_type: str,
    bet_side: str,
    bookmaker: Optional[str]
) -> Optional[Tuple[float, Optional[float]]]:
    """
    Get opening line as fallback.

    Args:
        game_id: Game ID
        bet_type: Bet type
        bet_side: Bet side
        bookmaker: Preferred bookmaker (optional)

    Returns:
        Tuple of (odds, line) or None if not found
    """
    conn = _get_connection()

    if bookmaker:
        # Try specific bookmaker first
        query = """
            SELECT opening_odds, opening_line
            FROM opening_lines
            WHERE game_id = ?
            AND bet_type = ?
            AND side = ?
            AND bookmaker = ?
            LIMIT 1
        """
        result = conn.execute(query, (game_id, bet_type, bet_side, bookmaker)).fetchone()

        if result and result['opening_odds'] is not None:
            conn.close()
            return (float(result['opening_odds']), float(result['opening_line']) if result['opening_line'] is not None else None)

    # Fall back to any bookmaker
    query = """
        SELECT opening_odds, opening_line
        FROM opening_lines
        WHERE game_id = ?
        AND bet_type = ?
        AND side = ?
        AND opening_odds IS NOT NULL
        ORDER BY is_true_opener DESC
        LIMIT 1
    """

    result = conn.execute(query, (game_id, bet_type, bet_side)).fetchone()
    conn.close()

    if result and result['opening_odds'] is not None:
        return (float(result['opening_odds']), float(result['opening_line']) if result['opening_line'] is not None else None)

    return None


def validate_bet_closing_line(bet: Dict) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Validate and select best closing line source for a bet.

    Priority:
    1. Snapshot (most accurate)
    2. API provisional (captured 30 min before game)
    3. Opening line (fallback)
    4. NULL (failure)

    Args:
        bet: Bet record dict

    Returns:
        Tuple of (source, closing_odds, closing_line)
        source is one of: 'snapshot', 'api', 'opening', 'none'
    """
    game_id = bet['game_id']
    bet_type = bet['bet_type']
    bet_side = bet['bet_side']
    bookmaker = bet.get('bookmaker')
    commence_time = bet.get('commence_time')

    # Priority 1: Snapshot
    if commence_time:
        snapshot_line = get_snapshot_closing_line(
            game_id, bet_type, bet_side, bookmaker, commence_time
        )

        if snapshot_line:
            logger.debug(f"Bet {bet['id']}: Using snapshot closing line")
            return ('snapshot', snapshot_line[0], snapshot_line[1])

    # Priority 2: API provisional
    if bet.get('provisional_closing_odds') is not None:
        logger.debug(f"Bet {bet['id']}: Using API provisional closing line")
        return (
            'api',
            float(bet['provisional_closing_odds']),
            float(bet['provisional_closing_line']) if bet.get('provisional_closing_line') is not None else None
        )

    # Priority 3: Opening line (fallback)
    opening_line = get_opening_line(game_id, bet_type, bet_side, bookmaker)

    if opening_line:
        logger.warning(f"Bet {bet['id']}: Falling back to opening line (no snapshots or provisional)")
        return ('opening', opening_line[0], opening_line[1])

    # Priority 4: NULL (failure)
    logger.error(f"Bet {bet['id']}: No closing line source available")
    return ('none', None, None)


def update_bet_closing_line(
    bet_id: str,
    source: str,
    closing_odds: Optional[float],
    closing_line: Optional[float]
) -> bool:
    """
    Update bet record with validated closing line.

    Args:
        bet_id: Bet ID to update
        source: Closing line source
        closing_odds: Validated closing odds
        closing_line: Validated closing line

    Returns:
        True if successful
    """
    conn = _get_connection()

    try:
        # Update closing line and source
        conn.execute("""
            UPDATE bets
            SET closing_odds = ?,
                closing_line = ?,
                closing_line_source = ?
            WHERE id = ?
        """, (closing_odds, closing_line, source, bet_id))

        # If we have closing odds, also update CLV
        if closing_odds is not None:
            bet = conn.execute("SELECT odds FROM bets WHERE id = ?", (bet_id,)).fetchone()
            if bet and bet['odds'] is not None:
                booked_odds = float(bet['odds'])

                # Calculate CLV (positive = better odds than close)
                if booked_odds > 0 and closing_odds > 0:
                    # American odds
                    booked_implied = 100 / (booked_odds + 100) if booked_odds > 0 else -booked_odds / (-booked_odds + 100)
                    closing_implied = 100 / (closing_odds + 100) if closing_odds > 0 else -closing_odds / (-closing_odds + 100)
                    clv = closing_implied - booked_implied
                else:
                    clv = None

                conn.execute("""
                    UPDATE bets
                    SET clv = ?,
                        clv_updated_at = ?
                    WHERE id = ?
                """, (clv, datetime.now(timezone.utc).isoformat(), bet_id))

        conn.commit()
        return True

    except Exception as e:
        logger.error(f"Error updating closing line for bet {bet_id}: {e}")
        return False

    finally:
        conn.close()


def generate_validation_stats(results: Dict[str, int]) -> None:
    """
    Generate and log validation statistics.

    Args:
        results: Dict with counts by source
    """
    total = sum(results.values())

    logger.info("\n=== Closing Line Validation Summary ===")
    logger.info(f"Total bets validated: {total}")

    if total > 0:
        for source in ['snapshot', 'api', 'opening', 'none']:
            count = results.get(source, 0)
            pct = (count / total) * 100
            logger.info(f"  {source.capitalize()}: {count} ({pct:.1f}%)")

        # Quality metrics
        quality_count = results.get('snapshot', 0) + results.get('api', 0)
        quality_pct = (quality_count / total) * 100
        logger.info(f"\nHigh-quality closing lines (snapshot + API): {quality_count} ({quality_pct:.1f}%)")

        failure_count = results.get('none', 0)
        if failure_count > 0:
            failure_pct = (failure_count / total) * 100
            logger.warning(f"Failed to find closing line: {failure_count} ({failure_pct:.1f}%)")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Validate and finalize closing lines for settled bets')
    parser.add_argument('--days', type=int, default=2, help='Lookback days (default: 2)')
    parser.add_argument('--force', action='store_true', help='Revalidate all bets')
    args = parser.parse_args()

    logger.info(f"=== Closing Line Validation Started (lookback: {args.days} days, force: {args.force}) ===")

    # Get bets needing validation
    bets = get_settled_bets_needing_validation(
        lookback_days=args.days,
        force_revalidate=args.force
    )

    if not bets:
        logger.info("No bets need closing line validation")
        return 0

    logger.info(f"Found {len(bets)} bets needing validation")

    # Track results by source
    results = {
        'snapshot': 0,
        'api': 0,
        'opening': 0,
        'none': 0
    }

    success_count = 0
    failed_count = 0

    # Process each bet
    for bet in bets:
        try:
            source, closing_odds, closing_line = validate_bet_closing_line(bet)

            # Update bet
            if update_bet_closing_line(bet['id'], source, closing_odds, closing_line):
                results[source] += 1
                success_count += 1

                logger.info(
                    f"Validated {bet['id']}: source={source}, "
                    f"odds={closing_odds if closing_odds else 'N/A'}, "
                    f"line={closing_line if closing_line else 'N/A'}"
                )
            else:
                failed_count += 1

        except Exception as e:
            logger.error(f"Error validating bet {bet['id']}: {e}")
            failed_count += 1

    # Generate summary stats
    generate_validation_stats(results)

    logger.info(f"\n=== Validation Complete: {success_count} succeeded, {failed_count} failed ===")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
