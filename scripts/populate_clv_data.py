#!/usr/bin/env python3
"""
Populate Multi-Snapshot CLV Data

Calculates CLV at multiple time windows (1hr, 4hr, 12hr, 24hr before game)
for settled bets. Runs daily to populate CLV data for recently settled bets.

Run daily at 6 AM via cron:
    0 6 * * * cd /path/to/nbamodels && python scripts/populate_clv_data.py >> logs/populate_clv.log 2>&1

For manual execution:
    python scripts/populate_clv_data.py

Usage:
    # Process last 48 hours (default)
    python scripts/populate_clv_data.py

    # Process last 7 days
    python scripts/populate_clv_data.py --days 7

    # Force reprocess all bets (even if CLV already calculated)
    python scripts/populate_clv_data.py --force
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from loguru import logger

from src.bet_tracker import (
    DB_PATH,
    calculate_multi_snapshot_clv,
    calculate_line_velocity
)


def _get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_settled_bets_needing_clv(
    lookback_days: int = 2,
    force_reprocess: bool = False
) -> List[Dict]:
    """
    Get settled bets that need CLV calculation.

    Args:
        lookback_days: How many days back to look for settled bets
        force_reprocess: If True, reprocess all bets even if CLV already calculated

    Returns:
        List of bet records
    """
    conn = _get_connection()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    if force_reprocess:
        # Reprocess all settled bets in time window
        query = """
            SELECT *
            FROM bets
            WHERE outcome IS NOT NULL
            AND settled_at >= ?
            AND closing_odds IS NOT NULL
            ORDER BY settled_at DESC
        """
        params = (cutoff.isoformat(),)
    else:
        # Only process bets without CLV data
        query = """
            SELECT *
            FROM bets
            WHERE outcome IS NOT NULL
            AND settled_at >= ?
            AND closing_odds IS NOT NULL
            AND clv_at_1hr IS NULL
            ORDER BY settled_at DESC
        """
        params = (cutoff.isoformat(),)

    bets = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(bet) for bet in bets]


def update_bet_clv_data(
    bet_id: str,
    clv_data: Dict[str, float],
    line_velocity: Optional[float],
    snapshot_coverage: float
) -> bool:
    """
    Update bet record with CLV calculations.

    Args:
        bet_id: Bet ID to update
        clv_data: Dict with clv_at_1hr, clv_at_4hr, etc.
        line_velocity: Line movement velocity
        snapshot_coverage: 0.0-1.0, percentage of windows with data

    Returns:
        True if successful
    """
    conn = _get_connection()

    try:
        # Extract CLV values
        clv_at_1hr = clv_data.get('clv_at_1hr')
        clv_at_4hr = clv_data.get('clv_at_4hr')
        clv_at_12hr = clv_data.get('clv_at_12hr')
        clv_at_24hr = clv_data.get('clv_at_24hr')

        # Calculate max CLV achieved
        clv_values = [v for v in [clv_at_1hr, clv_at_4hr, clv_at_12hr, clv_at_24hr] if v is not None]
        max_clv = max(clv_values) if clv_values else None

        # Update bet
        conn.execute("""
            UPDATE bets
            SET clv_at_1hr = ?,
                clv_at_4hr = ?,
                clv_at_12hr = ?,
                clv_at_24hr = ?,
                line_velocity = ?,
                max_clv_achieved = ?,
                snapshot_coverage = ?,
                clv_updated_at = ?
            WHERE id = ?
        """, (
            clv_at_1hr, clv_at_4hr, clv_at_12hr, clv_at_24hr,
            line_velocity, max_clv, snapshot_coverage,
            datetime.now(timezone.utc).isoformat(),
            bet_id
        ))

        conn.commit()
        return True

    except Exception as e:
        logger.error(f"Error updating CLV data for bet {bet_id}: {e}")
        return False

    finally:
        conn.close()


def calculate_snapshot_coverage(clv_data: Dict[str, float]) -> float:
    """
    Calculate percentage of time windows with snapshot data.

    Args:
        clv_data: Dict with clv_at_1hr, clv_at_4hr, etc.

    Returns:
        Coverage ratio (0.0-1.0)
    """
    windows = ['clv_at_1hr', 'clv_at_4hr', 'clv_at_12hr', 'clv_at_24hr']
    available = sum(1 for w in windows if clv_data.get(w) is not None)
    return available / len(windows)


def process_bet(bet: Dict) -> bool:
    """
    Process a single bet to calculate and store CLV data.

    Args:
        bet: Bet record dict

    Returns:
        True if successful
    """
    bet_id = bet['id']

    try:
        # Calculate multi-snapshot CLV
        clv_data = calculate_multi_snapshot_clv(
            bet_id=bet_id,
            snapshot_times=['1hr', '4hr', '12hr', '24hr']
        )

        if not clv_data:
            logger.warning(f"No CLV data calculated for bet {bet_id}")
            # Still update with NULL values to mark as processed
            clv_data = {}

        # Calculate line velocity
        line_velocity = None
        if bet['game_id'] and bet['bet_type'] and bet['bet_side']:
            line_velocity = calculate_line_velocity(
                game_id=bet['game_id'],
                bet_type=bet['bet_type'],
                bet_side=bet['bet_side'],
                window_hours=4
            )

        # Calculate snapshot coverage
        coverage = calculate_snapshot_coverage(clv_data)

        # Update bet record
        success = update_bet_clv_data(
            bet_id=bet_id,
            clv_data=clv_data,
            line_velocity=line_velocity,
            snapshot_coverage=coverage
        )

        if success:
            logger.info(
                f"Processed {bet_id}: "
                f"1hr={clv_data.get('clv_at_1hr'):.3f if clv_data.get('clv_at_1hr') else 'N/A'}, "
                f"4hr={clv_data.get('clv_at_4hr'):.3f if clv_data.get('clv_at_4hr') else 'N/A'}, "
                f"coverage={coverage:.1%}"
            )

        return success

    except Exception as e:
        logger.error(f"Error processing bet {bet_id}: {e}")
        return False


def generate_summary_stats(bets: List[Dict]) -> Dict:
    """
    Generate summary statistics for processed bets.

    Args:
        bets: List of processed bet records

    Returns:
        Summary stats dict
    """
    conn = _get_connection()

    # Get updated CLV stats
    stats = {}

    # Average CLV by time window
    clv_query = """
        SELECT
            AVG(clv_at_1hr) as avg_1hr,
            AVG(clv_at_4hr) as avg_4hr,
            AVG(clv_at_12hr) as avg_12hr,
            AVG(clv_at_24hr) as avg_24hr,
            AVG(snapshot_coverage) as avg_coverage,
            COUNT(CASE WHEN clv_at_1hr IS NOT NULL THEN 1 END) as count_1hr,
            COUNT(CASE WHEN clv_at_4hr IS NOT NULL THEN 1 END) as count_4hr,
            COUNT(CASE WHEN clv_at_12hr IS NOT NULL THEN 1 END) as count_12hr,
            COUNT(CASE WHEN clv_at_24hr IS NOT NULL THEN 1 END) as count_24hr
        FROM bets
        WHERE id IN ({})
    """.format(','.join(['?'] * len(bets)))

    bet_ids = [bet['id'] for bet in bets]
    result = conn.execute(clv_query, bet_ids).fetchone()

    stats['avg_clv'] = {
        '1hr': result['avg_1hr'],
        '4hr': result['avg_4hr'],
        '12hr': result['avg_12hr'],
        '24hr': result['avg_24hr']
    }

    stats['coverage_counts'] = {
        '1hr': result['count_1hr'],
        '4hr': result['count_4hr'],
        '12hr': result['count_12hr'],
        '24hr': result['count_24hr']
    }

    stats['avg_snapshot_coverage'] = result['avg_coverage']

    conn.close()

    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Populate multi-snapshot CLV data for settled bets')
    parser.add_argument('--days', type=int, default=2, help='Lookback days (default: 2)')
    parser.add_argument('--force', action='store_true', help='Reprocess all bets even if CLV exists')
    args = parser.parse_args()

    logger.info(f"=== CLV Population Started (lookback: {args.days} days, force: {args.force}) ===")

    # Get bets needing CLV calculation
    bets = get_settled_bets_needing_clv(
        lookback_days=args.days,
        force_reprocess=args.force
    )

    if not bets:
        logger.info("No bets need CLV calculation")
        return 0

    logger.info(f"Found {len(bets)} bets needing CLV calculation")

    # Process each bet
    success_count = 0
    failed_count = 0

    for bet in bets:
        if process_bet(bet):
            success_count += 1
        else:
            failed_count += 1

    # Generate summary stats
    if success_count > 0:
        logger.info("\n=== Summary Statistics ===")
        stats = generate_summary_stats(bets)

        logger.info("Average CLV by time window:")
        for window, avg_clv in stats['avg_clv'].items():
            count = stats['coverage_counts'][window]
            if avg_clv is not None:
                logger.info(f"  {window}: {avg_clv:+.3f} (n={count})")
            else:
                logger.info(f"  {window}: N/A (n={count})")

        if stats['avg_snapshot_coverage'] is not None:
            logger.info(f"\nAverage snapshot coverage: {stats['avg_snapshot_coverage']:.1%}")

    logger.info(
        f"\n=== CLV Population Complete: {success_count} succeeded, {failed_count} failed ==="
    )

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
