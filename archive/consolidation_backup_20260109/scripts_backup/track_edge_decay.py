#!/usr/bin/env python3
"""
Track Edge Decay

Daily script to track how edges decay from initial detection to game start.
Logs edge decay patterns for market efficiency analysis.

Usage:
    python scripts/track_edge_decay.py
    python scripts/track_edge_decay.py --days 1
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from loguru import logger

from src.market_analysis.market_efficiency import TimeToEfficiencyTracker


def get_recent_bets(days: int = 1):
    """Get bets from recent days that have been settled."""
    db_path = "data/bets/bets.db"
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            game_id,
            bet_type,
            bet_side,
            edge,
            logged_at,
            outcome
        FROM bets
        WHERE settled_at >= date('now', ?)
            AND outcome IS NOT NULL
            AND edge IS NOT NULL
            AND edge > 0
    """

    cursor = conn.execute(query, (f'-{days} days',))
    bets = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return bets


def save_edge_decay(pattern, db_path: str = "data/bets/bets.db"):
    """Save edge decay pattern to database."""
    conn = sqlite3.connect(db_path)

    conn.execute(
        """
        INSERT OR REPLACE INTO edge_decay_tracking
            (game_id, bet_type, bet_side, initial_edge, initial_edge_time,
             edge_at_1hr, edge_at_4hr, edge_at_12hr, closing_edge,
             time_to_half_edge_hrs, time_to_zero_edge_hrs, was_correct_side)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pattern.game_id,
            pattern.bet_type,
            None,  # bet_side - simplified
            pattern.initial_edge,
            None,  # initial_edge_time - simplified
            None,  # edge_at_1hr - would need recalculation
            None,  # edge_at_4hr
            None,  # edge_at_12hr
            0.0,   # closing_edge - assume edge went to 0
            pattern.time_to_half_edge,
            pattern.time_to_zero_edge,
            pattern.was_correct_side,
        ),
    )

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Track edge decay patterns")
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to look back (default: 1)",
    )

    args = parser.parse_args()

    logger.info(f"Tracking edge decay for bets from last {args.days} days")

    tracker = TimeToEfficiencyTracker()

    # Get recent settled bets
    bets = get_recent_bets(days=args.days)
    logger.info(f"Found {len(bets)} bets to analyze")

    if len(bets) == 0:
        logger.info("No bets to analyze")
        return

    # Track decay for each bet
    tracked = 0

    for bet in bets:
        try:
            pattern = tracker.track_edge_decay(
                game_id=bet['game_id'],
                bet_type=bet['bet_type'],
                bet_side=bet['bet_side'],
                initial_edge=bet['edge'],
                initial_edge_time=bet['logged_at'],
            )

            save_edge_decay(pattern)
            tracked += 1

            if pattern.was_arbitraged:
                logger.warning(
                    f"{bet['game_id']} {bet['bet_type']}: "
                    f"Edge {pattern.initial_edge:.2%} decayed quickly "
                    f"(velocity: {pattern.decay_velocity:.4f}/hr) - possibly arbitraged"
                )
            else:
                logger.debug(
                    f"{bet['game_id']} {bet['bet_type']}: "
                    f"Edge decay tracked (velocity: {pattern.decay_velocity:.4f}/hr)"
                )

        except Exception as e:
            logger.error(f"Failed to track decay for {bet['game_id']}: {e}")

    logger.success(f"Tracked edge decay for {tracked}/{len(bets)} bets")

    # Summary statistics
    try:
        avg_time = tracker.get_avg_time_to_efficiency()
        logger.info(f"Average time to efficiency: {avg_time:.1f} hours")

        # Check if edges are being arbitraged faster
        trend = tracker.are_edges_being_arbitraged_faster()
        logger.info("Edge decay trend:")
        for period, stats in trend.items():
            if stats['avg_time_to_zero']:
                logger.info(f"  {period}: {stats['avg_time_to_zero']:.1f} hrs average")

    except Exception as e:
        logger.error(f"Failed to calculate summary stats: {e}")


if __name__ == "__main__":
    main()
