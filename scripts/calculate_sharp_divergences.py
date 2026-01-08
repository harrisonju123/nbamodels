#!/usr/bin/env python3
"""
Calculate Sharp Divergences

Hourly script to calculate and store Pinnacle vs retail book divergences.
Runs alongside line snapshot collection.

Usage:
    python scripts/calculate_sharp_divergences.py
    python scripts/calculate_sharp_divergences.py --hours-ahead 24
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from loguru import logger

from src.data.odds_api import OddsAPIClient
from src.market_analysis.market_efficiency import MarketEfficiencyAnalyzer


def get_upcoming_games(hours_ahead: int = 24) -> list:
    """Get game IDs for upcoming games."""
    db_path = "data/bets/bets.db"
    conn = sqlite3.connect(db_path)

    cutoff_time = (datetime.now() + timedelta(hours=hours_ahead)).isoformat()

    query = """
        SELECT DISTINCT game_id
        FROM line_snapshots
        WHERE snapshot_time >= datetime('now', '-1 hour')
            AND game_id IN (
                SELECT DISTINCT game_id
                FROM line_snapshots
                WHERE snapshot_time <= ?
            )
    """

    cursor = conn.execute(query, (cutoff_time,))
    game_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    return game_ids


def save_divergence(divergence, db_path: str = "data/bets/bets.db"):
    """Save divergence to database."""
    conn = sqlite3.connect(db_path)

    conn.execute(
        """
        INSERT OR REPLACE INTO sharp_divergences
            (game_id, snapshot_time, bet_type, pinnacle_line, pinnacle_odds,
             retail_consensus_line, retail_consensus_odds, divergence, sharp_side)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            divergence.game_id,
            divergence.snapshot_time,
            divergence.bet_type,
            divergence.pinnacle_line,
            divergence.pinnacle_odds,
            divergence.retail_consensus_line,
            divergence.retail_consensus_odds,
            divergence.divergence,
            divergence.sharp_side,
        ),
    )

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Calculate sharp vs public divergences")
    parser.add_argument(
        "--hours-ahead",
        type=int,
        default=24,
        help="Look at games within this many hours (default: 24)",
    )
    parser.add_argument(
        "--min-divergence",
        type=float,
        default=0.5,
        help="Minimum divergence to record (points, default: 0.5)",
    )

    args = parser.parse_args()

    logger.info("Calculating sharp vs public divergences")

    analyzer = MarketEfficiencyAnalyzer(pinnacle_threshold=args.min_divergence)

    # Get upcoming games
    game_ids = get_upcoming_games(hours_ahead=args.hours_ahead)
    logger.info(f"Found {len(game_ids)} upcoming games")

    if len(game_ids) == 0:
        logger.info("No games to analyze")
        return

    # Calculate divergences for each game
    total_divergences = 0
    significant_divergences = 0

    for game_id in game_ids:
        for bet_type in ['spread', 'totals']:
            try:
                divergence = analyzer.calculate_pinnacle_divergence(
                    game_id=game_id,
                    bet_type=bet_type,
                )

                if divergence:
                    total_divergences += 1

                    if divergence.divergence >= args.min_divergence:
                        significant_divergences += 1
                        save_divergence(divergence)

                        logger.info(
                            f"{game_id} {bet_type}: "
                            f"Divergence {divergence.divergence:.2f} points "
                            f"(Pinnacle {divergence.pinnacle_line} vs Retail {divergence.retail_consensus_line:.2f}) "
                            f"Sharp side: {divergence.sharp_side}"
                        )

            except Exception as e:
                logger.error(f"Failed to calculate divergence for {game_id} {bet_type}: {e}")

    logger.success(
        f"Calculated {total_divergences} divergences, "
        f"{significant_divergences} significant (>={args.min_divergence} points)"
    )


if __name__ == "__main__":
    main()
