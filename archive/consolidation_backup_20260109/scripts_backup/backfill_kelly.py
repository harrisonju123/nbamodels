#!/usr/bin/env python3
"""
Backfill Kelly Criterion percentages for existing bets.

Calculates Kelly % for all bets that have odds and model_prob but missing kelly value.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from loguru import logger

from src.utils.constants import BETS_DB_PATH


def calculate_kelly(odds: float, model_prob: float) -> float:
    """
    Calculate Kelly Criterion percentage.

    Args:
        odds: American odds (e.g., -110, +150)
        model_prob: Model's win probability (0-1)

    Returns:
        Kelly percentage (0-1)
    """
    # Convert American odds to decimal
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1

    # Kelly formula: f* = (bp - q) / b
    # where b = decimal_odds - 1, p = model_prob, q = 1 - model_prob
    b = decimal_odds - 1

    if b > 0 and model_prob > 0:
        kelly_pct = (model_prob * b - (1 - model_prob)) / b
        kelly_pct = max(0, kelly_pct)  # Don't bet on negative Kelly
        return kelly_pct
    else:
        return 0


def backfill_kelly():
    """Backfill Kelly percentages for existing bets."""
    logger.info("=" * 80)
    logger.info("KELLY CRITERION BACKFILL")
    logger.info("=" * 80)

    conn = sqlite3.connect(BETS_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        # Get bets missing Kelly
        bets = conn.execute("""
            SELECT id, odds, model_prob, kelly
            FROM bets
            WHERE odds IS NOT NULL
              AND model_prob IS NOT NULL
              AND model_prob > 0
              AND (kelly IS NULL OR kelly = 0)
        """).fetchall()

        logger.info(f"Found {len(bets)} bets needing Kelly calculation")

        if not bets:
            logger.success("All bets already have Kelly values!")
            return

        updated = 0
        for bet in bets:
            bet_id = bet['id']
            odds = bet['odds']
            model_prob = bet['model_prob']

            # Calculate Kelly
            kelly_pct = calculate_kelly(odds, model_prob)

            # Update bet
            conn.execute("""
                UPDATE bets
                SET kelly = ?
                WHERE id = ?
            """, (kelly_pct, bet_id))

            updated += 1

            if updated % 10 == 0:
                logger.info(f"Updated {updated}/{len(bets)} bets...")

        conn.commit()
        logger.success(f"✅ Updated {updated} bets with Kelly percentages")

        # Show summary
        summary = conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(kelly) as avg_kelly,
                MIN(kelly) as min_kelly,
                MAX(kelly) as max_kelly,
                SUM(CASE WHEN kelly > 0 THEN 1 ELSE 0 END) as positive_kelly
            FROM bets
            WHERE kelly IS NOT NULL
        """).fetchone()

        logger.info("\n" + "=" * 80)
        logger.info("KELLY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total bets with Kelly: {summary['total']}")
        logger.info(f"Average Kelly: {summary['avg_kelly']:.2%}")
        logger.info(f"Min Kelly: {summary['min_kelly']:.2%}")
        logger.info(f"Max Kelly: {summary['max_kelly']:.2%}")
        logger.info(f"Positive Kelly bets: {summary['positive_kelly']}")
        logger.info("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    backfill_kelly()
