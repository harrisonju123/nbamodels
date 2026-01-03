#!/usr/bin/env python3
"""
Opening Line Capture Script

Run more frequently than snapshots (e.g., every 15 min) to catch openers.
Lightweight - only checks for new games, doesn't store full snapshots.

Usage:
    python scripts/capture_opening_lines.py

Cron schedule (every 15 minutes):
    */15 * * * * cd /path/to/nbamodels && python scripts/capture_opening_lines.py >> logs/opening_lines.log 2>&1
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data.line_snapshot_collector import LineSnapshotCollector


def main():
    """Capture opening lines for new games."""
    logger.info("Starting opening line capture...")

    # Use wider collection window to catch games further out
    collector = LineSnapshotCollector(collection_window_hours=72)

    try:
        # Fetch current odds
        odds_df = collector.odds_client.get_current_odds(
            markets=["h2h", "spreads", "totals"]
        )

        if odds_df.empty:
            logger.info("No odds data available")
            return 0

        # Capture opening lines
        count = collector.capture_opening_lines(odds_df)

        if count > 0:
            logger.success(f"Captured {count} new opening lines")
        else:
            logger.info("No new opening lines to capture")

        return 0

    except Exception as e:
        logger.error(f"Error during opening line capture: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
