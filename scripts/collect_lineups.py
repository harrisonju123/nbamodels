#!/usr/bin/env python3
"""
Collect Confirmed Lineups - Cron Script

Fetches confirmed starting lineups from ESPN and stores them in the database.
Run every 15 minutes during game hours (5-11 PM ET) to capture lineup updates.

Cron schedule:
*/15 17-23 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_lineups.py >> logs/lineups.log 2>&1
"""

import sys
import os
import fcntl
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from loguru import logger

from src.data.lineup_scrapers import ESPNLineupClient

# Load environment variables
load_dotenv()

# File locking to prevent concurrent execution
LOCK_FILE = "/tmp/nba_lineup_collector.lock"


def main():
    """Main collection function."""
    # Acquire file lock
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.warning("Another lineup collection process is already running - skipping")
        lock_fd.close()
        return 0

    try:
        logger.info("=" * 60)
        logger.info("Starting lineup collection")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("=" * 60)

        # Initialize client
        client = ESPNLineupClient()

        # Fetch today's lineups
        lineups = client.get_todays_lineups()

        if lineups.empty:
            logger.info("No lineup data found for today")
            logger.info("This is normal if no games are scheduled or lineups not yet posted")
            return 0

        total_players = len(lineups)
        starters = lineups[lineups["is_starter"] == 1]
        total_starters = len(starters)

        logger.info(f"Found {total_players} player records ({total_starters} confirmed starters)")

        # Save to database
        count = client.save_lineups(lineups)
        logger.success(f"Successfully saved {count} lineup records")

        # Log summary by team
        if not starters.empty:
            teams_with_lineups = starters["team_abbrev"].nunique()
            logger.info(f"Summary: Confirmed starters for {teams_with_lineups} teams")

            # Log teams with complete lineups (5 starters)
            starter_counts = starters.groupby("team_abbrev").size()
            complete_lineups = starter_counts[starter_counts == 5]
            if not complete_lineups.empty:
                logger.info(f"Complete lineups (5 starters): {', '.join(complete_lineups.index)}")

        logger.info("=" * 60)
        logger.info("Lineup collection completed successfully")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during lineup collection: {e}")
        logger.exception("Full traceback:")
        return 1

    finally:
        # Release file lock
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
