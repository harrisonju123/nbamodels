#!/usr/bin/env python3
"""
Collect Referee Assignments - Cron Script

Fetches today's referee assignments from NBA.com and stores them in the database.
Run daily at 10 AM ET to capture referee assignments when they're posted.

Cron schedule:
0 10 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_referees.py >> logs/referees.log 2>&1
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

from src.data.referee_data import RefereeDataClient

# Load environment variables
load_dotenv()

# File locking to prevent concurrent execution
LOCK_FILE = "/tmp/nba_referee_collector.lock"


def main():
    """Main collection function."""
    # Acquire file lock
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.warning("Another referee collection process is already running - skipping")
        lock_fd.close()
        return 0

    try:
        logger.info("=" * 60)
        logger.info("Starting referee assignment collection")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("=" * 60)

        # Initialize client
        client = RefereeDataClient()

        # Fetch today's referee assignments
        assignments = client.get_todays_referees()

        if assignments.empty:
            logger.warning("No referee assignments found for today")
            logger.info("This is normal if no games are scheduled")
            return 0

        logger.info(f"Found {len(assignments)} referee assignments")

        # Save to database
        count = client.save_referee_assignments(assignments)
        logger.success(f"Successfully saved {count} referee assignments")

        # Log summary
        unique_games = assignments["game_id"].nunique()
        unique_refs = assignments["ref_name"].nunique()
        logger.info(f"Summary: {unique_refs} referees across {unique_games} games")

        logger.info("=" * 60)
        logger.info("Referee collection completed successfully")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during referee collection: {e}")
        logger.exception("Full traceback:")
        return 1

    finally:
        # Release file lock
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
