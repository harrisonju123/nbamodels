#!/usr/bin/env python3
"""
Collect NBA News - Cron Script

Fetches NBA news from RSS feeds (ESPN, NBA.com, Yahoo Sports).
Run hourly to maintain fresh news data.

Cron schedule:
0 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_news.py >> logs/news.log 2>&1
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

from src.data.news_scrapers import NBANewsClient

# Load environment variables
load_dotenv()

# File locking to prevent concurrent execution
LOCK_FILE = "/tmp/nba_news_collector.lock"


def main():
    """Main collection function."""
    # Acquire file lock
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.warning("Another news collection process is already running - skipping")
        lock_fd.close()
        return 0

    try:
        logger.info("=" * 60)
        logger.info("Starting NBA news collection")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("=" * 60)

        # Initialize client
        client = NBANewsClient()

        # Fetch all RSS feeds
        articles = client.fetch_all_feeds()

        if articles.empty:
            logger.warning("No articles fetched from any feed")
            return 0

        logger.info(f"Fetched {len(articles)} total articles")

        # Log breakdown by source
        source_counts = articles.groupby("source").size()
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count} articles")

        # Save articles and extract entities
        count = client.save_articles(articles)
        logger.success(f"Successfully saved {count} new articles")

        if count == 0:
            logger.info("All articles were duplicates (already in database)")

        logger.info("=" * 60)
        logger.info("News collection completed successfully")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during news collection: {e}")
        logger.exception("Full traceback:")
        return 1

    finally:
        # Release file lock
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
